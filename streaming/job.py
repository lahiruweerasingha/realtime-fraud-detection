"""Real-time fraud scoring with PyFlink.

    make stream-run

    Kafka `transactions`
      -> validate                      bad messages -> `transactions-dlq` (+ counter)
      -> event-time timestamps         step s = 2024-01-01 00:00 UTC + s hours
      -> key by destination account
      -> per account, in arrival order: compute the features of
                                       sql/features.sql (fraud.online) and score
                                       with the MLflow champion
      -> `scores`                      every scored transaction
      -> hourly monitoring             volume, alerts, alert precision, score
                                       drift (PSI) -> InfluxDB -> Grafana

Why each transaction can be scored the moment it arrives: the producer keys
messages by destination account, so all of one account's messages sit in one
Kafka partition, are read by one source task and travel one network channel to
the scoring task. Each of those is first-in first-out, so an account's
transactions arrive in hour order. `AccountFeed` scores each one against the
account's earlier hours only and holds the current hour's totals aside until
the next hour starts, which is exactly the feature SQL's
`... AND 1 PRECEDING` rule (tests/test_online_parity.py checks the two agree).
A transaction from an hour earlier than one already seen for its account is
counted as late and not scored.

An earlier version buffered each account's hour in Flink state until the
watermark closed it. It was correct regardless of arrival order but cost
several state round trips and a timer per record, and ran at about 200
records a second; see docs/decisions/0005.
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd
from pyflink.common import Configuration, Duration, Types, WatermarkStrategy
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.time import Time
from pyflink.common.watermark_strategy import TimestampAssigner
from pyflink.datastream import OutputTag, RuntimeExecutionMode, StreamExecutionEnvironment
from pyflink.datastream.connectors.base import DeliveryGuarantee
from pyflink.datastream.connectors.kafka import (
    KafkaOffsetsInitializer,
    KafkaRecordSerializationSchema,
    KafkaSink,
    KafkaSource,
)
from pyflink.datastream.functions import (
    KeyedProcessFunction,
    ProcessAllWindowFunction,
    ProcessFunction,
)
from pyflink.datastream.state import ValueStateDescriptor
from pyflink.datastream.window import TumblingEventTimeWindows

from fraud.online import AccountFeed, DestinationState
from fraud.schema import MODEL_SCOPE_TYPES
from fraud.stream_records import STEP_MS, dead_letter, event_time_ms, hourly_summary, step_of
from fraud.stream_records import validate as validate_message

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
INPUT_TOPIC = os.getenv("INPUT_TOPIC", "transactions")
SCORES_TOPIC = os.getenv("SCORES_TOPIC", "scores")
DLQ_TOPIC = os.getenv("DLQ_TOPIC", "transactions-dlq")
SCORE_FROM_STEP = int(os.getenv("SCORE_FROM_STEP", "378"))
SCORE_TO_STEP = int(os.getenv("SCORE_TO_STEP", "408"))
# BOUNDED=1 reads the topic up to its current end and then finishes, which
# flushes every hour; used for the verification run. Otherwise the job runs live.
BOUNDED = os.getenv("BOUNDED", "0") == "1"
OUT_OF_ORDERNESS_STEPS = int(os.getenv("OUT_OF_ORDERNESS_STEPS", "2"))

DLQ = OutputTag("dead-letters", Types.STRING())


def influx_writer():
    """A synchronous InfluxDB write API, or None if InfluxDB is not configured."""
    url = os.getenv("INFLUX_URL")
    if not url:
        return None, None
    from influxdb_client import InfluxDBClient
    from influxdb_client.client.write_api import SYNCHRONOUS

    client = InfluxDBClient(url=url, token=os.environ["INFLUX_TOKEN"], org=os.environ["INFLUX_ORG"])
    return client, client.write_api(write_options=SYNCHRONOUS)


class Validate(ProcessFunction):
    """Checks every message BEFORE event time is assigned, so a malformed record
    can never move the watermark. Bad messages go to the dead-letter topic."""

    def open(self, runtime_context):
        self.dead_letters = runtime_context.get_metrics_group().counter("dead_letter_records")
        self.client, self.writer = influx_writer()

    def close(self):
        if self.client:
            self.client.close()

    def process_element(self, value, ctx):
        record, reason = validate_message(value)
        if record is not None:
            # (account, step, message): downstream operators read the key and
            # the event time without parsing the JSON again.
            yield record["name_dest"], record["step"], json.dumps(record)
            return
        self.dead_letters.inc()
        if self.writer:
            from influxdb_client import Point

            self.writer.write(
                bucket=os.environ["INFLUX_BUCKET"],
                record=Point("dead_letters").tag("reason", reason).field("count", 1),
            )
        yield DLQ, dead_letter(value, reason)


class EventTime(TimestampAssigner):
    def extract_timestamp(self, value, record_timestamp):
        return event_time_ms(value[1])


class ScoreAccount(KeyedProcessFunction):
    """Per destination account: featurise and score each transaction on arrival."""

    def open(self, runtime_context):
        from fraud.serving import load_champion

        self.feed = runtime_context.get_state(
            ValueStateDescriptor("destination_feed", Types.STRING())
        )
        self.champion = load_champion()
        metrics = runtime_context.get_metrics_group()
        self.scored = metrics.counter("scored_transactions")
        self.late = metrics.counter("late_transactions")

    def process_element(self, value, ctx):
        _, step, message = value
        record = json.loads(message)
        if record["kind"] == "state":
            # History from before the replay. It is written without reading the
            # state first: the producer sends an account's state message before
            # any of its transactions, on the same partition, so it always
            # arrives first.
            history = DestinationState(record["total_count"], record["total_amount"])
            self.feed.update(json.dumps(AccountFeed(history=history).to_dict()))
            return

        saved = self.feed.value()
        feed = AccountFeed.from_dict(json.loads(saved)) if saved else AccountFeed()
        features = feed.observe(step, record["txn_type"], record["amount"])
        if features is None:
            # An hour this account has already moved past; counted, not mixed in.
            self.late.inc()
            return
        self.feed.update(json.dumps(feed.to_dict()))

        if record["txn_type"] not in MODEL_SCOPE_TYPES:
            return
        if not SCORE_FROM_STEP <= step <= SCORE_TO_STEP:
            return  # warm-up: builds history, is not scored
        row = {**record, **features}
        score = float(self.champion.predict(pd.DataFrame([row]))[0])
        self.scored.inc()
        yield json.dumps(
            {
                **row,
                "score": score,
                "alert": bool(score >= self.champion.alert_threshold),
                "model_version": self.champion.version,
            }
        )


class HourlyMonitor(ProcessAllWindowFunction):
    """One summary per event-time hour, written to InfluxDB for Grafana."""

    def __init__(self, threshold: float, reference: dict):
        self.threshold = threshold
        self.reference = reference

    def open(self, runtime_context):
        self.client, self.writer = influx_writer()

    def close(self):
        if self.client:
            self.client.close()

    def process(self, context, elements):
        rows = [json.loads(e) for e in elements]
        summary = hourly_summary(
            [r["score"] for r in rows],
            [r["is_fraud"] for r in rows],
            self.threshold,
            self.reference,
        )
        start = context.window().start
        summary["step"] = step_of(start)
        if self.writer:
            from influxdb_client import Point, WritePrecision

            point = Point("hourly").time(start, WritePrecision.MS)
            for key, value in summary.items():
                if value is not None:  # undefined rates are left out, not written as 0
                    point = point.field(key, value)
            self.writer.write(bucket=os.environ["INFLUX_BUCKET"], record=point)
        yield json.dumps(summary)


def kafka_sink(topic: str) -> KafkaSink:
    return (
        KafkaSink.builder()
        .set_bootstrap_servers(BOOTSTRAP)
        .set_record_serializer(
            KafkaRecordSerializationSchema.builder()
            .set_topic(topic)
            .set_value_serialization_schema(SimpleStringSchema())
            .build()
        )
        .set_delivery_guarantee(DeliveryGuarantee.AT_LEAST_ONCE)
        .build()
    )


def main() -> int:
    from fraud.serving import load_champion

    champion = load_champion()  # read once here for the monitor's threshold/reference
    print(
        f"champion: v{champion.version} ({champion.model_type}), "
        f"alert threshold {champion.alert_threshold:.6f}"
    )

    config = Configuration()
    # Every record crosses between the JVM and a Python worker. Larger bundles
    # mean fewer round trips, and a state cache larger than the number of
    # active accounts (about 700,000 in the replay) keeps state reads inside
    # the Python worker. With the defaults (1,000 each) the replay slowed to
    # ~120 records a second.
    config.set_string("python.fn-execution.bundle.size", "10000")
    config.set_string("python.state.cache-size", "1000000")
    env = StreamExecutionEnvironment.get_execution_environment(config)
    env.set_runtime_mode(RuntimeExecutionMode.STREAMING)
    # One source task per partition of the 3-partition input topic.
    env.set_parallelism(int(os.getenv("PARALLELISM", "3")))
    env.enable_checkpointing(180_000)
    # The Python operator is the bottleneck, so the job runs under
    # backpressure. Aligned checkpoint barriers then queue behind the backlog
    # and block input while they wait (up to 160 s per checkpoint in testing);
    # unaligned checkpoints let barriers overtake the buffered records.
    env.get_checkpoint_config().enable_unaligned_checkpoints()

    builder = (
        KafkaSource.builder()
        .set_bootstrap_servers(BOOTSTRAP)
        .set_topics(INPUT_TOPIC)
        .set_group_id("fraud-scoring")
        .set_starting_offsets(KafkaOffsetsInitializer.earliest())
        .set_value_only_deserializer(SimpleStringSchema())
    )
    if BOUNDED:
        builder = builder.set_bounded(KafkaOffsetsInitializer.latest())
    source = builder.build()

    # Validation runs before any timestamp is assigned. Watermarks are attached
    # AFTER the source: a Python TimestampAssigner passed into from_source()
    # would be handed to the Java Kafka operator and silently ignored.
    validated = env.from_source(
        source, WatermarkStrategy.no_watermarks(), "kafka-transactions"
    ).process(Validate(), output_type=Types.TUPLE([Types.STRING(), Types.INT(), Types.STRING()]))

    # Event time drives only the hourly monitoring windows; scoring does not
    # wait for it. Out-of-orderness of two hours: a source task can read more
    # than one Kafka partition, and records arrive in per-partition batches,
    # so one partition can run up to about an hour ahead of another. With
    # monotonous timestamps the watermark would overtake the slower partition
    # and its scores would miss their monitoring window.
    strategy = WatermarkStrategy.for_bounded_out_of_orderness(
        Duration.of_millis(OUT_OF_ORDERNESS_STEPS * STEP_MS)
    ).with_timestamp_assigner(EventTime())
    if not BOUNDED:
        # Live only: let a quiet partition stop holding the watermark back. In a
        # replay every partition has data, and under backpressure a starved task
        # could be wrongly marked idle, so idleness is not used there.
        strategy = strategy.with_idleness(Duration.of_seconds(30))
    timed = validated.assign_timestamps_and_watermarks(strategy)

    scored = timed.key_by(lambda v: v[0], key_type=Types.STRING()).process(
        ScoreAccount(), output_type=Types.STRING()
    )

    scored.sink_to(kafka_sink(SCORES_TOPIC))
    validated.get_side_output(DLQ).sink_to(kafka_sink(DLQ_TOPIC))

    scored.window_all(TumblingEventTimeWindows.of(Time.milliseconds(STEP_MS))).process(
        HourlyMonitor(champion.alert_threshold, champion.score_reference),
        output_type=Types.STRING(),
    ).print()

    env.execute("fraud-scoring")
    return 0


if __name__ == "__main__":
    sys.exit(main())
