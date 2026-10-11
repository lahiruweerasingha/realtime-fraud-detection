"""Replay the prepared transactions into Kafka.

    make stream-run   (runs this, then the Flink job)

Sends, keyed by destination account so one account's messages stay in order
within a partition:
  1. one `state` message per destination with history before the replay;
  2. every transaction from the warm-up start to the last scored hour, in hour
     order, with the Kafka timestamp set to the transaction's event time;
  3. a fixed set of deliberately malformed messages, spread through the
     stream, to exercise the dead-letter path. What was injected is written to
     a manifest so the verification can check every one was caught.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
from confluent_kafka import Producer

from fraud.stream_records import event_time_ms

STREAM_DIR = Path(os.getenv("STREAM_DIR", "/data/processed/stream"))
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
TOPIC = os.getenv("INPUT_TOPIC", "transactions")
MALFORMED = int(os.getenv("INJECT_MALFORMED", "40"))
SEED = int(os.getenv("SEED", "7"))

# (reason the job must report, a message that should trigger it)
BAD_MESSAGES: list[tuple[str, str]] = [
    ("not_json", "{this is not json"),
    (
        "unknown_txn_type",
        '{"kind": "txn", "txn_id": 1, "step": 380, "txn_type": "BITCOIN", '
        '"amount": 10.0, "name_dest": "C1"}',
    ),
    (
        "bad_amount",
        '{"kind": "txn", "txn_id": 2, "step": 380, "txn_type": "TRANSFER", '
        '"amount": -50.0, "name_dest": "C1"}',
    ),
    (
        "bad_step",
        '{"kind": "txn", "txn_id": 3, "step": "late", "txn_type": "TRANSFER", '
        '"amount": 10.0, "name_dest": "C1"}',
    ),
    (
        "missing_name_dest",
        '{"kind": "txn", "txn_id": 4, "step": 380, "txn_type": "CASH_OUT", "amount": 10.0}',
    ),
]


def main() -> int:
    replay = pd.read_parquet(STREAM_DIR / "replay.parquet").sort_values(["step", "txn_id"])
    snapshot = pd.read_parquet(STREAM_DIR / "state_snapshot.parquet")
    summary = json.loads((STREAM_DIR / "summary.json").read_text())
    state_step = summary["warmup_from_step"] - 1

    producer = Producer(
        {
            "bootstrap.servers": BOOTSTRAP,
            "linger.ms": 50,
            "batch.num.messages": 10_000,
            "compression.type": "lz4",
            "enable.idempotence": True,
        }
    )

    def send(key: str, value: str, step: int) -> None:
        while True:
            try:
                producer.produce(
                    TOPIC, key=key.encode(), value=value.encode(), timestamp=event_time_ms(step)
                )
                return
            except BufferError:  # local queue full: let it drain
                producer.poll(0.5)

    started = time.perf_counter()
    for row in snapshot.itertuples(index=False):
        send(
            row.name_dest,
            json.dumps(
                {
                    "kind": "state",
                    "name_dest": row.name_dest,
                    "step": state_step,
                    "total_count": int(row.total_count),
                    "total_amount": float(row.total_amount),
                }
            ),
            state_step,
        )
    producer.flush()

    rng = random.Random(SEED)
    inject_at = sorted(rng.sample(range(len(replay)), MALFORMED)) if MALFORMED else []
    injected: list[str] = []
    next_bad = 0
    for i, row in enumerate(replay.itertuples(index=False)):
        while next_bad < len(inject_at) and inject_at[next_bad] == i:
            reason, payload = BAD_MESSAGES[next_bad % len(BAD_MESSAGES)]
            send(f"malformed-{next_bad}", payload, int(row.step))
            injected.append(reason)
            next_bad += 1
        send(
            row.name_dest,
            json.dumps(
                {
                    "kind": "txn",
                    "txn_id": int(row.txn_id),
                    "step": int(row.step),
                    "txn_type": row.txn_type,
                    "amount": float(row.amount),
                    "name_dest": row.name_dest,
                    "is_fraud": int(row.is_fraud),
                }
            ),
            int(row.step),
        )
        if i % 50_000 == 0:
            producer.poll(0)
    remaining = producer.flush(60)

    manifest = {
        "state_messages": len(snapshot),
        "transactions": len(replay),
        "malformed_injected": len(injected),
        "malformed_by_reason": {r: injected.count(r) for r in sorted(set(injected))},
        "undelivered": remaining,
        "seconds": round(time.perf_counter() - started, 1),
    }
    (STREAM_DIR / "producer_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0 if remaining == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
