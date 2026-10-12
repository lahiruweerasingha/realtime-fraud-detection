"""Prepare a streaming replay that reproduces the batch features exactly.

    make stream-data

The destination-history features count every earlier transaction of an
account, back to step 1. A stream that started empty at the first scored hour
would therefore compute different features from the ones the model was
trained on. This job writes:

  state_snapshot.parquet   per destination active in the replay: all-time
                           transaction count and amount BEFORE the warm-up
  replay.parquet           every transaction (all types) from the warm-up
                           start to the last scored step, in step order

The warm-up covers the 24 steps before scoring starts, which is exactly what
the 24-hour window features need; the snapshot supplies the rest.
"""

from __future__ import annotations

import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from fraud.io import PROCESSED_DIR, write_json
from fraud.online import WINDOW_STEPS

SCORE_FROM_STEP = int(os.getenv("SCORE_FROM_STEP", "378"))  # first test hour
SCORE_TO_STEP = int(os.getenv("SCORE_TO_STEP", "408"))  # last normal-volume hour
WARMUP_FROM_STEP = SCORE_FROM_STEP - WINDOW_STEPS


def main() -> int:
    spark = SparkSession.builder.appName("fraud-stream-state").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    txns = spark.read.parquet(str(PROCESSED_DIR / "transactions.parquet"))
    out = PROCESSED_DIR / "stream"

    replay = txns.filter(F.col("step").between(WARMUP_FROM_STEP, SCORE_TO_STEP)).select(
        "txn_id",
        "step",
        "txn_type",
        "amount",
        "name_orig",
        "name_dest",
        "is_fraud",
        "is_flagged_fraud",
    )
    active = replay.select("name_dest").distinct()
    snapshot = (
        txns.filter(F.col("step") < WARMUP_FROM_STEP)
        .join(active, "name_dest")
        .groupBy("name_dest")
        .agg(F.count(F.lit(1)).alias("total_count"), F.sum("amount").alias("total_amount"))
    )

    replay.orderBy("step", "txn_id").coalesce(1).write.mode("overwrite").parquet(
        str(out / "replay.parquet")
    )
    snapshot.coalesce(1).write.mode("overwrite").parquet(str(out / "state_snapshot.parquet"))

    summary = {
        "warmup_from_step": WARMUP_FROM_STEP,
        "score_from_step": SCORE_FROM_STEP,
        "score_to_step": SCORE_TO_STEP,
        "replay_rows": spark.read.parquet(str(out / "replay.parquet")).count(),
        "snapshot_accounts": spark.read.parquet(str(out / "state_snapshot.parquet")).count(),
        "active_destinations": active.count(),
    }
    write_json(summary, out / "summary.json")
    # The streaming image runs as a different user (flink, uid 9999) and writes
    # its run manifest here, so the folder must be writable by it.
    out.chmod(0o777)
    print(summary)
    spark.stop()
    if not summary["replay_rows"]:
        print(
            f"no transactions in hours {WARMUP_FROM_STEP}-{SCORE_TO_STEP}: set "
            "SCORE_FROM_STEP/SCORE_TO_STEP in .env to hours the data covers",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
