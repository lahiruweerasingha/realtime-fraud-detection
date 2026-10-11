"""Validate the raw PaySim CSV and write a typed Parquet copy.

    make validate

Stops with a non-zero exit code if any hard check fails, so nothing
downstream ever runs on a broken or unexpected file.
"""

from __future__ import annotations

import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from fraud.io import EVIDENCE_DIR, PROCESSED_DIR, find_raw_csv, sha256, write_json
from fraud.schema import read_transactions
from fraud.validate import failed_checks, profile


def main() -> int:
    spark = SparkSession.builder.appName("fraud-validate").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    source = find_raw_csv()
    print(f"input: {source.name}")

    txns = read_transactions(spark, str(source)).cache()
    measured = profile(txns)
    measured["source_file"] = source.name
    measured["source_bytes"] = source.stat().st_size
    measured["source_sha256"] = sha256(source)

    failures = failed_checks(measured)
    measured["failed_checks"] = failures
    write_json(measured, EVIDENCE_DIR / "data_profile.json")

    print(
        f"rows: {measured['rows']:,}   fraud rows: {measured['fraud_rows']:,} "
        f"({measured['fraud_rate']:.4%})"
    )
    if failures:
        print("VALIDATION FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        spark.stop()
        return 1

    # A stable id for every transaction. Written once here, so every later
    # stage refers to the same rows by the same id.
    out = str(PROCESSED_DIR / "transactions.parquet")
    txns.withColumn("txn_id", F.monotonically_increasing_id()).write.mode("overwrite").parquet(out)
    print(f"all checks passed -> {out}")
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
