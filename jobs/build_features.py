"""Build the point-in-time feature table from the validated transactions.

make features
"""

from __future__ import annotations

import sys

from pyspark.sql import SparkSession

from fraud.features import build_features
from fraud.io import PROCESSED_DIR


def main() -> int:
    spark = SparkSession.builder.appName("fraud-features").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    txns = spark.read.parquet(str(PROCESSED_DIR / "transactions.parquet"))
    features = build_features(spark, txns)

    out = str(PROCESSED_DIR / "features.parquet")
    features.write.mode("overwrite").parquet(out)

    written = spark.read.parquet(out)
    rows_in, rows_out = txns.count(), written.count()
    print(f"transactions: {rows_in:,}   feature rows: {rows_out:,}")
    spark.stop()
    # Window functions must neither drop nor duplicate rows.
    return 0 if rows_in == rows_out else 1


if __name__ == "__main__":
    sys.exit(main())
