"""Feature engineering: runs sql/features.sql over the prepared transactions."""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from fraud.schema import LEAKY_COLUMNS

SQL_FILE = Path(__file__).resolve().parents[2] / "sql" / "features.sql"

# Model inputs, in the order the models see them.
FEATURE_COLUMNS: tuple[str, ...] = (
    "log_amount",
    "hour_of_day",
    "is_transfer",
    "is_merchant_dest",
    "dest_prior_txn_count",
    "dest_prior_amount",
    "dest_txn_count_24h",
    "dest_amount_24h",
    "amount_vs_dest_mean",
    "orig_prior_txn_count",
)

# Leakage guard, checked at import time: no feature may be a leaky column.
assert not set(FEATURE_COLUMNS) & set(LEAKY_COLUMNS), "a leaky column is listed as a feature"


def build_features(
    spark: SparkSession, transactions: DataFrame, sql_file: Path = SQL_FILE
) -> DataFrame:
    """Register the prepared transactions as a view and run the feature SQL."""
    transactions.createOrReplaceTempView("transactions")
    return spark.sql(sql_file.read_text(encoding="utf-8"))
