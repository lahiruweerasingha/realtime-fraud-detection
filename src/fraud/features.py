"""Feature engineering: runs sql/features.sql over the prepared transactions."""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from fraud.schema import LEAKY_COLUMNS

SQL_FILE = Path(__file__).resolve().parents[2] / "sql" / "features.sql"

# Model inputs, in the order the models see them.
#
# Removed after the first full run (evidence/runs/01-initial-features):
#   is_merchant_dest      zero gain: in-scope transfers and cash-outs never go
#                         to merchants, so it is constant in the scored data.
#   orig_prior_txn_count  0.03% of total gain: origin accounts almost never
#                         repeat (9,298 of 6.35M appear more than once).
FEATURE_COLUMNS: tuple[str, ...] = (
    "log_amount",
    "hour_of_day",
    "hour_sin",
    "hour_cos",
    "is_transfer",
    "dest_prior_txn_count",
    "dest_prior_amount",
    "dest_txn_count_24h",
    "dest_amount_24h",
    "amount_vs_dest_mean",
)

# The time-of-day features. PaySim injects fraud at a steady rate around the
# clock while legitimate volume follows a daily cycle, so these carry a strong
# simulator-specific signal; training reports a model without them as well.
TIME_OF_DAY_FEATURES: tuple[str, ...] = ("hour_of_day", "hour_sin", "hour_cos")

# Leakage guard, checked at import time: no feature may be a leaky column.
assert not set(FEATURE_COLUMNS) & set(LEAKY_COLUMNS), "a leaky column is listed as a feature"


def build_features(
    spark: SparkSession, transactions: DataFrame, sql_file: Path = SQL_FILE
) -> DataFrame:
    """Register the prepared transactions as a view and run the feature SQL."""
    transactions.createOrReplaceTempView("transactions")
    return spark.sql(sql_file.read_text(encoding="utf-8"))
