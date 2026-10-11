"""Data validation and profiling.

`profile` measures the dataset; `failed_checks` turns those measurements into
hard pass/fail rules. The validation job stops the pipeline if any rule fails,
so a broken download or a changed file never reaches feature engineering.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from fraud.schema import LABEL, RENAMES, RULE_FLAG, TRANSACTION_TYPES


def profile(df: DataFrame) -> dict:
    """Return the measurements the checks and the README rely on."""
    columns = list(RENAMES.values())

    totals = df.agg(
        F.count(F.lit(1)).alias("rows"),
        F.min("step").alias("min_step"),
        F.max("step").alias("max_step"),
        F.countDistinct("step").alias("distinct_steps"),
        F.sum(LABEL).alias("fraud_rows"),
        F.sum(RULE_FLAG).alias("flagged_rows"),
        F.sum(F.when((F.col(LABEL) == 1) & (F.col(RULE_FLAG) == 1), 1).otherwise(0)).alias(
            "flagged_and_fraud"
        ),
        F.sum(F.when(F.col("amount") < 0, 1).otherwise(0)).alias("negative_amounts"),
        F.countDistinct("name_orig").alias("distinct_orig_accounts"),
        F.countDistinct("name_dest").alias("distinct_dest_accounts"),
        *[F.sum(F.col(c).isNull().cast("int")).alias(f"nulls__{c}") for c in columns],
    ).first()

    result = totals.asDict()
    result["nulls"] = {c: result.pop(f"nulls__{c}") for c in columns}
    result["fraud_rate"] = result["fraud_rows"] / result["rows"] if result["rows"] else 0.0
    result["duplicate_rows"] = result["rows"] - df.dropDuplicates().count()

    by_type = (
        df.groupBy("txn_type")
        .agg(F.count(F.lit(1)).alias("rows"), F.sum(LABEL).alias("fraud_rows"))
        .orderBy("txn_type")
        .collect()
    )
    result["by_type"] = {
        r["txn_type"]: {"rows": r["rows"], "fraud_rows": r["fraud_rows"]} for r in by_type
    }

    # How often do accounts reappear? If origin accounts almost never repeat,
    # "customer history" features carry no signal in this data.
    result["orig_accounts_seen_more_than_once"] = (
        df.groupBy("name_orig").count().filter(F.col("count") > 1).count()
    )
    result["dest_accounts_seen_more_than_once"] = (
        df.groupBy("name_dest").count().filter(F.col("count") > 1).count()
    )
    return result


def failed_checks(p: dict) -> list[str]:
    """Hard rules. An empty list means the data is fit to use."""
    failures: list[str] = []
    if p["rows"] == 0:
        failures.append("dataset is empty")
    for column, nulls in p["nulls"].items():
        if nulls:
            failures.append(f"{nulls} null or unparseable values in '{column}'")
    unknown = sorted(set(p["by_type"]) - set(TRANSACTION_TYPES))
    if unknown:
        failures.append(f"unknown transaction types: {unknown}")
    if p["negative_amounts"]:
        failures.append(f"{p['negative_amounts']} negative amounts")
    if p["min_step"] is not None and p["min_step"] < 1:
        failures.append(f"step starts at {p['min_step']}, expected >= 1")
    if p["fraud_rows"] == 0:
        failures.append("no fraud labels: wrong file or broken label column")
    return failures
