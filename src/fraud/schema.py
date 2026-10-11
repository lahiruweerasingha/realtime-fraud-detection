"""Column contract for the PaySim transaction log.

Every job reads the raw CSV through `read_transactions`, so column names,
types and the excluded columns are defined once.

PySpark is imported inside the functions that need it, so the constants here
can also be used by the streaming components, which run without Spark.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession
    from pyspark.sql import types as T

# Raw header as published, in file order -> snake_case name used everywhere else.
RENAMES: dict[str, str] = {
    "step": "step",
    "type": "txn_type",
    "amount": "amount",
    "nameOrig": "name_orig",
    "oldbalanceOrg": "old_balance_orig",
    "newbalanceOrig": "new_balance_orig",
    "nameDest": "name_dest",
    "oldbalanceDest": "old_balance_dest",
    "newbalanceDest": "new_balance_dest",
    "isFraud": "is_fraud",
    "isFlaggedFraud": "is_flagged_fraud",
}

LABEL = "is_fraud"

# The operator's existing rule. Kept out of the model and used as the
# baseline the model has to beat.
RULE_FLAG = "is_flagged_fraud"

# The dataset description states that fraudulent transactions are cancelled,
# so these balances are written *after* the outcome is known. A model that
# reads them is reading the label. They are never used as features.
LEAKY_COLUMNS: tuple[str, ...] = (
    "old_balance_orig",
    "new_balance_orig",
    "old_balance_dest",
    "new_balance_dest",
)

TRANSACTION_TYPES: tuple[str, ...] = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")

# Only these transaction types are scored. The data profile shows every fraud
# in PaySim is a TRANSFER or a CASH_OUT (see evidence/data_profile.json);
# the other types pass through without a score.
MODEL_SCOPE_TYPES: tuple[str, ...] = ("TRANSFER", "CASH_OUT")

# Spark SQL type of each raw column.
_RAW_TYPES: dict[str, str] = {
    "step": "int",
    "type": "string",
    "amount": "double",
    "nameOrig": "string",
    "oldbalanceOrg": "double",
    "newbalanceOrig": "double",
    "nameDest": "string",
    "oldbalanceDest": "double",
    "newbalanceDest": "double",
    "isFraud": "int",
    "isFlaggedFraud": "int",
}


def raw_schema() -> T.StructType:
    """Explicit schema: no inference pass over 6M rows, and no silent type drift."""
    from pyspark.sql import types as T

    spark_types = {"int": T.IntegerType(), "string": T.StringType(), "double": T.DoubleType()}
    return T.StructType(
        [T.StructField(name, spark_types[_RAW_TYPES[name]], True) for name in RENAMES]
    )


def read_transactions(spark: SparkSession, path: str) -> DataFrame:
    """Read the raw CSV with the explicit schema and snake_case column names.

    Rows that do not match the schema are kept as nulls (PERMISSIVE mode) so
    that validation can count them, rather than being dropped silently.
    """
    from pyspark.sql import functions as F

    raw = (
        spark.read.option("header", True)
        .option("mode", "PERMISSIVE")
        .schema(raw_schema())
        .csv(path)
    )
    return raw.select([F.col(src).alias(dst) for src, dst in RENAMES.items()])
