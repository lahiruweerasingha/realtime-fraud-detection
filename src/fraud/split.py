"""Time-based train / validation / test split.

A random split would put transactions from the last day into training and
let the model learn from the future. Splitting on `step` keeps the evaluation
honest: the model is always scored on hours it has never seen.
"""

from __future__ import annotations

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F


def split_name(step: int, train_end_step: int, valid_end_step: int) -> str:
    """Plain-Python version of the rule, used by tests and documentation."""
    if not train_end_step < valid_end_step:
        raise ValueError("train_end_step must be before valid_end_step")
    if step <= train_end_step:
        return "train"
    if step <= valid_end_step:
        return "valid"
    return "test"


def split_column(train_end_step: int, valid_end_step: int, step_col: str = "step") -> Column:
    if not train_end_step < valid_end_step:
        raise ValueError("train_end_step must be before valid_end_step")
    step = F.col(step_col)
    return (
        F.when(step <= train_end_step, "train")
        .when(step <= valid_end_step, "valid")
        .otherwise("test")
    )


def assign_split(df: DataFrame, train_end_step: int, valid_end_step: int) -> DataFrame:
    return df.withColumn("split", split_column(train_end_step, valid_end_step))
