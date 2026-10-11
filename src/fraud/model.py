"""Model definitions: a logistic-regression baseline and distributed XGBoost."""

from __future__ import annotations

from collections.abc import Sequence

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from xgboost.spark import SparkXGBClassifier

# Only these transaction types are scored. The data profile shows every fraud
# in PaySim is a TRANSFER or a CASH_OUT (see evidence/data_profile.json);
# the other types pass through without a score.
MODEL_SCOPE_TYPES: tuple[str, ...] = ("TRANSFER", "CASH_OUT")

XGB_PARAMS: dict = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 400,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "early_stopping_rounds": 30,
    "random_state": 7,
}


def class_weight(df: DataFrame, label: str) -> tuple[DataFrame, float]:
    """Add a `weight` column that balances the classes; return the positive weight.

    Computed on the frame it is given, so pass the TRAINING rows only.
    """
    counts = {r[label]: r["count"] for r in df.groupBy(label).count().collect()}
    positives, negatives = counts.get(1, 0), counts.get(0, 0)
    if positives == 0:
        raise ValueError("no positive rows to train on")
    pos_weight = negatives / positives
    weighted = df.withColumn("weight", F.when(F.col(label) == 1, pos_weight).otherwise(1.0))
    return weighted, pos_weight


def logistic_pipeline(features: Sequence[str], label: str) -> Pipeline:
    return Pipeline(
        stages=[
            VectorAssembler(inputCols=list(features), outputCol="features_raw"),
            # Counts and amounts span several orders of magnitude; scaling keeps
            # the regularisation penalty fair across features.
            StandardScaler(inputCol="features_raw", outputCol="features", withMean=True),
            LogisticRegression(
                featuresCol="features",
                labelCol=label,
                weightCol="weight",
                maxIter=100,
                regParam=0.01,
            ),
        ]
    )


def xgboost_pipeline(
    features: Sequence[str],
    label: str,
    scale_pos_weight: float,
    num_workers: int = 2,
    params: dict | None = None,
) -> Pipeline:
    """XGBoost trained across `num_workers` Spark tasks.

    Rows with `is_valid = true` are held out of training and used only for
    early stopping. The fitted booster is plain XGBoost, so it can be loaded
    and scored outside Spark (which a Spark MLlib GBT model cannot).
    """
    xgb = SparkXGBClassifier(
        features_col="features",
        label_col=label,
        validation_indicator_col="is_valid",
        num_workers=num_workers,
        scale_pos_weight=scale_pos_weight,
        **(params or XGB_PARAMS),
    )
    return Pipeline(stages=[VectorAssembler(inputCols=list(features), outputCol="features"), xgb])


def score(model: PipelineModel, df: DataFrame) -> DataFrame:
    """Probability of fraud as a plain `score` column."""
    return model.transform(df).withColumn("score", vector_to_array(F.col("probability")).getItem(1))
