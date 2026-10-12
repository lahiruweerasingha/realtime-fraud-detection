"""Turn trained Spark pipelines into artefacts that score WITHOUT Spark.

A model is only deployable if something outside the training cluster can
produce the same scores. Both exports here are checked against the Spark
predictions in tests/test_export.py.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import mlflow.pyfunc
import numpy as np
import pandas as pd
import xgboost as xgb

if TYPE_CHECKING:
    # Type hints only. A registered LogisticScorer is unpickled in the Flink
    # image, which has no Spark, so this module must import without PySpark.
    from pyspark.ml import PipelineModel


class LogisticScorer(mlflow.pyfunc.PythonModel):
    """Standardise -> linear score -> sigmoid, exactly as the Spark pipeline does."""

    def __init__(
        self,
        features: Sequence[str],
        mean: Sequence[float],
        std: Sequence[float],
        coefficients: Sequence[float],
        intercept: float,
    ):
        self.features = list(features)
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.intercept = float(intercept)

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.features].to_numpy(dtype=float)
        # Spark's StandardScaler outputs 0 for a constant (zero-variance) column.
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(self.std > 0, (x - self.mean) / self.std, 0.0)
        return 1.0 / (1.0 + np.exp(-(z @ self.coefficients + self.intercept)))

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        return self.score(model_input)


def logistic_scorer(model: PipelineModel, features: Sequence[str]) -> LogisticScorer:
    """Extract the fitted scaler and regression from `logistic_pipeline`."""
    scaler, regression = model.stages[1], model.stages[2]
    return LogisticScorer(
        features=features,
        mean=scaler.mean.toArray(),
        std=scaler.std.toArray(),
        coefficients=regression.coefficients.toArray(),
        intercept=regression.intercept,
    )


def xgboost_booster(model: PipelineModel) -> xgb.Booster:
    """The fitted booster from `xgboost_pipeline`, trimmed to its best round.

    With early stopping, training continues `early_stopping_rounds` past the
    best round. Spark's `transform` scores with the best round only, but a raw
    booster's `predict` uses EVERY tree. Exporting the untrimmed booster would
    deploy a different model from the one evaluated, so it is sliced here.
    """
    booster = model.stages[-1].get_booster()
    best = booster.attr("best_iteration")
    return booster[: int(best) + 1] if best is not None else booster


def booster_scores(
    booster: xgb.Booster, frame: pd.DataFrame, features: Sequence[str]
) -> np.ndarray:
    """Fraud probability from a booster, with features in training order."""
    matrix = xgb.DMatrix(frame[list(features)].to_numpy(dtype=float))
    return booster.predict(matrix)
