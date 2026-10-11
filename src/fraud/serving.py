"""Load the registered champion with everything a scoring service needs.

The model, its feature order, its alert threshold and its score reference are
all read from the MLflow registry entry, so a service always uses the exact
operating point the model was evaluated at, and nothing is copied by hand.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REGISTERED_MODEL = "fraud-detector"
ALIAS = "champion"


@dataclass
class Champion:
    version: str
    model_type: str
    features: list[str]
    alert_threshold: float
    score_reference: dict
    predict: Callable[[pd.DataFrame], np.ndarray]


def load_champion(name: str = REGISTERED_MODEL, alias: str = ALIAS) -> Champion:
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    version = client.get_model_version_by_alias(name, alias)
    tags = version.tags
    features = tags["features"].split(",")
    uri = f"models:/{name}@{alias}"

    if tags["model_type"] == "xgboost":
        import xgboost as xgb

        booster = mlflow.xgboost.load_model(uri)

        def predict(frame: pd.DataFrame) -> np.ndarray:
            # Plain arrays in training order: the booster was trained on an
            # assembled vector, so it has no feature names to match.
            return booster.predict(xgb.DMatrix(frame[features].to_numpy(dtype=float)))
    else:
        model = mlflow.pyfunc.load_model(uri)

        def predict(frame: pd.DataFrame) -> np.ndarray:
            return np.asarray(model.predict(frame[features]))

    with tempfile.TemporaryDirectory() as tmp:
        path = client.download_artifacts(version.run_id, "monitoring/score_reference.json", tmp)
        reference = json.loads(Path(path).read_text())["score_reference"]

    return Champion(
        version=version.version,
        model_type=tags["model_type"],
        features=features,
        alert_threshold=float(tags["alert_threshold"]),
        score_reference=reference,
        predict=predict,
    )
