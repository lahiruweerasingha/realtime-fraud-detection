"""The exported artefacts must score exactly like the Spark pipelines.

If they did not, the model that gets deployed would not be the model that was
evaluated: a train/serve mismatch that no metric would reveal.
"""

from __future__ import annotations

import numpy as np
import pytest
from pyspark.sql import functions as F

from fraud.export import booster_scores, logistic_scorer, xgboost_booster
from fraud.model import class_weight, logistic_pipeline, score, xgboost_pipeline

FEATURES = ("x1", "x2", "constant")


@pytest.fixture(scope="module")
def frame(spark):
    rng = np.random.default_rng(3)
    n = 400
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n) * 10
    label = ((x1 + x2 / 10 + rng.normal(scale=0.5, size=n)) > 1.2).astype(int)
    rows = [
        (int(i), float(a), float(b), 5.0, int(y))
        for i, (a, b, y) in enumerate(zip(x1, x2, label, strict=True))
    ]
    df = spark.createDataFrame(rows, ["txn_id", "x1", "x2", "constant", "is_fraud"])
    weighted, pos_weight = class_weight(df, "is_fraud")
    return weighted, pos_weight


def test_class_weight_balances_the_classes(frame):
    weighted, pos_weight = frame
    totals = {
        r["is_fraud"]: r["w"]
        for r in weighted.groupBy("is_fraud").agg(F.sum("weight").alias("w")).collect()
    }
    assert totals[1] == pytest.approx(totals[0])
    assert pos_weight > 1


def test_logistic_export_matches_spark(frame):
    weighted, _ = frame
    model = logistic_pipeline(FEATURES, "is_fraud").fit(weighted)
    spark_scores = score(model, weighted).select("txn_id", "score").toPandas().sort_values("txn_id")

    exported = logistic_scorer(model, FEATURES)
    local = exported.score(weighted.toPandas().sort_values("txn_id"))

    # The zero-variance column must not produce NaNs.
    assert not np.isnan(local).any()
    np.testing.assert_allclose(local, spark_scores["score"].to_numpy(), rtol=1e-9, atol=1e-12)


def test_xgboost_booster_matches_spark_with_early_stopping(frame):
    weighted, pos_weight = frame
    data = weighted.withColumn("is_valid", (F.col("txn_id") % 5) == 0)
    # Deep trees + many rounds + early stopping: training runs past the best
    # round, which is exactly the case where an untrimmed booster scores
    # differently from Spark. This configuration caught that bug.
    params = {
        "max_depth": 6,
        "n_estimators": 300,
        "learning_rate": 0.3,
        "eval_metric": "aucpr",
        "early_stopping_rounds": 5,
        "random_state": 7,
    }
    model = xgboost_pipeline(FEATURES, "is_fraud", pos_weight, num_workers=1, params=params).fit(
        data
    )
    full = model.stages[-1].get_booster()
    assert full.attr("best_iteration") is not None
    assert full.num_boosted_rounds() > int(full.attr("best_iteration")) + 1  # it ran past the best

    spark_scores = score(model, data).select("txn_id", "score").toPandas().sort_values("txn_id")
    rows = data.toPandas().sort_values("txn_id")
    expected = spark_scores["score"].to_numpy()

    # The untrimmed booster must disagree, or this test is not exercising the bug.
    untrimmed = booster_scores(full, rows, FEATURES)
    assert not np.allclose(untrimmed, expected, rtol=1e-5, atol=1e-6)

    local = booster_scores(xgboost_booster(model), rows, FEATURES)
    np.testing.assert_allclose(local, expected, rtol=1e-5, atol=1e-6)
