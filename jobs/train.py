"""Train, compare and register the fraud models.

    make train

1. Split the feature table by time (train / valid / test).
2. Fit the logistic-regression baseline and distributed XGBoost.
3. Pick the champion on VALIDATION average precision only.
4. Report every model on the untouched TEST hours, with bootstrap
   confidence intervals, alert-budget metrics, the existing rule as a
   baseline, and a cost analysis.
5. Leakage check: retrain XGBoost WITH the balance columns, to show how much
   of a "too good" score comes from them. That model is never registered.
6. Log everything to MLflow; register the champion as `fraud-detector`
   with the alias `champion`.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from pyspark.sql import DataFrame, SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from fraud import evaluate as ev  # noqa: E402
from fraud.export import booster_scores, logistic_scorer, xgboost_booster  # noqa: E402
from fraud.features import FEATURE_COLUMNS  # noqa: E402
from fraud.io import EVIDENCE_DIR, PROCESSED_DIR, write_json  # noqa: E402
from fraud.model import (  # noqa: E402
    MODEL_SCOPE_TYPES,
    XGB_PARAMS,
    class_weight,
    logistic_pipeline,
    score,
    xgboost_pipeline,
)
from fraud.schema import LABEL, LEAKY_COLUMNS, RULE_FLAG  # noqa: E402
from fraud.split import assign_split  # noqa: E402

EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "fraud-detection")
REGISTERED_MODEL = os.getenv("REGISTERED_MODEL", "fraud-detector")
BUDGETS = (0.001, 0.005, 0.01)  # share of in-scope transactions analysts can review
# Illustrative cost assumptions (PaySim has no currency): review cost per alert
# in the same units as `amount`, and the share of a missed fraud that is lost.
REVIEW_COSTS = (10.0, 100.0, 1000.0)
LOSS_FRACTION = 1.0
N_BOOT = int(os.getenv("N_BOOT", "500"))
TRAIN_SHARE, VALID_SHARE = 0.70, 0.15  # by rows, cut at whole steps


def split_boundaries(df: DataFrame) -> tuple[int, int]:
    """Step boundaries that put ~70% / 15% / 15% of rows in train / valid / test.

    Overridable with TRAIN_END_STEP / VALID_END_STEP. Cutting at whole steps
    keeps every hour entirely in one split.
    """
    if os.getenv("TRAIN_END_STEP") and os.getenv("VALID_END_STEP"):
        return int(os.environ["TRAIN_END_STEP"]), int(os.environ["VALID_END_STEP"])
    per_step = df.groupBy("step").count().orderBy("step").toPandas()
    share = per_step["count"].cumsum() / per_step["count"].sum()
    train_end = int(per_step.loc[share >= TRAIN_SHARE, "step"].iloc[0])
    valid_end = int(per_step.loc[share >= TRAIN_SHARE + VALID_SHARE, "step"].iloc[0])
    return train_end, valid_end


def collect_scores(model, df: DataFrame, features=FEATURE_COLUMNS) -> pd.DataFrame:
    # Features are collected too, so the exported artefact can be checked
    # against these exact Spark scores.
    return (
        score(model, df)
        .select("txn_id", "step", LABEL, RULE_FLAG, "amount", "score", *features)
        .toPandas()
    )


def summarise(name: str, valid: pd.DataFrame, test: pd.DataFrame) -> dict:
    y, s, a = test[LABEL].to_numpy(), test["score"].to_numpy(), test["amount"].to_numpy()
    out = {
        "model": name,
        "valid_average_precision": ev.average_precision(valid[LABEL], valid["score"]),
        "test_average_precision": ev.bootstrap_ci(ev.average_precision, (y, s), n_boot=N_BOOT),
        "test_at_budget": [ev.at_budget(y, s, a, b) for b in BUDGETS],
        "test_cost_optimal": {
            str(c): ev.cost_optimal(y, s, a, review_cost=c, loss_fraction=LOSS_FRACTION)
            for c in REVIEW_COSTS
        },
    }
    rule_alerts = int(test[RULE_FLAG].sum())
    if rule_alerts:
        # The model judged at exactly the rule's alert volume: like for like.
        out["test_at_rule_volume"] = ev.at_alert_count(y, s, a, rule_alerts)
    return out


def plot_pr(curves: dict[str, pd.DataFrame], rule: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, frame in curves.items():
        p, r, _ = precision_recall_curve(frame[LABEL], frame["score"])
        ax.plot(r, p, label=name)
    if rule["alerts"]:
        ax.scatter(
            [rule["recall"]],
            [rule["precision"]],
            color="black",
            zorder=5,
            label=f"existing rule ({rule['alerts']} alerts)",
        )
    ax.set_xlabel("Recall (share of fraud caught)")
    ax.set_ylabel("Precision (share of alerts that are fraud)")
    ax.set_title("Precision-recall on the test hours")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(importance: pd.DataFrame, path: Path) -> None:
    ordered = importance.sort_values("gain")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(ordered["feature"], ordered["gain"])
    ax.set_xlabel("Total gain")
    ax.set_title("XGBoost feature importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    spark = SparkSession.builder.appName("fraud-train").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    started = time.perf_counter()

    features = spark.read.parquet(str(PROCESSED_DIR / "features.parquet"))
    in_scope = features.filter(F.col("txn_type").isin(*MODEL_SCOPE_TYPES))
    train_end, valid_end = split_boundaries(in_scope)
    data = assign_split(in_scope, train_end, valid_end).cache()

    split_summary = {
        r["split"]: {"rows": r["rows"], "fraud": r["fraud"], "fraud_rate": r["fraud"] / r["rows"]}
        for r in data.groupBy("split")
        .agg(F.count(F.lit(1)).alias("rows"), F.sum(LABEL).alias("fraud"))
        .collect()
    }
    split_summary["boundaries"] = {"train_end_step": train_end, "valid_end_step": valid_end}
    print(json.dumps(split_summary, indent=2))

    train, pos_weight = class_weight(data.filter(F.col("split") == "train"), LABEL)
    valid = data.filter(F.col("split") == "valid")
    test = data.filter(F.col("split") == "test")

    # --- Models -------------------------------------------------------------
    lr_model = logistic_pipeline(FEATURE_COLUMNS, LABEL).fit(train)

    xgb_train = train.withColumn("is_valid", F.lit(False)).unionByName(
        valid.withColumn("weight", F.lit(1.0)).withColumn("is_valid", F.lit(True))
    )
    xgb_model = xgboost_pipeline(FEATURE_COLUMNS, LABEL, pos_weight).fit(xgb_train)

    scored = {
        "logistic_regression": (collect_scores(lr_model, valid), collect_scores(lr_model, test)),
        "xgboost": (collect_scores(xgb_model, valid), collect_scores(xgb_model, test)),
    }
    results = {name: summarise(name, v, t) for name, (v, t) in scored.items()}

    champion = max(results, key=lambda n: results[n]["valid_average_precision"])
    challenger = next(n for n in results if n != champion)

    test_lr, test_xgb = scored["logistic_regression"][1], scored["xgboost"][1]
    comparison = ev.paired_bootstrap_difference(
        ev.average_precision,
        test_lr[LABEL].to_numpy(),
        test_lr["score"].to_numpy(),
        test_xgb.set_index("txn_id").loc[test_lr["txn_id"], "score"].to_numpy(),
        n_boot=N_BOOT,
    )
    rule = ev.binary_rule_metrics(
        test_xgb[LABEL].to_numpy(), test_xgb[RULE_FLAG].to_numpy(), test_xgb["amount"].to_numpy()
    )

    # --- Leakage check: the same XGBoost, plus the balance columns ----------
    balances = spark.read.parquet(str(PROCESSED_DIR / "transactions.parquet")).select(
        "txn_id", *LEAKY_COLUMNS
    )
    leaky_cols = (*FEATURE_COLUMNS, *LEAKY_COLUMNS)
    leaky_model = xgboost_pipeline(leaky_cols, LABEL, pos_weight).fit(
        xgb_train.join(balances, "txn_id")
    )
    leaky_test = collect_scores(leaky_model, test.join(balances, "txn_id"), leaky_cols)
    leakage = {
        "features": list(leaky_cols),
        "test_average_precision": ev.average_precision(leaky_test[LABEL], leaky_test["score"]),
        "test_at_budget": [
            ev.at_budget(leaky_test[LABEL], leaky_test["score"], leaky_test["amount"], b)
            for b in BUDGETS
        ],
    }

    # --- Deployable artefacts, checked against the Spark scores --------------
    booster = xgboost_booster(xgb_model)
    lr_scorer = logistic_scorer(lr_model, FEATURE_COLUMNS)
    parity = {
        "xgboost": float(
            np.max(
                np.abs(
                    booster_scores(booster, test_xgb, FEATURE_COLUMNS)
                    - test_xgb["score"].to_numpy()
                )
            )
        ),
        "logistic_regression": float(
            np.max(np.abs(lr_scorer.score(test_lr) - test_lr["score"].to_numpy()))
        ),
    }
    print(f"max |exported - Spark| score difference on test: {parity}")
    if max(parity.values()) > 1e-4:
        raise RuntimeError(f"exported model does not reproduce Spark scores: {parity}")

    # --- Feature importance --------------------------------------------------
    gain = booster.get_score(importance_type="total_gain")
    importance = pd.DataFrame(
        {
            "feature": list(FEATURE_COLUMNS),
            "gain": [gain.get(f"f{i}", 0.0) for i in range(len(FEATURE_COLUMNS))],
        }
    ).sort_values("gain", ascending=False)

    metrics = {
        "split": split_summary,
        "positive_class_weight": pos_weight,
        "models": results,
        "champion": champion,
        "xgboost_minus_logistic_average_precision": comparison,
        "existing_rule_on_test": rule,
        "leakage_check": leakage,
        "export_parity_max_abs_diff": parity,
        "assumptions": {
            "alert_budgets": BUDGETS,
            "review_costs": REVIEW_COSTS,
            "loss_fraction": LOSS_FRACTION,
            "scope": MODEL_SCOPE_TYPES,
        },
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    write_json(metrics, EVIDENCE_DIR / "metrics.json")
    importance.to_csv(EVIDENCE_DIR / "feature_importance.csv", index=False)
    plot_pr(
        {"XGBoost": test_xgb, "Logistic regression": test_lr},
        rule,
        EVIDENCE_DIR / "pr_curve_test.png",
    )
    plot_importance(importance, EVIDENCE_DIR / "feature_importance.png")

    # --- MLflow ---------------------------------------------------------------
    mlflow.set_experiment(EXPERIMENT)
    data_profile = EVIDENCE_DIR / "data_profile.json"
    data_sha = (
        json.loads(data_profile.read_text())["source_sha256"] if data_profile.exists() else ""
    )
    common = {
        "features": ",".join(FEATURE_COLUMNS),
        "scope": ",".join(MODEL_SCOPE_TYPES),
        "train_end_step": train_end,
        "valid_end_step": valid_end,
        "data_sha256": data_sha,
    }

    model_uris = {}
    for name in results:
        with mlflow.start_run(run_name=name):
            mlflow.log_params(common)
            mlflow.set_tag("role", "champion" if name == champion else "challenger")
            r = results[name]
            mlflow.log_metric("valid_average_precision", r["valid_average_precision"])
            mlflow.log_metric("test_average_precision", r["test_average_precision"]["estimate"])
            for m in r["test_at_budget"]:
                tag = f"{m['budget']:.3%}".replace(".", "_").replace("%", "pct")
                mlflow.log_metric(f"test_recall_at_{tag}", m["recall"])
                mlflow.log_metric(f"test_precision_at_{tag}", m["precision"])
            mlflow.log_metric("export_parity_max_abs_diff", parity[name])
            if name == "xgboost":
                mlflow.log_params({f"xgb_{k}": v for k, v in XGB_PARAMS.items()})
                mlflow.log_param("scale_pos_weight", pos_weight)
                info = mlflow.xgboost.log_model(booster, name="model")
                mlflow.log_artifact(str(EVIDENCE_DIR / "feature_importance.png"))
            else:
                info = mlflow.pyfunc.log_model(
                    name="model",
                    python_model=lr_scorer,
                    code_paths=[str(Path(__file__).resolve().parents[1] / "src" / "fraud")],
                )
                coefficients = {
                    "features": list(FEATURE_COLUMNS),
                    "coefficients_on_standardised_inputs": lr_scorer.coefficients.tolist(),
                    "intercept": lr_scorer.intercept,
                }
                with tempfile.TemporaryDirectory() as tmp:
                    coef_path = Path(tmp) / "logistic_coefficients.json"
                    coef_path.write_text(json.dumps(coefficients, indent=2))
                    mlflow.log_artifact(str(coef_path))
            mlflow.log_artifact(str(EVIDENCE_DIR / "metrics.json"))
            mlflow.log_artifact(str(EVIDENCE_DIR / "pr_curve_test.png"))
            model_uris[name] = info.model_uri

    with mlflow.start_run(run_name="leakage-check (not deployable)"):
        mlflow.log_params({**common, "features": ",".join(leaky_cols)})
        mlflow.set_tag("role", "leakage-check")
        mlflow.log_metric("test_average_precision", leakage["test_average_precision"])

    # Whichever model won on validation is registered: both are deployable.
    version = mlflow.register_model(model_uris[champion], REGISTERED_MODEL)
    client = MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL, "champion", version.version)
    client.set_model_version_tag(REGISTERED_MODEL, version.version, "model_type", champion)
    print(f"registered {REGISTERED_MODEL} v{version.version} ({champion}) as @champion")

    print(
        json.dumps(
            {
                "champion": champion,
                "challenger": challenger,
                "test_average_precision": {
                    n: r["test_average_precision"] for n, r in results.items()
                },
                "leakage_check_average_precision": leakage["test_average_precision"],
                "existing_rule": rule,
                "runtime_seconds": metrics["runtime_seconds"],
            },
            indent=2,
        )
    )
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
