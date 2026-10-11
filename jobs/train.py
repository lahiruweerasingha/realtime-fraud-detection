"""Train, compare and register the fraud models.

    make train

1. Split the feature table by time (train / valid / test).
2. Fit the logistic-regression baseline and distributed XGBoost.
3. Pick the champion on VALIDATION average precision only.
4. Report every model on the untouched TEST hours, with bootstrap
   confidence intervals, alert-budget metrics, the existing rule as a
   baseline, and a cost analysis.
5. Two checks, retraining XGBoost with a changed feature set. Neither model
   is ever registered:
   - leakage: WITH the balance columns, to show how much of a "too good"
     score comes from them;
   - time of day: WITHOUT the hour features, to show how much of the score
     depends on PaySim's simulated daily cycle.
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
from matplotlib.ticker import PercentFormatter  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from pyspark.sql import DataFrame, SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from fraud import evaluate as ev  # noqa: E402
from fraud.export import booster_scores, logistic_scorer, xgboost_booster  # noqa: E402
from fraud.features import FEATURE_COLUMNS, TIME_OF_DAY_FEATURES  # noqa: E402
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
ALERT_BUDGET = 0.005  # the operating point a scoring service alerts at
# Illustrative cost assumptions (PaySim has no currency): review cost per alert
# in the same units as `amount`, and the share of a missed fraud that is lost.
REVIEW_COSTS = (10.0, 100.0, 1000.0)
LOSS_FRACTION = 1.0
N_BOOT = int(os.getenv("N_BOOT", "500"))
TRAIN_SHARE, VALID_SHARE = 0.70, 0.15  # by rows, cut at whole steps

# After day 17 (step 408) legitimate in-scope volume collapses from ~180k
# rows a day to under 25k while fraud continues at the same pace, so the
# fraud rate in those hours is several times higher (notebooks/01_eda.ipynb).
# Test metrics are reported separately for the normal-volume hours (the
# headline, closest to a realistic base rate) and the low-volume tail.
NORMAL_VOLUME_LAST_STEP = int(os.getenv("NORMAL_VOLUME_LAST_STEP", "408"))


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


def test_segments(test: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The test period split at NORMAL_VOLUME_LAST_STEP; empty segments are left out
    (e.g. a dataset that ends before the low-volume tail)."""
    normal = test["step"] <= NORMAL_VOLUME_LAST_STEP
    segments = {
        "test_normal_volume": test[normal],
        "test_low_volume_tail": test[~normal],
        "test_all": test,
    }
    return {name: frame for name, frame in segments.items() if len(frame)}


def evaluate_frame(frame: pd.DataFrame) -> dict:
    y, s, a = frame[LABEL].to_numpy(), frame["score"].to_numpy(), frame["amount"].to_numpy()
    out = {
        "rows": len(frame),
        "fraud": int(y.sum()),
        "fraud_rate": float(y.mean()),
        "average_precision": ev.bootstrap_ci(ev.average_precision, (y, s), n_boot=N_BOOT),
        "at_budget": [ev.at_budget(y, s, a, b) for b in BUDGETS],
        "cost_optimal": {
            str(c): ev.cost_optimal(y, s, a, review_cost=c, loss_fraction=LOSS_FRACTION)
            for c in REVIEW_COSTS
        },
    }
    rule_alerts = int(frame[RULE_FLAG].sum())
    if rule_alerts:
        # The model judged at exactly the rule's alert volume: like for like.
        out["at_rule_volume"] = ev.at_alert_count(y, s, a, rule_alerts)
    return out


def summarise(name: str, valid: pd.DataFrame, test: pd.DataFrame) -> dict:
    return {
        "model": name,
        "valid_average_precision": ev.average_precision(valid[LABEL], valid["score"]),
        **{seg: evaluate_frame(frame) for seg, frame in test_segments(test).items()},
    }


def plot_pr(curves: dict[str, pd.DataFrame], rule: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, frame in curves.items():
        p, r, _ = precision_recall_curve(frame[LABEL], frame["score"])
        ax.plot(r, p, label=name)
    if rule["alerts"]:
        n = rule["alerts"]
        ax.scatter(
            [rule["recall"]],
            [rule["precision"]],
            color="black",
            zorder=5,
            label=f"existing rule ({n} alert{'s' if n != 1 else ''})",
        )
    ax.set_xlabel("Recall (share of fraud caught)")
    ax.set_ylabel("Precision (share of alerts that are fraud)")
    ax.set_title("Precision-recall, normal-volume test hours")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(importance: pd.DataFrame, path: Path) -> None:
    ordered = importance.sort_values("gain")
    share = ordered["gain"] / ordered["gain"].sum()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(ordered["feature"], share)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Share of total gain")
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

    # Alert threshold and score reference, both from VALIDATION scores only.
    champion_valid = scored[champion][0]["score"].to_numpy()
    monitoring = {
        "alert_budget": ALERT_BUDGET,
        "alert_threshold": ev.threshold_for_budget(champion_valid, ALERT_BUDGET),
        "score_reference": ev.score_reference(champion_valid),
    }

    test_lr, test_xgb = scored["logistic_regression"][1], scored["xgboost"][1]
    # Align the two models row by row, so every comparison is paired.
    paired = test_lr[["txn_id", "step", LABEL, RULE_FLAG, "amount", "score"]].merge(
        test_xgb[["txn_id", "score"]], on="txn_id", suffixes=("_lr", "_xgb")
    )
    # Kept (gitignored) for any later analysis without retraining.
    paired.to_parquet(PROCESSED_DIR / "test_scores.parquet", index=False)
    comparison, rule = {}, {}
    for seg, frame in test_segments(paired).items():
        comparison[seg] = ev.paired_bootstrap_difference(
            ev.average_precision,
            frame[LABEL].to_numpy(),
            frame["score_lr"].to_numpy(),
            frame["score_xgb"].to_numpy(),
            n_boot=N_BOOT,
        )
        rule[seg] = ev.binary_rule_metrics(
            frame[LABEL].to_numpy(), frame[RULE_FLAG].to_numpy(), frame["amount"].to_numpy()
        )

    # --- Checks: the same XGBoost with a changed feature set -----------------
    def feature_check(columns: tuple[str, ...], train_df: DataFrame, test_df: DataFrame) -> dict:
        model = xgboost_pipeline(columns, LABEL, pos_weight).fit(train_df)
        scored_test = collect_scores(model, test_df, columns)
        result: dict = {"features": list(columns)}
        for seg, frame in test_segments(scored_test).items():
            result[seg] = {
                "average_precision": ev.average_precision(frame[LABEL], frame["score"]),
                "at_budget": [
                    ev.at_budget(frame[LABEL], frame["score"], frame["amount"], b) for b in BUDGETS
                ],
            }
        return result

    balances = spark.read.parquet(str(PROCESSED_DIR / "transactions.parquet")).select(
        "txn_id", *LEAKY_COLUMNS
    )
    leaky_cols = (*FEATURE_COLUMNS, *LEAKY_COLUMNS)
    leakage = feature_check(
        leaky_cols, xgb_train.join(balances, "txn_id"), test.join(balances, "txn_id")
    )
    no_time_cols = tuple(c for c in FEATURE_COLUMNS if c not in TIME_OF_DAY_FEATURES)
    without_time_of_day = feature_check(no_time_cols, xgb_train, test)

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
        "without_time_of_day_check": without_time_of_day,
        "export_parity_max_abs_diff": parity,
        "monitoring": monitoring,
        "assumptions": {
            "alert_budgets": BUDGETS,
            "review_costs": REVIEW_COSTS,
            "loss_fraction": LOSS_FRACTION,
            "scope": MODEL_SCOPE_TYPES,
            "normal_volume_last_step": NORMAL_VOLUME_LAST_STEP,
        },
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    write_json(metrics, EVIDENCE_DIR / "metrics.json")
    importance.to_csv(EVIDENCE_DIR / "feature_importance.csv", index=False)
    normal = "test_normal_volume"
    plot_pr(
        {
            "XGBoost": test_segments(test_xgb)[normal],
            "Logistic regression": test_segments(test_lr)[normal],
        },
        rule[normal],
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
            for seg in test_segments(test_lr):
                mlflow.log_metric(
                    f"{seg}_average_precision", r[seg]["average_precision"]["estimate"]
                )
                for m in r[seg]["at_budget"]:
                    tag = f"{m['budget']:.3%}".replace(".", "_").replace("%", "pct")
                    mlflow.log_metric(f"{seg}_recall_at_{tag}", m["recall"])
                    mlflow.log_metric(f"{seg}_precision_at_{tag}", m["precision"])
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
            if name == champion:
                mlflow.log_dict(monitoring, "monitoring/score_reference.json")
            model_uris[name] = info.model_uri

    for run_name, check in [
        ("leakage-check (not deployable)", leakage),
        ("without-time-of-day check (not deployable)", without_time_of_day),
    ]:
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({**common, "features": ",".join(check["features"])})
            mlflow.set_tag("role", "check")
            for seg in test_segments(test_lr):
                mlflow.log_metric(f"{seg}_average_precision", check[seg]["average_precision"])

    # Whichever model won on validation is registered: both are deployable.
    # The alert threshold and the score reference travel with the model
    # version, so a scoring service uses exactly what was evaluated.
    version = mlflow.register_model(model_uris[champion], REGISTERED_MODEL)
    client = MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL, "champion", version.version)
    client.set_model_version_tag(REGISTERED_MODEL, version.version, "model_type", champion)
    client.set_model_version_tag(
        REGISTERED_MODEL, version.version, "features", ",".join(FEATURE_COLUMNS)
    )
    client.set_model_version_tag(
        REGISTERED_MODEL, version.version, "alert_threshold", str(monitoring["alert_threshold"])
    )
    client.set_model_version_tag(
        REGISTERED_MODEL, version.version, "alert_budget", str(monitoring["alert_budget"])
    )
    print(f"registered {REGISTERED_MODEL} v{version.version} ({champion}) as @champion")

    print(
        json.dumps(
            {
                "champion": champion,
                "challenger": challenger,
                "valid_average_precision": {
                    n: r["valid_average_precision"] for n, r in results.items()
                },
                "average_precision_by_test_segment": {
                    n: {seg: r[seg]["average_precision"] for seg in test_segments(test_lr)}
                    for n, r in results.items()
                },
                "leakage_check_average_precision": {
                    seg: leakage[seg]["average_precision"] for seg in test_segments(test_lr)
                },
                "without_time_of_day_average_precision": {
                    seg: without_time_of_day[seg]["average_precision"]
                    for seg in test_segments(test_lr)
                },
                "alert_threshold": monitoring["alert_threshold"],
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
