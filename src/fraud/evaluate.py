"""Evaluation metrics for a rare-event classifier.

Accuracy is useless here: with fraud well under 1% of transactions, a model
that never raises an alert is over 99% accurate. The questions a fraud team
actually asks are:

* ranking quality across all thresholds   -> average precision (PR-AUC)
* "we can review N alerts; what do we catch?" -> precision / recall at a budget
* "how much money do we stop?"               -> value recall at a budget
* "which threshold minimises total cost?"    -> cost-optimal threshold

Everything takes plain NumPy arrays so it can be unit-tested without Spark.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score


def _as_arrays(*arrays: np.ndarray) -> list[np.ndarray]:
    out = [np.asarray(a) for a in arrays]
    n = len(out[0])
    if any(len(a) != n for a in out):
        raise ValueError("all inputs must have the same length")
    if n == 0:
        raise ValueError("inputs are empty")
    return out


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true, scores = _as_arrays(y_true, scores)
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def top_k_mask(scores: np.ndarray, budget: float) -> np.ndarray:
    """Boolean mask of the `ceil(budget * n)` highest scores."""
    if not 0 < budget <= 1:
        raise ValueError("budget must be in (0, 1]")
    (scores,) = _as_arrays(scores)
    return top_count_mask(scores, math.ceil(budget * len(scores)))


def top_count_mask(scores: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask of exactly the `k` highest scores.

    Ties are broken by position (stable sort) so the alert count is exactly
    k, as it would be for an analyst team with fixed capacity.
    """
    (scores,) = _as_arrays(scores)
    if not 0 <= k <= len(scores):
        raise ValueError("k must be between 0 and the number of rows")
    order = np.argsort(-scores, kind="stable")
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[:k]] = True
    return mask


def at_budget(
    y_true: np.ndarray, scores: np.ndarray, amounts: np.ndarray, budget: float
) -> dict[str, float]:
    """Precision, recall and value recall when only the top `budget` share is alerted."""
    if not 0 < budget <= 1:
        raise ValueError("budget must be in (0, 1]")
    (scores,) = _as_arrays(scores)
    result = at_alert_count(y_true, scores, amounts, math.ceil(budget * len(scores)))
    return {"budget": budget, **result}


def at_alert_count(
    y_true: np.ndarray, scores: np.ndarray, amounts: np.ndarray, k: int
) -> dict[str, float]:
    """Precision, recall and value recall when exactly the top `k` rows are alerted."""
    y_true, scores, amounts = _as_arrays(y_true, scores, amounts)
    alert = top_count_mask(scores, k)
    fraud = y_true == 1
    caught = alert & fraud
    fraud_value = float(amounts[fraud].sum())
    return {
        "alerts": int(alert.sum()),
        "precision": float(caught.sum() / alert.sum()) if alert.sum() else float("nan"),
        "recall": float(caught.sum() / fraud.sum()) if fraud.sum() else float("nan"),
        "value_recall": float(amounts[caught].sum() / fraud_value) if fraud_value else float("nan"),
    }


def binary_rule_metrics(
    y_true: np.ndarray, flagged: np.ndarray, amounts: np.ndarray
) -> dict[str, float]:
    """Metrics for an existing yes/no rule, for comparison with a model."""
    y_true, flagged, amounts = _as_arrays(y_true, flagged, amounts)
    alert = flagged.astype(bool)
    fraud = y_true == 1
    caught = alert & fraud
    fraud_value = float(amounts[fraud].sum())
    return {
        "alerts": int(alert.sum()),
        "alert_rate": float(alert.mean()),
        "precision": float(caught.sum() / alert.sum()) if alert.sum() else float("nan"),
        "recall": float(caught.sum() / fraud.sum()) if fraud.sum() else float("nan"),
        "value_recall": float(amounts[caught].sum() / fraud_value) if fraud_value else float("nan"),
    }


def cost_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    review_cost: float,
    loss_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Total cost at every possible alert count k = 0..n.

    cost(k) = k * review_cost + loss_fraction * (fraud value NOT among the top k)

    `review_cost` is the cost of an analyst checking one alert and
    `loss_fraction` the share of a missed fraud's amount that is lost. Both are
    business assumptions, passed in explicitly rather than hidden in the code.
    """
    y_true, scores, amounts = _as_arrays(y_true, scores, amounts)
    order = np.argsort(-scores, kind="stable")
    fraud_value_sorted = np.where(y_true[order] == 1, amounts[order], 0.0)
    caught_value = np.concatenate([[0.0], np.cumsum(fraud_value_sorted)])
    total_fraud_value = caught_value[-1]
    k = np.arange(len(scores) + 1)
    cost = k * review_cost + loss_fraction * (total_fraud_value - caught_value)
    return k, cost


def cost_optimal(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    review_cost: float,
    loss_fraction: float,
) -> dict[str, float]:
    """The alert count (and score threshold) that minimises total cost."""
    y_true, scores, amounts = _as_arrays(y_true, scores, amounts)
    k, cost = cost_curve(y_true, scores, amounts, review_cost, loss_fraction)
    best = int(np.argmin(cost))
    order = np.argsort(-scores, kind="stable")
    threshold = float(scores[order[best - 1]]) if best > 0 else float("inf")
    no_model_cost = float(cost[0])  # review nothing, lose every fraud
    return {
        "alerts": best,
        "alert_rate": best / len(scores),
        "threshold": threshold,
        "total_cost": float(cost[best]),
        "cost_without_model": no_model_cost,
        "saving": no_model_cost - float(cost[best]),
    }


def bootstrap_ci(
    metric: Callable[..., float],
    arrays: tuple[np.ndarray, ...],
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 7,
) -> dict[str, float]:
    """Percentile bootstrap confidence interval for `metric(*arrays)`.

    Rows are resampled with replacement; every array is resampled with the
    same indices so they stay aligned.
    """
    arrays = tuple(_as_arrays(*arrays))
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    point = metric(*arrays)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        samples[i] = metric(*(a[idx] for a in arrays))
    samples = samples[~np.isnan(samples)]
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return {"estimate": float(point), "ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot}


def paired_bootstrap_difference(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 7,
) -> dict[str, float]:
    """Is model B better than model A on the same rows?

    Both models are scored on the SAME resample each round, so the comparison
    is paired. `share_b_not_better` is the fraction of resamples where B did
    not beat A: a small value is evidence that B is genuinely better.
    """
    y_true, scores_a, scores_b = _as_arrays(y_true, scores_a, scores_b)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = metric(y_true[idx], scores_b[idx]) - metric(y_true[idx], scores_a[idx])
    diffs = diffs[~np.isnan(diffs)]
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {
        "difference": float(metric(y_true, scores_b) - metric(y_true, scores_a)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "share_b_not_better": float(np.mean(diffs <= 0)),
        "n_boot": n_boot,
    }
