"""Metric tests. Every expected value is worked out by hand from the arrays."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fraud import evaluate as ev

# Five transactions, two of them fraud (amounts 100 and 50, total 150).
Y = np.array([1, 0, 1, 0, 0])
SCORES = np.array([0.9, 0.8, 0.7, 0.1, 0.2])
AMOUNTS = np.array([100.0, 5.0, 50.0, 5.0, 5.0])


def test_average_precision_matches_hand_calculation():
    # Ranked: fraud, legit, fraud, legit, legit.
    # Precision at each fraud hit: 1/1 and 2/3 -> AP = (1 + 2/3) / 2
    assert ev.average_precision(Y, SCORES) == pytest.approx((1 + 2 / 3) / 2)


def test_average_precision_is_nan_without_positives():
    assert math.isnan(ev.average_precision(np.zeros(3), np.array([0.1, 0.2, 0.3])))


def test_budget_of_40_percent_alerts_top_two():
    m = ev.at_budget(Y, SCORES, AMOUNTS, budget=0.4)
    assert m["alerts"] == 2
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["value_recall"] == pytest.approx(100 / 150)


def test_budget_rounds_alert_count_up():
    m = ev.at_budget(Y, SCORES, AMOUNTS, budget=0.5)  # ceil(2.5) = 3 alerts
    assert m["alerts"] == 3
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(1.0)
    assert m["value_recall"] == pytest.approx(1.0)


def test_ties_give_exactly_the_budgeted_alert_count():
    mask = ev.top_k_mask(np.array([0.5, 0.5, 0.5, 0.5]), budget=0.5)
    assert mask.sum() == 2
    assert mask.tolist() == [True, True, False, False]  # stable: earlier rows first


def test_alert_count_is_exact_where_a_fraction_would_round_up():
    # 7/25 * 25 is 7.000000000000001 in floating point, so going through a
    # budget fraction would alert 8 rows instead of 7.
    assert math.ceil(7 / 25 * 25) == 8
    scores = np.linspace(1, 0, 25)
    m = ev.at_alert_count(np.zeros(25), scores, np.ones(25), k=7)
    assert m["alerts"] == 7


@pytest.mark.parametrize("budget", [0, -0.1, 1.5])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValueError):
        ev.top_k_mask(SCORES, budget)


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        ev.at_budget(Y, SCORES[:3], AMOUNTS, budget=0.4)


def test_binary_rule_metrics():
    flagged = np.array([1, 0, 0, 0, 1])
    m = ev.binary_rule_metrics(Y, flagged, AMOUNTS)
    assert m["alerts"] == 2
    assert m["alert_rate"] == pytest.approx(0.4)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["value_recall"] == pytest.approx(100 / 150)


def test_cost_curve_by_hand():
    # review cost 10 per alert, every missed fraud fully lost.
    # k=0: 150 | k=1: 10+50=60 | k=2: 20+50=70 | k=3: 30+0=30 | k=4: 40 | k=5: 50
    k, cost = ev.cost_curve(Y, SCORES, AMOUNTS, review_cost=10, loss_fraction=1.0)
    assert k.tolist() == [0, 1, 2, 3, 4, 5]
    assert cost.tolist() == pytest.approx([150, 60, 70, 30, 40, 50])


def test_cost_optimal_picks_minimum():
    best = ev.cost_optimal(Y, SCORES, AMOUNTS, review_cost=10, loss_fraction=1.0)
    assert best["alerts"] == 3
    assert best["threshold"] == pytest.approx(0.7)
    assert best["total_cost"] == pytest.approx(30)
    assert best["cost_without_model"] == pytest.approx(150)
    assert best["saving"] == pytest.approx(120)


def test_cost_optimal_alerts_nothing_when_review_is_too_expensive():
    best = ev.cost_optimal(Y, SCORES, AMOUNTS, review_cost=1_000, loss_fraction=1.0)
    assert best["alerts"] == 0
    assert best["threshold"] == math.inf


def test_threshold_for_budget_matches_top_k():
    scores = np.array([0.1, 0.9, 0.4, 0.7, 0.2])
    # top 40% of 5 rows = 2 rows: 0.9 and 0.7 -> threshold 0.7
    assert ev.threshold_for_budget(scores, 0.4) == pytest.approx(0.7)
    assert (scores >= ev.threshold_for_budget(scores, 0.4)).sum() == 2


def test_score_reference_bins_are_open_ended_and_sum_to_one():
    scores = np.linspace(0, 1, 1000)
    ref = ev.score_reference(scores, n_bins=10)
    assert ref["edges"][0] == -math.inf
    assert ref["edges"][-1] == math.inf
    assert sum(ref["proportions"]) == pytest.approx(1.0)
    assert all(p == pytest.approx(0.1) for p in ref["proportions"])


def test_psi_is_zero_for_the_same_distribution_and_grows_with_shift():
    rng = np.random.default_rng(0)
    base = rng.random(20_000)
    ref = ev.score_reference(base)
    assert ev.population_stability_index(ref, base) == pytest.approx(0.0, abs=1e-12)
    small = ev.population_stability_index(ref, np.clip(base + 0.02, 0, 1))
    large = ev.population_stability_index(ref, np.clip(base + 0.3, 0, 1))
    assert 0 < small < 0.1 < 0.25 < large


def test_psi_by_hand():
    # Two bins, reference 50/50, new 80/20:
    # (0.8-0.5)ln(0.8/0.5) + (0.2-0.5)ln(0.2/0.5)
    ref = {"edges": [-math.inf, 0.5, math.inf], "proportions": [0.5, 0.5]}
    new = np.array([0.1] * 8 + [0.9] * 2)
    expected = 0.3 * math.log(0.8 / 0.5) + (-0.3) * math.log(0.2 / 0.5)
    assert ev.population_stability_index(ref, new) == pytest.approx(expected)


def test_bootstrap_ci_is_reproducible_and_brackets_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    s = y * 0.6 + rng.random(400) * 0.5
    a = ev.bootstrap_ci(ev.average_precision, (y, s), n_boot=200, seed=1)
    b = ev.bootstrap_ci(ev.average_precision, (y, s), n_boot=200, seed=1)
    assert a == b
    assert a["ci_low"] <= a["estimate"] <= a["ci_high"]


def test_paired_bootstrap_detects_a_clearly_better_model():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    weak = rng.random(500)  # no signal
    strong = y + rng.random(500) * 0.1  # near-perfect ranking
    result = ev.paired_bootstrap_difference(ev.average_precision, y, weak, strong, n_boot=200)
    assert result["difference"] > 0
    assert result["ci_low"] > 0
    assert result["share_b_not_better"] == 0.0
