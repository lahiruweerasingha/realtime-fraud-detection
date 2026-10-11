"""Feature tests on the hand-built fixture.

The fixture's destination account C900 receives four transactions:
    step 1  100  (txn A)        step 2  200  (txn C)
    step 1  300  (txn B)        step 26 400  (txn D, fraud)
so its history at each step can be checked by hand.
"""

from __future__ import annotations

import math

import pytest
from pyspark.sql import functions as F

from fraud.features import FEATURE_COLUMNS, build_features
from fraud.schema import LABEL, LEAKY_COLUMNS, RULE_FLAG, read_transactions


@pytest.fixture(scope="module")
def features(spark, fixture_csv):
    txns = read_transactions(spark, fixture_csv).withColumn(
        "txn_id", F.monotonically_increasing_id()
    )
    rows = build_features(spark, txns).collect()
    # Key each row by (step, amount): unique in the fixture.
    return {(r["step"], r["amount"]): r.asDict() for r in rows}


def test_no_leaky_column_or_label_is_a_feature():
    assert not set(FEATURE_COLUMNS) & set(LEAKY_COLUMNS)
    assert LABEL not in FEATURE_COLUMNS
    assert RULE_FLAG not in FEATURE_COLUMNS


def test_every_feature_column_is_produced(features):
    row = next(iter(features.values()))
    assert set(FEATURE_COLUMNS) <= set(row)


def test_same_step_transactions_do_not_see_each_other(features):
    # A and B share step 1: neither may count the other as history.
    for key in [(1, 100.0), (1, 300.0)]:
        assert features[key]["dest_prior_txn_count"] == 0
        assert features[key]["dest_prior_amount"] == 0
        assert features[key]["amount_vs_dest_mean"] == 0


def test_history_uses_only_earlier_steps(features):
    c = features[(2, 200.0)]  # sees A and B (step 1)
    assert c["dest_prior_txn_count"] == 2
    assert c["dest_prior_amount"] == pytest.approx(400.0)
    assert c["amount_vs_dest_mean"] == pytest.approx(200.0 / 200.0)


def test_24_hour_window_excludes_older_history(features):
    d = features[(26, 400.0)]  # fraud row
    # All history: A, B, C.
    assert d["dest_prior_txn_count"] == 3
    assert d["dest_prior_amount"] == pytest.approx(600.0)
    assert d["amount_vs_dest_mean"] == pytest.approx(400.0 / 200.0)
    # Last 24 steps = steps 2..25: only C.
    assert d["dest_txn_count_24h"] == 1
    assert d["dest_amount_24h"] == pytest.approx(200.0)


def test_origin_history_and_merchant_flag(features):
    payment = features[(2, 50.0)]  # C1 paid merchant M100; C1 also sent A at step 1
    assert payment["orig_prior_txn_count"] == 1
    assert payment["is_merchant_dest"] == 1
    assert payment["dest_prior_txn_count"] == 1  # M100 was paid 10 at step 1
    assert features[(1, 100.0)]["is_merchant_dest"] == 0


def test_transaction_level_features(features):
    d = features[(26, 400.0)]
    assert d["hour_of_day"] == 26 % 24
    assert d["is_transfer"] == 1
    assert d["log_amount"] == pytest.approx(math.log1p(400.0))
    assert features[(30, 20.0)]["is_transfer"] == 0
