from __future__ import annotations

import pytest
from pyspark.sql import Row

from fraud.schema import RENAMES, read_transactions
from fraud.split import assign_split, split_name
from fraud.validate import failed_checks, profile


@pytest.fixture(scope="module")
def fixture_profile(spark, fixture_csv):
    return profile(read_transactions(spark, fixture_csv))


def test_profile_counts(fixture_profile):
    p = fixture_profile
    assert p["rows"] == 7
    assert p["fraud_rows"] == 1
    assert p["flagged_rows"] == 1
    assert p["flagged_and_fraud"] == 1
    assert (p["min_step"], p["max_step"]) == (1, 30)
    assert p["distinct_orig_accounts"] == 6
    assert p["distinct_dest_accounts"] == 3
    assert p["orig_accounts_seen_more_than_once"] == 1  # C1
    assert p["dest_accounts_seen_more_than_once"] == 2  # C900, M100
    assert p["by_type"]["TRANSFER"] == {"rows": 3, "fraud_rows": 1}
    assert p["duplicate_rows"] == 0


def test_clean_fixture_passes_all_checks(fixture_profile):
    assert failed_checks(fixture_profile) == []


def test_checks_catch_bad_data(spark):
    good = {name: None for name in RENAMES.values()}
    rows = [
        {
            **good,
            "step": 1,
            "txn_type": "TRANSFER",
            "amount": 10.0,
            "name_orig": "C1",
            "name_dest": "C2",
            "is_fraud": 1,
            "is_flagged_fraud": 0,
        },
        {
            **good,
            "step": 2,
            "txn_type": "BITCOIN",
            "amount": -5.0,
            "name_orig": "C3",
            "name_dest": "C4",
            "is_fraud": 0,
            "is_flagged_fraud": 0,
        },
    ]
    for r in rows:  # balances are irrelevant to the checks; fill them in
        for col in ("old_balance_orig", "new_balance_orig", "old_balance_dest", "new_balance_dest"):
            r[col] = 0.0
    rows[1]["amount"] = None  # one unparseable amount ...
    rows.append({**rows[0], "amount": -5.0})  # ... and one negative amount
    df = spark.createDataFrame([Row(**r) for r in rows])

    failures = failed_checks(profile(df))
    assert any("'amount'" in f for f in failures)
    assert any("BITCOIN" in f for f in failures)
    assert any("negative" in f for f in failures)


@pytest.mark.parametrize(
    ("step", "expected"),
    [(1, "train"), (10, "train"), (11, "valid"), (20, "valid"), (21, "test"), (743, "test")],
)
def test_split_boundaries_are_inclusive(step, expected):
    assert split_name(step, train_end_step=10, valid_end_step=20) == expected


def test_split_rejects_reversed_boundaries():
    with pytest.raises(ValueError):
        split_name(5, train_end_step=20, valid_end_step=10)


def test_spark_split_matches_python_rule(spark, fixture_csv):
    df = assign_split(read_transactions(spark, fixture_csv), train_end_step=1, valid_end_step=25)
    for r in df.select("step", "split").collect():
        assert r["split"] == split_name(r["step"], 1, 25)
