"""The streaming feature logic must equal the Spark SQL, row for row.

Both implementations run on the same synthetic transactions (with repeated
destination accounts, same-step collisions and gaps longer than 24 steps), and
every feature of every transaction is compared.
"""

from __future__ import annotations

import importlib.util
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pytest

from fraud.features import FEATURE_COLUMNS, build_features
from fraud.online import AccountFeed, DestinationState, process_step, transaction_features
from fraud.schema import RENAMES

_GEN = Path(__file__).resolve().parents[1] / "scripts" / "make_synthetic_data.py"
_spec = importlib.util.spec_from_file_location("make_synthetic_data", _GEN)
make_synthetic_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_synthetic_data)


def online_features(txns: pd.DataFrame) -> pd.DataFrame:
    """Replay the transactions per destination, one step at a time."""
    states: dict[str, DestinationState] = defaultdict(DestinationState)
    rows = []
    ordered = txns.sort_values(["step", "txn_id"])
    for step, step_rows in ordered.groupby("step", sort=True):
        for dest, group in step_rows.groupby("name_dest", sort=False):
            rows += process_step(states[dest], int(step), group.to_dict("records"))
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic(spark):
    raw = make_synthetic_data.generate(4_000, seed=11).rename(columns=RENAMES)
    # Squeeze 200 steps of activity into fewer destinations, so accounts repeat
    # often, share steps, and also go quiet for longer than 24 steps.
    raw["name_dest"] = raw["name_dest"].str[:4]
    raw["txn_id"] = range(len(raw))
    return raw


def test_online_features_equal_spark_sql(spark, synthetic):
    sdf = spark.createDataFrame(synthetic)
    batch = build_features(spark, sdf).toPandas().set_index("txn_id").sort_index()
    online = online_features(synthetic).set_index("txn_id").sort_index()

    assert len(batch) == len(online) == len(synthetic)
    for column in FEATURE_COLUMNS:
        pd.testing.assert_series_equal(
            online[column].astype(float),
            batch[column].astype(float),
            check_names=False,
            rtol=1e-9,
            atol=1e-9,
            obj=column,
        )


def test_the_fixture_exercises_the_hard_cases(synthetic):
    per_dest_step = synthetic.groupby(["name_dest", "step"]).size()
    assert (per_dest_step > 1).any(), "no same-step collisions"
    gaps = synthetic.sort_values("step").groupby("name_dest")["step"].diff()
    assert (gaps > 24).any(), "no gap longer than the 24-step window"


def test_state_survives_serialisation():
    state = DestinationState()
    state.add_step(5, 2, 300.0)
    state.add_step(9, 1, 50.0)
    restored = DestinationState.from_dict(state.to_dict())
    assert restored == state
    assert transaction_features(10, "TRANSFER", 10.0, restored) == transaction_features(
        10, "TRANSFER", 10.0, state
    )


def test_window_drops_steps_older_than_24():
    state = DestinationState()
    state.add_step(1, 1, 100.0)
    state.add_step(30, 1, 7.0)
    assert state.window(31) == (1, 7.0)  # step 1 is outside [7, 30]
    assert state.total_count == 2  # but still in the all-time history
    assert 1 not in state.recent  # and pruned from memory


def test_batch_must_share_a_step():
    with pytest.raises(ValueError):
        process_step(
            DestinationState(),
            3,
            [
                {"step": 3, "txn_type": "TRANSFER", "amount": 1.0},
                {"step": 4, "txn_type": "TRANSFER", "amount": 1.0},
            ],
        )


def test_account_feed_equals_whole_step_processing(synthetic):
    """The streaming job featurises records one by one as they arrive; that must
    give the same features as processing each account's hours whole."""
    by_step = online_features(synthetic).set_index("txn_id").sort_index()

    feeds: dict[str, AccountFeed] = defaultdict(AccountFeed)
    rows = []
    for t in synthetic.sort_values(["step", "txn_id"]).to_dict("records"):
        if t["txn_id"] % 7 == 0:  # a checkpoint/restore mid-stream changes nothing
            feeds[t["name_dest"]] = AccountFeed.from_dict(feeds[t["name_dest"]].to_dict())
        features = feeds[t["name_dest"]].observe(int(t["step"]), t["txn_type"], float(t["amount"]))
        rows.append({"txn_id": t["txn_id"], **features})
    one_by_one = pd.DataFrame(rows).set_index("txn_id").sort_index()

    for column in FEATURE_COLUMNS:
        pd.testing.assert_series_equal(
            one_by_one[column].astype(float), by_step[column].astype(float), obj=column
        )


def test_account_feed_rejects_an_earlier_hour():
    feed = AccountFeed()
    assert feed.observe(10, "TRANSFER", 5.0) is not None
    assert feed.observe(9, "TRANSFER", 5.0) is None
    assert feed.count == 1  # the late record changed nothing


def test_amount_vs_mean_is_zero_without_history():
    f = transaction_features(1, "CASH_OUT", 500.0, DestinationState())
    assert f["amount_vs_dest_mean"] == 0.0
    assert f["is_transfer"] == 0
    assert f["log_amount"] == pytest.approx(math.log1p(500.0))
