from __future__ import annotations

import json
import math

import numpy as np
import pytest

from fraud.evaluate import score_reference
from fraud.stream_records import (
    BASE_EPOCH_MS,
    dead_letter,
    event_time_ms,
    hourly_summary,
    step_of,
    validate,
)

GOOD_TXN = {
    "kind": "txn",
    "txn_id": 7,
    "step": 378,
    "txn_type": "TRANSFER",
    "amount": 1500.5,
    "name_dest": "C123",
    "is_fraud": 1,
}


def test_event_time_round_trips():
    assert event_time_ms(0) == BASE_EPOCH_MS
    for step in (1, 378, 743):
        assert step_of(event_time_ms(step)) == step
    assert event_time_ms(1) - event_time_ms(0) == 3_600_000


def test_valid_transaction_is_normalised():
    record, reason = validate(json.dumps(GOOD_TXN).encode())
    assert reason is None
    assert record == {**GOOD_TXN, "amount": 1500.5}


def test_valid_state_message():
    msg = {"kind": "state", "name_dest": "C1", "step": 353, "total_count": 4, "total_amount": 10}
    record, reason = validate(json.dumps(msg))
    assert reason is None
    assert record["total_amount"] == 10.0


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("not json at all", "not_json"),
        (b"\xff\xfe", "not_json"),
        ("[1, 2]", "not_an_object"),
        (json.dumps({**GOOD_TXN, "name_dest": ""}), "missing_name_dest"),
        (json.dumps({**GOOD_TXN, "step": "378"}), "bad_step"),
        (json.dumps({**GOOD_TXN, "step": -1}), "bad_step"),
        (json.dumps({**GOOD_TXN, "step": True}), "bad_step"),
        (json.dumps({**GOOD_TXN, "kind": "refund"}), "unknown_kind"),
        (json.dumps({**GOOD_TXN, "txn_type": "BITCOIN"}), "unknown_txn_type"),
        (json.dumps({**GOOD_TXN, "amount": -5}), "bad_amount"),
        (json.dumps({**GOOD_TXN, "amount": "12"}), "bad_amount"),
        (json.dumps({k: v for k, v in GOOD_TXN.items() if k != "amount"}), "bad_amount"),
        (json.dumps({**GOOD_TXN, "txn_id": "7"}), "bad_txn_id"),
        (
            json.dumps(
                {
                    "kind": "state",
                    "name_dest": "C1",
                    "step": 1,
                    "total_count": -1,
                    "total_amount": 1,
                }
            ),
            "bad_state_count",
        ),
    ],
)
def test_invalid_messages_are_rejected_with_a_reason(raw, reason):
    record, got = validate(raw)
    assert record is None
    assert got == reason


def test_nan_amount_is_rejected():
    # json.dumps writes NaN, which json.loads accepts; it must still be rejected.
    assert validate(json.dumps({**GOOD_TXN, "amount": math.nan}))[1] == "bad_amount"


def test_dead_letter_keeps_reason_and_truncated_payload():
    letter = json.loads(dead_letter("x" * 2000, "not_json"))
    assert letter["reason"] == "not_json"
    assert len(letter["raw"]) == 500
    assert letter["processed_at_ms"] > 0


def test_hourly_summary_by_hand():
    rng = np.random.default_rng(0)
    reference = score_reference(rng.random(10_000))
    scores = np.array([0.9, 0.8, 0.1, 0.05])
    labels = np.array([1, 0, 1, 0])
    s = hourly_summary(scores, labels, threshold=0.5, reference=reference)
    assert s["scored"] == 4
    assert s["alerts"] == 2
    assert s["alert_rate"] == pytest.approx(0.5)
    assert s["fraud"] == 2
    assert s["fraud_alerted"] == 1
    assert s["alert_precision"] == pytest.approx(0.5)
    assert s["recall"] == pytest.approx(0.5)
    assert s["mean_score"] == pytest.approx(0.4625)
    assert s["psi"] > 0


def test_hourly_summary_leaves_undefined_rates_empty():
    reference = score_reference(np.random.default_rng(0).random(1_000))
    s = hourly_summary(np.array([0.1, 0.2]), np.array([0, 0]), 0.5, reference)
    assert s["alerts"] == 0 and s["fraud"] == 0
    assert s["alert_precision"] is None  # no alerts: precision undefined, not 0
    assert s["recall"] is None  # no fraud: recall undefined, not 0


def test_export_module_imports_without_pyspark():
    # The Flink image unpickles LogisticScorer from fraud.export and has no Spark.
    # A fresh interpreter, so PySpark imported by other tests can't hide the bug.
    import os
    import subprocess
    import sys
    from pathlib import Path

    src = str(Path(__file__).resolve().parents[1] / "src")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([src, os.environ.get("PYTHONPATH", "")])}
    code = (
        "import sys; sys.modules['pyspark'] = None; "
        "import fraud.export; print(fraud.export.LogisticScorer.__name__)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "LogisticScorer"


def test_hourly_summary_rejects_empty_window():
    with pytest.raises(ValueError):
        hourly_summary(np.array([]), np.array([]), 0.5, {"edges": [0, 1], "proportions": [1.0]})


def test_shared_dir_is_writable_by_every_container_user(tmp_path):
    import stat

    from fraud.io import shared_dir

    target = shared_dir(tmp_path / "output" / "evidence")
    for folder in (target, target.parent):
        assert stat.S_IMODE(folder.stat().st_mode) == 0o777
