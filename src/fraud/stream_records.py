"""Message formats and checks for the streaming pipeline (no Flink imports).

Kept separate from the Flink job so every rule here is unit-tested directly.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np

from fraud.evaluate import population_stability_index
from fraud.schema import TRANSACTION_TYPES

# PaySim has no calendar, only hour numbers. For event time, step 0 is placed at
# 2024-01-01 00:00 UTC, so step s happens at that time plus s hours.
BASE_EPOCH_MS = 1_704_067_200_000
STEP_MS = 3_600_000
MAX_STEP = 10_000
RAW_PREVIEW_CHARS = 500


def event_time_ms(step: int) -> int:
    return BASE_EPOCH_MS + step * STEP_MS


def step_of(event_ms: int) -> int:
    return (event_ms - BASE_EPOCH_MS) // STEP_MS


def _is_number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def validate(raw: str | bytes) -> tuple[dict | None, str | None]:
    """Return (record, None) for a usable message, or (None, reason) for a dead letter.

    Two message kinds arrive on the input topic:
      txn    a transaction to score (or to learn from, before scoring starts)
      state  a destination account's history before the replay starts
    """
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes | bytearray) else raw
        msg = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, "not_json"
    if not isinstance(msg, dict):
        return None, "not_an_object"

    kind = msg.get("kind")
    name_dest = msg.get("name_dest")
    if not isinstance(name_dest, str) or not name_dest:
        return None, "missing_name_dest"
    step = msg.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or not 0 <= step <= MAX_STEP:
        return None, "bad_step"

    if kind == "state":
        count, amount = msg.get("total_count"), msg.get("total_amount")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            return None, "bad_state_count"
        if not _is_number(amount) or amount < 0:
            return None, "bad_state_amount"
        return {
            "kind": "state",
            "name_dest": name_dest,
            "step": step,
            "total_count": count,
            "total_amount": float(amount),
        }, None

    if kind != "txn":
        return None, "unknown_kind"
    if msg.get("txn_type") not in TRANSACTION_TYPES:
        return None, "unknown_txn_type"
    amount = msg.get("amount")
    if not _is_number(amount) or amount < 0:
        return None, "bad_amount"
    txn_id = msg.get("txn_id")
    if not isinstance(txn_id, int) or isinstance(txn_id, bool):
        return None, "bad_txn_id"
    return {
        "kind": "txn",
        "txn_id": txn_id,
        "step": step,
        "txn_type": msg["txn_type"],
        "amount": float(amount),
        "name_dest": name_dest,
        # Labels ride along in the replay so monitoring can show alert precision
        # once outcomes are known; they are never model inputs.
        "is_fraud": int(msg.get("is_fraud", 0) or 0),
    }, None


def dead_letter(raw: str | bytes, reason: str) -> str:
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes | bytearray) else raw
    return json.dumps(
        {
            "reason": reason,
            "raw": text[:RAW_PREVIEW_CHARS],
            "processed_at_ms": int(time.time() * 1000),
        }
    )


def hourly_summary(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    reference: dict,
) -> dict[str, float]:
    """Monitoring numbers for one event-time hour of scored transactions."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(scores) == 0:
        raise ValueError("no scores in this window")
    alerts = scores >= threshold
    fraud = labels == 1
    return {
        "scored": int(len(scores)),
        "alerts": int(alerts.sum()),
        "alert_rate": float(alerts.mean()),
        "fraud": int(fraud.sum()),
        "fraud_alerted": int((alerts & fraud).sum()),
        "alert_precision": float((alerts & fraud).sum() / alerts.sum()) if alerts.any() else 0.0,
        "recall": float((alerts & fraud).sum() / fraud.sum()) if fraud.any() else 0.0,
        "mean_score": float(scores.mean()),
        "psi": population_stability_index(reference, scores),
    }
