"""Check the streaming run against the batch pipeline, record by record.

    make stream-verify

Reads everything the Flink job wrote to the `scores` and dead-letter topics and
checks that:
  - every in-scope transaction in the scored hours was scored exactly once;
  - each streamed feature equals the batch feature for the same transaction;
  - each streamed score equals the champion's score on the batch features;
  - every injected malformed message reached the dead-letter topic with the
    right reason, and nothing else did.
Writes evidence/streaming_verification.json and exits non-zero on any failure.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from confluent_kafka import Consumer

from fraud.features import FEATURE_COLUMNS
from fraud.io import EVIDENCE_DIR, PROCESSED_DIR, write_json
from fraud.schema import MODEL_SCOPE_TYPES
from fraud.serving import load_champion

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
STREAM_DIR = Path(os.getenv("STREAM_DIR", "/data/processed/stream"))
FLINK_REST = os.getenv("FLINK_REST", "http://flink-jobmanager:8081")
TOLERANCE = 1e-6


def last_flink_run() -> dict:
    """Duration and record counts of the most recent finished scoring job."""
    import urllib.request

    def get(path: str) -> dict:
        with urllib.request.urlopen(f"{FLINK_REST}{path}", timeout=10) as response:
            return json.loads(response.read())

    try:
        jobs = [
            j
            for j in get("/jobs/overview")["jobs"]
            if j["name"] == "fraud-scoring" and j["state"] == "FINISHED"
        ]
    except OSError:
        return {}
    if not jobs:
        return {}
    job = max(jobs, key=lambda j: j["end-time"])
    detail = get(f"/jobs/{job['jid']}")
    # The source vertex has validation chained into it, so what it passes on
    # is the valid records (malformed ones leave through the dead-letter sink).
    valid = detail["vertices"][0]["metrics"].get("write-records", 0)
    seconds = job["duration"] / 1000
    return {
        "job_id": job["jid"],
        "duration_seconds": round(seconds, 1),
        "valid_records_processed": valid,
        "records_per_second": round(valid / seconds) if seconds else None,
        "parallelism": max(v["parallelism"] for v in detail["vertices"]),
    }


def read_topic(topic: str) -> list[dict]:
    consumer = Consumer(
        {
            "bootstrap.servers": BOOTSTRAP,
            "group.id": f"verify-{topic}-{os.getpid()}",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([topic])
    out, idle = [], 0
    while idle < 10:  # stop after ~10 s with nothing new
        batch = consumer.consume(10_000, timeout=1.0)
        if not batch:
            idle += 1
            continue
        idle = 0
        out += [json.loads(m.value()) for m in batch if m.error() is None]
    consumer.close()
    return out


def main() -> int:
    summary = json.loads((STREAM_DIR / "summary.json").read_text())
    manifest = json.loads((STREAM_DIR / "producer_manifest.json").read_text())
    lo, hi = summary["score_from_step"], summary["score_to_step"]

    streamed = pd.DataFrame(read_topic("scores"))
    dead = pd.DataFrame(read_topic("transactions-dlq"))

    batch = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    expected = batch[batch["step"].between(lo, hi) & batch["txn_type"].isin(MODEL_SCOPE_TYPES)]
    champion = load_champion()
    expected = expected.assign(batch_score=champion.predict(expected))

    checks: dict = {
        "scored_hours": [lo, hi],
        "expected_transactions": len(expected),
        "streamed_records": len(streamed),
        "duplicate_txn_ids": int(streamed["txn_id"].duplicated().sum()) if len(streamed) else 0,
    }
    joined = expected.merge(
        streamed, on="txn_id", how="outer", suffixes=("_batch", ""), indicator=True
    )
    checks["missing_from_stream"] = int((joined["_merge"] == "left_only").sum())
    checks["unexpected_in_stream"] = int((joined["_merge"] == "right_only").sum())

    both = joined[joined["_merge"] == "both"]
    feature_diffs = {
        c: float(np.max(np.abs(both[c].astype(float) - both[f"{c}_batch"].astype(float))))
        for c in FEATURE_COLUMNS
    }
    checks["max_abs_feature_difference"] = feature_diffs
    checks["max_abs_score_difference"] = float(np.max(np.abs(both["score"] - both["batch_score"])))
    checks["alerts"] = int(both["alert"].sum())
    checks["alert_threshold"] = champion.alert_threshold
    checks["fraud_in_scored_hours"] = int(both["is_fraud_batch"].sum())
    checks["fraud_alerted"] = int((both["alert"] & (both["is_fraud_batch"] == 1)).sum())

    got_reasons = dead["reason"].value_counts().to_dict() if len(dead) else {}
    checks["dead_letters"] = {
        "injected": manifest["malformed_by_reason"],
        "received": {k: int(v) for k, v in sorted(got_reasons.items())},
    }
    checks["model_version"] = champion.version

    failures = []
    if checks["duplicate_txn_ids"]:
        failures.append("transactions scored more than once")
    if checks["missing_from_stream"] or checks["unexpected_in_stream"]:
        failures.append("streamed transactions differ from the expected set")
    if max(feature_diffs.values(), default=0) > TOLERANCE:
        failures.append("a streamed feature differs from the batch feature")
    if checks["max_abs_score_difference"] > TOLERANCE:
        failures.append("a streamed score differs from the batch score")
    if checks["dead_letters"]["received"] != checks["dead_letters"]["injected"]:
        failures.append("dead letters do not match what was injected")
    checks["failures"] = failures
    checks["passed"] = not failures

    write_json(checks, EVIDENCE_DIR / "streaming_verification.json")
    write_json(
        {**summary, **manifest, "flink": last_flink_run()}, EVIDENCE_DIR / "streaming_run.json"
    )
    print(json.dumps(checks, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
