"""Small file helpers shared by the jobs."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path

# Overridable so a smoke test can run the full pipeline on synthetic data
# without touching the real dataset or the committed evidence.
RAW_DIR = Path(os.getenv("RAW_DIR", "/data/raw"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "/data/processed"))
EVIDENCE_DIR = Path(os.getenv("EVIDENCE_DIR", "/app/evidence"))


def find_raw_csv(raw_dir: Path = RAW_DIR) -> Path:
    """Return the single CSV in the raw folder, whatever Kaggle named it."""
    csvs = sorted(raw_dir.glob("*.csv"))
    if len(csvs) != 1:
        found = [p.name for p in csvs] or "none"
        raise FileNotFoundError(
            f"expected exactly one CSV in {raw_dir}, found: {found}. "
            "Download the PaySim dataset first (see README)."
        )
    return csvs[0]


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Fingerprint of the input file, recorded so results can be tied to the exact data."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def shared_dir(path: Path) -> Path:
    """Create a folder that every container can write to, and return it.

    The Spark image runs as `spark` and the Flink image as `flink`. A folder one
    of them creates is otherwise read-only to the other, e.g. an EVIDENCE_DIR
    made by the batch jobs and later written by the stream verification.
    """
    missing = [p for p in (path, *path.parents) if not p.exists()]
    path.mkdir(parents=True, exist_ok=True)
    for created in missing:
        # PermissionError: created concurrently by another container's user.
        with contextlib.suppress(PermissionError):
            created.chmod(0o777)
    return path


def write_json(data: dict, path: Path) -> None:
    shared_dir(path.parent)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
