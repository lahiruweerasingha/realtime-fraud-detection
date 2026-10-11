"""Generate a small PaySim-shaped CSV for smoke-testing the pipeline.

    python scripts/make_synthetic_data.py OUTPUT_CSV [ROWS]

The fraud pattern is planted on purpose (large amounts sent to a small pool
of mule accounts), so a working pipeline must find it. Scores on this data
say nothing about real fraud detection; it exists only to prove the jobs
run end to end without the real dataset.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd


def generate(rows: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    types = rng.choice(
        ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
        size=rows,
        p=[0.22, 0.35, 0.01, 0.34, 0.08],
    )
    step = np.sort(rng.integers(1, 201, rows))
    amount = np.round(rng.lognormal(mean=9, sigma=1.2, size=rows), 2)

    in_scope = np.isin(types, ["TRANSFER", "CASH_OUT"])
    fraud = in_scope & (rng.random(rows) < 0.01)
    amount[fraud] = np.round(rng.lognormal(mean=12, sigma=0.8, size=fraud.sum()), 2)

    dest = np.where(
        types == "PAYMENT",
        [f"M{i}" for i in rng.integers(0, 2_000, rows)],
        [f"C{i}" for i in rng.integers(100_000, 105_000, rows)],
    )
    mules = np.array([f"C{i}" for i in rng.integers(900_000, 900_050, fraud.sum())])
    dest[fraud] = mules

    old_orig = np.round(amount + rng.random(rows) * 1e5, 2)
    return pd.DataFrame(
        {
            "step": step,
            "type": types,
            "amount": amount,
            "nameOrig": [f"C{i}" for i in range(rows)],
            "oldbalanceOrg": old_orig,
            "newbalanceOrig": np.where(fraud, 0.0, np.round(old_orig - amount, 2)),
            "nameDest": dest,
            "oldbalanceDest": 0.0,
            "newbalanceDest": np.where(fraud, 0.0, amount),
            "isFraud": fraud.astype(int),
            "isFlaggedFraud": (fraud & (types == "TRANSFER") & (amount > 200_000)).astype(int),
        }
    )


if __name__ == "__main__":
    out = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60_000
    generate(n).to_csv(out, index=False)
    print(f"wrote {n:,} synthetic rows to {out}")
