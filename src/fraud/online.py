"""The feature definitions of sql/features.sql, computed one transaction at a time.

Training builds features in Spark SQL over the whole history; a live service
sees transactions one by one. If the two disagreed, the deployed model would be
fed different inputs from the ones it was trained and evaluated on. This module
is the streaming implementation, and tests/test_online_parity.py checks it
against the SQL on the same rows.

The contract that makes them equal: for one destination account, transactions
are handled one STEP at a time, in ascending step order. Every transaction in
step s is scored against the state of steps < s, and only then are the step-s
transactions added to the state. This is the SQL's
`RANGE BETWEEN ... AND 1 PRECEDING` rule, applied incrementally.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

WINDOW_STEPS = 24


@dataclass
class DestinationState:
    """What the features need to know about one destination account."""

    total_count: int = 0
    total_amount: float = 0.0
    # step -> (transactions, amount) for the most recent WINDOW_STEPS steps only
    recent: dict[int, tuple[int, float]] = field(default_factory=dict)

    def add_step(self, step: int, count: int, amount: float) -> None:
        """Record a COMPLETED step for this account."""
        self.total_count += count
        self.total_amount += amount
        prev_count, prev_amount = self.recent.get(step, (0, 0.0))
        self.recent[step] = (prev_count + count, prev_amount + amount)
        for old in [s for s in self.recent if s < step - WINDOW_STEPS]:
            del self.recent[old]

    def window(self, step: int) -> tuple[int, float]:
        """Transactions and amount in steps [step - 24, step - 1]."""
        count, amount = 0, 0.0
        for s, (c, a) in self.recent.items():
            if step - WINDOW_STEPS <= s <= step - 1:
                count += c
                amount += a
        return count, amount

    def to_dict(self) -> dict:
        return {
            "total_count": self.total_count,
            "total_amount": self.total_amount,
            "recent": {str(s): list(v) for s, v in self.recent.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> DestinationState:
        return cls(
            total_count=int(data["total_count"]),
            total_amount=float(data["total_amount"]),
            recent={int(s): (int(v[0]), float(v[1])) for s, v in data["recent"].items()},
        )


def transaction_features(
    step: int, txn_type: str, amount: float, state: DestinationState
) -> dict[str, float]:
    """Features for one transaction, given its destination's state BEFORE `step`."""
    hour = step % 24
    count_24h, amount_24h = state.window(step)
    if state.total_count and state.total_amount:
        vs_mean = amount / (state.total_amount / state.total_count)
    else:  # SQL: COALESCE(amount / NULLIF(AVG(...), 0), 0)
        vs_mean = 0.0
    return {
        "log_amount": math.log1p(amount),
        "hour_of_day": hour,
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "is_transfer": int(txn_type == "TRANSFER"),
        "dest_prior_txn_count": state.total_count,
        "dest_prior_amount": state.total_amount,
        "dest_txn_count_24h": count_24h,
        "dest_amount_24h": amount_24h,
        "amount_vs_dest_mean": vs_mean,
    }


@dataclass
class AccountFeed:
    """One account's transactions, featurised one at a time in arrival order.

    For use when an account's transactions arrive in hour order (the streaming
    job guarantees this: the producer keys by account, so one account's
    messages share a Kafka partition and a Flink channel, both first-in
    first-out). Each transaction is featurised against the history of earlier
    hours; the current hour's totals are held aside and committed to the
    history when the account's next hour begins. Equivalent to `process_step`
    on whole hours, with no buffering.
    """

    history: DestinationState = field(default_factory=DestinationState)
    step: int | None = None  # the hour currently being received
    count: int = 0
    amount: float = 0.0

    def observe(self, step: int, txn_type: str, amount: float) -> dict[str, float] | None:
        """Features for the next transaction, or None if it arrived out of order."""
        if self.step is not None and step < self.step:
            return None
        if self.step is not None and step > self.step:
            self.history.add_step(self.step, self.count, self.amount)
            self.count, self.amount = 0, 0.0
        self.step = step
        features = transaction_features(step, txn_type, amount, self.history)
        self.count += 1
        self.amount += amount
        return features

    def to_dict(self) -> dict:
        return {
            "history": self.history.to_dict(),
            "step": self.step,
            "count": self.count,
            "amount": self.amount,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AccountFeed:
        return cls(
            history=DestinationState.from_dict(data["history"]),
            step=data["step"],
            count=int(data["count"]),
            amount=float(data["amount"]),
        )


def process_step(state: DestinationState, step: int, transactions: Iterable[dict]) -> list[dict]:
    """Score every transaction of one step for one account, then commit the step.

    `transactions` are dicts with at least `step`, `txn_type` and `amount`.
    Returns the transactions with their features added.
    """
    batch = list(transactions)
    if any(t["step"] != step for t in batch):
        raise ValueError("all transactions in a batch must share the same step")
    out = [{**t, **transaction_features(step, t["txn_type"], t["amount"], state)} for t in batch]
    state.add_step(step, len(batch), sum(t["amount"] for t in batch))
    return out
