# 1. Split by time and build features only from earlier hours

**Status:** accepted

## Context

A fraud model is always used on transactions that happen *after* it was
trained. Two common shortcuts break that:

- **A random train/test split.** Transactions from the last day end up in
  training, so the model is tested partly on the past it has already seen.
- **History features that can see the future.** A feature such as "how many
  transfers this account has received" is only honest if it counts transfers
  that had already happened when the transaction was made.

PaySim records time as `step`, one per hour. Inside one hour the file order is
not a reliable time order, so "earlier in the file" does not mean "earlier in
time".

## Decision

- **Split on `step`:** about 70% of in-scope rows for training, 15% for
  validation, 15% for test, cut at whole hours (`jobs/train.py`,
  `split_boundaries`). With this data that gives train = steps 1–323,
  validation = 324–377, test = 378–743.
- **Features look only at strictly earlier hours.** Every window in
  `sql/features.sql` ends at `1 PRECEDING` on the step axis, so a transaction
  never sees itself, other transactions in the same hour, or anything later.
- **The champion is chosen on validation only.** The test hours are used once,
  to report results.

## Consequences

- `tests/test_features.py` pins this down on a hand-built fixture: two
  transactions in the same hour do not see each other, and the 24-hour window
  drops older history.
- The rule is conservative: within an hour, a real system would know about
  earlier transactions in the same hour. Here that information is given up in
  exchange for a guarantee of no leakage.
- Because the split follows time, it inherits PaySim's change in volume
  after day 17 (see [ADR 3](0003-evaluate-at-alert-budgets-and-segment-the-test-period.md)).
