# 5. Streaming features must equal the batch features, and that is verified

**Status:** accepted

## Context

The champion was trained on features computed in Spark SQL over the full
history. A live service sees one transaction at a time. Three things can make
the two disagree, and none of them raises an error:

1. **Missing history.** `dest_prior_txn_count` counts every earlier transaction
   of the destination account, back to the first hour. A stream that starts
   empty at the first scored hour would undercount it.
2. **Arrival order and same-hour visibility.** Transactions in the same hour
   must not see each other, because the SQL only looks at strictly earlier
   hours, and the stream must not see an account's hours out of order.
3. **Bad messages.** A malformed record that reached the timestamp assigner
   could move the event-time watermark and make later, valid records "late".

## Decision

- **One definition in two implementations.** `fraud.online` reimplements
  `sql/features.sql` for one account at a time, and
  `tests/test_online_parity.py` runs both on the same synthetic transactions
  (with same-hour collisions and gaps longer than the 24-hour window) and
  compares every feature of every row.
- **State, then warm-up, then scoring.** `jobs/export_stream_state.py` writes
  each active destination's all-time count and amount before the replay
  window, sent as `state` messages ahead of the transactions. The replay
  starts 24 hours before the first scored hour, which is exactly what the
  24-hour window features need.
- **Key by account end to end, score on arrival.** The producer keys Kafka
  messages by destination account, so one account's messages share a
  partition, a source task and a network channel to the scoring task, all
  first-in first-out: they arrive in hour order. `fraud.online.AccountFeed`
  scores each transaction against the account's *earlier* hours and holds the
  current hour's totals aside until the account's next hour starts.
  `tests/test_online_parity.py` checks this one-by-one path against the
  whole-hour path, which is itself checked against the Spark SQL.
- **Validate before event time.** Messages are checked before any timestamp is
  assigned. Failures go to a dead-letter topic with a reason, increment a Flink
  counter and are counted in InfluxDB, so they can never touch the watermark.
- **Verify end to end.** `streaming/verify.py` reads everything the job wrote
  and compares it with the batch pipeline, transaction by transaction.

## Result

From `evidence/streaming_verification.json` (model v3):

| Check | Result |
|---|---|
| Messages replayed | 959,016 transactions (all types, hours 354–408) + 284,974 account-history messages + 40 malformed |
| Transactions scored (TRANSFER and CASH_OUT, hours 378–408) | 253,490 of 253,490 expected; 0 missing, 0 unexpected, 0 duplicates |
| Largest difference, streamed vs batch feature | 3.0e-08 |
| Largest difference, streamed vs batch score | 0.0e+00 |
| Alerts raised (threshold 0.9478, model v3) | 997; 205 of 386 fraud cases caught |
| Malformed messages caught | 40 of 40, every reason matching what was injected |
| Run time (bounded replay, parallelism 3) | 26 min for 1,243,990 valid messages, about 806 a second |
| Verification | **passed** ([`evidence/streaming_verification.json`](../../evidence/streaming_verification.json)) |

## Alternative tried first

The first version buffered each account's transactions in Flink list state
and registered an event-time timer per hour, processing the hour when the
watermark passed it. That is correct whatever the arrival order, but it cost
several JVM-Python state round trips and a timer per record. On this laptop
it ran at about 200 records a second, with checkpoints taking 30-40 s, and
would have needed well over an hour for the 1.24 million-message replay. The
ordering guarantee it protected against was already provided by keying, so
it was replaced.

## Consequences

- Correctness now depends on per-account ordering. A record from an hour the
  account has already moved past is counted (`late_transactions`) and not
  scored, rather than mixed into the wrong hour, but the records that overtook
  it were scored without it. Keying by account rules this out here; a live
  system fed by several producers would need the same keying contract, plus a
  side output and a reconciliation process for late records. In the run above
  every expected transaction was scored and every feature matched the batch
  value, which could not happen if any record had arrived out of order.
- Per-account state is one small record (totals plus at most 24 hourly
  buckets). It grows with the number of accounts, so a long-running
  deployment would need a state TTL.
- The event-time watermark is still used, but only for the hourly monitoring
  windows, not for scoring. Scores are emitted immediately rather than when
  an hour closes.
