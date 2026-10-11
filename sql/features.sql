-- Point-in-time feature table.
--
-- Rule: a feature may only use transactions from STRICTLY EARLIER steps.
-- A step is one hour, and the log gives no reliable order for transactions
-- inside the same hour, so "earlier in the file" is not "earlier in time".
-- Every window therefore ends at `1 PRECEDING` on the step axis, which also
-- means a transaction never sees itself or anything that happened after it.
--
-- The balance columns are deliberately absent (see fraud/schema.py).

SELECT
    t.txn_id,
    t.step,
    t.txn_type,
    t.amount,
    t.is_fraud,
    t.is_flagged_fraud,

    -- Transaction itself
    LOG1P(t.amount)                                         AS log_amount,
    t.step % 24                                             AS hour_of_day,
    -- Hour as a point on a 24-hour circle, so 23:00 and 00:00 are neighbours.
    -- A linear model cannot use the raw hour otherwise.
    SIN(2 * PI() * (t.step % 24) / 24)                      AS hour_sin,
    COS(2 * PI() * (t.step % 24) / 24)                      AS hour_cos,
    CASE WHEN t.txn_type = 'TRANSFER' THEN 1 ELSE 0 END     AS is_transfer,

    -- Destination account history: has this account been receiving money?
    COUNT(1) OVER dest_all                                  AS dest_prior_txn_count,
    COALESCE(SUM(t.amount) OVER dest_all, 0)                AS dest_prior_amount,
    COUNT(1) OVER dest_24h                                  AS dest_txn_count_24h,
    COALESCE(SUM(t.amount) OVER dest_24h, 0)                AS dest_amount_24h,
    COALESCE(
        t.amount / NULLIF(AVG(t.amount) OVER dest_all, 0), 0
    )                                                       AS amount_vs_dest_mean

FROM transactions t
WINDOW
    dest_all AS (PARTITION BY t.name_dest ORDER BY t.step
                 RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
    dest_24h AS (PARTITION BY t.name_dest ORDER BY t.step
                 RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING)
