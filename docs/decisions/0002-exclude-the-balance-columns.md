# 2. Exclude the balance columns

**Status:** accepted

## Context

PaySim has four balance columns: `oldbalanceOrg`, `newbalanceOrig`,
`oldbalanceDest`, `newbalanceDest`. The dataset description says that
fraudulent transactions are cancelled. Many public notebooks feed these
columns to a model anyway and report near-perfect scores.

The EDA notebook measured how the balances behave in the transaction types
that can be fraud (TRANSFER and CASH_OUT):

| | Fraud | Legitimate |
|---|---:|---:|
| Destination balance never moves (old = new = 0) | 49.6% | 0.06% |
| Origin balance arithmetic adds up (old - amount = new) | 98.5% | 6.0% |

Neither pattern describes what a fraudster does. Both describe how the
simulator books a transaction once it already knows whether it is fraud. A
model that reads them learns the bookkeeping, and nothing it learns would carry
over to a real payment system.

## Decision

- The four balance columns are listed in `fraud.schema.LEAKY_COLUMNS`, never
  appear in `sql/features.sql`, and `fraud.features` asserts at import time
  that no feature is a leaky column.
- The training job **retrains the same XGBoost with the balance columns
  added**, scores the same test hours, and logs the result as a check run in
  MLflow. That model is never registered.

## Result

Same XGBoost, same split, scored on the same test hours (`evidence/metrics.json`, `leakage_check`):

| Test segment | Without balance columns (registered) | With balance columns (check) |
|---|---:|---:|
| normal-volume hours | 0.441 | 0.919 |
| low-volume tail | 0.590 | 0.986 |
| whole test period | 0.549 | 0.976 |

With the balance columns, average precision on the normal-volume hours rises from 0.441 to 0.919: most of the gap between this project's result and a "near-perfect" one is the leak.

## Consequences

- The reported scores are lower than many public PaySim results, deliberately.
  The gap is quantified rather than asserted.
- The check runs on every training run, so if a leaky column ever slipped
  back into the feature set, the gap between the two numbers would collapse
  and show it.
