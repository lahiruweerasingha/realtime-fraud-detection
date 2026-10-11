# 4. Register only models that score outside Spark, and prove they match

**Status:** accepted

## Context

Spark is the right tool to train on millions of rows, but a fraud score is
needed one transaction at a time, by a service that should not have to start a
Spark cluster. So the registered model has to work without Spark, and it has
to produce the **same** scores that were evaluated. If it does not, the
deployed model is not the model in the report, and no metric will show it.

## Decision

- **XGBoost** is trained with `xgboost.spark` across the Spark workers. Its
  fitted booster is ordinary XGBoost, so it can be saved and loaded without Spark.
  (A Spark MLlib gradient-boosted-trees model can only score inside Spark.)
- **Logistic regression** is exported as a small MLflow Python model that
  repeats the pipeline's arithmetic: standardise, multiply by the coefficients,
  apply the sigmoid (`src/fraud/export.py`).
- **The champion is chosen on validation**, and whichever model wins is
  registered as `fraud-detector@champion`.
- **A parity check guards every training run.** Before anything is logged,
  each exported model re-scores the test hours, and the job fails if any score
  differs from Spark's by more than 1e-4.

## What the check caught

The first version exported `booster` as returned by training. With early
stopping, XGBoost keeps training 30 rounds past its best round. Spark's
`transform` scores with the best round, but a raw booster's `predict` uses
every tree. On the smoke-test data the two disagreed by up to **0.19** in
predicted probability. The unit test had not caught it, because its settings
never triggered early stopping.

Fix: the exported booster is sliced to `best_iteration + 1`. The unit test now
forces early stopping, and asserts both that the untrimmed booster *disagrees*
(so the test really exercises the case) and that the trimmed one matches.

## Consequences

- One registry entry, one alias, loadable with
  `mlflow.pyfunc.load_model("models:/fraud-detector@champion")` (or with
  `mlflow.xgboost` for the native booster), with no Spark dependency.
- The parity difference is logged as a metric on every run, so a regression
  would be visible in MLflow, not only in a failed job.
