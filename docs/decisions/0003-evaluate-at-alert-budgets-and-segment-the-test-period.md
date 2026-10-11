# 3. Evaluate at alert budgets, and report the test period in two segments

**Status:** accepted

## Context

**Accuracy says nothing here.** In the scored transaction types, 0.296% of rows are
fraud. A model that never raises an alert would be 99.7% accurate.

**A fraud team has fixed capacity.** The practical question is not "what is
the F1 score?" but "if analysts can review N alerts, how much fraud, and how
much money, do we catch?"

**The test period changes character.** The EDA notebook shows that after day 17
(step 408), legitimate in-scope volume collapses from about 180,000 rows a
day to under 25,000, while fraud continues at about 250 a day. The test
period (steps 378–743) therefore mixes:

| Segment | Steps | Character |
|---|---|---|
| normal volume | 378–408 | similar to training and validation |
| low-volume tail | 409–743 | fraud rate several times higher; two days contain only fraud |

Precision depends directly on the fraud rate, so a single test number would
be inflated by the tail and would not describe normal operation.

## Decision

**Metrics** (`src/fraud/evaluate.py`, all unit-tested by hand):
- average precision (area under the precision-recall curve), with a 95%
  bootstrap confidence interval;
- precision, recall and **value recall** (share of fraud *amount* caught) when
  only the top 0.1%, 0.5% or 1% of transactions are alerted;
- the model at **exactly the existing rule's alert count**, for a like-for-like
  comparison with `isFlaggedFraud`;
- the alert count that minimises total cost
  (`alerts × review cost + missed fraud value × loss fraction`) for several
  review costs. These costs are labelled assumptions: PaySim has no
  currency, so the cost results are illustrative, not a business case;
- a **paired** bootstrap for XGBoost vs logistic regression on the same rows,
  so the comparison is not confused by resampling noise.

**Segments:** every test metric is reported for the normal-volume hours,
the low-volume tail, and the whole test period. The README leads with the
normal-volume figures.

## Consequences

- Results are comparable to how a fraud team works (alert volume), not to a
  leaderboard metric.
- The segment boundary (step 408) is a decision based on the EDA, kept as a
  named, overridable constant (`NORMAL_VOLUME_LAST_STEP`) rather than hidden.
- The normal-volume segment is smaller than the whole test period, so its
  confidence intervals are wider. They are reported, not hidden.
