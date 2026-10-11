# Results

Generated from `evidence/metrics.json` (run time 1279.2 s).
Input file SHA-256: `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b`

## Data

- Rows: 6,362,620; fraud: 8,213 (0.129%)
- Split by step: train <= 323, validation <= 377, test after

| Split | Rows | Fraud | Fraud rate |
|---|---:|---:|---:|
| train | 1,951,895 | 3,643 | 0.187% |
| valid | 410,437 | 560 | 0.136% |
| test | 408,077 | 4,010 | 0.983% |

Champion (chosen on validation): **XGBoost**

| Model | Validation AP |
|---|---:|
| Logistic regression | 0.097 |
| XGBoost | 0.389 |

## Normal-volume test hours (steps 378-408)

253,490 transactions, 386 fraud (0.152%).

| Model | Average precision (95% CI) | Budget | Alerts | Precision | Recall | Value recall |
|---|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.090 (0.065-0.121) | top 0.1% | 254 | 18.1% | 11.9% | 30.4% |
|  |  | top 0.5% | 1,268 | 7.3% | 23.8% | 44.9% |
|  |  | top 1.0% | 2,535 | 4.6% | 30.3% | 52.2% |
| XGBoost | 0.441 (0.395-0.493) | top 0.1% | 254 | 61.8% | 40.7% | 79.8% |
|  |  | top 0.5% | 1,268 | 17.2% | 56.5% | 92.8% |
|  |  | top 1.0% | 2,535 | 10.2% | 66.8% | 94.9% |
| *XGBoost + balance columns (check, not deployable)* | 0.919 | top 0.1% | 254 | 100.0% | 65.8% | 97.1% |
|  |  | top 0.5% | 1,268 | 29.1% | 95.6% | 99.9% |
|  |  | top 1.0% | 2,535 | 15.0% | 98.7% | 100.0% |
| *XGBoost without time of day (check, not deployable)* | 0.166 | top 0.1% | 254 | 30.3% | 19.9% | 61.6% |
|  |  | top 0.5% | 1,268 | 10.0% | 32.9% | 73.1% |
|  |  | top 1.0% | 2,535 | 6.8% | 44.8% | 76.6% |

XGBoost minus logistic regression, average precision: +0.351 (95% CI +0.307 to +0.395; share of resamples where XGBoost was not better: 0.000).

**Existing rule vs the model at the same alert count**

| | Alerts | Precision | Recall | Value recall |
|---|---:|---:|---:|---:|
| `isFlaggedFraud` rule | 1 | 100.0% | 0.3% | 0.7% |
| XGBoost | 1 | 100.0% | 0.3% | 0.4% |

## Low-volume tail (steps after 408)

154,587 transactions, 3,624 fraud (2.344%).

| Model | Average precision (95% CI) | Budget | Alerts | Precision | Recall | Value recall |
|---|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.330 (0.315-0.346) | top 0.1% | 155 | 96.8% | 4.1% | 9.6% |
|  |  | top 0.5% | 773 | 63.9% | 13.6% | 21.3% |
|  |  | top 1.0% | 1,546 | 55.0% | 23.5% | 32.4% |
| XGBoost | 0.590 (0.576-0.605) | top 0.1% | 155 | 100.0% | 4.3% | 12.9% |
|  |  | top 0.5% | 773 | 95.6% | 20.4% | 42.2% |
|  |  | top 1.0% | 1,546 | 78.3% | 33.4% | 61.9% |
| *XGBoost + balance columns (check, not deployable)* | 0.986 | top 0.1% | 155 | 100.0% | 4.3% | 14.4% |
|  |  | top 0.5% | 773 | 100.0% | 21.3% | 52.2% |
|  |  | top 1.0% | 1,546 | 100.0% | 42.7% | 83.5% |
| *XGBoost without time of day (check, not deployable)* | 0.414 | top 0.1% | 155 | 98.7% | 4.2% | 12.6% |
|  |  | top 0.5% | 773 | 78.4% | 16.7% | 57.0% |
|  |  | top 1.0% | 1,546 | 59.4% | 25.3% | 65.3% |

XGBoost minus logistic regression, average precision: +0.260 (95% CI +0.247 to +0.275; share of resamples where XGBoost was not better: 0.000).

**Existing rule vs the model at the same alert count**

| | Alerts | Precision | Recall | Value recall |
|---|---:|---:|---:|---:|
| `isFlaggedFraud` rule | 12 | 100.0% | 0.3% | 1.2% |
| XGBoost | 12 | 100.0% | 0.3% | 1.1% |

## Whole test period (steps from 378)

408,077 transactions, 4,010 fraud (0.983%).

| Model | Average precision (95% CI) | Budget | Alerts | Precision | Recall | Value recall |
|---|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.219 (0.205-0.232) | top 0.1% | 409 | 68.5% | 7.0% | 15.6% |
|  |  | top 0.5% | 2,041 | 39.3% | 20.0% | 28.8% |
|  |  | top 1.0% | 4,081 | 28.1% | 28.6% | 41.6% |
| XGBoost | 0.549 (0.534-0.563) | top 0.1% | 409 | 99.0% | 10.1% | 23.2% |
|  |  | top 0.5% | 2,041 | 74.2% | 37.8% | 68.7% |
|  |  | top 1.0% | 4,081 | 50.9% | 51.8% | 85.2% |
| *XGBoost + balance columns (check, not deployable)* | 0.976 | top 0.1% | 409 | 100.0% | 10.2% | 29.3% |
|  |  | top 0.5% | 2,041 | 100.0% | 50.9% | 91.6% |
|  |  | top 1.0% | 4,081 | 90.2% | 91.8% | 99.5% |
| *XGBoost without time of day (check, not deployable)* | 0.306 | top 0.1% | 409 | 95.4% | 9.7% | 28.1% |
|  |  | top 0.5% | 2,041 | 46.6% | 23.7% | 64.5% |
|  |  | top 1.0% | 4,081 | 32.4% | 32.9% | 66.7% |

XGBoost minus logistic regression, average precision: +0.330 (95% CI +0.318 to +0.342; share of resamples where XGBoost was not better: 0.000).

**Existing rule vs the model at the same alert count**

| | Alerts | Precision | Recall | Value recall |
|---|---:|---:|---:|---:|
| `isFlaggedFraud` rule | 13 | 100.0% | 0.3% | 1.1% |
| XGBoost | 13 | 100.0% | 0.3% | 1.4% |

## Cost analysis: computed, deliberately not used

`metrics.json` holds the cost-minimising alert volume for review costs of 10, 100, 1,000 per alert. With PaySim's large transaction amounts, every one of those costs makes reviewing a large share of all transactions look optimal, which no fraud team could staff. Without real review costs and capacity, the cost model would mislead, so the results above are reported at fixed alert budgets instead.

## Operating point stored with the model

- Alert threshold for the top 0.5% of transactions, set on validation scores: 0.947794
- Score reference for drift monitoring: 10 quantile bins of the validation scores

## Export parity

- Logistic regression: max |exported - Spark| = 3.33e-16
- XGBoost: max |exported - Spark| = 0.00e+00
