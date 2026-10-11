# Results

Generated from `evidence/metrics.json` (run time 783.2 s).
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
| Logistic regression | 0.052 |
| XGBoost | 0.371 |

## Normal-volume test hours (steps 378-408)

253,490 transactions, 386 fraud (0.152%).

| Model | Average precision (95% CI) | Budget | Alerts | Precision | Recall | Value recall |
|---|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.063 (0.037-0.091) | top 0.1% | 254 | 15.4% | 10.1% | 23.1% |
|  |  | top 0.5% | 1,268 | 5.4% | 17.9% | 41.7% |
|  |  | top 1.0% | 2,535 | 3.6% | 23.3% | 48.7% |
| XGBoost | 0.430 (0.376-0.478) | top 0.1% | 254 | 60.6% | 39.9% | 78.9% |
|  |  | top 0.5% | 1,268 | 16.8% | 55.2% | 92.9% |
|  |  | top 1.0% | 2,535 | 10.0% | 65.5% | 94.8% |
| *XGBoost + balance columns (not deployable)* | 0.918 | | | | | |

XGBoost minus logistic regression, average precision: +0.367 (95% CI +0.321 to +0.413; share of resamples where XGBoost was not better: 0.000).

**Existing rule vs the model at the same alert count**

| | Alerts | Precision | Recall | Value recall |
|---|---:|---:|---:|---:|
| `isFlaggedFraud` rule | 1 | 100.0% | 0.3% | 0.7% |
| XGBoost | 1 | 100.0% | 0.3% | 1.3% |

**Cost-optimal alert volume (illustrative costs, loss fraction 1.0)**

| Review cost per alert | Alerts | Alert rate | Cost without model | Cost with model | Saving |
|---:|---:|---:|---:|---:|---:|
| 10 | 210,855 | 83.18% | 723,162,540 | 2,108,550 | 721,053,990 |
| 100 | 21,761 | 8.58% | 723,162,540 | 10,018,421 | 713,144,119 |
| 1,000 | 6,495 | 2.56% | 723,162,540 | 23,183,462 | 699,979,078 |

## Low-volume tail (steps 409-743)

154,587 transactions, 3,624 fraud (2.344%).

| Model | Average precision (95% CI) | Budget | Alerts | Precision | Recall | Value recall |
|---|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.278 (0.263-0.294) | top 0.1% | 155 | 95.5% | 4.1% | 12.8% |
|  |  | top 0.5% | 773 | 63.0% | 13.4% | 28.6% |
|  |  | top 1.0% | 1,546 | 46.3% | 19.8% | 38.0% |
| XGBoost | 0.585 (0.569-0.601) | top 0.1% | 155 | 100.0% | 4.3% | 13.7% |
|  |  | top 0.5% | 773 | 95.7% | 20.4% | 38.3% |
|  |  | top 1.0% | 1,546 | 78.3% | 33.4% | 60.6% |
| *XGBoost + balance columns (not deployable)* | 0.987 | | | | | |

XGBoost minus logistic regression, average precision: +0.307 (95% CI +0.292 to +0.323; share of resamples where XGBoost was not better: 0.000).

**Existing rule vs the model at the same alert count**

| | Alerts | Precision | Recall | Value recall |
|---|---:|---:|---:|---:|
| `isFlaggedFraud` rule | 12 | 100.0% | 0.3% | 1.2% |
| XGBoost | 12 | 100.0% | 0.3% | 1.0% |

**Cost-optimal alert volume (illustrative costs, loss fraction 1.0)**

| Review cost per alert | Alerts | Alert rate | Cost without model | Cost with model | Saving |
|---:|---:|---:|---:|---:|---:|
| 10 | 125,816 | 81.39% | 5,600,645,230 | 1,258,160 | 5,599,387,070 |
| 100 | 125,816 | 81.39% | 5,600,645,230 | 12,581,600 | 5,588,063,630 |
| 1,000 | 36,586 | 23.67% | 5,600,645,230 | 89,018,533 | 5,511,626,697 |

## Whole test period (steps 378-743)

408,077 transactions, 4,010 fraud (0.983%).

| Model | Average precision (95% CI) | Budget | Alerts | Precision | Recall | Value recall |
|---|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.176 (0.164-0.189) | top 0.1% | 409 | 69.2% | 7.1% | 18.8% |
|  |  | top 0.5% | 2,041 | 31.5% | 16.0% | 33.1% |
|  |  | top 1.0% | 4,081 | 22.9% | 23.3% | 43.0% |
| XGBoost | 0.541 (0.526-0.557) | top 0.1% | 409 | 99.5% | 10.1% | 24.2% |
|  |  | top 0.5% | 2,041 | 71.2% | 36.2% | 65.0% |
|  |  | top 1.0% | 4,081 | 51.3% | 52.2% | 88.2% |
| *XGBoost + balance columns (not deployable)* | 0.976 | | | | | |

XGBoost minus logistic regression, average precision: +0.366 (95% CI +0.353 to +0.380; share of resamples where XGBoost was not better: 0.000).

**Existing rule vs the model at the same alert count**

| | Alerts | Precision | Recall | Value recall |
|---|---:|---:|---:|---:|
| `isFlaggedFraud` rule | 13 | 100.0% | 0.3% | 1.1% |
| XGBoost | 13 | 100.0% | 0.3% | 1.5% |

**Cost-optimal alert volume (illustrative costs, loss fraction 1.0)**

| Review cost per alert | Alerts | Alert rate | Cost without model | Cost with model | Saving |
|---:|---:|---:|---:|---:|---:|
| 10 | 337,018 | 82.59% | 6,323,807,770 | 3,370,180 | 6,320,437,590 |
| 100 | 337,018 | 82.59% | 6,323,807,770 | 33,701,800 | 6,290,105,970 |
| 1,000 | 44,761 | 10.97% | 6,323,807,770 | 126,850,032 | 6,196,957,738 |

## Export parity

- Logistic regression: max |exported - Spark| = 3.33e-16
- XGBoost: max |exported - Spark| = 0.00e+00
