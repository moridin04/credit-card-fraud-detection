# Credit Card Fraud Detection
Credit Card Fraud Detection by Detecting Anomalies using Machine Learning

# Fraud Anomaly Detection

> Isolation Forest, a reconstruction autoencoder, and Local Outlier Factor
> evaluated on 284,807 European card transactions (492 fraudulent, 0.172%)
> under a temporal holdout. **Isolation Forest reached PR-AUC [X] versus [X]
> for an amount-threshold baseline**, at roughly **[N] false alarms per true
> fraud detected**.

## The hypothesis
| | |
|---|---|
| **H0** | No better than a naive amount-threshold rule |
| **H1** | The full feature vector separates fraud better than that rule |

**Why PR-AUC and not accuracy:** predicting "never fraud" scores 99.83% accuracy.

## Data provenance
Dal Pozzolo, Borghesi & Pascal (ICDM 2018). ODbL 1.0.
284,807 × 31 | 492 fraud (0.172%) | 0 missing | 1,081 exact duplicates,
**19 of which are fraudulent** | Time monotonically increasing, fraud rate
varies 0.077%–0.327% across time deciles.

## Cleaning decisions
| Decision | Rows | Rationale |
|---|---|---|
| Drop exact duplicates | 1,081 (19 fraud) | Disclose the minority-class loss |
| Temporal split 80/20 | — | Time is ordered; a random split leaks the future |
| Scaling inside Pipeline | — | Fitting on full data leaks test statistics |

## Baseline

## Results

## Error analysis

## Limitations

## Reproduce
