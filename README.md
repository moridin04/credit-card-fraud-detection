# Credit Card Fraud Detection

> Credit Card Fraud Detection by Detecting Anomalies using Machine Learning

# Fraud Anomaly Detection

> Isolation Forest, a reconstruction Autoencoder, and Local Outlier Factor
> evaluated on 284,807 European card transactions (473 fraudulent, 0.167%
> after de-duplication) under a temporal holdout.
> **The Autoencoder reached PR-AUC 0.149 versus 0.0015 for an
> amount-threshold baseline**, a 100x improvement. At its best operating
> point it flags **2.3 false alarms per true fraud** detected.
>
> **Under the cost model stated below, flagging nothing is still cheaper.**
> See [Cost analysis](#cost-analysis).

## The hypothesis

| | |
|---|---|
| **H0** | No better than a naive amount-threshold rule |
| **H1** | The full feature vector separates fraud better than that rule |

**Verdict: H1 supported.** The autoencoder's PR-AUC of 0.1488 (95% CI
0.0982-0.2254) does not overlap the amount-threshold baseline's 0.0015, and
McNemar's test between Isolation Forest and the Autoencoder gives
chi2=145.47, p<0.0001 (348 discordant pairs).

**Why PR-AUC and not accuracy:** predicting "never fraud" scores 99.87%
accuracy. Accuracy is reported nowhere in this analysis for that reason.

## Data provenance

Dal Pozzolo, Borghesi & Pascal (ICDM 2018). ODbL 1.0.
284,807 x 31 | 492 fraud (0.1727%) | 0 missing | 1,081 exact duplicates,
**19 of which are fraudulent** | Time monotonically increasing, fraud rate
varies 0.077%-0.327% across time deciles.

**Dataset:** Credit Card Fraud Detection

**Source:** Dal Pozzolo, Borghesi & Pascal (ICDM 2018) - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

**License:** Open Database License (ODbL) v1.0

**Not redistributed.** `creditcard.csv` is excluded via `.gitignore`; download it
directly from Kaggle and place it in `data/`.

## Cleaning decisions

| Decision | Rows | Rationale |
|---|---|---|
| Drop exact duplicates | 1,081 (19 fraud) | Disclose the minority-class loss |
| Temporal split 80/20 | - | Time is ordered; a random split leaks the future |
| Drop `Time` as a feature | 1 of 30 | Under a temporal holdout its test values fall entirely outside the train range |
| Scaling inside Pipeline | - | Fitting on full data leaks test statistics |
| Fit scalers on normals only | - | Consistent across all three detectors; 399 fraud rows excluded |

Train: 226,980 rows, 399 fraud (0.1758%).
Test: 56,746 rows, **74 fraud (0.1304%)**, all strictly later in time.

## Baseline

Two baselines, both reported ahead of the models:

- **Always legitimate** - predict nothing. PR-AUC 0.0013 (= test prevalence).
- **Amount > threshold** - the threshold is the 90th percentile of
  *training fraud* amounts, targeting 10% recall. PR-AUC 0.0015,
  ROC-AUC 0.385 (worse than random: a large amount is not by itself evidence
  of fraud), 2,550 false alarms.

The second is deliberately naive. It exists to answer "is the model doing
anything an analyst with a spreadsheet could not?", not to be competitive.

## Results

### Ranking quality (threshold-free)

| Model | PR-AUC | 95% CI | ROC-AUC | P@0.1% | P@1% |
|---|---|---|---|---|---|
| Baseline: always legitimate | 0.0013 | - | 0.5000 | 0.0000 | 0.0000 |
| Baseline: Amount > threshold | 0.0015 | - | 0.3851 | 0.0000 | 0.0053 |
| Local Outlier Factor | 0.0168 | 0.0113-0.0254 | 0.8959 | 0.0000 | 0.0247 |
| Isolation Forest | 0.0434 | 0.0301-0.0613 | 0.9444 | 0.0000 | 0.0653 |
| **Autoencoder** | **0.1488** | **0.0982-0.2254** | 0.9360 | **0.3036** | **0.0882** |

CIs are 2,000-sample percentile bootstraps over the 74 test-set fraud cases.

### Operating points (each model's own calibrated threshold)

| Model | Precision | Recall | F1 | F2 | Alerts/10k | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| Baseline: always legitimate | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 | 0 | 0 | 74 |
| Baseline: Amount > threshold | 0.0023 | 0.0811 | 0.0046 | 0.0105 | 450.4 | 6 | 2550 | 68 |
| Isolation Forest | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10.2 | 0 | 58 | 74 |
| **Autoencoder** | **0.1243** | **0.6351** | 0.2080 | 0.3487 | 66.6 | 47 | 331 | 27 |
| Local Outlier Factor | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 13.6 | 0 | 77 | 74 |

Thresholds are calibrated on training scores at the measured fraud prior
(0.1758%), never on test data. Isolation Forest and LOF flag a sensible number
of transactions but land on none of the 74 fraud cases; only the Autoencoder
catches any at its own threshold.

### Threshold sweep - the Autoencoder

| Alert rate | Alerts | Precision | Recall | FP per TP |
|---|---|---|---|---|
| 0.05% | 28 | 0.2143 | 0.0811 | 3.7 |
| **0.10%** | **56** | **0.3036** | **0.2297** | **2.3** |
| 0.25% | 141 | 0.2128 | 0.4054 | 3.7 |
| 0.50% | 283 | 0.1484 | 0.5676 | 5.7 |
| 1.00% | 567 | 0.0882 | 0.6757 | 10.3 |
| 2.00% | 1,134 | 0.0467 | 0.7162 | 20.4 |
| 5.00% | 2,837 | 0.0201 | 0.7703 | 48.8 |

Precision peaks at the 0.1% budget: 56 alerts, 17 true frauds caught,
**2.3 false alarms per true fraud**. Recall saturates near 0.77 - roughly
a quarter of fraud in this window is not separable by reconstruction error at
any budget.

### Which features the Autoencoder uses

Permutation importance (30 shuffles per feature, drop in PR-AUC):

| Feature | Importance | Std |
|---|---|---|
| Amount | 0.0788 | 0.0013 |
| V2 | 0.0712 | 0.0019 |
| V21 | 0.0709 | 0.0018 |
| V20 | 0.0683 | 0.0019 |
| V1 | 0.0618 | 0.0019 |
| V14 | 0.0468 | 0.0023 |
| V8 | 0.0463 | 0.0019 |

`Amount` is the single strongest driver - the only feature in the dataset
that is not PCA-anonymised. V14 and V17/V18, the components usually cited in
the literature for this dataset, appear later or not at all.

## Cost analysis

Cost model: each undetected fraud costs 1 unit; each false alarm costs
`COST_RATIO` units. Flagging nothing therefore costs 74 units.

**Autoencoder at the 0.1% budget** (the best precision available):

| FP:FN | Cost | vs. flagging nothing (74) |
|---|---|---|
| 1:1 | 96 | worse |
| 5:1 | 252 | worse |
| 20:1 | 837 | worse |

Break-even at 1:1 requires precision above 50%. The Autoencoder peaks at
30.4%. **No budget, for any model, is cost-justified under this cost model.**

This is a result about the cost model as much as about the models. A ratioof 20:1 says a false alarm is twenty times as expensive as a missed fraud -
which holds when review is manual and costly, and breaks down when it is
automated at near-zero marginal cost. At 1:1 the autoencoder misses by 22%,
which is close enough to be worth a real look, but the analysis does not
support deploying any of these detectors as specified.

## Error analysis

Where the autoencoder's top-ranked alerts go wrong at the 0.1% budget:
17 of 74 frauds caught, 39 false alarms. The misses are not concentrated on
low amounts - median fraud amount is close to the population median, so
amount alone carries little signal (the baseline's ROC-AUC of 0.385 confirms
this).

The 27 frauds missed even at a 5% budget appear to be ones whose PCA profile
is close to legitimate traffic. Without the original feature meanings (the
dataset is anonymised) there is no way to attribute them to a merchant type,
channel, or card-present/card-not-present distinction.

## Limitations

- **74 fraud cases in the test set.** Every figure here carries wide
  intervals; a single re-split would move them materially.
- **Unsupervised only.** No supervised baseline (gradient boosting, random
  forest) was run. Published work on this dataset reaches PR-AUC above 0.8  with supervised models, so this study does *not* establish that fraud is
  hard to detect here - only that it is hard to detect without labels.
- **One split, no cross-validation.** Temporal CV would give a better
  estimate of variance across windows.
- **`Time` was dropped**, not because it is uninformative but because under a
  temporal holdout it extrapolates. A feature engineered relative to the
  training window (e.g. hour-of-day as a cyclical encoding) might work better.
- **Thresholds assume the training prior.** `contamination` is 0.1758% from
  train but test prevalence is 0.1304%; alert rates would drift in production.
- **Permutation importance is noisy** at this sample size - the gaps between
  adjacent features are within roughly one standard deviation.

## Reproduce

```bash
# 1. Get the data (not redistributed; ODbL 1.0)
mkdir -p data
#   download creditcard.csv from
#   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
#   and place it in data/

# 2. Run the notebook top to bottom
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=3600 \
    credit_card_fraud_detection.ipynb \
    --output executed.ipynb
