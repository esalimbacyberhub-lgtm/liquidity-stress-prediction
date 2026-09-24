# Liquidity Stress Prediction — Zindi Challenge

Predicting which mobile money customers are likely to experience financial/liquidity
stress in the next 30 days, using six months of transaction history.

**Challenge:** [Can you use mobile money transactions to spot early signs of financial hardship?](https://zindi.africa/) (Zindi)
**Evaluation:** weighted average of Log Loss (60%) and ROC-AUC (40%); submissions are raw probabilities, not class labels.

## Results

5-fold stratified cross-validation, averaged across 3 random seeds (bagging):

| Version | Log Loss | ROC-AUC | Combined (0.6·LL + 0.4·(1-AUC)) |
|---|---|---|---|
| Single seed, baseline features | 0.301 | 0.858 | 0.238 |
| Hyperparameter tuning (no gain, not adopted) | 0.301 | 0.858 | 0.237 |
| 3-seed bagged, baseline features | 0.298 | 0.862 | 0.234 |
| **3-seed bagged + acceleration/composition/peer-relative features (final)** | **0.296** | **0.865** | **0.232** |

The single most predictive feature in the final model is `bal_slope_z_by_segment`
— a customer's balance trend compared to peers in the same `segment` — by a
wide margin (more than double the importance of the next feature). This
validates the idea that "is this normal for someone like you" carries more
signal than the raw trend alone.

## Approach

The raw dataset gives one row per customer with 6 months (`m1` = most recent,
`m6` = oldest) of activity across 7 transaction types — deposits, merchant
payments, P2P transfers (`mm_send`), bill payments (`paybill`), money
received, bank transfers, and withdrawals — each with 4 metrics (transaction
volume, total value, largest single transaction, unique counterparties),
plus 6 months of daily average balance.

**Key finding from EDA:** the raw monthly values carry weak signal on their
own. What separates customers heading into stress is the *trend* across the
6 months:

| Signal | Stable customers | Customers heading into stress |
|---|---|---|
| Balance change (recent month vs. 6 months ago) | +1,175 | **−6,660** |
| Withdraw-to-received ratio | 0.50 | **0.80** |

This drove the feature engineering: for every transaction type and metric,
`src/features.py` builds a linear trend (slope), month-1-vs-month-6 change,
coefficient of variation, and month-1-vs-6-month-average ratio, plus
cross-type spending ratios (e.g. withdrawals as a share of money received).

Three further feature groups were added after the initial model (see
Results table):
- **Acceleration** — whether a decline is speeding up or slowing down
  (recent 3-month change vs. prior 3-month change), not just the overall
  6-month direction
- **Peer-relative z-scores** — a customer's balance trend compared to
  others in the same `segment`/`earning_pattern` (group stats computed from
  train only, to avoid leakage) — this turned out to be the single most
  useful feature in the model
- **Spending composition** — what share of a customer's total transaction
  value/volume each type (withdrawals, paybill, etc.) represents in the
  most recent month

**Model:** LightGBM (binary classification), 5-fold stratified CV, no
explicit class reweighting — the ~15% positive rate isn't severe enough to
need it here, and reweighting was tested and made both log loss and AUC
*worse* by distorting calibration (see the note in `src/train.py`).
Predictions are bagged across 3 random seeds to reduce variance (see
Results above).

## Files not yet used

`src/tune.py` runs an Optuna hyperparameter search. It's kept in the repo
for reference, but its result wasn't adopted (see Results) — `src/train.py`
uses hand-picked defaults plus seed-bagging instead, which tested better.

## Project structure

```
.
├── README.md
├── requirements.txt
├── data/                  # Train.csv, Test.csv, data_dictionary.csv (not committed — see below)
├── notebooks/
│   └── eda.ipynb          # exploratory analysis behind the findings above
└── src/
    ├── features.py        # feature engineering (trend/volatility/ratio features)
    └── train.py           # CV training, evaluation, submission generation
```

## Data

Competition data isn't committed to this repo (see `.gitignore`). Download
`Train.csv`, `Test.csv`, `SampleSubmission.csv`, and `data_dictionary.csv`
from the Zindi challenge page and place them in `data/`.

## Running it

```bash
pip install -r requirements.txt
cd src
python train.py --train ../data/Train.csv --test ../data/Test.csv --out ../submission.csv
```

This prints per-fold and overall CV metrics, feature importances, and
writes a submission file in the `ID,Target` format the challenge requires.

Note: this trains 3 seeds × 5 folds (15 models total) plus 3 final
retrains, so expect it to take several minutes to run, not seconds.

## Next steps / ideas not yet implemented

- Model diversity: ensemble LightGBM with a different model type (XGBoost/CatBoost) rather than just averaging seeds of the same model
- Target/frequency encoding for `region` instead of one-hot
- Interaction features between balance trend and other categoricals (only segment/earning_pattern tried so far)
