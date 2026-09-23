# Liquidity Stress Prediction — Zindi Challenge

Predicting which mobile money customers are likely to experience financial/liquidity
stress in the next 30 days, using six months of transaction history.

**Challenge:** [Can you use mobile money transactions to spot early signs of financial hardship?](https://zindi.africa/) (Zindi)
**Evaluation:** weighted average of Log Loss (60%) and ROC-AUC (40%); submissions are raw probabilities, not class labels.

## Results

5-fold stratified cross-validation, averaged across 3 random seeds (bagging):

| Metric | Single seed | 3-seed bagged (final) |
|---|---|---|
| Log Loss | 0.301 | **0.298** |
| ROC-AUC | 0.858 | **0.862** |
| Combined (0.6·log loss + 0.4·(1-AUC)) | 0.238 | **0.234** |

Hyperparameter tuning (Optuna, 15 trials) was also tried and gave no
meaningful improvement over hand-picked defaults (0.237 vs 0.238) — not
worth the compute given the deadline. Multi-seed bagging gave a real,
if modest, gain instead, so that's what's in the final pipeline.

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

**Model:** LightGBM (binary classification), 5-fold stratified CV, no
explicit class reweighting — the ~15% positive rate isn't severe enough to
need it here, and reweighting was tested and made both log loss and AUC
*worse* by distorting calibration (see the note in `src/train.py`).
Predictions are bagged across 3 random seeds to reduce variance (see
Results above).

Balance trend (`bal_slope`) is by a wide margin the single most predictive
feature, followed by trends in deposit and money-received totals.

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

- Hyperparameter tuning (currently using reasonable defaults, not tuned)
- Target/frequency encoding for `region` instead of one-hot
- Interaction features between balance trend and earning pattern
- Model ensembling (LightGBM + logistic regression blend) for the log-loss component
