# Liquidity Stress Prediction — Zindi Challenge

Predicting which mobile money customers are likely to experience financial/liquidity
stress in the next 30 days, using six months of transaction history.

**Challenge:** [Can you use mobile money transactions to spot early signs of financial hardship?](https://zindi.africa/) (Zindi)
**Evaluation:** weighted average of Log Loss (60%) and ROC-AUC (40%); submissions are raw probabilities, not class labels.

## Results

5-fold stratified cross-validation on the training set:

| Metric | Score |
|---|---|
| Log Loss | 0.301 |
| ROC-AUC | 0.858 |
| Combined (0.6·log loss + 0.4·(1-AUC)) | 0.238 |

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

Balance trend (`bal_slope`) is by a wide margin the single most predictive
feature, followed by trends in deposit and money-received totals.

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

## Next steps / ideas not yet implemented

- Hyperparameter tuning (currently using reasonable defaults, not tuned)
- Target/frequency encoding for `region` instead of one-hot
- Interaction features between balance trend and earning pattern
- Model ensembling (LightGBM + logistic regression blend) for the log-loss component
