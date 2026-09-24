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
| 3-seed bagged + acceleration/composition/peer-relative (balance only) | 0.296 | 0.865 | 0.232 |
| CatBoost, same features, native categoricals, 2-seed bagged | 0.296 | 0.864 | 0.232 |
| LightGBM + CatBoost blend, 50/50 | 0.295 | 0.867 | 0.230 |
| LightGBM, peer-relative expanded to 4 more trend features | 0.294 | 0.868 | 0.229 |
| CatBoost, same expanded features, 2-seed bagged | 0.295 | 0.866 | 0.230 |
| **LightGBM + CatBoost blend, 60/40 (final)** | **0.294** | **0.869** | **0.228** |

Model blending gave the single biggest gain of any change tried, and
extending the peer-relative z-score trick (originally just balance trend
vs. segment) to also cover deposit, received, withdraw, and bank-transfer
trends gave the second-biggest gain — both models improved when given the
richer feature set, confirming the peer-relative approach generalizes
beyond the one feature it was first tried on.

The blend weight shifted from 50/50 to 60/40 (favoring LightGBM slightly)
after the feature expansion, found via the same CV grid search approach
(see `src/blend.py`).

The single most predictive feature in the final model is `bal_slope_z_by_segment`
— a customer's balance trend compared to peers in the same `segment` — by a
wide margin (more than double the importance of the next feature). This
validates the idea that "is this normal for someone like you" carries more
signal than the raw trend alone.

**Explored and explicitly not worth pursuing:** target/frequency encoding
for `region` — checked the data first and found stress rates nearly
identical across all 7 regions (13.4%-15.5%), so the category carries weak
signal regardless of encoding, and one-hot isn't a bottleneck at only 7
categories. Skipped before investing time in it.

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
for reference, but its result wasn't adopted (see Results) — the final
pipeline uses hand-picked defaults plus seed-bagging and model blending
instead, which both tested better.

## Project structure

```
.
├── README.md
├── requirements.txt
├── data/                  # Train.csv, Test.csv, data_dictionary.csv (not committed — see below)
├── notebooks/
│   └── eda.ipynb          # exploratory analysis behind the findings above
└── src/
    ├── features.py        # feature engineering (trend/volatility/ratio/peer-relative features)
    ├── train.py            # LightGBM: CV training, evaluation, submission generation
    ├── train_catboost.py   # CatBoost: same features, native categorical handling
    ├── blend.py            # trains both models and blends predictions (final pipeline)
    └── tune.py             # Optuna hyperparameter search (kept for reference, not adopted)
```

## Data

Competition data isn't committed to this repo (see `.gitignore`). Download
`Train.csv`, `Test.csv`, `SampleSubmission.csv`, and `data_dictionary.csv`
from the Zindi challenge page and place them in `data/`.

## Running it

For the final blended submission (recommended):
```bash
pip install -r requirements.txt
cd src
python blend.py --train ../data/Train.csv --test ../data/Test.csv --out ../submission.csv
```

To run either model individually:
```bash
python train.py --train ../data/Train.csv --test ../data/Test.csv --out ../submission_lgb.csv
python train_catboost.py --train ../data/Train.csv --test ../data/Test.csv --out ../submission_cb.csv
```

Note: `blend.py` trains both models from scratch (LightGBM: 3 seeds × 5
folds; CatBoost: 2 seeds × 5 folds), so expect 15-20 minutes to run, not
seconds.

## Next steps / ideas not yet implemented

- Stacking (a meta-model on top of LightGBM + CatBoost predictions) instead of a fixed-weight blend
- Extending peer-relative z-scores to gender/smartphone/region groupings (only segment/earning_pattern tried)
- A third model type (e.g. XGBoost) added to the blend
