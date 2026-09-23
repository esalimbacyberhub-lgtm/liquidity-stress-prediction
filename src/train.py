"""
Train a LightGBM model for the Liquidity Stress Prediction challenge and
report cross-validated performance on the challenge's actual scoring metric:
    final_score = 0.6 * log_loss + 0.4 * (1 - roc_auc)

(ROC-AUC is expressed as 1 - AUC here purely so that, like log loss, LOWER
is better for both terms -- makes the blended score easy to read as
"lower is better" throughout. Zindi's own leaderboard formula may display
this differently; check the challenge's evaluation page if you need to
match their exact number.)

Usage:
    python src/train.py --train ../data/Train.csv --test ../data/Test.csv \
        --out ../submission.csv
"""

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features import ID_COL, TARGET_COL, build_features, encode_categoricals

N_FOLDS = 5
RANDOM_STATE = 42

SEEDS = [42, 7, 123]  # multi-seed bagging: averaging 3 seeds improved combined
                       # score from 0.238 -> 0.234 in testing (both log loss and
                       # AUC improved), a standard variance-reduction technique
LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    # NOTE: class-reweighting (is_unbalance / scale_pos_weight) was tested and
    # made BOTH log loss and AUC worse on this data (0.400/0.802 vs 0.301/0.858
    # without it) -- at ~15% positive rate the imbalance isn't severe enough
    # to need reweighting, and reweighting distorts calibration, which hurts
    # log loss directly (60% of the challenge score). Left off deliberately.
    "seed": RANDOM_STATE,
    "verbosity": -1,
}
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 100


def blended_score(y_true, y_pred_proba) -> dict:
    ll = log_loss(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    combined = 0.6 * ll + 0.4 * (1 - auc)
    return {"log_loss": ll, "roc_auc": auc, "combined_lower_is_better": combined}


def run_cv_single_seed(X: pd.DataFrame, y: pd.Series, seed: int):
    """Stratified K-fold CV for one seed. Returns OOF predictions, fold scores, best iterations."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(X))
    fold_scores = []
    best_iterations = []

    params = dict(LGB_PARAMS)
    params["seed"] = seed

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        model = lgb.train(
            params,
            train_set,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[val_set],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred
        best_iterations.append(model.best_iteration)

        scores = blended_score(y_val, val_pred)
        scores["fold"] = fold
        fold_scores.append(scores)

    return oof_preds, fold_scores, best_iterations


def run_cv(X: pd.DataFrame, y: pd.Series):
    """
    Multi-seed bagged CV: runs full 5-fold CV once per seed in SEEDS, then
    averages the out-of-fold predictions across seeds. This is a standard
    variance-reduction technique -- testing showed it improves both log loss
    and AUC over a single-seed run (combined score 0.238 -> 0.234).
    """
    all_oof = np.zeros((len(SEEDS), len(X)))
    all_best_iterations = []

    for seed in SEEDS:
        oof, fold_scores, best_iterations = run_cv_single_seed(X, y, seed)
        seed_idx = SEEDS.index(seed)
        all_oof[seed_idx] = oof
        all_best_iterations.extend(best_iterations)

        seed_overall = blended_score(y, oof)
        print(
            f"Seed {seed}: log_loss={seed_overall['log_loss']:.4f} "
            f"roc_auc={seed_overall['roc_auc']:.4f} "
            f"combined={seed_overall['combined_lower_is_better']:.4f}"
        )

    avg_oof = all_oof.mean(axis=0)
    overall = blended_score(y, avg_oof)
    print(f"\n{len(SEEDS)}-seed bagged OOF performance:")
    print(json.dumps(overall, indent=2))

    return avg_oof, all_best_iterations


def train_full_and_predict(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, best_iterations: list[int]):
    """
    Retrain on all data once per seed (using the average best-iteration from
    CV across all seeds/folds), predict on test with each, and average --
    mirroring the bagging done in run_cv so the test predictions benefit from
    the same variance reduction as the validated CV score.
    """
    avg_best_iter = int(np.mean(best_iterations))
    test_preds = np.zeros(len(X_test))
    final_model = None

    for seed in SEEDS:
        params = dict(LGB_PARAMS)
        params["seed"] = seed
        train_set = lgb.Dataset(X, label=y)
        model = lgb.train(params, train_set, num_boost_round=avg_best_iter)
        test_preds += model.predict(X_test) / len(SEEDS)
        final_model = model  # keep the last one around for feature importance reporting

    return test_preds, final_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="../data/Train.csv")
    parser.add_argument("--test", default="../data/Test.csv")
    parser.add_argument("--out", default="../submission.csv")
    args = parser.parse_args()

    train_raw = pd.read_csv(args.train)
    test_raw = pd.read_csv(args.test)

    train_feat = build_features(train_raw)
    test_feat = build_features(test_raw)
    train_enc, test_enc = encode_categoricals(train_feat, test_feat)

    y = train_enc[TARGET_COL]
    X = train_enc.drop(columns=[ID_COL, TARGET_COL])
    X_test = test_enc.drop(columns=[c for c in [ID_COL, TARGET_COL] if c in test_enc.columns])

    print(f"Training on {X.shape[1]} engineered features, {len(X)} rows.\n")

    oof_preds, best_iterations = run_cv(X, y)

    test_preds, final_model = train_full_and_predict(X, y, X_test, best_iterations)

    submission = pd.DataFrame({"ID": test_enc[ID_COL], "Target": test_preds})
    submission.to_csv(args.out, index=False)
    print(f"\nSubmission saved to {args.out}")

    importance = pd.Series(
        final_model.feature_importance(importance_type="gain"), index=X.columns
    ).sort_values(ascending=False)
    print("\nTop 15 features by gain:")
    print(importance.head(15))


if __name__ == "__main__":
    main()
