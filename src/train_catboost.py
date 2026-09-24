"""
CatBoost model for the Liquidity Stress Prediction challenge.

Unlike train.py (LightGBM + one-hot encoded categoricals), this uses
CatBoost's native categorical handling on the raw category columns
(gender, region, smartphone, segment, earning_pattern). Different model
types make different errors, so blending this with the LightGBM bag from
train.py is a more promising lever than further tuning either model alone.

Usage:
    python src/train_catboost.py --train ../data/Train.csv --test ../data/Test.csv \
        --out ../submission_catboost.csv
"""

import argparse
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features import CATEGORICAL_COLS, ID_COL, TARGET_COL, add_peer_relative_features, build_features

N_FOLDS = 5
SEEDS = [42, 7]  # 2 seeds (not 3) to keep runtime reasonable for blending experiments
CB_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "iterations": 1200,
    "learning_rate": 0.06,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "verbose": False,
}
EARLY_STOPPING_ROUNDS = 60


def blended_score(y_true, y_pred_proba) -> dict:
    ll = log_loss(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    return {"log_loss": ll, "roc_auc": auc, "combined_lower_is_better": 0.6 * ll + 0.4 * (1 - auc)}


def prepare_features(train_raw: pd.DataFrame, test_raw: pd.DataFrame):
    """Build features but KEEP categoricals as raw strings (no one-hot) for CatBoost."""
    train_feat = build_features(train_raw)
    test_feat = build_features(test_raw)
    train_feat, test_feat = add_peer_relative_features(train_feat, test_feat)

    cat_cols = [c for c in CATEGORICAL_COLS if c in train_feat.columns]
    for c in cat_cols:
        train_feat[c] = train_feat[c].astype(str)
        test_feat[c] = test_feat[c].astype(str)

    return train_feat, test_feat, cat_cols


def run_cv_single_seed(X: pd.DataFrame, y: pd.Series, cat_cols: list[str], seed: int):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(X))
    best_iterations = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_pool = Pool(X_train, y_train, cat_features=cat_cols)
        val_pool = Pool(X_val, y_val, cat_features=cat_cols)

        model = CatBoostClassifier(**CB_PARAMS, random_seed=seed)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=EARLY_STOPPING_ROUNDS, use_best_model=True)

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        best_iterations.append(model.get_best_iteration())

    return oof_preds, best_iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="../data/Train.csv")
    parser.add_argument("--test", default="../data/Test.csv")
    parser.add_argument("--out", default="../submission_catboost.csv")
    args = parser.parse_args()

    train_raw = pd.read_csv(args.train)
    test_raw = pd.read_csv(args.test)
    train_feat, test_feat, cat_cols = prepare_features(train_raw, test_raw)

    y = train_feat[TARGET_COL]
    X = train_feat.drop(columns=[ID_COL, TARGET_COL])
    X_test = test_feat.drop(columns=[c for c in [ID_COL, TARGET_COL] if c in test_feat.columns])

    print(f"Training CatBoost on {X.shape[1]} features ({len(cat_cols)} categorical), {len(X)} rows.\n")

    all_oof = np.zeros((len(SEEDS), len(X)))
    all_best_iters = []
    for i, seed in enumerate(SEEDS):
        oof, best_iters = run_cv_single_seed(X, y, cat_cols, seed)
        all_oof[i] = oof
        all_best_iters.extend(best_iters)
        scores = blended_score(y, oof)
        print(f"Seed {seed}: {json.dumps(scores, indent=2)}")

    avg_oof = all_oof.mean(axis=0)
    overall = blended_score(y, avg_oof)
    print(f"\n{len(SEEDS)}-seed bagged CatBoost OOF performance:")
    print(json.dumps(overall, indent=2))

    np.save("catboost_oof.npy", avg_oof)

    avg_best_iter = int(np.mean(all_best_iters))
    test_preds = np.zeros(len(X_test))
    for seed in SEEDS:
        model = CatBoostClassifier(**{**CB_PARAMS, "iterations": avg_best_iter}, random_seed=seed)
        model.fit(Pool(X, y, cat_features=cat_cols), verbose=False)
        test_preds += model.predict_proba(X_test)[:, 1] / len(SEEDS)

    np.save("catboost_test_preds.npy", test_preds)
    submission = pd.DataFrame({"ID": test_feat[ID_COL], "Target": test_preds})
    submission.to_csv(args.out, index=False)
    print(f"\nSubmission saved to {args.out}")


if __name__ == "__main__":
    main()
