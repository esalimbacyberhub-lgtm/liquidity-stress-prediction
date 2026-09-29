"""
Blend the LightGBM (train.py) and CatBoost (train_catboost.py) models.

Both models score similarly alone (~0.229 combined each) but make different
enough errors that combining their predictions beats either individually.

Combination method: logistic regression stacking in logit space, fit via
5-fold out-of-fold cross-fitting (to avoid the meta-model overfitting to
itself). This beat a fixed-weight average (0.2279) by finding a real
calibration correction -- the raw ensemble average under-predicted the
positive rate by about 1 point (14.0% vs the true 15.0%), and the learned
intercept term fixes exactly that (0.2271). See README for the full
comparison.

This script re-runs both models' OOF and test predictions from scratch (it
does not depend on any cached files from exploration), then stacks them.
Expect this to take a while: LightGBM does 3 seeds x 5 folds, CatBoost does
2 seeds x 5 folds -- roughly 15-20 minutes total on modest hardware.

Usage:
    python src/blend.py --train ../data/Train.csv --test ../data/Test.csv \
        --out ../submission.csv
"""

import argparse

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import KFold

from features import ID_COL, TARGET_COL, add_peer_relative_features, build_features, encode_categoricals
import train as lgb_train
import train_catboost as cb_train

STACK_FOLDS = 5
STACK_SEED = 42


def blended_score(y_true, y_pred_proba) -> dict:
    ll = log_loss(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    return {"log_loss": ll, "roc_auc": auc, "combined_lower_is_better": 0.6 * ll + 0.4 * (1 - auc)}


def _safe_logit(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(p, 1e-6, 1 - 1e-6))


def stack_oof(lgb_oof: np.ndarray, cb_oof: np.ndarray, y: pd.Series) -> np.ndarray:
    """Out-of-fold logistic stacking in logit space. Returns OOF stacked predictions."""
    meta_X = np.column_stack([_safe_logit(lgb_oof), _safe_logit(cb_oof)])
    kf = KFold(n_splits=STACK_FOLDS, shuffle=True, random_state=STACK_SEED)
    stacked = np.zeros(len(y))
    for train_idx, val_idx in kf.split(meta_X):
        meta = LogisticRegression()
        meta.fit(meta_X[train_idx], y.iloc[train_idx])
        stacked[val_idx] = meta.predict_proba(meta_X[val_idx])[:, 1]
    return stacked


def stack_test(lgb_oof: np.ndarray, cb_oof: np.ndarray, y: pd.Series, lgb_test: np.ndarray, cb_test: np.ndarray) -> np.ndarray:
    """Fit the meta-model on ALL OOF data, apply to test predictions."""
    meta_X = np.column_stack([_safe_logit(lgb_oof), _safe_logit(cb_oof)])
    meta = LogisticRegression()
    meta.fit(meta_X, y)
    test_meta_X = np.column_stack([_safe_logit(lgb_test), _safe_logit(cb_test)])
    return meta.predict_proba(test_meta_X)[:, 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="../data/Train.csv")
    parser.add_argument("--test", default="../data/Test.csv")
    parser.add_argument("--out", default="../submission.csv")
    args = parser.parse_args()

    train_raw = pd.read_csv(args.train)
    test_raw = pd.read_csv(args.test)

    # --- LightGBM pipeline (one-hot categoricals) ---
    print("=" * 60)
    print("Training LightGBM...")
    print("=" * 60)
    train_feat = build_features(train_raw)
    test_feat = build_features(test_raw)
    train_feat, test_feat = add_peer_relative_features(train_feat, test_feat)
    train_enc, test_enc = encode_categoricals(train_feat, test_feat)

    y = train_enc[TARGET_COL]
    X_lgb = train_enc.drop(columns=[ID_COL, TARGET_COL])
    X_test_lgb = test_enc.drop(columns=[c for c in [ID_COL, TARGET_COL] if c in test_enc.columns])

    lgb_oof, lgb_best_iters = lgb_train.run_cv(X_lgb, y)
    lgb_test_preds, _ = lgb_train.train_full_and_predict(X_lgb, y, X_test_lgb, lgb_best_iters)

    # --- CatBoost pipeline (native categoricals) ---
    print("\n" + "=" * 60)
    print("Training CatBoost...")
    print("=" * 60)
    train_feat_cb, test_feat_cb, cat_cols = cb_train.prepare_features(train_raw, test_raw)
    X_cb = train_feat_cb.drop(columns=[ID_COL, TARGET_COL])
    X_test_cb = test_feat_cb.drop(columns=[c for c in [ID_COL, TARGET_COL] if c in test_feat_cb.columns])

    all_cb_oof = np.zeros((len(cb_train.SEEDS), len(X_cb)))
    all_cb_best_iters = []
    for i, seed in enumerate(cb_train.SEEDS):
        oof, best_iters = cb_train.run_cv_single_seed(X_cb, y, cat_cols, seed)
        all_cb_oof[i] = oof
        all_cb_best_iters.extend(best_iters)
    cb_oof = all_cb_oof.mean(axis=0)

    from catboost import CatBoostClassifier, Pool

    avg_best_iter = int(np.mean(all_cb_best_iters))
    cb_test_preds = np.zeros(len(X_test_cb))
    for seed in cb_train.SEEDS:
        model = CatBoostClassifier(**{**cb_train.CB_PARAMS, "iterations": avg_best_iter}, random_seed=seed)
        model.fit(Pool(X_cb, y, cat_features=cat_cols))
        cb_test_preds += model.predict_proba(X_test_cb)[:, 1] / len(cb_train.SEEDS)

    # --- Stack (logistic regression in logit space, OOF-fit) ---
    print("\n" + "=" * 60)
    print("Stacking...")
    print("=" * 60)
    print("LightGBM OOF:", blended_score(y, lgb_oof))
    print("CatBoost OOF:", blended_score(y, cb_oof))

    stacked_oof = stack_oof(lgb_oof, cb_oof, y)
    print("Stacked OOF:", blended_score(y, stacked_oof))

    final_test_preds = stack_test(lgb_oof, cb_oof, y, lgb_test_preds, cb_test_preds)

    submission = pd.DataFrame({"ID": test_enc[ID_COL], "Target": final_test_preds})
    submission.to_csv(args.out, index=False)
    print(f"\nSubmission saved to {args.out}")


if __name__ == "__main__":
    main()
