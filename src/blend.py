"""
Blend the LightGBM (train.py) and CatBoost (train_catboost.py) models.

Both models score similarly alone (~0.232 combined each) but make different
enough errors that averaging their predictions beats either individually.
A 50/50 blend was found to be optimal via a CV grid search over blend
weights (see README for the full weight-vs-score table).

This script re-runs both models' OOF and test predictions from scratch (it
does not depend on any cached files from exploration), then blends them.
Expect this to take a while: LightGBM does 3 seeds x 5 folds, CatBoost does
2 seeds x 5 folds -- roughly 15-20 minutes total on modest hardware.

Usage:
    python src/blend.py --train ../data/Train.csv --test ../data/Test.csv \
        --out ../submission.csv
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from features import ID_COL, TARGET_COL, add_peer_relative_features, build_features, encode_categoricals
import train as lgb_train
import train_catboost as cb_train

BLEND_WEIGHT_LGB = 0.60  # found via CV grid search on the expanded feature set; see README


def blended_score(y_true, y_pred_proba) -> dict:
    ll = log_loss(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    return {"log_loss": ll, "roc_auc": auc, "combined_lower_is_better": 0.6 * ll + 0.4 * (1 - auc)}


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

    # --- Blend ---
    print("\n" + "=" * 60)
    print("Blending...")
    print("=" * 60)
    print("LightGBM OOF:", blended_score(y, lgb_oof))
    print("CatBoost OOF:", blended_score(y, cb_oof))

    blend_oof = BLEND_WEIGHT_LGB * lgb_oof + (1 - BLEND_WEIGHT_LGB) * cb_oof
    print(f"Blend (w_lgb={BLEND_WEIGHT_LGB}) OOF:", blended_score(y, blend_oof))

    blend_test_preds = BLEND_WEIGHT_LGB * lgb_test_preds + (1 - BLEND_WEIGHT_LGB) * cb_test_preds

    submission = pd.DataFrame({"ID": test_enc[ID_COL], "Target": blend_test_preds})
    submission.to_csv(args.out, index=False)
    print(f"\nSubmission saved to {args.out}")


if __name__ == "__main__":
    main()
