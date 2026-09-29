"""
XGBoost model for the Liquidity Stress Prediction challenge.

Third model type in the ensemble (alongside LightGBM in train.py and
CatBoost in train_catboost.py). Uses the same one-hot encoded features as
LightGBM (train.py) -- XGBoost's native categorical support requires a
newer training path than used here, so one-hot keeps this simple and
consistent with the LightGBM pipeline. The point of adding a third model
isn't better categorical handling (CatBoost already covers that) but
architectural diversity: XGBoost's level-wise tree growth and different
regularization defaults make different errors than LightGBM's leaf-wise
growth or CatBoost's symmetric trees and ordered boosting.

Usage:
    python src/train_xgboost.py --train ../data/Train.csv --test ../data/Test.csv \
        --out ../submission_xgboost.csv
"""

import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features import ID_COL, TARGET_COL, add_peer_relative_features, build_features, encode_categoricals

N_FOLDS = 5
SEEDS = [42, 7]  # 2 seeds, matching CatBoost's budget for a reasonable runtime
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
}
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 100


def blended_score(y_true, y_pred_proba) -> dict:
    ll = log_loss(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    return {"log_loss": ll, "roc_auc": auc, "combined_lower_is_better": 0.6 * ll + 0.4 * (1 - auc)}


def run_cv_single_seed(X: pd.DataFrame, y: pd.Series, seed: int):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(X))
    best_iterations = []

    params = dict(XGB_PARAMS)
    params["seed"] = seed

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dval, "val")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )

        oof_preds[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
        best_iterations.append(model.best_iteration)

    return oof_preds, best_iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="../data/Train.csv")
    parser.add_argument("--test", default="../data/Test.csv")
    parser.add_argument("--out", default="../submission_xgboost.csv")
    args = parser.parse_args()

    train_raw = pd.read_csv(args.train)
    test_raw = pd.read_csv(args.test)

    train_feat = build_features(train_raw)
    test_feat = build_features(test_raw)
    train_feat, test_feat = add_peer_relative_features(train_feat, test_feat)
    train_enc, test_enc = encode_categoricals(train_feat, test_feat)

    y = train_enc[TARGET_COL]
    X = train_enc.drop(columns=[ID_COL, TARGET_COL])
    X_test = test_enc.drop(columns=[c for c in [ID_COL, TARGET_COL] if c in test_enc.columns])

    print(f"Training XGBoost on {X.shape[1]} features, {len(X)} rows.\n")

    all_oof = np.zeros((len(SEEDS), len(X)))
    all_best_iters = []
    for i, seed in enumerate(SEEDS):
        oof, best_iters = run_cv_single_seed(X, y, seed)
        all_oof[i] = oof
        all_best_iters.extend(best_iters)
        scores = blended_score(y, oof)
        print(f"Seed {seed}: {json.dumps(scores, indent=2)}")

    avg_oof = all_oof.mean(axis=0)
    overall = blended_score(y, avg_oof)
    print(f"\n{len(SEEDS)}-seed bagged XGBoost OOF performance:")
    print(json.dumps(overall, indent=2))

    avg_best_iter = int(np.mean(all_best_iters))
    dtrain_full = xgb.DMatrix(X, label=y)
    dtest = xgb.DMatrix(X_test)
    test_preds = np.zeros(len(X_test))
    for seed in SEEDS:
        params = dict(XGB_PARAMS)
        params["seed"] = seed
        model = xgb.train(params, dtrain_full, num_boost_round=avg_best_iter)
        test_preds += model.predict(dtest) / len(SEEDS)

    submission = pd.DataFrame({"ID": test_enc[ID_COL], "Target": test_preds})
    submission.to_csv(args.out, index=False)
    print(f"\nSubmission saved to {args.out}")


if __name__ == "__main__":
    main()
