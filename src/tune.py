"""
Hyperparameter tuning for the Liquidity Stress Prediction LightGBM model.

Searches LightGBM hyperparameters using Optuna, optimizing the same 5-fold
CV blended metric (0.6*log_loss + 0.4*(1-roc_auc)) used in train.py, so the
result is directly comparable to the untuned baseline (0.238).

Usage:
    python src/tune.py --train ../data/Train.csv --n-trials 50
"""

import argparse

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from features import ID_COL, TARGET_COL, build_features, encode_categoricals

# Search uses a lighter CV setup than train.py's final evaluation (3 folds,
# fewer boosting rounds) purely for speed -- ~165s per trial at 5-fold/2000-round
# settings was too slow for a 3-day deadline. The winning params get validated
# with the full 5-fold setup afterward in train.py, so this speed/accuracy
# tradeoff only affects the search, not the final reported score.
N_FOLDS = 3
RANDOM_STATE = 42
NUM_BOOST_ROUND = 400
EARLY_STOPPING_ROUNDS = 25
SEARCH_SAMPLE_FRAC = 0.35  # subsample rows during search only, for speed

optuna.logging.set_verbosity(optuna.logging.WARNING)


def cv_score(params: dict, X: pd.DataFrame, y: pd.Series) -> float:
    """Run 5-fold CV for a given param set, return the blended combined score (lower is better)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X))

    full_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "seed": RANDOM_STATE,
        "verbosity": -1,
        **params,
    }

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        model = lgb.train(
            full_params,
            train_set,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[val_set],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)

    ll = log_loss(y, oof_preds)
    auc = roc_auc_score(y, oof_preds)
    return 0.6 * ll + 0.4 * (1 - auc)


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 5.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 5.0, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
    }
    return cv_score(params, X, y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="../data/Train.csv")
    parser.add_argument("--n-trials", type=int, default=50)
    args = parser.parse_args()

    train_raw = pd.read_csv(args.train)
    train_feat = build_features(train_raw)
    # tune only needs train-side encoding; use train as both sides for get_dummies alignment
    train_enc, _ = encode_categoricals(train_feat, train_feat.copy())

    y_full = train_enc[TARGET_COL]
    X_full = train_enc.drop(columns=[ID_COL, TARGET_COL])

    # Subsample (stratified) for search speed; final params get validated on full data in train.py
    sample_idx, _ = train_test_split(
        X_full.index, train_size=SEARCH_SAMPLE_FRAC, stratify=y_full, random_state=RANDOM_STATE
    )
    X, y = X_full.loc[sample_idx], y_full.loc[sample_idx]

    print(f"Tuning on {X.shape[1]} features, {len(X)} rows (subsampled from {len(X_full)}), {args.n_trials} trials...\n")

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(lambda trial: objective(trial, X, y), n_trials=args.n_trials, show_progress_bar=False)

    print("\nBest combined score:", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Save best params to a file for train.py to pick up
    with open("best_params.txt", "w") as f:
        f.write(str(study.best_params))
    print("\nSaved to best_params.txt")


if __name__ == "__main__":
    main()
