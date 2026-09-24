"""
Feature engineering for the Zindi Liquidity Stress Prediction challenge.

The raw data gives 6 months (m1 = most recent, m6 = oldest) of mobile money
behaviour across 7 transaction types (deposit, merchantpay, mm_send, paybill,
received, transfer_from_bank, withdraw), each with 4 metrics (volume,
total_value, highest_amount, unique-counterparty count) plus 6 months of
daily average balance.

EDA on the training data showed the strongest signals are not the raw
monthly values but how they *change* over the 6-month window:
  - balance trend (m1 - m6): customers heading toward stress show sharply
    declining balances (mean -6,660 vs +1,175 for stable customers)
  - withdraw-to-received ratio: stressed customers spend down a larger
    share of what they receive (0.80 vs 0.50 median)

This module turns the wide, repeated-per-month columns into a compact set
of trend, volatility, and ratio features per customer.
"""

import numpy as np
import pandas as pd

MONTHS = [1, 2, 3, 4, 5, 6]  # m1 = most recent, m6 = oldest
TXN_TYPES = [
    "deposit",
    "merchantpay",
    "mm_send",
    "paybill",
    "received",
    "transfer_from_bank",
    "withdraw",
]
# The "unique counterparty" column has a different suffix per transaction type
COUNTERPARTY_SUFFIX = {
    "deposit": "agents",
    "merchantpay": "merchants",
    "mm_send": "recipients",
    "paybill": "companies",
    "received": "senders",
    "transfer_from_bank": "banks",
    "withdraw": "agents",
}
CATEGORICAL_COLS = ["gender", "region", "smartphone", "segment", "earning_pattern"]
ID_COL = "ID"
TARGET_COL = "liquidity_stress_next_30d"


def _linear_slope(values: np.ndarray) -> np.ndarray:
    """Slope of a simple linear fit across months 6->1 (oldest to newest) per row."""
    # x runs oldest (m6) to newest (m1) so a positive slope means "increasing recently"
    x = np.arange(len(MONTHS))  # 0..5 in m6..m1 order
    x_mean = x.mean()
    x_centered = x - x_mean
    denom = (x_centered**2).sum()
    y_centered = values - values.mean(axis=1, keepdims=True)
    return (y_centered * x_centered).sum(axis=1) / denom


def _trend_features(df: pd.DataFrame, cols_recent_to_old: list[str], name: str) -> pd.DataFrame:
    """Given columns ordered m1..m6 (recent to old), build trend/volatility features."""
    values = df[cols_recent_to_old].to_numpy(dtype=float)
    # reorder to oldest->recent (m6..m1) for slope interpretation
    values_chrono = values[:, ::-1]

    m1 = values[:, 0]
    m6 = values[:, -1]
    mean_ = values.mean(axis=1)
    std_ = values.std(axis=1)

    out = pd.DataFrame(index=df.index)
    out[f"{name}_m1_m6_diff"] = m1 - m6
    with np.errstate(divide="ignore", invalid="ignore"):
        out[f"{name}_m1_m6_ratio"] = np.where(m6 != 0, m1 / m6, np.nan)
        out[f"{name}_cv"] = np.where(mean_ != 0, std_ / mean_, np.nan)  # coefficient of variation
        out[f"{name}_m1_over_avg"] = np.where(mean_ != 0, m1 / mean_, np.nan)
    out[f"{name}_slope"] = _linear_slope(values_chrono)
    out[f"{name}_mean"] = mean_
    out[f"{name}_std"] = std_
    out[f"{name}_m1"] = m1
    return out


def _acceleration(df: pd.DataFrame, cols_recent_to_old: list[str]) -> np.ndarray:
    """
    Whether change is speeding up or slowing down: compares the most recent
    3-month change to the prior 3-month change. Positive = recent change is
    larger (accelerating) than the earlier change; distinct from the overall
    6-month slope, which can look identical for a steady decline vs. one that
    just started crashing in the last 3 months.
    """
    values = df[cols_recent_to_old].to_numpy(dtype=float)  # m1..m6 (recent to old)
    recent_delta = values[:, 0] - values[:, 2]  # m1 - m3
    older_delta = values[:, 3] - values[:, 5]  # m4 - m6
    return recent_delta - older_delta


def add_peer_relative_features(train_feat: pd.DataFrame, test_feat: pd.DataFrame):
    """
    Add peer-relative z-scores for balance trend features, computed within
    each segment and earning_pattern group. Group statistics (mean/std) are
    computed from TRAIN ONLY and applied to both train and test, to avoid
    leaking test information into the normalization.

    Rationale: a declining balance may be normal for one segment/earning
    pattern and alarming for another -- comparing a customer to their own
    peer group can surface stress that a population-wide feature misses.
    """
    train_feat = train_feat.copy()
    test_feat = test_feat.copy()
    base_cols = ["bal_slope", "bal_m1_m6_diff"]
    group_cols = ["segment", "earning_pattern"]

    for group_col in group_cols:
        if group_col not in train_feat.columns:
            continue
        stats = train_feat.groupby(group_col)[base_cols].agg(["mean", "std"])
        for base_col in base_cols:
            group_mean = stats[(base_col, "mean")]
            group_std = stats[(base_col, "std")].replace(0, np.nan)
            new_col = f"{base_col}_z_by_{group_col}"

            for feat in (train_feat, test_feat):
                mapped_mean = feat[group_col].map(group_mean)
                mapped_std = feat[group_col].map(group_std)
                feat[new_col] = (feat[base_col] - mapped_mean) / mapped_std

    return train_feat, test_feat


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full engineered feature set from a raw Train/Test dataframe.
    Returns a new dataframe (does not mutate the input), keeping ID and,
    if present, the target column untouched alongside the engineered features.

    Note: peer-relative features (see add_peer_relative_features) are NOT
    included here since they require both train and test together to avoid
    leakage -- call add_peer_relative_features() separately after this.
    """
    df = df.copy()
    feature_frames = []

    # --- Balance trend/volatility ---
    bal_cols = [f"m{m}_daily_avg_bal" for m in MONTHS]
    feature_frames.append(_trend_features(df, bal_cols, "bal"))
    feature_frames.append(pd.DataFrame({"bal_acceleration": _acceleration(df, bal_cols)}, index=df.index))

    # personalized "low balance" flag: number of months below the customer's own median
    bal_values = df[bal_cols].to_numpy(dtype=float)
    row_median = np.median(bal_values, axis=1, keepdims=True)
    below_median_months = (bal_values < row_median).sum(axis=1)
    feature_frames.append(pd.DataFrame({"bal_months_below_own_median": below_median_months}, index=df.index))

    # --- Per transaction-type trend/volatility for each of the 4 metrics ---
    per_type_m1_total_value = {}
    per_type_m1_volume = {}
    for txn in TXN_TYPES:
        counterparty_col = f"m{{m}}_{txn}_{COUNTERPARTY_SUFFIX[txn]}"
        metrics = {
            "volume": f"m{{m}}_{txn}_volume",
            "total_value": f"m{{m}}_{txn}_total_value",
            "highest_amount": f"m{{m}}_{txn}_highest_amount",
            "counterparties": counterparty_col,
        }
        for metric_name, template in metrics.items():
            cols_recent_to_old = [template.format(m=m) for m in MONTHS]
            if not all(c in df.columns for c in cols_recent_to_old):
                continue
            feat_name = f"{txn}_{metric_name}"
            feature_frames.append(_trend_features(df, cols_recent_to_old, feat_name))
            if metric_name == "total_value":
                per_type_m1_total_value[txn] = df[cols_recent_to_old[0]]
                # acceleration only for total_value (the metric most tied to stress)
                feature_frames.append(
                    pd.DataFrame({f"{feat_name}_acceleration": _acceleration(df, cols_recent_to_old)}, index=df.index)
                )
            elif metric_name == "volume":
                per_type_m1_volume[txn] = df[cols_recent_to_old[0]]

    # --- Cross-type ratios in the most recent month (spending vs income patterns) ---
    received_m1 = df["m1_received_total_value"].replace(0, np.nan)
    ratio_frame = pd.DataFrame(index=df.index)
    for txn in ["withdraw", "paybill", "mm_send", "merchantpay"]:
        col = f"m1_{txn}_total_value"
        if col in df.columns:
            ratio_frame[f"{txn}_to_received_ratio_m1"] = df[col] / received_m1
    feature_frames.append(ratio_frame)

    # --- Spending composition: each type's share of total activity in month 1 ---
    # Captures WHERE money goes, not just how much -- two customers with the
    # same total withdrawal amount can look very different if one customer's
    # activity is almost entirely withdrawals vs. spread across several types.
    composition_frame = pd.DataFrame(index=df.index)
    total_value_sum = sum(per_type_m1_total_value.values()).replace(0, np.nan)
    for txn, series in per_type_m1_total_value.items():
        composition_frame[f"{txn}_total_value_share_m1"] = series / total_value_sum
    total_volume_sum = sum(per_type_m1_volume.values()).replace(0, np.nan)
    for txn, series in per_type_m1_volume.items():
        composition_frame[f"{txn}_volume_share_m1"] = series / total_volume_sum
    feature_frames.append(composition_frame)

    # --- Passthrough numeric/categorical customer-profile features ---
    passthrough_cols = ["arpu", "age", "x_90_d_activity_rate"] + CATEGORICAL_COLS
    passthrough_cols = [c for c in passthrough_cols if c in df.columns]
    feature_frames.append(df[passthrough_cols])

    # --- ID (and target, if present) passthrough ---
    keep_cols = [c for c in [ID_COL, TARGET_COL] if c in df.columns]
    feature_frames.append(df[keep_cols])

    result = pd.concat(feature_frames, axis=1)
    # de-fragment
    return result.copy()


def encode_categoricals(train_feat: pd.DataFrame, test_feat: pd.DataFrame):
    """One-hot encode categorical columns, aligning train/test columns."""
    cat_cols = [c for c in CATEGORICAL_COLS if c in train_feat.columns]
    train_enc = pd.get_dummies(train_feat, columns=cat_cols, drop_first=True)
    test_enc = pd.get_dummies(test_feat, columns=cat_cols, drop_first=True)
    train_enc, test_enc = train_enc.align(test_enc, join="left", axis=1, fill_value=0)
    return train_enc, test_enc
