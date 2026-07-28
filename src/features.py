"""
features.py
-----------
Step 1 (Data Cleaning) + Step 3 (Feature Engineering) of the pipeline.
"""

import numpy as np
import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Missing values, duplicates, date handling, basic outlier capping."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    before = len(df)
    df = df.drop_duplicates(subset=["date", "store_id", "product_id"])
    dupes_removed = before - len(df)

    df = df.sort_values(["product_id", "store_id", "date"])

    # Fill missing units_sold with the product/store rolling median (fallback to category median)
    df["units_sold"] = (
        df.groupby(["product_id", "store_id"])["units_sold"]
        .transform(lambda s: s.fillna(s.rolling(7, min_periods=1, center=True).median()))
    )
    df["units_sold"] = df["units_sold"].fillna(
        df.groupby("category")["units_sold"].transform("median")
    )
    df["units_sold"] = df["units_sold"].clip(lower=0)

    # Cap extreme outliers at the 1st/99th percentile *per product* (winsorize),
    # rather than deleting them -- true anomalies are handled later by Isolation Forest.
    def winsorize(s):
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        return s.clip(lo, hi)

    df["units_sold_capped"] = df.groupby("product_id")["units_sold"].transform(winsorize)

    meta = {"duplicates_removed": int(dupes_removed), "rows_after_clean": len(df)}
    return df, meta


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["week_number"] = df["date"].dt.isocalendar().week.astype(int)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame, target_col="units_sold_capped") -> pd.DataFrame:
    df = df.copy()
    grp = df.groupby(["product_id", "store_id"])[target_col]

    for lag in [1, 7, 14, 30]:
        df[f"lag_{lag}"] = grp.shift(lag)

    key = ["product_id", "store_id"]
    df["roll_mean_7"] = df.groupby(key)[target_col].transform(lambda s: s.shift(1).rolling(7).mean())
    df["roll_mean_14"] = df.groupby(key)[target_col].transform(lambda s: s.shift(1).rolling(14).mean())
    df["roll_mean_30"] = df.groupby(key)[target_col].transform(lambda s: s.shift(1).rolling(30).mean())
    df["roll_std_7"] = df.groupby(key)[target_col].transform(lambda s: s.shift(1).rolling(7).std())
    df["roll_std_30"] = df.groupby(key)[target_col].transform(lambda s: s.shift(1).rolling(30).std())

    return df


def add_price_and_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_ratio_to_base"] = df["price"] / df["base_price"]
    df["discount_pct"] = 1 - df["price_ratio_to_base"]

    df = df.sort_values(["product_id", "store_id", "date"])
    df["promo_frequency_30"] = (
        df.groupby(["product_id", "store_id"])["promo_flag"]
        .transform(lambda s: s.shift(1).rolling(30, min_periods=1).mean())
    )
    return df


def build_feature_table(raw: pd.DataFrame):
    df, clean_meta = clean_data(raw)
    df = add_calendar_features(df)
    df = add_lag_and_rolling_features(df)
    df = add_price_and_promo_features(df)

    # Drop the warm-up rows that don't yet have a 30-day lag (can't be modeled reliably)
    model_df = df.dropna(subset=["lag_30", "roll_mean_30"]).reset_index(drop=True)
    return df, model_df, clean_meta


FEATURE_COLUMNS = [
    "lag_1", "lag_7", "lag_14", "lag_30",
    "roll_mean_7", "roll_mean_14", "roll_mean_30",
    "roll_std_7", "roll_std_30",
    "price_ratio_to_base", "discount_pct", "promo_frequency_30",
    "promo_flag", "is_holiday", "is_festival",
    "day_of_week", "is_weekend", "month", "quarter", "week_number",
]

CATEGORICAL_FOR_ENCODING = ["category", "region", "store_id"]
