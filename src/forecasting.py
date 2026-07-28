"""
forecasting.py
--------------
Step 6: Demand forecasting with XGBoost (primary MVP model) + LightGBM
(comparison), evaluated with RMSE / MAE / MAPE on a held-out time window.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from features import FEATURE_COLUMNS, CATEGORICAL_FOR_ENCODING

TARGET = "units_sold_capped"


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0.5  # avoid divide-by-near-zero blowing up the metric
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def prepare_model_matrix(df: pd.DataFrame):
    df = df.copy()
    cat_cols = [c for c in CATEGORICAL_FOR_ENCODING if c in df.columns]
    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    encoded_cat_cols = [c for c in df_enc.columns if any(c.startswith(f"{cc}_") for cc in cat_cols)]
    feature_cols = FEATURE_COLUMNS + encoded_cat_cols
    feature_cols = [c for c in feature_cols if c in df_enc.columns]
    return df_enc, feature_cols


def time_split(df: pd.DataFrame, test_days: int = 60):
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    train = df[df["date"] <= cutoff]
    test = df[df["date"] > cutoff]
    return train, test


def train_and_evaluate(model_df: pd.DataFrame, test_days: int = 60):
    df_enc, feature_cols = prepare_model_matrix(model_df)
    # df_enc keeps the same index as model_df, so we can recover original
    # (pre-one-hot) columns like store_id/category/region after the split.
    df_enc[TARGET] = model_df[TARGET].values
    df_enc["date"] = model_df["date"].values
    train, test = time_split(df_enc, test_days=test_days)

    X_train, y_train = train[feature_cols], train[TARGET]
    X_test, y_test = test[feature_cols], test[TARGET]

    xgb = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
        n_jobs=-1, tree_method="hist",
    )
    xgb.fit(X_train, y_train)
    xgb_pred = np.clip(xgb.predict(X_test), 0, None)

    lgbm = LGBMRegressor(
        n_estimators=400, max_depth=-1, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42, verbosity=-1,
    )
    lgbm.fit(X_train, y_train)
    lgbm_pred = np.clip(lgbm.predict(X_test), 0, None)

    metrics = {}
    for name, pred in [("xgboost", xgb_pred), ("lightgbm", lgbm_pred)]:
        metrics[name] = dict(
            RMSE=float(np.sqrt(mean_squared_error(y_test, pred))),
            MAE=float(mean_absolute_error(y_test, pred)),
            MAPE=float(mape(y_test, pred)),
        )

    meta_cols = ["date", "store_id", "product_id", "category", "region"]
    test_out = model_df.loc[test.index, meta_cols].copy()
    test_out[TARGET] = y_test.values
    test_out["xgb_pred"] = xgb_pred
    test_out["lgbm_pred"] = lgbm_pred

    return xgb, lgbm, feature_cols, metrics, test_out
