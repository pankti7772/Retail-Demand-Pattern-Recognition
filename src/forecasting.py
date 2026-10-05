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


def get_cutoff(max_date, test_days: int = 60):
    """Last training date for the holdout: train <= cutoff < test."""
    return pd.Timestamp(max_date) - pd.Timedelta(days=test_days)


def time_split(df: pd.DataFrame, test_days: int = 60):
    cutoff = get_cutoff(df["date"].max(), test_days)
    train = df[df["date"] <= cutoff]
    test = df[df["date"] > cutoff]
    return train, test


def evaluate(y_true, y_pred) -> dict:
    """Shared RMSE / MAE / MAPE used for models AND baselines."""
    return dict(
        RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        MAE=float(mean_absolute_error(y_true, y_pred)),
        MAPE=float(mape(y_true, y_pred)),
    )


BASELINES = {"naive_lag1": "lag_1", "seasonal_naive_lag7": "lag_7", "moving_avg_7": "roll_mean_7"}


def baseline_metrics(model_df: pd.DataFrame, test_days: int = 60):
    """Naive baselines on the exact same holdout rows/target as the models."""
    _, test = time_split(model_df, test_days=test_days)
    metrics, preds = {}, pd.DataFrame(index=test.index)
    for name, col in BASELINES.items():
        metrics[name] = evaluate(test[TARGET], test[col])
        preds[f"{name}_pred"] = test[col].values
    return metrics, preds


def _fit_models(X_train, y_train):
    xgb = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
        n_jobs=-1, tree_method="hist",
    )
    xgb.fit(X_train, y_train)
    lgbm = LGBMRegressor(
        n_estimators=400, max_depth=-1, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42, verbosity=-1,
    )
    lgbm.fit(X_train, y_train)
    return xgb, lgbm


def rolling_origin_cv(model_df: pd.DataFrame, n_folds: int = 3, test_days: int = 60):
    """Expanding-window CV: fold k trains on all data up to its cutoff and tests
    on the following `test_days`. The last fold equals the main holdout."""
    df_enc, feature_cols = prepare_model_matrix(model_df)
    df_enc[TARGET] = model_df[TARGET].values
    df_enc["date"] = model_df["date"].values
    max_date = df_enc["date"].max()

    folds = []
    for k in range(n_folds):
        test_end = max_date - pd.Timedelta(days=test_days * (n_folds - 1 - k))
        cutoff = test_end - pd.Timedelta(days=test_days)
        train = df_enc[df_enc["date"] <= cutoff]
        test = df_enc[(df_enc["date"] > cutoff) & (df_enc["date"] <= test_end)]
        xgb, lgbm = _fit_models(train[feature_cols], train[TARGET])
        row = dict(fold=k + 1, train_end=str(cutoff.date()), test_start=str(test["date"].min().date()),
                   test_end=str(test_end.date()), n_train=len(train), n_test=len(test))
        for name, m in [("xgboost", xgb), ("lightgbm", lgbm)]:
            pred = np.clip(m.predict(test[feature_cols]), 0, None)
            for metric, val in evaluate(test[TARGET], pred).items():
                row[f"{name}_{metric}"] = val
        folds.append(row)

    folds_df = pd.DataFrame(folds)
    metric_cols = [c for c in folds_df.columns if c.startswith(("xgboost_", "lightgbm_"))]
    summary = {c: {"mean": float(folds_df[c].mean()), "std": float(folds_df[c].std(ddof=1))} for c in metric_cols}
    return folds_df, summary


def train_and_evaluate(model_df: pd.DataFrame, test_days: int = 60):
    df_enc, feature_cols = prepare_model_matrix(model_df)
    # df_enc keeps the same index as model_df, so we can recover original
    # (pre-one-hot) columns like store_id/category/region after the split.
    df_enc[TARGET] = model_df[TARGET].values
    df_enc["date"] = model_df["date"].values
    train, test = time_split(df_enc, test_days=test_days)

    X_train, y_train = train[feature_cols], train[TARGET]
    X_test, y_test = test[feature_cols], test[TARGET]

    xgb, lgbm = _fit_models(X_train, y_train)
    xgb_pred = np.clip(xgb.predict(X_test), 0, None)
    lgbm_pred = np.clip(lgbm.predict(X_test), 0, None)

    metrics = {}
    for name, pred in [("xgboost", xgb_pred), ("lightgbm", lgbm_pred)]:
        metrics[name] = evaluate(y_test, pred)

    meta_cols = ["date", "store_id", "product_id", "category", "region"]
    test_out = model_df.loc[test.index, meta_cols].copy()
    test_out[TARGET] = y_test.values
    test_out["xgb_pred"] = xgb_pred
    test_out["lgbm_pred"] = lgbm_pred

    split_info = dict(n_train_rows=len(train), n_test_rows=len(test))
    return xgb, lgbm, feature_cols, metrics, test_out, split_info
