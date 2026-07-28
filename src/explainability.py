"""
explainability.py
------------------
Step 7: SHAP-based explainability for the XGBoost forecasting model.
Produces global feature importance + a helper to explain a single
product/store/date prediction in plain business language.
"""

import numpy as np
import pandas as pd
import shap


def compute_shap_values(model, X: pd.DataFrame, sample_size: int = 3000, random_state: int = 42):
    """Sample rows for speed on the MVP; swap to full X for production."""
    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=random_state)
    else:
        X_sample = X
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    return explainer, shap_values, X_sample


def global_feature_importance(shap_values, feature_names) -> pd.DataFrame:
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return imp


def explain_row(explainer, row_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Explain a single prediction (one-row dataframe with the model's feature columns)."""
    sv = explainer(row_df)
    contrib = pd.DataFrame({
        "feature": row_df.columns,
        "value": row_df.iloc[0].values,
        "shap_contribution": sv.values[0],
    })
    contrib["direction"] = np.where(contrib["shap_contribution"] >= 0, "increases demand", "decreases demand")
    contrib["abs_contribution"] = contrib["shap_contribution"].abs()
    return contrib.sort_values("abs_contribution", ascending=False).head(top_n).drop(columns="abs_contribution")
