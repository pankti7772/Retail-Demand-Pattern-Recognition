"""
anomaly.py
----------
Step 5: Anomaly Detection via Isolation Forest, run on each product/store
series' residual-style features (actual vs recent rolling mean, volatility)
so it flags unusual spikes/drops rather than just "high sales" days.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ANOMALY_FEATURES = ["units_sold_capped", "roll_mean_7", "roll_std_7", "deviation_from_roll_mean"]


def run_anomaly_detection(model_df: pd.DataFrame, contamination: float = 0.02, random_state: int = 42,
                          train_cutoff=None):
    """train_cutoff: if given, the forest is fitted only on rows with
    date <= train_cutoff and then used to score all rows."""
    df = model_df.copy()
    df["deviation_from_roll_mean"] = (
        (df["units_sold_capped"] - df["roll_mean_7"]) / (df["roll_std_7"] + 1e-6)
    )

    feats = df[ANOMALY_FEATURES].fillna(0).values
    fit_mask = np.ones(len(df), dtype=bool) if train_cutoff is None else (df["date"] <= pd.Timestamp(train_cutoff)).values
    iso = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=200)
    iso.fit(feats[fit_mask])
    preds = iso.predict(feats)              # -1 = anomaly, 1 = normal
    scores = iso.decision_function(feats)   # lower = more anomalous

    df["is_anomaly"] = (preds == -1)
    df["anomaly_score"] = scores
    df["anomaly_type"] = np.where(
        df["is_anomaly"] & (df["deviation_from_roll_mean"] > 0), "Spike",
        np.where(df["is_anomaly"] & (df["deviation_from_roll_mean"] < 0), "Drop / Possible Stock-Out", "Normal")
    )
    return df, iso
