"""
clustering.py
-------------
Step 4: Pattern Recognition via clustering. Groups PRODUCTS (not rows)
by demand trend, seasonality, volatility and price behaviour, then
gives each cluster a human-readable name based on its characteristics.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def build_product_profile(df: pd.DataFrame) -> pd.DataFrame:
    """One row per product summarizing its demand behaviour."""
    daily = df.groupby(["product_id", "date"], as_index=False)["units_sold_capped"].sum()

    profiles = []
    for pid, g in daily.groupby("product_id"):
        g = g.sort_values("date")
        vals = g["units_sold_capped"].values
        mean_demand = vals.mean()
        cv = vals.std() / (mean_demand + 1e-6)  # volatility, coefficient of variation

        # crude seasonality strength: variance explained by month-of-year means
        month = g["date"].dt.month.values
        month_means = pd.Series(vals).groupby(month).transform("mean")
        seasonality_strength = 1 - (np.var(vals - month_means) / (np.var(vals) + 1e-6))

        # simple linear trend slope (per day, normalized by mean)
        t = np.arange(len(vals))
        slope = np.polyfit(t, vals, 1)[0] / (mean_demand + 1e-6)

        prod_rows = df[df["product_id"] == pid]
        avg_price = prod_rows["price"].mean()
        promo_rate = prod_rows["promo_flag"].mean()
        category = prod_rows["category"].iloc[0]

        profiles.append(dict(
            product_id=pid, category=category,
            mean_demand=mean_demand, volatility_cv=cv,
            seasonality_strength=seasonality_strength, trend_slope=slope,
            avg_price=avg_price, promo_rate=promo_rate,
        ))

    return pd.DataFrame(profiles)


CLUSTER_FEATURES = ["mean_demand", "volatility_cv", "seasonality_strength", "trend_slope", "avg_price", "promo_rate"]


def name_cluster(row) -> str:
    """Heuristic label based on cluster-centroid characteristics."""
    if row["promo_rate"] > row["promo_rate_median"] * 1.3 and row["promo_rate"] > 0.15:
        return "Highly Promotional Products"
    if row["seasonality_strength"] > 0.35:
        return "Seasonal Products"
    if row["avg_price"] > row["avg_price_median"] * 1.3 and row["mean_demand"] < row["mean_demand_median"]:
        return "Luxury / Low-Velocity Products"
    if row["mean_demand"] > row["mean_demand_median"] * 2:
        return "Core High-Velocity Essentials"
    return "Fast-Moving Essentials"


def run_clustering(product_df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42):
    X = product_df[CLUSTER_FEATURES].fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(Xs)

    out = product_df.copy()
    out["cluster"] = labels

    centroids = out.groupby("cluster")[CLUSTER_FEATURES].mean().reset_index()
    centroids["promo_rate_median"] = out["promo_rate"].median()
    centroids["avg_price_median"] = out["avg_price"].median()
    centroids["mean_demand_median"] = out["mean_demand"].median()
    centroids["cluster_name"] = centroids.apply(name_cluster, axis=1)

    out = out.merge(centroids[["cluster", "cluster_name"]], on="cluster", how="left")
    return out, km, scaler, centroids[["cluster", "cluster_name"] + CLUSTER_FEATURES]
