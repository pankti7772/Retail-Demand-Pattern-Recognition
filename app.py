"""
app.py -- DemandSense AI Streamlit Dashboard
Step 8 of the pipeline: an interactive view over everything the pipeline
produced (forecast, demand trend, clusters, SHAP explanations, anomalies,
region-wise sales).

Run:
    streamlit run app.py
(run src/run_pipeline.py at least once first to generate ./artifacts/)
"""

import json

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="DemandSense AI", layout="wide", page_icon="📦")

ARTIFACT_DIR = "artifacts"


# ---------------------------------------------------------------- loaders --
@st.cache_data
def load_data():
    model_df = pd.read_parquet(f"{ARTIFACT_DIR}/model_df.parquet")
    clustered_products = pd.read_parquet(f"{ARTIFACT_DIR}/clustered_products.parquet")
    test_pred = pd.read_parquet(f"{ARTIFACT_DIR}/test_predictions.parquet")
    importance = pd.read_csv(f"{ARTIFACT_DIR}/shap_global_importance.csv")
    centroids = pd.read_csv(f"{ARTIFACT_DIR}/cluster_centroids.csv")
    with open(f"{ARTIFACT_DIR}/metrics.json") as f:
        metrics = json.load(f)
    with open(f"{ARTIFACT_DIR}/clean_meta.json") as f:
        clean_meta = json.load(f)
    return model_df, clustered_products, test_pred, importance, centroids, metrics, clean_meta


@st.cache_resource
def load_models():
    feature_cols = joblib.load(f"{ARTIFACT_DIR}/feature_cols.joblib")
    xgb_model = joblib.load(f"{ARTIFACT_DIR}/xgb_model.joblib")
    explainer = joblib.load(f"{ARTIFACT_DIR}/shap_explainer.joblib")
    return feature_cols, xgb_model, explainer


try:
    model_df, clustered_products, test_pred, importance, centroids, metrics, clean_meta = load_data()
    feature_cols, xgb_model, explainer = load_models()
except FileNotFoundError:
    st.error(
        "No artifacts found. Run the pipeline first:\n\n"
        "```\npython src/generate_data.py\npython src/run_pipeline.py\n```"
    )
    st.stop()


# ------------------------------------------------------------------ sidebar --
st.sidebar.title("📦 DemandSense AI")
st.sidebar.caption("Intelligent Retail Demand Pattern Recognition & Sales Forecasting")

regions = ["All"] + sorted(model_df["region"].unique().tolist())
sel_region = st.sidebar.selectbox("Region", regions)

df_scope = model_df if sel_region == "All" else model_df[model_df["region"] == sel_region]

categories = ["All"] + sorted(df_scope["category"].unique().tolist())
sel_category = st.sidebar.selectbox("Category", categories)
if sel_category != "All":
    df_scope = df_scope[df_scope["category"] == sel_category]

products = sorted(df_scope["product_id"].unique().tolist())
sel_product = st.sidebar.selectbox("Product", products, index=0 if products else None)

stores = sorted(df_scope[df_scope["product_id"] == sel_product]["store_id"].unique().tolist())
sel_store = st.sidebar.selectbox("Store", stores)

st.sidebar.divider()
st.sidebar.metric("XGBoost Test MAPE", f"{metrics['xgboost']['MAPE']:.1f}%")
st.sidebar.metric("XGBoost Test RMSE", f"{metrics['xgboost']['RMSE']:.1f}")
st.sidebar.caption(
    f"Trained on synthetic M5-style data — {clean_meta['rows_after_clean']:,} rows "
    f"({clean_meta['duplicates_removed']} dupes removed during cleaning)."
)

series = df_scope[(df_scope["product_id"] == sel_product) & (df_scope["store_id"] == sel_store)].sort_values("date")

# ----------------------------------------------------------------- header --
st.title("Intelligent Retail Demand Pattern Recognition & Sales Forecasting")
st.caption(f"Viewing **{sel_product}** at **{sel_store}** ({series['category'].iloc[0] if len(series) else '—'})")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Avg Daily Units", f"{series['units_sold_capped'].mean():.0f}" if len(series) else "—")
k2.metric("Promo Days (last 30)", f"{int(series['promo_flag'].tail(30).sum())}" if len(series) else "—")
k3.metric("Anomalies Detected", f"{int(series['is_anomaly'].sum())}" if len(series) else "—")
cluster_name = series["cluster_name"].iloc[0] if len(series) else "—"
k4.metric("Cluster", cluster_name)

tabs = st.tabs([
    "📈 Sales Forecast", "📊 Demand Trend", "🧩 Product Clusters",
    "🔍 SHAP Explanation", "🚨 Anomaly Alerts", "🗺️ Region-wise Sales",
])

# ------------------------------------------------------------- 1. Forecast --
with tabs[0]:
    st.subheader("Forecast vs. Actual (held-out test window)")
    pred_series = test_pred[(test_pred["product_id"] == sel_product) & (test_pred["store_id"] == sel_store)].sort_values("date")
    if len(pred_series):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pred_series["date"], y=pred_series["units_sold_capped"], name="Actual", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=pred_series["date"], y=pred_series["xgb_pred"], name="XGBoost Forecast", mode="lines"))
        fig.add_trace(go.Scatter(x=pred_series["date"], y=pred_series["lgbm_pred"], name="LightGBM Forecast", mode="lines"))
        fig.update_layout(height=420, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Model performance (all products, test window)**")
            st.dataframe(pd.DataFrame(metrics).T.style.format("{:.2f}"), use_container_width=True)
        with c2:
            err = (pred_series["xgb_pred"] - pred_series["units_sold_capped"]).abs()
            st.markdown("**This product's forecast error (XGBoost)**")
            st.metric("Mean Absolute Error", f"{err.mean():.1f} units")
    else:
        st.info("This product/store falls outside the held-out test window — pick another combination.")

# --------------------------------------------------------- 2. Demand Trend --
with tabs[1]:
    st.subheader("Historical Demand Trend")
    if len(series):
        fig = px.line(series, x="date", y="units_sold_capped", title=None, labels={"units_sold_capped": "Units Sold"})
        fig.add_scatter(x=series[series["promo_flag"] == 1]["date"], y=series[series["promo_flag"] == 1]["units_sold_capped"],
                         mode="markers", marker=dict(color="orange", size=6), name="Promo Day")
        fig.add_scatter(x=series[series["is_holiday"] == 1]["date"], y=series[series["is_holiday"] == 1]["units_sold_capped"],
                         mode="markers", marker=dict(color="red", size=6, symbol="diamond"), name="Holiday")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("30-day Rolling Mean", f"{series['roll_mean_30'].iloc[-1]:.1f}" if series['roll_mean_30'].notna().any() else "—")
        c2.metric("30-day Volatility (std)", f"{series['roll_std_30'].iloc[-1]:.1f}" if series['roll_std_30'].notna().any() else "—")
        c3.metric("Promo Frequency (30d)", f"{series['promo_frequency_30'].iloc[-1]*100:.0f}%" if series['promo_frequency_30'].notna().any() else "—")

        st.markdown("**Weekday seasonality (this product/store)**")
        wk = series.groupby("day_of_week")["units_sold_capped"].mean().reindex(range(7))
        wk.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        st.bar_chart(wk)
    else:
        st.info("No data for this selection.")

# ------------------------------------------------------- 3. Product Clusters --
with tabs[2]:
    st.subheader("Product Pattern Clusters (K-Means)")
    st.caption("Products grouped by demand level, volatility, seasonality strength, trend, price and promo behaviour.")
    fig = px.scatter(
        clustered_products, x="mean_demand", y="volatility_cv", color="cluster_name", size="avg_price",
        hover_data=["product_id", "category", "seasonality_strength", "promo_rate"],
        labels={"mean_demand": "Mean Daily Demand", "volatility_cv": "Volatility (CV)"},
    )
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Cluster centroids**")
    st.dataframe(centroids.round(2), use_container_width=True)

    st.markdown("**Products in the same cluster as your current selection**")
    this_cluster = clustered_products[clustered_products["product_id"] == sel_product]["cluster_name"]
    if len(this_cluster):
        st.dataframe(
            clustered_products[clustered_products["cluster_name"] == this_cluster.iloc[0]]
            .sort_values("mean_demand", ascending=False).head(10),
            use_container_width=True,
        )

# ----------------------------------------------------- 4. SHAP Explanation --
with tabs[3]:
    st.subheader("Why the model predicts what it predicts")
    st.markdown("**Global feature importance (mean |SHAP value|, sample of test rows)**")
    fig = px.bar(importance.head(12).sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature", orientation="h")
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Explain one prediction**")
    if len(series):
        row = df_scope[(df_scope["product_id"] == sel_product) & (df_scope["store_id"] == sel_store)].sort_values("date").tail(1)
        row_enc = pd.get_dummies(row, columns=["category", "region", "store_id"])
        for c in feature_cols:
            if c not in row_enc.columns:
                row_enc[c] = 0
        row_X = row_enc[feature_cols]
        sv = explainer(row_X)
        contrib = pd.DataFrame({
            "feature": feature_cols,
            "value": row_X.iloc[0].values,
            "shap_contribution": sv.values[0],
        })
        contrib["direction"] = contrib["shap_contribution"].apply(lambda x: "⬆ increases demand" if x >= 0 else "⬇ decreases demand")
        contrib = contrib.reindex(contrib["shap_contribution"].abs().sort_values(ascending=False).index).head(6)
        st.dataframe(contrib, use_container_width=True)
        st.caption(f"Base value: {sv.base_values[0]:.1f} units → prediction for {sel_product}'s most recent day.")

# ------------------------------------------------------- 5. Anomaly Alerts --
with tabs[4]:
    st.subheader("Anomaly Alerts (Isolation Forest)")
    anomalies = df_scope[df_scope["is_anomaly"]].sort_values("anomaly_score").head(200)
    st.caption(f"{len(df_scope[df_scope['is_anomaly']]):,} anomalies detected in current scope "
               f"({len(df_scope):,} rows).")
    if len(anomalies):
        st.dataframe(
            anomalies[["date", "product_id", "store_id", "category", "units_sold_capped",
                       "roll_mean_7", "anomaly_type", "anomaly_score"]],
            use_container_width=True, height=350,
        )
        fig = px.histogram(df_scope[df_scope["is_anomaly"]], x="anomaly_type", color="category",
                            title="Anomaly type breakdown")
        st.plotly_chart(fig, use_container_width=True)

        if len(series):
            st.markdown(f"**Anomalies for {sel_product} @ {sel_store}**")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=series["date"], y=series["units_sold_capped"], mode="lines", name="Units Sold"))
            anom_pts = series[series["is_anomaly"]]
            fig2.add_trace(go.Scatter(x=anom_pts["date"], y=anom_pts["units_sold_capped"], mode="markers",
                                       marker=dict(color="red", size=9, symbol="x"), name="Anomaly"))
            fig2.update_layout(height=380)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No anomalies in the current filter scope.")

# ---------------------------------------------------- 6. Region-wise Sales --
with tabs[5]:
    st.subheader("Region-wise Sales")
    region_sales = model_df.groupby(["region", "date"], as_index=False)["units_sold_capped"].sum()
    fig = px.line(region_sales, x="date", y="units_sold_capped", color="region", labels={"units_sold_capped": "Total Units"})
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        region_totals = model_df.groupby("region")["units_sold_capped"].sum().reset_index()
        fig2 = px.pie(region_totals, names="region", values="units_sold_capped", title="Share of total units by region")
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        region_cat = model_df.groupby(["region", "category"])["units_sold_capped"].sum().reset_index()
        fig3 = px.bar(region_cat, x="region", y="units_sold_capped", color="category", title="Category mix by region", barmode="stack")
        st.plotly_chart(fig3, use_container_width=True)
