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


@st.cache_data
def load_evaluation():
    """Evaluation artifacts written by the pipeline (read-only; nothing is recomputed here)."""
    try:
        comparison = pd.read_csv(f"{ARTIFACT_DIR}/model_comparison.csv")
        folds = pd.read_csv(f"{ARTIFACT_DIR}/rolling_cv_folds.csv", parse_dates=["train_end", "test_start", "test_end"])
        with open(f"{ARTIFACT_DIR}/baseline_metrics.json") as f:
            baselines = json.load(f)
        with open(f"{ARTIFACT_DIR}/rolling_cv_summary.json") as f:
            cv_summary = json.load(f)
    except FileNotFoundError:
        return None
    return comparison, folds, baselines, cv_summary


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
    "🧪 Evaluation & Model Comparison",
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

# ------------------------------------------ 7. Evaluation & Model Comparison --
with tabs[6]:
    st.subheader("Evaluation & Model Comparison")
    st.caption(
        "All metrics on this page are evaluation results on a **synthetic** retail dataset. "
        "They do not represent real-world retail performance."
    )
    eval_artifacts = load_evaluation()
    if eval_artifacts is None:
        st.info("Evaluation artifacts not found. Re-run `python src/run_pipeline.py` to generate them.")
    else:
        comparison, cv_folds, baseline_metrics, cv_summary = eval_artifacts
        ML_MODELS = ["xgboost", "lightgbm"]
        LABELS = {
            "xgboost": "XGBoost", "lightgbm": "LightGBM", "naive_lag1": "Naive (lag-1)",
            "seasonal_naive_lag7": "Seasonal-Naive (lag-7)", "moving_avg_7": "7-Day Moving Average",
        }
        best_baseline = min(baseline_metrics, key=lambda m: baseline_metrics[m]["MAE"])
        bb = baseline_metrics[best_baseline]

        cmp_df = comparison.set_index("model")
        cmp_df["RMSE_improvement_vs_best_baseline_pct"] = (bb["RMSE"] - cmp_df["RMSE"]) / bb["RMSE"] * 100
        order = ML_MODELS + [m for m in cmp_df.index if m not in ML_MODELS]
        cmp_df = cmp_df.loc[order]

        # ---- headline: best baseline and improvement over it
        h1, h2, h3 = st.columns(3)
        h1.metric("Best baseline (lowest MAE)", LABELS.get(best_baseline, best_baseline), f"MAE {bb['MAE']:.2f}", delta_color="off")
        for col, m in zip((h2, h3), ML_MODELS):
            col.metric(
                f"{LABELS[m]} MAE vs. best baseline",
                f"{cmp_df.loc[m, 'MAE']:.2f}",
                f"{cmp_df.loc[m, 'MAE_improvement_vs_best_baseline_pct']:.1f}% lower MAE",
            )

        # ---- comparison table
        st.markdown("**Holdout comparison (60-day time-based holdout, one-day-ahead forecasts)**")
        table = pd.DataFrame({
            "Model": [LABELS.get(m, m) for m in cmp_df.index],
            "Type": ["ML model" if m in ML_MODELS else ("Best baseline" if m == best_baseline else "Baseline") for m in cmp_df.index],
            "MAE": cmp_df["MAE"].values,
            "RMSE": cmp_df["RMSE"].values,
            "MAPE (%)": cmp_df["MAPE"].values,
            "MAE improvement vs. best baseline (%)": cmp_df["MAE_improvement_vs_best_baseline_pct"].values,
            "RMSE improvement vs. best baseline (%)": cmp_df["RMSE_improvement_vs_best_baseline_pct"].values,
        })

        def _highlight(row):
            color = {"ML model": "rgba(46,160,67,0.18)", "Best baseline": "rgba(210,153,34,0.22)"}.get(row["Type"], "")
            return [f"background-color: {color}" if color else ""] * len(row)

        st.dataframe(
            table.style.apply(_highlight, axis=1).format({c: "{:.2f}" for c in table.columns if c not in ("Model", "Type")}),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Green = ML models, amber = best baseline. MAE improvement is read from `model_comparison.csv`; "
            "RMSE improvement is the relative difference between the table's RMSE values and the best baseline's RMSE."
        )

        # ---- MAE / RMSE chart
        chart_df = table.melt(id_vars="Model", value_vars=["MAE", "RMSE"], var_name="Metric", value_name="Value")
        fig = px.bar(chart_df, x="Model", y="Value", color="Metric", barmode="group", text_auto=".1f",
                     category_orders={"Model": table["Model"].tolist()},
                     labels={"Value": "Error (units)"})
        fig.update_layout(height=380, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

        # ---- rolling-origin validation
        st.markdown("**3-fold expanding-window validation (mean ± std across folds)**")
        rows = []
        for m in ML_MODELS:
            row = {"Model": LABELS[m]}
            for metric, label in (("MAE", "MAE"), ("RMSE", "RMSE"), ("MAPE", "MAPE (%)")):
                stat = cv_summary[f"{m}_{metric}"]
                row[label] = f"{stat['mean']:.2f} ± {stat['std']:.2f}"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("**Validation windows (time-based, no random split)**")
        data_start = model_df["date"].min()
        win = []
        for _, r in cv_folds.iterrows():
            win.append({"Fold": f"Fold {int(r['fold'])}", "Phase": "Train (expanding)", "Start": data_start, "End": r["train_end"] + pd.Timedelta(days=1)})
            win.append({"Fold": f"Fold {int(r['fold'])}", "Phase": "Test", "Start": r["test_start"], "End": r["test_end"] + pd.Timedelta(days=1)})
        fig_w = px.timeline(pd.DataFrame(win), x_start="Start", x_end="End", y="Fold", color="Phase")
        fig_w.update_yaxes(autorange="reversed", title=None)
        fig_w.update_layout(height=260, legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_w, use_container_width=True)

        fold_table = pd.DataFrame({
            "Fold": cv_folds["fold"],
            "Train end": cv_folds["train_end"].dt.date,
            "Test window": cv_folds["test_start"].dt.strftime("%Y-%m-%d") + " → " + cv_folds["test_end"].dt.strftime("%Y-%m-%d"),
            "Train rows": cv_folds["n_train"],
            "Test rows": cv_folds["n_test"],
            "XGB MAE": cv_folds["xgboost_MAE"], "XGB RMSE": cv_folds["xgboost_RMSE"], "XGB MAPE (%)": cv_folds["xgboost_MAPE"],
            "LGBM MAE": cv_folds["lightgbm_MAE"], "LGBM RMSE": cv_folds["lightgbm_RMSE"], "LGBM MAPE (%)": cv_folds["lightgbm_MAPE"],
        })
        num_cols = [c for c in fold_table.columns if c.startswith(("XGB", "LGBM"))]
        st.dataframe(fold_table.style.format({c: "{:.2f}" for c in num_cols}), use_container_width=True, hide_index=True)
        st.caption("The last fold uses the same split as the main 60-day holdout.")

        # ---- methodology
        st.markdown("**Methodology**")
        st.markdown(
            "- Synthetic retail dataset (M5-style), not real retail data\n"
            "- 60-day time-based holdout\n"
            "- One-day-ahead forecasting\n"
            "- Baseline benchmarking: naive (lag-1), seasonal-naive (lag-7), 7-day moving average, scored on the same holdout rows\n"
            "- 3-fold expanding-window (rolling-origin) validation, 60-day test window per fold"
        )