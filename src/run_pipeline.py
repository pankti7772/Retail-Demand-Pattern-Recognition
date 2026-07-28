"""
run_pipeline.py
---------------
End-to-end MVP pipeline:
  raw data -> clean -> feature engineer -> cluster -> anomaly detect
  -> forecast (XGBoost/LightGBM) -> SHAP explain -> save artifacts

Run:
    python src/run_pipeline.py

Artifacts written to ../artifacts/ for the Streamlit dashboard to consume.
"""

import json
import time
import joblib
import pandas as pd
import mlflow

from features import build_feature_table, FEATURE_COLUMNS
from clustering import build_product_profile, run_clustering
from anomaly import run_anomaly_detection
from forecasting import prepare_model_matrix, train_and_evaluate
from explainability import compute_shap_values, global_feature_importance

ARTIFACT_DIR = "artifacts"


def main():
    t0 = time.time()
    print("1/6  Loading raw data...")
    raw = pd.read_csv("data/raw_sales.csv", parse_dates=["date"])

    print("2/6  Cleaning + feature engineering...")
    full_df, model_df, clean_meta = build_feature_table(raw)
    print(f"     rows raw={len(raw):,} -> cleaned={clean_meta['rows_after_clean']:,} "
          f"-> modelable={len(model_df):,} (dupes removed: {clean_meta['duplicates_removed']})")

    print("3/6  Clustering products (pattern recognition)...")
    product_profile = build_product_profile(full_df)
    clustered_products, kmeans_model, scaler, centroids = run_clustering(product_profile)
    print(centroids.round(2).to_string(index=False))

    print("4/6  Anomaly detection (Isolation Forest)...")
    model_df, iso_model = run_anomaly_detection(model_df)
    n_anom = int(model_df["is_anomaly"].sum())
    print(f"     flagged {n_anom:,} anomalous rows out of {len(model_df):,} "
          f"({n_anom/len(model_df)*100:.2f}%)")

    print("5/6  Training forecasting models (XGBoost + LightGBM)...")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("demandsense-ai-mvp")
    with mlflow.start_run(run_name="xgb_lgbm_forecast"):
        xgb_model, lgbm_model, feature_cols, metrics, test_out = train_and_evaluate(model_df)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_train_rows", len(model_df))
        mlflow.log_param("test_days", 60)
        for model_name, m in metrics.items():
            for metric_name, val in m.items():
                mlflow.log_metric(f"{model_name}_{metric_name}", val)
    print(json.dumps(metrics, indent=2))

    print("6/6  Computing SHAP explanations...")
    df_enc, _ = prepare_model_matrix(model_df)
    explainer, shap_values, X_sample = compute_shap_values(xgb_model, df_enc[feature_cols])
    importance = global_feature_importance(shap_values, feature_cols)
    print(importance.head(10).to_string(index=False))

    # ---- Save everything the dashboard needs ----
    joblib.dump(xgb_model, f"{ARTIFACT_DIR}/xgb_model.joblib")
    joblib.dump(lgbm_model, f"{ARTIFACT_DIR}/lgbm_model.joblib")
    joblib.dump(explainer, f"{ARTIFACT_DIR}/shap_explainer.joblib")
    joblib.dump(feature_cols, f"{ARTIFACT_DIR}/feature_cols.joblib")

    model_df.merge(
        clustered_products[["product_id", "cluster", "cluster_name"]], on="product_id", how="left"
    ).to_parquet(f"{ARTIFACT_DIR}/model_df.parquet", index=False)

    clustered_products.to_parquet(f"{ARTIFACT_DIR}/clustered_products.parquet", index=False)
    centroids.to_csv(f"{ARTIFACT_DIR}/cluster_centroids.csv", index=False)
    test_out.to_parquet(f"{ARTIFACT_DIR}/test_predictions.parquet", index=False)
    importance.to_csv(f"{ARTIFACT_DIR}/shap_global_importance.csv", index=False)

    with open(f"{ARTIFACT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{ARTIFACT_DIR}/clean_meta.json", "w") as f:
        json.dump(clean_meta, f, indent=2)

    print(f"\nDone in {time.time()-t0:.1f}s. Artifacts saved to ./{ARTIFACT_DIR}/")


if __name__ == "__main__":
    main()
