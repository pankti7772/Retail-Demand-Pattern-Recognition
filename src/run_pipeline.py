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
from forecasting import (prepare_model_matrix, train_and_evaluate, baseline_metrics,
                         rolling_origin_cv, get_cutoff)
from explainability import compute_shap_values, global_feature_importance

ARTIFACT_DIR = "artifacts"
TEST_DAYS = 60
N_CV_FOLDS = 3


def _log_artifact_safe(path):
    """mlflow.db's experiment artifact_location may point to a path from another
    machine; metrics are still logged, so a failed file upload is non-fatal."""
    try:
        mlflow.log_artifact(path)
    except OSError as e:
        print(f"     [warn] MLflow artifact upload skipped for {path}: {e}")


def main():
    t0 = time.time()
    print("1/6  Loading raw data...")
    raw = pd.read_csv("data/raw_sales.csv", parse_dates=["date"])
    # Single holdout cutoff shared by every train-only fit (capping, Isolation Forest, models).
    cutoff = get_cutoff(raw["date"].max(), TEST_DAYS)
    print(f"     holdout: train <= {cutoff.date()} < test ({TEST_DAYS} days)")

    print("2/6  Cleaning + feature engineering...")
    full_df, model_df, clean_meta = build_feature_table(raw, cap_cutoff_date=cutoff)
    print(f"     rows raw={len(raw):,} -> cleaned={clean_meta['rows_after_clean']:,} "
          f"-> modelable={len(model_df):,} (dupes removed: {clean_meta['duplicates_removed']})")

    print("3/6  Clustering products (pattern recognition)...")
    product_profile = build_product_profile(full_df)
    clustered_products, kmeans_model, scaler, centroids = run_clustering(product_profile)
    print(centroids.round(2).to_string(index=False))

    print("4/6  Anomaly detection (Isolation Forest, fitted on training period)...")
    model_df, iso_model = run_anomaly_detection(model_df, train_cutoff=cutoff)
    n_anom = int(model_df["is_anomaly"].sum())
    print(f"     flagged {n_anom:,} anomalous rows out of {len(model_df):,} "
          f"({n_anom/len(model_df)*100:.2f}%)")

    print("5/6  Training forecasting models (XGBoost + LightGBM) + baselines + rolling CV...")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("demandsense-ai-mvp")
    with mlflow.start_run(run_name="xgb_lgbm_forecast"):
        xgb_model, lgbm_model, feature_cols, metrics, test_out, split_info = train_and_evaluate(
            model_df, test_days=TEST_DAYS)
        base_metrics, base_preds = baseline_metrics(model_df, test_days=TEST_DAYS)
        assert len(base_preds) == split_info["n_test_rows"], "baseline/model holdout mismatch"
        test_out = test_out.join(base_preds)

        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_train_rows", split_info["n_train_rows"])
        mlflow.log_param("n_test_rows", split_info["n_test_rows"])
        mlflow.log_param("n_modelable_rows", len(model_df))
        mlflow.log_param("test_days", TEST_DAYS)
        mlflow.log_param("train_cutoff", str(cutoff.date()))
        mlflow.log_param("dataset", "synthetic (src/generate_data.py, seed 42)")
        for model_name, m in {**metrics, **base_metrics}.items():
            for metric_name, val in m.items():
                mlflow.log_metric(f"{model_name}_{metric_name}", val)

        comparison = pd.DataFrame({**metrics, **base_metrics}).T
        best_base_mae = comparison.loc[list(base_metrics), "MAE"].min()
        comparison["MAE_improvement_vs_best_baseline_pct"] = (1 - comparison["MAE"] / best_base_mae) * 100
        comparison.index.name = "model"
        for model_name in metrics:
            mlflow.log_metric(f"{model_name}_MAE_improvement_vs_best_baseline_pct",
                              float(comparison.loc[model_name, "MAE_improvement_vs_best_baseline_pct"]))
        comparison.to_csv(f"{ARTIFACT_DIR}/model_comparison.csv")
        _log_artifact_safe(f"{ARTIFACT_DIR}/model_comparison.csv")

        cv_folds, cv_summary = rolling_origin_cv(model_df, n_folds=N_CV_FOLDS, test_days=TEST_DAYS)
        mlflow.log_param("cv_folds", N_CV_FOLDS)
        for key, s in cv_summary.items():
            mlflow.log_metric(f"cv_{key}_mean", s["mean"])
            mlflow.log_metric(f"cv_{key}_std", s["std"])
        cv_folds.to_csv(f"{ARTIFACT_DIR}/rolling_cv_folds.csv", index=False)
        with open(f"{ARTIFACT_DIR}/rolling_cv_summary.json", "w") as f:
            json.dump(cv_summary, f, indent=2)
        _log_artifact_safe(f"{ARTIFACT_DIR}/rolling_cv_folds.csv")

    print(comparison.round(2).to_string())
    print(cv_folds.round(2).to_string(index=False))

    print("6/6  Computing SHAP explanations (test window)...")
    df_enc, _ = prepare_model_matrix(model_df)
    test_mask = (model_df["date"] > cutoff).values
    explainer, shap_values, X_sample = compute_shap_values(xgb_model, df_enc.loc[test_mask, feature_cols])
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
    with open(f"{ARTIFACT_DIR}/baseline_metrics.json", "w") as f:
        json.dump(base_metrics, f, indent=2)
    with open(f"{ARTIFACT_DIR}/clean_meta.json", "w") as f:
        json.dump(clean_meta, f, indent=2)

    print(f"\nDone in {time.time()-t0:.1f}s. Artifacts saved to ./{ARTIFACT_DIR}/")


if __name__ == "__main__":
    main()
