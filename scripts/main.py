"""
Credit Card Fraud Detection using Ensemble Learning
====================================================
DMDW Lab Project — Complete ML Pipeline

Usage:
    cd scripts
    python main.py

This script runs the full pipeline:
    1. Load data
    2. Exploratory Data Analysis
    3. Preprocessing (scaling, splitting, SMOTE)
    4. Model training (5 models)
    5. Evaluation (metrics, plots, comparisons)
    6. Prediction demo on sample transactions

All output plots are saved to the scripts/output/ folder.
"""

from data_loader import load_data
from eda import run_full_eda
from preprocessing import preprocess_pipeline
from models import build_models, train_models
from evaluation import (
    evaluate_all_models,
    print_best_models,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_feature_importance,
    compare_smote_vs_undersampling,
    plot_threshold_tuning,
    plot_metrics_summary,
    predict_transaction,
    print_final_summary,
)
from config import OUTPUT_DIR, setup_environment
import joblib


def main():
    # ── 0. Setup Environment ──────────────────────────────────────
    setup_environment()

    # ── 1. Load Data ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SECTION 1: LOADING DATA")
    print("=" * 70 + "\n")

    df = load_data()

    # ── 2. Exploratory Data Analysis ──────────────────────────────
    print("\n" + "=" * 70)
    print("  SECTION 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 70 + "\n")

    run_full_eda(df)

    # ── 3. Preprocessing ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SECTION 3: DATA PREPROCESSING")
    print("=" * 70 + "\n")

    data = preprocess_pipeline(df)

    # ── 4. Model Training ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SECTION 4: MODEL TRAINING")
    print("=" * 70 + "\n")

    models = build_models()
    print(f"Created {len(models)} models: {list(models.keys())}\n")
    trained_models = train_models(models, data["X_train_smote"], data["y_train_smote"])

    # ── 5. Evaluation ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SECTION 5: MODEL EVALUATION")
    print("=" * 70 + "\n")

    results_df, predictions, probabilities = evaluate_all_models(
        trained_models, data["X_test"], data["y_test"]
    )
    print_best_models(results_df)

    plot_confusion_matrices(predictions, data["y_test"])
    plot_roc_curves(probabilities, data["y_test"])
    plot_feature_importance(trained_models, data["X_train"].columns)

    # ── 6. SMOTE vs Undersampling ────────────────────────────────
    print("\n" + "=" * 70)
    print("  SECTION 6: SMOTE vs UNDERSAMPLING COMPARISON")
    print("=" * 70)

    compare_smote_vs_undersampling(
        trained_models, data["X_train_under"], data["y_train_under"],
        data["X_test"], data["y_test"],
    )

    # ── 7. Threshold Tuning ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SECTION 7: THRESHOLD TUNING")
    print("=" * 70 + "\n")

    plot_threshold_tuning(results_df, trained_models, probabilities, data["y_test"])

    # ── 8. Metrics Summary ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SECTION 8: METRICS VISUALIZATION")
    print("=" * 70 + "\n")

    plot_metrics_summary(results_df)

    # ── 9. Prediction Demo ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SECTION 9: PREDICTION DEMO")
    print("=" * 70 + "\n")

    best_model_name = results_df["F1-Score"].idxmax()
    best_model = trained_models[best_model_name]

    fraud_indices = data["y_test"][data["y_test"] == 1].index
    sample_fraud = data["X_test"].loc[fraud_indices[0]].values
    print("Testing with a KNOWN FRAUD transaction:\n")
    predict_transaction(best_model, sample_fraud, data["X_test"].columns, best_model_name)

    print()

    normal_indices = data["y_test"][data["y_test"] == 0].index
    sample_normal = data["X_test"].loc[normal_indices[0]].values
    print("Testing with a KNOWN NORMAL transaction:\n")
    predict_transaction(best_model, sample_normal, data["X_test"].columns, best_model_name)

    # ── 10. Final Summary ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SECTION 10: FINAL SUMMARY")
    print("=" * 70)

    print_final_summary(results_df)

    print(f"\n\nSaving best model ({best_model_name}) and scaler to {OUTPUT_DIR}...")
    joblib.dump(best_model, OUTPUT_DIR / "best_model.pkl")
    joblib.dump(data["scaler"], OUTPUT_DIR / "scaler.pkl")

    print(f"Pipeline complete! All visualizations and models saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
