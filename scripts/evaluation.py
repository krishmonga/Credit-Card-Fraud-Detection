import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.ensemble import RandomForestClassifier
from config import RANDOM_STATE, BAR_COLORS, ROC_COLORS, ROC_LINE_STYLES, OUTPUT_DIR


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a single model and return metrics dictionary."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }
    return metrics, y_pred, y_prob


def evaluate_all_models(trained_models, X_test, y_test):
    """Evaluate all trained models and return results."""
    all_results = []
    predictions = {}
    probabilities = {}

    print("=" * 90)
    print(f"{'MODEL EVALUATION RESULTS':^90}")
    print("=" * 90)

    for name, model in trained_models.items():
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test, name)
        all_results.append(metrics)
        predictions[name] = y_pred
        probabilities[name] = y_prob

        print(f"\n--- {name} ---")
        print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall:    {metrics['Recall']:.4f}  <-- catches {metrics['Recall']*100:.1f}% of frauds")
        print(f"  F1-Score:  {metrics['F1-Score']:.4f}")
        print(f"  ROC-AUC:   {metrics['ROC-AUC']:.4f}")

    print("\n" + "=" * 90)

    results_df = pd.DataFrame(all_results).set_index("Model")
    return results_df, predictions, probabilities


def print_best_models(results_df):
    """Print the best model for each metric."""
    print("\nBest model by each metric:")
    for metric in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
        best = results_df[metric].idxmax()
        print(f"  {metric:<12}: {best} ({results_df.loc[best, metric]:.4f})")


def plot_confusion_matrices(predictions, y_test):
    """Plot confusion matrices for all models."""
    fig, axes = plt.subplots(1, len(predictions), figsize=(5 * len(predictions), 5))

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"],
            linewidths=1, linecolor="black",
        )
        ax.set_title(name, fontweight="bold", fontsize=10)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    plt.suptitle("Confusion Matrices - All Models", fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices.png"), bbox_inches="tight")
    plt.close()


def plot_roc_curves(probabilities, y_test):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 7))

    for (name, y_prob), ls, color in zip(probabilities.items(), ROC_LINE_STYLES, ROC_COLORS):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, ls, color=color, linewidth=2, label=f"{name} (AUC={auc_val:.4f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC=0.5)")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate (Recall)", fontsize=12)
    plt.title("ROC Curve Comparison - All Models", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), bbox_inches="tight")
    plt.close()


def plot_feature_importance(trained_models, feature_names):
    """Plot feature importance from tree-based models."""
    tree_models = {
        name: trained_models[name]
        for name in ["Decision Tree", "Random Forest", "Gradient Boosting"]
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    for ax, (name, model) in zip(axes, tree_models.items()):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[-15:]
        ax.barh(
            range(len(sorted_idx)),
            importances[sorted_idx],
            color="#3498db", edgecolor="black", linewidth=0.3,
        )
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels(feature_names[sorted_idx])
        ax.set_title(f"{name}\nTop 15 Features", fontweight="bold")
        ax.set_xlabel("Importance")

    plt.suptitle("Feature Importance - Tree-Based Models", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), bbox_inches="tight")
    plt.close()

    print("\nTop 10 Most Important Features (Random Forest):")
    rf_importances = pd.Series(
        trained_models["Random Forest"].feature_importances_, index=feature_names
    ).sort_values(ascending=False)
    for i, (feat, imp) in enumerate(rf_importances.head(10).items(), 1):
        print(f"  {i:>2}. {feat:<18} {imp:.4f}")


def compare_smote_vs_undersampling(trained_models, X_train_under, y_train_under, X_test, y_test):
    """Compare Random Forest performance with SMOTE vs Undersampling."""
    rf_under = RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_split=5,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_under.fit(X_train_under, y_train_under)

    metrics_smote, _, _ = evaluate_model(trained_models["Random Forest"], X_test, y_test, "RF + SMOTE")
    metrics_under, _, _ = evaluate_model(rf_under, X_test, y_test, "RF + Undersampling")

    comparison = pd.DataFrame([metrics_smote, metrics_under]).set_index("Model")
    print("\nRandom Forest: SMOTE vs Undersampling")
    print("=" * 60)
    print(comparison.to_string(float_format="{:.4f}".format))


def plot_threshold_tuning(results_df, trained_models, probabilities, y_test):
    """Perform and visualize threshold tuning on the best model."""
    best_model_name = results_df["F1-Score"].idxmax()
    y_prob_best = probabilities[best_model_name]

    print(f"Performing threshold tuning on: {best_model_name}\n")

    precisions_pr, recalls_pr, _ = precision_recall_curve(y_test, y_prob_best)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(recalls_pr, precisions_pr, color="#e74c3c", linewidth=2)
    axes[0].set_xlabel("Recall", fontsize=12)
    axes[0].set_ylabel("Precision", fontsize=12)
    axes[0].set_title(f"Precision-Recall Curve\n({best_model_name})", fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].fill_between(recalls_pr, precisions_pr, alpha=0.15, color="#e74c3c")

    test_thresholds = np.arange(0.1, 0.9, 0.05)
    t_precisions, t_recalls, t_f1s = [], [], []

    for t in test_thresholds:
        y_pred_t = (y_prob_best >= t).astype(int)
        t_precisions.append(precision_score(y_test, y_pred_t, zero_division=0))
        t_recalls.append(recall_score(y_test, y_pred_t, zero_division=0))
        t_f1s.append(f1_score(y_test, y_pred_t, zero_division=0))

    axes[1].plot(test_thresholds, t_precisions, "b-", linewidth=2, label="Precision")
    axes[1].plot(test_thresholds, t_recalls, "r-", linewidth=2, label="Recall")
    axes[1].plot(test_thresholds, t_f1s, "g--", linewidth=2, label="F1-Score")
    axes[1].axvline(x=0.5, color="gray", linestyle=":", label="Default Threshold (0.5)")

    optimal_idx = np.argmax(t_f1s)
    optimal_threshold = test_thresholds[optimal_idx]
    axes[1].axvline(x=optimal_threshold, color="green", linestyle="--", alpha=0.7,
                    label=f"Optimal Threshold ({optimal_threshold:.2f})")

    axes[1].set_xlabel("Threshold", fontsize=12)
    axes[1].set_ylabel("Score", fontsize=12)
    axes[1].set_title("Threshold Tuning Analysis", fontweight="bold")
    axes[1].legend(loc="center left", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "threshold_tuning.png"), bbox_inches="tight")
    plt.close()

    print(f"\nOptimal threshold (maximizes F1): {optimal_threshold:.2f}")
    print(f"  Precision: {t_precisions[optimal_idx]:.4f}")
    print(f"  Recall:    {t_recalls[optimal_idx]:.4f}")
    print(f"  F1-Score:  {t_f1s[optimal_idx]:.4f}")


def plot_metrics_summary(results_df):
    """Plot bar charts comparing all models across key metrics."""
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    model_names = results_df.index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics_to_plot):
        values = results_df[metric].values
        bars = ax.bar(range(len(model_names)), values, color=BAR_COLORS, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
        ax.set_title(metric, fontweight="bold", fontsize=13)
        ax.set_ylim(0, 1.15)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", fontweight="bold", fontsize=9)
        best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(3)

    axes[-1].axis("off")
    plt.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), bbox_inches="tight")
    plt.close()


def predict_transaction(model, sample, feature_names, model_name="Model"):
    """Predict whether a single transaction is fraud or normal."""
    sample_df = pd.DataFrame([sample], columns=feature_names)
    prediction = model.predict(sample_df)[0]
    probability = model.predict_proba(sample_df)[0]

    print("=" * 50)
    print(f"  FRAUD DETECTION PREDICTION ({model_name})")
    print("=" * 50)
    result = ">>> FRAUD DETECTED <<<" if prediction == 1 else "Normal Transaction"
    print(f"  Prediction:  {result}")
    print(f"  Confidence:")
    print(f"    Normal: {probability[0]*100:.2f}%")
    print(f"    Fraud:  {probability[1]*100:.2f}%")
    print("=" * 50)
    return prediction, probability


def print_final_summary(results_df):
    """Print the final model comparison summary."""
    print("\n" + "=" * 80)
    print(f"{'FINAL MODEL COMPARISON SUMMARY':^80}")
    print("=" * 80)

    results_sorted = results_df.sort_values("F1-Score", ascending=False)

    print(f"\n{'Rank':<6} {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'ROC-AUC':>10}")
    print("-" * 85)

    for rank, (name, row) in enumerate(results_sorted.iterrows(), 1):
        marker = " <-- BEST" if rank == 1 else ""
        print(f"{rank:<6} {name:<25} {row['Accuracy']:>10.4f} {row['Precision']:>10.4f} {row['Recall']:>10.4f} "
              f"{row['F1-Score']:>10.4f} {row['ROC-AUC']:>10.4f}{marker}")

    best = results_sorted.index[0]
    print(f"\n{'=' * 85}")
    print(f"\nBEST OVERALL MODEL: {best}")
    print(f"  - Accuracy:  {results_sorted.loc[best, 'Accuracy']:.4f}")
    print(f"  - F1-Score:  {results_sorted.loc[best, 'F1-Score']:.4f}")
    print(f"  - Recall:    {results_sorted.loc[best, 'Recall']:.4f} "
          f"(catches {results_sorted.loc[best, 'Recall']*100:.1f}% of frauds)")
    print(f"  - Precision: {results_sorted.loc[best, 'Precision']:.4f}")
    print(f"  - ROC-AUC:   {results_sorted.loc[best, 'ROC-AUC']:.4f}")

    return best
