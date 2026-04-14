import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import COLORS, LABELS, OUTPUT_DIR


def plot_class_distribution(class_counts, class_pct):
    """Plot bar chart and pie chart of class distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(LABELS, class_counts.values, color=COLORS, edgecolor="black", linewidth=0.8)
    for i, (count, pct) in enumerate(zip(class_counts.values, class_pct.values)):
        axes[0].text(i, count + 3000, f"{count:,}\n({pct:.2f}%)", ha="center", fontweight="bold")
    axes[0].set_title("Class Distribution (Count)", fontweight="bold")
    axes[0].set_ylabel("Number of Transactions")

    axes[1].pie(
        class_counts.values,
        labels=LABELS,
        colors=COLORS,
        autopct="%1.3f%%",
        startangle=90,
        explode=(0, 0.15),
        shadow=True,
        textprops={"fontweight": "bold"},
    )
    axes[1].set_title("Class Distribution (Proportion)", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), bbox_inches="tight")
    plt.close()

    print("\nKEY INSIGHT: The dataset is EXTREMELY imbalanced.")
    print("Only 0.17% of transactions are fraudulent.")


def plot_amount_analysis(normal_df, fraud_df):
    """Plot transaction amount distribution and box plots."""
    print("=" * 55)
    print("  Transaction Amount Statistics")
    print("=" * 55)
    for label, subset in [("Normal", normal_df), ("Fraud", fraud_df)]:
        print(f"\n  {label} Transactions:")
        print(f"    Count : {len(subset):>10,}")
        print(f"    Mean  : ${subset['Amount'].mean():>10,.2f}")
        print(f"    Median: ${subset['Amount'].median():>10,.2f}")
        print(f"    Max   : ${subset['Amount'].max():>10,.2f}")
        print(f"    Std   : ${subset['Amount'].std():>10,.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(normal_df["Amount"], bins=80, color="#2ecc71", alpha=0.7, label="Normal", edgecolor="black", linewidth=0.3)
    axes[0].hist(fraud_df["Amount"], bins=80, color="#e74c3c", alpha=0.8, label="Fraud", edgecolor="black", linewidth=0.3)
    axes[0].set_title("Transaction Amount Distribution", fontweight="bold")
    axes[0].set_xlabel("Amount ($)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_xlim(0, 2000)
    axes[0].legend()

    data_box = [normal_df["Amount"].values, fraud_df["Amount"].values]
    bp = axes[1].boxplot(data_box, labels=["Normal", "Fraud"], patch_artist=True, notch=True)
    bp["boxes"][0].set_facecolor("#2ecc71")
    bp["boxes"][1].set_facecolor("#e74c3c")
    axes[1].set_title("Amount: Normal vs Fraud (Box Plot)", fontweight="bold")
    axes[1].set_ylabel("Amount ($)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "amount_analysis.png"), bbox_inches="tight")
    plt.close()

    print("\nINSIGHT: Fraudulent transactions tend to have LOWER amounts.")


def plot_time_analysis(normal_df, fraud_df):
    """Plot transaction time distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(normal_df["Time"] / 3600, bins=48, color="#2ecc71", alpha=0.7, label="Normal", edgecolor="black", linewidth=0.3)
    axes[0].set_title("Normal Transactions Over Time", fontweight="bold")
    axes[0].set_xlabel("Time (hours)")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(fraud_df["Time"] / 3600, bins=48, color="#e74c3c", alpha=0.8, label="Fraud", edgecolor="black", linewidth=0.3)
    axes[1].set_title("Fraudulent Transactions Over Time", fontweight="bold")
    axes[1].set_xlabel("Time (hours)")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "time_analysis.png"), bbox_inches="tight")
    plt.close()

    print("INSIGHT: Normal transactions show a clear day/night pattern.")
    print("Fraud transactions are more uniformly distributed.")


def plot_correlation_analysis(df):
    """Plot feature correlations with the target variable."""
    corr_with_class = df.corr()["Class"].drop("Class").sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors_bar = ["#e74c3c" if v < 0 else "#2ecc71" for v in corr_with_class.values]
    axes[0].barh(corr_with_class.index, corr_with_class.values, color=colors_bar, edgecolor="black", linewidth=0.3)
    axes[0].set_title("Feature Correlation with Fraud (Class)", fontweight="bold")
    axes[0].set_xlabel("Pearson Correlation")
    axes[0].axvline(x=0, color="black", linewidth=0.8)

    top_features = list(corr_with_class.head(5).index) + list(corr_with_class.tail(5).index) + ["Class"]
    sns.heatmap(
        df[top_features].corr(),
        annot=True, fmt=".2f", cmap="RdYlGn", center=0,
        ax=axes[1], square=True, linewidths=0.5,
    )
    axes[1].set_title("Correlation Heatmap (Top Features)", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_analysis.png"), bbox_inches="tight")
    plt.close()

    print("\nTop negatively correlated features:")
    for feat, val in corr_with_class.head(5).items():
        print(f"  {feat}: {val:.4f}")
    print("\nTop positively correlated features:")
    for feat, val in corr_with_class.tail(5).items():
        print(f"  {feat}: {val:.4f}")


def plot_feature_comparison(normal_df, fraud_df):
    """Plot distribution comparison of important features."""
    important_features = ["V17", "V14", "V12", "V10", "V11", "V4", "V2"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for i, feat in enumerate(important_features):
        axes[i].hist(normal_df[feat], bins=50, alpha=0.6, color="#2ecc71", label="Normal", density=True)
        axes[i].hist(fraud_df[feat], bins=50, alpha=0.7, color="#e74c3c", label="Fraud", density=True)
        axes[i].set_title(f"{feat} Distribution", fontweight="bold")
        axes[i].legend(fontsize=8)

    axes[-1].axis("off")
    axes[-1].text(
        0.5, 0.5,
        "Fraud transactions\nshow clearly different\ndistributions for\nthese key features",
        ha="center", va="center", fontsize=13, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f0f0f0", edgecolor="gray"),
    )

    plt.suptitle("Feature Distributions: Normal vs Fraud", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_comparison.png"), bbox_inches="tight")
    plt.close()


def run_full_eda(df):
    """Run the complete EDA pipeline."""
    from data_loader import print_class_distribution

    class_counts, class_pct = print_class_distribution(df)

    fraud_df = df[df["Class"] == 1]
    normal_df = df[df["Class"] == 0]

    plot_class_distribution(class_counts, class_pct)
    plot_amount_analysis(normal_df, fraud_df)
    plot_time_analysis(normal_df, fraud_df)
    plot_correlation_analysis(df)
    plot_feature_comparison(normal_df, fraud_df)
