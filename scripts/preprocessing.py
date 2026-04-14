import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from config import RANDOM_STATE, TEST_SIZE, SMOTE_RATIO, COLORS, OUTPUT_DIR


def check_missing_values(df):
    """Check for missing values and duplicates, remove duplicates if found."""
    missing = df.isnull().sum()
    total_missing = missing.sum()

    print(f"Total missing values: {total_missing}")
    if total_missing == 0:
        print("No missing values found - the dataset is clean.")
    else:
        print("\nMissing values per column:")
        print(missing[missing > 0])

    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates:,}")

    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Shape after removing duplicates: {df.shape}")

    return df


def scale_features(df):
    """Scale Amount and Time features using StandardScaler."""
    scaler = StandardScaler()
    df = df.copy()

    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    df["Time_scaled"] = scaler.fit_transform(df[["Time"]])
    df = df.drop(columns=["Amount", "Time"])

    print("Scaling complete.")
    print(f"  Amount_scaled -- mean: {df['Amount_scaled'].mean():.4f}, std: {df['Amount_scaled'].std():.4f}")
    print(f"  Time_scaled   -- mean: {df['Time_scaled'].mean():.4f}, std: {df['Time_scaled'].std():.4f}")
    print(f"  Final columns ({len(df.columns)}): {df.columns.tolist()}")

    return df


def split_data(df):
    """Separate features/target and perform stratified train-test split."""
    X = df.drop(columns=["Class"])
    y = df["Class"]

    print(f"Features (X) shape: {X.shape}")
    print(f"Target   (y) shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nTraining set: {X_train.shape[0]:,} samples")
    print(f"Testing set:  {X_test.shape[0]:,} samples")
    print(f"Fraud % in train: {y_train.mean()*100:.3f}%")
    print(f"Fraud % in test:  {y_test.mean()*100:.3f}%")

    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):
    """Apply SMOTE oversampling to balance the training data."""
    smote = SMOTE(sampling_strategy=SMOTE_RATIO, random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print("After SMOTE (oversampling):")
    print(f"  Normal: {(y_resampled == 0).sum():,}")
    print(f"  Fraud:  {(y_resampled == 1).sum():,}")

    return X_resampled, y_resampled


def apply_undersampling(X_train, y_train):
    """Apply random undersampling to balance the training data."""
    undersample = RandomUnderSampler(random_state=RANDOM_STATE)
    X_resampled, y_resampled = undersample.fit_resample(X_train, y_train)

    print("After Undersampling:")
    print(f"  Normal: {(y_resampled == 0).sum():,}")
    print(f"  Fraud:  {(y_resampled == 1).sum():,}")

    return X_resampled, y_resampled


def plot_resampling_comparison(y_train, y_train_smote, y_train_under):
    """Visualize the effect of resampling techniques."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ["Original", "After SMOTE", "After Undersampling"]
    datasets = [
        pd.Series(y_train).value_counts().values,
        pd.Series(y_train_smote).value_counts().values,
        pd.Series(y_train_under).value_counts().values,
    ]

    for ax, title, data in zip(axes, titles, datasets):
        ax.bar(["Normal", "Fraud"], data, color=COLORS, edgecolor="black", linewidth=0.8)
        for i, v in enumerate(data):
            ax.text(i, v + max(data) * 0.02, f"{v:,}", ha="center", fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Count")

    plt.suptitle("Effect of Resampling Techniques", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "resampling_comparison.png"), bbox_inches="tight")
    plt.show()


def preprocess_pipeline(df):
    """Run the full preprocessing pipeline. Returns all needed datasets."""
    print("=" * 60)
    print("  STEP 1: Missing Values & Duplicates")
    print("=" * 60)
    df = check_missing_values(df)

    print(f"\n{'=' * 60}")
    print("  STEP 2: Feature Scaling")
    print("=" * 60)
    df = scale_features(df)

    print(f"\n{'=' * 60}")
    print("  STEP 3: Train-Test Split")
    print("=" * 60)
    X_train, X_test, y_train, y_test = split_data(df)

    print(f"\n{'=' * 60}")
    print("  STEP 4: Resampling")
    print("=" * 60)
    print(f"\nBefore resampling:")
    print(f"  Normal: {(y_train == 0).sum():,}")
    print(f"  Fraud:  {(y_train == 1).sum():,}\n")

    X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    print()
    X_train_under, y_train_under = apply_undersampling(X_train, y_train)

    plot_resampling_comparison(y_train, y_train_smote, y_train_under)

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_smote": X_train_smote, "y_train_smote": y_train_smote,
        "X_train_under": X_train_under, "y_train_under": y_train_under,
    }
