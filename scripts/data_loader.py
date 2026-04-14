import pandas as pd
from config import DATA_PATH


def load_data(path=DATA_PATH):
    """Load the credit card fraud dataset and print basic info."""
    df = pd.read_csv(path)

    print(f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"\nColumn names:\n{df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    return df


def print_class_distribution(df):
    """Print the class distribution of the dataset."""
    class_counts = df["Class"].value_counts()
    class_pct = df["Class"].value_counts(normalize=True) * 100

    print("\nClass Distribution:")
    print(f"  Normal (0):  {class_counts[0]:>7,}  ({class_pct[0]:.3f}%)")
    print(f"  Fraud  (1):  {class_counts[1]:>7,}  ({class_pct[1]:.3f}%)")
    print(f"  Imbalance ratio: 1 fraud per {class_counts[0] // class_counts[1]} normal transactions")

    return class_counts, class_pct
