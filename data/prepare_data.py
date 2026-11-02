# data/prepare_data.py

import sys
from pathlib import Path

import pandas as pd

# Add project root to system path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.config import RAW_DATA_PATH, PROCESSED_DATA_DIR


def load_and_split_temporal_data():
    """
    Create chronological training, validation, and test partitions.

    Approximate proportions before embargo removal:
        Training:   64%
        Validation: 16%
        Testing:    20%

    An embargo gap is removed between partitions.
    """

    print(f"[*] Checking raw data path: {RAW_DATA_PATH}")

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {RAW_DATA_PATH}"
        )

    # Load dataset
    df = pd.read_csv(RAW_DATA_PATH)

    print(f"[*] Dataset loaded: {df.shape}")

    # Validate required columns
    required_columns = {"Time", "Class"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    # Sort chronologically
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)

    # Check labels
    if df["Class"].isna().any():
        raise ValueError("Class column contains missing labels.")

    if not set(df["Class"].unique()).issubset({0, 1}):
        raise ValueError("Class column must contain only 0 and 1.")

    # Define partition boundaries
    n_rows = len(df)

    train_end = int(n_rows * 0.64)
    validation_end = int(n_rows * 0.80)

    # Embargo: remove 0.1% of rows at each partition boundary.
    # This is a row-count gap, not a fixed time-duration gap.
    embargo = max(1, int(n_rows * 0.001))

    train_df = df.iloc[:train_end].copy()

    validation_start = min(train_end + embargo, n_rows)
    validation_df = df.iloc[validation_start:validation_end].copy()

    test_start = min(validation_end + embargo, n_rows)
    test_df = df.iloc[test_start:].copy()

    # Report partition information
    partitions = {
        "Training": train_df,
        "Validation": validation_df,
        "Testing": test_df,
    }

    for name, partition in partitions.items():
        fraud_count = int(partition["Class"].sum())
        fraud_rate = partition["Class"].mean() * 100

        print(
            f"{name}: rows={len(partition):,}, "
            f"fraud={fraud_count}, "
            f"fraud_rate={fraud_rate:.4f}%, "
            f"Time range="
            f"{partition['Time'].min():.6f} to "
            f"{partition['Time'].max():.6f}"
        )

    # Check chronological separation
    assert train_df["Time"].max() <= validation_df["Time"].min()
    assert validation_df["Time"].max() <= test_df["Time"].min()

    # Save partitions
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_path = PROCESSED_DATA_DIR / "train_data.csv"
    validation_path = PROCESSED_DATA_DIR / "validation_data.csv"
    test_path = PROCESSED_DATA_DIR / "test_data.csv"

    train_df.to_csv(train_path, index=False)
    validation_df.to_csv(validation_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n[+] Temporal partitions saved successfully:")
    print(f"    Training:   {train_path}")
    print(f"    Validation: {validation_path}")
    print(f"    Testing:    {test_path}")

    print(
        "\n[!] These files retain the dataset's existing feature values. "
        "The original scaling procedure has not been verified."
    )


if __name__ == "__main__":
    load_and_split_temporal_data()