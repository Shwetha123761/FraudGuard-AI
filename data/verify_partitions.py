from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd()
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

files = [
    "train_data.csv",
    "validation_data.csv",
    "test_data.csv",
]

total_rows = 0
total_fraud = 0

print("PARTITION VERIFICATION")
print("-" * 50)

for filename in files:
    path = PROCESSED_DATA_DIR / filename

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        continue

    df = pd.read_csv(path)

    rows = len(df)
    fraud = int(df["Class"].sum())
    missing = int(df.isna().sum().sum())

    total_rows += rows
    total_fraud += fraud

    print(f"\nFile: {filename}")
    print(f"Transactions: {rows:,}")
    print(f"Fraud cases: {fraud}")
    print(f"Columns: {len(df.columns)}")
    print(f"Missing values: {missing}")

print("\n" + "-" * 50)
print(f"Total transactions: {total_rows:,}")
print(f"Total fraud cases: {total_fraud}")
print("Verification completed.")
