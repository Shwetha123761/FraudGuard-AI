
import json
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "test_data.csv"
API_URL = "http://127.0.0.1:8000/v1/evaluate-hybrid"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Use the first held-out test row.
    row = df.iloc[0]
    features = {
        f"V{i}": float(row[f"V{i}"])
        for i in range(1, 29)
    }

    payload = {
        "card_id": "fraudguard_test_card",
        "amount": float(row["Amount"]),
        "features": features,
        "timestamp": float(row["Time"]),
    }

    print("Sending transaction to the hybrid API...")
    print(f"Dataset label (0=legitimate, 1=fraud): {int(row['Class'])}")

    response = requests.post(API_URL, json=payload, timeout=60)

    print(f"\nHTTP status: {response.status_code}")
    response.raise_for_status()

    result = response.json()
    print("\nAPI response:")
    print(json.dumps(result, indent=2))

    print("\nAPI endpoint test: PASS")


if __name__ == "__main__":
    main()

