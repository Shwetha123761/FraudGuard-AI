from pathlib import Path
import json
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from engine.hybrid_inference import HybridInference, FEATURES

ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data" / "processed" / "test_data.csv"
RESULTS_DIR = ROOT / "results"
OUTPUT_PATH = RESULTS_DIR / "gnn_hybrid_test_results.json"

# The threshold is fixed before evaluating the test set.
# Do not tune this threshold using test labels.
FUSION_THRESHOLD = 0.5

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading held-out test dataset...")
    df = pd.read_csv(TEST_PATH)

    missing = [name for name in FEATURES + ["Class"] if name not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    y_true = df["Class"].astype(int).to_numpy()
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("Class must contain only binary labels 0 and 1.")

    print(f"Rows: {len(df):,}")
    print(f"Fraud cases: {int(y_true.sum()):,}")
    print("Initializing existing hybrid inference model...")
    engine = HybridInference()

    scores = []
    decisions = []
    started = time.perf_counter()

    # Use the same inference path as the dashboard.
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        transaction = {name: float(row[name]) for name in FEATURES}
        result = engine.predict(transaction)

        scores.append(float(result["risk_score"]))
        decisions.append(result["decision"] != "APPROVE")

        if i % 5000 == 0 or i == len(df):
            print(f"Processed {i:,}/{len(df):,} rows")

    elapsed = time.perf_counter() - started
    y_pred = np.asarray(decisions, dtype=int)
    scores = np.asarray(scores, dtype=float)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    result = {
        "model": "Autoencoder + feature-similarity GNN prototype",
        "partition": "held_out_chronological_test",
        "rows": int(len(df)),
        "fraud_cases": int(y_true.sum()),
        "legitimate_cases": int((y_true == 0).sum()),
        "fusion_threshold": FUSION_THRESHOLD,
        "threshold_selection": (
            "Fixed prototype threshold from inference implementation; "
            "not tuned on test labels. Validation-based calibration remains pending."
        ),
        "test_used_for_training_or_threshold_selection": False,
        "score_note": (
            "Prototype fusion score is not a calibrated fraud probability. "
            "These metrics evaluate the current inference decision rule."
        ),
        "metrics": {
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, scores)),
            "average_precision": float(average_precision_score(y_true, scores)),
            "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "fraud_detected": int(tp),
            "fraud_missed": int(fn),
            "legitimate_transactions_flagged": int(fp),
        },
        "runtime_seconds": round(elapsed, 2),
        "graph_method": "training-reference feature-similarity graph",
        "limitations": [
            "Graph edges represent feature similarity, not real card/account relationships.",
            "The fusion threshold has not been calibrated using validation predictions.",
            "The dataset has few fraud cases in the held-out test partition.",
            "Velocity rules are not evaluated by this script.",
        ],
    }

    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n===== HYBRID TEST RESULTS =====")
    print(json.dumps(result, indent=2))
    print(f"\nSaved results to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

