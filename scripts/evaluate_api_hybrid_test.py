import json
import sys
import time
from pathlib import Path

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.hybrid_inference import HybridInference, FEATURES

TEST_PATH = ROOT / "data" / "processed" / "test_data.csv"
RESULTS_DIR = ROOT / "results"

# Match the current API implementation.
FUSION_THRESHOLD = 0.5

# Begin with a small sample before running the full test set.
SAMPLE_SIZE = 100


def main():
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_PATH}")

    test_df = pd.read_csv(TEST_PATH)

    required = set(FEATURES + ["Class"])
    missing = required - set(test_df.columns)
    if missing:
        raise ValueError(f"Test dataset is missing: {sorted(missing)}")

    # Deterministic sample for the initial consistency check.
    sample = test_df.head(min(SAMPLE_SIZE, len(test_df))).copy()

    y_true = sample["Class"].to_numpy(dtype=np.int32)

    print(f"Initializing actual HybridInference engine...")
    start_init = time.perf_counter()
    engine = HybridInference()
    init_seconds = time.perf_counter() - start_init

    print(f"Initialization: {init_seconds:.2f} seconds")
    print(f"Transactions to evaluate: {len(sample)}")
    print(f"Actual fraud cases in sample: {int(y_true.sum())}")
    print("Evaluating. Please wait...")

    scores = []
    predictions = []
    records = []

    start_eval = time.perf_counter()

    for row_number, (_, row) in enumerate(sample.iterrows(), start=1):
        transaction = {
            feature: float(row[feature])
            for feature in FEATURES
        }

        result = engine.predict(transaction)

        score = float(result["risk_score"])
        prediction = int(score >= FUSION_THRESHOLD)

        scores.append(score)
        predictions.append(prediction)

        records.append({
            "test_row": int(row_number),
            "actual_label": int(row["Class"]),
            "fusion_score": score,
            "predicted_fraud": prediction,
            "decision": result["decision"],
            "reconstruction_error": result["breakdown"][
                "reconstruction_error"
            ],
            "gnn_fraud_score": result["breakdown"]["gnn_fraud_score"],
        })

        if row_number % 10 == 0 or row_number == len(sample):
            print(f"Processed {row_number}/{len(sample)} transactions")

    elapsed = time.perf_counter() - start_eval

    y_score = np.asarray(scores)
    y_pred = np.asarray(predictions)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    metrics = {
        "evaluation": "actual HybridInference engine",
        "dataset": str(TEST_PATH.relative_to(ROOT)),
        "sample_size": len(sample),
        "actual_fraud_cases": int(y_true.sum()),
        "fusion_threshold": FUSION_THRESHOLD,
        "initialization_seconds": round(init_seconds, 4),
        "evaluation_seconds": round(elapsed, 4),
        "mean_latency_seconds": round(elapsed / len(sample), 6),
        "throughput_transactions_per_second": round(
            len(sample) / elapsed, 3
        ),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "false_positive_rate": float(fp / (fp + tn))
        if (fp + tn) else 0.0,
        "roc_auc": float(roc_auc_score(y_true, y_score))
        if len(np.unique(y_true)) == 2 else None,
        "average_precision": float(
            average_precision_score(y_true, y_score)
        )
        if len(np.unique(y_true)) == 2 else None,
        "confusion_matrix": {
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn),
        },
        "note": (
            "Initial sample evaluation using the actual inference engine. "
            "Not a full-test result. Fusion score is not a calibrated "
            "fraud probability."
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = RESULTS_DIR / "api_hybrid_sample_evaluation.json"
    predictions_path = RESULTS_DIR / "api_hybrid_sample_predictions.csv"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    pd.DataFrame(records).to_csv(predictions_path, index=False)

    print("\n=== ACTUAL INFERENCE ENGINE: SAMPLE RESULTS ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nReport saved to: {report_path}")
    print(f"Predictions saved to: {predictions_path}")


if __name__ == "__main__":
    main()