
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

FUSION_THRESHOLD = 0.5
CHECKPOINT_INTERVAL = 250


def main():
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_PATH}")

    df = pd.read_csv(TEST_PATH)

    required = set(FEATURES + ["Class"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")

    y_true = df["Class"].to_numpy(dtype=np.int32)

    print("Initializing actual HybridInference engine...")
    init_start = time.perf_counter()
    engine = HybridInference()
    initialization_seconds = time.perf_counter() - init_start

    print(f"Initialization completed in {initialization_seconds:.2f} seconds")
    print(f"Total test transactions: {len(df):,}")
    print(f"Fraud cases: {int(y_true.sum()):,}")
    print(f"Legitimate cases: {int((y_true == 0).sum()):,}")
    print(f"Frozen fusion threshold: {FUSION_THRESHOLD}")
    print("Starting full held-out test evaluation...")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = RESULTS_DIR / "api_hybrid_full_test_checkpoint.csv"
    report_path = RESULTS_DIR / "api_hybrid_full_test_results.json"
    predictions_path = RESULTS_DIR / "api_hybrid_full_test_predictions.csv"

    records = []
    start = time.perf_counter()

    try:
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            transaction = {
                feature: float(row[feature])
                for feature in FEATURES
            }

            result = engine.predict(transaction)
            score = float(result["risk_score"])
            prediction = int(score >= FUSION_THRESHOLD)

            records.append({
                "source_test_index": i - 1,
                "actual_label": int(row["Class"]),
                "fusion_score": score,
                "predicted_fraud": prediction,
                "decision": result["decision"],
                "reconstruction_error": result["breakdown"][
                    "reconstruction_error"
                ],
                "gnn_fraud_score": result["breakdown"][
                    "gnn_fraud_score"
                ],
            })

            if i % CHECKPOINT_INTERVAL == 0 or i == len(df):
                pd.DataFrame(records).to_csv(
                    checkpoint_path,
                    index=False,
                )

                elapsed = time.perf_counter() - start
                rate = i / elapsed if elapsed else 0
                remaining = len(df) - i
                eta_minutes = remaining / rate / 60 if rate else 0

                print(
                    f"Processed {i:,}/{len(df):,} "
                    f"| elapsed={elapsed / 60:.1f} min "
                    f"| speed={rate:.2f} transactions/sec "
                    f"| estimated remaining={eta_minutes:.1f} min"
                )

    except KeyboardInterrupt:
        pd.DataFrame(records).to_csv(checkpoint_path, index=False)
        print("\nEvaluation interrupted by user.")
        print(f"Completed predictions saved to: {checkpoint_path}")
        print("This checkpoint is not a complete test evaluation.")
        return

    evaluation_seconds = time.perf_counter() - start

    predictions_df = pd.DataFrame(records)
    predictions_df.to_csv(predictions_path, index=False)

    actual = predictions_df["actual_label"].to_numpy(dtype=np.int32)
    scores = predictions_df["fusion_score"].to_numpy(dtype=np.float64)
    predicted = predictions_df["predicted_fraud"].to_numpy(dtype=np.int32)

    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    metrics = {
        "evaluation": "actual HybridInference engine",
        "dataset": str(TEST_PATH.relative_to(ROOT)),
        "sample_size": int(len(predictions_df)),
        "actual_fraud_cases": int(actual.sum()),
        "actual_legitimate_cases": int((actual == 0).sum()),
        "fusion_threshold": FUSION_THRESHOLD,
        "initialization_seconds": round(initialization_seconds, 4),
        "evaluation_seconds": round(evaluation_seconds, 4),
        "mean_latency_seconds": round(
            evaluation_seconds / len(predictions_df), 6
        ),
        "throughput_transactions_per_second": round(
            len(predictions_df) / evaluation_seconds, 3
        ),
        "precision": float(
            precision_score(actual, predicted, zero_division=0)
        ),
        "recall": float(
            recall_score(actual, predicted, zero_division=0)
        ),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)),
        "roc_auc": float(roc_auc_score(actual, scores)),
        "average_precision": float(
            average_precision_score(actual, scores)
        ),
        "confusion_matrix": {
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn),
        },
        "note": (
            "Full held-out test evaluation using the actual inference "
            "engine. Threshold remained fixed at 0.5. The risk score "
            "is not a calibrated fraud probability."
        ),
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\n=== FULL TEST EVALUATION RESULTS ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nFinal report: {report_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()

