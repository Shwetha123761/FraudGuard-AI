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
LEGITIMATE_SAMPLE_SIZE = 5000
RANDOM_SEED = 42


def main():
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_PATH}")

    df = pd.read_csv(TEST_PATH)

    required = set(FEATURES + ["Class"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")

    fraud = df[df["Class"] == 1]
    legitimate = df[df["Class"] == 0]

    if len(fraud) == 0 or len(legitimate) == 0:
        raise ValueError("Test dataset must contain both classes.")

    # Keep every fraud case and reproducibly sample legitimate cases.
    legitimate_sample = legitimate.sample(
        n=min(LEGITIMATE_SAMPLE_SIZE, len(legitimate)),
        random_state=RANDOM_SEED,
    )

    sample = (
        pd.concat([fraud, legitimate_sample])
        .sample(frac=1, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )

    y_true = sample["Class"].to_numpy(dtype=np.int32)

    print("Initializing the actual HybridInference engine...")
    init_start = time.perf_counter()
    engine = HybridInference()
    initialization_seconds = time.perf_counter() - init_start

    print(f"Initialization: {initialization_seconds:.2f} seconds")
    print(f"Transactions: {len(sample):,}")
    print(f"Fraud cases: {int(y_true.sum()):,}")
    print(f"Legitimate cases: {int((y_true == 0).sum()):,}")
    print("Evaluation started. This may take several minutes.")

    scores = []
    predictions = []
    records = []

    evaluation_start = time.perf_counter()

    for i, (_, row) in enumerate(sample.iterrows(), start=1):
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
            "source_test_index": int(row.name),
            "actual_label": int(row["Class"]),
            "fusion_score": score,
            "predicted_fraud": prediction,
            "decision": result["decision"],
            "reconstruction_error": result["breakdown"][
                "reconstruction_error"
            ],
            "gnn_fraud_score": result["breakdown"]["gnn_fraud_score"],
        })

        if i % 250 == 0 or i == len(sample):
            elapsed = time.perf_counter() - evaluation_start
            rate = i / elapsed if elapsed else 0
            print(
                f"Processed {i:,}/{len(sample):,} "
                f"| elapsed={elapsed / 60:.1f} min "
                f"| rate={rate:.2f} transactions/sec"
            )

    evaluation_seconds = time.perf_counter() - evaluation_start

    y_score = np.asarray(scores, dtype=np.float64)
    y_pred = np.asarray(predictions, dtype=np.int32)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    metrics = {
        "evaluation": "actual HybridInference engine",
        "dataset": str(TEST_PATH.relative_to(ROOT)),
        "sampling_method": (
            "All fraud cases plus reproducible random sample "
            "of legitimate transactions"
        ),
        "random_seed": RANDOM_SEED,
        "sample_size": int(len(sample)),
        "actual_fraud_cases": int(y_true.sum()),
        "actual_legitimate_cases": int((y_true == 0).sum()),
        "fusion_threshold": FUSION_THRESHOLD,
        "initialization_seconds": round(initialization_seconds, 4),
        "evaluation_seconds": round(evaluation_seconds, 4),
        "mean_latency_seconds": round(
            evaluation_seconds / len(sample), 6
        ),
        "throughput_transactions_per_second": round(
            len(sample) / evaluation_seconds, 3
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
        "false_positive_rate": float(fp / (fp + tn)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(
            average_precision_score(y_true, y_score)
        ),
        "confusion_matrix": {
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn),
        },
        "note": (
            "This is a stratified sample, not the full test set. "
            "Its fraud prevalence differs from the original test set, "
            "so precision and average precision may not represent "
            "deployment prevalence. Fusion score is not a calibrated "
            "fraud probability."
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = RESULTS_DIR / "api_hybrid_stratified_results.json"
    predictions_path = (
        RESULTS_DIR / "api_hybrid_stratified_predictions.csv"
    )

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    pd.DataFrame(records).to_csv(predictions_path, index=False)

    print("\n=== STRATIFIED API-ENGINE EVALUATION ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nReport saved to: {report_path}")
    print(f"Predictions saved to: {predictions_path}")


if __name__ == "__main__":
    main()