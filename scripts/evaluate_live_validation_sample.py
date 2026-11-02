"""
FraudGuard AI: live-style validation evaluation using the experimentally verified
local GCN calculation.

This is a validation-only diagnostic. It does not modify model weights,
thresholds, or project results. It scores all 52 validation fraud cases plus a
reproducible random sample of legitimate transactions to keep runtime bounded.

Run from the project root:
    python .\scripts\evaluate_live_validation_sample.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from engine.hybrid_inference import HybridInference
from scripts.benchmark_local_gnn import LocalGCNExperiment

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
DATA_PATH = ROOT / "data" / "processed" / "validation_data.csv"
LEGITIMATE_SAMPLE_SIZE = 5000
RANDOM_SEED = 42
FUSION_THRESHOLD = 0.5


def main():
    torch.set_num_threads(1)

    print("=" * 72)
    print("FRAUDGUARD AI - LIVE-STYLE VALIDATION SAMPLE EVALUATION")
    print("=" * 72)
    print("This script does not change models, thresholds, or project result files.")

    df = pd.read_csv(DATA_PATH)
    fraud = df[df["Class"] == 1]
    legitimate = df[df["Class"] == 0]

    if len(fraud) == 0:
        raise RuntimeError("Validation data contains no fraud cases.")

    n_legitimate = min(LEGITIMATE_SAMPLE_SIZE, len(legitimate))
    legitimate_sample = legitimate.sample(
        n=n_legitimate, random_state=RANDOM_SEED
    )
    sample = pd.concat([fraud, legitimate_sample]).sample(
        frac=1, random_state=RANDOM_SEED
    ).reset_index(drop=True)

    y_true = sample["Class"].to_numpy(dtype=int)

    print(f"Validation rows available: {len(df):,}")
    print(f"Fraud cases included: {int(y_true.sum()):,}")
    print(f"Legitimate cases sampled: {int((y_true == 0).sum()):,}")
    print(f"Total evaluated: {len(sample):,}")
    print("\nLoading inference engine...")

    start = time.perf_counter()
    engine = HybridInference()
    init_time = time.perf_counter() - start
    experiment = LocalGCNExperiment(engine)
    print(f"Engine initialization: {init_time:.3f} seconds")
    print("Scoring transactions with the local GCN experiment...")

    scores = np.empty(len(sample), dtype=np.float64)
    latencies = np.empty(len(sample), dtype=np.float64)

    start_all = time.perf_counter()

    for i, (_, row) in enumerate(sample.iterrows()):
        start = time.perf_counter()

        values = row[FEATURES].to_numpy(dtype=np.float32)
        ae_input = values.reshape(1, -1)
        reconstruction = engine.autoencoder.predict(ae_input, verbose=0)
        reconstruction_error = float(np.mean(np.square(ae_input - reconstruction)))
        anomaly_score = float(
            np.log1p(max(reconstruction_error, 0.0) / engine.ae_threshold)
        )

        scaled = (
            (values - engine.scaler_mean) / engine.scaler_scale
        ).reshape(1, -1)
        _, neighbors = engine.neighbor_search.kneighbors(scaled)
        gnn_probability = experiment.fraud_probability(scaled, neighbors[0])

        scores[i] = 0.5 * gnn_probability + 0.5 * min(anomaly_score / 3.0, 1.0)
        latencies[i] = time.perf_counter() - start

        if (i + 1) % 500 == 0 or i + 1 == len(sample):
            elapsed = time.perf_counter() - start_all
            print(
                f"Completed {i + 1:,}/{len(sample):,} | "
                f"elapsed={elapsed:.1f}s | "
                f"mean={elapsed / (i + 1):.4f}s/transaction"
            )

    predictions = (scores >= FUSION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)

    print("\n--- RESULTS AT CURRENT FUSION THRESHOLD ---")
    print(f"Fusion threshold: {FUSION_THRESHOLD:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1-score: {f1:.6f}")
    print(f"False-positive rate: {fpr:.6f}")
    print(f"ROC-AUC: {roc_auc_score(y_true, scores):.6f}")
    print(f"Average Precision: {average_precision_score(y_true, scores):.6f}")
    print(f"TP={tp} | FP={fp} | TN={tn} | FN={fn}")
    print(f"Fraud cases approved: {fn}")
    print(f"Transactions flagged: {int(predictions.sum())}")

    total_elapsed = time.perf_counter() - start_all
    print("\n--- LATENCY ---")
    print(f"Total scoring time: {total_elapsed:.2f} seconds")
    print(f"Mean per-transaction latency: {np.mean(latencies):.4f} seconds")
    print(f"Median latency: {np.median(latencies):.4f} seconds")
    print(f"95th-percentile latency: {np.percentile(latencies, 95):.4f} seconds")
    print(f"Approximate throughput: {1 / np.mean(latencies):.2f} transactions/second")

    print("\nIMPORTANT:")
    print("- This is a sampled validation evaluation, not a full-validation result.")
    print("- All validation fraud cases are included, but only a sample of legitimate cases.")
    print("- Because the legitimate class is sampled, FPR/precision and threshold behavior")
    print("  may differ from the complete validation dataset.")
    print("- Threshold selection and the final test evaluation have not been performed.")
    print("- No project files were modified.")


if __name__ == "__main__":
    main()
