"""
FraudGuard AI: final test-set evaluation with a threshold frozen from validation.

Frozen operating threshold: 0.55
Rationale: among the displayed validation grid points in the previous run,
0.55 had the highest F1 (0.6735), with validation recall 0.6346 and FPR 0.0287%.
This threshold is fixed before reading test labels for metric calculation.

This script does NOT train models or modify fusion_config.json.
It scores the untouched chronological test partition and saves a new report.

Run from project root:
    python .\scripts\evaluate_live_fusion_test.py
"""

import sys
import json
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
DATA_PATH = ROOT / "data" / "processed" / "test_data.csv"
OUTPUT_PATH = ROOT / "results" / "live_style_final_test_results.json"

# Frozen before evaluating test labels; do not tune this using test results.
FROZEN_THRESHOLD = 0.55
AE_BATCH_SIZE = 512
PROGRESS_EVERY = 5000


def main():
    torch.set_num_threads(1)

    print("=" * 76)
    print("FRAUDGUARD AI - FINAL TEST EVALUATION")
    print("=" * 76)
    print(f"Dataset: {DATA_PATH.relative_to(ROOT)}")
    print(f"Frozen fusion threshold: {FROZEN_THRESHOLD:.4f}")
    print("No model training or configuration changes will be made.")
    print("Do not tune this threshold using the test results.")

    df = pd.read_csv(DATA_PATH)
    if "Class" not in df.columns or not set(FEATURES).issubset(df.columns):
        raise ValueError("Test CSV is missing expected feature or Class columns.")

    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df["Class"].to_numpy(dtype=np.int32)
    n = len(y)
    n_pos = int(y.sum())
    n_neg = int(n - n_pos)

    print(f"\nTest transactions: {n:,}")
    print(f"Fraud cases: {n_pos:,}")
    print(f"Legitimate cases: {n_neg:,}")
    print("\nLoading inference engine...")

    t0 = time.perf_counter()
    engine = HybridInference()
    experiment = LocalGCNExperiment(engine)
    init_seconds = time.perf_counter() - t0
    print(f"Engine and local GCN setup: {init_seconds:.2f}s")

    print("\nComputing Autoencoder reconstruction errors...")
    t0 = time.perf_counter()
    reconstructed = engine.autoencoder.predict(
        X, batch_size=AE_BATCH_SIZE, verbose=0
    )
    errors = np.mean(np.square(X - reconstructed), axis=1)
    anomaly = np.log1p(np.maximum(errors, 0.0) / engine.ae_threshold)
    ae_seconds = time.perf_counter() - t0
    print(f"Autoencoder: {ae_seconds:.2f}s")

    print("\nScaling features and finding reference neighbors...")
    t0 = time.perf_counter()
    scaled = (X - engine.scaler_mean) / engine.scaler_scale
    _, neighbors = engine.neighbor_search.kneighbors(scaled)
    neighbor_seconds = time.perf_counter() - t0
    print(f"Scaling + neighbor search: {neighbor_seconds:.2f}s")

    print("\nScoring GNN query nodes...")
    gnn_probs = np.empty(n, dtype=np.float64)
    t0 = time.perf_counter()
    for i in range(n):
        gnn_probs[i] = experiment.fraud_probability(
            scaled[i:i + 1], neighbors[i]
        )
        if (i + 1) % PROGRESS_EVERY == 0 or i + 1 == n:
            elapsed = time.perf_counter() - t0
            print(
                f"Completed {i + 1:,}/{n:,}; elapsed={elapsed:.1f}s; "
                f"average={elapsed / (i + 1):.5f}s/row"
            )
    gnn_seconds = time.perf_counter() - t0

    # Same live-style fusion formula used in the validation evaluation.
    scores = 0.5 * gnn_probs + 0.5 * np.minimum(anomaly / 3.0, 1.0)
    predictions = (scores >= FROZEN_THRESHOLD).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if fp + tn else 0.0
    precision = float(precision_score(y, predictions, zero_division=0))
    recall = float(recall_score(y, predictions, zero_division=0))
    f1 = float(f1_score(y, predictions, zero_division=0))
    roc_auc = float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None
    ap = float(average_precision_score(y, scores)) if n_pos else None

    report = {
        "evaluation": "final_live_style_test",
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "sampled": False,
        "threshold_frozen_before_test_evaluation": True,
        "threshold_selection_basis": (
            "Validation grid: threshold 0.55 had the highest F1 among displayed "
            "grid points; test data was not used to choose it."
        ),
        "fusion_threshold": FROZEN_THRESHOLD,
        "fusion_formula": "0.5 * gnn_probability + 0.5 * min(log1p(reconstruction_error / ae_threshold) / 3, 1)",
        "transactions": int(n),
        "fraud_cases": n_pos,
        "legitimate_cases": n_neg,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "roc_auc": roc_auc,
        "average_precision": ap,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "fraud_cases_missed": int(fn),
        "transactions_flagged": int(predictions.sum()),
        "engine_setup_seconds": float(init_seconds),
        "autoencoder_seconds": float(ae_seconds),
        "neighbor_search_seconds": float(neighbor_seconds),
        "gnn_scoring_seconds": float(gnn_seconds),
        "mean_gnn_query_seconds": float(gnn_seconds / n),
        "gnn_query_throughput_per_second": float(n / gnn_seconds),
        "note": (
            "Final test evaluation with a threshold frozen from validation. "
            "Do not alter threshold based on this test report; any further changes "
            "require a new holdout or nested validation protocol."
        ),
    }

    print("\n" + "=" * 76)
    print("FINAL TEST RESULTS")
    print("=" * 76)
    print(f"Threshold:          {FROZEN_THRESHOLD:.4f}")
    print(f"Precision:          {precision:.6f}")
    print(f"Recall:             {recall:.6f}")
    print(f"F1-score:           {f1:.6f}")
    print(f"False-positive rate:{fpr:.6f}")
    print(f"ROC-AUC:            {roc_auc:.6f}" if roc_auc is not None else "ROC-AUC: undefined")
    print(f"Average Precision:  {ap:.6f}" if ap is not None else "Average Precision: undefined")
    print(f"TP={tp} | FP={fp} | TN={tn} | FN={fn}")
    print(f"Fraud cases missed: {fn}")
    print(f"Transactions flagged: {int(predictions.sum())}")
    print("\nTiming:")
    print(f"Autoencoder:        {ae_seconds:.2f}s")
    print(f"Neighbor search:    {neighbor_seconds:.2f}s")
    print(f"GNN scoring:        {gnn_seconds:.2f}s")
    print(f"Mean GNN query:     {gnn_seconds / n:.6f}s")
    print(f"GNN throughput:     {n / gnn_seconds:.2f} queries/second")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report: {OUTPUT_PATH}")
    print("\nDo not use this test report to tune the threshold.")


if __name__ == "__main__":
    main()
