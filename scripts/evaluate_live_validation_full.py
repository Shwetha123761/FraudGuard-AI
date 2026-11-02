"""
FraudGuard AI: full live-style validation evaluation.

Scores every row in data/processed/validation_data.csv using:
- the existing Autoencoder and its current threshold,
- the experimental local GCN scoring implementation,
- the current live fusion formula and threshold (0.5).

This does not train models or alter model weights/configuration.
It writes one new report to results/live_style_full_validation_results.json.

Run from project root:
    python .\scripts\evaluate_live_validation_full.py
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
DATA_PATH = ROOT / "data" / "processed" / "validation_data.csv"
OUTPUT_PATH = ROOT / "results" / "live_style_full_validation_results.json"
FUSION_THRESHOLD = 0.5
AE_BATCH_SIZE = 512
PROGRESS_EVERY = 1000


def main():
    torch.set_num_threads(1)

    print("=" * 76)
    print("FRAUDGUARD AI - FULL LIVE-STYLE VALIDATION EVALUATION")
    print("=" * 76)
    print("No models will be trained and no thresholds/configurations will be changed.")
    print("A new JSON report will be written to:")
    print(OUTPUT_PATH)

    df = pd.read_csv(DATA_PATH)
    if "Class" not in df.columns or not set(FEATURES).issubset(df.columns):
        raise ValueError("Validation CSV is missing expected feature or Class columns.")

    X = df[FEATURES].to_numpy(dtype=np.float32)
    y_true = df["Class"].to_numpy(dtype=np.int32)

    print(f"\nValidation transactions: {len(df):,}")
    print(f"Fraud cases: {int(y_true.sum()):,}")
    print(f"Legitimate cases: {int((y_true == 0).sum()):,}")
    print(f"Fusion threshold: {FUSION_THRESHOLD}")
    print("\nLoading inference engine...")

    t0 = time.perf_counter()
    engine = HybridInference()
    engine_init_seconds = time.perf_counter() - t0
    experiment = LocalGCNExperiment(engine)
    print(f"Engine initialization: {engine_init_seconds:.3f} seconds")
    print("\nCalculating Autoencoder reconstruction errors in batches...")

    ae_start = time.perf_counter()
    reconstructed = engine.autoencoder.predict(
        X, batch_size=AE_BATCH_SIZE, verbose=0
    )
    reconstruction_errors = np.mean(np.square(X - reconstructed), axis=1)
    anomaly_scores = np.log1p(
        np.maximum(reconstruction_errors, 0.0) / engine.ae_threshold
    )
    ae_seconds = time.perf_counter() - ae_start
    print(f"Autoencoder pass completed in {ae_seconds:.2f} seconds")

    print("\nScaling features and finding reference neighbors...")
    scale_start = time.perf_counter()
    scaled = (X - engine.scaler_mean) / engine.scaler_scale
    _, neighbor_indices = engine.neighbor_search.kneighbors(scaled)
    neighbor_seconds = time.perf_counter() - scale_start
    print(f"Scaling + neighbor search completed in {neighbor_seconds:.2f} seconds")

    print("\nScoring GNN query nodes. This stage may take a while.")
    gnn_probabilities = np.empty(len(df), dtype=np.float64)
    gnn_start = time.perf_counter()
    stage_start = gnn_start

    for i in range(len(df)):
        gnn_probabilities[i] = experiment.fraud_probability(
            scaled[i:i + 1], neighbor_indices[i]
        )

        if (i + 1) % PROGRESS_EVERY == 0 or i + 1 == len(df):
            elapsed = time.perf_counter() - gnn_start
            interval = time.perf_counter() - stage_start
            print(
                f"Completed {i + 1:,}/{len(df):,} | "
                f"GNN stage elapsed={elapsed / 60:.1f} min | "
                f"average={elapsed / (i + 1):.4f}s/transaction | "
                f"last block={interval:.1f}s"
            )
            stage_start = time.perf_counter()

    gnn_seconds = time.perf_counter() - gnn_start

    # Matches the current formula in engine/hybrid_inference.py.
    scores = (
        0.5 * gnn_probabilities
        + 0.5 * np.minimum(anomaly_scores / 3.0, 1.0)
    )
    predictions = (scores >= FUSION_THRESHOLD).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    metrics = {
        "evaluation": "full_live_style_validation",
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "sampled": False,
        "transactions": int(len(y_true)),
        "fraud_cases": int(y_true.sum()),
        "legitimate_cases": int((y_true == 0).sum()),
        "fusion_threshold": FUSION_THRESHOLD,
        "fusion_formula": "0.5 * gnn_probability + 0.5 * min(log1p(reconstruction_error / ae_threshold) / 3, 1)",
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "false_positive_rate": float(fpr),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "fraud_cases_missed": int(fn),
        "transactions_flagged": int(predictions.sum()),
        "engine_initialization_seconds": float(engine_init_seconds),
        "autoencoder_batch_seconds": float(ae_seconds),
        "scaling_neighbor_search_seconds": float(neighbor_seconds),
        "gnn_scoring_seconds": float(gnn_seconds),
        "total_evaluation_seconds": float(
            engine_init_seconds + ae_seconds + neighbor_seconds + gnn_seconds
        ),
        "mean_gnn_query_seconds": float(gnn_seconds / len(df)),
        "mean_gnn_query_throughput_tps": float(len(df) / gnn_seconds),
        "score_mean": float(np.mean(scores)),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "note": (
            "Validation-only diagnostic. The experimental local GCN implementation "
            "was previously checked against the baseline on a 200-row sample. "
            "Threshold 0.5 was not tuned by this script. These are not test-set results."
        ),
    }

    print("\n" + "=" * 76)
    print("FULL VALIDATION RESULTS")
    print("=" * 76)
    print(f"Precision:          {metrics['precision']:.6f}")
    print(f"Recall:             {metrics['recall']:.6f}")
    print(f"F1-score:           {metrics['f1']:.6f}")
    print(f"False-positive rate:{metrics['false_positive_rate']:.6f}")
    print(f"ROC-AUC:            {metrics['roc_auc']:.6f}")
    print(f"Average Precision:  {metrics['average_precision']:.6f}")
    print(f"TP={tp} | FP={fp} | TN={tn} | FN={fn}")
    print(f"Fraud cases missed: {fn}")
    print(f"Transactions flagged: {int(predictions.sum())}")
    print("\nTiming:")
    print(f"Autoencoder:        {ae_seconds:.2f}s")
    print(f"Neighbor search:    {neighbor_seconds:.2f}s")
    print(f"GNN scoring:        {gnn_seconds / 60:.2f} minutes")
    print(f"Mean GNN query:     {gnn_seconds / len(df):.4f}s")
    print(f"GNN throughput:     {len(df) / gnn_seconds:.2f} queries/second")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved report: {OUTPUT_PATH}")
    print("\nReminder: this uses the current threshold without tuning.")
    print("Do not use these validation results as final test-set performance.")


if __name__ == "__main__":
    main()
