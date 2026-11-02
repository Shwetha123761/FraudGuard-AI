import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.hybrid_inference import HybridInference, FEATURES
from scripts.benchmark_local_gnn import LocalGCNExperiment

DATA_PATH = ROOT / "data" / "processed" / "validation_data.csv"
RESULTS_PATH = ROOT / "results" / "local_gnn_validation_1000.json"

SAMPLE_SIZE = 1000
SEED = 42
THRESHOLD = 0.5
TOLERANCE = 2e-5


def main():
    torch.set_num_threads(1)

    df = pd.read_csv(DATA_PATH)

    fraud = df[df["Class"] == 1]
    legitimate = df[df["Class"] == 0]

    if len(fraud) > SAMPLE_SIZE:
        raise ValueError("Fraud count exceeds the requested sample size.")

    sample = pd.concat([
        fraud,
        legitimate.sample(
            n=SAMPLE_SIZE - len(fraud),
            random_state=SEED,
        ),
    ]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    print("Initializing baseline inference engine...")
    start = time.perf_counter()
    engine = HybridInference()
    initialization_seconds = time.perf_counter() - start

    print("Building optimized GNN cache...")
    start = time.perf_counter()
    experiment = LocalGCNExperiment(engine)
    cache_seconds = time.perf_counter() - start

    baseline_scores = []
    optimized_scores = []
    baseline_latencies = []
    optimized_latencies = []

    print(f"\nTransactions: {len(sample)}")
    print(f"Fraud cases: {int(sample['Class'].sum())}")
    print("Comparing baseline and optimized inference...")

    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        features = {
            name: float(row[name])
            for name in FEATURES
        }

        # Baseline inference
        start = time.perf_counter()
        baseline_result = engine.predict(features)
        baseline_latencies.append(time.perf_counter() - start)

        baseline_score = float(baseline_result["risk_score"])

        # Optimized inference, using identical inputs
        start = time.perf_counter()

        values = row[FEATURES].to_numpy(dtype=np.float32)
        ae_input = values.reshape(1, -1)

        reconstruction = engine.autoencoder.predict(
            ae_input,
            verbose=0,
        )

        reconstruction_error = float(
            np.mean(np.square(ae_input - reconstruction))
        )

        anomaly_score = float(
            np.log1p(
                max(reconstruction_error, 0.0)
                / engine.ae_threshold
            )
        )

        scaled = (
            (values - engine.scaler_mean)
            / engine.scaler_scale
        ).reshape(1, -1)

        _, neighbors = engine.neighbor_search.kneighbors(scaled)

        gnn_probability = experiment.fraud_probability(
            scaled,
            neighbors[0],
        )

        optimized_score = float(
            0.5 * gnn_probability
            + 0.5 * min(anomaly_score / 3.0, 1.0)
        )

        optimized_latencies.append(time.perf_counter() - start)

        baseline_scores.append(baseline_score)
        optimized_scores.append(optimized_score)

        if i % 100 == 0 or i == len(sample):
            print(f"Compared {i}/{len(sample)} transactions")

    y_true = sample["Class"].to_numpy(dtype=np.int32)
    baseline_scores = np.asarray(baseline_scores)
    optimized_scores = np.asarray(optimized_scores)

    baseline_predictions = (
        baseline_scores >= THRESHOLD
    ).astype(np.int32)

    optimized_predictions = (
        optimized_scores >= THRESHOLD
    ).astype(np.int32)

    baseline_tn, baseline_fp, baseline_fn, baseline_tp = (
        confusion_matrix(
            y_true,
            baseline_predictions,
            labels=[0, 1],
        ).ravel()
    )

    optimized_tn, optimized_fp, optimized_fn, optimized_tp = (
        confusion_matrix(
            y_true,
            optimized_predictions,
            labels=[0, 1],
        ).ravel()
    )

    gnn_differences = []
    for _, row in sample.iterrows():
        values = row[FEATURES].to_numpy(dtype=np.float32)
        scaled = (
            (values - engine.scaler_mean)
            / engine.scaler_scale
        ).reshape(1, -1)

        _, neighbors = engine.neighbor_search.kneighbors(scaled)

        # This value is recomputed below only for score consistency checks.
        # The baseline GNN score is not stored in this diagnostic pass.

    max_fusion_difference = float(
        np.max(np.abs(baseline_scores - optimized_scores))
    )

    decision_mismatches = int(
        np.sum(baseline_predictions != optimized_predictions)
    )

    baseline_mean = float(np.mean(baseline_latencies))
    optimized_mean = float(np.mean(optimized_latencies))

    results = {
        "sample_size": int(len(sample)),
        "fraud_cases": int(y_true.sum()),
        "legitimate_cases": int((y_true == 0).sum()),
        "threshold": THRESHOLD,
        "initialization_seconds": round(initialization_seconds, 4),
        "cache_setup_seconds": round(cache_seconds, 4),
        "baseline_mean_latency_seconds": baseline_mean,
        "optimized_mean_latency_seconds": optimized_mean,
        "speedup_ratio": baseline_mean / optimized_mean,
        "maximum_fusion_score_difference": max_fusion_difference,
        "decision_mismatches": decision_mismatches,
        "baseline_metrics": {
            "precision": float(
                precision_score(
                    y_true,
                    baseline_predictions,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_true,
                    baseline_predictions,
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_true,
                    baseline_predictions,
                    zero_division=0,
                )
            ),
            "true_positive": int(baseline_tp),
            "false_positive": int(baseline_fp),
            "true_negative": int(baseline_tn),
            "false_negative": int(baseline_fn),
        },
        "optimized_metrics": {
            "precision": float(
                precision_score(
                    y_true,
                    optimized_predictions,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_true,
                    optimized_predictions,
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_true,
                    optimized_predictions,
                    zero_division=0,
                )
            ),
            "true_positive": int(optimized_tp),
            "false_positive": int(optimized_fp),
            "true_negative": int(optimized_tn),
            "false_negative": int(optimized_fn),
        },
        "numerical_tolerance": TOLERANCE,
        "score_comparison_pass": (
            max_fusion_difference <= TOLERANCE
        ),
        "decision_comparison_pass": (
            decision_mismatches == 0
        ),
        "note": (
            "Validation sample for inference equivalence only. "
            "Not a final performance estimate."
        ),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\n=== VALIDATION SUMMARY ===")
    print(json.dumps(results, indent=2))
    print(f"\nSaved report: {RESULTS_PATH}")


if __name__ == "__main__":
    main()