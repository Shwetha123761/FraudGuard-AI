
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.hybrid_inference import HybridInference as OptimizedEngine


BACKUP_PATH = ROOT / "engine" / "hybrid_inference_backup.py"
VALIDATION_PATH = ROOT / "data" / "processed" / "validation_data.csv"
RESULTS_PATH = ROOT / "results" / "optimized_engine_comparison.json"


def load_backup_engine():
    if not BACKUP_PATH.exists():
        raise FileNotFoundError(f"Backup not found: {BACKUP_PATH}")

    spec = importlib.util.spec_from_file_location(
        "original_hybrid_inference",
        BACKUP_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HybridInference


def main():
    print("=" * 68)
    print("FRAUDGUARD AI - ORIGINAL VS OPTIMIZED ENGINE")
    print("=" * 68)

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(f"Validation dataset not found: {VALIDATION_PATH}")

    validation = pd.read_csv(VALIDATION_PATH)

    # Include every validation fraud and 248 legitimate transactions.
    fraud = validation[validation["Class"] == 1]
    legitimate = validation[validation["Class"] == 0].sample(
        n=min(248, int((validation["Class"] == 0).sum())),
        random_state=42,
    )

    sample = pd.concat([fraud, legitimate]).sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    features = [f"V{i}" for i in range(1, 29)] + ["Amount"]

    print(f"Transactions to compare: {len(sample)}")
    print(f"Fraud transactions: {int(sample['Class'].sum())}")
    print("Loading original and optimized engines...")

    BackupEngine = load_backup_engine()

    start = time.perf_counter()
    original = BackupEngine()
    original_init = time.perf_counter() - start

    start = time.perf_counter()
    optimized = OptimizedEngine()
    optimized_init = time.perf_counter() - start

    print(f"Original engine initialization: {original_init:.2f}s")
    print(f"Optimized engine initialization: {optimized_init:.2f}s")
    print("\nComparing predictions...")

    original_scores = []
    optimized_scores = []
    original_gnn_scores = []
    optimized_gnn_scores = []
    original_decisions = []
    optimized_decisions = []

    original_times = []
    optimized_times = []

    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        transaction = {name: float(row[name]) for name in features}

        start = time.perf_counter()
        old_result = original.predict(transaction)
        original_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        new_result = optimized.predict(transaction)
        optimized_times.append(time.perf_counter() - start)

        original_scores.append(float(old_result["risk_score"]))
        optimized_scores.append(float(new_result["risk_score"]))

        original_gnn_scores.append(
            float(old_result["breakdown"]["gnn_fraud_score"])
        )
        optimized_gnn_scores.append(
            float(new_result["breakdown"]["gnn_fraud_score"])
        )

        original_decisions.append(old_result["decision"])
        optimized_decisions.append(new_result["decision"])

        if i % 50 == 0 or i == len(sample):
            print(f"Compared {i}/{len(sample)} transactions")

    original_scores = np.asarray(original_scores)
    optimized_scores = np.asarray(optimized_scores)
    original_gnn_scores = np.asarray(original_gnn_scores)
    optimized_gnn_scores = np.asarray(optimized_gnn_scores)

    fusion_diff = np.abs(original_scores - optimized_scores)
    gnn_diff = np.abs(original_gnn_scores - optimized_gnn_scores)

    decision_mismatches = sum(
        old != new
        for old, new in zip(original_decisions, optimized_decisions)
    )

    report = {
        "transactions_compared": len(sample),
        "fraud_transactions": int(sample["Class"].sum()),
        "original_initialization_seconds": original_init,
        "optimized_initialization_seconds": optimized_init,
        "original_mean_latency_seconds": float(np.mean(original_times)),
        "optimized_mean_latency_seconds": float(np.mean(optimized_times)),
        "max_absolute_fusion_score_difference": float(fusion_diff.max()),
        "max_absolute_gnn_score_difference": float(gnn_diff.max()),
        "mean_absolute_fusion_score_difference": float(fusion_diff.mean()),
        "decision_mismatches": int(decision_mismatches),
        "decision_mismatch_count": int(decision_mismatches),
        "tolerance": 2e-5,
        "score_comparison_pass": bool(
            fusion_diff.max() <= 2e-5 and gnn_diff.max() <= 2e-5
        ),
        "decision_comparison_pass": decision_mismatches == 0,
    }

    report["overall_pass"] = (
        report["score_comparison_pass"]
        and report["decision_comparison_pass"]
    )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 68)
    print("COMPARISON RESULTS")
    print("=" * 68)
    print(f"Maximum fusion score difference: {fusion_diff.max():.10f}")
    print(f"Maximum GNN score difference: {gnn_diff.max():.10f}")
    print(f"Decision mismatches: {decision_mismatches}/{len(sample)}")
    print(f"Original mean latency: {np.mean(original_times):.5f}s")
    print(f"Optimized mean latency: {np.mean(optimized_times):.5f}s")
    print(f"Score comparison: {'PASS' if report['score_comparison_pass'] else 'FAIL'}")
    print(f"Decision comparison: {'PASS' if report['decision_comparison_pass'] else 'FAIL'}")
    print(f"OVERALL: {'PASS' if report['overall_pass'] else 'FAIL'}")
    print(f"\nReport saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()

