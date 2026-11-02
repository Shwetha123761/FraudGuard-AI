"""
FraudGuard AI: efficient validation-only hybrid fusion threshold analysis.

Computes live-style scores for all validation rows, then evaluates thresholds
using vectorized sorting/cumulative counts (rather than repeatedly scanning the
entire dataset for every threshold).

Does not train models or modify model weights/configuration.
Writes: results/hybrid_threshold_tuning_validation.json

Run from project root:
    python .\scripts\tune_live_fusion_threshold.py
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
from sklearn.metrics import average_precision_score, roc_auc_score

from engine.hybrid_inference import HybridInference
from scripts.benchmark_local_gnn import LocalGCNExperiment

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
DATA_PATH = ROOT / "data" / "processed" / "validation_data.csv"
OUTPUT_PATH = ROOT / "results" / "hybrid_threshold_tuning_validation.json"

MAX_ALLOWED_FPR = 0.005
CURRENT_THRESHOLD = 0.5
AE_BATCH_SIZE = 512
PROGRESS_EVERY = 5000


def metrics_at_threshold(y, scores, threshold):
    pred = scores >= threshold
    positives = y == 1
    negatives = ~positives
    tp = int(np.count_nonzero(pred & positives))
    fp = int(np.count_nonzero(pred & negatives))
    fn = int(np.count_nonzero(~pred & positives))
    tn = int(np.count_nonzero(~pred & negatives))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fpr),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "flagged": tp + fp,
        "missed_fraud": fn,
    }


def main():
    torch.set_num_threads(1)
    print("=" * 76)
    print("FRAUDGUARD AI - EFFICIENT VALIDATION-ONLY THRESHOLD ANALYSIS")
    print("=" * 76)
    print("No model training or configuration changes will be made.")

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df["Class"].to_numpy(dtype=np.int32)
    n = len(y)

    print(f"Validation transactions: {n:,}")
    print(f"Fraud cases: {int(y.sum()):,}")
    print(f"Legitimate cases: {int((y == 0).sum()):,}")
    print(f"Maximum experimental FPR constraint: {MAX_ALLOWED_FPR:.3%}")

    t0 = time.perf_counter()
    engine = HybridInference()
    experiment = LocalGCNExperiment(engine)
    print(f"Engine and local GCN setup: {time.perf_counter() - t0:.2f}s")

    print("\nComputing Autoencoder reconstruction errors...")
    t0 = time.perf_counter()
    reconstructed = engine.autoencoder.predict(X, batch_size=AE_BATCH_SIZE, verbose=0)
    errors = np.mean(np.square(X - reconstructed), axis=1)
    anomaly = np.log1p(np.maximum(errors, 0.0) / engine.ae_threshold)
    print(f"Autoencoder: {time.perf_counter() - t0:.2f}s")

    print("\nScaling and finding reference neighbors...")
    t0 = time.perf_counter()
    scaled = (X - engine.scaler_mean) / engine.scaler_scale
    _, neighbors = engine.neighbor_search.kneighbors(scaled)
    print(f"Neighbor search: {time.perf_counter() - t0:.2f}s")

    print("\nComputing GNN probabilities...")
    gnn_probs = np.empty(n, dtype=np.float64)
    t0 = time.perf_counter()
    for i in range(n):
        gnn_probs[i] = experiment.fraud_probability(scaled[i:i + 1], neighbors[i])
        if (i + 1) % PROGRESS_EVERY == 0 or i + 1 == n:
            elapsed = time.perf_counter() - t0
            print(f"Completed {i + 1:,}/{n:,}; elapsed {elapsed:.1f}s; "
                  f"average {elapsed / (i + 1):.5f}s/row")

    # Exactly the current live-style fusion formula.
    scores = 0.5 * gnn_probs + 0.5 * np.minimum(anomaly / 3.0, 1.0)
    y_bool = y == 1
    n_pos = int(y_bool.sum())
    n_neg = n - n_pos

    print("\nScoring complete.")
    print(f"ROC-AUC: {roc_auc_score(y, scores):.6f}")
    print(f"Average Precision: {average_precision_score(y, scores):.6f}")

    current = metrics_at_threshold(y, scores, CURRENT_THRESHOLD)

    # Sort scores descending once. A candidate threshold at a distinct score
    # includes every row tied at that score, as required by score >= threshold.
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_y = y_bool[order]
    cumulative_tp = np.cumsum(sorted_y, dtype=np.int64)
    cumulative_fp = np.cumsum(~sorted_y, dtype=np.int64)

    # End index of each equal-score group; thresholds at these scores correspond
    # to predictions including the entire tied group.
    group_ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), n - 1]
    thresholds = sorted_scores[group_ends]
    tp = cumulative_tp[group_ends]
    fp = cumulative_fp[group_ends]
    fn = n_pos - tp
    tn = n_neg - fp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall = tp / n_pos if n_pos else np.zeros_like(tp, dtype=float)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) != 0)
    fpr = fp / n_neg if n_neg else np.zeros_like(fp, dtype=float)

    # Include threshold just above the maximum score (no positive predictions),
    # current threshold, and threshold 0 for a complete operational comparison.
    no_positive_threshold = float(np.nextafter(float(np.max(scores)), np.inf))
    extra_thresholds = [no_positive_threshold, CURRENT_THRESHOLD, 0.0, 1.0]
    extra_rows = [metrics_at_threshold(y, scores, t) for t in extra_thresholds]

    eligible = np.flatnonzero(fpr <= MAX_ALLOWED_FPR)
    if len(eligible) == 0:
        candidate_rows = extra_rows
        eligible_extra = [r for r in extra_rows if r["false_positive_rate"] <= MAX_ALLOWED_FPR]
        if not eligible_extra:
            raise RuntimeError("No threshold met the requested FPR constraint.")
        best = sorted(eligible_extra, key=lambda r: (r["recall"], r["precision"], r["f1"], -r["fp"], r["threshold"]), reverse=True)[0]
    else:
        max_recall = float(np.max(recall[eligible]))
        best_indices = eligible[np.isclose(recall[eligible], max_recall)]
        best_idx = sorted(
            best_indices.tolist(),
            key=lambda i: (precision[i], f1[i], -int(fp[i]), float(thresholds[i])),
            reverse=True,
        )[0]
        best = {
            "threshold": float(thresholds[best_idx]),
            "precision": float(precision[best_idx]),
            "recall": float(recall[best_idx]),
            "f1": float(f1[best_idx]),
            "false_positive_rate": float(fpr[best_idx]),
            "tp": int(tp[best_idx]), "fp": int(fp[best_idx]),
            "tn": int(tn[best_idx]), "fn": int(fn[best_idx]),
            "flagged": int(tp[best_idx] + fp[best_idx]),
            "missed_fraud": int(fn[best_idx]),
        }

    print("\n" + "=" * 76)
    print("CURRENT THRESHOLD VS SELECTED VALIDATION THRESHOLD")
    print("=" * 76)
    print(f"Current threshold: {CURRENT_THRESHOLD:.8f}")
    print(f"  Precision={current['precision']:.4f} | Recall={current['recall']:.4f} | "
          f"F1={current['f1']:.4f} | FPR={current['false_positive_rate']:.4%} | "
          f"TP={current['tp']} FP={current['fp']} TN={current['tn']} FN={current['fn']}")
    print(f"\nBest threshold under FPR <= {MAX_ALLOWED_FPR:.3%}: {best['threshold']:.10f}")
    print(f"  Precision={best['precision']:.4f} | Recall={best['recall']:.4f} | "
          f"F1={best['f1']:.4f} | FPR={best['false_positive_rate']:.4%} | "
          f"TP={best['tp']} FP={best['fp']} TN={best['tn']} FN={best['fn']}")

    print("\nReadable threshold grid:")
    print("threshold     precision  recall     F1         FPR        TP   FP   FN")
    grid = sorted(set(np.round(np.arange(0.0, 1.0001, 0.05), 4).tolist()
                      + [CURRENT_THRESHOLD, round(best["threshold"], 10)]))
    for threshold in grid:
        r = metrics_at_threshold(y, scores, threshold)
        marker = " <-- selected" if np.isclose(threshold, best["threshold"], rtol=0, atol=1e-9) else (
            " <-- current" if np.isclose(threshold, CURRENT_THRESHOLD) else "")
        print(f"{r['threshold']:<13.8f} {r['precision']:<10.4f} {r['recall']:<10.4f} "
              f"{r['f1']:<10.4f} {r['false_positive_rate']:<10.4%} "
              f"{r['tp']:<4} {r['fp']:<4} {r['fn']:<4}{marker}")

    report = {
        "evaluation": "validation_only_hybrid_threshold_tuning",
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "sampled": False,
        "transactions": int(n),
        "fraud_cases": n_pos,
        "legitimate_cases": n_neg,
        "fusion_formula": "0.5 * gnn_probability + 0.5 * min(log1p(reconstruction_error / ae_threshold) / 3, 1)",
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "current_threshold": current,
        "max_allowed_fpr_constraint": MAX_ALLOWED_FPR,
        "selection_objective": "maximize recall subject to FPR constraint; tie-break by precision, F1, fewer false positives, then higher threshold",
        "selected_threshold": best,
        "note": "Validation-only threshold selection. No model/configuration was modified. Freeze before final test evaluation.",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report: {OUTPUT_PATH}")
    print("Do not edit fusion_config.json yet. Review this report first.")


if __name__ == "__main__":
    main()
