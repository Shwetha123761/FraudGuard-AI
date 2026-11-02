import json
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

from engine.hybrid_inference import HybridInference, FEATURES

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "validation_data.csv"
OUTPUT_PATH = ROOT / "results" / "gnn_hybrid_validation_results.json"


def main():
    df = pd.read_csv(DATA_PATH)
    y_true = df["Class"].astype(int).to_numpy()

    print(f"Validation transactions: {len(df):,}")
    print(f"Fraud cases: {int(y_true.sum()):,}")
    print("Loading hybrid model...")

    engine = HybridInference()
    scores = []

    start = time.perf_counter()

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        transaction = {name: float(row[name]) for name in FEATURES}
        prediction = engine.predict(transaction)
        scores.append(float(prediction["risk_score"]))

        if i % 500 == 0 or i == len(df):
            elapsed = time.perf_counter() - start
            print(f"Processed {i:,}/{len(df):,} in {elapsed:.1f}s")

    scores = np.asarray(scores, dtype=float)

    if not np.isfinite(scores).all():
        raise ValueError("Non-finite hybrid scores detected.")

    # Choose a threshold using validation labels only.
    # Among thresholds with FPR <= 0.5%, choose the one
    # with the highest F1-score.
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    best = None

    for threshold in candidates:
        y_pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if fpr <= 0.005:
            candidate = (f1, recall_score(y_true, y_pred, zero_division=0),
                         -threshold, threshold, tn, fp, fn, tp, fpr)

            if best is None or candidate[:3] > best[:3]:
                best = candidate

    if best is None:
        raise RuntimeError("No threshold satisfied the FPR constraint.")

    _, _, _, threshold, tn, fp, fn, tp, fpr = best
    y_pred = (scores >= threshold).astype(int)

    result = {
        "model": "Autoencoder + feature-similarity GNN prototype",
        "partition": "validation",
        "rows": int(len(df)),
        "fraud_cases": int(y_true.sum()),
        "legitimate_cases": int((y_true == 0).sum()),
        "threshold": float(threshold),
        "threshold_selection": (
            "Selected on validation data to maximize F1 subject to FPR <= 0.005."
        ),
        "test_used_for_training_or_threshold_selection": False,
        "score_note": "Prototype fusion score; not a calibrated fraud probability.",
        "metrics": {
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, scores)),
            "average_precision": float(average_precision_score(y_true, scores)),
            "false_positive_rate": float(fpr),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "runtime_seconds": round(time.perf_counter() - start, 2),
        "limitations": [
            "The graph represents feature similarity, not real card/account relationships.",
            "The score is not a calibrated probability.",
            "Threshold selection and reported metrics use the same validation set, so these metrics are selection results, not an unbiased final estimate.",
        ],
    }

    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n===== HYBRID VALIDATION RESULTS =====")
    print(json.dumps(result, indent=2))
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
