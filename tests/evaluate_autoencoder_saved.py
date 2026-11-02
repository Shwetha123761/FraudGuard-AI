import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "weights" / "autoencoder.keras"
VAL_PATH = ROOT / "data" / "processed" / "validation_data.csv"
TEST_PATH = ROOT / "data" / "processed" / "test_data.csv"
RESULT_PATH = ROOT / "results" / "autoencoder_results.json"

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
FPR_TARGET = 0.005
RECALL_TARGET = 0.80


def load_data(path):
    df = pd.read_csv(path)
    missing = set(FEATURES + ["Class"]) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    if df[FEATURES + ["Class"]].isna().any().any():
        raise ValueError(f"{path.name} contains missing values")
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df["Class"].to_numpy(dtype=np.int32)
    return X, y


def reconstruction_scores(model, X, batch_size=1024):
    reconstructed = model.predict(X, batch_size=batch_size, verbose=0)
    return np.mean(np.square(X - reconstructed), axis=1)


def choose_threshold(y, scores):
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_y = y[order]

    ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(y) - 1]
    thresholds = sorted_scores[ends]
    tp = np.cumsum(sorted_y)[ends]
    fp = np.cumsum(1 - sorted_y)[ends]

    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Validation set must contain both classes.")

    fn = positives - tp
    tn = negatives - fp
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    fpr = fp / negatives
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-15)

    eligible = np.flatnonzero(fpr <= FPR_TARGET)
    if len(eligible) == 0:
        best = min(range(len(thresholds)),
                   key=lambda i: (fpr[i], -recall[i], -thresholds[i]))
        status = "fpr_target_not_met"
    else:
        candidates = eligible[recall[eligible] >= RECALL_TARGET]
        if len(candidates):
            status = "both_validation_targets_met"
        else:
            candidates = eligible
            status = "recall_target_not_met"

        best = max(candidates, key=lambda i: (
            precision[i], recall[i], f1[i], thresholds[i]
        ))

    metrics = {
        "threshold": float(thresholds[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "f1": float(f1[best]),
        "false_positive_rate": float(fpr[best]),
        "tn": int(tn[best]),
        "fp": int(fp[best]),
        "fn": int(fn[best]),
        "tp": int(tp[best]),
    }
    return metrics, status


def evaluate(y, scores, threshold):
    predictions = (scores >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc_average_precision": float(average_precision_score(y, scores)),
    }


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")

    print("Loading saved Autoencoder; training will NOT run.")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    X_val, y_val = load_data(VAL_PATH)
    X_test, y_test = load_data(TEST_PATH)

    print(f"Validation rows: {len(y_val)} | Fraud: {int(y_val.sum())}")
    print(f"Test rows: {len(y_test)} | Fraud: {int(y_test.sum())}")

    val_scores = reconstruction_scores(model, X_val)
    val_metrics, status = choose_threshold(y_val, val_scores)
    threshold = val_metrics["threshold"]

    print(f"Validation threshold status: {status}")
    print("Evaluating held-out test data with the validation-selected threshold.")
    test_scores = reconstruction_scores(model, X_test)
    test_metrics = evaluate(y_test, test_scores, threshold)

    results = {
        "model": "Autoencoder reconstruction-error baseline",
        "model_path": str(MODEL_PATH),
        "training_performed_during_evaluation": False,
        "validation_status": status,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "feature_columns": FEATURES,
        "threshold_selection": (
            f"Selected on validation data with FPR target <= {FPR_TARGET} "
            f"and recall target >= {RECALL_TARGET} where feasible."
        ),
        "caveat": (
            "Chronological holdout has few fraud cases. Metrics are uncertain. "
            "This evaluates the Autoencoder baseline only, not the integrated "
            "Autoencoder + GNN + rules hybrid."
        ),
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nHELD-OUT TEST RESULTS")
    print(json.dumps(test_metrics, indent=2))
    print(f"\nSaved results to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
