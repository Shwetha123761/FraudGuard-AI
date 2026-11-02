import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
WEIGHTS = ROOT / "models" / "weights"
RESULTS = ROOT / "results"

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
LABEL = "Class"
MAX_VALIDATION_FPR = 0.005
RANDOM_STATE = 42


def load_partition(filename):
    path = DATA / filename
    if not path.exists():
        raise FileNotFoundError(f"Required dataset not found: {path}")

    frame = pd.read_csv(path)

    required = set(FEATURES + [LABEL])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{filename} is missing columns: {missing}")

    if frame[FEATURES + [LABEL]].isna().any().any():
        raise ValueError(f"{filename} contains missing model values.")

    if not np.isfinite(frame[FEATURES + [LABEL]].to_numpy(dtype=float)).all():
        raise ValueError(f"{filename} contains non-finite values.")

    if not set(frame[LABEL].unique()).issubset({0, 1}):
        raise ValueError(f"{filename} has unexpected Class labels.")

    return frame


def metrics_at_threshold(y_true, scores, threshold):
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(
            y_true, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, predictions, zero_division=0
        )),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def main():
    print("Loading train and validation partitions...")
    train = load_partition("train_data.csv")
    validation = load_partition("validation_data.csv")

    X_train = train[FEATURES].astype(np.float64)
    y_train = train[LABEL].astype(int)
    X_val = validation[FEATURES].astype(np.float64)
    y_val = validation[LABEL].astype(int)

    print(f"Training rows: {len(train):,}")
    print(f"Training frauds: {int(y_train.sum()):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Validation frauds: {int(y_val.sum()):,}")

    if y_train.nunique() != 2 or y_val.nunique() != 2:
        raise ValueError("Both partitions must contain legitimate and fraudulent rows.")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=300,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )),
    ])

    print("\nTraining Logistic Regression...")
    model.fit(X_train, y_train)

    val_scores = model.predict_proba(X_val)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_val, val_scores)
    eligible = np.flatnonzero(fpr <= MAX_VALIDATION_FPR)

    if len(eligible) == 0:
        raise RuntimeError("No threshold satisfies the validation FPR constraint.")

    best_recall = float(np.max(tpr[eligible]))
    candidates = eligible[np.isclose(tpr[eligible], best_recall)]
    best_index = int(candidates[np.argmax(thresholds[candidates])])
    threshold = float(thresholds[best_index])

    validation_metrics = metrics_at_threshold(y_val, val_scores, threshold)
    validation_metrics.update({
        "roc_auc": float(roc_auc_score(y_val, val_scores)),
        "average_precision": float(average_precision_score(y_val, val_scores)),
        "target_max_fpr": MAX_VALIDATION_FPR,
        "threshold_selection": (
            "Maximum validation recall subject to FPR <= 0.005; "
            "ties resolved using the highest threshold."
        ),
        "validation_rows": int(len(validation)),
        "validation_frauds": int(y_val.sum()),
    })

    bundle = {
        "model": model,
        "features": FEATURES,
        "threshold": threshold,
        "model_name": "class_weighted_logistic_regression",
        "random_state": RANDOM_STATE,
        "validation_metrics": validation_metrics,
    }

    model_path = WEIGHTS / "supervised_model.joblib"
    metrics_path = RESULTS / "supervised_validation_results.json"

    joblib.dump(bundle, model_path)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(validation_metrics, file, indent=2)

    print("\nVALIDATION RESULTS")
    print(json.dumps(validation_metrics, indent=2))
    print(f"\nSaved model: {model_path}")
    print(f"Saved validation metrics: {metrics_path}")
    print("\nTraining and threshold selection completed.")
    print("The held-out test dataset was not used.")


if __name__ == "__main__":
    main()
