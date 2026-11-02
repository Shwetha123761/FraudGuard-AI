import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

import json
import random
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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "weights"
RESULTS_DIR = PROJECT_ROOT / "results"

TRAIN_PATH = DATA_DIR / "train_data.csv"
VALIDATION_PATH = DATA_DIR / "validation_data.csv"
TEST_PATH = DATA_DIR / "test_data.csv"

MODEL_PATH = MODEL_DIR / "autoencoder.keras"
RESULT_PATH = RESULTS_DIR / "autoencoder_results.json"

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]

EPOCHS = 40
BATCH_SIZE = 512
RECALL_TARGET = 0.80
FPR_TARGET = 0.005


def load_partition(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    missing = set(FEATURES + ["Class"]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df[FEATURES].isna().any().any():
        raise ValueError(f"Missing feature values in {path.name}")

    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df["Class"].to_numpy(dtype=np.int32)
    return X, y


def reconstruction_errors(model, X):
    reconstructed = model.predict(
        X, batch_size=2048, verbose=0
    )
    return np.mean(np.square(X - reconstructed), axis=1)


def calculate_metrics(y_true, scores, threshold):
    predictions = (scores >= threshold).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "false_positive_rate": float(fp / (fp + tn)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_threshold(y_true, scores):
    import numpy as np

    y_true = np.asarray(y_true, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)

    total_positive = int(np.sum(y_true))
    total_negative = len(y_true) - total_positive

    if total_positive == 0 or total_negative == 0:
        raise ValueError("Validation data must contain both classes.")

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_y = y_true[order]

    cumulative_tp = np.cumsum(sorted_y)
    cumulative_fp = np.cumsum(1 - sorted_y)

    group_ends = np.r_[
        np.flatnonzero(np.diff(sorted_scores) != 0),
        len(sorted_scores) - 1
    ]

    thresholds = sorted_scores[group_ends]
    tp = cumulative_tp[group_ends]
    fp = cumulative_fp[group_ends]
    fn = total_positive - tp
    tn = total_negative - fp

    precision = np.divide(
        tp, tp + fp,
        out=np.zeros(len(tp), dtype=float),
        where=(tp + fp) != 0
    )
    recall = tp / total_positive
    fpr = fp / total_negative

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(tp), dtype=float),
        where=(precision + recall) != 0
    )

    eligible = np.flatnonzero(fpr <= FPR_TARGET)

    if len(eligible) == 0:
        best = min(
            range(len(thresholds)),
            key=lambda i: (fpr[i], -recall[i], -thresholds[i])
        )
        status = "fpr_target_not_met"
    else:
        recall_eligible = eligible[recall[eligible] >= RECALL_TARGET]

        if len(recall_eligible) > 0:
            candidates = recall_eligible
            status = "both_validation_targets_met"
        else:
            candidates = eligible
            status = "recall_target_not_met"

        best = max(
            candidates,
            key=lambda i: (
                precision[i],
                recall[i],
                f1[i],
                thresholds[i]
            )
        )

    metrics = {
        "threshold": float(thresholds[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "f1": float(f1[best]),
        "fpr": float(fpr[best]),
        "tn": int(tn[best]),
        "fp": int(fp[best]),
        "fn": int(fn[best]),
        "tp": int(tp[best])
    }

    return metrics, status


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_partition(TRAIN_PATH)
    X_val, y_val = load_partition(VALIDATION_PATH)
    X_test, y_test = load_partition(TEST_PATH)

    print(f"Training rows: {len(y_train):,} | Fraud: {int(y_train.sum())}")
    print(f"Validation rows: {len(y_val):,} | Fraud: {int(y_val.sum())}")
    print(f"Test rows: {len(y_test):,} | Fraud: {int(y_test.sum())}")

    X_normal = X_train[y_train == 0]
    X_val_normal = X_val[y_val == 0]

    print(f"Legitimate training examples: {len(X_normal):,}")

    input_dim = X_normal.shape[1]

    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(24, activation="relu")(inputs)
    x = tf.keras.layers.Dense(12, activation="relu")(x)
    x = tf.keras.layers.Dense(6, activation="relu")(x)
    x = tf.keras.layers.Dense(12, activation="relu")(x)
    x = tf.keras.layers.Dense(24, activation="relu")(x)
    outputs = tf.keras.layers.Dense(input_dim)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    model.fit(
        X_normal,
        X_normal,
        validation_data=(X_val_normal, X_val_normal),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=[early_stopping],
        verbose=2,
    )

    model.save(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    val_scores = reconstruction_errors(model, X_val)
    validation_metrics, status = select_threshold(y_val, val_scores)
    threshold = validation_metrics["threshold"]

    print(f"Validation status: {status}")
    print("Validation metrics:")
    print(json.dumps(validation_metrics, indent=2))

    test_scores = reconstruction_errors(model, X_test)
    test_metrics = calculate_metrics(y_test, test_scores, threshold)
    test_metrics["roc_auc"] = float(
        roc_auc_score(y_test, test_scores)
    )
    test_metrics["pr_auc_average_precision"] = float(
        average_precision_score(y_test, test_scores)
    )

    results = {
        "model": "Autoencoder",
        "features": FEATURES,
        "training_strategy": "Legitimate training transactions only",
        "validation_status": status,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "epochs_completed": int(len(model.history.history["loss"]))
        if hasattr(model, "history") and model.history is not None
        else None,
        "model_path": str(MODEL_PATH),
        "note": (
            "Existing feature scaling provenance is unverified. "
            "Threshold was selected on validation data only."
        ),
    }

    with open(RESULT_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nHeld-out test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(f"Results saved to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
