import json
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
from tensorflow.keras.models import load_model
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "test_data.csv"
AE_PATH = ROOT / "models" / "weights" / "autoencoder.keras"
LR_PATH = ROOT / "models" / "weights" / "supervised_model.joblib"
AE_RESULTS_PATH = ROOT / "results" / "autoencoder_results.json"
OUTPUT_PATH = ROOT / "results" / "hybrid_test_results.json"

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
BATCH_SIZE = 2048


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    for path in [DATA_PATH, AE_PATH, LR_PATH, AE_RESULTS_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    df = pd.read_csv(DATA_PATH)

    missing = [column for column in FEATURES + ["Class"] if column not in df.columns]
    if missing:
        raise ValueError(f"Test dataset is missing columns: {missing}")

    X = df[FEATURES].astype(np.float32)
    y = df["Class"].astype(int).to_numpy()

    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Class must contain only 0 and 1.")

    print(f"Test rows: {len(df):,}")
    print(f"Fraud cases: {int(y.sum()):,}")
    print("Loading trained models...")

    autoencoder = load_model(AE_PATH, compile=False)
    bundle = joblib.load(LR_PATH)

    lr_model = bundle["model"]
    lr_features = list(bundle["features"])
    lr_threshold = float(bundle["threshold"])

    if lr_features != FEATURES:
        raise ValueError(
            "Supervised model feature order does not match the evaluation features."
        )

    ae_results = load_json(AE_RESULTS_PATH)
    ae_threshold = float(ae_results["validation_metrics"]["threshold"])

    print(f"Autoencoder threshold: {ae_threshold:.10f}")
    print(f"Supervised threshold: {lr_threshold:.10f}")
    print("Calculating Autoencoder reconstruction errors...")

    errors = []
    values = X.to_numpy(dtype=np.float32)

    for start in range(0, len(values), BATCH_SIZE):
        batch = values[start:start + BATCH_SIZE]
        reconstructed = autoencoder.predict(batch, verbose=0)
        batch_errors = np.mean(np.square(batch - reconstructed), axis=1)
        errors.extend(batch_errors.tolist())

    reconstruction_error = np.asarray(errors, dtype=np.float64)
    anomaly_flag = reconstruction_error > ae_threshold

    print("Calculating supervised fraud probabilities...")
    supervised_probability = lr_model.predict_proba(X)[:, 1]
    supervised_flag = supervised_probability >= lr_threshold

    # Match the current inference engine's scoring formula.
    # Velocity values are neutral because the public test dataset has no card IDs
    # or transaction-history stream with which to calculate genuine velocity.
    velocity_1m = np.ones(len(df), dtype=np.float64)
    velocity_1h = np.ones(len(df), dtype=np.float64)

    supervised_component = 50.0 * supervised_probability
    anomaly_component = np.minimum(
        30.0,
        30.0 * reconstruction_error / ae_threshold,
    )
    velocity_component = (
        np.where(velocity_1m > 3, 12.0, 0.0)
        + np.where(velocity_1h > 10, 8.0, 0.0)
    )

    risk_score = np.minimum(
        100.0,
        supervised_component + anomaly_component + velocity_component,
    )

    # Match inference.py: either model flag raises a low score to at least 75.
    any_model_flag = anomaly_flag | supervised_flag
    risk_score = np.where(
        any_model_flag & (risk_score < 75.0),
        75.0,
        risk_score,
    )

    # Match current routing: <30 APPROVE, <=75 OTP_VERIFY, >75 BLOCK.
    decision = np.where(
        risk_score < 30.0,
        "APPROVE",
        np.where(risk_score <= 75.0, "OTP_VERIFY", "BLOCK"),
    )

    # For binary classification metrics, OTP_VERIFY and BLOCK are alerts.
    predicted_fraud = decision != "APPROVE"

    tn, fp, fn, tp = confusion_matrix(
        y, predicted_fraud.astype(int), labels=[0, 1]
    ).ravel()

    result = {
        "model": "autoencoder_plus_logistic_regression_operational_heuristic",
        "partition": "held_out_chronological_test",
        "rows": int(len(y)),
        "fraud_cases": int(y.sum()),
        "legitimate_cases": int((y == 0).sum()),
        "features": FEATURES,
        "autoencoder_threshold": ae_threshold,
        "supervised_threshold": lr_threshold,
        "threshold_selection": (
            "Thresholds loaded from prior validation results; "
            "test data was not used for training or threshold selection."
        ),
        "velocity_evaluation": (
            "Neutralized: the dataset lacks card IDs and transaction history. "
            "Velocity rules are not validated by these results."
        ),
        "scoring_note": (
            "Risk score is an operational heuristic, not a calibrated fraud "
            "probability. ROC-AUC and average precision for this score are "
            "exploratory ranking metrics."
        ),
        "decision_policy": {
            "approve": "risk_score < 30",
            "otp_verify": "30 <= risk_score <= 75",
            "block": "risk_score > 75",
            "binary_fraud_alert_metric": (
                "OTP_VERIFY or BLOCK is counted as a positive fraud alert."
            ),
        },
        "metrics": {
            "precision": float(precision_score(
                y, predicted_fraud, zero_division=0
            )),
            "recall": float(recall_score(
                y, predicted_fraud, zero_division=0
            )),
            "f1_score": float(f1_score(
                y, predicted_fraud, zero_division=0
            )),
            "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
            "roc_auc": float(roc_auc_score(y, risk_score)),
            "average_precision": float(average_precision_score(y, risk_score)),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "fraud_detected": int(tp),
            "fraud_missed": int(fn),
            "legitimate_transactions_flagged": int(fp),
        },
        "decision_counts": {
            str(name): int(count)
            for name, count in zip(
                *np.unique(decision, return_counts=True)
            )
        },
        "component_counts": {
            "autoencoder_flagged": int(anomaly_flag.sum()),
            "supervised_model_flagged": int(supervised_flag.sum()),
            "either_model_flagged": int(any_model_flag.sum()),
        },
        "test_used_for_training_or_threshold_selection": False,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print("\n=== HYBRID TEST RESULTS ===")
    print(json.dumps(result, indent=2))
    print(f"\nSaved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

