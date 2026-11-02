import json
from pathlib import Path

import joblib
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

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "test_data.csv"
AE_PATH = ROOT / "models" / "weights" / "autoencoder.keras"
LR_PATH = ROOT / "models" / "weights" / "supervised_model.joblib"
AE_RESULTS = ROOT / "results" / "autoencoder_results.json"
FUSION_CONFIG = ROOT / "models" / "weights" / "fusion_config.json"
LR_TEST_RESULTS = ROOT / "results" / "supervised_test_results.json"
OUTPUT = ROOT / "results" / "fusion_test_results.json"

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
BATCH_SIZE = 2048


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def evaluate(y, scores, threshold):
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y, predicted, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1_score": float(f1_score(y, predicted, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if tn + fp else 0.0,
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fraud_detected": int(tp),
        "fraud_missed": int(fn),
        "legitimate_transactions_flagged": int(fp),
    }


def main():
    required = [DATA, AE_PATH, LR_PATH, AE_RESULTS, FUSION_CONFIG, LR_TEST_RESULTS]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    df = pd.read_csv(DATA)
    missing = [c for c in FEATURES + ["Class"] if c not in df.columns]
    if missing:
        raise ValueError(f"Test dataset missing columns: {missing}")

    X = df[FEATURES].astype(np.float32)
    y = df["Class"].astype(int).to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError("Test dataset must contain both classes.")

    print(f"Test rows: {len(df):,}")
    print(f"Fraud cases: {int(y.sum()):,}")
    print("Loading saved models and frozen fusion configuration...")

    ae = load_model(AE_PATH, compile=False)
    bundle = joblib.load(LR_PATH)
    if list(bundle["features"]) != FEATURES:
        raise ValueError("Supervised model feature order mismatch.")

    ae_results = load_json(AE_RESULTS)
    fusion = load_json(FUSION_CONFIG)
    lr_baseline = load_json(LR_TEST_RESULTS)

    ae_threshold = float(ae_results["validation_metrics"]["threshold"])
    lr_threshold = float(bundle["threshold"])
    alpha = float(fusion["alpha"])
    fusion_threshold = float(fusion["threshold"])

    if fusion.get("selected_on") != "validation_data_only":
        raise ValueError("Fusion configuration is not marked as validation-only.")

    if not np.isclose(
        float(fusion["autoencoder_threshold"]), ae_threshold
    ):
        raise ValueError("Fusion config Autoencoder threshold mismatch.")

    if not np.isclose(
        float(fusion["supervised_model_threshold_reference"]), lr_threshold
    ):
        raise ValueError("Fusion config supervised threshold reference mismatch.")

    values = X.to_numpy(dtype=np.float32)
    errors = []

    print("Calculating Autoencoder reconstruction errors...")
    for start in range(0, len(values), BATCH_SIZE):
        batch = values[start:start + BATCH_SIZE]
        reconstructed = ae.predict(batch, verbose=0)
        errors.extend(
            np.mean(np.square(batch - reconstructed), axis=1).tolist()
        )

    errors = np.asarray(errors, dtype=np.float64)

    print("Calculating Logistic Regression fraud scores...")
    probabilities = bundle["model"].predict_proba(X)[:, 1]

    anomaly_signal = np.log1p(np.maximum(errors, 0.0) / ae_threshold)
    fusion_scores = probabilities + alpha * anomaly_signal

    # The original Logistic Regression baseline uses its validation-selected
    # threshold. The fusion threshold is loaded unchanged from validation tuning.
    lr_scores_for_ranking = probabilities
    ae_scores_for_ranking = errors

    lr_metrics = evaluate(y, lr_scores_for_ranking, lr_threshold)
    ae_metrics = evaluate(y, ae_scores_for_ranking, ae_threshold)
    fusion_metrics = evaluate(y, fusion_scores, fusion_threshold)

    # Verify the saved LR test results against the loaded artifact and test data.
    saved_lr_threshold = float(lr_baseline["threshold_from_validation"])
    if not np.isclose(saved_lr_threshold, lr_threshold):
        raise ValueError("Saved LR test threshold differs from model artifact.")

    result = {
        "partition": "held_out_chronological_test",
        "rows": int(len(y)),
        "fraud_cases": int(y.sum()),
        "legitimate_cases": int((y == 0).sum()),
        "test_used_for_training_or_threshold_selection": False,
        "fusion_weight_and_threshold_source": "validation_data_only",
        "velocity_evaluation": (
            "Not evaluated: dataset lacks card IDs and transaction history."
        ),
        "fusion_score_note": (
            "Fusion score is not a calibrated fraud probability. ROC-AUC "
            "and average precision measure ranking and are exploratory."
        ),
        "autoencoder": ae_metrics,
        "logistic_regression": lr_metrics,
        "selected_fusion": {
            "method": fusion["fusion_method"],
            "alpha": alpha,
            "threshold": fusion_threshold,
            "metrics": fusion_metrics,
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== FROZEN FUSION TEST RESULTS ===")
    print(json.dumps(result, indent=2))
    print(f"\nSaved results to: {OUTPUT}")


if __name__ == "__main__":
    main()

