
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
BATCH_SIZE = 2048

DATASETS = {
    "validation": ROOT / "data" / "processed" / "validation_data.csv",
    "test": ROOT / "data" / "processed" / "test_data.csv",
}

AE_PATH = ROOT / "models" / "weights" / "autoencoder.keras"
LR_PATH = ROOT / "models" / "weights" / "supervised_model.joblib"
RF_PATH = ROOT / "models" / "weights" / "random_forest_proper.joblib"

AE_RESULTS = ROOT / "results" / "autoencoder_results.json"
FUSION_CONFIG = ROOT / "models" / "weights" / "fusion_config.json"

OUTPUT_DIR = ROOT / "results" / "scores"


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def verify_artifacts():
    required = [
        AE_PATH,
        LR_PATH,
        RF_PATH,
        AE_RESULTS,
        FUSION_CONFIG,
        *DATASETS.values(),
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    ae_results = load_json(AE_RESULTS)
    fusion = load_json(FUSION_CONFIG)

    lr_bundle = joblib.load(LR_PATH)
    rf_bundle = joblib.load(RF_PATH)

    if list(lr_bundle["features"]) != FEATURES:
        raise ValueError("LR feature order mismatch.")

    if list(rf_bundle["features"]) != FEATURES:
        raise ValueError("RF feature order mismatch.")

    if fusion.get("selected_on") != "validation_data_only":
        raise ValueError("Fusion configuration is not validation-only.")

    ae_threshold = float(
        ae_results["validation_metrics"]["threshold"]
    )
    lr_threshold = float(lr_bundle["threshold"])
    rf_threshold = float(rf_bundle["threshold"])
    alpha = float(fusion["alpha"])
    fusion_threshold = float(fusion["threshold"])

    if not np.isclose(
        float(fusion["autoencoder_threshold"]), ae_threshold
    ):
        raise ValueError("Fusion AE threshold mismatch.")

    if not np.isclose(
        float(fusion["supervised_model_threshold_reference"]),
        lr_threshold,
    ):
        raise ValueError("Fusion LR threshold mismatch.")

    if rf_bundle["threshold_selection_method"] != "validation_only":
        raise ValueError("RF threshold was not selected on validation.")

    return {
        "lr_bundle": lr_bundle,
        "rf_bundle": rf_bundle,
        "ae_threshold": ae_threshold,
        "lr_threshold": lr_threshold,
        "rf_threshold": rf_threshold,
        "alpha": alpha,
        "fusion_threshold": fusion_threshold,
    }


def calculate_ae_errors(model, values):
    errors = []

    for start in range(0, len(values), BATCH_SIZE):
        batch = values[start:start + BATCH_SIZE]

        reconstructed = model.predict(batch, verbose=0)

        batch_errors = np.mean(
            np.square(batch - reconstructed),
            axis=1,
        )

        errors.extend(batch_errors.tolist())

    return np.asarray(errors, dtype=np.float64)


def score_dataset(
    name,
    data_path,
    ae,
    artifacts,
):
    print(f"\nScoring {name} dataset...")

    df = pd.read_csv(data_path)

    required_columns = ["Time", "Class"] + FEATURES
    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} dataset missing columns: {missing}"
        )

    if df.empty:
        raise ValueError(f"{name} dataset is empty.")

    X = df[FEATURES].astype(np.float32)
    values = X.to_numpy(dtype=np.float32)
    y = df["Class"].astype(int).to_numpy()

    if not np.isin(y, [0, 1]).all():
        raise ValueError(f"{name} dataset has invalid Class labels.")

    if len(np.unique(y)) != 2:
        raise ValueError(
            f"{name} dataset must contain both classes."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            f"{name} dataset contains non-finite feature values."
        )

    print(f"Rows: {len(df):,}")
    print(f"Fraud cases: {int(y.sum()):,}")

    # 1. Autoencoder reconstruction error
    print("Calculating AE reconstruction errors...")

    ae_errors = calculate_ae_errors(ae, values)

    # 2. AE anomaly signal
    anomaly_signal = np.log1p(
        np.maximum(ae_errors, 0.0)
        / artifacts["ae_threshold"]
    )

    # 3. Logistic Regression scores
    print("Calculating LR fraud scores...")

    lr_scores = artifacts["lr_bundle"]["model"].predict_proba(X)[:, 1]

    # 4. Random Forest scores
    print("Calculating RF fraud scores...")

    rf_scores = artifacts["rf_bundle"]["model"].predict_proba(X)[:, 1]

    # 5. AE + LR fusion
    fusion_scores = (
        lr_scores + artifacts["alpha"] * anomaly_signal
    )

    result = pd.DataFrame({
        "Time": df["Time"].to_numpy(),
        "Class": y,
        "AE": ae_errors,
        "AE_Anomaly": anomaly_signal,
        "LR": lr_scores,
        "RF": rf_scores,
        "AE_LR_Fusion": fusion_scores,
    })

    score_columns = [
        "AE",
        "AE_Anomaly",
        "LR",
        "RF",
        "AE_LR_Fusion",
    ]

    if not np.isfinite(
        result[score_columns].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(
            f"{name} scores contain non-finite values."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / f"{name}_scores.csv"
    result.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Output rows: {len(result):,}")

    return result


def main():
    print("Verifying saved model artifacts...")

    artifacts = verify_artifacts()

    print("Loading saved Autoencoder...")

    ae = tf.keras.models.load_model(
        AE_PATH,
        compile=False,
    )

    print("\nSaved thresholds (unchanged):")
    print(f"AE: {artifacts['ae_threshold']:.12f}")
    print(f"LR: {artifacts['lr_threshold']:.12f}")
    print(f"RF: {artifacts['rf_threshold']:.12f}")
    print(f"AE+LR Fusion: {artifacts['fusion_threshold']:.12f}")
    print(f"Fusion alpha: {artifacts['alpha']}")

    # Generate validation scores first, then test scores.
    for name, data_path in DATASETS.items():
        score_dataset(
            name=name,
            data_path=data_path,
            ae=ae,
            artifacts=artifacts,
        )

    print("\n=== SCORE GENERATION COMPLETED ===")
    print("Validation: results/scores/validation_scores.csv")
    print("Test:       results/scores/test_scores.csv")
    print("No model training or threshold tuning was performed.")


if __name__ == "__main__":
    main()
