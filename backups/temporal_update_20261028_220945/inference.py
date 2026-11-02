from pathlib import Path
import json
import sys

import joblib
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import AUTOENCODER_WEIGHTS

ROOT = Path(__file__).resolve().parent.parent
SUPERVISED_WEIGHTS = ROOT / "models" / "weights" / "supervised_model.joblib"
FUSION_CONFIG_PATH = ROOT / "models" / "weights" / "fusion_config.json"

AUTOENCODER_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]


class FraudGuardInferenceEngine:
    """Fraud detection using Autoencoder + Logistic Regression fusion."""

    def __init__(self):
        print("[*] Initializing FraudGuard inference engine...")

        self.autoencoder = None
        self.supervised_bundle = None
        self.fusion_config = None
        self.model_status = {}

        self._load_autoencoder()
        self._load_supervised_model()
        self._load_fusion_config()

        # Velocity rules remain available as a separate operational signal.
        # Their effectiveness has not been evaluated on the current dataset.
        self.model_status["gnn"] = "not_integrated"

    def _load_autoencoder(self):
        if not AUTOENCODER_WEIGHTS.exists():
            self.model_status["autoencoder"] = "missing"
            return

        try:
            self.autoencoder = tf.keras.models.load_model(
                AUTOENCODER_WEIGHTS,
                compile=False,
            )
            self.model_status["autoencoder"] = "loaded"
            print("[+] Autoencoder loaded.")
        except Exception as exc:
            self.model_status["autoencoder"] = "load_failed"
            print(f"[!] Autoencoder loading failed: {exc}")

    def _load_supervised_model(self):
        if not SUPERVISED_WEIGHTS.exists():
            self.model_status["supervised_model"] = "missing"
            return

        try:
            bundle = joblib.load(SUPERVISED_WEIGHTS)
            required = {"model", "features", "threshold"}

            if not required.issubset(bundle):
                raise ValueError(
                    "Supervised model artifact is missing required fields."
                )

            if list(bundle["features"]) != AUTOENCODER_FEATURES:
                raise ValueError(
                    "Supervised model feature order does not match API features."
                )

            self.supervised_bundle = bundle
            self.model_status["supervised_model"] = "loaded"
            print("[+] Supervised Logistic Regression model loaded.")
        except Exception as exc:
            self.model_status["supervised_model"] = "load_failed"
            print(f"[!] Supervised model loading failed: {exc}")

    def _load_fusion_config(self):
        if not FUSION_CONFIG_PATH.exists():
            self.model_status["fusion_config"] = "missing"
            return

        try:
            with FUSION_CONFIG_PATH.open("r", encoding="utf-8") as file:
                config = json.load(file)

            required = {
                "fusion_method",
                "alpha",
                "threshold",
                "autoencoder_threshold",
                "selected_on",
            }
            if not required.issubset(config):
                raise ValueError("Fusion configuration is missing required fields.")

            if config["fusion_method"] != (
                "logistic_probability_plus_weighted_log_anomaly"
            ):
                raise ValueError("Unsupported fusion method.")

            if config["selected_on"] != "validation_data_only":
                raise ValueError(
                    "Fusion config must identify validation-only selection."
                )

            alpha = float(config["alpha"])
            threshold = float(config["threshold"])
            ae_threshold = float(config["autoencoder_threshold"])

            if not np.isfinite([alpha, threshold, ae_threshold]).all():
                raise ValueError("Fusion configuration values must be finite.")

            if alpha < 0 or ae_threshold <= 0:
                raise ValueError("Fusion weight or Autoencoder threshold is invalid.")

            if threshold < 0:
                raise ValueError("Fusion threshold cannot be negative.")

            self.fusion_config = config
            self.model_status["fusion_config"] = "loaded"
            print(
                "[+] Fusion configuration loaded "
                f"(alpha={alpha}, threshold={threshold})."
            )
        except Exception as exc:
            self.model_status["fusion_config"] = "load_failed"
            print(f"[!] Fusion configuration loading failed: {exc}")

    def health(self):
        active = []

        if self.autoencoder is not None:
            active.append("autoencoder")
        if self.supervised_bundle is not None:
            active.append("supervised_model")
        if self.fusion_config is not None:
            active.append("fusion_scoring")
        active.append("velocity_rules")

        return {
            "autoencoder": self.model_status.get("autoencoder"),
            "supervised_model": self.model_status.get("supervised_model"),
            "fusion_config": self.model_status.get("fusion_config"),
            "gnn": self.model_status.get("gnn"),
            "active_detection_components": active,
        }

    def evaluate_transaction(
        self,
        transaction_features: dict,
        velocity_metrics: dict,
    ) -> dict:
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder is unavailable.")
        if self.supervised_bundle is None:
            raise RuntimeError("Supervised model is unavailable.")
        if self.fusion_config is None:
            raise RuntimeError("Fusion configuration is unavailable.")

        missing = [
            name for name in AUTOENCODER_FEATURES
            if name not in transaction_features
        ]
        if missing:
            raise ValueError(
                "Missing required model features: " + ", ".join(missing)
            )

        try:
            values = np.asarray(
                [float(transaction_features[name]) for name in AUTOENCODER_FEATURES],
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("All model features must be numeric.") from exc

        if not np.isfinite(values).all():
            raise ValueError("Transaction features must be finite.")

        model_input = values.reshape(1, -1).astype(np.float32)

        reconstructed = self.autoencoder.predict(model_input, verbose=0)
        if reconstructed.shape != model_input.shape:
            raise RuntimeError(
                f"Autoencoder output shape {reconstructed.shape} "
                f"does not match input shape {model_input.shape}."
            )

        reconstruction_error = float(
            np.mean(np.square(model_input - reconstructed))
        )

        config = self.fusion_config
        ae_threshold = float(config["autoencoder_threshold"])
        alpha = float(config["alpha"])
        fusion_threshold = float(config["threshold"])

        anomaly_flag = reconstruction_error >= ae_threshold

        bundle = self.supervised_bundle
        supervised_input = np.asarray(
            [[float(transaction_features[name]) for name in bundle["features"]]],
            dtype=np.float64,
        )
        supervised_probability = float(
            bundle["model"].predict_proba(supervised_input)[0, 1]
        )
        supervised_threshold = float(bundle["threshold"])
        supervised_flag = supervised_probability >= supervised_threshold

        # This is the exact fusion score evaluated on the held-out test set.
        anomaly_signal = float(
            np.log1p(max(reconstruction_error, 0.0) / ae_threshold)
        )
        fusion_score = float(supervised_probability + alpha * anomaly_signal)
        fusion_flag = fusion_score >= fusion_threshold

        velocity_1m = max(0, int(velocity_metrics.get("velocity_1m", 1)))
        velocity_1h = max(0, int(velocity_metrics.get("velocity_1h", 1)))

        risk_factors = []
        if anomaly_flag:
            risk_factors.append(
                "Autoencoder reconstruction error exceeds its threshold"
            )
        if supervised_flag:
            risk_factors.append(
                "Supervised fraud score meets its validation threshold"
            )
        if fusion_flag:
            risk_factors.append(
                "Combined fusion score meets its validation-selected threshold"
            )

        # Velocity remains a separate signal. Do not alter the fusion score
        # or fusion_flag with unvalidated velocity heuristics.
        velocity_alerts = []
        if velocity_1m > 3:
            velocity_alerts.append(
                "High transaction velocity within one minute"
            )
        if velocity_1h > 10:
            velocity_alerts.append(
                "High transaction velocity within one hour"
            )

        if not risk_factors:
            risk_factors.append("Fusion score is below its selected threshold")

        # The selected fusion threshold reproduces the binary classifier
        # evaluated in evaluate_fusion_test.py. OTP/block is an operational
        # policy layered on top and is not itself validated by dataset metrics.
        if fusion_flag:
            action = "OTP_VERIFY"
            risk_level = "MEDIUM"
        else:
            action = "APPROVE"
            risk_level = "LOW"

        return {
            "risk_score": round(fusion_score, 6),
            "action": action,
            "risk_level": risk_level,
            "breakdown": {
                "reconstruction_error": reconstruction_error,
                "anomaly_threshold": ae_threshold,
                "anomaly_flag": bool(anomaly_flag),
                "supervised_fraud_score": round(supervised_probability, 8),
                "supervised_threshold": supervised_threshold,
                "supervised_flag": bool(supervised_flag),
                "fusion_method": config["fusion_method"],
                "fusion_alpha": alpha,
                "fusion_threshold": fusion_threshold,
                "fusion_score": round(fusion_score, 8),
                "fusion_flag": bool(fusion_flag),
                "velocity_1m": velocity_1m,
                "velocity_1h": velocity_1h,
                "velocity_alerts": velocity_alerts,
                "risk_factors": risk_factors + velocity_alerts,
                "model_status": self.health(),
                "score_type": "fusion_ranking_score_not_calibrated_probability",
                "decision_policy_note": (
                    "Fusion threshold selected on validation data. "
                    "Velocity alerts are reported separately and do not alter "
                    "the evaluated fusion score."
                ),
            },
        }
