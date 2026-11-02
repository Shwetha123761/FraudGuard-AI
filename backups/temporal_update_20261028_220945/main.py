
from pathlib import Path
import sys
import time
import math

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.feature_store import RealTimeFeatureStore
from engine.inference import FraudGuardInferenceEngine
from engine.hybrid_inference import HybridInference, FEATURES


# ============================================================
# CONFIGURATION
# ============================================================

# Fixed project exchange-rate assumption.
# Change this value when you intentionally update the assumption.
INR_PER_EUR = 106.06

# Verified from the original and scaled dataset comparison:
# scaled Amount = original Amount / 25691.16
ORIGINAL_AMOUNT_MAX = 25691.16


app = FastAPI(
    title="FraudGuard AI",
    description=(
        "Credit-card fraud detection using Autoencoder, supervised "
        "learning, a feature-similarity GNN, and transaction-velocity rules."
    ),
    version="2.3.0",
)

feature_store = RealTimeFeatureStore()
inference_engine = FraudGuardInferenceEngine()

hybrid_engine = None
hybrid_error = None

try:
    hybrid_engine = HybridInference()
    print("[+] Hybrid Autoencoder + GNN inference loaded.")
except Exception as exc:
    hybrid_error = str(exc)
    print(f"[!] Hybrid inference unavailable: {exc}")


# ============================================================
# REQUEST VALIDATION
# ============================================================

class TransactionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    card_id: str = Field(..., min_length=1, max_length=100)
    amount: float = Field(..., ge=0)
    features: dict[str, float]
    timestamp: float = Field(default_factory=time.time)

    @field_validator("card_id")
    @classmethod
    def validate_card_id(cls, value):
        value = value.strip()

        if not value:
            raise ValueError("card_id must not be blank")

        return value

    @field_validator("timestamp", "amount")
    @classmethod
    def validate_finite_numbers(cls, value):
        if not math.isfinite(value):
            raise ValueError("Value must be finite")

        return value

    @field_validator("features")
    @classmethod
    def validate_feature_values(cls, values):
        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(
                    f"Feature {name} must be finite"
                )

        return values


# ============================================================
# SHARED HELPERS
# ============================================================

def validate_features(features: dict[str, float]):
    required_features = {f"V{i}" for i in range(1, 29)}
    provided_features = set(features)

    missing = sorted(required_features - provided_features)
    unexpected = sorted(provided_features - required_features)

    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Missing required features",
                "features": missing,
            },
        )

    if unexpected:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Unexpected features provided",
                "features": unexpected,
            },
        )


def convert_inr_to_model_amount(amount_inr: float) -> dict:
    """
    Convert INR to the numerical Amount scale used by the hybrid model.

    Steps:
        INR -> EUR using a fixed project exchange-rate assumption
        EUR -> scaled Amount using the verified dataset maximum

    This conversion does not establish that the model is calibrated
    or validated for Indian banking transactions.
    """

    if not math.isfinite(amount_inr) or amount_inr < 0:
        raise ValueError(
            "Transaction amount in INR must be finite and non-negative."
        )

    amount_eur = amount_inr / INR_PER_EUR

    model_amount = amount_eur / ORIGINAL_AMOUNT_MAX

    if not math.isfinite(model_amount):
        raise ValueError(
            "Converted model amount is not finite."
        )

    return {
        "amount_inr": float(amount_inr),
        "amount_eur_assumed": float(amount_eur),
        "model_amount": float(model_amount),
        "inr_per_eur": INR_PER_EUR,
        "original_amount_max": ORIGINAL_AMOUNT_MAX,
    }


# ============================================================
# ROOT AND HEALTH
# ============================================================

@app.get("/")
def read_root():
    return {
        "service": "FraudGuard AI",
        "status": "online",
        "version": app.version,
        "documentation": "/docs",
        "health_endpoint": "/health",
    }


@app.get("/health")
def health():
    component_status = inference_engine.health()

    component_status["hybrid_gnn"] = (
        "loaded" if hybrid_engine is not None else "unavailable"
    )

    return {
        "status": (
            "ready"
            if component_status.get("autoencoder") == "loaded"
            else "degraded"
        ),
        "components": component_status,
        "hybrid_error": hybrid_error,
        "amount_conversion": {
            "input_currency": "INR",
            "assumed_dataset_currency": "EUR",
            "inr_per_eur": INR_PER_EUR,
            "original_amount_max": ORIGINAL_AMOUNT_MAX,
            "method": "INR -> EUR -> dataset-scaled Amount",
        },
        "note": (
            "The original evaluation endpoint retains its existing "
            "amount-input behavior. The hybrid endpoint accepts INR "
            "and converts it to the model's expected Amount scale. "
            "The hybrid score is experimental and is not a calibrated "
            "fraud probability."
        ),
    }


# ============================================================
# EXISTING EVALUATION ENDPOINT
# ============================================================

@app.post(
    "/v1/evaluate",
    summary="Evaluate using the existing fusion pipeline",
)
def evaluate_transaction_endpoint(payload: TransactionRequest):
    validate_features(payload.features)

    # Compatibility preserved:
    # This existing endpoint continues to receive Amount on its
    # existing model-input scale.
    tx_data = dict(payload.features)
    tx_data["Amount"] = payload.amount

    try:
        velocity_metrics = feature_store.update_and_get_velocity(
            card_id=payload.card_id,
            current_timestamp=payload.timestamp,
        )

        result = inference_engine.evaluate_transaction(
            transaction_features=tx_data,
            velocity_metrics=velocity_metrics,
        )

        return {
            "card_id": payload.card_id,
            "timestamp": payload.timestamp,
            "decision": result["action"],
            "risk_score": result["risk_score"],
            "risk_level": result["risk_level"],
            "breakdown": result["breakdown"],
        }

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Transaction evaluation failed. Check server logs.",
        ) from exc


# ============================================================
# HYBRID AUTOENCODER + GNN ENDPOINT
# ============================================================

@app.post(
    "/v1/evaluate-hybrid",
    summary="Evaluate a transaction amount entered in INR",
)
def evaluate_hybrid_endpoint(payload: TransactionRequest):
    if hybrid_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Hybrid inference is unavailable. Check API server logs.",
        )

    validate_features(payload.features)

    try:
        # Convert the user-entered INR amount into the model's
        # existing Amount scale.
        conversion = convert_inr_to_model_amount(payload.amount)

        tx_data = dict(payload.features)
        tx_data["Amount"] = conversion["model_amount"]

        result = hybrid_engine.predict(tx_data)

        return {
            "card_id": payload.card_id,
            "timestamp": payload.timestamp,
            "decision": result["decision"],
            "risk_score": result["risk_score"],
            "risk_level": result["risk_level"],
            "breakdown": result["breakdown"],
            "score_type": result["score_type"],
            "model": "Autoencoder + feature-similarity GNN prototype",
            "amount_conversion": conversion,
            "note": (
                "The amount was converted from INR using a fixed "
                "project exchange-rate assumption and the verified "
                "dataset scaling constant. The risk score is not a "
                "calibrated probability, and this conversion does not "
                "establish validation for Indian banking transactions."
            ),
        }

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Hybrid evaluation failed. Check server logs.",
        ) from exc

