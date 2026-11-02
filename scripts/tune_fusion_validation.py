import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "validation_data.csv"
AE_PATH = ROOT / "models" / "weights" / "autoencoder.keras"
LR_PATH = ROOT / "models" / "weights" / "supervised_model.joblib"
AE_RESULTS = ROOT / "results" / "autoencoder_results.json"
OUTPUT = ROOT / "results" / "fusion_tuning_results.json"
CONFIG = ROOT / "models" / "weights" / "fusion_config.json"

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
MAX_FPR = 0.005
BATCH_SIZE = 2048
ALPHAS = [0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]


def metrics_at_threshold(y, scores, threshold):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1_score": float(f1_score(y, pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if tn + fp else 0.0,
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def choose_threshold(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    n_negative = int((y == 0).sum())
    max_fp = int(np.floor(MAX_FPR * n_negative + 1e-12))

    # Candidate thresholds are evaluated using vectorized cumulative counts.
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_y = y[order]

    # Only consider thresholds at score boundaries.
    boundary = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(scores) - 1]
    tp_cum = np.cumsum(sorted_y)[boundary]
    fp_cum = (boundary + 1) - tp_cum
    fn_cum = int(y.sum()) - tp_cum
    tn_cum = n_negative - fp_cum

    valid = fp_cum <= max_fp
    if not np.any(valid):
        raise RuntimeError("No threshold satisfies the validation FPR constraint.")

    tp_v = tp_cum[valid]
    fp_v = fp_cum[valid]
    fn_v = fn_cum[valid]
    tn_v = tn_cum[valid]
    boundary_v = boundary[valid]

    rec = tp_v / max(int(y.sum()), 1)
    prec = np.divide(tp_v, tp_v + fp_v, out=np.zeros_like(tp_v, dtype=float),
                     where=(tp_v + fp_v) != 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec),
                   where=(prec + rec) != 0)
    fpr = fp_v / max(n_negative, 1)

    # Objective: recall, then F1, precision, and finally higher threshold.
    order_best = np.lexsort((
        sorted_scores[boundary_v],
        prec,
        f1,
        rec,
    ))
    idx = order_best[-1]
    threshold = float(sorted_scores[boundary_v[idx]])
    result_metrics = metrics_at_threshold(y, scores, threshold)

    if result_metrics["false_positive_rate"] > MAX_FPR + 1e-12:
        raise RuntimeError("Internal threshold selection exceeded FPR constraint.")

    return {"threshold": threshold, "metrics": result_metrics}


def main():
    for path in (DATA, AE_PATH, LR_PATH, AE_RESULTS):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    df = pd.read_csv(DATA)
    missing = [c for c in FEATURES + ["Class"] if c not in df.columns]
    if missing:
        raise ValueError(f"Validation dataset missing columns: {missing}")

    X = df[FEATURES].astype(np.float32)
    y = df["Class"].astype(int).to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError("Validation data must contain legitimate and fraud cases.")

    print(f"Validation rows: {len(df):,}")
    print(f"Validation fraud cases: {int(y.sum()):,}")
    print("Loading existing models...")

    ae = load_model(AE_PATH, compile=False)
    bundle = joblib.load(LR_PATH)
    if list(bundle["features"]) != FEATURES:
        raise ValueError("Supervised model feature order mismatch.")

    ae_threshold = float(json.loads(AE_RESULTS.read_text(encoding="utf-8"))
                         ["validation_metrics"]["threshold"])
    lr_model = bundle["model"]
    lr_reference_threshold = float(bundle["threshold"])

    values = X.to_numpy(dtype=np.float32)
    errors = []
    print("Calculating Autoencoder reconstruction errors...")
    for start in range(0, len(values), BATCH_SIZE):
        batch = values[start:start + BATCH_SIZE]
        reconstructed = ae.predict(batch, verbose=0)
        errors.extend(np.mean(np.square(batch - reconstructed), axis=1).tolist())

    errors = np.asarray(errors, dtype=np.float64)
    print("Calculating Logistic Regression probabilities...")
    probabilities = lr_model.predict_proba(X)[:, 1]
    anomaly_signal = np.log1p(np.maximum(errors, 0.0) / ae_threshold)

    print("Searching fusion weights with vectorized threshold selection...")
    candidates = []
    for alpha in ALPHAS:
        scores = probabilities + alpha * anomaly_signal
        choice = choose_threshold(y, scores)
        candidates.append({
            "alpha": alpha,
            "threshold": choice["threshold"],
            "metrics": choice["metrics"],
        })
        m = choice["metrics"]
        print(f"alpha={alpha:>5}: recall={m['recall']:.4f}, "
              f"precision={m['precision']:.4f}, F1={m['f1_score']:.4f}, "
              f"FPR={m['false_positive_rate']:.4f}")

    # Pick highest recall, then F1, precision, average precision; prefer
    # the simpler LR-only score if all metrics tie.
    best = max(
        candidates,
        key=lambda c: (
            c["metrics"]["recall"],
            c["metrics"]["f1_score"],
            c["metrics"]["precision"],
            c["metrics"]["average_precision"],
            -c["alpha"],
        ),
    )

    result = {
        "partition": "validation_only",
        "rows": int(len(y)),
        "fraud_cases": int(y.sum()),
        "max_allowed_validation_fpr": MAX_FPR,
        "threshold_selection": (
            "Fusion weights and thresholds selected on validation data only. "
            "Held-out test data was not loaded."
        ),
        "autoencoder_threshold": ae_threshold,
        "supervised_validation_threshold_reference": lr_reference_threshold,
        "selection_objective": (
            "Maximize recall subject to FPR <= 0.005; ties resolved by F1, "
            "precision, average precision, and lower fusion weight."
        ),
        "fusion_candidates": candidates,
        "selected_fusion": best,
    }

    config = {
        "fusion_method": "logistic_probability_plus_weighted_log_anomaly",
        "alpha": best["alpha"],
        "threshold": best["threshold"],
        "max_validation_fpr": MAX_FPR,
        "validation_metrics": best["metrics"],
        "autoencoder_threshold": ae_threshold,
        "supervised_model_threshold_reference": lr_reference_threshold,
        "selected_on": "validation_data_only",
        "risk_score_is_calibrated_probability": False,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    CONFIG.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\n=== SELECTED VALIDATION FUSION ===")
    print(json.dumps(config, indent=2))
    print(f"\nDetailed results saved to: {OUTPUT}")
    print(f"Configuration saved to: {CONFIG}")


if __name__ == "__main__":
    main()
