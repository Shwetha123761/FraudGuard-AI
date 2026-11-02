import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tensorflow as tf

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GCNConv


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
WEIGHTS_DIR = ROOT / "models" / "weights"
RESULTS_DIR = ROOT / "results"

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
K_NEIGHBORS = 5
BATCH_SIZE = 2048
FUSION_ALPHA = 0.5
MAX_FPR = 0.005


class FraudGCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels=32, dropout=0.25):
        super().__init__()
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels // 2)
        self.output = torch.nn.Linear(hidden_channels // 2, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.output(x)


def build_graph(x):
    if len(x) < 2:
        raise ValueError("At least two transactions are required.")

    nn_model = NearestNeighbors(
        n_neighbors=min(K_NEIGHBORS + 1, len(x)),
        metric="euclidean",
        n_jobs=-1,
    )
    nn_model.fit(x)
    indices = nn_model.kneighbors(x, return_distance=False)

    source = np.repeat(np.arange(len(x)), indices.shape[1] - 1)
    target = indices[:, 1:].reshape(-1)

    edge_index = torch.tensor(
        np.vstack([source, target]), dtype=torch.long
    )
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return torch.unique(edge_index, dim=1)


def calculate_metrics(y_true, scores, threshold):
    predictions = (scores >= threshold).astype(np.int32)
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
        "false_positive_rate": float(fp / max(tn + fp, 1)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "actual_fraud": int(np.sum(y_true)),
        "predicted_fraud": int(np.sum(predictions)),
    }


def choose_threshold(y_true, scores):
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    best = None

    for threshold in candidates:
        predictions = scores >= threshold
        negatives = y_true == 0
        positives = y_true == 1

        fp = int(np.sum(predictions & negatives))
        tn = int(np.sum(~predictions & negatives))
        tp = int(np.sum(predictions & positives))
        fn = int(np.sum(~predictions & positives))

        fpr = fp / max(fp + tn, 1)
        recall = tp / max(tp + fn, 1)
        f1 = f1_score(y_true, predictions, zero_division=0)

        if fpr <= MAX_FPR:
            candidate = (recall, f1, float(threshold))
            if best is None or candidate[:2] > best[:2]:
                best = candidate

    if best is None:
        return float(np.nextafter(np.max(scores), np.inf))

    return best[2]


def get_autoencoder_errors(autoencoder, x):
    errors = []

    for start in range(0, len(x), BATCH_SIZE):
        batch = x[start:start + BATCH_SIZE]
        reconstructed = autoencoder.predict(
            batch, batch_size=BATCH_SIZE, verbose=0
        )
        errors.append(np.mean(np.square(batch - reconstructed), axis=1))

    return np.concatenate(errors)


def evaluate_partition(name, csv_path, autoencoder, gnn, scaler, ae_threshold):
    print(f"\nLoading {name} dataset...", flush=True)
    frame = pd.read_csv(csv_path)

    missing = set(FEATURES + ["Class"]) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} dataset is missing columns: {sorted(missing)}")

    x_raw = frame[FEATURES].to_numpy(dtype=np.float32)
    y = frame["Class"].to_numpy(dtype=np.int32)

    if not np.isfinite(x_raw).all():
        raise ValueError(f"{name} features contain NaN or infinite values.")

    started = time.perf_counter()

    print(f"Scaling {len(frame):,} rows...", flush=True)
    x_scaled = scaler.transform(x_raw).astype(np.float32)

    print(f"Building {name} graph...", flush=True)
    edge_index = build_graph(x_scaled)

    print(f"Running GNN on {name} graph...", flush=True)
    gnn.eval()
    with torch.no_grad():
        logits = gnn(
            torch.tensor(x_scaled, dtype=torch.float32),
            edge_index,
        )
        gnn_scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    print(f"Calculating autoencoder errors for {name}...", flush=True)
    ae_errors = get_autoencoder_errors(autoencoder, x_raw)

    anomaly_scores = np.log1p(
        np.maximum(ae_errors, 0) / max(ae_threshold, 1e-12)
    )

    # Prototype score, not a calibrated probability.
    hybrid_scores = (
        FUSION_ALPHA * gnn_scores
        + (1.0 - FUSION_ALPHA) * np.minimum(anomaly_scores / 3.0, 1.0)
    )

    elapsed = time.perf_counter() - started

    result = {
        "dataset": name,
        "rows_evaluated": int(len(frame)),
        "actual_fraud": int(np.sum(y)),
        "graph_method": "partition-level k-nearest-neighbor feature-similarity graph",
        "graph_nodes": int(len(frame)),
        "graph_edges": int(edge_index.shape[1]),
        "runtime_seconds": round(elapsed, 3),
        "seconds_per_transaction": round(elapsed / max(len(frame), 1), 6),
        "score_type": "prototype hybrid score; not a calibrated probability",
        "fusion_alpha": FUSION_ALPHA,
        "autoencoder_threshold": float(ae_threshold),
        "gnn_score_min": float(np.min(gnn_scores)),
        "gnn_score_max": float(np.max(gnn_scores)),
        "hybrid_score_min": float(np.min(hybrid_scores)),
        "hybrid_score_max": float(np.max(hybrid_scores)),
    }

    return result, y, hybrid_scores


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    required = [
        DATA_DIR / "validation_data.csv",
        DATA_DIR / "test_data.csv",
        WEIGHTS_DIR / "autoencoder.keras",
        WEIGHTS_DIR / "gnn_model_state.pt",
        WEIGHTS_DIR / "gnn_scaler.npz",
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print("Loading saved autoencoder...", flush=True)
    autoencoder = tf.keras.models.load_model(
        WEIGHTS_DIR / "autoencoder.keras", compile=False
    )

    print("Loading saved GNN...", flush=True)
    try:
        checkpoint = torch.load(
            WEIGHTS_DIR / "gnn_model_state.pt",
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            WEIGHTS_DIR / "gnn_model_state.pt", map_location="cpu"
        )

    if checkpoint["features"] != FEATURES:
        raise ValueError("Saved GNN feature order does not match.")

    gnn = FraudGCN(
        input_channels=checkpoint["input_channels"],
        hidden_channels=checkpoint["hidden_channels"],
    )
    gnn.load_state_dict(checkpoint["state_dict"])
    gnn.eval()

    scaler_data = np.load(WEIGHTS_DIR / "gnn_scaler.npz")
    scaler = StandardScaler()
    scaler.mean_ = scaler_data["mean"]
    scaler.scale_ = scaler_data["scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(FEATURES)

    with open(RESULTS_DIR / "autoencoder_results.json", encoding="utf-8") as f:
        ae_results = json.load(f)

    if "validation_metrics" not in ae_results or "threshold" not in ae_results["validation_metrics"]:
        raise KeyError("Autoencoder results JSON has no 'threshold' field.")

    ae_threshold = float(ae_results["validation_metrics"]["threshold"])

    # Select the threshold using validation labels only.
    validation, y_val, val_scores = evaluate_partition(
        "validation",
        DATA_DIR / "validation_data.csv",
        autoencoder,
        gnn,
        scaler,
        ae_threshold,
    )

    threshold = choose_threshold(y_val, val_scores)
    validation["metrics"] = calculate_metrics(y_val, val_scores, threshold)
    validation["threshold_selection"] = (
        "Validation only: maximize recall subject to FPR <= 0.5%"
    )

    validation_output = {
        "evaluation": "full validation partition",
        "test_used_for_threshold_selection": False,
        "validation": validation,
    }

    validation_path = RESULTS_DIR / "gnn_hybrid_validation_results.json"
    with validation_path.open("w", encoding="utf-8") as f:
        json.dump(validation_output, f, indent=2)

    print("\n=== VALIDATION RESULTS ===", flush=True)
    print(json.dumps(validation_output, indent=2), flush=True)
    print(f"Saved: {validation_path}", flush=True)

    # Freeze the validation threshold before testing.
    test, y_test, test_scores = evaluate_partition(
        "test",
        DATA_DIR / "test_data.csv",
        autoencoder,
        gnn,
        scaler,
        ae_threshold,
    )
    test["metrics"] = calculate_metrics(y_test, test_scores, threshold)

    test_output = {
        "evaluation": "full held-out test partition",
        "threshold_source": "validation dataset only",
        "threshold": float(threshold),
        "test_used_for_training_or_threshold_selection": False,
        "test": test,
    }

    test_path = RESULTS_DIR / "gnn_hybrid_full_test_results.json"
    with test_path.open("w", encoding="utf-8") as f:
        json.dump(test_output, f, indent=2)

    print("\n=== FULL TEST RESULTS ===", flush=True)
    print(json.dumps(test_output, indent=2), flush=True)
    print(f"Saved: {test_path}", flush=True)

    print(
        "\nNOTE: These are partition-level graph results, not direct "
        "measurements of the API's per-transaction reference-graph inference. "
        "The hybrid score is not a calibrated probability.",
        flush=True,
    )


if __name__ == "__main__":
    main()



