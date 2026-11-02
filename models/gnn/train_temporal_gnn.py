import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
WEIGHTS_DIR = ROOT / "models" / "weights"
RESULTS_DIR = ROOT / "results"

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]

SEED = 42
MAX_TRAIN_NODES = 6000
TEMPORAL_WINDOW = 5
HIDDEN_CHANNELS = 16
EPOCHS = 20
LEARNING_RATE = 0.003
DROPOUT = 0.15
MAX_FPR = 0.005

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")


# ============================================================
# DATA PREPARATION
# ============================================================

def load_data():
    train_path = DATA_DIR / "train_data.csv"
    val_path = DATA_DIR / "validation_data.csv"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            "Expected train_data.csv and validation_data.csv "
            "inside data/processed."
        )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    required = set(FEATURES + ["Time", "Class"])

    for name, df in (("training", train_df), ("validation", val_df)):
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{name} data is missing columns: {sorted(missing)}"
            )

        checked = df[["Time", "Class"] + FEATURES]
        if checked.isna().any().any():
            raise ValueError(f"{name} data contains missing values.")

        numeric = df[["Time"] + FEATURES].to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError(f"{name} data contains non-finite values.")

        if not set(df["Class"].unique()).issubset({0, 1}):
            raise ValueError(f"{name} Class must contain only 0 and 1.")

    # Stable ordering makes equal-time ordering deterministic.
    train_df = train_df.sort_values(
        "Time", kind="mergesort"
    ).reset_index(drop=True)
    val_df = val_df.sort_values(
        "Time", kind="mergesort"
    ).reset_index(drop=True)

    if train_df.empty or val_df.empty:
        raise ValueError("Training and validation datasets must be non-empty.")

    if train_df["Time"].max() > val_df["Time"].min():
        raise ValueError(
            "Training and validation time ranges overlap or are reversed. "
            "Inspect the chronological split before continuing."
        )

    # Keep all fraud cases and sample legitimate training rows.
    # The selected rows are re-sorted chronologically afterward.
    if len(train_df) > MAX_TRAIN_NODES:
        fraud_df = train_df[train_df["Class"] == 1]
        legitimate_df = train_df[train_df["Class"] == 0]

        if len(fraud_df) >= MAX_TRAIN_NODES:
            raise ValueError(
                "The number of training fraud cases exceeds MAX_TRAIN_NODES."
            )

        legitimate_count = MAX_TRAIN_NODES - len(fraud_df)
        legitimate_sample = legitimate_df.sample(
            n=legitimate_count,
            random_state=SEED,
        )

        train_df = pd.concat(
            [fraud_df, legitimate_sample], axis=0
        ).sort_values(
            "Time", kind="mergesort"
        ).reset_index(drop=True)

    # Use the complete validation dataset. Do not sample based on labels.
    val_df = val_df.reset_index(drop=True)

    return train_df, val_df


# ============================================================
# CAUSAL TEMPORAL NEIGHBORS
# ============================================================

def build_temporal_neighbors(n_nodes, window=TEMPORAL_WINDOW):
    """
    For node i, list up to `window` preceding nodes.

    Neighbors always have indices smaller than i. The current node
    and all future nodes are excluded. Missing positions are -1.
    """
    if n_nodes < 1:
        raise ValueError("At least one node is required.")
    if window < 1:
        raise ValueError("window must be at least 1.")

    neighbors = np.full((n_nodes, window), -1, dtype=np.int64)

    for i in range(n_nodes):
        start = max(0, i - window)
        previous = np.arange(start, i, dtype=np.int64)
        if previous.size:
            neighbors[i, -previous.size:] = previous

    # Explicit invariant check: every real edge points backward.
    rows = np.arange(n_nodes)[:, None]
    if np.any(neighbors[neighbors >= 0] >= np.broadcast_to(rows, neighbors.shape)[neighbors >= 0]):
        raise RuntimeError("Temporal graph contains a non-causal edge.")

    return torch.tensor(neighbors, dtype=torch.long)


# ============================================================
# CAUSAL TEMPORAL MESSAGE-PASSING NETWORK
# ============================================================

class TemporalFraudGNN(nn.Module):
    """
    Two-layer causal temporal message-passing network.

    Each node combines its own representation with the mean
    representation of up to five preceding nodes. No class labels
    are used in graph construction or message passing.
    """

    def __init__(
        self,
        input_channels,
        hidden_channels=HIDDEN_CHANNELS,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.self1 = nn.Linear(input_channels, hidden_channels)
        self.neighbor1 = nn.Linear(input_channels, hidden_channels)
        self.self2 = nn.Linear(hidden_channels, hidden_channels)
        self.neighbor2 = nn.Linear(hidden_channels, hidden_channels)
        self.output = nn.Linear(hidden_channels, 2)
        self.dropout = dropout

    @staticmethod
    def aggregate_previous(h, neighbors):
        """
        Aggregate h for each row's previous-node indices.
        `neighbors` may contain rows for all nodes or only target nodes.
        """
        valid = neighbors >= 0
        safe_indices = neighbors.clamp(min=0)

        # Shape: [number of requested rows, window, feature_count]
        gathered = h[safe_indices]
        mask = valid.unsqueeze(-1).to(dtype=h.dtype)
        gathered = gathered * mask

        counts = mask.sum(dim=1).clamp(min=1.0)
        return gathered.sum(dim=1) / counts

    def forward(self, x, neighbors, target_indices=None):
        """
        x contains the complete chronological context.
        neighbors must contain one row per node in x.
        target_indices optionally selects which node logits to return.
        """
        if neighbors.shape[0] != x.shape[0]:
            raise ValueError(
                "neighbors must have one row per node in x."
            )

        # Layer 1: each node only receives messages from earlier nodes.
        previous_x = self.aggregate_previous(x, neighbors)
        h = F.relu(self.self1(x) + self.neighbor1(previous_x))
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2: each node aggregates first-layer states of its
        # preceding nodes. Since layer 1 is causal, this remains causal.
        previous_h = self.aggregate_previous(h, neighbors)
        h = F.relu(self.self2(h) + self.neighbor2(previous_h))
        h = F.dropout(h, p=self.dropout, training=self.training)

        if target_indices is not None:
            h = h[target_indices]

        return self.output(h)


# ============================================================
# METRICS AND VALIDATION THRESHOLD
# ============================================================

def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int64)
    negatives = int(np.sum(y_true == 0))
    false_positives = int(
        np.sum((y_true == 0) & (predictions == 1))
    )
    fpr = false_positives / negatives if negatives else 0.0

    return {
        "threshold": float(threshold),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "false_positive_rate": float(fpr),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "roc_auc": (
            float(roc_auc_score(y_true, probabilities))
            if len(np.unique(y_true)) == 2
            else None
        ),
        "predicted_fraud": int(predictions.sum()),
        "actual_fraud": int(y_true.sum()),
    }


def choose_threshold(y_true, probabilities):
    """
    Use validation labels only to maximize recall with FPR <= MAX_FPR.
    F1 breaks ties in recall.
    """
    candidates = np.unique(
        np.concatenate(([0.0, 1.0], probabilities))
    )
    best = None
    negatives = int(np.sum(y_true == 0))

    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(np.int64)
        false_positives = int(
            np.sum((y_true == 0) & (predictions == 1))
        )
        fpr = false_positives / negatives if negatives else 0.0

        if fpr > MAX_FPR:
            continue

        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
        candidate = (float(recall), float(f1), float(threshold))

        if best is None or candidate[:2] > best[:2]:
            best = candidate

    return 1.0 if best is None else best[2]


# ============================================================
# TRAINING AND VALIDATION
# ============================================================

def main():
    print("Loading chronological datasets...")
    train_df, val_df = load_data()

    print(f"Training nodes: {len(train_df):,}")
    print(f"Training fraud cases: {int(train_df['Class'].sum()):,}")
    print(f"Validation nodes: {len(val_df):,}")
    print(f"Validation fraud cases: {int(val_df['Class'].sum()):,}")

    # Fit preprocessing on training data only.
    scaler = StandardScaler()
    x_train = scaler.fit_transform(
        train_df[FEATURES].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    x_val = scaler.transform(
        val_df[FEATURES].to_numpy(dtype=np.float32)
    ).astype(np.float32)

    y_train = train_df["Class"].to_numpy(dtype=np.int64)
    y_val = val_df["Class"].to_numpy(dtype=np.int64)

    # Training graph: sampled training sequence only.
    train_neighbors = build_temporal_neighbors(len(x_train)).to(DEVICE)
    train_x = torch.tensor(x_train, dtype=torch.float32, device=DEVICE)
    train_y = torch.tensor(y_train, dtype=torch.long, device=DEVICE)

    # Validation graph context includes the training feature history
    # followed by all validation features in chronological order.
    # Validation labels are never used for graph construction.
    combined_x = np.concatenate([x_train, x_val], axis=0)
    combined_neighbors = build_temporal_neighbors(
        len(combined_x)
    ).to(DEVICE)
    combined_x_tensor = torch.tensor(
        combined_x, dtype=torch.float32, device=DEVICE
    )

    val_start = len(x_train)
    val_indices = torch.arange(
        val_start, len(combined_x), dtype=torch.long, device=DEVICE
    )

    model = TemporalFraudGNN(input_channels=len(FEATURES)).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    counts = np.bincount(y_train, minlength=2)
    class_weights = counts.sum() / (2.0 * np.maximum(counts, 1))
    weight_tensor = torch.tensor(
        class_weights, dtype=torch.float32, device=DEVICE
    )

    best_ap = -1.0
    best_state = None

    print("\nStarting temporal GNN training on CPU...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        train_logits = model(train_x, train_neighbors)
        loss = F.cross_entropy(
            train_logits, train_y, weight=weight_tensor
        )
        loss.backward()
        optimizer.step()

        # Validation model selection uses validation AP.
        model.eval()
        with torch.no_grad():
            val_logits = model(
                combined_x_tensor,
                combined_neighbors,
                target_indices=val_indices,
            )
            val_probabilities = torch.softmax(
                val_logits, dim=1
            )[:, 1].cpu().numpy()

        ap = float(average_precision_score(y_val, val_probabilities))
        if ap > best_ap:
            best_ap = ap
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:02d}/{EPOCHS} "
                f"| loss={loss.item():.4f} "
                f"| validation AP={ap:.4f}"
            )

    if best_state is None:
        raise RuntimeError("Training did not produce a usable model.")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        val_logits = model(
            combined_x_tensor,
            combined_neighbors,
            target_indices=val_indices,
        )
        val_probabilities = torch.softmax(
            val_logits, dim=1
        )[:, 1].cpu().numpy()

    threshold = choose_threshold(y_val, val_probabilities)
    metrics = calculate_metrics(y_val, val_probabilities, threshold)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    weights_path = WEIGHTS_DIR / "temporal_gnn_model_state.pt"
    scaler_path = WEIGHTS_DIR / "temporal_gnn_scaler.npz"
    results_path = RESULTS_DIR / "temporal_gnn_validation_results.json"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_channels": len(FEATURES),
            "hidden_channels": HIDDEN_CHANNELS,
            "temporal_window": TEMPORAL_WINDOW,
            "features": FEATURES,
            "threshold": float(threshold),
            "seed": SEED,
            "architecture": "two-layer causal temporal message-passing network",
            "time_used_for_graph_order": True,
            "time_used_as_node_feature": False,
        },
        weights_path,
    )

    np.savez(
        scaler_path,
        mean=scaler.mean_,
        scale=scaler.scale_,
    )

    results = {
        "model": "Causal temporal GNN experiment",
        "architecture": "Two-layer causal temporal message-passing network",
        "graph_method": (
            "Each node aggregates up to five preceding nodes in the "
            "chronologically ordered sequence."
        ),
        "temporal_window": TEMPORAL_WINDOW,
        "training_nodes": int(len(train_df)),
        "validation_nodes": int(len(val_df)),
        "training_fraud_cases": int(y_train.sum()),
        "validation_fraud_cases": int(y_val.sum()),
        "validation_only_threshold_selection": True,
        "test_used_for_training_or_threshold_selection": False,
        "threshold_selection_fpr_constraint": MAX_FPR,
        "best_validation_average_precision": best_ap,
        "validation_metrics": metrics,
        "limitations": [
            "Temporal proximity does not establish cardholder or account relationships.",
            "This is not a reproduction of the paper's GTAN architecture.",
            (
                "Training uses a stratified sample of training transactions; "
                "edges connect preceding nodes in that sampled sequence."
            ),
            (
                "Validation metrics are used for checkpoint selection and "
                "threshold selection, so they are model-selection metrics, "
                "not an unbiased final test estimate."
            ),
            (
                "This experiment is separate from the existing GCN and hybrid "
                "inference pipeline."
            ),
        ],
        "weights_path": str(weights_path.relative_to(ROOT)),
        "scaler_path": str(scaler_path.relative_to(ROOT)),
    }

    with results_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\n=== TEMPORAL GNN VALIDATION RESULTS ===")
    print(json.dumps(results, indent=2))
    print("\nSaved weights:", weights_path)
    print("Saved scaler:", scaler_path)
    print("Saved results:", results_path)


if __name__ == "__main__":
    main()
