import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
WEIGHTS_DIR = ROOT / "models" / "weights"
RESULTS_DIR = ROOT / "results"

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
SEED = 42
K_NEIGHBORS = 5
EPOCHS = 40
HIDDEN_CHANNELS = 32
MAX_TRAIN_NODES = 12000

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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


def build_graph(x, k=K_NEIGHBORS):
    # Each node represents a transaction. Edges represent feature similarity.
    # Class labels are never used to create graph edges.
    if len(x) < 2:
        raise ValueError("At least two transactions are required.")

    neighbors = min(k + 1, len(x))
    nn = NearestNeighbors(n_neighbors=neighbors, metric="euclidean", n_jobs=-1)
    nn.fit(x)
    indices = nn.kneighbors(x, return_distance=False)

    source = np.repeat(np.arange(len(x)), indices.shape[1] - 1)
    target = indices[:, 1:].reshape(-1)

    edge_index = torch.tensor(
        np.vstack([source, target]), dtype=torch.long
    )
    # Make graph edges undirected.
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index


def metrics_at_threshold(y, probability, threshold):
    prediction = (probability >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "predicted_fraud": int(prediction.sum()),
        "actual_fraud": int(np.sum(y)),
    }


def main():
    train_path = DATA_DIR / "train_data.csv"
    validation_path = DATA_DIR / "validation_data.csv"

    for p in [train_path, validation_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required data file missing: {p}")

    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)

    for name, df in [("train", train_df), ("validation", validation_df)]:
        missing = set(FEATURES + ["Class"]) - set(df.columns)
        if missing:
            raise ValueError(f"{name} data is missing columns: {sorted(missing)}")

    # CPU graph construction is bounded for the deadline.
    # Stratified sampling retains fraud cases and uses only training data.
    if len(train_df) > MAX_TRAIN_NODES:
        fraud_df = train_df[train_df["Class"] == 1]
        legit_df = train_df[train_df["Class"] == 0]
        keep_legit = MAX_TRAIN_NODES - len(fraud_df)
        train_df = pd.concat(
            [
                legit_df.sample(n=keep_legit, random_state=SEED),
                fraud_df,
            ]
        ).sort_index().reset_index(drop=True)

    x_train_raw = train_df[FEATURES].to_numpy(dtype=np.float32)
    y_train = train_df["Class"].to_numpy(dtype=np.int64)
    x_val_raw = validation_df[FEATURES].to_numpy(dtype=np.float32)
    y_val = validation_df["Class"].to_numpy(dtype=np.int64)

    # Fit scaling on training data only.
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw).astype(np.float32)
    x_val = scaler.transform(x_val_raw).astype(np.float32)

    # Build train graph from training nodes only.
    train_edges = build_graph(x_train)
    train_x = torch.tensor(x_train, dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.long)

    # Build a separate validation graph. No validation labels enter graph building.
    val_edges = build_graph(x_val)
    val_x = torch.tensor(x_val, dtype=torch.float32)
    val_y_tensor = torch.tensor(y_val, dtype=torch.long)

    model = FraudGCN(input_channels=len(FEATURES), hidden_channels=HIDDEN_CHANNELS)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    counts = np.bincount(y_train, minlength=2)
    class_weights = counts.sum() / (2.0 * np.maximum(counts, 1))
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

    print(f"Training nodes: {len(train_x):,}")
    print(f"Training fraud cases: {int(y_train.sum()):,}")
    print(f"Training edges (directed entries): {train_edges.shape[1]:,}")
    print(f"Validation nodes: {len(val_x):,}")
    print(f"Validation fraud cases: {int(y_val.sum()):,}")
    print("Training GCN on CPU...")

    best_state = None
    best_ap = -1.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_x, train_edges)
        loss = F.cross_entropy(logits, train_y, weight=weight_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x, val_edges)
            val_prob = torch.softmax(val_logits, dim=1)[:, 1].numpy()

        ap = average_precision_score(y_val, val_prob)
        if ap > best_ap:
            best_ap = float(ap)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:02d}/{EPOCHS} | loss={loss.item():.4f} | validation AP={ap:.4f}")

    if best_state is None:
        raise RuntimeError("Training did not produce a usable model.")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        val_logits = model(val_x, val_edges)
        val_prob = torch.softmax(val_logits, dim=1)[:, 1].numpy()

    # Choose a threshold using validation data only:
    # maximize recall subject to FPR <= 0.5%.
    candidates = np.unique(np.concatenate(([0.0, 1.0], val_prob)))
    best_choice = None
    for threshold in candidates:
        prediction = (val_prob >= threshold).astype(int)
        tn = int(np.sum((y_val == 0) & (prediction == 0)))
        fp = int(np.sum((y_val == 0) & (prediction == 1)))
        fpr = fp / max(tn + fp, 1)
        if fpr <= 0.005:
            recall = recall_score(y_val, prediction, zero_division=0)
            f1 = f1_score(y_val, prediction, zero_division=0)
            candidate = (recall, f1, float(threshold), fpr)
            if best_choice is None or candidate[:2] > best_choice[:2]:
                best_choice = candidate

    if best_choice is None:
        threshold = 1.0
        fpr = 0.0
    else:
        _, _, threshold, fpr = best_choice

    val_metrics = metrics_at_threshold(y_val, val_prob, threshold)
    val_metrics["false_positive_rate"] = float(fpr)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = WEIGHTS_DIR / "gnn_model_state.pt"
    scaler_path = WEIGHTS_DIR / "gnn_scaler.npz"
    results_path = RESULTS_DIR / "gnn_validation_results.json"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_channels": len(FEATURES),
            "hidden_channels": HIDDEN_CHANNELS,
            "k_neighbors": K_NEIGHBORS,
            "features": FEATURES,
            "threshold": float(threshold),
            "seed": SEED,
            "graph_method": "k-nearest-neighbor feature similarity graph",
            "train_nodes": int(len(train_df)),
            "validation_nodes": int(len(validation_df)),
        },
        model_path,
    )
    np.savez(
        scaler_path,
        mean=scaler.mean_,
        scale=scaler.scale_,
    )

    results = {
        "model": "GCN feature-similarity graph",
        "validation_only_threshold_selection": True,
        "test_used_for_training_or_threshold_selection": False,
        "graph_method": "k-nearest-neighbor feature similarity graph",
        "graph_limitations": (
            "Edges represent feature similarity, not known real-world card "
            "or account relationships. Validation graph was built separately."
        ),
        "sampled_training_nodes": int(len(train_df)),
        "training_fraud_cases": int(y_train.sum()),
        "validation_metrics": val_metrics,
        "best_validation_average_precision": float(best_ap),
        "weights_path": str(model_path.relative_to(ROOT)),
        "scaler_path": str(scaler_path.relative_to(ROOT)),
    }

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== GNN VALIDATION RESULTS ===")
    print(json.dumps(results, indent=2))
    print("\nSaved GNN weights:", model_path)
    print("Saved scaler:", scaler_path)
    print("Saved validation results:", results_path)


if __name__ == "__main__":
    main()
