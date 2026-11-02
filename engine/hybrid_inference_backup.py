import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]

AE_PATH = ROOT / "models" / "weights" / "autoencoder.keras"
GNN_PATH = ROOT / "models" / "weights" / "gnn_model_state.pt"
SCALER_PATH = ROOT / "models" / "weights" / "gnn_scaler.npz"
TRAIN_PATH = ROOT / "data" / "processed" / "train_data.csv"
AE_RESULTS_PATH = ROOT / "results" / "autoencoder_results.json"


class FraudGCN(torch.nn.Module):
    def __init__(self, input_channels=29, hidden_channels=32, dropout=0.25):
        super().__init__()
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels // 2)
        self.output = torch.nn.Linear(hidden_channels // 2, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.output(x)


class HybridInference:
    def __init__(self):
        required = [AE_PATH, GNN_PATH, SCALER_PATH, TRAIN_PATH, AE_RESULTS_PATH]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(f"Required hybrid artifact missing: {path}")

        self.autoencoder = tf.keras.models.load_model(AE_PATH, compile=False)

        self.checkpoint = torch.load(
            GNN_PATH, map_location="cpu", weights_only=True
        )
        self.gnn = FraudGCN(
            input_channels=self.checkpoint["input_channels"],
            hidden_channels=self.checkpoint["hidden_channels"],
        )
        self.gnn.load_state_dict(self.checkpoint["state_dict"])
        self.gnn.eval()

        scaler_data = np.load(SCALER_PATH)
        self.scaler_mean = scaler_data["mean"]
        self.scaler_scale = scaler_data["scale"]
        self.threshold = float(self.checkpoint["threshold"])
        self.k = int(self.checkpoint["k_neighbors"])

        train_df = pd.read_csv(TRAIN_PATH)

        # Reproduce the training node sampling strategy.
        target_nodes = int(self.checkpoint["train_nodes"])
        if len(train_df) > target_nodes:
            fraud = train_df[train_df["Class"] == 1]
            legitimate = train_df[train_df["Class"] == 0]
            keep_legitimate = target_nodes - len(fraud)
            train_df = pd.concat([
                legitimate.sample(
                    n=keep_legitimate,
                    random_state=int(self.checkpoint["seed"]),
                ),
                fraud,
            ]).sort_index().reset_index(drop=True)

        x_raw = train_df[FEATURES].to_numpy(dtype=np.float32)
        x_scaled = (x_raw - self.scaler_mean) / self.scaler_scale
        self.reference_x = torch.tensor(x_scaled, dtype=torch.float32)

        self.neighbor_search = NearestNeighbors(
            metric="euclidean",
            algorithm="auto",
            n_jobs=-1,
        )
        self.neighbor_search.fit(x_scaled)

        # Build the reference graph in batched queries, not one query per node.
        # Query k+1 neighbors and remove the node itself when it appears.
        node_count = len(x_scaled)
        query_count = min(self.k + 1, node_count)
        neighbor_indices = self.neighbor_search.kneighbors(
            x_scaled, n_neighbors=query_count, return_distance=False
        )

        src_parts = []
        dst_parts = []

        for node_id, neighbors in enumerate(neighbor_indices):
            neighbors = neighbors[neighbors != node_id][:self.k]
            if len(neighbors):
                src_parts.append(np.full(len(neighbors), node_id, dtype=np.int64))
                dst_parts.append(neighbors.astype(np.int64))

        if src_parts:
            src = np.concatenate(src_parts)
            dst = np.concatenate(dst_parts)
            # Make the reference graph undirected.
            edges = np.concatenate([
                np.stack([src, dst]),
                np.stack([dst, src]),
            ], axis=1)
            self.reference_edge_index = torch.tensor(
                np.unique(edges, axis=1), dtype=torch.long
            )
        else:
            self.reference_edge_index = torch.empty((2, 0), dtype=torch.long)

        with AE_RESULTS_PATH.open("r", encoding="utf-8") as f:
            ae_results = json.load(f)
        self.ae_threshold = float(
            ae_results["validation_metrics"]["threshold"]
        )

    def health(self):
        return {
            "autoencoder": "loaded",
            "gnn": "loaded",
            "graph_method": "nearest-neighbor feature similarity",
            "reference_nodes": len(self.reference_x),
            "reference_edges": int(self.reference_edge_index.shape[1]),
            "gnn_threshold": self.threshold,
        }

    def predict(self, transaction_features):
        missing = [name for name in FEATURES if name not in transaction_features]
        if missing:
            raise ValueError("Missing features: " + ", ".join(missing))

        values = np.asarray(
            [float(transaction_features[name]) for name in FEATURES],
            dtype=np.float32,
        )
        if not np.isfinite(values).all():
            raise ValueError("All model features must be finite.")

        # Autoencoder reconstruction-error signal.
        ae_input = values.reshape(1, -1)
        reconstruction = self.autoencoder.predict(ae_input, verbose=0)
        reconstruction_error = float(
            np.mean(np.square(ae_input - reconstruction))
        )
        anomaly_score = float(
            np.log1p(
                max(reconstruction_error, 0.0) / self.ae_threshold
            )
        )

        # Scale the incoming transaction with the saved training scaler.
        x_new_np = (
            (values - self.scaler_mean) / self.scaler_scale
        ).reshape(1, -1)

        _, indices = self.neighbor_search.kneighbors(x_new_np)
        neighbor_ids = indices[0].astype(np.int64)

        reference_count = len(self.reference_x)
        query_id = reference_count
        x_all = torch.cat([
            self.reference_x,
            torch.tensor(x_new_np, dtype=torch.float32),
        ], dim=0)

        # Reuse the cached reference graph and add query-to-neighbor edges.
        query_edges = np.stack([
            np.concatenate([np.full(len(neighbor_ids), query_id), neighbor_ids]),
            np.concatenate([neighbor_ids, np.full(len(neighbor_ids), query_id)]),
        ]).astype(np.int64)

        edge_index = torch.cat([
            self.reference_edge_index,
            torch.tensor(query_edges, dtype=torch.long),
        ], dim=1)

        with torch.no_grad():
            logits = self.gnn(x_all, edge_index)
            gnn_probability = float(
                torch.softmax(logits[query_id], dim=0)[1]
            )

        # Prototype fusion score; this is not a calibrated probability.
        fusion_score = float(
            0.5 * gnn_probability
            + 0.5 * min(anomaly_score / 3.0, 1.0)
        )
        fusion_flag = fusion_score >= 0.5

        return {
            "decision": "OTP_VERIFY" if fusion_flag else "APPROVE",
            "risk_level": "MEDIUM" if fusion_flag else "LOW",
            "risk_score": round(fusion_score, 6),
            "score_type": "prototype_fusion_score_not_calibrated_probability",
            "breakdown": {
                "reconstruction_error": reconstruction_error,
                "autoencoder_threshold": self.ae_threshold,
                "gnn_fraud_score": round(gnn_probability, 6),
                "gnn_threshold": self.threshold,
                "fusion_threshold": 0.5,
                "fusion_flag": bool(fusion_flag),
                "graph_neighbors": int(len(neighbor_ids)),
                "graph_method": "training-reference feature-similarity graph",
            },
        }
