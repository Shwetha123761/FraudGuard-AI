
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

FUSION_THRESHOLD = 0.5


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
        required = [
            AE_PATH,
            GNN_PATH,
            SCALER_PATH,
            TRAIN_PATH,
            AE_RESULTS_PATH,
        ]

        for path in required:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required hybrid artifact missing: {path}"
                )

        self.autoencoder = tf.keras.models.load_model(
            AE_PATH,
            compile=False,
        )

        self.checkpoint = torch.load(
            GNN_PATH,
            map_location="cpu",
            weights_only=True,
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

            train_df = pd.concat(
                [
                    legitimate.sample(
                        n=keep_legitimate,
                        random_state=int(self.checkpoint["seed"]),
                    ),
                    fraud,
                ]
            ).sort_index().reset_index(drop=True)

        x_raw = train_df[FEATURES].to_numpy(dtype=np.float32)
        x_scaled = (
            (x_raw - self.scaler_mean) / self.scaler_scale
        )

        self.reference_x = torch.tensor(
            x_scaled,
            dtype=torch.float32,
        )

        self.neighbor_search = NearestNeighbors(
            metric="euclidean",
            algorithm="auto",
            n_jobs=-1,
        )

        self.neighbor_search.fit(x_scaled)

        # Build the reference graph.
        node_count = len(x_scaled)
        query_count = min(self.k + 1, node_count)

        neighbor_indices = self.neighbor_search.kneighbors(
            x_scaled,
            n_neighbors=query_count,
            return_distance=False,
        )

        src_parts = []
        dst_parts = []

        for node_id, neighbors in enumerate(neighbor_indices):
            neighbors = neighbors[neighbors != node_id][:self.k]

            if len(neighbors):
                src_parts.append(
                    np.full(
                        len(neighbors),
                        node_id,
                        dtype=np.int64,
                    )
                )
                dst_parts.append(
                    neighbors.astype(np.int64)
                )

        if src_parts:
            src = np.concatenate(src_parts)
            dst = np.concatenate(dst_parts)

            # Make the reference graph undirected.
            edges = np.concatenate(
                [
                    np.stack([src, dst]),
                    np.stack([dst, src]),
                ],
                axis=1,
            )

            self.reference_edge_index = torch.tensor(
                np.unique(edges, axis=1),
                dtype=torch.long,
            )
        else:
            self.reference_edge_index = torch.empty(
                (2, 0),
                dtype=torch.long,
            )

        with AE_RESULTS_PATH.open("r", encoding="utf-8") as file:
            ae_results = json.load(file)

        self.ae_threshold = float(
            ae_results["validation_metrics"]["threshold"]
        )

        # ----------------------------------------------------------
        # Optimized local-GCN cache.
        # Reference features remain fixed between transactions.
        # ----------------------------------------------------------
        self.reference_node_count = len(self.reference_x)

        destination_degree = torch.bincount(
            self.reference_edge_index[1],
            minlength=self.reference_node_count,
        ).to(torch.float32)

        # GCNConv adds a self-loop to each node.
        self.base_degree = destination_degree + 1.0

        # Cache the first layer's linear transformation.
        with torch.no_grad():
            self.reference_x1_linear = self.gnn.conv1.lin(
                self.reference_x
            )

        # Store incoming reference sources per target node.
        edge_src = self.reference_edge_index[0].cpu().numpy()
        edge_dst = self.reference_edge_index[1].cpu().numpy()

        incoming_sources = [
            [] for _ in range(self.reference_node_count)
        ]

        for src, dst in zip(edge_src, edge_dst):
            incoming_sources[int(dst)].append(int(src))

        self.incoming_sources = [
            np.asarray(sources, dtype=np.int64)
            for sources in incoming_sources
        ]

    def health(self):
        return {
            "autoencoder": "loaded",
            "gnn": "loaded",
            "gnn_inference": "optimized_local_gcn",
            "graph_method": "nearest-neighbor feature similarity",
            "reference_nodes": len(self.reference_x),
            "reference_edges": int(
                self.reference_edge_index.shape[1]
            ),
            "gnn_threshold": self.threshold,
            "fusion_threshold": FUSION_THRESHOLD,
        }

    def _local_gnn_probability(self, scaled_query, neighbor_ids):
        """
        Compute the query node's two-layer GCN output locally.

        This is specific to the current GCN architecture and graph
        construction. It must be checked against full-graph inference
        whenever either changes.
        """
        neighbor_ids = np.asarray(
            neighbor_ids,
            dtype=np.int64,
        )

        if len(neighbor_ids) == 0:
            raise ValueError(
                "At least one reference neighbor is required."
            )

        device = self.reference_x.device

        query_degree = torch.tensor(
            float(len(neighbor_ids) + 1),
            dtype=torch.float32,
            device=device,
        )

        neighbor_tensor = torch.as_tensor(
            neighbor_ids,
            dtype=torch.long,
            device=device,
        )

        neighbor_degrees = (
            self.base_degree[neighbor_tensor] + 1.0
        )

        query_x = torch.as_tensor(
            scaled_query,
            dtype=torch.float32,
            device=device,
        ).reshape(1, -1)

        conv1 = self.gnn.conv1
        conv2 = self.gnn.conv2

        def normalization_weight(degree_src, degree_dst):
            return torch.rsqrt(degree_src * degree_dst)

        with torch.no_grad():
            query_x1_linear = conv1.lin(query_x).squeeze(0)

            # First layer: query representation.
            query_aggregate = self.reference_x1_linear.new_zeros(
                self.reference_x1_linear.shape[1]
            )

            query_aggregate += (
                query_x1_linear / query_degree
            )

            for neighbor_id, neighbor_degree in zip(
                neighbor_ids,
                neighbor_degrees,
            ):
                query_aggregate += (
                    self.reference_x1_linear[int(neighbor_id)]
                    * normalization_weight(
                        neighbor_degree,
                        query_degree,
                    )
                )

            query_h1 = torch.relu(
                query_aggregate + conv1.bias
            )

            # First-layer representations for query neighbors.
            neighbor_h1_rows = []

            updated_degree_by_id = {
                int(node_id): neighbor_degrees[i]
                for i, node_id in enumerate(neighbor_ids)
            }

            for target_id, target_degree in zip(
                neighbor_ids,
                neighbor_degrees,
            ):
                aggregate = self.reference_x1_linear.new_zeros(
                    self.reference_x1_linear.shape[1]
                )

                # Existing incoming reference edges.
                for source_id in self.incoming_sources[
                    int(target_id)
                ]:
                    source_degree = updated_degree_by_id.get(
                        int(source_id),
                        self.base_degree[int(source_id)],
                    )

                    aggregate += (
                        self.reference_x1_linear[int(source_id)]
                        * normalization_weight(
                            source_degree,
                            target_degree,
                        )
                    )

                # Self-loop.
                aggregate += (
                    self.reference_x1_linear[int(target_id)]
                    / target_degree
                )

                # New edge from the query node.
                aggregate += (
                    query_x1_linear
                    * normalization_weight(
                        query_degree,
                        target_degree,
                    )
                )

                neighbor_h1_rows.append(
                    torch.relu(aggregate + conv1.bias)
                )

            # Second layer: calculate only the query's output.
            h1_query_and_neighbors = torch.stack(
                [query_h1] + neighbor_h1_rows,
                dim=0,
            )

            h2_linear = conv2.lin(
                h1_query_and_neighbors
            )

            query_h2_aggregate = (
                h2_linear[0] / query_degree
            )

            for i, neighbor_degree in enumerate(
                neighbor_degrees,
                start=1,
            ):
                query_h2_aggregate += (
                    h2_linear[i]
                    * normalization_weight(
                        neighbor_degree,
                        query_degree,
                    )
                )

            query_h2 = torch.relu(
                query_h2_aggregate + conv2.bias
            )

            logits = self.gnn.output(
                query_h2.reshape(1, -1)
            )

            probability = torch.softmax(
                logits,
                dim=1,
            )[0, 1].item()

        return float(probability)

    def predict(self, transaction_features):
        missing = [
            name
            for name in FEATURES
            if name not in transaction_features
        ]

        if missing:
            raise ValueError(
                "Missing features: " + ", ".join(missing)
            )

        try:
            values = np.asarray(
                [
                    float(transaction_features[name])
                    for name in FEATURES
                ],
                dtype=np.float32,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "All model features must be numeric."
            ) from exc

        if not np.isfinite(values).all():
            raise ValueError(
                "All model features must be finite."
            )

        # Autoencoder reconstruction-error signal.
        ae_input = values.reshape(1, -1)

        reconstruction = self.autoencoder.predict(
            ae_input,
            verbose=0,
        )

        reconstruction_error = float(
            np.mean(np.square(ae_input - reconstruction))
        )

        anomaly_score = float(
            np.log1p(
                max(reconstruction_error, 0.0)
                / self.ae_threshold
            )
        )

        # Scale the incoming transaction with the saved scaler.
        x_new_np = (
            (values - self.scaler_mean) / self.scaler_scale
        ).reshape(1, -1)

        _, indices = self.neighbor_search.kneighbors(
            x_new_np
        )

        neighbor_ids = indices[0].astype(np.int64)

        # Optimized local GNN inference.
        gnn_probability = self._local_gnn_probability(
            x_new_np,
            neighbor_ids,
        )

        # Prototype fusion score; this is not a calibrated probability.
        fusion_score = float(
            0.5 * gnn_probability
            + 0.5 * min(anomaly_score / 3.0, 1.0)
        )

        fusion_flag = fusion_score >= FUSION_THRESHOLD

        return {
            "decision": (
                "OTP_VERIFY" if fusion_flag else "APPROVE"
            ),
            "risk_level": (
                "MEDIUM" if fusion_flag else "LOW"
            ),
            "risk_score": round(fusion_score, 6),
            "score_type": (
                "prototype_fusion_score_not_calibrated_probability"
            ),
            "breakdown": {
                "reconstruction_error": reconstruction_error,
                "autoencoder_threshold": self.ae_threshold,
                "gnn_fraud_score": round(gnn_probability, 6),
                "gnn_threshold": self.threshold,
                "fusion_threshold": FUSION_THRESHOLD,
                "fusion_flag": bool(fusion_flag),
                "graph_neighbors": int(len(neighbor_ids)),
                "graph_method": (
                    "training-reference feature-similarity graph"
                ),
            },
        }

