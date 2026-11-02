"""
FraudGuard AI: experimental local GCN inference benchmark.

This script does not modify project files or model artifacts.
It compares a local, normalization-aware query computation against the
existing HybridInference.predict() implementation.

Run from the project root with:
    python scripts/benchmark_local_gnn.py
"""

import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.hybrid_inference import HybridInference
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
DATA_PATH = "data/processed/validation_data.csv"
SAMPLE_SIZE = 200
FRAUD_SAMPLE = 10
SEED = 42
TOLERANCE = 2e-5


class LocalGCNExperiment:
    """Experimental local computation for the query node of the current 2-layer GCN."""

    def __init__(self, engine):
        self.engine = engine
        self.gnn = engine.gnn
        self.gnn.eval()

        self.reference_x = engine.reference_x
        edge_index = engine.reference_edge_index
        self.edge_index = edge_index
        self.node_count = len(self.reference_x)
        self.k = engine.k

        # GCNConv adds one self-loop per node. The stored reference graph
        # itself has no self-loop entries, so degree = reference degree + 1.
        destination_degree = torch.bincount(
            edge_index[1], minlength=self.node_count
        ).to(torch.float32)
        self.base_degree = destination_degree + 1.0

        # Cache the first layer's linear transform for fixed reference features.
        # The graph-dependent aggregation is still recomputed with query-adjusted
        # degree normalization for the affected query neighborhood.
        with torch.no_grad():
            self.reference_x1_linear = self.gnn.conv1.lin(self.reference_x)

        # Store incoming reference sources per target node for local aggregation.
        edge_src = edge_index[0].cpu().numpy()
        edge_dst = edge_index[1].cpu().numpy()
        incoming = [[] for _ in range(self.node_count)]
        for src, dst in zip(edge_src, edge_dst):
            incoming[int(dst)].append(int(src))
        self.incoming_sources = [
            np.asarray(sources, dtype=np.int64) for sources in incoming
        ]

    @staticmethod
    def _normalization_weight(degree_src, degree_dst):
        return torch.rsqrt(degree_src * degree_dst)

    def fraud_probability(self, scaled_query, neighbor_ids):
        """
        Compute the query's 2-layer GCN output using the query and its
        immediate reference neighbors, while retaining full-graph degrees
        adjusted for the newly added query edges.
        """
        device = self.reference_x.device
        neighbor_ids = np.asarray(neighbor_ids, dtype=np.int64)
        q_degree = torch.tensor(
            float(len(neighbor_ids) + 1), dtype=torch.float32, device=device
        )

        neighbor_tensor = torch.as_tensor(
            neighbor_ids, dtype=torch.long, device=device
        )
        neighbor_degrees = self.base_degree[neighbor_tensor] + 1.0

        query_x = torch.as_tensor(
            scaled_query, dtype=torch.float32, device=device
        ).reshape(1, -1)

        conv1 = self.gnn.conv1
        conv2 = self.gnn.conv2

        with torch.no_grad():
            query_x1_linear = conv1.lin(query_x).squeeze(0)

            # First-layer representation for the query node:
            # its self-loop plus the five query-to-reference edges.
            q_aggregate = self.reference_x1_linear.new_zeros(
                self.reference_x1_linear.shape[1]
            )
            q_aggregate = q_aggregate + query_x1_linear / q_degree
            for neighbor_id, neighbor_degree in zip(
                neighbor_ids, neighbor_degrees
            ):
                q_aggregate = q_aggregate + (
                    self.reference_x1_linear[int(neighbor_id)]
                    * self._normalization_weight(neighbor_degree, q_degree)
                )
            q_h1 = torch.relu(q_aggregate + conv1.bias)

            # First-layer representations for the reference neighbors.
            # Their degrees increase by one because the query connects to them.
            neighbor_h1_rows = []
            neighbor_degree_by_id = {
                int(node_id): neighbor_degrees[i]
                for i, node_id in enumerate(neighbor_ids)
            }

            for target_id, target_degree in zip(
                neighbor_ids, neighbor_degrees
            ):
                aggregate = self.reference_x1_linear.new_zeros(
                    self.reference_x1_linear.shape[1]
                )

                # Original reference edges into this target. The normalization
                # uses updated degrees for query neighbors and original degrees
                # for all other reference nodes.
                for source_id in self.incoming_sources[int(target_id)]:
                    source_degree = neighbor_degree_by_id.get(
                        int(source_id), self.base_degree[int(source_id)]
                    )
                    weight = self._normalization_weight(
                        source_degree, target_degree
                    )
                    aggregate = aggregate + (
                        self.reference_x1_linear[int(source_id)] * weight
                    )

                # Self-loop of this reference target.
                aggregate = aggregate + (
                    self.reference_x1_linear[int(target_id)] / target_degree
                )

                # New incoming edge from query to this target.
                aggregate = aggregate + (
                    query_x1_linear
                    * self._normalization_weight(q_degree, target_degree)
                )

                neighbor_h1_rows.append(torch.relu(aggregate + conv1.bias))

            # Second-layer output for the query only. Its incoming nodes are
            # itself and its selected reference neighbors.
            h1_query_and_neighbors = torch.stack(
                [q_h1] + neighbor_h1_rows, dim=0
            )
            h2_linear = conv2.lin(h1_query_and_neighbors)

            q_h2_aggregate = h2_linear[0] / q_degree
            for i, neighbor_degree in enumerate(neighbor_degrees, start=1):
                q_h2_aggregate = q_h2_aggregate + (
                    h2_linear[i]
                    * self._normalization_weight(neighbor_degree, q_degree)
                )

            q_h2 = torch.relu(q_h2_aggregate + conv2.bias)
            logits = self.gnn.output(q_h2.reshape(1, -1))
            probability = torch.softmax(logits, dim=1)[0, 1].item()

        return float(probability)


def main():
    torch.set_num_threads(1)

    print("=" * 70)
    print("FRAUDGUARD AI - LOCAL GCN OPTIMIZATION EXPERIMENT")
    print("=" * 70)
    print("This is an experimental comparison; it does not change project files.")

    df = pd.read_csv(DATA_PATH)
    fraud = df[df["Class"] == 1].head(FRAUD_SAMPLE)
    legitimate = df[df["Class"] == 0].head(SAMPLE_SIZE - len(fraud))
    sample = pd.concat([legitimate, fraud]).sample(
        frac=1, random_state=SEED
    ).reset_index(drop=True)

    print("Sample transactions:", len(sample))
    print("Fraud cases:", int(sample["Class"].sum()))

    start = time.perf_counter()
    engine = HybridInference()
    initialization_seconds = time.perf_counter() - start
    print(f"Engine initialization: {initialization_seconds:.3f} seconds")

    start = time.perf_counter()
    experiment = LocalGCNExperiment(engine)
    optimizer_init_seconds = time.perf_counter() - start
    print(f"Experimental cache setup: {optimizer_init_seconds:.3f} seconds")

    # Warm-up baseline and experimental path; exclude warm-up from timings.
    first_features = sample.iloc[0][FEATURES].to_dict()
    first_result = engine.predict(first_features)
    first_values = sample.iloc[0][FEATURES].to_numpy(dtype=np.float32)
    first_scaled = (
        (first_values - engine.scaler_mean) / engine.scaler_scale
    ).reshape(1, -1)
    _, first_neighbors = engine.neighbor_search.kneighbors(first_scaled)
    experiment.fraud_probability(first_scaled, first_neighbors[0])

    baseline_latencies = []
    experimental_latencies = []
    gnn_abs_differences = []
    fusion_abs_differences = []
    baseline_scores = []
    experimental_scores = []
    mismatched_flags = 0

    for _, row in sample.iterrows():
        features = row[FEATURES].to_dict()

        start = time.perf_counter()
        baseline_result = engine.predict(features)
        baseline_latencies.append(time.perf_counter() - start)

        # Time the experimental end-to-end path, including the autoencoder,
        # scaling, nearest-neighbor lookup, local GNN computation, and fusion.
        start = time.perf_counter()

        values = row[FEATURES].to_numpy(dtype=np.float32)
        ae_input = values.reshape(1, -1)
        reconstruction = engine.autoencoder.predict(ae_input, verbose=0)
        reconstruction_error = float(
            np.mean(np.square(ae_input - reconstruction))
        )
        anomaly_score = float(
            np.log1p(
                max(reconstruction_error, 0.0) / engine.ae_threshold
            )
        )

        scaled = (
            (values - engine.scaler_mean) / engine.scaler_scale
        ).reshape(1, -1)
        _, neighbors = engine.neighbor_search.kneighbors(scaled)
        local_gnn_probability = experiment.fraud_probability(
            scaled, neighbors[0]
        )
        local_fusion = 0.5 * local_gnn_probability + 0.5 * min(
            anomaly_score / 3.0, 1.0
        )
        experimental_latencies.append(time.perf_counter() - start)

        baseline_gnn = float(
            baseline_result["breakdown"]["gnn_fraud_score"]
        )
        baseline_fusion = float(baseline_result["risk_score"])

        gnn_abs_differences.append(
            abs(baseline_gnn - local_gnn_probability)
        )
        fusion_abs_differences.append(abs(baseline_fusion - local_fusion))
        baseline_scores.append(baseline_fusion)
        experimental_scores.append(local_fusion)

        local_flag = local_fusion >= 0.5
        baseline_flag = baseline_fusion >= 0.5
        if local_flag != baseline_flag:
            mismatched_flags += 1

    gnn_max_diff = float(np.max(gnn_abs_differences))
    fusion_max_diff = float(np.max(fusion_abs_differences))
    baseline_mean = float(np.mean(baseline_latencies))
    local_mean = float(np.mean(experimental_latencies))

    print("\n--- NUMERICAL COMPARISON ---")
    print(f"Maximum absolute GNN score difference: {gnn_max_diff:.10f}")
    print(f"Maximum absolute fusion score difference: {fusion_max_diff:.10f}")
    print(f"Final decision mismatches: {mismatched_flags}/{len(sample)}")
    print(f"Predefined tolerance: {TOLERANCE:g}")
    scores_match = (
        gnn_max_diff <= TOLERANCE
        and fusion_max_diff <= TOLERANCE
        and mismatched_flags == 0
    )
    print("Comparison status:", "PASS" if scores_match else "NEEDS INVESTIGATION")

    print("\n--- LATENCY COMPARISON ---")
    print(f"Baseline mean predict latency: {baseline_mean:.6f} seconds")
    print(
        "Experimental end-to-end mean latency: "
        f"{local_mean:.6f} seconds"
    )
    if local_mean > 0:
        print(f"Measured ratio (baseline / experimental): {baseline_mean/local_mean:.2f}x")
    print(
        "Both timings include autoencoder inference, scaling, neighbor search, "
        "GNN scoring, and fusion."
    )

    print("\nNo project files were modified.")
    print("Do not replace the existing inference engine based on this experiment alone.")


if __name__ == "__main__":
    main()



