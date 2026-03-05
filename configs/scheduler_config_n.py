# utils/scheduler_config.py

# ----------------------------
# Model (GNN/RNN) parameters
# ----------------------------
MODEL_CFG = {
    "gnn_hidden_dim": 32,
    "gnn_out_dim": 16,
    "rnn_hidden_dim": 32,
    "heads": 4,
}

# ----------------------------
# Clustering head parameters
# ----------------------------
CLUSTER_CFG = {
    "temperature": 5.0,
}

# ----------------------------
# Train/Test parameters
# ----------------------------
TRAIN_CFG = {
    "n_samples_train": 800,
    "n_samples_test": 200,
    "batch_size": 4,
    "n_epochs": 50,
    "lr": 1e-4,

    # different seed bases => no overlap between train/test
    "seed_base_train": 42,
    "seed_base_test": 1000,
}

# ----------------------------
# Dataset/segmentation parameters
# ----------------------------
DATASET_CFG = {
    "segmentation_mode": "jaccard",   # or "layer"
    "segment_threshold": 0.3,
}

# ----------------------------
# Circuit source selection
# ----------------------------
# Set CIRCUIT_SOURCE_CFG["name"] to:
#   - "random_custom"
#   - "roi_composed"
#
# For future sources, we just add them to utils/circuit_sources.py registry.
CIRCUIT_SOURCE_CFG = {
    "name": "roi_composed",
    "kwargs": {
        # Core size
        "num_qubits": 50,
        "num_layers": 160,

        # One of: "op1", "op2a", "op2b", "op3"
        "option": "op2a",

        # Per-circuit ROI subset size (excluding idle)
        "n_rois": 3,

        # Defaults / targets
        "twoq_to_oneq_ratio": 0.7,
        "idle_density": 0.20,  # volume fraction of (num_qubits * num_layers)

        # Bridge probabilities are sampled per circuit from these ranges
        "p_bridge_boundary": (0.10, 0.20),
        "p_bridge_interior": (0.01, 0.05),

        # Block-local noise (softly breaks layer guarantees)
        "noise_1q_prob": 0.02,
        "noise_2q_prob": 0.004,

        # End-only measurements (set 0.0 to omit measure ops)
        "measure_frac": 0.0,

        # Rectangle bounds
        "min_block_w": 2,
        "max_block_w": 18,
        "min_block_h": 2,
        "max_block_h": 16,

        # Long/tall blocks (spatial/temporal modularity proxies)
        "n_long": (2, 5),
        "long_w_min": 12,
        "long_w_max": 40,
        "n_tall": (1, 3),
        "tall_h_min": 10,
        "tall_h_max": 30,

        "use_barriers": True,
    },

    # Only used by "random_custom" provider.
    # If not None, provider samples per-circuit two_qubit_ratio ~ Uniform(low, high).
    "two_qubit_bounds": None,
}

# Example for random_custom:
# CIRCUIT_SOURCE_CFG = {
#     "name": "random_custom",
#     "kwargs": {
#         "num_qubits": 50,
#         "depth": 160,
#         "gate_density": 0.3,
#         "use_barriers": True,
#     },
#     "two_qubit_bounds": (0.1, 0.9),
# }
