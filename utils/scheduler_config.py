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
        "num_qubits": 10,
        "num_segments": 5,
        "segment_depth": 6,
    },

    # Only used by "random_custom" provider.
    # If not None, provider samples per-circuit two_qubit_ratio ~ Uniform(low, high).
    "two_qubit_bounds": None,
}

# Example for random_custom:
# CIRCUIT_SOURCE_CFG = {
#     "name": "random_custom",
#     "kwargs": {
#         "num_qubits": 10,
#         "depth": 20,
#         "gate_density": 0.3,
#     },
#     "two_qubit_bounds": (0.1, 0.9),
# }