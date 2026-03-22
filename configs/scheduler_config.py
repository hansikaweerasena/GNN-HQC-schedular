# utils/scheduler_config.py

# ----------------------------
# Model (GNN/RNN) parameters
# ----------------------------
# Dimensions follow the pipeline:
#   raw node features [N, node_feat_dim]
#   -> MLP out        [N, mlp_out_dim]      (mlp_hidden_dim is the hidden width)
#   -> GATv2 out      [N, gnn_out_dim]
#   -> GRU hidden     [N, gru_hidden_dim]   (= clustering head input dim)
MODEL_CFG = {
    # Input/feature dims — must match qubit_interaction_graph.NODE_FEAT_DIM / EDGE_FEAT_DIM
    "node_feat_dim":  16,
    "edge_feat_dim":  5,

    # MLP node encoder
    "mlp_hidden_dim": 32,
    "mlp_out_dim":    64,

    # GATv2 spatial encoder
    "gnn_out_dim":    64,
    "heads":          4,

    # GRU temporal encoder
    "gru_hidden_dim": 64,

    # Regularisation
    "dropout":        0.1,

    # Truncated BPTT: detach hidden state every N layers (0 = no truncation)
    "bptt_steps":     3,

    # Activation in MLP: "relu" or "gelu"
    "activation":     "relu",
}

# ----------------------------
# Clustering head parameters
# ----------------------------
# The clustering head converts GRU embeddings [N, H] into soft technology
# assignments P_t [N, K] via:
#   1. Per-qubit MLP projection (nonlinear feature space for assignment)
#   2. Cosine similarity against L2-normalised prototypes
#   3. Sparse neighbor-logit coordination (edge-restricted message passing)
#   4. Temperature-scaled softmax with epoch annealing
CLUSTER_CFG = {
    # Per-qubit MLP projection hidden dim (None = same as hidden_dim)
    "proj_hidden_dim": None,

    # Temperature annealing schedule: T(e) = max(T_min, T_init * gamma^e)
    #   Early training: T_init=3.0 -> soft exploratory assignments
    #   Late training:  T_min=0.5  -> sharp decisive assignments
    "temperature_init":  3.0,
    "temperature_min":   0.5,
    "temperature_gamma": 0.95,

    # Neighbor-logit coordination initial mixing weight (raw logit, sigmoid-bounded).
    # 0.0 -> sigmoid(0)=0.5 initial mixing; learned during training.
    "neighbor_alpha_init": 0.0,
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
# Dataset parameters
# ----------------------------
# segmentation_mode="layer" means one "segment" per circuit layer.
# segment_threshold is kept for API compat but unused in layer mode.
DATASET_CFG = {
    "segmentation_mode": "layer",
    "segment_threshold": 0.3,
}

# ----------------------------
# Circuit source selection
# ----------------------------
# Set CIRCUIT_SOURCE_CFG["name"] to:
#   - "random_custom"
#   - "roi_composed"
#
# For future sources, add them to utils/circuit_sources.py registry.
CIRCUIT_SOURCE_CFG = {
    "name": "roi_composed",

    # These kwargs are passed directly into generate_roi_composed_circuit(...)
    "kwargs": {
        # Core size
        # "num_qubits": 20,
        "num_qubits": 30,
        "num_layers": 80,

        # Fallback option if no mix is enabled
        # One of: "op1", "op2a", "op2b", "op3"
        "option": "op2a",

        # Per-circuit ROI subset size (excluding idle)
        "n_rois": 5,

        # Defaults / targets
        "twoq_to_oneq_ratio": 0.4,
        "idle_density": [0.15, 0.40],  # fraction of total (num_qubits * num_layers) canvas (VQE, QFT have 10- 15 and general its 35 -50%)

        # Bridge probabilities are sampled INSIDE the generator per circuit
        # from these ranges
        "p_bridge_boundary": [0.10, 0.25],
        "p_bridge_interior": [0.01, 0.05],

        # Block-local noise (softly breaks layer guarantees)
        "noise_1q_prob": 0.05,
        "noise_2q_prob": 0.02,

        # End-only measurements (set 0.0 to omit measure ops)
        "measure_frac": 0.0,

        # Rectangle bounds - 30
        "min_block_w": 2,
        "max_block_w": 15,
        "min_block_h": 2,
        "max_block_h": 10,

        # Rectangle bounds - 20
        # "min_block_w": 2,
        # "max_block_w": 15,
        # "min_block_h": 2,
        # "max_block_h": 8,

        # Long/tall blocks (spatial/temporal modularity proxies)- 30
        "n_long": (2, 5),
        "long_w_min": 12,
        "long_w_max": 18,
        "n_tall": (1, 3),
        "tall_h_min": 6,
        "tall_h_max": 10,


        # Long/tall blocks (spatial/temporal modularity proxies)- 20
        # "n_long": (2, 5),
        # "long_w_min": 12,
        # "long_w_max": 18,
        # "n_tall": (1, 3),
        # "tall_h_min": 6,
        # "tall_h_max": 10,

        "use_barriers": False,
    },

    # Provider-side per-sample sampling knobs.
    # For ROI circuits, this samples the option per circuit using the seed derived
    # from seed_base + idx, so train/test remain reproducible.
    #
    # Remove this block (or set to {}) if you want a single fixed option only.
    "sampled_kwargs": {
        "option_mix": {
            "op1": 0.25,
            "op2a": 0.25,
            "op2b": 0.25,
            "op3": 0.25,
        }
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
#     "sampled_kwargs": {},
#     "two_qubit_bounds": (0.1, 0.9),
# }
