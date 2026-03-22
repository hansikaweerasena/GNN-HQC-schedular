# analyze_clusters.py

import os, sys
from copy import deepcopy
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.circuit_sources import build_provider
from src.circuit_representation import CircuitRepresentation
from src.circuit_segmentation import segment_circuit
from src.qubit_interaction_graph import (
    build_layer_graph_arrays,
    compute_window_sizes_from_config,
    NODE_FEAT_DIM,
    EDGE_FEAT_DIM,
)
from src.inference_utils import enforce_capacity_sequence
from src.evolving_gnn import EvolvingGNN
from src.clustering_head import SegmentClustering
from utils.circuit_visualization import (
    visualize_circuit,
    visualize_layer_activity,
    visualize_segmentation,
)
from src.cost_function import TotalCost
from utils.cost_config_reader import load_cost_config
from utils.plot_utils import compute_drivers, plot_cost_dashboard
from utils.print_utils import print_run_config_analyze
from utils.cost_config_reader import load_scheduler_cfg

def build_layer_data_list(rep, w_short: int, w_long: int, device="cpu"):
    """Build one PyG Data object per circuit layer — matches the training pipeline exactly."""
    arrays = build_layer_graph_arrays(rep, w_short, w_long)
    return [
        Data(
            x          = torch.tensor(x_np,  dtype=torch.float32).to(device),
            edge_index = torch.tensor(ei_np, dtype=torch.long).to(device),
            edge_attr  = torch.tensor(ea_np, dtype=torch.float32).to(device),
        )
        for x_np, ei_np, ea_np in arrays
    ]


def compute_total_cost_for_fixed_tech(total_cost_module, segments, rep, tech_index, device="cpu"):
    N = rep.num_qubits
    K = total_cost_module.K
    T = len(segments)

    P_seq = []
    for _ in range(T):
        P_t = torch.zeros(N, K, dtype=torch.float32, device=device)
        P_t[:, tech_index] = 1.0
        P_seq.append(P_t)

    out = total_cost_module(P_seq, segments, rep, debug=False)
    return out["total_cost"].item()


def analyze_circuit(seed, circuit_cfg, dataset_cfg, model_cfg, cluster_cfg, evol_ckpt, cluster_ckpt, total_cost_module, K, tech_names, w_short, w_long, config, device="cpu"):
    src_name = circuit_cfg["name"]
    print(f"\n=== circuit_source={src_name} | seed={seed} ===")

    # 1) Generate circuit from provider (seed-controlled)
    provider = build_provider(circuit_cfg, seed_base=seed)
    qc = provider.get(0)
    visualize_circuit(qc)

    # 2) Representation + segmentation + activity viz
    rep = CircuitRepresentation(qc)
    activity = visualize_layer_activity(rep.layers, rep.num_qubits)

    thr = dataset_cfg["segment_threshold"]
    segment_mode = dataset_cfg["segmentation_mode"]
    segments, seg_ids = segment_circuit(rep.layers, mode=segment_mode, threshold=thr)

    visualize_segmentation(
        activity,
        segments,
        title_suffix=f"({src_name}, seed={seed}, thr={thr})",
    )

    # 3) Cost sanity check: all-tech-k baselines (K-generic)
    costs_all = []
    for k in range(K):
        c = compute_total_cost_for_fixed_tech(total_cost_module, segments, rep, tech_index=k, device=device)
        costs_all.append(c)

    print(
        f"[Cost check] {src_name} seed={seed}  " +
        ",  ".join([f"Cost(all {tech_names[k]})={costs_all[k]:.3f}" for k in range(K)])
    )
    best_k = min(range(K), key=lambda k: costs_all[k])
    best = costs_all[best_k]

    # 4) Build layer graphs (one per circuit layer — matches training pipeline)
    layer_data_list = build_layer_data_list(rep, w_short, w_long, device=device)

    # 5) Recreate models and load weights
    evol_model = EvolvingGNN(
        node_feat_dim  = NODE_FEAT_DIM,
        edge_feat_dim  = EDGE_FEAT_DIM,
        mlp_hidden_dim = model_cfg["mlp_hidden_dim"],
        mlp_out_dim    = model_cfg["mlp_out_dim"],
        gnn_out_dim    = model_cfg["gnn_out_dim"],
        gru_hidden_dim = model_cfg["gru_hidden_dim"],
        heads          = model_cfg["heads"],
        dropout        = model_cfg.get("dropout", 0.1),
        bptt_steps     = model_cfg.get("bptt_steps", 3),
        activation     = model_cfg.get("activation", "relu"),
    ).to(device)

    # SegmentClustering: pass config-driven parameters
    cluster_module = SegmentClustering(
        hidden_dim          = evol_model.rnn_hidden_dim,
        num_clusters        = K,
        proj_hidden_dim     = cluster_cfg.get("proj_hidden_dim"),
        temperature_init    = cluster_cfg.get("temperature_init", cluster_cfg.get("temperature", 3.0)),
        temperature_min     = cluster_cfg.get("temperature_min", 0.5),
        temperature_gamma   = cluster_cfg.get("temperature_gamma", 0.95),
        neighbor_alpha_init = cluster_cfg.get("neighbor_alpha_init", 0.0),
    ).to(device)

    evol_model.load_state_dict(torch.load(evol_ckpt, map_location=device))
    cluster_module.load_state_dict(torch.load(cluster_ckpt, map_location=device))
    evol_model.eval()
    cluster_module.eval()

    # 6) Run model to get P_seq
    with torch.no_grad():
        h_seq, z_seq = evol_model(layer_data_list)
        P_seq = cluster_module(h_seq, graphs=layer_data_list)

    # --- Cost + dashboard for SOFT schedule (keep your existing dashboard code) ---
    cost_soft = total_cost_module(P_seq, segments, rep, debug=False)
    drivers_soft = compute_drivers(total_cost_module, P_seq, segments, rep, device=device)
    plot_cost_dashboard(
        cost_soft,
        drivers_soft,
        title_prefix=f"SOFT schedule ({src_name}, seed={seed})",
        show=True,
        save_path_prefix=None,
    )

    # 7) Stack soft assignments for display: [T, N, K]
    T = len(P_seq)
    N = P_seq[0].size(0)

    # 8) Hard assignment via capacity-feasible inference repair
    # Simple argmax is NOT used here — it can produce infeasible schedules where
    # a technology is assigned more qubits than its capacity allows.
    # enforce_capacity_sequence repairs violations by reassigning the least-confident
    # qubits to the best technology with remaining capacity.
    caps = torch.tensor(
        [float(t["capacity"]["max_qubits"]) for t in config["techs"]],
        dtype=torch.float32,
        device=device,
    )
    hard_assignments = enforce_capacity_sequence(P_seq, caps)  # list of [N] int tensors
    hard_np = torch.stack(hard_assignments, dim=0).cpu().numpy()  # [T, N]

    plt.figure(figsize=(6, 4))
    plt.imshow(hard_np.T, aspect="auto", origin="lower", vmin=0, vmax=K-1, cmap="tab20")
    cbar = plt.colorbar(ticks=list(range(K)))
    cbar.ax.set_yticklabels(tech_names)
    cbar.set_label("Technology")
    plt.xlabel("Segment index"); plt.ylabel("Qubit index")
    plt.title(f"Hard assignments - capacity feasible ({src_name}, seed={seed})")
    plt.tight_layout(); plt.show()

    # 9) Build one-hot P_seq from capacity-feasible hard assignments
    P_seq_hard = []
    for t_idx in range(T):
        P_t = torch.zeros(N, K, dtype=torch.float32, device=device)
        idx = hard_assignments[t_idx].to(device)
        P_t[torch.arange(N, device=device), idx] = 1.0
        P_seq_hard.append(P_t)

    res = total_cost_module(P_seq_hard, segments, rep, debug=False)
    total = res["total_cost"].item()

    print(f"[Hard schedule] total_cost={total:.3f}")
    gap = (total - best) / best * 100 if best > 0 else float("nan")
    print(f"[Gap vs best single-tech baseline ({tech_names[best_k]})] {gap:+.2f}%")

    drivers_hard = compute_drivers(total_cost_module, P_seq_hard, segments, rep, device=device)
    plot_cost_dashboard(
        res,
        drivers_hard,
        title_prefix=f"HARD schedule ({src_name}, seed={seed})",
        show=True,
        save_path_prefix=None,
    )

def main():
    evol_ckpt    = "evol_model_final.pt"
    cluster_ckpt = "cluster_head_final.pt"
    device = "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--sched_cfg", type=str, default="configs.scheduler_config")
    parser.add_argument("--cost_cfg", type=str, default="cost_config_v3.json")
    args = parser.parse_args()

    MODEL_CFG, CLUSTER_CFG, TRAIN_CFG, DATASET_CFG, CIRCUIT_SOURCE_CFG = load_scheduler_cfg(args.sched_cfg)

    config = load_cost_config(args.cost_cfg)
    K = len(config["techs"])
    tech_names = [t.get("name", f"tech{k}") for k, t in enumerate(config["techs"])]

    w_short, w_long = compute_window_sizes_from_config(config)
    print(f"Window sizes: W_short={w_short}, W_long={w_long}")

    total_cost_module = TotalCost(config).to(device)

    print_run_config_analyze(
        device=device,
        K=K,
        tech_names=tech_names,
        MODEL_CFG=MODEL_CFG,
        CLUSTER_CFG=CLUSTER_CFG,
        DATASET_CFG=DATASET_CFG,
        CIRCUIT_SOURCE_CFG=CIRCUIT_SOURCE_CFG,
    )

    # If you want ROI / fixed-seed analysis:
    fixed_seeds = [200, 30]
    for seed in fixed_seeds:
        analyze_circuit(seed, CIRCUIT_SOURCE_CFG, DATASET_CFG, MODEL_CFG, CLUSTER_CFG, evol_ckpt, cluster_ckpt, total_cost_module, K, tech_names, w_short, w_long, config, device=device)

    # Optional: keep your old ratio sweep when source is random_custom
    # ratios = [0.1, 0.5, 0.9, 1.0]
    # if CIRCUIT_SOURCE_CFG["name"] == "random_custom":
    #     for r in ratios:
    #         cfg_r = deepcopy(CIRCUIT_SOURCE_CFG)
    #         cfg_r["two_qubit_bounds"] = None
    #         cfg_r["kwargs"] = dict(cfg_r.get("kwargs", {}))
    #         cfg_r["kwargs"]["two_qubit_ratio"] = r
    #         analyze_circuit(123, cfg_r, evol_ckpt, cluster_ckpt, total_cost_module, K, tech_names, device=device)

if __name__ == "__main__":
    main()
