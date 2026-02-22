import os, sys
from copy import deepcopy
import inspect
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.scheduler_config import MODEL_CFG, CLUSTER_CFG, DATASET_CFG, CIRCUIT_SOURCE_CFG
from utils.circuit_sources import build_provider
from src.circuit_representation import CircuitRepresentation
from src.circuit_segmentation import segment_circuit
from src.qubit_interaction_graph import build_segment_graph_arrays
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

def build_segment_data_list(rep, segments):
    per_segment_graphs = build_segment_graph_arrays(rep, segments)
    segment_data_list = []
    for seg_id, x_s, edge_index_s, edge_attr_s in per_segment_graphs:
        x_t  = torch.tensor(x_s, dtype=torch.float32)
        ei_t = torch.tensor(edge_index_s, dtype=torch.long)
        ea_t = torch.tensor(edge_attr_s, dtype=torch.float32)
        segment_data_list.append(Data(x=x_t, edge_index=ei_t, edge_attr=ea_t))
    return segment_data_list


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


def analyze_circuit(seed, circuit_cfg, evol_ckpt, cluster_ckpt, total_cost_module, K, tech_names, device="cpu"):
    src_name = circuit_cfg["name"]
    print(f"\n=== circuit_source={src_name} | seed={seed} ===")

    # 1) Generate circuit from provider (seed-controlled)
    provider = build_provider(circuit_cfg, seed_base=seed)
    qc = provider.get(0)
    visualize_circuit(qc)

    # 2) Representation + segmentation + activity viz
    rep = CircuitRepresentation(qc)
    activity = visualize_layer_activity(rep.layers, rep.num_qubits)

    thr = DATASET_CFG["segment_threshold"]
    segments, seg_ids = segment_circuit(rep.layers, threshold=thr)

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

    # 4) Build segment graphs
    segment_data_list = build_segment_data_list(rep, segments)

    # 5) Recreate models and load weights (from config)
    in_dim_node = segment_data_list[0].x.size(1)
    in_dim_edge = segment_data_list[0].edge_attr.size(1) if segment_data_list[0].edge_attr.numel() > 0 else 0

    evol_model = EvolvingGNN(
        in_dim_node=in_dim_node,
        in_dim_edge=in_dim_edge,
        gnn_hidden_dim=MODEL_CFG["gnn_hidden_dim"],
        gnn_out_dim=MODEL_CFG["gnn_out_dim"],
        rnn_hidden_dim=MODEL_CFG["rnn_hidden_dim"],
        heads=MODEL_CFG["heads"],
    ).to(device)

    # SegmentClustering: pass temperature only if your class supports it
    cluster_kwargs = dict(hidden_dim=evol_model.rnn_hidden_dim, num_clusters=K)
    sig = inspect.signature(SegmentClustering.__init__)
    if "temperature" in sig.parameters:
        cluster_kwargs["temperature"] = CLUSTER_CFG["temperature"]

    cluster_module = SegmentClustering(**cluster_kwargs).to(device)

    evol_model.load_state_dict(torch.load(evol_ckpt, map_location=device))
    cluster_module.load_state_dict(torch.load(cluster_ckpt, map_location=device))
    evol_model.eval()
    cluster_module.eval()

    # 6) Run model to get P_seq
    with torch.no_grad():
        h_seq, z_seq = evol_model(segment_data_list)   # list[T] of [N,H]
        P_seq = cluster_module(h_seq)                  # list[T] of [N,K]

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

    # 7) Stack soft assignments: [T, N, K]
    T = len(P_seq)
    N = P_seq[0].size(0)
    P_stack = torch.stack(P_seq, dim=0)  # [T, N, K]

    # 8) Hard assignment via argmax
    hard_idx = P_stack.argmax(dim=2)  # [T, N]
    hard_np = hard_idx.cpu().numpy()

    plt.figure(figsize=(6, 4))
    plt.imshow(hard_np.T, aspect="auto", origin="lower", vmin=0, vmax=K-1, cmap="tab20")
    cbar = plt.colorbar(ticks=list(range(K)))
    cbar.ax.set_yticklabels(tech_names)
    cbar.set_label("Technology")
    plt.xlabel("Segment index"); plt.ylabel("Qubit index")
    plt.title(f"Hard assignments (argmax) ({src_name}, seed={seed})")
    plt.tight_layout(); plt.show()

    # 9) Build one-hot P_seq from hard assignments
    P_seq_hard = []
    for t in range(T):
        P_t = torch.zeros(N, K, dtype=torch.float32, device=device)
        idx = hard_idx[t].to(device)
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

    cfg_path = os.path.join(os.path.dirname(__file__), "..", "data", "cost_config_v3.json")
    config = load_cost_config(cfg_path)
    K = len(config["techs"])
    tech_names = [t.get("name", f"tech{k}") for k, t in enumerate(config["techs"])]

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
        analyze_circuit(seed, CIRCUIT_SOURCE_CFG, evol_ckpt, cluster_ckpt, total_cost_module, K, tech_names, device=device)

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
