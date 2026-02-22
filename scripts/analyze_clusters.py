import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.circuit_generation import generate_random_circuit_custom
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


def analyze_ratio(two_qubit_ratio, evol_ckpt, cluster_ckpt, total_cost_module, K, tech_names, device="cpu"):
    print(f"\n=== two_qubit_ratio = {two_qubit_ratio} ===")

    # 1) Generate one circuit with fixed ratio
    qc = generate_random_circuit_custom(
        n_qubits=10,
        depth=20,
        gate_density=0.3,
        two_qubit_ratio=two_qubit_ratio,
        seed=123,
    )
    visualize_circuit(qc)

    # 2) Representation + segmentation + activity viz
    rep = CircuitRepresentation(qc)
    activity = visualize_layer_activity(rep.layers, rep.num_qubits)
    segments, seg_ids = segment_circuit(rep.layers, threshold=0.3)
    visualize_segmentation(
        activity,
        segments,
        title_suffix=f"(two_qubit_ratio={two_qubit_ratio})",
    )

    # 3) Cost sanity check: all-tech-k baselines (K-generic)
    costs_all = []
    for k in range(K):
        c = compute_total_cost_for_fixed_tech(
            total_cost_module, segments, rep, tech_index=k, device=device
        )
        costs_all.append(c)

    print(
        f"[Cost check] two_qubit_ratio={two_qubit_ratio:.1f}  " +
        ",  ".join([f"Cost(all {tech_names[k]})={costs_all[k]:.3f}" for k in range(K)])
    )

    best_k = min(range(K), key=lambda k: costs_all[k])
    print(f"[Cost check] best single-tech baseline = {tech_names[best_k]}")

    # 4) Build segment graphs
    segment_data_list = build_segment_data_list(rep, segments)

    # 5) Recreate models and load weights (same hyperparams as training)
    in_dim_node = segment_data_list[0].x.size(1)
    in_dim_edge = (
        segment_data_list[0].edge_attr.size(1)
        if segment_data_list[0].edge_attr.numel() > 0
        else 0
    )

    evol_model = EvolvingGNN(
        in_dim_node=in_dim_node,
        in_dim_edge=in_dim_edge,
        gnn_hidden_dim=32,
        gnn_out_dim=16,
        rnn_hidden_dim=32,
        heads=4,
    ).to(device)

    cluster_module = SegmentClustering(
        hidden_dim=evol_model.rnn_hidden_dim,
        num_clusters=K,
    ).to(device)

    evol_model.load_state_dict(torch.load(evol_ckpt, map_location=device))
    cluster_module.load_state_dict(torch.load(cluster_ckpt, map_location=device))
    evol_model.eval()
    cluster_module.eval()

    # 6) Run model to get P_seq
    with torch.no_grad():
        h_seq, z_seq = evol_model(segment_data_list)   # list[T] of [N,H]
        P_seq = cluster_module(h_seq)                  # list[T] of [N,2]

    # --- Cost + dashboard for SOFT schedule ---
    cost_soft = total_cost_module(P_seq, segments, rep, debug=False)
    drivers_soft = compute_drivers(total_cost_module, P_seq, segments, rep, device=device)
    plot_cost_dashboard(
        cost_soft,
        drivers_soft,
        title_prefix=f"SOFT schedule (two_qubit_ratio={two_qubit_ratio})",
        show=True,
        save_path_prefix=None,  # or f"soft_r{two_qubit_ratio}"
    )

    # 7) Stack soft assignments: [T, N, K]
    T = len(P_seq)
    N = P_seq[0].size(0)
    K = P_seq[0].size(1)
    P_stack = torch.stack(P_seq, dim=0)  # [T, N, K]

    # Soft visualization:
    # - if K==2, show P(tech1)
    # - else, show max probability (confidence)
    if K == 2:
        M_np = P_stack[:, :, 1].cpu().numpy()  # [T, N]
        plt.figure(figsize=(6, 4))
        plt.imshow(M_np.T, aspect="auto", origin="lower", cmap="bwr", vmin=0.0, vmax=1.0)
        plt.colorbar(label=f"P({tech_names[1]})")
        plt.xlabel("Segment index"); plt.ylabel("Qubit index")
        plt.title(f"Soft cluster assignments (two_qubit_ratio={two_qubit_ratio})")
        plt.tight_layout(); plt.show()
    else:
        conf_np = P_stack.max(dim=2).values.cpu().numpy()  # [T, N]
        # Optional: entropy heatmap for K>2 (uncertainty)
        eps = 1e-12
        P_np = P_stack.cpu().numpy()  # [T,N,K]
        entropy = -(P_np * np.log(P_np + eps)).sum(axis=2)  # [T,N]
        plt.figure(figsize=(6, 4))
        plt.imshow(entropy.T, aspect="auto", origin="lower")
        plt.colorbar(label="Assignment entropy")
        plt.xlabel("Segment index"); plt.ylabel("Qubit index")
        plt.title(f"Soft assignment uncertainty (entropy) (two_qubit_ratio={two_qubit_ratio})")
        plt.tight_layout(); plt.show()

    # 8 Hard assignment via argmax for ANY K: [T, N]
    hard_idx = P_stack.argmax(dim=2)  # [T, N]
    hard_np = hard_idx.cpu().numpy()

    plt.figure(figsize=(6, 4))
    plt.imshow(hard_np.T, aspect="auto", origin="lower", vmin=0, vmax=K-1, cmap="tab20")
    cbar = plt.colorbar(ticks=list(range(K)))
    cbar.ax.set_yticklabels(tech_names)
    cbar.set_label("Technology")
    plt.xlabel("Segment index"); plt.ylabel("Qubit index")
    plt.title(f"Hard cluster assignments (argmax) (two_qubit_ratio={two_qubit_ratio})")
    plt.tight_layout(); plt.show()

    # 9 Tech usage per segment (fraction of qubits assigned to each tech)
    usage = np.zeros((T, K), dtype=float)
    for t in range(T):
        idx = hard_np[t]  # [N]
        for k in range(K):
            usage[t, k] = (idx == k).mean()

    plt.figure(figsize=(7, 3.5))
    for k in range(K):
        plt.plot(np.arange(T), usage[:, k], label=tech_names[k])
    plt.xlabel("Segment index")
    plt.ylabel("Fraction of qubits (hard assigned)")
    plt.title(f"Tech usage over segments (two_qubit_ratio={two_qubit_ratio})")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()

    # 10) Build P_seq from hard assignments (one-hot), ANY K
    P_seq_hard = []
    for t in range(T):
        P_t = torch.zeros(N, K, dtype=torch.float32, device=device)
        idx = hard_idx[t].to(device)  # [N]
        P_t[torch.arange(N, device=device), idx] = 1.0
        P_seq_hard.append(P_t)

    # 11) Compute cost for this hard schedule
    res = total_cost_module(P_seq_hard, segments, rep, debug=False)

    # Optional: show final overall tech usage across all segments/qubits
    overall_counts = np.bincount(hard_np.reshape(-1), minlength=K)
    overall_frac = overall_counts / overall_counts.sum()
    print("[Hard schedule] overall tech fractions:",
        ", ".join([f"{tech_names[k]}={overall_frac[k]:.2f}" for k in range(K)]))
    total = res["total_cost"].item()
    print(f"[Hard schedule] total_cost={total:.3f}")

    # --- Cost + dashboard for HARD schedule ---
    drivers_hard = compute_drivers(total_cost_module, P_seq_hard, segments, rep, device=device)
    plot_cost_dashboard(
        res,
        drivers_hard,
        title_prefix=f"HARD schedule (two_qubit_ratio={two_qubit_ratio})",
        show=True,
        save_path_prefix=None,  # or f"hard_r{two_qubit_ratio}"
    )



def main():
    evol_ckpt    = "evol_model_final.pt"
    cluster_ckpt = "cluster_head_final.pt"
    device = "cpu"  # or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_path = os.path.join(os.path.dirname(__file__), "..", "data", "cost_config_v3.json")
    config = load_cost_config(cfg_path)
    K = len(config["techs"])
    tech_names = [t.get("name", f"tech{k}") for k, t in enumerate(config["techs"])]

    total_cost_module = TotalCost(config).to(device)

    for r in [0.1, 0.5, 0.9, 1.0]:
        analyze_ratio(r, evol_ckpt, cluster_ckpt, total_cost_module, K, tech_names, device=device)

if __name__ == "__main__":
    main()
