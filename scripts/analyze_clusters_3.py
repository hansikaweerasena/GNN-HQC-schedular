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
    """
    Compute total cost if ALL qubits use a single tech (0 or 1) in all segments.
    """
    T = len(segments)
    N = rep.num_qubits
    K = total_cost_module.K

    P_seq = []
    for _ in range(T):
        P_t = torch.zeros(N, K, device=device)
        P_t[:, tech_index] = 1.0
        P_seq.append(P_t)

    with torch.no_grad():
        res = total_cost_module(P_seq, segments, rep, debug=False)
    return res["total_cost"].item()


def analyze_ratio(two_qubit_ratio, evol_ckpt, cluster_ckpt, total_cost_module, device="cpu"):
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


    # 3) Cost sanity check: all-tech0, all-tech1, all-tech2
    cost_all_0 = compute_total_cost_for_fixed_tech(total_cost_module, segments, rep, tech_index=0, device=device)
    cost_all_1 = compute_total_cost_for_fixed_tech(total_cost_module, segments, rep, tech_index=1, device=device)
    cost_all_2 = compute_total_cost_for_fixed_tech(total_cost_module, segments, rep, tech_index=2, device=device)
    cost_all_3 = compute_total_cost_for_fixed_tech(total_cost_module, segments, rep, tech_index=3, device=device)

    print(
        f"[Cost check] two_qubit_ratio={two_qubit_ratio:.1f}  "
        f"all0={cost_all_0:.3f},  all1={cost_all_1:.3f},  all2={cost_all_2:.3f}, all2={cost_all_3:.3f}"
    )
    best = min(cost_all_0, cost_all_1, cost_all_2, cost_all_3)


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

    K = 4
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

    # 7) Build [T, N] matrix of P(tech1)
    T = len(P_seq)
    N = P_seq[0].size(0)
    K = total_cost_module.K   # should be 3

    # Stack → [T, N, K]
    P_stack = torch.stack(P_seq, dim=0)

    # Hard assignments: argmax over tech dimension → [T, N], values in {0,1,2}
    hard_assign = P_stack.argmax(dim=2)          # [T, N]
    hard_np = hard_assign.cpu().numpy()


    # 8) Soft heatmap of cluster probabilities
    M = P_stack[:, :, 2]          # [T, N] P(tech2)
    M_np = M.cpu().numpy()

    plt.figure(figsize=(6, 4))
    plt.imshow(
        M_np.T,
        aspect="auto",
        origin="lower",
        cmap="bwr",
        vmin=0.0,
        vmax=1.0,
    )
    plt.colorbar(label="P(tech2)")
    plt.xlabel("Segment index")
    plt.ylabel("Qubit index")
    plt.title(f"Soft P(tech2) (two_qubit_ratio={two_qubit_ratio})")
    plt.tight_layout()
    plt.show()


    # 9) Hard 0/1 heatmap
    plt.figure(figsize=(6, 4))
    plt.imshow(
        hard_np.T,
        aspect="auto",
        origin="lower",
        cmap="tab10",  # categorical colormap: different color per tech
        vmin=0,
        vmax=K-1,
    )
    plt.xlabel("Segment index")
    plt.ylabel("Qubit index")
    plt.title(f"Hard tech assignments (two_qubit_ratio={two_qubit_ratio})")
    plt.tight_layout()
    plt.show()


    # 10) Build P_seq from hard assignments
    # 10) Build P_seq from hard assignments
    P_seq_hard = []
    T, N = hard_np.shape
    K = total_cost_module.K  # = 3

    for t in range(T):
        P_t = torch.zeros(N, K, dtype=torch.float32)
        P_t[torch.arange(N), hard_np[t]] = 1.0
        P_seq_hard.append(P_t)

    # 11) Compute cost for this hard schedule
    res = total_cost_module(P_seq_hard, segments, rep, debug=False)
    total = res["total_cost"].item()

    print(f"[Hard schedule] total_cost={total:.3f}")
    gap = (total - best) / best * 100
    print(f"[Gap vs best baseline] {gap:+.2f}%")




def main():
    evol_ckpt    = "evol_model_final.pt"
    cluster_ckpt = "cluster_head_final.pt"
    device = "cpu"

    # Use the SAME costs as in training
    exec_costs_1q = [0.03, 0.06, 0.10, 0.11]   #tech0 best for 1q
    exec_costs_2q = [0.08, 0.06, 0.15, 0.10]   #tech1 best for 2q
    idle_costs    = [0.05, 0.04, 0.02, 0.04]   #tech2 best for idle
    move_costs    = [0.01, 0.01, 0.01, 0.01]  # symmetric for now


    total_cost_module = TotalCost(
        exec_costs_1q,
        exec_costs_2q,
        idle_costs,
        move_costs,
    )

    for r in [0.1, 0.5, 0.9, 1.0]:
        analyze_ratio(r, evol_ckpt, cluster_ckpt, total_cost_module, device=device)


if __name__ == "__main__":
    main()
