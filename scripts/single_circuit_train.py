import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.circuit_generation import generate_random_circuit_custom
from src.circuit_representation import CircuitRepresentation
from src.circuit_segmentation import segment_circuit
from src.qubit_interaction_graph import build_segment_graph_arrays
from src.evolving_gnn import EvolvingGNN
from src.clustering_head import SegmentClustering
from src.cost_function import TotalCost
from src.train_utils import train_step


def build_segment_data_list(rep, segments):
    per_segment_graphs = build_segment_graph_arrays(rep, segments)
    segment_data_list = []
    for seg_id, x_s, edge_index_s, edge_attr_s in per_segment_graphs:
        x_t = torch.tensor(x_s, dtype=torch.float32)
        ei_t = torch.tensor(edge_index_s, dtype=torch.long)
        ea_t = torch.tensor(edge_attr_s, dtype=torch.float32)
        segment_data_list.append(Data(x=x_t, edge_index=ei_t, edge_attr=ea_t))
    return segment_data_list


def main():
    # ===== 1) Generate one random circuit =====
    qc = generate_random_circuit_custom(
        n_qubits=10,
        depth=20,
        gate_density=0.3,
        seed=42,
    )
    rep = CircuitRepresentation(qc)
    segments, seg_ids = segment_circuit(rep.layers, threshold=0.3)
    print(f"Num segments: {len(segments)}")

    # ===== 2) Build per-segment graphs =====
    segment_data_list = build_segment_data_list(rep, segments)

    # ===== 3) Build models =====
    in_dim_node = segment_data_list[0].x.size(1)
    in_dim_edge = segment_data_list[0].edge_attr.size(1) if segment_data_list[0].edge_attr.numel() > 0 else 0

    evol_model = EvolvingGNN(
        in_dim_node=in_dim_node,
        in_dim_edge=in_dim_edge,
        gnn_hidden_dim=32,
        gnn_out_dim=16,
        rnn_hidden_dim=32,
        heads=4,
    )

    K = 2  # number of techs/clusters
    cluster_module = SegmentClustering(
        hidden_dim=evol_model.rnn_hidden_dim,
        num_clusters=K,
    )

    # costs per tech (example values)
    exec_costs_1q = [0.05, 0.25]
    exec_costs_2q = [0.10, 0.50]
    idle_costs    = [0.5, 0.1]
    move_costs    = [0.3, 0.3]

    total_cost_module = TotalCost(
        exec_costs_1q,
        exec_costs_2q,
        idle_costs,
        move_costs,
    )

    # ===== 4) Optimizer =====
    optimizer = torch.optim.Adam(
        list(evol_model.parameters()) + list(cluster_module.parameters()),
        lr=1e-3,
    )

    # ===== 5) Training loop =====
    num_epochs = 50
    losses = []
    per_seg_history = []

    for epoch in range(num_epochs):
        loss, per_seg = train_step(
            evol_model,
            cluster_module,
            total_cost_module,
            segment_data_list,
            segments,
            rep,
            optimizer,
        )
        losses.append(loss)
        per_seg_history.append(per_seg.cpu().numpy())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={loss:.4f}, per_segment={per_seg.cpu().numpy()}")

    # ===== 6) Simple loss curve =====
    plt.figure(figsize=(6, 4))
    plt.plot(range(num_epochs), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Total cost")
    plt.title("Training loss over epochs (single 10-qubit circuit)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
