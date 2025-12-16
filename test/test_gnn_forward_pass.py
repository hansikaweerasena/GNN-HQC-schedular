import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.circuit_generation import (generate_qft_circuit, generate_random_circuit_custom)
from src.circuit_representation import CircuitRepresentation
from src.circuit_segmentation import ( segment_circuit, analyze_segmentation)
from src.qubit_interaction_graph import (
    build_qubit_interaction_multigraph,
    print_graph_stats,
    print_segment_info,
    visualize_segment_graph,
    build_segment_graph_arrays,
)
from src.circuit_visualization import (
    visualize_circuit,
    visualize_layer_activity,
    visualize_segmentation,
)
import torch
from src.evolving_gnn import EvolvingGNN
from torch_geometric.data import Data


def run_segmentation(rep, threshold):
    segments, seg_ids = segment_circuit(rep.layers, threshold=threshold)
    print("Segment IDs per layer:", seg_ids)
    for seg in segments:
        print(
            f"Segment {seg.segment_idx}: "
            f"layers={seg.layers}, active_qubits={sorted(seg.active_qubits)}"
        )
    return segments, seg_ids


if __name__ == "__main__":

    # qc = generate_qft_circuit(num_qubits=10)
    qc = generate_random_circuit_custom(n_qubits=10, depth=30, gate_density=0.4, seed=42)
    print("✓ Circuit generated")
    print(f"  Depth: {qc.depth()}, Gates: {qc.size()}")

    # Visualize circuit
    visualize_circuit(qc)

    # Representation
    rep = CircuitRepresentation(qc)
    print("\n✓ Circuit representation")
    print(rep.summary())

    # Visualize layer activity
    activity = visualize_layer_activity(rep.layers, rep.num_qubits)

    threshold = 0.3
    segments, seg_ids = run_segmentation(rep, threshold)
    stats = analyze_segmentation(segments, rep.num_qubits)

    print(f"\n✓ Segmentation (threshold={threshold})")
    print(f"  Num segments: {stats['num_segments']}")
    for seg in segments:
        print(seg)

    # Visualize segmentation
    visualize_segmentation(activity, segments, title_suffix=f"(threshold={threshold})")

    G, x, edge_index, edge_attr = build_qubit_interaction_multigraph(rep, seg_ids)

    print_graph_stats(G)
    print_segment_info(G, seg_ids)

    for seg_id in sorted(set(seg_ids)):
        visualize_segment_graph(G, seg_id, title=f"Segment {seg_id} (QFT)")

     # ==============================
    # Evolving GNN forward pass test
    # ==============================
    per_segment_graphs = build_segment_graph_arrays(rep, segments)

    # Convert per-segment arrays to PyG Data objects
    segment_data_list = []
    for seg_id, x_s, edge_index_s, edge_attr_s in per_segment_graphs:
        x_seg = torch.tensor(x_s, dtype=torch.float32)
        ei_seg = torch.tensor(edge_index_s, dtype=torch.long)
        ea_seg = torch.tensor(edge_attr_s, dtype=torch.float32)
        segment_data_list.append(Data(x=x_seg, edge_index=ei_seg, edge_attr=ea_seg))

    print("\n=== EvolvingGNN Test ===")
    print("Num segments:", len(segment_data_list))
    print("Segment 0 x shape:", segment_data_list[0].x.shape)
    print("Segment 0 edge_index shape:", segment_data_list[0].edge_index.shape)
    print("Segment 0 edge_attr shape:", segment_data_list[0].edge_attr.shape)

    in_dim_node_seg = segment_data_list[0].x.size(1)
    in_dim_edge_seg = (
        segment_data_list[0].edge_attr.size(1)
        if segment_data_list[0].edge_attr.numel() > 0
        else 0
    )

    evol_model = EvolvingGNN(
        in_dim_node=in_dim_node_seg,
        in_dim_edge=in_dim_edge_seg,
        gnn_hidden_dim=32,
        gnn_out_dim=16,   # match or differ from encoder; here 16
        rnn_hidden_dim=32,
        heads=4,
    )

    with torch.no_grad():
        h_seq, z_seq = evol_model(segment_data_list)

    print("Len h_seq (num segments):", len(h_seq))
    print("h_seq[0] shape:", h_seq[0].shape)
    print("z_seq[0] shape:", z_seq[0].shape)

    # Print first few qubit embeddings for first segment
    for q in range(min(5, h_seq[0].shape[0])):
        print(f"  Segment 0, Qubit {q}: h = {h_seq[0][q].cpu().numpy()}")


