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

    # === Global qubit interaction graph (for analysis/debug) ===
    G, x, edge_index, edge_attr = build_qubit_interaction_multigraph(rep, seg_ids)

    print_graph_stats(G)
    print_segment_info(G, seg_ids)

    for seg_id in sorted(set(seg_ids)):
        visualize_segment_graph(G, seg_id, title=f"Segment {seg_id} (QFT)")

    # === Per-segment graphs (for evolving GNN) ===
    per_segment_graphs = build_segment_graph_arrays(rep, segments)

    print("\n=== Per-segment graph arrays (for evolving GNN) ===")
    for seg_id, x_s, edge_index_s, edge_attr_s in per_segment_graphs:
        print(
            f"Segment {seg_id}: "
            f"x_s {x_s.shape}, "
            f"edge_index_s {edge_index_s.shape}, "
            f"edge_attr_s {edge_attr_s.shape}"
        )



