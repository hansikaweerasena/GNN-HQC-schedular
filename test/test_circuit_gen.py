import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.circuit_generation import (
    generate_random_circuit_custom,
    generate_roi_composed_circuit,
)
from src.circuit_representation import CircuitRepresentation
from src.circuit_segmentation import ( segment_circuit, analyze_segmentation)
from src.qubit_interaction_graph import (
    build_qubit_interaction_multigraph,
    print_graph_stats,
    print_segment_info,
    visualize_segment_graph,
    build_segment_graph_arrays,
)
from utils.circuit_visualization import (
    visualize_circuit,
    visualize_layer_activity,
    visualize_segmentation,
)


def run_segmentation(rep, threshold, mode="layer"):
    # Keep this consistent with train_test_eval_debug.py usage.
    segments, seg_ids = segment_circuit(rep.layers, mode=mode, threshold=threshold)
    print("Segment IDs per layer:", seg_ids)
    for seg in segments:
        print(
            f"Segment {seg.segment_idx}: "
            f"layers={seg.layers}, active_qubits={sorted(seg.active_qubits)}"
        )
    return segments, seg_ids


if __name__ == "__main__":

    # ------------------------------
    # Choose what to test
    #   - "roi": new ROI-based generator (spatiotemporal rectangles)
    #   - "random": legacy random circuit
    # ------------------------------
    WHICH = "roi"

    if WHICH == "random":
        qc = generate_random_circuit_custom(
            num_qubits=10,
            depth=40,
            gate_density=0.4,
            seed=42,
            two_qubit_ratio=0.5,
        )
        title = "Random"
    else:
        # New ROI-based generator (see src/circuit_generation.py)
        qc = generate_roi_composed_circuit(
            num_qubits=20,
            num_layers=80,
            option="op2b",          # op1 | op2a | op2b | op3
            n_rois=5,                # excluding idle
            twoq_to_oneq_ratio=0.1,   # default bias for non-2Q-dense ROIs
            idle_density=0.20,
            p_bridge_boundary=(0.15, 0.25),
            p_bridge_interior=(0.05, 0.10),
            noise_1q_prob=0.05,
            noise_2q_prob=0.02,
            measure_frac=0.0,
            # rectangle bounds
            min_block_w=2,
            max_block_w=15,
            min_block_h=2,
            max_block_h=10,
            # long/tall block allocation
            n_long=(2, 3),
            long_w_min=15,
            long_w_max=30,
            n_tall=(1, 3),
            tall_h_min=10,
            tall_h_max=20,
            use_barriers=False,
            seed=643,
            debug=True
        )
        title = "ROI"
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
    seg_mode = "layer"  # try "jaccard" if you want structure-based segmentation
    segments, seg_ids = run_segmentation(rep, threshold, mode=seg_mode)
    stats = analyze_segmentation(segments, rep.num_qubits)

    print(f"\n✓ Segmentation (mode={seg_mode}, threshold={threshold})")
    print(f"  Num segments: {stats['num_segments']}")
    for seg in segments:
        print(seg)

    # Visualize segmentation
    visualize_segmentation(activity, segments, title_suffix=f"({title}, mode={seg_mode}, thr={threshold})")

    # === Global qubit interaction graph (for analysis/debug) ===
    G, x, edge_index, edge_attr = build_qubit_interaction_multigraph(rep, seg_ids)

    print_graph_stats(G)
    print_segment_info(G, seg_ids)

    for seg_id in sorted(set(seg_ids)):
        visualize_segment_graph(G, seg_id, title=f"Segment {seg_id} ({title})")

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



