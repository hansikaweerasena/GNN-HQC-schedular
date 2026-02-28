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
from utils.circuit_visualization import (
    visualize_circuit,
    visualize_layer_activity,
    visualize_segmentation,
)
import torch
from src.evolving_gnn import EvolvingGNN
from torch_geometric.data import Data
from src.clustering_head import SegmentClustering
from src.cost_function import TotalCost
from utils.cost_config_reader import load_cost_config
from utils.plot_utils import compute_drivers, plot_cost_dashboard


def run_segmentation(rep, threshold, mode="jaccard"):
    segments, seg_ids = segment_circuit(rep.layers, mode=mode, threshold=threshold)
    print("Segment IDs per layer:", seg_ids)
    for seg in segments:
        print(
            f"Segment {seg.segment_idx}: "
            f"layers={seg.layers}, active_qubits={sorted(seg.active_qubits)}"
        )
    return segments, seg_ids

def _print_vec(name, x, max_items=5, fmt="{:.6f}"):
    if x is None:
        print(f"{name}: None")
        return
    try:
        arr = x.detach().cpu().flatten().tolist()
        head = arr[:max_items]
        tail = arr[-max_items:] if len(arr) > max_items else []
        if len(arr) <= max_items:
            s = ", ".join(fmt.format(v) for v in head)
            print(f"{name} [{len(arr)}]: {s}")
        else:
            s1 = ", ".join(fmt.format(v) for v in head)
            s2 = ", ".join(fmt.format(v) for v in tail)
            print(f"{name} [{len(arr)}]: {s1} ... {s2}")
    except Exception as e:
        print(f"{name}: <non-tensor or unsupported> ({e})")


def dump_cost_out_debug(cost_out, max_items=5):
    """
    Print all debug fields that exist in cost_out.
    This assumes debug=True was passed to TotalCost.forward.
    """
    print("\n=== DEBUG DUMP (available keys) ===")
    keys = sorted(cost_out.keys())
    print("Keys:", keys)

    # Always present core arrays
    for k in [
        "per_segment_total",
        "per_segment_exec",
        "per_segment_idle",
        "per_segment_comm",
        "per_segment_move",
        "per_segment_C1q",
        "per_segment_C2q_local",
        "per_segment_Cm",
    ]:
        if k in cost_out:
            _print_vec(k, cost_out[k], max_items=max_items)

    # Exec debug
    for k in [
        "exec_num_edges",
        "exec_twoq_ops",
        "exec_gamma",
        "exec_avg_local_prob",
        "exec_1q_ops",
        "exec_meas_ops",
    ]:
        if k in cost_out:
            _print_vec(k, cost_out[k], max_items=max_items, fmt="{:.4f}")

    # Comm debug
    for k in [
        "comm_num_edges",
        "comm_twoq_ops",
        "comm_avg_cut_prob",
    ]:
        if k in cost_out:
            _print_vec(k, cost_out[k], max_items=max_items, fmt="{:.4f}")

    # Move debug
    for k in [
        "move_total_change",
        "move_avg_change",
    ]:
        if k in cost_out:
            _print_vec(k, cost_out[k], max_items=max_items, fmt="{:.4f}")

    # Idle debug
    for k in [
        "idle_dt",
        "idle_sum_w_invT",
    ]:
        if k in cost_out:
            _print_vec(k, cost_out[k], max_items=max_items, fmt="{:.6f}")


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

    threshold = 0.2
    segments, seg_ids = run_segmentation(rep, threshold, mode="layer")
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

    # ==============================
    # Clustering head test
    # ==============================

    # Load v3 config from ../data/
    # Example filename: ../data/cost_config_v3.json
    config = load_cost_config("cost_config_v3.json")
    num_clusters = len(config["techs"])

    # ==============================
    # Clustering head test
    # ==============================
    num_clusters = 2

    cluster_module = SegmentClustering(
        hidden_dim=evol_model.rnn_hidden_dim,
        num_clusters=num_clusters,
    )

    with torch.no_grad():
        P_seq = cluster_module(h_seq)  # list of [num_qubits, K]

    print("\n=== Clustering Head Test ===")
    print("Num segments (P_seq):", len(P_seq))
    print("P_seq[0] shape:", P_seq[0].shape)  # [num_qubits, K]

    # Check that probabilities per qubit sum to 1
    for seg_idx, segment in enumerate(P_seq):
        print(f"\n=== Segment {seg_idx} ===")
        for q_idx, probs in enumerate(segment):
            probs_np = probs.cpu().numpy()
            print(f" Qubit {q_idx} probs: {probs_np}")
            print(f"  Sum: {float(probs.sum())}")

    # ==============================
    # Total Cost v3 Test (probabilistic LaTeX model)
    # ==============================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move P_seq to device (segments/rep stay as python objects; stats tensors are created on device)
    P_seq = [p.to(device) for p in P_seq]

    # Instantiate v3 cost model
    total_cost_module = TotalCost(config).to(device)

    stats = total_cost_module.stats_extractor(segments, rep, N=P_seq[0].shape[0], device=device, dtype=P_seq[0].dtype)

    mode = (config.get("connectivity_proxy", {}).get("mode", "") or "").lower()
    expect_pair = mode.startswith("pair_")

    for s, e in enumerate(stats["edges"]):
        E = int(e["u"].numel())
        if E == 0:
            continue

    has_gamma_e = "gamma_e" in e
    print(f"[seg {s}] E={E} has_gamma_e={has_gamma_e}")

    if expect_pair:
        assert has_gamma_e, f"Expected gamma_e for mode={mode}, but missing in segment {s}"
        print("  gamma_e sample:", e["gamma_e"][: min(5, E)].detach().cpu().tolist())

    print("\n=== Total Cost v3 Test ===")
    cost_out = total_cost_module(P_seq, segments, rep, debug=True)
    # loss = cost_out["total_cost"]
    # loss.backward()


    # Core outputs
    print(f"Total cost: {cost_out['total_cost'].item():.6f}")

    # Tripartite breakdown (+ move)
    print(f"Exec cost sum: {cost_out['per_segment_exec'].sum().item():.6f}")
    print(f"Idle cost sum: {cost_out['per_segment_idle'].sum().item():.6f}")
    print(f"Comm cost sum: {cost_out['per_segment_comm'].sum().item():.6f}")
    print(f"Move cost sum: {cost_out['per_segment_move'].sum().item():.6f}")

    # Show first few segments
    S = len(cost_out["per_segment_total"])
    print("\nFirst 5 segments breakdown:")
    for s in range(min(5, S)):
        print(
            f"  Seg {s}: total={cost_out['per_segment_total'][s].item():.6f} | "
            f"exec={cost_out['per_segment_exec'][s].item():.6f}, "
            f"idle={cost_out['per_segment_idle'][s].item():.6f}, "
            f"comm={cost_out['per_segment_comm'][s].item():.6f}, "
            f"move={cost_out['per_segment_move'][s].item():.6f}"
        )

    # Exec sub-breakdown (always present)
    print("\nExec sub-breakdown (first 5 segments):")
    for s in range(min(5, S)):
        print(
            f"  Seg {s}: C1q={cost_out['per_segment_C1q'][s].item():.6f}, "
            f"C2q_local={cost_out['per_segment_C2q_local'][s].item():.6f}, "
            f"Cm={cost_out['per_segment_Cm'][s].item():.6f}"
        )

    drivers = compute_drivers(total_cost_module, P_seq, segments, rep, device=device)
    plot_cost_dashboard(
        cost_out,
        drivers,
        title_prefix="Cost breakdown (soft schedule)",
        show=True,
        save_path_prefix=None
    )

    # Optional debug-only fields (exist only when debug=True)
    if "comm_avg_cut_prob" in cost_out and cost_out["comm_avg_cut_prob"] is not None:
        print("\nComm debug (first 5 segments):")
        for s in range(min(5, S)):
            print(
                f"  Seg {s}: avg_cut={cost_out['comm_avg_cut_prob'][s].item():.4f}, "
                f"twoq_ops={cost_out['comm_twoq_ops'][s].item():.1f}"
            )

    if "move_avg_change" in cost_out and cost_out["move_avg_change"] is not None:
        print("\nMove debug (first 5 segments):")
        for s in range(min(5, S)):
            print(f"  Seg {s}: avg_change={cost_out['move_avg_change'][s].item():.4f}")
