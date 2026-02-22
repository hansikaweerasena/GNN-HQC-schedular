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


def run_segmentation(rep, threshold):
    segments, seg_ids = segment_circuit(rep.layers, threshold=threshold)
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


import numpy as np
import matplotlib.pyplot as plt


def compute_drivers(total_cost_module, P_seq, segments, rep, device):
    """
    Computes interpretable drivers:
      - L, dt
      - twoq_ops, avg_cut_prob
      - avg_move_change
    using the same stats_extractor used by the cost model.
    """
    dtype = P_seq[0].dtype
    N = P_seq[0].shape[0]
    S = len(P_seq)

    stats = total_cost_module.stats_extractor(segments, rep, N=N, device=device, dtype=dtype)

    L = stats["L"]                               # [S]
    dt = L * total_cost_module.delta.to(dtype)   # [S]

    W = torch.stack(P_seq, dim=0).to(dtype)      # [S,N,K]

    # --- comm drivers: twoq_ops + avg_cut_prob ---
    twoq_ops = torch.zeros((S,), device=device, dtype=dtype)
    avg_cut_prob = torch.zeros((S,), device=device, dtype=dtype)

    for s in range(S):
        e = stats["edges"][s]
        u_idx, v_idx = e["u"], e["v"]
        omega = e["w"].to(dtype)

        if u_idx.numel() == 0:
            continue

        Wu = W[s, u_idx, :]                 # [E,K]
        Wv = W[s, v_idx, :]                 # [E,K]
        local_prob = (Wu * Wv).sum(dim=1)   # [E]
        cut_prob = 1.0 - local_prob         # [E]

        twoq_ops[s] = omega.sum()
        denom = torch.clamp(omega.sum(), min=1e-12)
        avg_cut_prob[s] = (omega * cut_prob).sum() / denom

    # --- move driver: avg_change_prob ---
    avg_move_change = torch.zeros((S,), device=device, dtype=dtype)
    if S >= 2:
        stay_prob = (W[:-1] * W[1:]).sum(dim=2)     # [S-1,N]
        change_prob = 1.0 - stay_prob               # [S-1,N]
        avg_move_change[:-1] = change_prob.mean(dim=1)
        avg_move_change[-1] = 0.0

    # return cpu numpy for plotting
    def to_np(x): return x.detach().cpu().numpy()
    return {
        "L": to_np(L),
        "dt": to_np(dt),
        "twoq_ops": to_np(twoq_ops),
        "avg_cut_prob": to_np(avg_cut_prob),
        "avg_move_change": to_np(avg_move_change),
    }


def plot_costs(cost_out, drivers):
    """
    Draws:
      (1) stacked breakdown over segments
      (2) heatmap components x segments
      (3) overlays with drivers
    """
    # segment axis
    total = cost_out["per_segment_total"].detach().cpu().numpy()
    exec_c = cost_out["per_segment_exec"].detach().cpu().numpy()
    idle_c = cost_out["per_segment_idle"].detach().cpu().numpy()
    comm_c = cost_out["per_segment_comm"].detach().cpu().numpy()
    move_c = cost_out["per_segment_move"].detach().cpu().numpy()
    S = len(total)
    x = np.arange(S)

    # --------------------------
    # (1) Stacked area breakdown
    # --------------------------
    plt.figure(figsize=(12, 4))
    plt.stackplot(x, exec_c, idle_c, comm_c, move_c, labels=["exec", "idle", "comm", "move"])
    plt.plot(x, total, linewidth=2, label="total")
    plt.xlabel("Segment index")
    plt.ylabel("Cost")
    plt.title("Cost breakdown across segments (stacked)")
    plt.legend(loc="upper right")
    plt.tight_layout()

    # --------------------------
    # (2) Heatmap (components x segments)
    # --------------------------
    # include total + exec sub-breakdown if present
    rows = [
        ("exec", exec_c),
        ("idle", idle_c),
        ("comm", comm_c),
        ("move", move_c),
        ("total", total),
    ]

    # optional: exec subcomponents
    if "per_segment_C1q" in cost_out:
        rows.insert(1, ("C1q", cost_out["per_segment_C1q"].detach().cpu().numpy()))
    if "per_segment_C2q_local" in cost_out:
        rows.insert(2, ("C2q_local", cost_out["per_segment_C2q_local"].detach().cpu().numpy()))
    if "per_segment_Cm" in cost_out:
        rows.insert(3, ("Cm", cost_out["per_segment_Cm"].detach().cpu().numpy()))

    labels = [r[0] for r in rows]
    mat = np.vstack([r[1] for r in rows])  # [R,S]

    # log1p scaling helps visibility when one term dominates
    mat_show = np.log1p(mat)

    plt.figure(figsize=(12, 3.5))
    plt.imshow(mat_show, aspect="auto")
    plt.yticks(np.arange(len(labels)), labels)
    plt.xticks(np.arange(S))
    plt.xlabel("Segment index")
    plt.title("Heatmap of costs (log1p scaled)")
    plt.colorbar(label="log(1 + cost)")
    plt.tight_layout()

    # --------------------------
    # (3) Overlays with drivers
    # --------------------------

    # (3a) Comm cost vs avg cut prob (with twoq_ops as context)
    plt.figure(figsize=(12, 4))
    ax1 = plt.gca()
    ax1.plot(x, comm_c, linewidth=2)
    ax1.set_xlabel("Segment index")
    ax1.set_ylabel("Comm cost")
    ax1.set_title("Comm cost vs avg cut probability")

    ax2 = ax1.twinx()
    ax2.plot(x, drivers["avg_cut_prob"], linestyle="--", linewidth=2)
    ax2.set_ylabel("Avg cut probability")

    plt.tight_layout()

    # (3b) Comm cost vs 2Q ops (helps separate “many edges” vs “high cut”)
    plt.figure(figsize=(12, 4))
    ax1 = plt.gca()
    ax1.plot(x, comm_c, linewidth=2)
    ax1.set_xlabel("Segment index")
    ax1.set_ylabel("Comm cost")
    ax1.set_title("Comm cost vs total 2Q ops")

    ax2 = ax1.twinx()
    ax2.plot(x, drivers["twoq_ops"], linestyle="--", linewidth=2)
    ax2.set_ylabel("Total 2Q ops (sum ω)")
    plt.tight_layout()

    # (3c) Idle cost vs dt = L * delta
    plt.figure(figsize=(12, 4))
    ax1 = plt.gca()
    ax1.plot(x, idle_c, linewidth=2)
    ax1.set_xlabel("Segment index")
    ax1.set_ylabel("Idle cost")
    ax1.set_title("Idle cost vs segment duration proxy (L * delta)")

    ax2 = ax1.twinx()
    ax2.plot(x, drivers["dt"], linestyle="--", linewidth=2)
    ax2.set_ylabel("dt = L * delta")
    plt.tight_layout()

    # (3d) Move cost vs avg move change
    plt.figure(figsize=(12, 4))
    ax1 = plt.gca()
    ax1.plot(x, move_c, linewidth=2)
    ax1.set_xlabel("Segment index")
    ax1.set_ylabel("Move cost")
    ax1.set_title("Move cost vs avg assignment change probability")

    ax2 = ax1.twinx()
    ax2.plot(x, drivers["avg_move_change"], linestyle="--", linewidth=2)
    ax2.set_ylabel("Avg change probability")
    plt.tight_layout()

    plt.show()

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

    print("\n=== Total Cost v3 Test ===")
    cost_out = total_cost_module(P_seq, segments, rep, debug=True)

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

    dump_cost_out_debug(cost_out, max_items=5)
    drivers = compute_drivers(total_cost_module, P_seq, segments, rep, device=device)
    plot_costs(cost_out, drivers)

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
