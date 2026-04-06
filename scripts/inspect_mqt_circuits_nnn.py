#!/usr/bin/env python3
"""
inspect_mqt_circuits.py — Browse MQT Bench circuits + MOSAIC schedule visualization.

Loads MQT Bench circuits, preprocesses them (same pipeline as eval_scheduler_v2.py),
optionally loads a trained MOSAIC model and runs MOSAIC + B3/B4/B5 baselines.

For each circuit, displays:
  1) Raw Qiskit circuit (pre-preprocessing, with measurements, layer barriers)
  2) Two-panel: custom circuit diagram (top) + 2Q activity heatmap (bottom)
  3) If --run_dir given: MOSAIC vs best-baseline assignment heatmaps
  4) If --run_dir given: remaining two baseline assignment heatmaps

Usage (browse only):
    python inspect_mqt_circuits.py --qubit_min 8 --qubit_max 12 --algorithms bv,wstate

Usage (with model evaluation):
    python inspect_mqt_circuits.py \
        --qubit_min 8 --qubit_max 12 --algorithms bv,wstate \
        --run_dir results/20250601_run_v1 --checkpoint best
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit


# ── MQT algorithm registry (same as eval_scheduler_v2.py) ──────────────────

MQT_ALGORITHMS = [
    "qaoa", "qft", "qftentangled", "vqe_real_amp", "vqe_su2",
    "vqe_two_local", "wstate", "qgan", "qnn",
    "portfolioqaoa", "portfoliovqe", "bv", "randomcircuit",
]


# ── Logging ─────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def log_section(title: str):
    w = 72
    print(flush=True)
    print("=" * w, flush=True)
    print(f"  {title}", flush=True)
    print("=" * w, flush=True)


# ── Circuit loading (identical pipeline to eval_scheduler_v2.py) ────────────

def _strip_classical_registers(qc: QuantumCircuit) -> QuantumCircuit:
    new_qc = QuantumCircuit(*qc.qregs)
    for instruction in qc.data:
        if len(instruction.clbits) == 0:
            new_qc.append(instruction)
    return new_qc


def _has_multiqubit_gates(qc: QuantumCircuit) -> bool:
    for instruction in qc.data:
        if len(instruction.qubits) >= 3:
            return True
    return False


def _decompose_to_1q2q(qc: QuantumCircuit, max_rounds: int = 15) -> QuantumCircuit:
    """
    Selectively decompose ONLY gates acting on 3+ qubits.

    Unlike qc.decompose() which decomposes ALL gates with definitions,
    this uses the DAG to substitute only 3Q+ gate nodes, leaving 1Q/2Q
    gates completely untouched:
      - CZ stays CZ   (not exploded into H+CX+H)
      - CP stays CP    (not exploded into CX+Rz chains)
      - Ry stays Ry    (not rewritten as Rz+SX)
      - QFTGate(N) → individual H, CP, SWAP gates
      - Toffoli     → CX + 1Q gates
    """
    qc_out = qc
    for _ in range(max_rounds):
        if not _has_multiqubit_gates(qc_out):
            break

        dag = circuit_to_dag(qc_out)
        did_something = False

        for node in dag.op_nodes():
            if len(node.qargs) < 3:
                continue  # skip 1Q/2Q gates entirely

            defn = node.op.definition
            if defn is None:
                # Primitive 3Q+ gate with no definition — can't decompose further
                continue

            # Replace this 3Q+ node with its definition sub-DAG
            sub_dag = circuit_to_dag(defn)
            dag.substitute_node_with_dag(node, sub_dag)
            did_something = True

        qc_out = dag_to_circuit(dag)

        if not did_something:
            break  # nothing left to decompose

    return qc_out


def load_mqt_circuits(
    algorithms: List[str],
    qubit_min: int,
    qubit_max: int,
) -> List[Tuple[str, int, QuantumCircuit, QuantumCircuit]]:
    """Returns list of (algo, num_qubits, qc_clean, qc_raw) tuples."""
    try:
        from mqt.bench import BenchmarkLevel, get_benchmark
    except ImportError:
        raise ImportError("mqt.bench not installed. Install: pip install mqt.bench")

    circuits, skipped = [], []

    for algo in algorithms:
        for nq in range(qubit_min, qubit_max + 1):
            try:
                # Step 1: generate at algorithmic level
                qc_raw = get_benchmark(
                    benchmark=algo, level=BenchmarkLevel.ALG, circuit_size=nq)

                # Step 2: decompose 3Q+ gates only (preserve native 1Q/2Q gates)
                qc_decomposed = _decompose_to_1q2q(qc_raw)

                # Step 3: remove final measurements
                qc_decomposed.remove_final_measurements(inplace=True)

                # Step 4: strip classical registers
                qc_clean = _strip_classical_registers(qc_decomposed)

                # Step 5: validate
                if qc_clean.num_clbits != 0:
                    skipped.append((algo, nq, "classical bits remain")); continue
                if qc_clean.num_qubits < qubit_min or qc_clean.num_qubits > qubit_max:
                    skipped.append((algo, nq, f"qubit count -> {qc_clean.num_qubits}")); continue
                if _has_multiqubit_gates(qc_clean):
                    skipped.append((algo, nq, "3Q+ gates remain after decompose")); continue
                if qc_clean.size() == 0:
                    skipped.append((algo, nq, "empty circuit")); continue

                circuits.append((algo, qc_clean.num_qubits, qc_clean, qc_raw))
                log(f"  [OK] {algo:20s} N={qc_clean.num_qubits:2d}  "
                    f"gates(raw)={qc_raw.size():5d}  gates(clean)={qc_clean.size():5d}")

            except Exception as e:
                skipped.append((algo, nq, str(e)[:80]))

    if skipped:
        log(f"  Skipped {len(skipped)} (algo, nq) combinations:")
        for a, n, r in skipped[:15]:
            log(f"    {a:20s} N={n:2d}  reason: {r}")
        if len(skipped) > 15:
            log(f"    ... and {len(skipped) - 15} more")

    return circuits


# ── Layer extraction (mirrors CircuitRepresentation._extract_layers) ────────

def extract_layers(qc: QuantumCircuit):
    dag = circuit_to_dag(qc)
    qubit_index = {qb: i for i, qb in enumerate(qc.qubits)}
    layers = []

    for dag_layer in dag.layers():
        subdag = dag_layer["graph"]
        layer = {"gates_1q": [], "gates_2q": [], "active_qubits": set(), "twoq_qubits": set()}

        for node in subdag.op_nodes():
            name = node.op.name
            if name in ("barrier", "measure", "reset"):
                continue
            qargs = tuple(qubit_index[qb] for qb in node.qargs)
            layer["active_qubits"].update(qargs)

            if len(qargs) == 1:
                layer["gates_1q"].append((name, qargs))
            elif len(qargs) == 2:
                layer["gates_2q"].append((name, qargs))
                layer["twoq_qubits"].update(qargs)

        if layer["gates_1q"] or layer["gates_2q"]:
            layers.append(layer)

    return layers


# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING + EVALUATION  (copied from eval_scheduler_v2.py)
# ═══════════════════════════════════════════════════════════════════════════

def _setup_project_imports(run_dir: str):
    """Add project root to sys.path so src/ and utils/ imports work."""
    for candidate in [
        os.path.abspath(os.path.join(run_dir, "..")),
        os.path.abspath(os.path.join(run_dir, "../..")),
        os.path.abspath("."),
        os.path.abspath(".."),
    ]:
        if os.path.isdir(os.path.join(candidate, "src")):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return candidate
    raise RuntimeError(
        "Cannot find project root (directory containing src/). "
        "Run this script from the project root or pass --project_root.")


def _load_snapshot_cfg(snapshot_path: str) -> dict:
    ns: dict = {}
    with open(snapshot_path, "r") as f:
        exec(f.read(), ns)  # noqa: S102
    return ns


def load_run_artifacts(run_dir: str, checkpoint: str, device: str = "cpu") -> dict:
    """Load trained model, cost config, and dataset config from a run directory."""
    import torch
    from src.evolving_gnn import EvolvingGNN
    from src.clustering_head import SegmentClustering
    from src.cost_function import TotalCost
    from utils.cost_config_reader import load_cost_config
    from src.qubit_interaction_graph import compute_window_sizes_from_config

    log(f"Loading run artifacts from: {run_dir}")

    arch_path = os.path.join(run_dir, "model_arch_params.json")
    with open(arch_path) as f:
        arch = json.load(f)
    gnn_arch = arch["EvolvingGNN"]
    cls_arch = arch["SegmentClustering"]
    log(f"  arch: gru_hidden={gnn_arch['gru_hidden_dim']}, K={cls_arch['num_clusters']}")

    cost_cfg_path = os.path.join(run_dir, "cost_config_snapshot.json")
    config = load_cost_config(cost_cfg_path)
    K = len(config["techs"])
    tech_names = [t.get("name", f"tech{k}") for k, t in enumerate(config["techs"])]
    caps = torch.tensor(
        [float(t["capacity"]["max_qubits"]) for t in config["techs"]],
        dtype=torch.float32)
    w_short, w_long = compute_window_sizes_from_config(config)
    log(f"  cost config: K={K}, techs={tech_names}, caps={caps.tolist()}, "
        f"w_short={w_short}, w_long={w_long}")

    snap_path = os.path.join(run_dir, "scheduler_config_snapshot.py")
    snap = _load_snapshot_cfg(snap_path)
    dataset_cfg = snap["DATASET_CFG"]
    log(f"  dataset cfg: seg_mode={dataset_cfg['segmentation_mode']}")

    evol_model = EvolvingGNN(
        node_feat_dim  = gnn_arch["node_feat_dim"],
        edge_feat_dim  = gnn_arch["edge_feat_dim"],
        mlp_hidden_dim = gnn_arch["mlp_hidden_dim"],
        mlp_out_dim    = gnn_arch["mlp_out_dim"],
        gnn_out_dim    = gnn_arch["gnn_out_dim"],
        gru_hidden_dim = gnn_arch["gru_hidden_dim"],
        heads          = gnn_arch["heads"],
        dropout        = gnn_arch["dropout"],
        bptt_steps     = gnn_arch["bptt_steps"],
        activation     = gnn_arch.get("activation", "relu"),
    ).to(device)

    cluster_module = SegmentClustering(
        hidden_dim          = cls_arch["hidden_dim"],
        num_clusters        = K,
        proj_hidden_dim     = cls_arch.get("proj_hidden_dim"),
        temperature_init    = cls_arch["temperature_init"],
        temperature_min     = cls_arch["temperature_min"],
        temperature_gamma   = cls_arch["temperature_gamma"],
        neighbor_alpha_init = cls_arch.get("neighbor_alpha_learned", 0.1),
    ).to(device)

    ckpt_lower = checkpoint.lower()
    if ckpt_lower == "final":
        evol_model.load_state_dict(
            torch.load(os.path.join(run_dir, "evol_model.pt"), map_location=device))
        cluster_module.load_state_dict(
            torch.load(os.path.join(run_dir, "cluster_head.pt"), map_location=device))
        log(f"  weights: final")
    else:
        if ckpt_lower == "best":
            ckpt_file = os.path.join(run_dir, "checkpoint_best.pt")
        elif ckpt_lower == "last":
            ckpt_file = os.path.join(run_dir, "checkpoint_last.pt")
        elif ckpt_lower.startswith("epoch_"):
            n = ckpt_lower.split("_")[1]
            ckpt_file = os.path.join(run_dir, f"checkpoint_epoch_{n.zfill(3)}.pt")
        else:
            raise ValueError(f"Unknown checkpoint: '{checkpoint}'")
        ckpt_dict = torch.load(ckpt_file, map_location=device)
        evol_model.load_state_dict(ckpt_dict["evol_model"])
        cluster_module.load_state_dict(ckpt_dict["cluster_head"])
        epoch     = ckpt_dict.get("epoch", "?")
        test_loss = ckpt_dict.get("test_loss", float("nan"))
        log(f"  weights: {os.path.basename(ckpt_file)} "
            f"(epoch={epoch}, test_loss={test_loss:.4f})")

    evol_model.eval()
    cluster_module.eval()

    total_cost_module = TotalCost(config).to(device)
    tau_min = total_cost_module._tau_min
    total_cost_module.tau.fill_(tau_min)
    log(f"  TotalCost built (tau=tau_min={tau_min:.4f})")

    return {
        "evol_model":     evol_model,
        "cluster_module": cluster_module,
        "cost_module":    total_cost_module,
        "config":         config,
        "dataset_cfg":    dataset_cfg,
        "K":              K,
        "tech_names":     tech_names,
        "caps":           caps,
        "w_short":        w_short,
        "w_long":         w_long,
        "device":         device,
    }


def preprocess_circuit(qc, dataset_cfg, w_short, w_long):
    """Convert a clean QuantumCircuit to internal representation."""
    import torch
    from torch_geometric.data import Data
    from src.circuit_representation import CircuitRepresentation
    from src.circuit_segmentation import segment_circuit
    from src.qubit_interaction_graph import build_layer_graph_arrays

    rep = CircuitRepresentation(qc)
    seg_mode = dataset_cfg["segmentation_mode"]
    seg_thr  = float(dataset_cfg["segment_threshold"])
    segments, _ = segment_circuit(rep.layers, mode=seg_mode, threshold=seg_thr)

    arrays = build_layer_graph_arrays(rep, w_short, w_long)
    layer_data_list = [
        Data(
            x          = torch.tensor(x_np,  dtype=torch.float32),
            edge_index = torch.tensor(ei_np, dtype=torch.long),
            edge_attr  = torch.tensor(ea_np, dtype=torch.float32),
        )
        for x_np, ei_np, ea_np in arrays
    ]
    return rep, segments, layer_data_list


def run_inference(evol_model, cluster_module, layer_data_list):
    import torch
    with torch.no_grad():
        h_seq, _ = evol_model(layer_data_list)
        P_seq    = cluster_module(h_seq, graphs=layer_data_list)
    return P_seq


def compute_metrics_v1(hard_assignments, rep, segments, cost_module, caps, K, config, device):
    """Compute hard cost + secondary metrics (identical to eval_scheduler_v2.py)."""
    import torch
    from baselines_tier1 import rank_techs_by

    T = len(hard_assignments)
    N = rep.num_qubits

    P_seq_hard = []
    for t in range(T):
        P_t = torch.zeros(N, K, dtype=torch.float32, device=device)
        P_t[torch.arange(N, device=device), hard_assignments[t].to(device)] = 1.0
        P_seq_hard.append(P_t)

    with torch.no_grad():
        hard_out = cost_module(P_seq_hard, segments, rep)
    hard_cost = hard_out["total_cost"].item()

    total_2q = cut_2q = 0
    for t in range(min(T, len(rep.layers))):
        ha_t = hard_assignments[t].cpu()
        for gate_name, qargs in rep.layers[t].gates:
            if len(qargs) == 2:
                u, v = qargs
                total_2q += 1
                if ha_t[u].item() != ha_t[v].item():
                    cut_2q += 1
    remote_2q_cut_rate = cut_2q / max(total_2q, 1)

    movements = []
    for t in range(T - 1):
        moved = (hard_assignments[t].cpu() != hard_assignments[t + 1].cpu()
                 ).float().sum().item()
        movements.append(moved)
    mean_movement = float(np.mean(movements)) if movements else 0.0

    # Idle decoherence placement
    best_T2 = rank_techs_by(config, "T2")[0]
    total_idle = correct_idle = 0
    for t in range(min(T, len(rep.layers))):
        ha_t = hard_assignments[t].cpu()
        active = set()
        for _, qargs in rep.layers[t].gates:
            for q in qargs:
                active.add(q)
        for q in range(N):
            if q not in active:
                total_idle += 1
                if ha_t[q].item() == best_T2:
                    correct_idle += 1
    idle_decoherence_rate = correct_idle / max(total_idle, 1)

    return {
        "hard_cost":             hard_cost,
        "remote_2q_cut_rate":    remote_2q_cut_rate,
        "mean_movement":         mean_movement,
        "idle_decoherence_rate": idle_decoherence_rate,
        "T": T, "N": N,
    }


def run_all_methods(qc, art):
    """Run MOSAIC + B3/B4/B5 on a single circuit. Returns dict of method->(hard, metrics)."""
    import torch
    from utils.inference_utils import enforce_capacity_sequence
    from baselines_tier1 import baseline_b3
    from baselines_tier2 import baseline_b4, baseline_b5

    device      = art["device"]
    config      = art["config"]
    dataset_cfg = art["dataset_cfg"]
    K           = art["K"]
    caps        = art["caps"]
    w_short     = art["w_short"]
    w_long      = art["w_long"]
    cost_module = art["cost_module"]

    rep, segments, layer_data_list = preprocess_circuit(qc, dataset_cfg, w_short, w_long)

    results = {}

    # MOSAIC
    P_seq       = run_inference(art["evol_model"], art["cluster_module"], layer_data_list)
    mosaic_hard = enforce_capacity_sequence(P_seq, caps)
    mosaic_m    = compute_metrics_v1(mosaic_hard, rep, segments, cost_module, caps, K, config, device)
    results["MOSAIC"] = {"hard": mosaic_hard, "metrics": mosaic_m}

    # B3
    b3_hard = baseline_b3(rep, caps, config, K)
    b3_m    = compute_metrics_v1(b3_hard, rep, segments, cost_module, caps, K, config, device)
    results["B3"] = {"hard": b3_hard, "metrics": b3_m}

    # B4
    b4_hard = baseline_b4(rep, caps, config, K, seed=42)
    b4_m    = compute_metrics_v1(b4_hard, rep, segments, cost_module, caps, K, config, device)
    results["B4"] = {"hard": b4_hard, "metrics": b4_m}

    # B5
    b5_hard = baseline_b5(rep, caps, config, K)
    b5_m    = compute_metrics_v1(b5_hard, rep, segments, cost_module, caps, K, config, device)
    results["B5"] = {"hard": b5_hard, "metrics": b5_m}

    return results, rep


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

# ── Raw circuit helpers ─────────────────────────────────────────────────────

NEEDS_DECOMPOSE = {"qft", "qftentangled"}

PRIMITIVE_GATES = {
    "cx", "cz", "cy", "swap", "ccx", "cswap",
    "h", "x", "y", "z", "s", "t", "sdg", "tdg",
    "sx", "sxdg", "rx", "ry", "rz", "u", "u1", "u2", "u3",
    "p", "cp", "crx", "cry", "crz",
    "measure", "barrier", "reset", "id",
}


def _needs_further_decompose(qc):
    for instruction in qc.data:
        if instruction.operation.name not in PRIMITIVE_GATES:
            return True
    return False


def _decompose_fully(qc, max_rounds=10):
    qc_out = qc
    for _ in range(max_rounds):
        if not _needs_further_decompose(qc_out):
            break
        qc_out = qc_out.decompose()
    return qc_out


def _rebuild_with_barriers(qc, max_layers=None):
    dag = circuit_to_dag(qc)
    all_layers = list(dag.layers())
    total = len(all_layers)
    cap = max_layers if max_layers is not None else total
    T_show = min(total, cap)
    truncated = total > cap

    new_qc = QuantumCircuit(*qc.qregs, *qc.cregs)
    for i, layer in enumerate(all_layers[:T_show]):
        subdag = layer["graph"]
        for node in subdag.op_nodes():
            new_qc.append(node.op, node.qargs, node.cargs)
        if i < T_show - 1:
            new_qc.barrier()

    return new_qc, total, truncated


def plot_raw_circuit(algo, nq, qc_raw, max_layers=None, save_dir=None):
    total_depth = qc_raw.depth()
    total_gates = qc_raw.size()

    if algo.lower() in NEEDS_DECOMPOSE:
        log(f"    decomposing composite gates for {algo}...")
        qc_expanded = _decompose_fully(qc_raw)
        decompose_note = f"  [decomposed: {qc_raw.size()}->{qc_expanded.size()} gates]"
    else:
        qc_expanded = qc_raw
        decompose_note = ""

    qc_draw, total_layers, truncated = _rebuild_with_barriers(qc_expanded, max_layers)
    trunc_note = f"  (first {max_layers} of {total_layers} layers)" if truncated else ""

    try:
        fig = qc_draw.draw(
            output="mpl", fold=-1, idle_wires=True,
            style={"backgroundcolor": "#FFFFFF"})
        fig.suptitle(
            f"RAW circuit:  {algo}  |  N={qc_raw.num_qubits}  depth={total_depth}  "
            f"gates={total_gates}{decompose_note}{trunc_note}",
            fontsize=10, fontweight="bold", y=1.02)
        fig.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"{algo}_N{nq}_raw.png"),
                        dpi=150, bbox_inches="tight")
        plt.show(); plt.close(fig)
    except Exception as e:
        log(f"    mpl draw failed ({e}), falling back to text:")
        print(qc_draw.draw(output="text", fold=80))


# ── Gate abbreviations for circuit diagram ──────────────────────────────────

GATE_ABBREV = {
    "cx": "CX", "cz": "CZ", "swap": "SW",
    "h": "H", "x": "X", "y": "Y", "z": "Z",
    "s": "S", "t": "T", "sx": "√X",
    "rx": "Rx", "ry": "Ry", "rz": "Rz",
    "sdg": "S†", "tdg": "T†",
}

def _abbrev(gate_name):
    return GATE_ABBREV.get(gate_name, gate_name[:3])


# ── Two-panel: circuit diagram + activity heatmap ───────────────────────────

def plot_circuit_and_heatmap(algo, nq, layers, max_layers, save_dir=None):
    T_full = len(layers)
    T_show = min(T_full, max_layers)
    truncated = T_full > max_layers

    twoq_activity = np.zeros((T_show, nq), dtype=float)
    oneq_activity = np.zeros((T_show, nq), dtype=float)
    twoq_edges_per_layer = []

    for t in range(T_show):
        layer = layers[t]
        edges = []
        for _, (q0, q1) in layer["gates_2q"]:
            twoq_activity[t, q0] = 1.0
            twoq_activity[t, q1] = 1.0
            edges.append((q0, q1))
        for _, (q,) in layer["gates_1q"]:
            if twoq_activity[t, q] == 0:
                oneq_activity[t, q] = 1.0
        twoq_edges_per_layer.append(edges)

    combined = twoq_activity + 0.4 * oneq_activity

    total_2q = sum(len(l["gates_2q"]) for l in layers)
    total_1q = sum(len(l["gates_1q"]) for l in layers)
    density_2q = twoq_activity.sum() / max(T_show * nq, 1) * 100

    fig_w = max(7.0, T_show * 0.45 + 2.0)
    fig_h = max(5.0, nq * 0.55 + 3.0)
    fig, (ax_circ, ax_heat) = plt.subplots(
        2, 1, figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": [1.2, 1]}, sharex=True)

    trunc_note = f"  (first {T_show}/{T_full})" if truncated else ""
    fig.suptitle(
        f"{algo}  |  N={nq}  T={T_full}{trunc_note}  |  "
        f"2Q={total_2q}  1Q={total_1q}  dens={density_2q:.1f}%",
        fontsize=10, fontweight="bold")

    # ── Top: circuit diagram ──
    ax = ax_circ
    for q in range(nq):
        ax.hlines(q, -0.5, T_show - 0.5, color="lightgray", linewidth=0.6, zorder=0)

    for t in range(T_show):
        layer = layers[t]
        for gate_name, (q,) in layer["gates_1q"]:
            ax.add_patch(plt.Rectangle(
                (t - 0.25, q - 0.25), 0.5, 0.5,
                facecolor="#a8d8ea", edgecolor="#3a7ca5", linewidth=0.6, zorder=2))
            ax.text(t, q, _abbrev(gate_name), ha="center", va="center",
                    fontsize=4.5, color="#1a4a6e", zorder=3, fontweight="bold")
        for gate_name, (q0, q1) in layer["gates_2q"]:
            qlo, qhi = min(q0, q1), max(q0, q1)
            ax.vlines(t, qlo, qhi, color="#c0392b", linewidth=1.5, zorder=2)
            if gate_name == "cx":
                ax.plot(t, q0, "o", color="#c0392b", markersize=4, zorder=3)
                ax.plot(t, q1, "o", color="#c0392b", markersize=6, zorder=3,
                        markerfacecolor="white", markeredgewidth=1.2)
                ax.plot([t - 0.08, t + 0.08], [q1, q1], color="#c0392b", linewidth=0.8, zorder=4)
                ax.plot([t, t], [q1 - 0.12, q1 + 0.12], color="#c0392b", linewidth=0.8, zorder=4)
            else:
                ax.plot(t, q0, "o", color="#c0392b", markersize=5, zorder=3)
                ax.plot(t, q1, "o", color="#c0392b", markersize=5, zorder=3)

    ax.set_ylabel("Qubit", fontsize=9)
    ax.set_yticks(range(nq))
    ax.set_yticklabels([f"q{q}" for q in range(nq)], fontsize=6)
    ax.set_ylim(nq - 0.5, -0.5)
    ax.set_xlim(-0.5, T_show - 0.5)
    ax.set_title("Circuit Diagram", fontsize=9, pad=4)
    ax.tick_params(axis="x", labelbottom=False)

    # ── Bottom: activity heatmap ──
    ax = ax_heat
    heatmap_data = combined.T
    ax.imshow(heatmap_data, aspect="auto", origin="upper", cmap="Blues",
              interpolation="nearest", vmin=0, vmax=1.0)
    for t, edges in enumerate(twoq_edges_per_layer):
        for q0, q1 in edges:
            ax.plot([t, t], [q0, q1], color="darkred", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Layer", fontsize=9)
    ax.set_ylabel("Qubit", fontsize=9)
    ax.set_yticks(range(nq))
    ax.set_yticklabels([f"q{q}" for q in range(nq)], fontsize=6)
    ax.set_title("2Q Activity Heatmap", fontsize=9, pad=4)
    if T_show <= 40:
        step = 1 if T_show <= 20 else 2 if T_show <= 30 else 5
        ax.set_xticks(range(0, T_show, step))
        ax.set_xticklabels([str(t) for t in range(0, T_show, step)], fontsize=6)

    legend_patches = [
        mpatches.Patch(color=plt.cm.Blues(1.0), label="2Q gate"),
        mpatches.Patch(color=plt.cm.Blues(0.4), label="1Q only"),
        mpatches.Patch(facecolor="white", edgecolor="gray", label="Idle"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=7, framealpha=0.8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{algo}_N{nq}_circuit_activity.png"),
                    dpi=180, bbox_inches="tight")
    plt.show(); plt.close(fig)


# ── Schedule assignment heatmaps ────────────────────────────────────────────

# Technology colors — consistent with eval_scheduler.py
TECH_COLORS = [
    "#2196F3",  # SC  — blue
    "#FF9800",  # TI  — orange
    "#4CAF50",  # NA  — green
    "#9C27B0",  # ES  — purple
]


def _build_assignment_matrix(hard_assignments, max_layers=None):
    """Convert list of assignment tensors to numpy [T_show, N]."""
    T = len(hard_assignments)
    T_show = min(T, max_layers) if max_layers else T
    N = hard_assignments[0].shape[0]
    mat = np.zeros((T_show, N), dtype=int)
    for t in range(T_show):
        mat[t] = hard_assignments[t].cpu().numpy()
    return mat


def plot_schedule_comparison(
    algo: str,
    nq: int,
    method_pairs: List[Tuple[str, np.ndarray, float]],
    tech_names: List[str],
    K: int,
    max_layers: int,
    title_prefix: str = "",
    save_dir: str = None,
    save_suffix: str = "",
):
    """
    Plot 1 x len(method_pairs) heatmaps comparing technology assignments.
    Each sub-panel: x=layer, y=qubit, color=technology.
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm

    n_methods = len(method_pairs)
    T_show = method_pairs[0][1].shape[0]

    fig_w = max(7.0, T_show * 0.35 + 2.0) * (n_methods / 2.0)
    fig_h = max(3.5, nq * 0.35 + 2.0)
    fig, axes = plt.subplots(1, n_methods, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes[0]

    fig.suptitle(
        f"{title_prefix}{algo}  |  N={nq}  T(shown)={T_show}  |  "
        f"K={K} ({', '.join(tech_names)})",
        fontsize=10, fontweight="bold")

    cmap = ListedColormap(TECH_COLORS[:K])
    bounds = np.arange(-0.5, K + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    legend_patches = [
        mpatches.Patch(color=TECH_COLORS[k], label=tech_names[k])
        for k in range(K)
    ]

    for idx, (method_name, assign_mat, cost) in enumerate(method_pairs):
        ax = axes[idx]
        # assign_mat [T_show, N] -> transpose to [N, T_show] for x=layer, y=qubit
        ax.imshow(assign_mat.T, aspect="auto", origin="upper",
                  cmap=cmap, norm=norm, interpolation="nearest")

        ax.set_title(f"{method_name}\ncost={cost:.4f}", fontsize=9, pad=4)
        ax.set_xlabel("Layer", fontsize=8)
        if idx == 0:
            ax.set_ylabel("Qubit", fontsize=8)

        ax.set_yticks(range(nq))
        ax.set_yticklabels([f"q{q}" for q in range(nq)], fontsize=5)
        if T_show <= 40:
            step = 1 if T_show <= 20 else 2 if T_show <= 30 else 5
            ax.set_xticks(range(0, T_show, step))
            ax.set_xticklabels([str(t) for t in range(0, T_show, step)], fontsize=5)

    axes[-1].legend(handles=legend_patches, loc="lower right", fontsize=7, framealpha=0.8)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{algo}_N{nq}_schedule{save_suffix}.png"),
                    dpi=180, bbox_inches="tight")
    plt.show(); plt.close(fig)


# ── Summary / score tables ──────────────────────────────────────────────────

def print_summary_table(circuits_info):
    print("\n" + "=" * 72)
    print(f"  {'Algorithm':<20s} {'N':>4s} {'T(layers)':>10s} {'2Q gates':>9s} "
          f"{'1Q gates':>9s} {'2Q dens%':>9s}")
    print("-" * 72)
    for info in circuits_info:
        print(f"  {info['algo']:<20s} {info['N']:>4d} {info['T']:>10d} "
              f"{info['total_2q']:>9d} {info['total_1q']:>9d} "
              f"{info['density_2q']:>8.1f}%")
    print("=" * 72)


def print_scores_table(algo, nq, results):
    """Print compact score comparison for one circuit."""
    methods_sorted = sorted(results.keys(),
                            key=lambda m: results[m]["metrics"]["hard_cost"])
    print(f"\n  Scores for {algo} N={nq}:")
    print(f"  {'Method':<10s} {'HardCost':>10s} {'CutRate%':>9s} "
          f"{'Movement':>9s} {'IdlePlac':>9s}")
    print(f"  {'-'*48}")
    for m in methods_sorted:
        met = results[m]["metrics"]
        print(f"  {m:<10s} {met['hard_cost']:>10.4f} "
              f"{met['remote_2q_cut_rate']*100:>8.2f}% "
              f"{met['mean_movement']:>9.3f} "
              f"{met['idle_decoherence_rate']:>9.4f}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Inspect MQT circuits + MOSAIC schedule visualization")
    p.add_argument("--qubit_min", type=int, default=8)
    p.add_argument("--qubit_max", type=int, default=12)
    p.add_argument("--max_layers", type=int, default=30,
                   help="Cap displayed layers (default 30)")
    p.add_argument("--algorithms", type=str, default=None,
                   help="Comma-separated algorithm filter (default: all)")
    p.add_argument("--save_dir", type=str, default=None,
                   help="Save figures to this directory")
    p.add_argument("--no_plot", action="store_true",
                   help="Print summary table only, skip plots")
    # Model evaluation args
    p.add_argument("--run_dir", type=str, default=None,
                   help="Path to trained MOSAIC run directory (enables evaluation)")
    p.add_argument("--checkpoint", type=str, default="best",
                   help="Which weights: 'final' | 'best' | 'last' | 'epoch_NNN'")
    p.add_argument("--project_root", type=str, default=None,
                   help="Project root containing src/ (auto-detected if omitted)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.algorithms:
        algorithms = [a.strip() for a in args.algorithms.split(",")]
    else:
        algorithms = MQT_ALGORITHMS

    # ── Optional: set up project imports and load model ──
    art = None
    if args.run_dir:
        if args.project_root:
            if args.project_root not in sys.path:
                sys.path.insert(0, os.path.abspath(args.project_root))
        else:
            _setup_project_imports(args.run_dir)

        log_section("LOADING MOSAIC MODEL")
        art = load_run_artifacts(args.run_dir, args.checkpoint)

    # ── Load circuits ──
    log_section("LOADING MQT BENCH CIRCUITS")
    log(f"N=[{args.qubit_min}, {args.qubit_max}], algorithms={algorithms}")
    mqt_circuits = load_mqt_circuits(algorithms, args.qubit_min, args.qubit_max)

    if not mqt_circuits:
        log("No circuits loaded. Check mqt.bench installation.")
        sys.exit(1)
    log(f"Loaded {len(mqt_circuits)} circuits total.")

    # ── Extract layers and stats ──
    circuits_info = []
    for algo, nq, qc, qc_raw in mqt_circuits:
        layers = extract_layers(qc)
        T = len(layers)
        total_2q = sum(len(l["gates_2q"]) for l in layers)
        total_1q = sum(len(l["gates_1q"]) for l in layers)
        T_show = min(T, args.max_layers)
        twoq_cells = sum(len(layers[t]["twoq_qubits"]) for t in range(T_show))
        density = twoq_cells / max(T_show * nq, 1) * 100

        circuits_info.append({
            "algo": algo, "N": nq, "T": T,
            "total_2q": total_2q, "total_1q": total_1q,
            "density_2q": density,
            "layers": layers, "qc": qc, "qc_raw": qc_raw,
        })

    print_summary_table(circuits_info)

    if args.no_plot:
        return

    # ── Per-circuit: visualize + optionally evaluate ──
    for info in circuits_info:
        algo, nq = info["algo"], info["N"]
        log(f"\n{'─'*60}")
        log(f"  Circuit: {algo}  N={nq}  T={info['T']}")
        log(f"{'─'*60}")

        # 1) Raw circuit
        plot_raw_circuit(
            algo=algo, nq=nq, qc_raw=info["qc_raw"],
            max_layers=args.max_layers, save_dir=args.save_dir)

        # 2) Circuit diagram + activity heatmap
        plot_circuit_and_heatmap(
            algo=algo, nq=nq, layers=info["layers"],
            max_layers=args.max_layers, save_dir=args.save_dir)

        # 3) If model loaded: run MOSAIC + baselines and show schedules
        if art is not None:
            try:
                log(f"  Running MOSAIC + B3/B4/B5...")
                t0 = time.time()
                results, rep = run_all_methods(info["qc"], art)
                elapsed = time.time() - t0
                log(f"  Done in {elapsed:.1f}s")

                # Print scores
                print_scores_table(algo, nq, results)

                # Rank baselines by cost (lower is better)
                baseline_names = ["B3", "B4", "B5"]
                baseline_ranked = sorted(
                    baseline_names,
                    key=lambda m: results[m]["metrics"]["hard_cost"])
                best_bl   = baseline_ranked[0]
                other_bls = baseline_ranked[1:]

                max_T = args.max_layers

                # Build assignment matrices
                mosaic_mat = _build_assignment_matrix(results["MOSAIC"]["hard"], max_T)
                best_mat   = _build_assignment_matrix(results[best_bl]["hard"], max_T)

                # Figure 3: MOSAIC vs best baseline
                plot_schedule_comparison(
                    algo=algo, nq=nq,
                    method_pairs=[
                        ("MOSAIC", mosaic_mat, results["MOSAIC"]["metrics"]["hard_cost"]),
                        (best_bl, best_mat, results[best_bl]["metrics"]["hard_cost"]),
                    ],
                    tech_names=art["tech_names"], K=art["K"],
                    max_layers=max_T,
                    title_prefix="MOSAIC vs Best Baseline:  ",
                    save_dir=args.save_dir, save_suffix="_mosaic_vs_best",
                )

                # Figure 4: remaining two baselines
                other_pairs = []
                for bl in other_bls:
                    mat = _build_assignment_matrix(results[bl]["hard"], max_T)
                    other_pairs.append(
                        (bl, mat, results[bl]["metrics"]["hard_cost"]))
                plot_schedule_comparison(
                    algo=algo, nq=nq,
                    method_pairs=other_pairs,
                    tech_names=art["tech_names"], K=art["K"],
                    max_layers=max_T,
                    title_prefix="Other Baselines:  ",
                    save_dir=args.save_dir, save_suffix="_other_baselines",
                )

            except Exception as e:
                log(f"  EVALUATION FAILED for {algo} N={nq}: {e}")
                import traceback
                traceback.print_exc()

    log_section("DONE")


if __name__ == "__main__":
    main()
