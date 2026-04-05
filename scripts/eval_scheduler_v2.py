"""
eval_scheduler_v2.py  —  MOSAIC MQT Bench Zero-Shot Evaluation Script

Evaluates a trained MOSAIC model against B1-B5 baselines on real algorithm
circuits from MQT Bench. This is a zero-shot evaluation — circuits come from
a different distribution than training (structured algorithms vs ROI-composed).

Selected algorithms (measurement-free, 28-32 qubits, compatible structure):
    qaoa, qft, qftentangled, vqe_real_amp, vqe_su2, vqe_two_local,
    wstate, qgan, qnn, portfolioqaoa, portfoliovqe

Preprocessing pipeline applied to every MQT Bench circuit:
    1. Generate at BenchmarkLevel.ALG (Qiskit QuantumCircuit objects)
    2. Transpile to {cx, rz, h, x, sx} basis — ensures all gates are 1Q or 2Q,
       no multi-qubit gates that CircuitRepresentation cannot parse
    3. Remove final measurements (remove_final_measurements)
    4. Strip classical registers (rebuild QuantumCircuit with quantum regs only)
    5. Verify num_clbits == 0, num_qubits in [qubit_min, qubit_max]
    6. Verify no 3+-qubit gates remain after transpilation
    7. Pass clean QuantumCircuit directly to CircuitRepresentation

All subsequent preprocessing (segment_circuit, build_layer_graph_arrays) and
evaluation (compute_metrics_v1, baselines) is identical to eval_scheduler_v1.py.

Differences from eval_scheduler_v1.py:
    - Circuit source: MQT Bench algorithms instead of ROI generator
    - Summary table and figure are grouped by algorithm type
    - Additional per-algorithm breakdown saved to JSON
    - No --n_circuits or --seed arguments (circuit set is deterministic)
    - --qubit_min / --qubit_max / --algorithms arguments for filtering

Usage:
    python eval_scheduler_v2.py \\
        --run_dir  results/20250101_120000_run_v1 \\
        --checkpoint best \\
        --save_dir eval_v2_out \\
        --qubit_min 28 --qubit_max 32 \\
        --show
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from qiskit import QuantumCircuit, transpile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.circuit_representation import CircuitRepresentation
from src.circuit_segmentation import segment_circuit
from src.qubit_interaction_graph import (
    build_layer_graph_arrays,
    compute_window_sizes_from_config,
)
from src.evolving_gnn import EvolvingGNN
from src.clustering_head import SegmentClustering
from src.cost_function import TotalCost
from utils.inference_utils import enforce_capacity_sequence
from utils.cost_config_reader import load_cost_config
from baselines_tier1 import baseline_b1, baseline_b3, rank_techs_by
from baselines_tier2 import baseline_b4, baseline_b5


# =============================================================================
# MQT Bench algorithm registry
# =============================================================================

# Algorithms selected for evaluation:
#   - No mid-circuit measurements or resets (terminal measurement only)
#   - Structurally compatible with CircuitRepresentation (1Q/2Q gates after transpile)
#   - Not structurally trivial (skip GHZ, graphstate which have near-zero depth)
#
# Note on qft: depth ~450 at 30q — valid zero-shot extrapolation in time dimension.
#   Results reported with caveat that T is beyond training range (55-105).
# Note on wstate: linear CX chain, low 2Q density, structurally simple.
#   Included for completeness; scheduling problem is relatively easy.

# Benchmark names for MQT Bench v2.x
# Old names (v0.x/v1.x) -> New names (v2.x):
#   realamprandom -> vqe_real_amp
#   su2random     -> vqe_su2
#   twolocalrandom -> vqe_two_local
MQT_ALGORITHMS = [
    "qaoa",           # Quantum Approximate Optimization Algorithm
    "qft",            # Quantum Fourier Transform (deep, T>105 at 30q — extrapolation)
    "qftentangled",   # QFT with entangled input
    "vqe_real_amp",   # VQE RealAmplitudes ansatz (was: realamprandom)
    "vqe_su2",        # VQE Efficient SU2 ansatz (was: su2random)
    "vqe_two_local",  # VQE Two-Local ansatz (was: twolocalrandom)
    "wstate",         # W-state preparation
    "qgan",           # Quantum GAN
    "qnn",            # Quantum Neural Network
    "portfolioqaoa",  # Portfolio optimization with QAOA
    "portfoliovqe",   # Portfolio optimization with VQE
    "bv",             # Bernstein-Vazirani (structured oracle circuit)
    "randomcircuit",  # Random circuit (good diversity proxy)
]

# Basis gate set for transpilation — ensures all gates are exactly 1Q or 2Q
MQT_BASIS_GATES = ["cx", "rz", "h", "x", "sx"]


# =============================================================================
# Logging helpers
# =============================================================================

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def log_section(title: str):
    width = 72
    print(flush=True)
    print("=" * width, flush=True)
    print(f"  {title}", flush=True)
    print("=" * width, flush=True)


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="MOSAIC MQT Bench Zero-Shot Evaluation v2")
    p.add_argument("--run_dir",    type=str, required=True,
                   help="Path to the HiPerGator run directory")
    p.add_argument("--checkpoint", type=str, default="best",
                   help="Which weights to load: 'final' | 'best' | 'last' | 'epoch_NNN'")
    p.add_argument("--save_dir",   type=str, default=None,
                   help="Directory for output files.")
    p.add_argument("--qubit_min",  type=int, default=28,
                   help="Minimum qubit count for MQT circuits (default 28)")
    p.add_argument("--qubit_max",  type=int, default=32,
                   help="Maximum qubit count for MQT circuits (default 32)")
    p.add_argument("--algorithms", type=str, default=None,
                   help="Comma-separated list of algorithms to use. "
                        "Defaults to all 10 selected algorithms.")
    p.add_argument("--show",       action="store_true",
                   help="Show summary plots interactively")
    return p.parse_args()


# =============================================================================
# MQT Bench circuit loading and preprocessing
# =============================================================================

def _strip_classical_registers(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Rebuild a QuantumCircuit keeping only quantum registers.
    Necessary because some MQT circuits carry classical registers even after
    remove_final_measurements — these cause CircuitRepresentation to fail.
    """
    new_qc = QuantumCircuit(*qc.qregs)
    for instruction in qc.data:
        # Only keep instructions that act on qubits only (no classical bits)
        if len(instruction.clbits) == 0:
            new_qc.append(instruction)
    return new_qc


def _has_multiqubit_gates(qc: QuantumCircuit) -> bool:
    """Return True if any gate acts on 3+ qubits."""
    for instruction in qc.data:
        if len(instruction.qubits) >= 3:
            return True
    return False


def load_mqt_circuits(
    algorithms:  List[str],
    qubit_min:   int,
    qubit_max:   int,
) -> List[Tuple[str, int, QuantumCircuit]]:
    """
    Load, transpile, and filter MQT Bench circuits.

    For each algorithm in `algorithms`, attempts to generate circuits for
    qubit counts in [qubit_min, qubit_max]. Applies full preprocessing pipeline:
        1. Generate at BenchmarkLevel.ALG
        2. Transpile to MQT_BASIS_GATES
        3. Remove final measurements
        4. Strip classical registers
        5. Filter: num_clbits == 0, num_qubits in range, no 3+-qubit gates

    Returns list of (algorithm_name, num_qubits, QuantumCircuit) triples.
    Logs each circuit's status.
    """
    try:
        from mqt.bench import BenchmarkLevel, get_benchmark
    except ImportError:
        raise ImportError(
            "mqt.bench is not installed. Install with: pip install mqt.bench"
        )

    circuits = []
    skipped  = []

    for algo in algorithms:
        for nq in range(qubit_min, qubit_max + 1):
            try:
                # Step 1: generate at algorithmic level
                qc_raw = get_benchmark(
                    benchmark=algo,
                    level=BenchmarkLevel.ALG,
                    circuit_size=nq,
                )

                # Step 2: transpile to 1Q/2Q basis
                qc_basis = transpile(
                    qc_raw,
                    basis_gates=MQT_BASIS_GATES,
                    optimization_level=1,
                )

                # Step 3: remove final measurements
                qc_basis.remove_final_measurements(inplace=True)

                # Step 4: strip classical registers
                qc_clean = _strip_classical_registers(qc_basis)

                # Step 5: validate
                if qc_clean.num_clbits != 0:
                    skipped.append((algo, nq, "classical bits remain"))
                    continue

                if qc_clean.num_qubits < qubit_min or qc_clean.num_qubits > qubit_max:
                    # Transpilation may have added ancillas
                    skipped.append((algo, nq,
                        f"qubit count changed to {qc_clean.num_qubits}"))
                    continue

                # Step 6: no 3+-qubit gates
                if _has_multiqubit_gates(qc_clean):
                    skipped.append((algo, nq, "3+-qubit gates after transpile"))
                    continue

                # Step 7: must have at least some gates
                if qc_clean.size() == 0:
                    skipped.append((algo, nq, "empty circuit after preprocessing"))
                    continue

                circuits.append((algo, qc_clean.num_qubits, qc_clean))
                log(f"  [OK] {algo:20s} N={qc_clean.num_qubits:2d}  "
                    f"gates={qc_clean.size():5d}")

            except Exception as e:
                skipped.append((algo, nq, str(e)[:60]))

    if skipped:
        log(f"  Skipped {len(skipped)} (algo, nq) combinations:")
        for algo, nq, reason in skipped[:10]:
            log(f"    {algo:20s} N={nq:2d}  reason: {reason}")
        if len(skipped) > 10:
            log(f"    ... and {len(skipped) - 10} more")

    return circuits


# =============================================================================
# Load run artifacts  (identical to eval_scheduler_v1.py)
# =============================================================================

def _load_snapshot_cfg(snapshot_path: str) -> dict:
    ns: dict = {}
    with open(snapshot_path, "r") as f:
        exec(f.read(), ns)  # noqa: S102
    return ns


def load_run_artifacts(run_dir: str, checkpoint: str, device: str = "cpu") -> dict:
    log(f"Loading run artifacts from: {run_dir}")

    arch_path = os.path.join(run_dir, "model_arch_params.json")
    with open(arch_path) as f:
        arch = json.load(f)
    gnn_arch = arch["EvolvingGNN"]
    cls_arch  = arch["SegmentClustering"]
    log(f"  arch loaded: gru_hidden={gnn_arch['gru_hidden_dim']}, K={cls_arch['num_clusters']}")

    cost_cfg_path = os.path.join(run_dir, "cost_config_snapshot.json")
    config = load_cost_config(cost_cfg_path)
    K = len(config["techs"])
    tech_names = [t.get("name", f"tech{k}") for k, t in enumerate(config["techs"])]
    caps = torch.tensor(
        [float(t["capacity"]["max_qubits"]) for t in config["techs"]],
        dtype=torch.float32,
    )
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
        "evol_model":    evol_model,
        "cluster_module": cluster_module,
        "cost_module":   total_cost_module,
        "config":        config,
        "dataset_cfg":   dataset_cfg,
        "K":             K,
        "tech_names":    tech_names,
        "caps":          caps,
        "w_short":       w_short,
        "w_long":        w_long,
        "device":        device,
    }


# =============================================================================
# Circuit preprocessing
# =============================================================================

def _build_layer_data_list(rep: CircuitRepresentation,
                            w_short: int, w_long: int) -> List[Data]:
    arrays = build_layer_graph_arrays(rep, w_short, w_long)
    return [
        Data(
            x          = torch.tensor(x_np,  dtype=torch.float32),
            edge_index = torch.tensor(ei_np, dtype=torch.long),
            edge_attr  = torch.tensor(ea_np, dtype=torch.float32),
        )
        for x_np, ei_np, ea_np in arrays
    ]


def preprocess_circuit(qc: QuantumCircuit, dataset_cfg: dict,
                        w_short: int, w_long: int):
    """
    Convert a clean QuantumCircuit to the internal representation used by
    the training pipeline. Identical to eval_scheduler_v1.py.
    """
    rep = CircuitRepresentation(qc)
    seg_mode = dataset_cfg["segmentation_mode"]
    seg_thr  = float(dataset_cfg["segment_threshold"])
    segments, _ = segment_circuit(rep.layers, mode=seg_mode, threshold=seg_thr)
    layer_data_list = _build_layer_data_list(rep, w_short, w_long)
    return rep, segments, layer_data_list


# =============================================================================
# MOSAIC inference
# =============================================================================

def run_inference(
    evol_model:      EvolvingGNN,
    cluster_module:  SegmentClustering,
    layer_data_list: List[Data],
) -> List[torch.Tensor]:
    with torch.no_grad():
        h_seq, _ = evol_model(layer_data_list)
        P_seq    = cluster_module(h_seq, graphs=layer_data_list)
    return P_seq


# =============================================================================
# Metrics  (identical to eval_scheduler_v1.py)
# =============================================================================

def compute_idle_decoherence_placement(
    hard_assignments: List[torch.Tensor],
    rep:              CircuitRepresentation,
    config:           dict,
    K:                int,
) -> float:
    best_T2 = rank_techs_by(config, "T2")[0]
    T = min(len(hard_assignments), len(rep.layers))
    total_idle = correct_idle = 0
    for t in range(T):
        ha_t   = hard_assignments[t].cpu()
        active: set = set()
        for _, qargs in rep.layers[t].gates:
            for q in qargs:
                active.add(q)
        for q in range(ha_t.shape[0]):
            if q not in active:
                total_idle += 1
                if ha_t[q].item() == best_T2:
                    correct_idle += 1
    return correct_idle / max(total_idle, 1)


def compute_metrics_v1(
    hard_assignments: List[torch.Tensor],
    rep:              CircuitRepresentation,
    segments,
    cost_module:      TotalCost,
    caps:             torch.Tensor,
    K:                int,
    config:           dict,
    device:           str,
) -> dict:
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

    idle_decoherence_rate = compute_idle_decoherence_placement(
        hard_assignments, rep, config, K)

    return {
        "hard_cost":             hard_cost,
        "remote_2q_cut_rate":    remote_2q_cut_rate,
        "mean_movement":         mean_movement,
        "idle_decoherence_rate": idle_decoherence_rate,
        "T": T,
        "N": N,
    }


# =============================================================================
# Summary table, per-algorithm table, figures, JSON
# =============================================================================

METHOD_NAMES = ["MOSAIC", "B1", "B3", "B4", "B5"]

METRICS_CFG = [
    ("hard_cost",             "Hard TotalCost",            "↓", ".4f"),
    ("remote_2q_cut_rate",    "Remote 2Q Cut Rate (%)",    "↓", ".2f"),
    ("mean_movement",         "Mean Temporal Movement",    "↓", ".3f"),
    ("idle_decoherence_rate", "Idle Decoherence Placement","↑", ".4f"),
]

SCALE_PCT = {"remote_2q_cut_rate"}


def _scale(key: str, val: float) -> float:
    return val * 100.0 if key in SCALE_PCT else val


def _format_aggregate_table_lines(
    all_metrics:   Dict[str, List[dict]],
    circuit_info:  List[Tuple[str, int]],
    tech_names:    List[str],
    K:             int,
) -> List[str]:
    """Build aggregate comparison table as lines (shared by print and txt save)."""
    n = len(circuit_info)
    lines = []
    lines.append(f"  Circuits evaluated : {n}")
    lines.append(f"  Technologies (K={K}): {', '.join(tech_names)}")
    lines.append("")

    col_w = 38
    val_w = 18
    header = f"  {'Metric':<{col_w}}"
    for m in METHOD_NAMES:
        header += f"  {m:^{val_w}}"
    lines.append(header)
    lines.append("  " + "-" * (col_w + (val_w + 2) * len(METHOD_NAMES)))

    for key, label, direction, fmt in METRICS_CFG:
        row = f"  {label + ' ' + direction:<{col_w}}"
        for method in METHOD_NAMES:
            vals = [_scale(key, m[key]) for m in all_metrics[method]]
            cell = f"{np.mean(vals):{fmt}} ± {np.std(vals):{fmt}}"
            row += f"  {cell:^{val_w}}"
        lines.append(row)

    lines.append("")
    for key, label, direction, fmt in METRICS_CFG:
        means = {m: np.mean([_scale(key, x[key]) for x in all_metrics[m]])
                 for m in METHOD_NAMES}
        winner = (min if direction == "↓" else max)(means, key=lambda m: means[m])
        lines.append(f"  {'Best ' + label + ':':<{col_w + 2}}  {winner}")
    lines.append("")

    # Win rates (hard_cost only)
    baselines = [m for m in METHOD_NAMES if m != "MOSAIC"]
    n_c = len(all_metrics["MOSAIC"])
    lines.append("  Win Rates (MOSAIC hard_cost < baseline):")
    for bl in baselines:
        wins = sum(
            1 for i in range(n_c)
            if all_metrics["MOSAIC"][i]["hard_cost"] < all_metrics[bl][i]["hard_cost"]
        )
        lines.append(f"    MOSAIC vs {bl}: {wins}/{n_c}  ({100.0*wins/max(n_c,1):.1f}%)")
    lines.append("")
    return lines


def _format_per_algorithm_table_lines(
    all_metrics:  Dict[str, List[dict]],
    circuit_info: List[Tuple[str, int]],
) -> List[str]:
    """Build per-algorithm breakdown as lines (shared by print and txt save)."""
    algo_groups: Dict[str, List[int]] = {}
    for i, (algo, _nq) in enumerate(circuit_info):
        algo_groups.setdefault(algo, []).append(i)

    col_w = 20
    lines = []
    lines.append(
        f"  {'Algorithm':<{col_w}}  {'N_circ':>6}  "
        f"{'MOSAIC':>10}  {'BestBase':>10}  {'Winner':>10}  {'MOSAIC_cut%':>11}"
    )
    lines.append("  " + "-" * 72)

    for algo in sorted(algo_groups.keys()):
        idxs = algo_groups[algo]
        mosaic_costs = [all_metrics["MOSAIC"][i]["hard_cost"] for i in idxs]
        base_costs_by_method = {}
        for method in METHOD_NAMES:
            if method == "MOSAIC":
                continue
            base_costs_by_method[method] = np.mean(
                [all_metrics[method][i]["hard_cost"] for i in idxs])

        best_base_method = min(base_costs_by_method, key=lambda m: base_costs_by_method[m])
        best_base_cost   = base_costs_by_method[best_base_method]
        mosaic_mean      = np.mean(mosaic_costs)
        winner           = "MOSAIC" if mosaic_mean <= best_base_cost else best_base_method

        mosaic_cut = np.mean([all_metrics["MOSAIC"][i]["remote_2q_cut_rate"] * 100
                              for i in idxs])

        lines.append(
            f"  {algo:<{col_w}}  {len(idxs):>6}  "
            f"{mosaic_mean:>10.4f}  {best_base_cost:>10.4f}  "
            f"{winner:>10}  {mosaic_cut:>10.1f}%"
        )
    lines.append("")
    return lines


def print_aggregate_table(
    all_metrics:   Dict[str, List[dict]],
    circuit_info:  List[Tuple[str, int]],
    tech_names:    List[str],
    K:             int,
):
    """Aggregate comparison table across all MQT circuits."""
    log_section("AGGREGATE COMPARISON TABLE (all MQT circuits)")
    for line in _format_aggregate_table_lines(all_metrics, circuit_info, tech_names, K):
        print(line)
    print()


def print_per_algorithm_table(
    all_metrics:  Dict[str, List[dict]],
    circuit_info: List[Tuple[str, int]],
):
    """Print a compact per-algorithm breakdown showing MOSAIC vs best baseline."""
    log_section("PER-ALGORITHM BREAKDOWN (MOSAIC hard cost vs best baseline)")
    for line in _format_per_algorithm_table_lines(all_metrics, circuit_info):
        print(line)
    print()


def plot_mqt_figures(
    all_metrics:  Dict[str, List[dict]],
    circuit_info: List[Tuple[str, int]],
    save_dir:     str,
    show:         bool = False,
):
    """
    Two figures:
    Fig 1: 4-panel aggregate bar chart (same as eval_scheduler_v1.py)
    Fig 2: Per-algorithm MOSAIC vs best-baseline hard cost comparison
    """
    colors_all = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]  # MOSAIC, B1, B3, B4, B5

    # --- Figure 1: aggregate bar chart ---
    fig1, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig1.suptitle("MOSAIC vs Baselines — MQT Bench (aggregate)",
                  fontsize=12, fontweight="bold")

    x = np.arange(len(METHOD_NAMES))
    for ax, (key, label, direction, fmt) in zip(axes, METRICS_CFG):
        means = [np.mean([_scale(key, m[key]) for m in all_metrics[method]])
                 for method in METHOD_NAMES]
        stds  = [np.std( [_scale(key, m[key]) for m in all_metrics[method]])
                 for method in METHOD_NAMES]
        ax.bar(x, means, yerr=stds, capsize=4, color=colors_all,
               alpha=0.85, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_NAMES, fontsize=8, rotation=30, ha="right")
        ax.set_title(f"{label}\n({direction} better)", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.annotate(f"{means[0]:{fmt}}",
                    xy=(0, means[0] + stds[0]),
                    ha="center", va="bottom", fontsize=7, color="#2196F3")

    plt.tight_layout()
    p1 = os.path.join(save_dir, "mqt_aggregate_comparison.png")
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    log(f"  Fig 1 saved: {p1}")
    if show:
        plt.show()
    plt.close(fig1)

    # --- Figure 2: per-algorithm MOSAIC vs best-baseline hard cost ---
    algo_groups: Dict[str, List[int]] = {}
    for i, (algo, _nq) in enumerate(circuit_info):
        algo_groups.setdefault(algo, []).append(i)

    algos = sorted(algo_groups.keys())
    mosaic_means = []
    best_base_means = []
    best_base_labels = []

    for algo in algos:
        idxs = algo_groups[algo]
        mosaic_means.append(
            np.mean([all_metrics["MOSAIC"][i]["hard_cost"] for i in idxs]))
        base_by_m = {m: np.mean([all_metrics[m][i]["hard_cost"] for i in idxs])
                     for m in METHOD_NAMES if m != "MOSAIC"}
        best_m = min(base_by_m, key=lambda m: base_by_m[m])
        best_base_means.append(base_by_m[best_m])
        best_base_labels.append(best_m)

    fig2, ax = plt.subplots(figsize=(max(8, len(algos) * 1.1), 5))
    xb = np.arange(len(algos))
    w  = 0.35
    bars_m = ax.bar(xb - w/2, mosaic_means,  width=w, label="MOSAIC",
                    color="#2196F3", alpha=0.85, edgecolor="black", linewidth=0.6)
    bars_b = ax.bar(xb + w/2, best_base_means, width=w, label="Best Baseline",
                    color="#FF9800", alpha=0.85, edgecolor="black", linewidth=0.6)

    # Annotate best-baseline label above each bar
    for j, (bar, lbl) in enumerate(zip(bars_b, best_base_labels)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * max(best_base_means),
                lbl, ha="center", va="bottom", fontsize=7, color="#FF9800")

    ax.set_xticks(xb)
    ax.set_xticklabels(algos, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Hard TotalCost (↓ better)")
    ax.set_title("MOSAIC vs Best Baseline per Algorithm — MQT Bench", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(save_dir, "mqt_per_algorithm_comparison.png")
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    log(f"  Fig 2 saved: {p2}")
    if show:
        plt.show()
    plt.close(fig2)


def save_results_json(
    all_metrics:  Dict[str, List[dict]],
    circuit_info: List[Tuple[str, int]],
    save_dir:     str,
    tech_names:   List[str],
    run_dir:      str,
):
    """Save per-circuit metrics with algorithm labels for all methods."""
    baselines = [m for m in METHOD_NAMES if m != "MOSAIC"]
    n = len(all_metrics["MOSAIC"])
    win_rates = {
        bl: {
            "wins":    sum(1 for i in range(n)
                           if all_metrics["MOSAIC"][i]["hard_cost"]
                              < all_metrics[bl][i]["hard_cost"]),
            "total":   n,
            "win_pct": round(
                100.0 * sum(1 for i in range(n)
                            if all_metrics["MOSAIC"][i]["hard_cost"]
                               < all_metrics[bl][i]["hard_cost"]) / max(n, 1), 2),
        }
        for bl in baselines
    }

    summary = {
        "run_dir":      run_dir,
        "n_circuits":   len(circuit_info),
        "circuit_info": [{"algo": a, "nq": nq} for a, nq in circuit_info],
        "tech_names":   tech_names,
        "win_rates":    win_rates,
        "methods":      {},
    }
    for method in METHOD_NAMES:
        mlist = all_metrics[method]
        summary["methods"][method] = {
            "per_circuit": [
                {"algo": circuit_info[i][0],
                 "nq":   circuit_info[i][1],
                 **{k: (int(v) if k in ("T", "N") else float(v))
                    for k, v in m.items()}}
                for i, m in enumerate(mlist)
            ],
            "aggregate_means": {
                key: float(np.mean([_scale(key, m[key]) for m in mlist]))
                for key, *_ in METRICS_CFG
            },
            "aggregate_stds": {
                key: float(np.std([_scale(key, m[key]) for m in mlist]))
                for key, *_ in METRICS_CFG
            },
        }

    # Per-algorithm means
    algo_groups: Dict[str, List[int]] = {}
    for i, (algo, _nq) in enumerate(circuit_info):
        algo_groups.setdefault(algo, []).append(i)

    summary["per_algorithm"] = {}
    for algo, idxs in algo_groups.items():
        summary["per_algorithm"][algo] = {}
        for method in METHOD_NAMES:
            summary["per_algorithm"][algo][method] = {
                key: float(np.mean([_scale(key, all_metrics[method][i][key])
                                    for i in idxs]))
                for key, *_ in METRICS_CFG
            }

    out_path = os.path.join(save_dir, "mqt_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"  Results saved: {out_path}")


def save_summary_txt(
    all_metrics:  Dict[str, List[dict]],
    circuit_info: List[Tuple[str, int]],
    save_dir:     str,
    tech_names:   List[str],
    K:            int,
):
    """Save summary.txt with aggregate table and per-algorithm breakdown."""
    lines = ["AGGREGATE COMPARISON TABLE (all MQT circuits)", "=" * 72, ""]
    lines += _format_aggregate_table_lines(all_metrics, circuit_info, tech_names, K)
    lines += ["", "PER-ALGORITHM BREAKDOWN (MOSAIC hard cost vs best baseline)", "=" * 72, ""]
    lines += _format_per_algorithm_table_lines(all_metrics, circuit_info)
    out_path = os.path.join(save_dir, "summary.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    log(f"  Summary txt saved: {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    args   = parse_args()
    device = "cpu"

    # ---- Output directory ----
    if args.save_dir is None:
        save_dir = os.path.join(args.run_dir, f"eval_mqt_{args.checkpoint}")
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    log_section("MOSAIC MQT BENCH ZERO-SHOT EVALUATION v2")
    log(f"Run dir     : {args.run_dir}")
    log(f"Checkpoint  : {args.checkpoint}")
    log(f"Qubit range : [{args.qubit_min}, {args.qubit_max}]")
    log(f"Save dir    : {save_dir}")

    # ---- Parse algorithm list ----
    if args.algorithms:
        algorithms = [a.strip() for a in args.algorithms.split(",")]
        unknown = [a for a in algorithms if a not in MQT_ALGORITHMS]
        if unknown:
            log(f"WARNING: unknown algorithms (will attempt anyway): {unknown}")
    else:
        algorithms = MQT_ALGORITHMS

    log(f"Algorithms  : {', '.join(algorithms)}")

    # ---- Load model ----
    log_section("LOADING RUN ARTIFACTS")
    art = load_run_artifacts(args.run_dir, args.checkpoint, device=device)

    evol_model     = art["evol_model"]
    cluster_module = art["cluster_module"]
    cost_module    = art["cost_module"]
    config         = art["config"]
    dataset_cfg    = art["dataset_cfg"]
    K              = art["K"]
    tech_names     = art["tech_names"]
    caps           = art["caps"]
    w_short        = art["w_short"]
    w_long         = art["w_long"]

    # ---- Load and preprocess MQT circuits ----
    log_section("LOADING MQT BENCH CIRCUITS")
    mqt_circuits = load_mqt_circuits(algorithms, args.qubit_min, args.qubit_max)

    if not mqt_circuits:
        log("ERROR: No circuits loaded. Check mqt.bench installation and algorithm names.")
        return

    log(f"  Total circuits loaded: {len(mqt_circuits)}")

    # ---- Per-circuit evaluation loop ----
    log_section("RUNNING MOSAIC + BASELINES ON MQT CIRCUITS")
    all_metrics: Dict[str, List[dict]] = {m: [] for m in METHOD_NAMES}
    circuit_info: List[Tuple[str, int]] = []
    t0 = time.time()

    for i, (algo, nq, qc) in enumerate(mqt_circuits):
        t_circ = time.time()

        try:
            rep, segments, layer_data_list = preprocess_circuit(
                qc, dataset_cfg, w_short, w_long)
        except Exception as e:
            log(f"  [{i+1:3d}] SKIP {algo}(N={nq}): preprocess failed — {e}")
            continue

        T = len(layer_data_list)
        N = rep.num_qubits

        # Warn if circuit depth is outside training range
        depth_note = ""
        if T > 105:
            depth_note = f" [T={T} > train max=105, extrapolation]"
        elif T < 55:
            depth_note = f" [T={T} < train min=55, extrapolation]"

        # --- MOSAIC ---
        try:
            P_seq       = run_inference(evol_model, cluster_module, layer_data_list)
            mosaic_hard = enforce_capacity_sequence(P_seq, caps)
            mosaic_m    = compute_metrics_v1(
                mosaic_hard, rep, segments, cost_module, caps, K, config, device)
        except Exception as e:
            log(f"  [{i+1:3d}] SKIP {algo}(N={nq}): MOSAIC inference failed — {e}")
            continue

        # --- B1, B3 ---
        b1_hard = baseline_b1(rep, caps, config, K)
        b3_hard = baseline_b3(rep, caps, config, K)
        b1_m = compute_metrics_v1(b1_hard, rep, segments, cost_module, caps, K, config, device)
        b3_m = compute_metrics_v1(b3_hard, rep, segments, cost_module, caps, K, config, device)

        # --- B4: Wu beam search ---
        b4_hard = baseline_b4(rep, caps, config, K, seed=i)
        b4_m    = compute_metrics_v1(b4_hard, rep, segments, cost_module, caps, K, config, device)

        # --- B5: Burt-style FM + gate-grouping ---
        b5_hard = baseline_b5(rep, caps, config, K)
        b5_m    = compute_metrics_v1(b5_hard, rep, segments, cost_module, caps, K, config, device)

        # Record
        all_metrics["MOSAIC"].append(mosaic_m)
        all_metrics["B1"].append(b1_m)
        all_metrics["B3"].append(b3_m)
        all_metrics["B4"].append(b4_m)
        all_metrics["B5"].append(b5_m)
        circuit_info.append((algo, nq))

        elapsed = time.time() - t_circ
        log(
            f"  [{i+1:3d}] {algo:15s} N={N:2d} T={T:4d}{depth_note} | "
            f"MOSAIC={mosaic_m['hard_cost']:.3f} "
            f"B1={b1_m['hard_cost']:.3f} "
            f"B3={b3_m['hard_cost']:.3f} "
            f"B4={b4_m['hard_cost']:.3f} "
            f"B5={b5_m['hard_cost']:.3f} "
            f"({elapsed:.1f}s)"
        )

    total_time = time.time() - t0
    n_done = len(circuit_info)
    log(f"\nDone: {n_done} circuits in {total_time:.1f}s "
        f"({total_time / max(n_done, 1):.2f}s/circuit)")

    if n_done == 0:
        log("No circuits successfully evaluated. Exiting.")
        return

    # ---- Summary ----
    print_aggregate_table(all_metrics, circuit_info, tech_names, K)
    print_per_algorithm_table(all_metrics, circuit_info)

    log_section("GENERATING FIGURES")
    plot_mqt_figures(all_metrics, circuit_info, save_dir, show=args.show)

    log_section("SAVING RESULTS")
    save_results_json(all_metrics, circuit_info, save_dir, tech_names, args.run_dir)
    save_summary_txt(all_metrics, circuit_info, save_dir, tech_names, K)

    log_section("EVALUATION COMPLETE")
    log(f"All outputs saved to: {save_dir}")


if __name__ == "__main__":
    main()
