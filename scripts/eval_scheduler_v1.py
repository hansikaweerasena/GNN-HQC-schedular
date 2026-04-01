"""
eval_scheduler_v1.py  —  MOSAIC Baseline Comparison Evaluation Script

Evaluates a trained MOSAIC model against Tier-1 baselines (B1, B3) and
freshly-generated circuits. Reports the four comparison metrics:

    1. Hard TotalCost             (↓ lower is better)
    2. Remote 2Q Cut Rate (%)     (↓ lower is better)
    3. Mean Temporal Movement     (↓ lower is better)
    4. Idle Decoherence Placement (↑ higher is better)

The MOSAIC model loading, circuit generation, and preprocessing pipeline are
identical to eval_scheduler.py. Per-circuit visual panels and summary figures
from the old script are intentionally omitted; use eval_scheduler.py for those.

Usage:
    python eval_scheduler_v1.py \\
        --run_dir  results/20250101_120000_run_v1 \\
        --checkpoint best \\
        --n_circuits 300 \\
        --seed 99999 \\
        --save_dir eval_v1_out
"""

import argparse
import importlib.util
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.circuit_sources import build_provider
from src.circuit_representation import CircuitRepresentation
from src.circuit_segmentation import segment_circuit
from src.qubit_interaction_graph import (
    build_layer_graph_arrays,
    compute_window_sizes_from_config,
    NODE_FEAT_DIM,
    EDGE_FEAT_DIM,
)
from src.evolving_gnn import EvolvingGNN
from src.clustering_head import SegmentClustering
from src.cost_function import TotalCost, CapacityPenalty
from utils.inference_utils import enforce_capacity_sequence
from utils.cost_config_reader import load_cost_config
from baselines_tier1 import baseline_b1, baseline_b3, rank_techs_by
from baselines_tier2 import baseline_b4, baseline_b5


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
    p = argparse.ArgumentParser(description="MOSAIC Baseline Comparison Evaluation v1")
    p.add_argument("--run_dir",    type=str, required=True,
                   help="Path to the HiPerGator run directory")
    p.add_argument("--checkpoint", type=str, default="best",
                   help="Which weights to load: 'final' | 'best' | 'last' | 'epoch_NNN'")
    p.add_argument("--n_circuits", type=int, default=300,
                   help="Number of circuits to evaluate")
    p.add_argument("--seed",       type=int, default=99999,
                   help="Seed base for eval circuit generation")
    p.add_argument("--save_dir",   type=str, default=None,
                   help="Directory for output files. Defaults to <run_dir>/eval_syn_<checkpoint>/")
    p.add_argument("--num_qubits", type=int, default=None,
                   help="Override num_qubits for generated circuits. "
                        "Defaults to value in CIRCUIT_SOURCE_CFG from run_dir.")
    p.add_argument("--show",       action="store_true",
                   help="Show summary comparison plot interactively")
    return p.parse_args()


# =============================================================================
# Load run artifacts  (identical to eval_scheduler.py)
# =============================================================================

def _load_snapshot_cfg(snapshot_path: str) -> dict:
    ns: dict = {}
    with open(snapshot_path, "r") as f:
        exec(f.read(), ns)  # noqa: S102
    return ns


def load_run_artifacts(run_dir: str, checkpoint: str, device: str = "cpu") -> dict:
    """
    Load every artifact saved by train_hipergator.py and reconstruct:
      - EvolvingGNN + SegmentClustering (weights from chosen checkpoint)
      - TotalCost (tau set to tau_min for eval)
      - CIRCUIT_SOURCE_CFG, DATASET_CFG
      - tech_names, caps, K, w_short, w_long, config (raw dict)
    """
    log(f"Loading run artifacts from: {run_dir}")

    # --- arch params ---
    arch_path = os.path.join(run_dir, "model_arch_params.json")
    with open(arch_path) as f:
        arch = json.load(f)
    gnn_arch = arch["EvolvingGNN"]
    cls_arch  = arch["SegmentClustering"]
    log(f"  arch loaded: gru_hidden={gnn_arch['gru_hidden_dim']}, K={cls_arch['num_clusters']}")

    # --- cost config ---
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

    # --- scheduler config ---
    snap_path = os.path.join(run_dir, "scheduler_config_snapshot.py")
    snap = _load_snapshot_cfg(snap_path)
    circuit_source_cfg = snap["CIRCUIT_SOURCE_CFG"]
    dataset_cfg        = snap["DATASET_CFG"]
    log(f"  circuit source: {circuit_source_cfg['name']}, "
        f"seg_mode={dataset_cfg['segmentation_mode']}")

    # --- rebuild models ---
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

    # --- load weights ---
    ckpt_lower = checkpoint.lower()
    if ckpt_lower == "final":
        evol_model.load_state_dict(
            torch.load(os.path.join(run_dir, "evol_model.pt"), map_location=device))
        cluster_module.load_state_dict(
            torch.load(os.path.join(run_dir, "cluster_head.pt"), map_location=device))
        log(f"  weights: final (evol_model.pt + cluster_head.pt)")
    else:
        if ckpt_lower == "best":
            ckpt_file = os.path.join(run_dir, "checkpoint_best.pt")
        elif ckpt_lower == "last":
            ckpt_file = os.path.join(run_dir, "checkpoint_last.pt")
        elif ckpt_lower.startswith("epoch_"):
            n = ckpt_lower.split("_")[1]
            ckpt_file = os.path.join(run_dir, f"checkpoint_epoch_{n.zfill(3)}.pt")
        else:
            raise ValueError(f"Unknown checkpoint option: '{checkpoint}'. "
                             f"Use: final | best | last | epoch_NNN")
        ckpt_dict = torch.load(ckpt_file, map_location=device)
        evol_model.load_state_dict(ckpt_dict["evol_model"])
        cluster_module.load_state_dict(ckpt_dict["cluster_head"])
        epoch     = ckpt_dict.get("epoch", "?")
        test_loss = ckpt_dict.get("test_loss", float("nan"))
        log(f"  weights: {os.path.basename(ckpt_file)} "
            f"(epoch={epoch}, test_loss={test_loss:.4f})")

    evol_model.eval()
    cluster_module.eval()

    # --- cost module: set tau = tau_min for eval ---
    total_cost_module = TotalCost(config).to(device)
    tau_min = total_cost_module._tau_min
    total_cost_module.tau.fill_(tau_min)
    log(f"  TotalCost built (tau set to tau_min={tau_min:.4f})")

    return {
        "evol_model":         evol_model,
        "cluster_module":     cluster_module,
        "cost_module":        total_cost_module,
        "config":             config,
        "circuit_source_cfg": circuit_source_cfg,
        "dataset_cfg":        dataset_cfg,
        "K":                  K,
        "tech_names":         tech_names,
        "caps":               caps,
        "w_short":            w_short,
        "w_long":             w_long,
        "device":             device,
    }


# =============================================================================
# Circuit preprocessing  (identical to eval_scheduler.py)
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


def preprocess_circuit(qc, dataset_cfg: dict, w_short: int, w_long: int):
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
    evol_model:     EvolvingGNN,
    cluster_module: SegmentClustering,
    layer_data_list: List[Data],
) -> List[torch.Tensor]:
    """Return P_seq: List[Tensor[N, K]] — soft assignment per layer."""
    with torch.no_grad():
        h_seq, _ = evol_model(layer_data_list)
        P_seq    = cluster_module(h_seq, graphs=layer_data_list)
    return P_seq


# =============================================================================
# Metrics
# =============================================================================

def compute_idle_decoherence_placement(
    hard_assignments: List[torch.Tensor],
    rep:              CircuitRepresentation,
    config:           dict,
    K:                int,
) -> float:
    """
    Idle-qubit decoherence placement rate.

    Fraction of (qubit, layer) pairs where the qubit is idle AND is assigned
    to the technology with the longest T2 (best decoherence robustness).
    Higher is better. Returns 1.0 if there are no idle qubit-layers.
    """
    best_T2 = rank_techs_by(config, "T2")[0]
    T = min(len(hard_assignments), len(rep.layers))
    total_idle   = 0
    correct_idle = 0

    for t in range(T):
        ha_t   = hard_assignments[t].cpu()
        active: set = set()
        for _, qargs in rep.layers[t].gates:
            for q in qargs:
                active.add(q)
        N = ha_t.shape[0]
        for q in range(N):
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
    """
    Compute the four comparison metrics for a given hard assignment schedule.

    Uses one-hot P_seq constructed from hard_assignments — identical for both
    MOSAIC (post-hardening) and baselines (which produce hard assignments directly).

    Returns dict with keys:
        hard_cost, remote_2q_cut_rate, mean_movement, idle_decoherence_rate
        (plus T and N for bookkeeping)
    """
    T = len(hard_assignments)
    N = rep.num_qubits

    # Build one-hot P_seq from hard assignments (same for all methods)
    P_seq_hard = []
    for t in range(T):
        P_t = torch.zeros(N, K, dtype=torch.float32, device=device)
        P_t[torch.arange(N, device=device), hard_assignments[t].to(device)] = 1.0
        P_seq_hard.append(P_t)

    # 1. Hard TotalCost
    with torch.no_grad():
        hard_out = cost_module(P_seq_hard, segments, rep)
    hard_cost = hard_out["total_cost"].item()

    # 2. Remote 2Q cut rate
    total_2q = 0
    cut_2q   = 0
    for t in range(min(T, len(rep.layers))):
        ha_t = hard_assignments[t].cpu()
        for gate_name, qargs in rep.layers[t].gates:
            if len(qargs) == 2:
                u, v = qargs
                total_2q += 1
                if ha_t[u].item() != ha_t[v].item():
                    cut_2q += 1
    remote_2q_cut_rate = cut_2q / max(total_2q, 1)

    # 3. Mean temporal movement
    movements = []
    for t in range(T - 1):
        moved = (hard_assignments[t].cpu() != hard_assignments[t + 1].cpu()
                 ).float().sum().item()
        movements.append(moved)
    mean_movement = float(np.mean(movements)) if movements else 0.0

    # 4. Idle decoherence placement rate
    idle_decoherence_rate = compute_idle_decoherence_placement(
        hard_assignments, rep, config, K)

    return {
        "hard_cost":            hard_cost,
        "remote_2q_cut_rate":   remote_2q_cut_rate,
        "mean_movement":        mean_movement,
        "idle_decoherence_rate": idle_decoherence_rate,
        "T": T,
        "N": N,
    }


# =============================================================================
# Summary table and figure
# =============================================================================

METHOD_NAMES = ["MOSAIC", "B1", "B3", "B4", "B5"]

METRICS_CFG = [
    # (key,                    display_label,                  direction, fmt)
    ("hard_cost",              "Hard TotalCost",               "↓",       ".4f"),
    ("remote_2q_cut_rate",     "Remote 2Q Cut Rate (%)",       "↓",       ".2f"),
    ("mean_movement",          "Mean Temporal Movement",       "↓",       ".3f"),
    ("idle_decoherence_rate",  "Idle Decoherence Placement",   "↑",       ".4f"),
]

# For display: cut rate is stored as fraction [0,1], show as percentage
SCALE_PCT = {"remote_2q_cut_rate"}


def _scale(key: str, val: float) -> float:
    return val * 100.0 if key in SCALE_PCT else val


def _format_comparison_table_lines(
    all_metrics:    Dict[str, List[dict]],
    tech_names:     List[str],
    K:              int,
    n_circuits:     int,
    number_of_qubits: int,
) -> List[str]:
    """Build comparison table as a list of lines (shared by print and txt save)."""
    lines = []
    lines.append(f"  Circuits evaluated : {n_circuits}")
    lines.append(f"  Number of qubits   : {number_of_qubits}")
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
    lines.append("  " + "-" * (col_w + (val_w + 2) * len(METHOD_NAMES)))
    for key, label, direction, fmt in METRICS_CFG:
        means = {m: np.mean([_scale(key, x[key]) for x in all_metrics[m]])
                 for m in METHOD_NAMES}
        winner = (min if direction == "↓" else max)(means, key=lambda m: means[m])
        lines.append(f"  {'Best ' + label + ':':<{col_w + 2}}  {winner}")

    lines.append("")

    # Win rates (hard_cost only)
    baselines = [m for m in METHOD_NAMES if m != "MOSAIC"]
    n = len(all_metrics["MOSAIC"])
    lines.append("  Win Rates (MOSAIC hard_cost < baseline):")
    for bl in baselines:
        wins = sum(
            1 for i in range(n)
            if all_metrics["MOSAIC"][i]["hard_cost"] < all_metrics[bl][i]["hard_cost"]
        )
        lines.append(f"    MOSAIC vs {bl}: {wins}/{n}  ({100.0*wins/max(n,1):.1f}%)")
    lines.append("")
    return lines


def print_comparison_table(
    all_metrics:      Dict[str, List[dict]],
    tech_names:       List[str],
    K:                int,
    n_circuits:       int,
    number_of_qubits: int,
):
    log_section("BASELINE COMPARISON TABLE")
    for line in _format_comparison_table_lines(
            all_metrics, tech_names, K, n_circuits, number_of_qubits):
        print(line)
    print()


def plot_comparison_figure(
    all_metrics:      Dict[str, List[dict]],
    save_dir:         str,
    number_of_qubits: int,
    show:             bool = False,
):
    """Simple 4-panel bar chart: one panel per metric, bars = methods."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("MOSAIC vs Tier-1 Baselines", fontsize=13, fontweight="bold")

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]  # MOSAIC, B1, B3, B4, B5
    x = np.arange(len(METHOD_NAMES))

    for ax, (key, label, direction, fmt) in zip(axes, METRICS_CFG):
        means = [np.mean([_scale(key, m[key]) for m in all_metrics[method]])
                 for method in METHOD_NAMES]
        stds  = [np.std([_scale(key, m[key])  for m in all_metrics[method]])
                 for method in METHOD_NAMES]

        bars = ax.bar(x, means, yerr=stds, capsize=4,
                      color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_NAMES, fontsize=9)
        ax.set_title(f"{label}\n({direction} better)", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Annotate MOSAIC bar with its value
        ax.annotate(
            f"{means[0]:{fmt}}",
            xy=(0, means[0] + stds[0]),
            ha="center", va="bottom", fontsize=7.5, color="#2196F3",
        )

    plt.tight_layout()
    fig_path = os.path.join(save_dir, f"baseline_comparison_N{number_of_qubits}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    log(f"  Comparison figure saved: {fig_path}")
    if show:
        plt.show()
    plt.close(fig)


def save_results_json(
    all_metrics:      Dict[str, List[dict]],
    save_dir:         str,
    n_circuits:       int,
    tech_names:       List[str],
    number_of_qubits: int,
    run_dir:          str,
):
    """Save per-circuit metrics for all methods as JSON for downstream use."""
    baselines = [m for m in METHOD_NAMES if m != "MOSAIC"]
    n = len(all_metrics["MOSAIC"])
    win_rates = {
        bl: {
            "wins":  sum(1 for i in range(n)
                         if all_metrics["MOSAIC"][i]["hard_cost"]
                            < all_metrics[bl][i]["hard_cost"]),
            "total": n,
            "win_pct": round(
                100.0 * sum(1 for i in range(n)
                            if all_metrics["MOSAIC"][i]["hard_cost"]
                               < all_metrics[bl][i]["hard_cost"]) / max(n, 1), 2),
        }
        for bl in baselines
    }

    summary = {
        "run_dir":          run_dir,
        "n_circuits":       n_circuits,
        "number_of_qubits": number_of_qubits,
        "tech_names":       tech_names,
        "win_rates":        win_rates,
        "methods":          {},
    }
    for method in METHOD_NAMES:
        mlist = all_metrics[method]
        summary["methods"][method] = {
            "per_circuit": [
                {k: float(v) for k, v in m.items()
                 if k not in ("T", "N")}
                for m in mlist
            ],
            "means": {
                key: float(np.mean([_scale(key, m[key]) for m in mlist]))
                for key, *_ in METRICS_CFG
            },
            "stds": {
                key: float(np.std([_scale(key, m[key]) for m in mlist]))
                for key, *_ in METRICS_CFG
            },
        }
    out_path = os.path.join(save_dir, f"comparison_results_N{number_of_qubits}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"  Results saved: {out_path}")


def save_summary_txt(
    all_metrics:      Dict[str, List[dict]],
    save_dir:         str,
    tech_names:       List[str],
    K:                int,
    n_circuits:       int,
    number_of_qubits: int,
):
    """Save summary.txt mirroring the console comparison table."""
    lines = ["BASELINE COMPARISON TABLE", "=" * 72, ""]
    lines += _format_comparison_table_lines(
        all_metrics, tech_names, K, n_circuits, number_of_qubits)
    out_path = os.path.join(save_dir, f"summary_N{number_of_qubits}.txt")
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
        save_dir = os.path.join(args.run_dir, f"eval_syn_{args.checkpoint}")
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    log_section("MOSAIC BASELINE COMPARISON EVALUATION v1")
    log(f"Run dir     : {args.run_dir}")
    log(f"Checkpoint  : {args.checkpoint}")
    log(f"N circuits  : {args.n_circuits}")
    log(f"Eval seed   : {args.seed}")
    log(f"Save dir    : {save_dir}")

    # ---- Load model + cost config ----
    log_section("LOADING RUN ARTIFACTS")
    art = load_run_artifacts(args.run_dir, args.checkpoint, device=device)

    evol_model      = art["evol_model"]
    cluster_module  = art["cluster_module"]
    cost_module     = art["cost_module"]
    config          = art["config"]
    circuit_src_cfg = art["circuit_source_cfg"]
    dataset_cfg     = art["dataset_cfg"]
    K               = art["K"]
    tech_names      = art["tech_names"]
    caps            = art["caps"]
    w_short         = art["w_short"]
    w_long          = art["w_long"]

    # ---- Resolve num_qubits (CLI override or from run config) ----
    default_nq = circuit_src_cfg.get("kwargs", {}).get("num_qubits", None)
    if args.num_qubits is not None:
        number_of_qubits = args.num_qubits
        circuit_src_cfg.setdefault("kwargs", {})["num_qubits"] = number_of_qubits
        log(f"  num_qubits  : {number_of_qubits} (CLI override; default was {default_nq})")
    else:
        number_of_qubits = default_nq
        log(f"  num_qubits  : {number_of_qubits} (from run config)")

    # ---- Build eval circuit provider ----
    log_section("GENERATING EVALUATION CIRCUITS")
    provider = build_provider(circuit_src_cfg, seed_base=args.seed)
    log(f"Provider built: source={circuit_src_cfg['name']}, seed_base={args.seed}")
    if "sampled_kwargs" in circuit_src_cfg and circuit_src_cfg["sampled_kwargs"]:
        mix = circuit_src_cfg["sampled_kwargs"].get("option_mix", {})
        log(f"  Option mix: {mix}")

    # ---- Per-circuit loop ----
    log_section("RUNNING MOSAIC + BASELINES")
    all_metrics: Dict[str, List[dict]] = {m: [] for m in METHOD_NAMES}
    t0 = time.time()

    for i in range(args.n_circuits):
        t_circ = time.time()
        qc  = provider.get(i)
        rep, segments, layer_data_list = preprocess_circuit(qc, dataset_cfg, w_short, w_long)
        T   = len(layer_data_list)
        N   = rep.num_qubits

        # --- MOSAIC ---
        P_seq            = run_inference(evol_model, cluster_module, layer_data_list)
        mosaic_hard      = enforce_capacity_sequence(P_seq, caps)
        mosaic_metrics   = compute_metrics_v1(
            mosaic_hard, rep, segments, cost_module, caps, K, config, device)
        all_metrics["MOSAIC"].append(mosaic_metrics)

        # --- B1, B3 ---
        b1_hard    = baseline_b1(rep, caps, config, K)
        b3_hard    = baseline_b3(rep, caps, config, K)

        b1_metrics = compute_metrics_v1(
            b1_hard, rep, segments, cost_module, caps, K, config, device)
        b3_metrics = compute_metrics_v1(
            b3_hard, rep, segments, cost_module, caps, K, config, device)

        all_metrics["B1"].append(b1_metrics)
        all_metrics["B3"].append(b3_metrics)

        # --- B4: Wu beam search (seed=i for per-circuit reproducibility) ---
        b4_hard    = baseline_b4(rep, caps, config, K, seed=i)
        b4_metrics = compute_metrics_v1(
            b4_hard, rep, segments, cost_module, caps, K, config, device)
        all_metrics["B4"].append(b4_metrics)

        # --- B5: Burt-style FM + gate-grouping ---
        b5_hard    = baseline_b5(rep, caps, config, K)
        b5_metrics = compute_metrics_v1(
            b5_hard, rep, segments, cost_module, caps, K, config, device)
        all_metrics["B5"].append(b5_metrics)

        elapsed = time.time() - t_circ
        log(
            f"  [{i+1:3d}/{args.n_circuits}] N={N:2d}, T={T:3d} | "
            f"MOSAIC={mosaic_metrics['hard_cost']:.3f}  "
            f"B1={b1_metrics['hard_cost']:.3f}  "
            f"B3={b3_metrics['hard_cost']:.3f}  "
            f"B4={b4_metrics['hard_cost']:.3f}  "
            f"B5={b5_metrics['hard_cost']:.3f}  "
            f"({elapsed:.1f}s)"
        )

    total_time = time.time() - t0
    log(f"\nDone: {args.n_circuits} circuits in {total_time:.1f}s "
        f"({total_time / args.n_circuits:.2f}s/circuit)")

    # ---- Summary ----
    print_comparison_table(all_metrics, tech_names, K, args.n_circuits, number_of_qubits)

    log_section("GENERATING COMPARISON FIGURE")
    plot_comparison_figure(all_metrics, save_dir, number_of_qubits, show=args.show)

    log_section("SAVING RESULTS")
    save_results_json(all_metrics, save_dir, args.n_circuits, tech_names,
                      number_of_qubits, args.run_dir)
    save_summary_txt(all_metrics, save_dir, tech_names, K, args.n_circuits,
                     number_of_qubits)

    log_section("EVALUATION COMPLETE")
    log(f"All outputs saved to: {save_dir}")


if __name__ == "__main__":
    main()
