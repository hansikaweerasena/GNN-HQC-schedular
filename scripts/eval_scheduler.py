"""
eval_scheduler.py  —  MOSAIC Scheduler Evaluation Script

Loads a trained MOSAIC model from a HiPerGator run directory and evaluates
it on freshly-generated circuits using the exact same preprocessing pipeline
as training (CircuitRepresentation → segment_circuit → build_layer_graph_arrays).

Usage:
    python eval_scheduler.py \\
        --run_dir  results/20250101_120000_run_v1 \\
        --checkpoint best \\          # final | best | last | epoch_NNN
        --n_circuits 50 \\
        --seed 99999 \\
        --n_visual 3 \\
        --save_dir eval_out \\
        --show
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
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import torch
from torch_geometric.data import Data

# ---- make project root importable (same pattern as train_hipergator.py) ----
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


# =============================================================================
# Logging helpers (mirrors train_hipergator style)
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
    p = argparse.ArgumentParser(description="MOSAIC Scheduler Evaluation")
    p.add_argument("--run_dir",    type=str, required=True,
                   help="Path to the HiPerGator run directory (contains model_arch_params.json etc.)")
    p.add_argument("--checkpoint", type=str, default="best",
                   help="Which weights to load: 'final' | 'best' | 'last' | 'epoch_NNN'")
    p.add_argument("--n_circuits", type=int, default=50,
                   help="Number of circuits to evaluate")
    p.add_argument("--seed",       type=int, default=99999,
                   help="Seed base for eval circuit generation (use something far from train/test seeds)")
    p.add_argument("--n_visual",   type=int, default=3,
                   help="Number of circuits to show full 4-panel visual inspection")
    p.add_argument("--save_dir",   type=str, default=None,
                   help="Directory to save figures. Defaults to <run_dir>/eval_<timestamp>/")
    p.add_argument("--show",       action="store_true",
                   help="Show plots interactively (requires a display / GUI backend)")
    return p.parse_args()


# =============================================================================
# Load run artifacts from directory
# =============================================================================

def _load_snapshot_cfg(snapshot_path: str) -> dict:
    """
    Execute scheduler_config_snapshot.py and return its module-level names.
    The snapshot file is pure-Python dicts with no imports, so exec is safe.
    """
    ns: dict = {}
    with open(snapshot_path, "r") as f:
        exec(f.read(), ns)  # noqa: S102
    return ns


def load_run_artifacts(run_dir: str, checkpoint: str, device: str = "cpu"):
    """
    Load every artifact saved by train_hipergator.py and reconstruct:
      - EvolvingGNN + SegmentClustering (weights from chosen checkpoint)
      - TotalCost, CapacityPenalty
      - CIRCUIT_SOURCE_CFG, DATASET_CFG (from scheduler snapshot)
      - tech_names, caps, K, w_short, w_long

    Checkpoint options:
      "final"     -> evol_model.pt + cluster_head.pt (plain state dicts)
      "best"      -> checkpoint_best.pt (dict with evol_model / cluster_head keys)
      "last"      -> checkpoint_last.pt
      "epoch_NNN" -> checkpoint_epoch_NNN.pt
    """
    log(f"Loading run artifacts from: {run_dir}")

    # --- arch params ---
    arch_path = os.path.join(run_dir, "model_arch_params.json")
    with open(arch_path) as f:
        arch = json.load(f)
    gnn_arch = arch["EvolvingGNN"]
    cls_arch = arch["SegmentClustering"]
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

    # --- scheduler config (CIRCUIT_SOURCE_CFG, DATASET_CFG, etc.) ---
    snap_path = os.path.join(run_dir, "scheduler_config_snapshot.py")
    snap = _load_snapshot_cfg(snap_path)
    circuit_source_cfg = snap["CIRCUIT_SOURCE_CFG"]
    dataset_cfg = snap["DATASET_CFG"]
    log(f"  circuit source: {circuit_source_cfg['name']}, "
        f"seg_mode={dataset_cfg['segmentation_mode']}")

    # --- rebuild models (architecture only) ---
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
        evol_path    = os.path.join(run_dir, "evol_model.pt")
        cluster_path = os.path.join(run_dir, "cluster_head.pt")
        evol_model.load_state_dict(torch.load(evol_path, map_location=device))
        cluster_module.load_state_dict(torch.load(cluster_path, map_location=device))
        epoch = "unknown"   # plain state dicts don't carry epoch metadata
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
        epoch = ckpt_dict.get("epoch", "?")
        test_loss = ckpt_dict.get("test_loss", float("nan"))
        log(f"  weights: {os.path.basename(ckpt_file)} (epoch={epoch}, test_loss={test_loss:.4f})")

    evol_model.eval()
    cluster_module.eval()

    # --- cost modules ---
    total_cost_module  = TotalCost(config).to(device)
    cap_penalty_module = CapacityPenalty(total_cost_module, config).to(device)

    # Set tau to its annealed floor (tau_min) rather than replaying set_epoch().
    # TotalCost initialises with tau=tau0 (e.g. 400.0). By eval time the model
    # trained with fully-decayed tau=tau_min (e.g. 25.0). Driving it to the
    # floor is the correct eval-time state regardless of which checkpoint is used.
    tau_min = total_cost_module._tau_min
    total_cost_module.tau.fill_(tau_min)
    log(f"  TotalCost + CapacityPenalty built "
        f"(lambda_cap={cap_penalty_module.lambda_cap.item():.2f}, tau={tau_min:.4f})")

    return {
        "evol_model":      evol_model,
        "cluster_module":  cluster_module,
        "cost_module":     total_cost_module,
        "cap_penalty":     cap_penalty_module,
        "config":          config,
        "circuit_source_cfg": circuit_source_cfg,
        "dataset_cfg":     dataset_cfg,
        "K":               K,
        "tech_names":      tech_names,
        "caps":            caps,
        "w_short":         w_short,
        "w_long":          w_long,
        "device":          device,
    }


# =============================================================================
# Circuit preprocessing — exact mirror of train_hipergator.py pipeline
# =============================================================================

def _build_layer_data_list(rep: CircuitRepresentation, w_short: int, w_long: int) -> List[Data]:
    """CPU Data construction — matches CircuitDataset.__getitem__ exactly."""
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
    """CircuitRepresentation → segment_circuit → layer_data_list."""
    rep = CircuitRepresentation(qc)
    seg_mode = dataset_cfg["segmentation_mode"]
    seg_thr  = float(dataset_cfg["segment_threshold"])
    segments, _ = segment_circuit(rep.layers, mode=seg_mode, threshold=seg_thr)
    layer_data_list = _build_layer_data_list(rep, w_short, w_long)
    return rep, segments, layer_data_list


# =============================================================================
# Inference
# =============================================================================

def run_inference(
    evol_model: EvolvingGNN,
    cluster_module: SegmentClustering,
    layer_data_list: List[Data],
) -> List[torch.Tensor]:
    """Return P_seq: List[Tensor[N, K]] — one soft assignment per segment."""
    with torch.no_grad():
        h_seq, _ = evol_model(layer_data_list)
        P_seq    = cluster_module(h_seq, graphs=layer_data_list)
    return P_seq


# =============================================================================
# Per-circuit metrics
# =============================================================================

def compute_circuit_metrics(
    P_seq:            List[torch.Tensor],   # List[N, K]
    hard_assignments: List[torch.Tensor],   # List[N]
    segments,
    rep:              CircuitRepresentation,
    cost_module:      TotalCost,
    cap_penalty_module: CapacityPenalty,
    caps:             torch.Tensor,         # [K]
    K:                int,
    device:           str,
) -> dict:
    """Compute all per-circuit evaluation metrics. Returns a flat dict."""
    T = len(P_seq)
    N = rep.num_qubits

    # --- build one-hot P_seq_hard ---
    P_seq_hard = []
    for t in range(T):
        P_t = torch.zeros(N, K, dtype=torch.float32, device=device)
        P_t[torch.arange(N, device=device), hard_assignments[t].to(device)] = 1.0
        P_seq_hard.append(P_t)

    # --- costs ---
    with torch.no_grad():
        soft_out  = cost_module(P_seq,      segments, rep)
        hard_out  = cost_module(P_seq_hard, segments, rep)
        cap_out   = cap_penalty_module(P_seq)

    soft_cost      = soft_out["total_cost"].item()
    hard_cost      = hard_out["total_cost"].item()
    cap_penalty_val = cap_out["penalty"].item()
    soft_cost_with_cap = soft_cost + cap_penalty_val

    hardening_gap = (hard_cost - soft_cost) / max(abs(soft_cost), 1e-9)

    # --- hardening burden: qubits moved per layer ---
    soft_argmax_list = [P_t.argmax(dim=1).cpu() for P_t in P_seq]
    burden_per_layer = [
        (soft_argmax_list[t] != hard_assignments[t].cpu()).float().sum().item()
        for t in range(T)
    ]
    mean_burden = float(np.mean(burden_per_layer)) if burden_per_layer else 0.0
    max_burden  = float(np.max(burden_per_layer))  if burden_per_layer else 0.0

    # --- capacity overflow (expected occupancy — NOT argmax) ---
    # Using expected occupancy n_k = Σ_q P[q,k] rather than argmax counts,
    # because a blurry [0.51, 0.49] assignment and a sharp [0.99, 0.01] one
    # are identical after argmax but have very different soft capacity pressure.
    # This matches the quantity CapacityPenalty uses internally.
    overflows = []
    violating_layers = 0
    for t, P_t in enumerate(P_seq):
        expected_counts = P_t.detach().cpu().sum(dim=0)   # [K] expected qubits per tech
        overflow = torch.relu(expected_counts - caps.cpu()).sum().item()
        overflows.append(overflow)
        if overflow > 0:
            violating_layers += 1

    mean_overflow = float(np.mean(overflows)) if overflows else 0.0
    max_overflow  = float(np.max(overflows))  if overflows else 0.0
    pct_violating = violating_layers / T * 100.0

    # --- remote 2Q cut rate (hard assignments) ---
    total_2q = 0
    cut_2q   = 0
    # Segments correspond 1-to-1 with rep.layers in mode="layer"
    for t in range(min(T, len(rep.layers))):
        ha_t = hard_assignments[t].cpu()
        for gate_name, qargs in rep.layers[t].gates:
            if len(qargs) == 2:
                u, v = qargs
                total_2q += 1
                if ha_t[u].item() != ha_t[v].item():
                    cut_2q += 1
    remote_2q_cut_rate = cut_2q / max(total_2q, 1)

    # --- temporal movement (hard: qubit changes tech between adjacent segments) ---
    movements = []
    for t in range(T - 1):
        moved = (hard_assignments[t].cpu() != hard_assignments[t + 1].cpu()).float().sum().item()
        movements.append(moved)
    mean_movement = float(np.mean(movements)) if movements else 0.0
    max_movement  = float(np.max(movements))  if movements else 0.0

    # --- assignment sharpness: mean entropy of P[qubit, :] per layer ---
    entropies = []
    max_probs = []
    for P_t in P_seq:
        p = P_t.detach().cpu().clamp(min=1e-9)
        ent = -(p * p.log()).sum(dim=1).mean().item()   # mean over qubits
        entropies.append(ent)
        max_probs.append(p.max(dim=1).values.mean().item())
    mean_entropy   = float(np.mean(entropies))
    mean_max_prob  = float(np.mean(max_probs))

    # --- per-tech occupancy per layer (hard) ---
    occupancy = np.zeros((T, K))
    for t, ha in enumerate(hard_assignments):
        for k in range(K):
            occupancy[t, k] = (ha.cpu() == k).sum().item()

    return {
        "soft_cost":           soft_cost,
        "soft_cost_with_cap":  soft_cost_with_cap,
        "hard_cost":           hard_cost,
        "cap_penalty":         cap_penalty_val,
        "hardening_gap":       hardening_gap,
        "mean_burden":         mean_burden,
        "max_burden":          max_burden,
        "burden_per_layer":    burden_per_layer,
        "mean_overflow":       mean_overflow,
        "max_overflow":        max_overflow,
        "pct_violating":       pct_violating,
        "remote_2q_cut_rate":  remote_2q_cut_rate,
        "total_2q_gates":      total_2q,
        "cut_2q_gates":        cut_2q,
        "mean_movement":       mean_movement,
        "max_movement":        max_movement,
        "movements":           movements,
        "mean_entropy":        mean_entropy,
        "mean_max_prob":       mean_max_prob,
        "occupancy":           occupancy,  # [T, K]
        "T":                   T,
        "N":                   N,
    }


# =============================================================================
# Visual inspection panel (4-panel per circuit)
# =============================================================================

def plot_circuit_panel(
    circuit_idx:      int,
    rep:              CircuitRepresentation,
    P_seq:            List[torch.Tensor],
    hard_assignments: List[torch.Tensor],
    tech_names:       List[str],
    K:                int,
    caps:             torch.Tensor,
    save_path:        Optional[str] = None,
    show:             bool = False,
):
    T = len(P_seq)
    N = rep.num_qubits

    # ---- Build numpy arrays ----
    # Activity: [T, N] binary
    activity = np.zeros((T, N))
    for t in range(min(T, len(rep.layers))):
        for _, qargs in rep.layers[t].gates:
            for q in qargs:
                if q < N:
                    activity[t, q] = 1.0

    # Soft argmax and confidence: [T, N]
    soft_argmax_np = np.zeros((T, N), dtype=int)
    soft_confidence = np.zeros((T, N))
    for t, P_t in enumerate(P_seq):
        probs = P_t.detach().cpu().numpy()     # [N, K]
        soft_argmax_np[t] = probs.argmax(axis=1)
        soft_confidence[t] = probs.max(axis=1)

    # Hard assignments: [T, N]
    hard_np = np.zeros((T, N), dtype=int)
    for t, ha in enumerate(hard_assignments):
        hard_np[t] = ha.cpu().numpy()

    # Moved mask: where hard != soft argmax
    moved_mask = hard_np != soft_argmax_np     # [T, N] bool

    # ---- Discrete tech colormap ----
    base_colors = [plt.cm.tab10.colors[k % 10] for k in range(K)]

    def build_rgba(assignment: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """assignment [T,N] int, alpha [T,N] float -> [T,N,4] RGBA."""
        rgb = np.array([c[:3] for c in base_colors])   # [K, 3]
        img = rgb[assignment]                           # [T, N, 3]
        return np.concatenate([img, alpha[:, :, np.newaxis]], axis=2)

    soft_rgba = build_rgba(soft_argmax_np, np.clip(soft_confidence, 0.2, 1.0))
    hard_rgba = build_rgba(hard_np, np.ones((T, N)))

    # Overlay moved qubits in bright white
    moved_overlay = np.zeros((T, N, 4))
    moved_overlay[moved_mask] = [1.0, 1.0, 1.0, 0.65]

    # ---- Figure ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        f"Circuit {circuit_idx}  |  {N} qubits  |  {T} segments",
        fontsize=12, fontweight="bold",
    )

    legend_patches = [
        mpatches.Patch(color=base_colors[k], label=tech_names[k])
        for k in range(K)
    ]

    # Panel 1: Qubit activity
    ax = axes[0, 0]
    ax.imshow(activity, aspect="auto", origin="lower", cmap="Blues",
              interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Qubit Activity (blue = active)")
    ax.set_xlabel("Qubit"); ax.set_ylabel("Layer")

    # Panel 2: Occupancy balance (per-tech qubit count over layers)
    ax = axes[0, 1]
    occupancy = np.zeros((T, K))
    for t, ha in enumerate(hard_assignments):
        for k in range(K):
            occupancy[t, k] = (ha.cpu() == k).sum().item()
    for k in range(K):
        ax.plot(occupancy[:, k], label=tech_names[k], color=base_colors[k], linewidth=1.5)
    # Draw capacity ceilings as dashed horizontal lines per tech
    for k in range(K):
        ax.axhline(
            caps[k].item(),
            color=base_colors[k], linestyle="--", linewidth=1.0, alpha=0.6,
            label=f"{tech_names[k]} cap ({int(caps[k].item())})",
        )
    ax.set_title("Occupancy Balance (hard assignments per segment)")
    ax.set_xlabel("Segment"); ax.set_ylabel("# Qubits")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Soft assignment heatmap
    ax = axes[1, 0]
    ax.imshow(soft_rgba, aspect="auto", origin="lower", interpolation="nearest")
    ax.legend(handles=legend_patches, loc="upper right", fontsize=7,
              framealpha=0.7)
    ax.set_title("Soft Assignment  (color=tech, brightness=confidence)")
    ax.set_xlabel("Qubit"); ax.set_ylabel("Segment")

    # Panel 4: Hard assignment heatmap + moved qubits highlighted
    ax = axes[1, 1]
    ax.imshow(hard_rgba, aspect="auto", origin="lower", interpolation="nearest")
    ax.imshow(moved_overlay, aspect="auto", origin="lower", interpolation="nearest")
    moved_count = int(moved_mask.sum())
    ax.legend(handles=legend_patches + [
        mpatches.Patch(color="white", label=f"Moved ({moved_count} cells)")
    ], loc="upper right", fontsize=7, framealpha=0.7)
    ax.set_title("Hard Assignment  (white = moved by hardening)")
    ax.set_xlabel("Qubit"); ax.set_ylabel("Segment")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        log(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# Summary figures (across all N circuits)
# =============================================================================

def plot_summary_figures(
    all_metrics: List[dict],
    tech_names:  List[str],
    K:           int,
    caps:        torch.Tensor,
    save_dir:    str,
    show:        bool = False,
):
    n = len(all_metrics)
    base_colors = [plt.cm.tab10.colors[k % 10] for k in range(K)]

    def _savefig(fig, name: str):
        path = os.path.join(save_dir, name)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        log(f"  Saved: {path}")

    # -------------------------------------------------------------------------
    # Figure 1: Cost trio + hardening gap distribution
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Cost Overview", fontsize=12, fontweight="bold")

    # Bar chart: mean soft / soft+cap / hard
    costs = {
        "Soft\n(real cost)":           np.mean([m["soft_cost"]          for m in all_metrics]),
        "Soft+CapPen\n(train obj.)":   np.mean([m["soft_cost_with_cap"] for m in all_metrics]),
        "Hard\n(real cost)":           np.mean([m["hard_cost"]           for m in all_metrics]),
    }
    ax = axes[0]
    bars = ax.bar(costs.keys(), costs.values(),
                  color=["steelblue", "darkorange", "firebrick"], alpha=0.85)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_title("Mean Costs Across All Circuits")
    ax.set_ylabel("Total Cost")
    ax.grid(axis="y", alpha=0.3)

    # Hardening gap histogram
    gaps = [m["hardening_gap"] * 100 for m in all_metrics]
    ax = axes[1]
    ax.hist(gaps, bins=20, color="firebrick", alpha=0.8, edgecolor="white")
    ax.axvline(np.mean(gaps), color="black", linestyle="--", linewidth=1.5,
               label=f"mean={np.mean(gaps):.1f}%")
    ax.axvline(np.median(gaps), color="gray", linestyle=":", linewidth=1.5,
               label=f"median={np.median(gaps):.1f}%")
    ax.set_title("Hardening Gap Distribution (%)")
    ax.set_xlabel("(Hard − Soft) / |Soft| × 100")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Hardening burden distribution
    burdens = [m["mean_burden"] for m in all_metrics]
    ax = axes[2]
    ax.hist(burdens, bins=20, color="darkorange", alpha=0.8, edgecolor="white")
    ax.axvline(np.mean(burdens), color="black", linestyle="--", linewidth=1.5,
               label=f"mean={np.mean(burdens):.2f}")
    ax.set_title("Hardening Burden Distribution")
    ax.set_xlabel("Mean Qubits Reassigned per Layer")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    _savefig(fig, "summary_costs.png")
    if show: plt.show()
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 2: Capacity & cut rate
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Capacity Compliance & 2Q Cut Rate", fontsize=12, fontweight="bold")

    # % layers violating capacity
    pct_viol = [m["pct_violating"] for m in all_metrics]
    ax = axes[0]
    ax.hist(pct_viol, bins=20, color="purple", alpha=0.8, edgecolor="white")
    ax.axvline(np.mean(pct_viol), color="black", linestyle="--", linewidth=1.5,
               label=f"mean={np.mean(pct_viol):.1f}%")
    ax.set_title("% Layers Violating Capacity (Soft Argmax)")
    ax.set_xlabel("% Layers")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Mean overflow
    overflows = [m["mean_overflow"] for m in all_metrics]
    ax = axes[1]
    ax.hist(overflows, bins=20, color="teal", alpha=0.8, edgecolor="white")
    ax.axvline(np.mean(overflows), color="black", linestyle="--", linewidth=1.5,
               label=f"mean={np.mean(overflows):.2f}")
    ax.set_title("Mean Capacity Overflow per Layer (Soft)")
    ax.set_xlabel("Sum of Excess Qubits")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Remote 2Q cut rate
    cut_rates = [m["remote_2q_cut_rate"] * 100 for m in all_metrics]
    ax = axes[2]
    ax.hist(cut_rates, bins=20, color="tomato", alpha=0.8, edgecolor="white")
    ax.axvline(np.mean(cut_rates), color="black", linestyle="--", linewidth=1.5,
               label=f"mean={np.mean(cut_rates):.1f}%")
    ax.set_title("Remote 2Q Cut Rate (Hard)")
    ax.set_xlabel("% of 2Q Gates with Cross-Tech Endpoints")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    _savefig(fig, "summary_capacity_cut.png")
    if show: plt.show()
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 3: Sharpness, movement, occupancy balance
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Model Behaviour", fontsize=12, fontweight="bold")

    # Sharpness vs hardening burden scatter
    entropies = [m["mean_entropy"]  for m in all_metrics]
    burdens_s = [m["mean_burden"]   for m in all_metrics]
    ax = axes[0]
    ax.scatter(entropies, burdens_s, alpha=0.6, s=20, color="steelblue", edgecolors="none")
    ax.set_title("Sharpness vs Hardening Burden")
    ax.set_xlabel("Mean Assignment Entropy (lower = sharper)")
    ax.set_ylabel("Mean Qubits Reassigned per Layer")
    ax.grid(alpha=0.3)

    # Temporal movement distribution
    movements = [m["mean_movement"] for m in all_metrics]
    ax = axes[1]
    ax.hist(movements, bins=20, color="slateblue", alpha=0.8, edgecolor="white")
    ax.axvline(np.mean(movements), color="black", linestyle="--", linewidth=1.5,
               label=f"mean={np.mean(movements):.2f}")
    ax.set_title("Temporal Movement (Hard)")
    ax.set_xlabel("Mean Qubits Changing Tech per Segment Boundary")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Mean occupancy balance across all circuits (normalized to fraction of N)
    # Build a normalized time axis 0→1 for each circuit, interpolate to 100 points
    n_pts = 100
    norm_occ = np.zeros((n_pts, K))
    counts = np.zeros(n_pts)
    for m in all_metrics:
        T_i = m["T"]
        N_i = m["N"]
        occ_i = m["occupancy"]         # [T_i, K]
        occ_frac = occ_i / max(N_i, 1) # fractional occupancy [T_i, K]
        for k in range(K):
            interp = np.interp(
                np.linspace(0, 1, n_pts),
                np.linspace(0, 1, T_i),
                occ_frac[:, k],
            )
            norm_occ[:, k] += interp
        counts += 1
    norm_occ /= np.maximum(counts[:, np.newaxis], 1)

    ax = axes[2]
    t_axis = np.linspace(0, 1, n_pts)
    for k in range(K):
        ax.plot(t_axis, norm_occ[:, k], label=tech_names[k],
                color=base_colors[k], linewidth=2)
    # Draw capacity lines
    N_mean = float(np.mean([m["N"] for m in all_metrics]))
    for k in range(K):
        cap_frac = caps[k].item() / max(N_mean, 1)
        ax.axhline(cap_frac, color=base_colors[k], linestyle=":", alpha=0.5)
    ax.set_title("Mean Occupancy Balance over Time\n(dotted = capacity)")
    ax.set_xlabel("Normalised Circuit Time (0→1)")
    ax.set_ylabel("Fraction of Total Qubits")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    _savefig(fig, "summary_behaviour.png")
    if show: plt.show()
    plt.close(fig)


# =============================================================================
# Summary table
# =============================================================================

def print_summary_table(all_metrics: List[dict], tech_names: List[str], K: int):
    n = len(all_metrics)

    def _fmt(vals, fmt=".3f"):
        return f"{np.mean(vals):{fmt}} ± {np.std(vals):{fmt}}"

    log_section("EVALUATION SUMMARY TABLE")
    log(f"  Circuits evaluated : {n}")
    log(f"  Technologies (K={K}): {', '.join(tech_names)}")
    print()
    rows = [
        ("Soft cost (no cap pen.) [real]",    [m["soft_cost"]          for m in all_metrics], ".4f"),
        ("Soft cost + cap pen. [train obj.]",  [m["soft_cost_with_cap"] for m in all_metrics], ".4f"),
        ("Hard cost [real]",                   [m["hard_cost"]           for m in all_metrics], ".4f"),
        ("Hardening gap (hard−soft)/|soft| %", [m["hardening_gap"]*100  for m in all_metrics], ".2f"),
        ("Hardening burden (mean/layer)",  [m["mean_burden"]         for m in all_metrics], ".3f"),
        ("Hardening burden (max/layer)",   [m["max_burden"]          for m in all_metrics], ".1f"),
        ("% layers violating cap (soft exp.occ.)", [m["pct_violating"]       for m in all_metrics], ".1f"),
        ("Mean soft overflow / layer (exp.occ.)",  [m["mean_overflow"]       for m in all_metrics], ".3f"),
        ("Max soft overflow / layer (exp.occ.)",   [m["max_overflow"]        for m in all_metrics], ".1f"),
        ("Remote 2Q cut rate (%)",         [m["remote_2q_cut_rate"]*100 for m in all_metrics], ".2f"),
        ("Mean temporal movement",         [m["mean_movement"]       for m in all_metrics], ".3f"),
        ("Max temporal movement",          [m["max_movement"]        for m in all_metrics], ".1f"),
        ("Assignment entropy (lower=sharp)",[m["mean_entropy"]       for m in all_metrics], ".4f"),
        ("Mean max probability",           [m["mean_max_prob"]       for m in all_metrics], ".4f"),
    ]
    col_w = 42
    val_w = 26
    print(f"  {'Metric':<{col_w}}  {'Mean ± Std':>{val_w}}  {'Min':>8}  {'Max':>8}")
    print("  " + "-" * (col_w + val_w + 22))
    for label, vals, fmt in rows:
        mean_std = f"{np.mean(vals):{fmt}} ± {np.std(vals):{fmt}}"
        print(f"  {label:<{col_w}}  {mean_std:>{val_w}}  "
              f"{np.min(vals):>8{fmt}}  {np.max(vals):>8{fmt}}")
    print()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    device = "cpu"

    # ---- Output directory ----
    if args.save_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(args.run_dir, f"eval_{stamp}_{args.checkpoint}")
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    log_section("MOSAIC SCHEDULER EVALUATION")
    log(f"Run dir     : {args.run_dir}")
    log(f"Checkpoint  : {args.checkpoint}")
    log(f"N circuits  : {args.n_circuits}")
    log(f"Eval seed   : {args.seed}")
    log(f"N visual    : {args.n_visual}")
    log(f"Save dir    : {save_dir}")

    # ---- Load everything from run directory ----
    log_section("LOADING RUN ARTIFACTS")
    art = load_run_artifacts(args.run_dir, args.checkpoint, device=device)

    evol_model      = art["evol_model"]
    cluster_module  = art["cluster_module"]
    cost_module     = art["cost_module"]
    cap_penalty_mod = art["cap_penalty"]
    config          = art["config"]
    circuit_src_cfg = art["circuit_source_cfg"]
    dataset_cfg     = art["dataset_cfg"]
    K               = art["K"]
    tech_names      = art["tech_names"]
    caps            = art["caps"]
    w_short         = art["w_short"]
    w_long          = art["w_long"]

    # ---- Build eval provider (same mechanism as training) ----
    # Use the exact same circuit_source_cfg from the snapshot (including option_mix)
    # but with a fresh seed base far from train (42) and test (10000) seeds.
    log_section("GENERATING EVALUATION CIRCUITS")
    provider = build_provider(circuit_src_cfg, seed_base=args.seed)
    log(f"Provider built: source={circuit_src_cfg['name']}, seed_base={args.seed}")
    if "sampled_kwargs" in circuit_src_cfg and circuit_src_cfg["sampled_kwargs"]:
        mix = circuit_src_cfg["sampled_kwargs"].get("option_mix", {})
        log(f"  Option mix: {mix}")

    # ---- Preprocess + infer + metrics loop ----
    log_section("PREPROCESSING & INFERENCE")
    all_metrics: List[dict] = []
    t0 = time.time()

    for i in range(args.n_circuits):
        t_circ = time.time()
        qc  = provider.get(i)
        rep, segments, layer_data_list = preprocess_circuit(qc, dataset_cfg, w_short, w_long)
        T   = len(layer_data_list)
        N   = rep.num_qubits

        # Inference
        P_seq = run_inference(evol_model, cluster_module, layer_data_list)

        # Hard assignments (capacity-feasible)
        hard_assignments = enforce_capacity_sequence(P_seq, caps)

        # Metrics
        metrics = compute_circuit_metrics(
            P_seq, hard_assignments, segments, rep,
            cost_module, cap_penalty_mod, caps, K, device,
        )
        all_metrics.append(metrics)

        elapsed = time.time() - t_circ
        log(f"  [{i+1:3d}/{args.n_circuits}] N={N}, T={T} | "
            f"soft={metrics['soft_cost']:.3f}, hard={metrics['hard_cost']:.3f}, "
            f"gap={metrics['hardening_gap']*100:+.1f}%, "
            f"burden={metrics['mean_burden']:.2f}, "
            f"cut={metrics['remote_2q_cut_rate']*100:.1f}% "
            f"({elapsed:.1f}s)")

        # ---- Visual panel for first n_visual circuits ----
        if i < args.n_visual:
            save_path = os.path.join(save_dir, f"circuit_{i:03d}_panel.png")
            plot_circuit_panel(
                circuit_idx=i,
                rep=rep,
                P_seq=P_seq,
                hard_assignments=hard_assignments,
                tech_names=tech_names,
                K=K,
                caps=caps,
                save_path=save_path,
                show=args.show,
            )

    total_time = time.time() - t0
    log(f"\nInference done: {args.n_circuits} circuits in {total_time:.1f}s "
        f"({total_time / args.n_circuits:.2f}s / circuit)")

    # ---- Summary table ----
    print_summary_table(all_metrics, tech_names, K)

    # ---- Summary figures ----
    log_section("GENERATING SUMMARY FIGURES")
    plot_summary_figures(all_metrics, tech_names, K, caps, save_dir, show=args.show)

    log_section("EVALUATION COMPLETE")
    log(f"All outputs saved to: {save_dir}")


if __name__ == "__main__":
    main()
