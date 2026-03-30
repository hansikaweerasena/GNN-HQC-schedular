"""
train_hipergator.py

HiPerGator production training script for the MOSAIC scheduler.
Drop-in replacement for train_test_eval_debug.py — all interactive
elements removed, structured for SLURM capture.

Usage:
    python train_hipergator.py \\
        --sched_cfg configs.scheduler_config \\
        --cost_cfg  cost_config_v3.json \\
        --run_tag   baseline_v1

    # Quick smoke-test before submitting full job:
    python train_hipergator.py --dry_run --run_tag dry

Outputs (all written to results/<YYYYMMDD_HHMMSS>_<run_tag>/):
    evol_model.pt               — final EvolvingGNN state dict
    cluster_head.pt             — final SegmentClustering state dict
    model_arch_params.json      — constructor kwargs for both models (inference rebuild)
    scheduler_config_snapshot.py — copy of scheduler_config used
    cost_config_snapshot.json   — copy of cost_config used
    checkpoint_best.pt          — best test-loss checkpoint (dict with metadata)
    checkpoint_last.pt          — latest epoch (overwritten every epoch, kill-recovery)
    checkpoint_epoch_NNN.pt     — periodic snapshots (configurable)
    training_curve.png
    temperature_schedule.png
    capacity_penalty_curve.png

Dataset pipeline:
  1. Oversample-and-filter: generate oversample_factor * N candidates, compute
     effective post-layering depth T for each, retain the N circuits closest to
     target_depth. This removes Qiskit-induced outliers without any data loss on
     the kept circuits.
  2. SortedBatchSampler: sort all circuits by T and group into contiguous
     batches of size B. Each batch contains circuits of similar depth; only the
     final batch of the full dataset may be smaller, giving full data coverage
     with at most one partial batch per epoch.
  3. Masked-max batched GNN: circuits in a batch are processed in parallel up
     to Lmax = max(T_b). At each step only alive circuits are updated; ended
     circuits are frozen. ClusteringHead runs batched on the same disjoint graph.
"""

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
from src.cost_function import TotalCost, CapacityPenalty, SegmentStatsExtractor
from utils.train_utils import train_step, batch_train_step, transfer_layer_data_list
from utils.cost_config_reader import load_cost_config, get_cost_config_path, load_scheduler_cfg
from utils.print_utils import print_run_config


# =============================================================================
# Argument parsing
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser(description="MOSAIC HiPerGator training script")
    p.add_argument("--sched_cfg", type=str, default="configs.scheduler_config",
                   help="Module path for scheduler config (dotted import)")
    p.add_argument("--cost_cfg",  type=str, default="cost_config_v3.json",
                   help="Path to cost config JSON")
    p.add_argument("--run_tag",   type=str, default="run",
                   help="Human-readable label appended to the run directory name")
    p.add_argument("--results_root", type=str, default="results",
                   help="Parent directory for all run output dirs")
    p.add_argument("--checkpoint_every", type=int, default=None,
                   help="Save periodic checkpoint every N epochs "
                        "(overrides TRAIN_CFG['checkpoint_every'] if set)")
    p.add_argument("--dry_run", action="store_true",
                   help="Quick smoke-test: 2 epochs, 10 train / 5 test samples")
    return p.parse_args()


# =============================================================================
# Run directory setup
# =============================================================================


def make_run_dir(results_root: str, run_tag: str) -> str:
    """Create and return a timestamped run directory."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_root, f"{stamp}_{run_tag}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# =============================================================================
# Structured logging helpers
# =============================================================================


def log(msg: str):
    """Single-line log with timestamp — all output to stdout for SLURM capture."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def log_section(title: str):
    width = 72
    print(flush=True)
    print("=" * width, flush=True)
    print(f"  {title}", flush=True)
    print("=" * width, flush=True)


# =============================================================================
# Input construction
# =============================================================================


def build_layer_data_list_cpu(circuit, w_short: int, w_long: int) -> List[Data]:
    """
    Build layer Data objects on CPU. Device transfer happens in the training
    loop via transfer_layer_data_list (imported from train_utils).
    Building on CPU is required for num_workers > 0 (subprocesses cannot
    access CUDA).
    """
    arrays = build_layer_graph_arrays(circuit, w_short, w_long)
    return [
        Data(
            x          = torch.tensor(x_np,  dtype=torch.float32),
            edge_index = torch.tensor(ei_np, dtype=torch.long),
            edge_attr  = torch.tensor(ea_np, dtype=torch.float32),
        )
        for x_np, ei_np, ea_np in arrays
    ]


# =============================================================================
# Depth pre-pass — cheap T measurement before any caching
# =============================================================================


def compute_depths_cheap(
    provider,
    n_candidates: int,
    n_target: int,
    target_depth: int,
) -> Tuple[List[int], List[int]]:
    """
    Generate n_candidates circuits, measure effective post-layering depth T
    cheaply (CircuitRepresentation layer count only — no PyG Data construction),
    then return the n_target circuits whose T is closest to target_depth.

    This runs before CircuitDataset is constructed so that:
      - Only the n_target kept circuits ever enter the dataset cache.
      - The 0.5x discarded candidates pay only the cheap Qiskit + layer-count
        cost, not the full PyG tensor construction cost.

    Args:
        provider:      GeneratedCircuitProvider
        n_candidates:  total circuits to generate (= oversample_factor * n_target)
        n_target:      number of circuits to keep
        target_depth:  T* — nominal post-layering depth to centre selection on

    Returns:
        (selected_indices, selected_depths)
        selected_indices[i]: provider index of the i-th kept circuit
        selected_depths[i]:  effective T of the i-th kept circuit
        Both lists are ordered by increasing |T - target_depth|.
    """
    log(f"  Depth pre-pass: measuring T for {n_candidates} candidates "
        f"(target T*={target_depth}) ...")
    scored: List[Tuple[int, int, int]] = []   # (|T - T*|, idx, T)

    for idx in range(n_candidates):
        qc  = provider.get(idx)
        rep = CircuitRepresentation(qc)
        T   = len(rep.layers)
        scored.append((abs(T - target_depth), idx, T))

        if (idx + 1) % 200 == 0 or (idx + 1) == n_candidates:
            log(f"    {idx + 1}/{n_candidates} circuits measured ...")

    scored.sort(key=lambda x: x[0])
    kept = scored[:n_target]

    selected_indices = [x[1] for x in kept]
    selected_depths  = [x[2] for x in kept]

    T_arr = np.array(selected_depths)
    log(f"  Kept {n_target}/{n_candidates}: "
        f"T mean={T_arr.mean():.1f}, std={T_arr.std():.1f}, "
        f"min={T_arr.min()}, max={T_arr.max()}")

    return selected_indices, selected_depths


# =============================================================================
# Dataset  (always stores CPU tensors — safe for num_workers > 0)
# =============================================================================


class CircuitDataset(Dataset):
    """
    Always builds and caches tensors on CPU.
    Device transfer is the caller's responsibility (see transfer_layer_data_list).
    This is required to support num_workers > 0 in DataLoader.

    valid_indices: if provided, the dataset exposes exactly len(valid_indices)
    items. Item i maps to provider.get(valid_indices[i]). This is the mechanism
    for the oversample-filter pipeline: only the filtered circuits enter the
    cache — discarded candidates are never fetched again after the depth pre-pass.

    If stats_extractor is provided, segment stats are pre-computed on CPU during
    __getitem__ and stored in the cache alongside the circuit. This eliminates
    the dominant per-forward-pass cost: NetworkX BFS, Jaccard similarity, and
    dense scoring loops.
    """

    def __init__(
        self,
        provider,
        n_samples: int,
        segment_mode: str,
        segment_threshold: float,
        w_short: int,
        w_long: int,
        stats_extractor=None,
        valid_indices: Optional[List[int]] = None,
    ):
        self.provider          = provider
        self.segment_mode      = segment_mode
        self.segment_threshold = float(segment_threshold)
        self.w_short           = w_short
        self.w_long            = w_long
        self.stats_extractor   = stats_extractor

        # valid_indices maps dataset position i → provider index.
        # If None, dataset exposes indices 0..n_samples-1 directly.
        if valid_indices is not None:
            self.valid_indices = list(valid_indices)
        else:
            self.valid_indices = list(range(int(n_samples)))

        self._cache: Dict = {}

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, i):
        if i in self._cache:
            return self._cache[i]

        # Map dataset position i to provider index
        provider_idx = self.valid_indices[i]

        qc  = self.provider.get(provider_idx)
        rep = CircuitRepresentation(qc)

        segments, _ = segment_circuit(
            rep.layers,
            mode=self.segment_mode,
            threshold=self.segment_threshold,
        )

        # Always build on CPU
        layer_data_list = build_layer_data_list_cpu(rep, self.w_short, self.w_long)

        # Pre-compute segment stats on CPU (eliminates NetworkX/BFS per forward pass)
        stats_cpu = None
        if self.stats_extractor is not None:
            stats_cpu = self.stats_extractor.compute_stats_cpu(
                segments, rep, N=rep.num_qubits,
            )

        item = (layer_data_list, segments, rep, stats_cpu)
        self._cache[i] = item
        return item


def collate_fn(batch):
    return batch


# =============================================================================
# SortedBatchSampler
# =============================================================================


class SortedBatchSampler(torch.utils.data.Sampler):
    """
    Sorts all circuits by effective layer count T then groups them into
    contiguous batches of size batch_size.

    Because circuits are contiguous in the sorted order, each batch contains
    circuits of similar depth — achieving the T-variance bound of bucket
    batching without fixed-width buckets and without per-bucket leftovers.
    Only the final batch of the entire dataset may be smaller than batch_size,
    giving full data coverage with at most one partial batch per epoch.

    Each epoch the batch *order* is shuffled so the model sees circuits in a
    different sequence while within-batch T-similarity is preserved.

    Args:
        depths:     List[int] — effective T for each dataset item (0-indexed)
        batch_size: number of circuits per batch
        shuffle:    if True, shuffle batch order each epoch (default True)
                    within-batch order is always by ascending T
        seed:       RNG seed for reproducibility
    """

    def __init__(
        self,
        depths: List[int],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self._rng       = np.random.RandomState(seed)

        # Sort all indices by T (ascending) then slice into contiguous batches
        sorted_indices = sorted(range(len(depths)), key=lambda i: depths[i])

        self.batches: List[List[int]] = []
        for start in range(0, len(sorted_indices), batch_size):
            self.batches.append(sorted_indices[start:start + batch_size])

        T_arr     = np.array(depths)
        n_full    = sum(1 for b in self.batches if len(b) == batch_size)
        n_partial = len(self.batches) - n_full
        log(f"  SortedBatchSampler: {len(self.batches)} batches "
            f"({n_full} full, {n_partial} partial at tail), "
            f"T range [{T_arr.min()}, {T_arr.max()}], mean {T_arr.mean():.1f}")

    def __iter__(self):
        order = list(range(len(self.batches)))
        if self.shuffle:
            self._rng.shuffle(order)
        for i in order:
            yield self.batches[i]

    def __len__(self):
        return len(self.batches)


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_model(model, cluster_module, cost_module, test_loader, device,
                   capacity_penalty=None):
    """
    Evaluate on the test set using the same batched forward as training.
    Returns (avg_loss, avg_cap_penalty) averaged over batches.
    """
    model.eval()
    cluster_module.eval()
    total_loss, total_cap, n_batches = 0.0, 0.0, 0

    with torch.no_grad():
        for batch in test_loader:
            loss, cap = batch_train_step(
                model, cluster_module, cost_module,
                batch, device,
                capacity_penalty=capacity_penalty,
                training=False,
            )
            total_loss += loss.item()
            total_cap  += cap
            n_batches  += 1

    denom = max(n_batches, 1)
    return total_loss / denom, total_cap / denom


# =============================================================================
# Checkpoint helpers
# =============================================================================


def make_checkpoint(evol_model, cluster_module, optimizer, epoch, test_loss):
    return {
        "evol_model":   evol_model.state_dict(),
        "cluster_head": cluster_module.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "epoch":        epoch,
        "test_loss":    test_loss,
    }


def save_checkpoint(ckpt_dict, path: str):
    torch.save(ckpt_dict, path)


# =============================================================================
# Artifact helpers
# =============================================================================


def save_model_arch_params(evol_model: EvolvingGNN, cluster_module: SegmentClustering,
                            K: int, run_dir: str):
    arch = {
        "EvolvingGNN": {
            "node_feat_dim":  evol_model.mlp.fc1.in_features,
            "mlp_hidden_dim": evol_model.mlp.fc1.out_features,
            "mlp_out_dim":    evol_model.mlp.fc2.out_features,
            "gnn_out_dim":    evol_model.gnn_out_dim,
            "gru_hidden_dim": evol_model.rnn_hidden_dim,
            "edge_feat_dim":  evol_model.edge_feat_dim,
            "heads":          evol_model.gat.heads,
            "dropout":        evol_model.gat_dropout.p,
            "bptt_steps":     evol_model.bptt_steps,
        },
        "SegmentClustering": {
            "hidden_dim":             evol_model.rnn_hidden_dim,
            "num_clusters":           K,
            "proj_hidden_dim":        cluster_module.head.proj[0].out_features
                                      if hasattr(cluster_module.head, "proj") else None,
            "temperature_init":       cluster_module.head._temperature_init,
            "temperature_min":        cluster_module.head._temperature_min,
            "temperature_gamma":      cluster_module.head._temperature_gamma,
            "neighbor_alpha_learned": float(cluster_module.head.alpha.item()),
        },
    }
    path = os.path.join(run_dir, "model_arch_params.json")
    with open(path, "w") as f:
        json.dump(arch, f, indent=2)
    log(f"Saved model_arch_params.json -> {path}")


def save_config_snapshots(sched_cfg_module_path: str, cost_cfg_resolved: str, run_dir: str):
    module_rel = sched_cfg_module_path.replace(".", os.sep) + ".py"
    found = None
    for root in sys.path:
        candidate = os.path.join(root, module_rel)
        if os.path.isfile(candidate):
            found = candidate
            break
    if found:
        shutil.copy2(found, os.path.join(run_dir, "scheduler_config_snapshot.py"))
        log("Saved scheduler_config_snapshot.py")
    else:
        log(f"WARNING: Could not locate scheduler config file for snapshot ({module_rel})")

    if os.path.isfile(cost_cfg_resolved):
        shutil.copy2(cost_cfg_resolved, os.path.join(run_dir, "cost_config_snapshot.json"))
        log("Saved cost_config_snapshot.json")
    else:
        log(f"WARNING: Could not locate cost config file for snapshot ({cost_cfg_resolved})")


# =============================================================================
# Plotting
# =============================================================================


def save_plots(run_dir, train_losses, test_losses, test_epochs,
               cluster_T_history, cost_tau_history, cap_penalty_history):

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_losses, label="Train", alpha=0.85)
    if test_losses:
        ax.plot(test_epochs, test_losses, label="Test", alpha=0.85, marker="o", markersize=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Average Total Cost")
    ax.set_title("Training Curve"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "training_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(cluster_T_history, label="cluster_T", alpha=0.85)
    ax.plot(cost_tau_history,  label="cost_tau",  alpha=0.85)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Temperature / Tau")
    ax.set_title("Annealing Schedule"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "temperature_schedule.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(cap_penalty_history, label="Cap Penalty", color="tab:orange", alpha=0.85)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Avg Capacity Penalty")
    ax.set_title("Capacity Penalty over Training"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "capacity_penalty_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    log("Plots saved: training_curve.png, temperature_schedule.png, capacity_penalty_curve.png")


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    MODEL_CFG, CLUSTER_CFG, TRAIN_CFG, DATASET_CFG, CIRCUIT_SOURCE_CFG = \
        load_scheduler_cfg(args.sched_cfg)

    # --dry_run overrides: tiny dataset, 2 epochs
    if args.dry_run:
        TRAIN_CFG = dict(TRAIN_CFG)
        TRAIN_CFG["n_samples_train"] = 10
        TRAIN_CFG["n_samples_test"]  = 5
        TRAIN_CFG["n_epochs"]        = 2
        TRAIN_CFG["eval_every"]      = 1
        args.run_tag = f"{args.run_tag}_DRYRUN"

    eval_every       = int(TRAIN_CFG.get("eval_every", 10))
    checkpoint_every = args.checkpoint_every or int(TRAIN_CFG.get("checkpoint_every", 20))

    # New dataset pipeline params (with sensible defaults for backward compat)
    oversample_factor = float(TRAIN_CFG.get("oversample_factor", 1.5))
    target_depth      = int(TRAIN_CFG.get("target_depth", 80))
    batch_size        = int(TRAIN_CFG["batch_size"])

    # ---- Run directory ----
    run_dir = make_run_dir(args.results_root, args.run_tag)

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Header ----
    log_section("MOSAIC TRAINING — HEADER")
    log(f"Run dir    : {run_dir}")
    log(f"Run tag    : {args.run_tag}")
    log(f"Device     : {device}")
    if torch.cuda.is_available():
        log(f"GPU        : {torch.cuda.get_device_name(0)}")
        log(f"VRAM       : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    for slurm_var in ("SLURM_JOB_ID", "SLURM_NODELIST", "SLURM_GPUS_ON_NODE"):
        val = os.environ.get(slurm_var)
        if val:
            log(f"{slurm_var}: {val}")
    if args.dry_run:
        log("*** DRY RUN MODE — reduced dataset and 2 epochs ***")

    # ---- Cost config ----
    config            = load_cost_config(args.cost_cfg)
    cost_cfg_resolved = get_cost_config_path(args.cost_cfg)
    w_short, w_long   = compute_window_sizes_from_config(config)
    K = len(config["techs"])

    derived = {
        "device":            str(device),
        "K_num_clusters":    K,
        "w_short":           w_short,
        "w_long":            w_long,
        "node_feat_dim":     NODE_FEAT_DIM,
        "edge_feat_dim":     EDGE_FEAT_DIM,
        "eval_every":        eval_every,
        "checkpoint_every":  checkpoint_every,
        "oversample_factor": oversample_factor,
        "target_depth":      target_depth,
    }
    print_run_config(
        MODEL_CFG=MODEL_CFG, CLUSTER_CFG=CLUSTER_CFG, TRAIN_CFG=TRAIN_CFG,
        DATASET_CFG=DATASET_CFG, CIRCUIT_SOURCE_CFG=CIRCUIT_SOURCE_CFG,
        derived=derived,
    )
    log_section("END HEADER")

    # ---- Providers ----
    train_provider = build_provider(CIRCUIT_SOURCE_CFG, seed_base=TRAIN_CFG["seed_base_train"])
    test_provider  = build_provider(CIRCUIT_SOURCE_CFG, seed_base=TRAIN_CFG["seed_base_test"])

    stats_extractor = SegmentStatsExtractor(config)
    log("SegmentStatsExtractor built — will pre-compute gamma/gate stats during warm-up.")

    n_train      = int(TRAIN_CFG["n_samples_train"])
    n_test       = int(TRAIN_CFG["n_samples_test"])
    n_cand_train = int(n_train * oversample_factor)
    n_cand_test  = int(n_test  * oversample_factor)

    # ---- Stage 1: depth pre-pass (cheap — no PyG construction) ----
    # Measure T for all candidates using only CircuitRepresentation.layers count.
    # Only the n_target kept circuits will ever enter the dataset cache.
    log_section("DATASET FILTERING")

    log(f"Train: generating {n_cand_train} candidates, keeping {n_train} closest to T*={target_depth}")
    train_indices, train_depths = compute_depths_cheap(
        train_provider, n_cand_train, n_train, target_depth,
    )

    log(f"Test: generating {n_cand_test} candidates, keeping {n_test} closest to T*={target_depth}")
    test_indices, test_depths = compute_depths_cheap(
        test_provider, n_cand_test, n_test, target_depth,
    )

    # ---- Stage 2: build datasets (only kept circuits get cached) ----
    train_dataset = CircuitDataset(
        train_provider,
        n_samples         = n_cand_train,   # provider capacity (for index safety)
        segment_mode      = DATASET_CFG["segmentation_mode"],
        segment_threshold = DATASET_CFG["segment_threshold"],
        w_short=w_short, w_long=w_long,
        stats_extractor=stats_extractor,
        valid_indices=train_indices,        # only these are ever accessed
    )
    test_dataset = CircuitDataset(
        test_provider,
        n_samples         = n_cand_test,
        segment_mode      = DATASET_CFG["segmentation_mode"],
        segment_threshold = DATASET_CFG["segment_threshold"],
        w_short=w_short, w_long=w_long,
        stats_extractor=stats_extractor,
        valid_indices=test_indices,
    )
    log(f"Datasets built: train={len(train_dataset)}, test={len(test_dataset)}")

    # ---- Stage 3: sorted batch samplers ----
    # SortedBatchSampler orders all circuits by T then groups them into
    # contiguous batches of batch_size. Within each batch T-variance is
    # minimised (neighbouring circuits in the sorted order). Only the final
    # batch of the full dataset may be smaller — one partial batch at most.
    log("Building SortedBatchSamplers ...")
    log("  Train:")
    train_sampler = SortedBatchSampler(
        train_depths, batch_size, shuffle=True, seed=0,
    )
    log("  Test:")
    test_sampler = SortedBatchSampler(
        test_depths, batch_size, shuffle=False, seed=0,
    )

    num_workers = int(TRAIN_CFG.get("num_workers", 0))

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )
    log(f"DataLoaders ready: {len(train_loader)} train batches, "
        f"{len(test_loader)} test batches, num_workers={num_workers}")

    # ---- Pre-warm dataset cache ----
    # DataLoader workers are forked from the main process (Linux/HiPerGator).
    # Populating the cache here means all workers inherit it via CoW — each
    # circuit is built exactly once across all epochs.
    if num_workers > 0:
        log("Pre-warming dataset cache in main process (fork-safe, runs once) ...")
        t_warm = time.time()
        for i in range(len(train_dataset)):
            train_dataset[i]
            if (i + 1) % 100 == 0 or (i + 1) == len(train_dataset):
                log(f"  train {i + 1}/{len(train_dataset)} ...")
        for i in range(len(test_dataset)):
            test_dataset[i]
            if (i + 1) % 50 == 0 or (i + 1) == len(test_dataset):
                log(f"  test  {i + 1}/{len(test_dataset)} ...")
        log(f"Cache warm-up done in {(time.time() - t_warm) / 60:.1f} min.")
    else:
        log("num_workers=0: cache will populate inline during epoch 0 "
            "(expect first epoch to be slower; all subsequent epochs use the cache).")

    # ---- Build models ----
    log_section("MODEL CONSTRUCTION")
    evol_model = EvolvingGNN(
        node_feat_dim  = NODE_FEAT_DIM,
        edge_feat_dim  = EDGE_FEAT_DIM,
        mlp_hidden_dim = MODEL_CFG["mlp_hidden_dim"],
        mlp_out_dim    = MODEL_CFG["mlp_out_dim"],
        gnn_out_dim    = MODEL_CFG["gnn_out_dim"],
        gru_hidden_dim = MODEL_CFG["gru_hidden_dim"],
        heads          = MODEL_CFG["heads"],
        dropout        = MODEL_CFG["dropout"],
        bptt_steps     = MODEL_CFG["bptt_steps"],
        activation     = MODEL_CFG["activation"],
    ).to(device)

    cluster_module = SegmentClustering(
        hidden_dim          = evol_model.rnn_hidden_dim,
        num_clusters        = K,
        proj_hidden_dim     = CLUSTER_CFG.get("proj_hidden_dim"),
        temperature_init    = CLUSTER_CFG["temperature_init"],
        temperature_min     = CLUSTER_CFG["temperature_min"],
        temperature_gamma   = CLUSTER_CFG["temperature_gamma"],
        neighbor_alpha_init = CLUSTER_CFG["neighbor_alpha_init"],
    ).to(device)

    total_cost_module  = TotalCost(config).to(device)
    cap_penalty_module = CapacityPenalty(total_cost_module, config).to(device)

    log(f"EvolvingGNN params  : {sum(p.numel() for p in evol_model.parameters()):,}")
    log(f"ClusterHead params  : {sum(p.numel() for p in cluster_module.parameters()):,}")
    log(f"Cap penalty         : lambda={cap_penalty_module.lambda_cap.item():.6f}, "
        f"beta={cap_penalty_module.beta}, caps={cap_penalty_module.cap.tolist()}")

    optimizer = torch.optim.Adam(
        list(evol_model.parameters()) + list(cluster_module.parameters()),
        lr=TRAIN_CFG["lr"],
    )

    # ---- Save config snapshots ----
    save_config_snapshots(args.sched_cfg, cost_cfg_resolved, run_dir)

    # ---- Metric history (for plots) ----
    train_losses        = []
    test_losses         = []
    test_epochs         = []
    cluster_T_history   = []
    cost_tau_history    = []
    cap_penalty_history = []

    best_test_loss = float("inf")
    patience_count = 0
    n_epochs       = int(TRAIN_CFG["n_epochs"])

    # ---- Training Loop ----
    log_section("TRAINING")
    t_start = time.time()

    for epoch in range(n_epochs):

        # Anneal temperatures
        total_cost_module.set_epoch(epoch)
        cluster_module.set_epoch(epoch)

        cluster_T = cluster_module.head.temperature.item()
        cost_tau  = total_cost_module.tau.item()
        cluster_T_history.append(cluster_T)
        cost_tau_history.append(cost_tau)

        evol_model.train()
        cluster_module.train()

        epoch_train_loss  = 0.0
        epoch_cap_penalty = 0.0
        epoch_grad_norm   = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            # True batched forward: GNN + clustering head in parallel (masked-max),
            # then per-circuit cost accumulation.
            avg_loss, avg_cap = batch_train_step(
                evol_model, cluster_module, total_cost_module,
                batch, device,
                capacity_penalty=cap_penalty_module,
                training=True,
            )
            avg_loss.backward()

            # torch.nn.utils.clip_grad_norm_(
            #     list(evol_model.parameters()) + list(cluster_module.parameters()),
            #     max_norm=80.0
            # )

            # Grad norm (reads already-computed grads; cheap)
            _gnorm_sq = 0.0
            for _p in list(evol_model.parameters()) + list(cluster_module.parameters()):
                if _p.grad is not None:
                    _gnorm_sq += _p.grad.data.norm(2).item() ** 2
            epoch_grad_norm += _gnorm_sq ** 0.5

            optimizer.step()

            epoch_train_loss  += avg_loss.item()
            epoch_cap_penalty += avg_cap

        avg_train = epoch_train_loss  / len(train_loader)
        avg_cap   = epoch_cap_penalty / len(train_loader)
        avg_gnorm = epoch_grad_norm   / len(train_loader)
        train_losses.append(avg_train)
        cap_penalty_history.append(avg_cap)

        # ---- LR decay at temperature floor ----
        if epoch == 115:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * 0.5
            log("LR decayed: 1e-4 -> 5e-5 (both temperatures at floor)")

        # ---- Evaluate ----
        is_last = (epoch == n_epochs - 1)
        do_eval = (epoch % eval_every == 0) or is_last

        test_loss = float("nan")
        test_cap  = float("nan")
        if do_eval:
            test_loss, test_cap = evaluate_model(
                evol_model, cluster_module, total_cost_module,
                test_loader, device, capacity_penalty=cap_penalty_module,
            )
            test_losses.append(test_loss)
            test_epochs.append(epoch)

        alpha_val  = cluster_module.head.alpha.item()
        proto_norm = cluster_module.head.cluster_prototypes.norm(dim=-1).mean().item()

        print(
            f"[E {epoch:03d}/{n_epochs}] "
            f"train={avg_train:.4f}  "
            f"test={test_loss:.4f}  "
            f"cap_tr={avg_cap:.4f}  "
            f"cap_te={test_cap:.4f}  "
            f"T={cluster_T:.3f}  "
            f"tau={cost_tau:.1f}  "
            f"alpha={alpha_val:.3f}  "
            f"gnorm={avg_gnorm:.3f}  "
            f"pnorm={proto_norm:.3f}",
            flush=True,
        )

        # ---- Checkpointing ----
        ckpt = make_checkpoint(evol_model, cluster_module, optimizer, epoch, test_loss)
        save_checkpoint(ckpt, os.path.join(run_dir, "checkpoint_last.pt"))

        if do_eval:
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                patience_count = 0
                save_checkpoint(ckpt, os.path.join(run_dir, "checkpoint_best.pt"))
                log(f"  >> New best checkpoint at epoch {epoch:03d}  test={test_loss:.4f}")
            else:
                patience_count += 1
                log(f"  >> No improvement ({patience_count}/3)")
                if patience_count >= 3:
                    log("Early stopping triggered — best model already saved")
                    break

        if (epoch + 1) % checkpoint_every == 0:
            periodic_path = os.path.join(run_dir, f"checkpoint_epoch_{epoch:03d}.pt")
            save_checkpoint(ckpt, periodic_path)
            log(f"  >> Periodic checkpoint saved: checkpoint_epoch_{epoch:03d}.pt")

    elapsed = time.time() - t_start

    # ---- Save final artifacts ----
    torch.save(evol_model.state_dict(),     os.path.join(run_dir, "evol_model.pt"))
    torch.save(cluster_module.state_dict(), os.path.join(run_dir, "cluster_head.pt"))
    save_model_arch_params(evol_model, cluster_module, K, run_dir)
    save_plots(
        run_dir, train_losses, test_losses, test_epochs,
        cluster_T_history, cost_tau_history, cap_penalty_history,
    )

    # ---- Final summary ----
    log_section("FINAL SUMMARY")
    log(f"Run dir         : {run_dir}")
    log(f"Total time      : {elapsed / 60:.1f} min")
    log(f"Epochs          : {n_epochs}")
    log(f"Best test loss  : {best_test_loss:.4f}")
    log(f"Final train loss: {train_losses[-1]:.4f}")
    log(f"Final test loss : {test_losses[-1]:.4f}" if test_losses else "Final test loss : N/A")
    log(f"Post-train cluster_T  : {cluster_module.head.temperature.item():.4f}")
    log(f"Post-train cost_tau   : {total_cost_module.tau.item():.2f}")
    log(f"Post-train proto norm : "
        f"{cluster_module.head.cluster_prototypes.norm(dim=-1).mean().item():.4f}")
    log("Artifacts saved:")
    for fname in [
        "evol_model.pt", "cluster_head.pt", "model_arch_params.json",
        "scheduler_config_snapshot.py", "cost_config_snapshot.json",
        "checkpoint_best.pt", "checkpoint_last.pt",
        "training_curve.png", "temperature_schedule.png", "capacity_penalty_curve.png",
    ]:
        fpath  = os.path.join(run_dir, fname)
        status = "OK" if os.path.isfile(fpath) else "MISSING"
        log(f"  [{status}] {fname}")
    log_section("DONE")


if __name__ == "__main__":
    main()
