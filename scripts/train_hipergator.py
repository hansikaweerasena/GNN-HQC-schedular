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
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from typing import List

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
from utils.train_utils import train_step
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
    Build layer Data objects on CPU.  Device transfer happens in the training
    loop after the DataLoader yields the batch — this is required for
    num_workers > 0, which spawns subprocesses that cannot access CUDA.
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


def transfer_layer_data_list(layer_data_list: List[Data], device: torch.device) -> List[Data]:
    """Move a list of CPU Data objects to device (non-blocking)."""
    if device.type == "cpu":
        return layer_data_list
    return [
        Data(
            x          = d.x.to(device, non_blocking=True),
            edge_index = d.edge_index.to(device, non_blocking=True),
            edge_attr  = d.edge_attr.to(device, non_blocking=True),
        )
        for d in layer_data_list
    ]


# =============================================================================
# Dataset  (always stores CPU tensors — safe for num_workers > 0)
# =============================================================================


class CircuitDataset(Dataset):
    """
    Always builds and caches tensors on CPU.
    Device transfer is the caller's responsibility (see transfer_layer_data_list).
    This is required to support num_workers > 0 in DataLoader.

    If stats_extractor is provided, segment stats (gamma scoring, gate counts,
    edge tensors) are pre-computed on CPU during __getitem__ and stored in the
    cache alongside the circuit. This eliminates the dominant per-forward-pass
    cost: NetworkX BFS, Jaccard similarity, and dense scoring loops — all pure
    Python and re-run on every circuit every epoch without pre-computation.
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
    ):
        self.provider          = provider
        self.n_samples         = int(n_samples)
        self.segment_mode      = segment_mode
        self.segment_threshold = float(segment_threshold)
        self.w_short           = w_short
        self.w_long            = w_long
        self.stats_extractor   = stats_extractor   # SegmentStatsExtractor | None
        self._cache            = {}

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]

        qc  = self.provider.get(idx)
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
        self._cache[idx] = item
        return item


def collate_fn(batch):
    return batch


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_model(model, cluster_module, cost_module, test_loader, device,
                   capacity_penalty=None):
    model.eval()
    cluster_module.eval()
    total_loss, total_cap = 0.0, 0.0

    with torch.no_grad():
        for batch in test_loader:
            for layer_data_list_cpu, segments, rep, stats_cpu in batch:
                layer_data_list = transfer_layer_data_list(layer_data_list_cpu, device)
                loss, _, cap_val = train_step(
                    model, cluster_module, cost_module,
                    layer_data_list, segments, rep,
                    training=False,
                    capacity_penalty=capacity_penalty,
                    precomp_stats=stats_cpu,
                )
                total_loss += loss.item()
                total_cap  += cap_val

    n = len(test_loader.dataset)
    return total_loss / n, total_cap / n


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
    """
    Save a flat JSON of all constructor arguments needed to rebuild both models.
    Inference workflow: load JSON -> construct models -> load_state_dict.
    No guessing what the model was built with.
    """
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
            # activation not stored as attribute — record from config side
        },
        "SegmentClustering": {
            "hidden_dim":           evol_model.rnn_hidden_dim,
            "num_clusters":         K,
            "proj_hidden_dim":      cluster_module.head.proj[0].out_features
                                    if hasattr(cluster_module.head, "proj") else None,
            "temperature_init":     cluster_module.head._temperature_init,
            "temperature_min":      cluster_module.head._temperature_min,
            "temperature_gamma":    cluster_module.head._temperature_gamma,
            # neighbor_alpha_init is not re-used after construction; record learned value
            "neighbor_alpha_learned": float(cluster_module.head.alpha.item()),
        },
    }
    path = os.path.join(run_dir, "model_arch_params.json")
    with open(path, "w") as f:
        json.dump(arch, f, indent=2)
    log(f"Saved model_arch_params.json -> {path}")


def save_config_snapshots(sched_cfg_module_path: str, cost_cfg_resolved: str, run_dir: str):
    """
    Copy both config files verbatim into the run directory.
    cost_cfg_resolved must be the absolute path already returned by load_cost_config —
    not the raw CLI arg, which may be just a filename with no directory component.
    """
    # scheduler config: resolve module path to file path
    module_rel = sched_cfg_module_path.replace(".", os.sep) + ".py"
    found = None
    for root in sys.path:
        candidate = os.path.join(root, module_rel)
        if os.path.isfile(candidate):
            found = candidate
            break
    if found:
        shutil.copy2(found, os.path.join(run_dir, "scheduler_config_snapshot.py"))
        log(f"Saved scheduler_config_snapshot.py")
    else:
        log(f"WARNING: Could not locate scheduler config file for snapshot ({module_rel})")

    if os.path.isfile(cost_cfg_resolved):
        shutil.copy2(cost_cfg_resolved, os.path.join(run_dir, "cost_config_snapshot.json"))
        log(f"Saved cost_config_snapshot.json")
    else:
        log(f"WARNING: Could not locate cost config file for snapshot ({cost_cfg_resolved})")


# =============================================================================
# Plotting
# =============================================================================


def save_plots(run_dir, train_losses, test_losses, test_epochs,
               cluster_T_history, cost_tau_history, cap_penalty_history):

    # --- training_curve.png ---
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_losses, label="Train", alpha=0.85)
    if test_losses:
        ax.plot(test_epochs, test_losses, label="Test", alpha=0.85, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Total Cost")
    ax.set_title("Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "training_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- temperature_schedule.png ---
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(cluster_T_history, label="cluster_T", alpha=0.85)
    ax.plot(cost_tau_history,  label="cost_tau",  alpha=0.85)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Temperature / Tau")
    ax.set_title("Annealing Schedule")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "temperature_schedule.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- capacity_penalty_curve.png ---
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(cap_penalty_history, label="Cap Penalty", color="tab:orange", alpha=0.85)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg Capacity Penalty")
    ax.set_title("Capacity Penalty over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
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

    # eval_every: TRAIN_CFG key, CLI checkpoint_every override
    eval_every       = int(TRAIN_CFG.get("eval_every", 10))
    checkpoint_every = args.checkpoint_every or int(TRAIN_CFG.get("checkpoint_every", 20))

    # ---- Run directory ----
    run_dir = make_run_dir(args.results_root, args.run_tag)

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- HEADER BLOCK ----
    log_section("MOSAIC TRAINING — HEADER")
    log(f"Run dir    : {run_dir}")
    log(f"Run tag    : {args.run_tag}")
    log(f"Device     : {device}")
    if torch.cuda.is_available():
        log(f"GPU        : {torch.cuda.get_device_name(0)}")
        log(f"VRAM       : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # SLURM environment (present only inside a SLURM job)
    for slurm_var in ("SLURM_JOB_ID", "SLURM_NODELIST", "SLURM_GPUS_ON_NODE"):
        val = os.environ.get(slurm_var)
        if val:
            log(f"{slurm_var}: {val}")

    if args.dry_run:
        log("*** DRY RUN MODE — reduced dataset and 2 epochs ***")

    # ---- Cost config ----
    config            = load_cost_config(args.cost_cfg)
    cost_cfg_resolved = get_cost_config_path(args.cost_cfg)  # absolute path for snapshot copy
    w_short, w_long   = compute_window_sizes_from_config(config)
    K = len(config["techs"])

    derived = {
        "device":        str(device),
        "K_num_clusters": K,
        "w_short":       w_short,
        "w_long":        w_long,
        "node_feat_dim": NODE_FEAT_DIM,
        "edge_feat_dim": EDGE_FEAT_DIM,
        "eval_every":    eval_every,
        "checkpoint_every": checkpoint_every,
    }
    print_run_config(
        MODEL_CFG=MODEL_CFG, CLUSTER_CFG=CLUSTER_CFG, TRAIN_CFG=TRAIN_CFG,
        DATASET_CFG=DATASET_CFG, CIRCUIT_SOURCE_CFG=CIRCUIT_SOURCE_CFG,
        derived=derived,
    )
    log_section("END HEADER")

    # ---- Datasets & DataLoaders ----
    train_provider = build_provider(CIRCUIT_SOURCE_CFG, seed_base=TRAIN_CFG["seed_base_train"])
    test_provider  = build_provider(CIRCUIT_SOURCE_CFG, seed_base=TRAIN_CFG["seed_base_test"])

    # Build stats_extractor before datasets so it can be passed in for pre-computation.
    # SegmentStatsExtractor only needs the cost config (tech profiles, gate names) —
    # it does NOT need the full TotalCost module and can be constructed here cheaply.
    stats_extractor = SegmentStatsExtractor(config)
    log("SegmentStatsExtractor built — will pre-compute gamma/gate stats during warm-up.")

    train_dataset = CircuitDataset(
        train_provider,
        n_samples         = TRAIN_CFG["n_samples_train"],
        segment_mode      = DATASET_CFG["segmentation_mode"],
        segment_threshold = DATASET_CFG["segment_threshold"],
        w_short=w_short, w_long=w_long,
        stats_extractor=stats_extractor,
    )
    test_dataset = CircuitDataset(
        test_provider,
        n_samples         = TRAIN_CFG["n_samples_test"],
        segment_mode      = DATASET_CFG["segmentation_mode"],
        segment_threshold = DATASET_CFG["segment_threshold"],
        w_short=w_short, w_long=w_long,
        stats_extractor=stats_extractor,
    )

    # num_workers from config — on HiPerGator set to 4 or SLURM_CPUS_PER_TASK-1
    num_workers = int(TRAIN_CFG.get("num_workers", 0))

    # pin_memory disabled: the B200's PCIe bandwidth makes the async-transfer
    # benefit negligible, and pinning opens one /dev/shm fd per tensor which
    # exhausts the OS default of 1024 open files within the first few batches
    # (7 workers × prefetch × 80 layers × 3 tensors >> 1024).
    train_loader = DataLoader(
        train_dataset, batch_size=TRAIN_CFG["batch_size"],
        shuffle=True, collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=TRAIN_CFG["batch_size"],
        shuffle=False, collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )
    log(f"Dataset    : train={len(train_dataset)}, test={len(test_dataset)}, "
        f"batch={TRAIN_CFG['batch_size']}, num_workers={num_workers}")

    # ---- Pre-warm dataset cache (fix for num_workers > 0 on Linux/HiPerGator) ----
    # DataLoader workers are created via os.fork(). Forked children inherit the
    # parent's memory pages (copy-on-write), so a cache populated HERE — in the
    # main process, before the first DataLoader iteration — is visible to all
    # workers at zero extra cost. Without this, each worker starts with an empty
    # cache and regenerates every circuit from scratch every epoch (~6 min/epoch).
    # With this, circuit generation happens exactly once; epoch 2+ costs only GNN
    # forward/backward time.
    # NOTE: only works with fork-based multiprocessing (Linux default). Safe on
    # HiPerGator. Would need rework if start method is ever changed to "spawn".
    if num_workers > 0:
        log("Pre-warming dataset cache in main process (fork-safe, runs once)...")
        t_warm = time.time()
        n_train = len(train_dataset)
        n_test  = len(test_dataset)
        for i in range(n_train):
            train_dataset[i]
            if (i + 1) % 100 == 0 or (i + 1) == n_train:
                log(f"  train {i+1}/{n_train} ...")
        for i in range(n_test):
            test_dataset[i]
            if (i + 1) % 50 == 0 or (i + 1) == n_test:
                log(f"  test  {i+1}/{n_test} ...")
        log(f"Cache warm-up done in {(time.time() - t_warm)/60:.1f} min — "
            f"all subsequent epochs will skip circuit generation entirely.")
    else:
        log("num_workers=0: cache will populate on first epoch (no pre-warm needed).")

    # ---- Build models ----
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
        hidden_dim         = evol_model.rnn_hidden_dim,
        num_clusters       = K,
        proj_hidden_dim    = CLUSTER_CFG.get("proj_hidden_dim"),
        temperature_init   = CLUSTER_CFG["temperature_init"],
        temperature_min    = CLUSTER_CFG["temperature_min"],
        temperature_gamma  = CLUSTER_CFG["temperature_gamma"],
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

    # ---- Save config snapshots and arch params ----
    save_config_snapshots(args.sched_cfg, cost_cfg_resolved, run_dir)
    # arch params saved after training (captures learned alpha)

    # ---- Metric history (for plots) ----
    train_losses       = []
    test_losses        = []
    test_epochs        = []
    cluster_T_history  = []
    cost_tau_history   = []
    cap_penalty_history = []

    best_test_loss = float("inf")
    n_epochs       = int(TRAIN_CFG["n_epochs"])

    # ---- Training Loop ----
    log_section("TRAINING")
    t_start = time.time()

    for epoch in range(n_epochs):

        # Anneal both temperatures
        total_cost_module.set_epoch(epoch)
        cluster_module.set_epoch(epoch)

        # Record temperatures AFTER annealing for this epoch
        cluster_T  = cluster_module.head.temperature.item()
        cost_tau   = total_cost_module.tau.item()
        cluster_T_history.append(cluster_T)
        cost_tau_history.append(cost_tau)

        evol_model.train()
        cluster_module.train()

        epoch_train_loss  = 0.0
        epoch_cap_penalty = 0.0

        for batch in train_loader:
            batch_loss_tensor = None
            batch_loss_float  = 0.0
            batch_cap         = 0.0
            batch_count       = 0

            optimizer.zero_grad()
            for layer_data_list_cpu, segments, rep, stats_cpu in batch:
                layer_data_list = transfer_layer_data_list(layer_data_list_cpu, device)
                loss, _, cap_val = train_step(
                    evol_model, cluster_module, total_cost_module,
                    layer_data_list, segments, rep,
                    training=True,
                    capacity_penalty=cap_penalty_module,
                    precomp_stats=stats_cpu,
                )
                batch_loss_tensor = loss if batch_loss_tensor is None \
                    else batch_loss_tensor + loss
                batch_loss_float += loss.item()
                batch_cap        += cap_val
                batch_count      += 1

            (batch_loss_tensor / batch_count).backward()
            optimizer.step()

            epoch_train_loss  += batch_loss_float / batch_count
            epoch_cap_penalty += batch_cap        / batch_count

        avg_train  = epoch_train_loss  / len(train_loader)
        avg_cap    = epoch_cap_penalty / len(train_loader)
        train_losses.append(avg_train)
        cap_penalty_history.append(avg_cap)

        # ---- Evaluate (every eval_every epochs, and always on final epoch) ----
        is_last = (epoch == n_epochs - 1)
        do_eval = (epoch % eval_every == 0) or is_last

        test_loss = float("nan")
        if do_eval:
            test_loss, test_cap = evaluate_model(
                evol_model, cluster_module, total_cost_module,
                test_loader, device, capacity_penalty=cap_penalty_module,
            )
            test_losses.append(test_loss)
            test_epochs.append(epoch)

        # ---- Per-epoch log line ----
        print(
            f"[E {epoch:03d}/{n_epochs}] "
            f"train={avg_train:.4f}  "
            f"test={test_loss:.4f}  "
            f"cap={avg_cap:.6f}  "
            f"T={cluster_T:.3f}  "
            f"tau={cost_tau:.1f}",
            flush=True,
        )

        # ---- Checkpointing ----
        # 1. checkpoint_last.pt — always overwrite
        ckpt = make_checkpoint(evol_model, cluster_module, optimizer, epoch, test_loss)
        save_checkpoint(ckpt, os.path.join(run_dir, "checkpoint_last.pt"))

        # 2. checkpoint_best.pt — only when test loss improves
        if do_eval and test_loss < best_test_loss:
            best_test_loss = test_loss
            save_checkpoint(ckpt, os.path.join(run_dir, "checkpoint_best.pt"))
            log(f"  >> New best checkpoint at epoch {epoch:03d}  test={test_loss:.4f}")

        # 3. checkpoint_epoch_NNN.pt — periodic snapshot
        if (epoch + 1) % checkpoint_every == 0:
            periodic_path = os.path.join(run_dir, f"checkpoint_epoch_{epoch:03d}.pt")
            save_checkpoint(ckpt, periodic_path)
            log(f"  >> Periodic checkpoint saved: checkpoint_epoch_{epoch:03d}.pt")

    elapsed = time.time() - t_start

    # ---- Save final model weights ----
    torch.save(evol_model.state_dict(),    os.path.join(run_dir, "evol_model.pt"))
    torch.save(cluster_module.state_dict(), os.path.join(run_dir, "cluster_head.pt"))

    # ---- Save model arch params (post-training: captures learned alpha) ----
    save_model_arch_params(evol_model, cluster_module, K, run_dir)

    # ---- Save plots ----
    save_plots(
        run_dir, train_losses, test_losses, test_epochs,
        cluster_T_history, cost_tau_history, cap_penalty_history,
    )

    # ---- Final summary block ----
    log_section("FINAL SUMMARY")
    log(f"Run dir         : {run_dir}")
    log(f"Total time      : {elapsed/60:.1f} min")
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
        fpath = os.path.join(run_dir, fname)
        status = "OK" if os.path.isfile(fpath) else "MISSING"
        log(f"  [{status}] {fname}")
    log_section("DONE")


# =============================================================================
# train_utils.py patch note
# =============================================================================
#
#  train_step() must accept a `training: bool` parameter instead of using
#  `optimizer is not None` as a mode flag.  The signature change is:
#
#    def train_step(evol_model, cluster_module, total_cost_module,
#                   layer_data_list, segments, circuit,
#                   training: bool = True,       # <-- replaces optimizer flag
#                   optimizer=None,              # kept for backward compat, unused here
#                   capacity_penalty=None):
#        if training:
#            evol_model.train(); cluster_module.train()
#        else:
#            evol_model.eval();  cluster_module.eval()
#        ...
#
#  This script calls train_step with training=True/False and never passes
#  optimizer — the optimizer.zero_grad / backward / step are managed here.
#
# =============================================================================


if __name__ == "__main__":
    main()
