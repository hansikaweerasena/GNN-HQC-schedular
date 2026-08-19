#!/usr/bin/env python3
"""
measure_inference_time.py — Measure MOSAIC scheduling latency.

Loads a trained model, generates circuits, and times:
  1. Preprocessing (circuit → graph representation)
  2. Model inference (GNN + clustering head → soft assignments)
  3. Hardening (soft → hard assignments via capacity enforcement)
  4. Total (preprocessing + inference + hardening)

Reports per-circuit mean, std, min, max for each stage.

Usage:
    python measure_inference_time.py \
        --run_dir results/20260324_135332_v1_batched_hip \
        --checkpoint best \
        --n_circuits 300 \
        --seed 99999
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.circuit_sources import build_provider
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


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def log_section(title: str):
    print(flush=True)
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72, flush=True)


def _load_snapshot_cfg(path: str) -> dict:
    spec = importlib.util.spec_from_file_location("snap_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {
        "CIRCUIT_SOURCE_CFG": getattr(mod, "CIRCUIT_SOURCE_CFG"),
        "DATASET_CFG": getattr(mod, "DATASET_CFG"),
    }


def load_run_artifacts(run_dir: str, checkpoint: str, device: str = "cpu") -> dict:
    log(f"Loading run artifacts from: {run_dir}")

    arch_path = os.path.join(run_dir, "model_arch_params.json")
    with open(arch_path) as f:
        arch = json.load(f)
    gnn_arch = arch["EvolvingGNN"]
    cls_arch = arch["SegmentClustering"]

    cost_cfg_path = os.path.join(run_dir, "cost_config_snapshot.json")
    config = load_cost_config(cost_cfg_path)
    K = len(config["techs"])
    tech_names = [t.get("name", f"tech{k}") for k, t in enumerate(config["techs"])]
    caps = torch.tensor(
        [float(t["capacity"]["max_qubits"]) for t in config["techs"]],
        dtype=torch.float32,
    )
    w_short, w_long = compute_window_sizes_from_config(config)

    snap_path = os.path.join(run_dir, "scheduler_config_snapshot.py")
    snap = _load_snapshot_cfg(snap_path)
    circuit_source_cfg = snap["CIRCUIT_SOURCE_CFG"]
    dataset_cfg = snap["DATASET_CFG"]

    evol_model = EvolvingGNN(
        node_feat_dim=gnn_arch["node_feat_dim"],
        edge_feat_dim=gnn_arch["edge_feat_dim"],
        mlp_hidden_dim=gnn_arch["mlp_hidden_dim"],
        mlp_out_dim=gnn_arch["mlp_out_dim"],
        gnn_out_dim=gnn_arch["gnn_out_dim"],
        gru_hidden_dim=gnn_arch["gru_hidden_dim"],
        heads=gnn_arch["heads"],
        dropout=gnn_arch["dropout"],
        bptt_steps=gnn_arch["bptt_steps"],
        activation=gnn_arch.get("activation", "relu"),
    ).to(device)

    cluster_module = SegmentClustering(
        hidden_dim=cls_arch["hidden_dim"],
        num_clusters=K,
        proj_hidden_dim=cls_arch.get("proj_hidden_dim"),
        temperature_init=cls_arch["temperature_init"],
        temperature_min=cls_arch["temperature_min"],
        temperature_gamma=cls_arch["temperature_gamma"],
        neighbor_alpha_init=cls_arch.get("neighbor_alpha_learned", 0.1),
    ).to(device)

    ckpt_lower = checkpoint.lower()
    if ckpt_lower == "final":
        evol_model.load_state_dict(
            torch.load(os.path.join(run_dir, "evol_model.pt"), map_location=device))
        cluster_module.load_state_dict(
            torch.load(os.path.join(run_dir, "cluster_head.pt"), map_location=device))
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

    evol_model.eval()
    cluster_module.eval()

    return {
        "evol_model": evol_model,
        "cluster_module": cluster_module,
        "config": config,
        "circuit_source_cfg": circuit_source_cfg,
        "dataset_cfg": dataset_cfg,
        "K": K,
        "tech_names": tech_names,
        "caps": caps,
        "w_short": w_short,
        "w_long": w_long,
        "device": device,
    }


def _build_layer_data_list(rep, w_short, w_long):
    arrays = build_layer_graph_arrays(rep, w_short, w_long)
    return [
        Data(
            x=torch.tensor(x_np, dtype=torch.float32),
            edge_index=torch.tensor(ei_np, dtype=torch.long),
            edge_attr=torch.tensor(ea_np, dtype=torch.float32),
        )
        for x_np, ei_np, ea_np in arrays
    ]


def main():
    p = argparse.ArgumentParser(description="Measure MOSAIC inference latency")
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default="best")
    p.add_argument("--n_circuits", type=int, default=300)
    p.add_argument("--seed", type=int, default=99999)
    p.add_argument("--device", type=str, default="cpu",
                   help="Device for inference (cpu recommended for latency measurement)")
    p.add_argument("--warmup", type=int, default=10,
                   help="Number of warmup circuits (not included in timing)")
    p.add_argument("--num_qubits", type=int, default=None,
                   help="Override num_qubits from config")
    args = p.parse_args()

    # ── Load model ──
    log_section("LOADING MODEL")
    art = load_run_artifacts(args.run_dir, args.checkpoint, args.device)
    evol_model = art["evol_model"]
    cluster_module = art["cluster_module"]
    caps = art["caps"]
    w_short = art["w_short"]
    w_long = art["w_long"]
    dataset_cfg = art["dataset_cfg"]
    circuit_source_cfg = art["circuit_source_cfg"]
    K = art["K"]
    tech_names = art["tech_names"]
    device = art["device"]

    log(f"  Device: {device}")
    log(f"  K={K}, techs={tech_names}, caps={caps.tolist()}")

    # ── Override num_qubits if requested ──
    if args.num_qubits is not None:
        circuit_source_cfg["kwargs"]["num_qubits"] = args.num_qubits
        log(f"  Overriding num_qubits to {args.num_qubits}")

    num_qubits = circuit_source_cfg["kwargs"]["num_qubits"]

    # ── Generate circuits ──
    log_section("GENERATING CIRCUITS")
    total_needed = args.n_circuits + args.warmup
    provider = build_provider(circuit_source_cfg, seed_base=args.seed)
    circuits = []
    for i in range(total_needed):
        qc = provider.get(i)
        circuits.append(qc)
    log(f"  Generated {total_needed} circuits ({args.warmup} warmup + {args.n_circuits} timed)")

    # ── Warmup (JIT, caches, etc.) ──
    log_section("WARMUP")
    for i in range(args.warmup):
        qc = circuits[i]
        rep = CircuitRepresentation(qc)
        seg_mode = dataset_cfg["segmentation_mode"]
        seg_thr = float(dataset_cfg["segment_threshold"])
        segments, _ = segment_circuit(rep.layers, mode=seg_mode, threshold=seg_thr)
        layer_data_list = _build_layer_data_list(rep, w_short, w_long)
        with torch.no_grad():
            h_seq, _ = evol_model(layer_data_list)
            P_seq = cluster_module(h_seq, graphs=layer_data_list)
        hard = enforce_capacity_sequence(P_seq, caps)
    log(f"  Warmup complete ({args.warmup} circuits)")

    # ── Timed evaluation ──
    log_section("TIMED INFERENCE")
    times_preprocess = []
    times_inference = []
    times_harden = []
    times_total = []
    circuit_depths = []

    for i in range(args.warmup, total_needed):
        qc = circuits[i]

        # --- Preprocessing ---
        t0 = time.perf_counter()
        rep = CircuitRepresentation(qc)
        seg_mode = dataset_cfg["segmentation_mode"]
        seg_thr = float(dataset_cfg["segment_threshold"])
        segments, _ = segment_circuit(rep.layers, mode=seg_mode, threshold=seg_thr)
        layer_data_list = _build_layer_data_list(rep, w_short, w_long)
        t1 = time.perf_counter()

        # --- Model inference ---
        with torch.no_grad():
            h_seq, _ = evol_model(layer_data_list)
            P_seq = cluster_module(h_seq, graphs=layer_data_list)
        t2 = time.perf_counter()

        # --- Hardening ---
        hard = enforce_capacity_sequence(P_seq, caps)
        t3 = time.perf_counter()

        T = len(layer_data_list)
        N = rep.num_qubits

        times_preprocess.append((t1 - t0) * 1000)  # ms
        times_inference.append((t2 - t1) * 1000)
        times_harden.append((t3 - t2) * 1000)
        times_total.append((t3 - t0) * 1000)
        circuit_depths.append(T)

        if (i - args.warmup + 1) % 50 == 0:
            idx = i - args.warmup + 1
            log(f"  [{idx:3d}/{args.n_circuits}] N={N}, T={T}, "
                f"total={times_total[-1]:.1f}ms "
                f"(pre={times_preprocess[-1]:.1f} + "
                f"infer={times_inference[-1]:.1f} + "
                f"hard={times_harden[-1]:.1f})")

    # ── Summary ──
    log_section("TIMING SUMMARY")
    log(f"  Circuits timed : {args.n_circuits}")
    log(f"  Qubits         : {num_qubits}")
    log(f"  Device         : {device}")
    log(f"  Depth range    : {min(circuit_depths)}–{max(circuit_depths)} layers")
    log("")

    def _report(name, vals):
        arr = np.array(vals)
        log(f"  {name:20s}  {arr.mean():8.2f} ± {arr.std():6.2f} ms  "
            f"[min={arr.min():.2f}, max={arr.max():.2f}]")

    _report("Preprocessing", times_preprocess)
    _report("Model inference", times_inference)
    _report("Hardening", times_harden)
    _report("Total (end-to-end)", times_total)

    log("")
    log(f"  Mean total latency: {np.mean(times_total):.1f} ms/circuit")
    log(f"  Mean inference only: {np.mean(times_inference):.1f} ms/circuit")

    # ── Save results ──
    save_dir = os.path.join(args.run_dir, "timing_results")
    os.makedirs(save_dir, exist_ok=True)
    results = {
        "n_circuits": args.n_circuits,
        "num_qubits": num_qubits,
        "device": device,
        "checkpoint": args.checkpoint,
        "warmup": args.warmup,
        "depth_range": [int(min(circuit_depths)), int(max(circuit_depths))],
        "mean_depth": float(np.mean(circuit_depths)),
        "mean_total_ms": float(np.mean(times_total)),
        "std_total_ms": float(np.std(times_total)),
        "mean_preprocess_ms": float(np.mean(times_preprocess)),
        "mean_inference_ms": float(np.mean(times_inference)),
        "mean_harden_ms": float(np.mean(times_harden)),
        "per_circuit_total_ms": [float(t) for t in times_total],
        "per_circuit_depths": [int(d) for d in circuit_depths],
    }
    out_path = os.path.join(save_dir, f"timing_N{num_qubits}_{device}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
