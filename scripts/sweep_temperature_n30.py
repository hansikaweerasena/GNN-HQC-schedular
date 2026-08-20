#!/usr/bin/env python3
"""
sweep_temperature_n30.py

Offline temperature sweep for a trained MOSAIC Sinkhorn checkpoint.

Goal
----
Hold ALL learned weights and evaluation circuits fixed, vary only the
clustering-head inference temperature T, and measure whether sharper Sinkhorn
assignments improve the final capacity-feasible hard schedule.

Default sweep:
    T = {0.50, 0.40, 0.35, 0.30, 0.25}

For every T this script records:
  - soft TotalCost
  - raw-argmax TotalCost (diagnostic only; may violate capacity)
  - capacity-feasible hard TotalCost
  - B1 hard TotalCost (fixed reference, same circuits)
  - soft->hard gap, soft->argmax gap, argmax->repair gap
  - entropy / mean max probability / top1-top2 margin
  - soft movement mass / raw argmax flips / hard movement
  - repair burden / argmax overflow / violating-layer fraction
  - soft and hard exec/idle/comm/move cost components
  - Sinkhorn row/column residual maxima across the entire sweep set

The same circuits and the same GNN hidden states are reused for every T.  Only
head temperature and the Sinkhorn projection change.

Recommended command
-------------------
Place this file beside eval_scheduler_v1.py (normally scripts/) and run:

python scripts/sweep_temperature_n30.py \
    --run_dir results/pilot30_sinkhorn2 \
    --checkpoint best \
    --n_circuits 100 \
    --seed 99999 \
    --num_qubits 30 \
    --temperatures 0.5,0.4,0.35,0.30,0.25 \
    --sinkhorn_iters 200 \
    --save_dir results/pilot30_sinkhorn2/temp_sweep_N30

Why 200 Sinkhorn iterations by default?
----------------------------------------
This is an offline diagnostic, not training.  A conservative iteration budget
is cheap here and prevents a lower-T result from being confounded by an
under-converged capacity projection.  The script still reports max row/column
residual and flags a T if the requested tolerance is not met.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

# This script is intended to live beside eval_scheduler_v1.py.
try:
    import eval_scheduler_v1 as ev1
except ImportError as exc:
    raise ImportError(
        "Could not import eval_scheduler_v1.py. Put this script in the same "
        "directory as eval_scheduler_v1.py (normally scripts/) and run it from "
        "the project root."
    ) from exc

from utils.circuit_sources import build_provider
from utils.inference_utils import enforce_capacity_sequence
from baselines_tier1 import baseline_b1


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser(
        description="Offline MOSAIC Sinkhorn inference-temperature sweep"
    )
    p.add_argument("--run_dir", required=True,
                   help="HiPerGator run directory with checkpoint/config snapshots")
    p.add_argument("--checkpoint", default="best",
                   help="best | final | last | epoch_NNN (default: best)")
    p.add_argument("--n_circuits", type=int, default=100,
                   help="Number of fixed-N synthetic circuits (default: 100)")
    p.add_argument("--seed", type=int, default=99999,
                   help="Eval provider seed; match prior evaluator to reuse circuits")
    p.add_argument("--num_qubits", type=int, default=30,
                   help="Fixed logical qubit count (default: 30)")
    p.add_argument("--temperatures", type=str,
                   default="0.5,0.4,0.35,0.30,0.25",
                   help="Comma-separated inference temperatures")
    p.add_argument("--sinkhorn_iters", type=int, default=200,
                   help="Sinkhorn iterations used at EVERY T (default: 200)")
    p.add_argument("--residual_tol", type=float, default=1e-4,
                   help="Max acceptable row/column residual (default: 1e-4)")
    p.add_argument("--save_dir", default=None,
                   help="Output dir; default <run_dir>/temp_sweep_N<num_qubits>")
    p.add_argument("--progress_every", type=int, default=20,
                   help="Progress print frequency while preparing circuits")
    return p.parse_args()


def parse_temperatures(s: str) -> List[float]:
    vals = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        t = float(tok)
        if t <= 0:
            raise ValueError(f"Temperature must be > 0, got {t}")
        vals.append(t)
    if not vals:
        raise ValueError("No temperatures supplied")
    # Keep caller order, but remove exact duplicates.
    out = []
    for t in vals:
        if t not in out:
            out.append(t)
    return out


# =============================================================================
# Metric helpers
# =============================================================================


def one_hot_schedule(assignments: Sequence[torch.Tensor], N: int, K: int,
                     device: str = "cpu") -> List[torch.Tensor]:
    out = []
    qidx = torch.arange(N, device=device)
    for a in assignments:
        P = torch.zeros(N, K, dtype=torch.float32, device=device)
        P[qidx, a.to(device).long()] = 1.0
        out.append(P)
    return out


def sum_cost_components(cost_out: dict) -> Dict[str, float]:
    def _sum(key: str) -> float:
        x = cost_out[key]
        return float(x.detach().sum().cpu().item())

    return {
        "total": float(cost_out["total_cost"].detach().cpu().item()),
        "exec": _sum("per_segment_exec"),
        "idle": _sum("per_segment_idle"),
        "comm": _sum("per_segment_comm"),
        "move": _sum("per_segment_move"),
    }


def assignment_movement(assignments: Sequence[torch.Tensor]) -> float:
    """Mean number of qubits whose discrete tech label changes per boundary."""
    if len(assignments) < 2:
        return 0.0
    return float(np.mean([
        float((a.cpu() != b.cpu()).sum().item())
        for a, b in zip(assignments[:-1], assignments[1:])
    ]))


def soft_movement_mass(P_seq: Sequence[torch.Tensor]) -> float:
    """Mean 0.5*L1 moved assignment mass per boundary."""
    if len(P_seq) < 2:
        return 0.0
    return float(np.mean([
        float(torch.relu(a.detach() - b.detach()).sum().cpu().item())
        for a, b in zip(P_seq[:-1], P_seq[1:])
    ]))


def repair_burden(raw_argmax: Sequence[torch.Tensor],
                  hard: Sequence[torch.Tensor]) -> float:
    if not raw_argmax:
        return 0.0
    return float(np.mean([
        float((a.cpu() != h.cpu()).sum().item())
        for a, h in zip(raw_argmax, hard)
    ]))


def argmax_overflow_stats(raw_argmax: Sequence[torch.Tensor],
                          caps: torch.Tensor, K: int):
    caps_cpu = caps.detach().cpu().float()
    overflows = []
    violating = 0
    for a in raw_argmax:
        counts = torch.stack([(a.cpu() == k).sum() for k in range(K)]).float()
        ov = float(torch.relu(counts - caps_cpu).sum().item())
        overflows.append(ov)
        if ov > 0:
            violating += 1
    mean_ov = float(np.mean(overflows)) if overflows else 0.0
    pct = 100.0 * violating / max(len(raw_argmax), 1)
    return mean_ov, pct


def sharpness_stats(P_seq: Sequence[torch.Tensor], K: int):
    ent, mx, margins = [], [], []
    low_conf = 0
    n_rows = 0

    for P in P_seq:
        p = P.detach().cpu().float().clamp(min=1e-12)
        row_ent = -(p * p.log()).sum(dim=1)
        ent.extend(row_ent.tolist())

        top = torch.topk(p, k=min(2, K), dim=1).values
        m = top[:, 0]
        mx.extend(m.tolist())
        if K >= 2:
            margins.extend((top[:, 0] - top[:, 1]).tolist())
        else:
            margins.extend([1.0] * p.shape[0])
        low_conf += int((m < 0.60).sum().item())
        n_rows += int(p.shape[0])

    mean_ent = float(np.mean(ent)) if ent else 0.0
    norm_ent = mean_ent / math.log(K) if K > 1 else 0.0
    mean_max = float(np.mean(mx)) if mx else 1.0
    mean_margin = float(np.mean(margins)) if margins else 1.0
    frac_low = low_conf / max(n_rows, 1)
    return mean_ent, norm_ent, mean_max, mean_margin, frac_low


def mean_of(rows: List[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if np.isfinite(float(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def std_of(rows: List[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if np.isfinite(float(r[key]))]
    return float(np.std(vals)) if vals else float("nan")


# =============================================================================
# Model-state controls
# =============================================================================


def set_head_temperature(cluster_module, T: float) -> None:
    """Set BOTH the registered buffer and the Python-float hot-path mirror."""
    head = cluster_module.head
    with torch.no_grad():
        head.temperature.fill_(float(T))
    # Current Sinkhorn head reads the Python mirror on the hot path.
    if hasattr(head, "_T"):
        head._T = float(T)


def set_sinkhorn_iterations(cluster_module, n_iters: int) -> None:
    head = cluster_module.head
    sk = getattr(head, "sinkhorn", None)
    if sk is None:
        raise RuntimeError(
            "This checkpoint/head is not using Sinkhorn. Temperature sweep is "
            "intended for the Sinkhorn arm."
        )
    sk.n_iters = int(n_iters)
    # Keep the checkpoint-traveling buffer consistent too, even though forward()
    # uses the Python integer mirror.
    if hasattr(sk, "n_iters_buf"):
        with torch.no_grad():
            sk.n_iters_buf.fill_(int(n_iters))


def reset_sinkhorn_diagnostics(cluster_module) -> None:
    head = cluster_module.head
    if hasattr(head, "reset_diagnostics"):
        head.reset_diagnostics()
    elif getattr(head, "sinkhorn", None) is not None:
        head.sinkhorn.reset_diagnostics()


def get_sinkhorn_residuals(cluster_module):
    head = cluster_module.head
    if hasattr(head, "diagnostics"):
        d = head.diagnostics
        return float(d.get("row_residual", 0.0)), float(d.get("col_residual", 0.0))
    sk = getattr(head, "sinkhorn", None)
    if sk is not None and hasattr(sk, "residuals"):
        return sk.residuals()
    return float("nan"), float("nan")


# =============================================================================
# Preparation: circuit + GNN state computed ONCE
# =============================================================================


def prepare_circuits(args, art: dict) -> List[dict]:
    circuit_cfg = copy.deepcopy(art["circuit_source_cfg"])
    circuit_cfg.setdefault("kwargs", {})["num_qubits"] = args.num_qubits
    provider = build_provider(circuit_cfg, seed_base=args.seed)

    prepared = []
    cost_module = art["cost_module"]
    K = art["K"]
    caps = art["caps"]
    device = art["device"]

    print("\nPreparing circuits and caching GNN hidden states once...")
    for i in range(args.n_circuits):
        qc = provider.get(i)
        rep, segments, layer_data = ev1.preprocess_circuit(
            qc, art["dataset_cfg"], art["w_short"], art["w_long"]
        )
        N = rep.num_qubits

        # GNN output is temperature-independent, so compute once and reuse for
        # all temperatures.
        with torch.no_grad():
            h_seq, _ = art["evol_model"](layer_data)

        # Cost-model circuit stats are also temperature-independent. Precompute
        # once to avoid repeating gamma/NetworkX work five times per circuit.
        stats_cpu = cost_module.stats_extractor.compute_stats_cpu(
            segments, rep, N=N, dtype=torch.float32
        )

        # B1 is also temperature-independent and gives a useful fixed reference.
        b1 = baseline_b1(rep, caps, art["config"], K)
        P_b1 = one_hot_schedule(b1, N, K, device)
        with torch.no_grad():
            b1_out = cost_module(
                P_b1, segments, rep, precomp_stats=stats_cpu
            )
        b1_c = sum_cost_components(b1_out)

        prepared.append({
            "circuit_idx": i,
            "provider_seed": args.seed + i,
            "rep": rep,
            "segments": segments,
            "layer_data": layer_data,
            "h_seq": h_seq,
            "stats_cpu": stats_cpu,
            "b1_cost": b1_c,
            "N": N,
            "T_layers": len(layer_data),
        })

        if ((i + 1) % args.progress_every == 0 or i == 0 or
                (i + 1) == args.n_circuits):
            print(f"  prepared {i+1:3d}/{args.n_circuits}  "
                  f"T={len(layer_data):3d}  B1={b1_c['total']:.3f}")

    return prepared


# =============================================================================
# One temperature
# =============================================================================


def evaluate_temperature(T: float, prepared: List[dict], art: dict,
                         sinkhorn_iters: int) -> List[dict]:
    cluster = art["cluster_module"]
    cost_module = art["cost_module"]
    caps = art["caps"]
    K = art["K"]
    device = art["device"]

    set_head_temperature(cluster, T)
    set_sinkhorn_iterations(cluster, sinkhorn_iters)
    reset_sinkhorn_diagnostics(cluster)

    rows = []
    with torch.no_grad():
        for item in prepared:
            N = item["N"]
            P_seq = cluster(item["h_seq"], graphs=item["layer_data"])

            raw_argmax = [P.argmax(dim=1) for P in P_seq]
            hard = enforce_capacity_sequence(P_seq, caps)

            P_argmax = one_hot_schedule(raw_argmax, N, K, device)
            P_hard = one_hot_schedule(hard, N, K, device)

            soft_out = cost_module(
                P_seq, item["segments"], item["rep"],
                precomp_stats=item["stats_cpu"]
            )
            argmax_out = cost_module(
                P_argmax, item["segments"], item["rep"],
                precomp_stats=item["stats_cpu"]
            )
            hard_out = cost_module(
                P_hard, item["segments"], item["rep"],
                precomp_stats=item["stats_cpu"]
            )

            soft_c = sum_cost_components(soft_out)
            argmax_c = sum_cost_components(argmax_out)
            hard_c = sum_cost_components(hard_out)
            b1_c = item["b1_cost"]

            soft_move = soft_movement_mass(P_seq)
            argmax_move = assignment_movement(raw_argmax)
            hard_move = assignment_movement(hard)
            burden = repair_burden(raw_argmax, hard)
            argmax_ov, argmax_viol_pct = argmax_overflow_stats(raw_argmax, caps, K)
            ent, ent_norm, maxp, margin, frac_low = sharpness_stats(P_seq, K)

            hardening_abs = hard_c["total"] - soft_c["total"]
            hardening_pct = 100.0 * hardening_abs / max(abs(soft_c["total"]), 1e-12)

            row = {
                "temperature": float(T),
                "sinkhorn_iters": int(sinkhorn_iters),
                "circuit_idx": item["circuit_idx"],
                "provider_seed": item["provider_seed"],
                "N": N,
                "T_layers": item["T_layers"],

                "soft_total": soft_c["total"],
                "argmax_total_infeasible_diag": argmax_c["total"],
                "hard_total": hard_c["total"],
                "b1_total": b1_c["total"],
                "mosaic_minus_b1": hard_c["total"] - b1_c["total"],
                "mosaic_beats_b1": int(hard_c["total"] < b1_c["total"]),

                "hardening_gap_abs": hardening_abs,
                "hardening_gap_pct": hardening_pct,
                "rounding_gap_soft_to_argmax": argmax_c["total"] - soft_c["total"],
                "repair_gap_argmax_to_feasible": hard_c["total"] - argmax_c["total"],

                "soft_move_mass_mean": soft_move,
                "argmax_movement_mean": argmax_move,
                "hard_movement_mean": hard_move,
                "repair_burden_mean": burden,
                "argmax_overflow_mean": argmax_ov,
                "argmax_violating_layers_pct": argmax_viol_pct,

                "mean_entropy_nats": ent,
                "mean_entropy_normalized": ent_norm,
                "mean_max_prob": maxp,
                "mean_top1_top2_margin": margin,
                "frac_rows_maxprob_lt_0p60": frac_low,
            }

            for prefix, comp in (("soft", soft_c), ("argmax", argmax_c),
                                 ("hard", hard_c), ("b1", b1_c)):
                for k in ("exec", "idle", "comm", "move"):
                    row[f"{prefix}_{k}"] = comp[k]

            rows.append(row)

    return rows


# =============================================================================
# Summary/output
# =============================================================================


SUMMARY_METRICS = [
    "soft_total", "argmax_total_infeasible_diag", "hard_total", "b1_total",
    "mosaic_minus_b1", "hardening_gap_abs", "hardening_gap_pct",
    "rounding_gap_soft_to_argmax", "repair_gap_argmax_to_feasible",
    "soft_move_mass_mean", "argmax_movement_mean", "hard_movement_mean",
    "repair_burden_mean", "argmax_overflow_mean", "argmax_violating_layers_pct",
    "mean_entropy_nats", "mean_entropy_normalized", "mean_max_prob",
    "mean_top1_top2_margin", "frac_rows_maxprob_lt_0p60",
    "soft_exec", "soft_idle", "soft_comm", "soft_move",
    "hard_exec", "hard_idle", "hard_comm", "hard_move",
]


def summarize_temperature(T: float, rows: List[dict], cluster_module,
                          residual_tol: float, sinkhorn_iters: int) -> dict:
    row_res, col_res = get_sinkhorn_residuals(cluster_module)
    out = {
        "temperature": float(T),
        "sinkhorn_iters": int(sinkhorn_iters),
        "sinkhorn_row_res_max": float(row_res),
        "sinkhorn_col_res_max": float(col_res),
        "sinkhorn_residual_ok": bool(max(row_res, col_res) <= residual_tol),
        "mosaic_b1_wins": int(sum(r["mosaic_beats_b1"] for r in rows)),
        "mosaic_b1_win_pct": 100.0 * sum(r["mosaic_beats_b1"] for r in rows) / max(len(rows), 1),
    }
    for key in SUMMARY_METRICS:
        out[f"{key}_mean"] = mean_of(rows, key)
        out[f"{key}_std"] = std_of(rows, key)
    return out


def render_table(summary_rows: List[dict], baseline_T: float) -> str:
    base = next((r for r in summary_rows if abs(r["temperature"] - baseline_T) < 1e-12),
                summary_rows[0])
    base_hard = base["hard_total_mean"]

    lines = []
    lines.append("=" * 126)
    lines.append("MOSAIC OFFLINE TEMPERATURE SWEEP — SAME CHECKPOINT / SAME CIRCUITS")
    lines.append("=" * 126)
    lines.append(
        "   T   iters    col_res    soft     argmax*    hard      B1    "
        "hard-vs-T0   gap%   entropy   maxP   repair   hardMove  win%B1"
    )
    lines.append("-" * 126)

    for r in summary_rows:
        dh = r["hard_total_mean"] - base_hard
        lines.append(
            f" {r['temperature']:>4.2f}  {r['sinkhorn_iters']:>5d}  "
            f"{r['sinkhorn_col_res_max']:>9.2e}  "
            f"{r['soft_total_mean']:>7.3f}  "
            f"{r['argmax_total_infeasible_diag_mean']:>8.3f}  "
            f"{r['hard_total_mean']:>7.3f}  "
            f"{r['b1_total_mean']:>7.3f}  "
            f"{dh:>+10.3f}  "
            f"{r['hardening_gap_pct_mean']:>6.1f}  "
            f"{r['mean_entropy_normalized_mean']:>7.3f}  "
            f"{r['mean_max_prob_mean']:>5.3f}  "
            f"{r['repair_burden_mean_mean']:>6.3f}  "
            f"{r['hard_movement_mean_mean']:>8.3f}  "
            f"{r['mosaic_b1_win_pct']:>6.1f}"
            + ("  !RES" if not r["sinkhorn_residual_ok"] else "")
        )

    lines.append("\n* argmax is diagnostic only and may violate capacity.")
    lines.append("hard-vs-T0 is relative to the first/baseline T shown; negative is better.")
    lines.append("!RES means max(row,col) Sinkhorn residual exceeded --residual_tol.")

    # Components table
    lines.append("\nHARD EFCL COMPONENTS")
    lines.append("   T        exec       idle       comm       move")
    lines.append("-" * 58)
    for r in summary_rows:
        lines.append(
            f" {r['temperature']:>4.2f}  "
            f"{r['hard_exec_mean']:>10.4f} "
            f"{r['hard_idle_mean']:>10.4f} "
            f"{r['hard_comm_mean']:>10.4f} "
            f"{r['hard_move_mean']:>10.4f}"
        )

    lines.append("\nSOFT EFCL COMPONENTS")
    lines.append("   T        exec       idle       comm       move")
    lines.append("-" * 58)
    for r in summary_rows:
        lines.append(
            f" {r['temperature']:>4.2f}  "
            f"{r['soft_exec_mean']:>10.4f} "
            f"{r['soft_idle_mean']:>10.4f} "
            f"{r['soft_comm_mean']:>10.4f} "
            f"{r['soft_move_mean']:>10.4f}"
        )

    valid = [r for r in summary_rows if r["sinkhorn_residual_ok"]]
    if valid:
        best = min(valid, key=lambda r: r["hard_total_mean"])
        lines.append("\nBEST RESIDUAL-VALID TEMPERATURE")
        lines.append(
            f"  T={best['temperature']:.3f}: hard={best['hard_total_mean']:.5f}, "
            f"B1={best['b1_total_mean']:.5f}, win={best['mosaic_b1_win_pct']:.1f}%, "
            f"gap={best['hardening_gap_pct_mean']:.2f}%, "
            f"entropy={best['mean_entropy_normalized_mean']:.4f}, "
            f"maxP={best['mean_max_prob_mean']:.4f}"
        )
        improvement = base_hard - best["hard_total_mean"]
        pct = 100.0 * improvement / max(abs(base_hard), 1e-12)
        lines.append(
            f"  improvement vs T={base['temperature']:.2f}: "
            f"{improvement:+.5f} cost ({pct:+.2f}%; positive means improvement)"
        )
    else:
        lines.append("\nWARNING: no temperature satisfied the Sinkhorn residual tolerance.")

    return "\n".join(lines) + "\n"


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()
    temps = parse_temperatures(args.temperatures)
    if args.sinkhorn_iters <= 0:
        raise SystemExit("--sinkhorn_iters must be >= 1")

    t0 = time.time()
    save_dir = args.save_dir or os.path.join(
        args.run_dir, f"temp_sweep_N{args.num_qubits}"
    )
    os.makedirs(save_dir, exist_ok=True)

    # Keep evaluation behavior aligned with eval_scheduler_v1.py.
    art = ev1.load_run_artifacts(args.run_dir, args.checkpoint, device="cpu")

    head = art["cluster_module"].head
    if getattr(head, "sinkhorn", None) is None:
        raise SystemExit("Loaded checkpoint is not a Sinkhorn model.")

    c_total = int(art["caps"].sum().item())
    if args.num_qubits > c_total:
        raise SystemExit(
            f"N={args.num_qubits} exceeds total capacity C={c_total}; infeasible."
        )

    original_T = float(getattr(head, "_T", float(head.temperature)))
    original_iters = int(head.sinkhorn.n_iters)

    print("\nOffline temperature sweep")
    print(f"  run_dir          : {args.run_dir}")
    print(f"  checkpoint       : {args.checkpoint}")
    print(f"  N / caps         : {args.num_qubits} / {art['caps'].tolist()}")
    print(f"  circuits / seed  : {args.n_circuits} / {args.seed}")
    print(f"  temperatures     : {temps}")
    print(f"  Sinkhorn iters   : {args.sinkhorn_iters}")
    print(f"  residual tol     : {args.residual_tol:g}")
    print(f"  checkpoint T     : {original_T}")
    print(f"  checkpoint iters : {original_iters}")

    prepared = prepare_circuits(args, art)

    all_rows: List[dict] = []
    summary_rows: List[dict] = []

    for j, T in enumerate(temps, start=1):
        ts = time.time()
        print(f"\n[{j}/{len(temps)}] Evaluating T={T:.3f} ...")
        rows = evaluate_temperature(T, prepared, art, args.sinkhorn_iters)
        summary = summarize_temperature(
            T, rows, art["cluster_module"], args.residual_tol, args.sinkhorn_iters
        )
        all_rows.extend(rows)
        summary_rows.append(summary)

        print(
            f"  hard={summary['hard_total_mean']:.4f}  "
            f"soft={summary['soft_total_mean']:.4f}  "
            f"B1={summary['b1_total_mean']:.4f}  "
            f"gap={summary['hardening_gap_pct_mean']:.2f}%  "
            f"entropy={summary['mean_entropy_normalized_mean']:.3f}  "
            f"maxP={summary['mean_max_prob_mean']:.3f}  "
            f"repair={summary['repair_burden_mean_mean']:.3f}  "
            f"hardMove={summary['hard_movement_mean_mean']:.3f}  "
            f"col_res={summary['sinkhorn_col_res_max']:.2e}  "
            f"winB1={summary['mosaic_b1_win_pct']:.1f}%  "
            f"({time.time()-ts:.1f}s)"
        )

    # Restore in-memory state for cleanliness.
    set_head_temperature(art["cluster_module"], original_T)
    set_sinkhorn_iterations(art["cluster_module"], original_iters)

    # Add paired delta vs first temperature to summary CSV/JSON.
    base_hard = summary_rows[0]["hard_total_mean"]
    base_T = summary_rows[0]["temperature"]
    for r in summary_rows:
        r["hard_delta_vs_baseline_T"] = r["hard_total_mean"] - base_hard
        r["hard_pct_change_vs_baseline_T"] = (
            100.0 * (r["hard_total_mean"] - base_hard) / max(abs(base_hard), 1e-12)
        )
        r["baseline_temperature"] = base_T

    # Per-circuit long-form CSV.
    per_path = os.path.join(save_dir, "temperature_per_circuit.csv")
    with open(per_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    # One row per temperature.
    sum_csv = os.path.join(save_dir, "temperature_summary.csv")
    with open(sum_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    valid = [r for r in summary_rows if r["sinkhorn_residual_ok"]]
    best = min(valid, key=lambda r: r["hard_total_mean"]) if valid else None

    out_json = {
        "run_dir": args.run_dir,
        "checkpoint": args.checkpoint,
        "num_qubits": args.num_qubits,
        "n_circuits": args.n_circuits,
        "seed": args.seed,
        "temperatures": temps,
        "sinkhorn_iters": args.sinkhorn_iters,
        "residual_tol": args.residual_tol,
        "checkpoint_temperature": original_T,
        "checkpoint_sinkhorn_iters": original_iters,
        "summary": summary_rows,
        "best_residual_valid_temperature": best,
    }
    json_path = os.path.join(save_dir, "temperature_summary.json")
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)

    text = render_table(summary_rows, base_T)
    txt_path = os.path.join(save_dir, "temperature_summary.txt")
    with open(txt_path, "w") as f:
        f.write(text)

    print("\n" + text)
    print(f"Saved: {per_path}")
    print(f"Saved: {sum_csv}")
    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
