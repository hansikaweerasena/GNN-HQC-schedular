#!/usr/bin/env python3
"""
diagnose_soft_hard_n30.py

Cheap post-training diagnostic for MOSAIC/Sinkhorn.

Purpose
-------
For fresh ROI circuits generated exactly like eval_scheduler_v1.py (fixed-N mode),
measure whether MOSAIC's gap to B1 comes from:
  1) a soft -> discrete rounding gap,
  2) capacity-repair/hardener corrections,
  3) soft movement mass becoming many hard assignment flips, or
  4) the learned hard policy itself choosing a worse exec/idle/comm/move trade-off.

Outputs
-------
  diagnostic_per_circuit.csv
  diagnostic_summary.json
  diagnostic_summary.txt

Recommended use
---------------
Place this file beside eval_scheduler_v1.py (normally scripts/) and run from the
project root, e.g.:

  python scripts/diagnose_soft_hard_n30.py \
      --run_dir results/pilot30_sinkhorn2 \
      --checkpoint best \
      --n_circuits 100 \
      --seed 99999 \
      --num_qubits 30 \
      --save_dir results/pilot30_sinkhorn2/diag_soft_hard_N30

Use the SAME --seed as the eval_scheduler_v1.py run if you want these to be the
same first N circuits as the baseline-comparison JSON.
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
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch

# This script is intended to live beside eval_scheduler_v1.py.
# Importing it also installs the project root into sys.path using the same logic
# as the evaluator itself.
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


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="MOSAIC soft-vs-hard diagnostic (cheap, fixed-N synthetic circuits)"
    )
    p.add_argument("--run_dir", required=True,
                   help="HiPerGator run directory containing checkpoint/config snapshots")
    p.add_argument("--checkpoint", default="best",
                   help="best | final | last | epoch_NNN (default: best)")
    p.add_argument("--n_circuits", type=int, default=100,
                   help="Number of fresh circuits to diagnose (default: 100)")
    p.add_argument("--seed", type=int, default=99999,
                   help="Eval provider seed; match eval_scheduler_v1.py to reuse circuits")
    p.add_argument("--num_qubits", type=int, default=30,
                   help="Fixed logical qubit count (default: 30)")
    p.add_argument("--save_dir", default=None,
                   help="Output directory; defaults to <run_dir>/diag_soft_hard_N<num_qubits>")
    p.add_argument("--progress_every", type=int, default=10,
                   help="Print progress every N circuits (default: 10)")
    return p.parse_args()


def one_hot_schedule(assignments: List[torch.Tensor], N: int, K: int,
                     device: str = "cpu") -> List[torch.Tensor]:
    out = []
    qidx = torch.arange(N, device=device)
    for a in assignments:
        P = torch.zeros(N, K, dtype=torch.float32, device=device)
        P[qidx, a.to(device).long()] = 1.0
        out.append(P)
    return out


def sum_cost_components(cost_out: dict) -> Dict[str, float]:
    """Return circuit-total EFCL components from TotalCost output."""
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


def assignment_movement(assignments: List[torch.Tensor]) -> float:
    """Mean number of qubits whose hard tech label changes per boundary."""
    if len(assignments) < 2:
        return 0.0
    vals = []
    for a, b in zip(assignments[:-1], assignments[1:]):
        vals.append(float((a.cpu() != b.cpu()).sum().item()))
    return float(np.mean(vals))


def soft_movement_mass(P_seq: List[torch.Tensor]) -> float:
    """
    Mean moved probability mass per boundary, using the SAME definition as the
    current CommMoveCostV3:
        sum_u sum_k relu(P_t[u,k] - P_{t+1}[u,k])
      = 0.5 * sum_u ||P_t[u] - P_{t+1}[u]||_1.
    Units are 'soft qubit moves per boundary'.
    """
    if len(P_seq) < 2:
        return 0.0
    vals = []
    for a, b in zip(P_seq[:-1], P_seq[1:]):
        mass = torch.relu(a.detach() - b.detach()).sum().item()
        vals.append(float(mass))
    return float(np.mean(vals))


def repair_burden(argmax_sched: List[torch.Tensor],
                  hard_sched: List[torch.Tensor]) -> float:
    if not argmax_sched:
        return 0.0
    vals = [float((a.cpu() != h.cpu()).sum().item())
            for a, h in zip(argmax_sched, hard_sched)]
    return float(np.mean(vals))


def argmax_overflow_stats(argmax_sched: List[torch.Tensor],
                          caps: torch.Tensor, K: int):
    caps_cpu = caps.detach().cpu().float()
    overflows = []
    violating = 0
    for a in argmax_sched:
        counts = torch.stack([(a.cpu() == k).sum() for k in range(K)]).float()
        ov = torch.relu(counts - caps_cpu).sum().item()
        overflows.append(float(ov))
        if ov > 0:
            violating += 1
    mean_ov = float(np.mean(overflows)) if overflows else 0.0
    pct = 100.0 * violating / max(len(argmax_sched), 1)
    return mean_ov, pct


def sharpness_stats(P_seq: List[torch.Tensor], K: int):
    ent = []
    mx = []
    margins = []
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


def mean_dict(rows: List[dict], keys: List[str]) -> dict:
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if np.isfinite(float(r[k]))]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def std_dict(rows: List[dict], keys: List[str]) -> dict:
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if np.isfinite(float(r[k]))]
        out[k] = float(np.std(vals)) if vals else float("nan")
    return out


def fmt(x):
    return "nan" if not np.isfinite(float(x)) else f"{float(x):.5f}"


# -----------------------------------------------------------------------------
# Per-circuit diagnostic
# -----------------------------------------------------------------------------

def diagnose_one(qc, art: dict) -> dict:
    rep, segments, layer_data = ev1.preprocess_circuit(
        qc, art["dataset_cfg"], art["w_short"], art["w_long"]
    )
    N = rep.num_qubits
    K = art["K"]
    caps = art["caps"]
    device = art["device"]
    cost_module = art["cost_module"]

    # MOSAIC soft output and two discrete versions:
    #   raw_argmax: no capacity repair (may be infeasible; diagnostic only)
    #   hard:       capacity-feasible inference schedule used in evaluation
    P_seq = ev1.run_inference(art["evol_model"], art["cluster_module"], layer_data)
    raw_argmax = [P.argmax(dim=1) for P in P_seq]
    hard = enforce_capacity_sequence(P_seq, caps)

    P_argmax = one_hot_schedule(raw_argmax, N, K, device)
    P_hard = one_hot_schedule(hard, N, K, device)

    # B1 is cheap and is the main competitor we need to explain.
    b1 = baseline_b1(rep, caps, art["config"], K)
    P_b1 = one_hot_schedule(b1, N, K, device)

    with torch.no_grad():
        soft_out = cost_module(P_seq, segments, rep)
        # raw argmax cost can exploit an infeasible occupancy; keep only as a
        # diagnostic to split rounding vs capacity-repair effects.
        argmax_out = cost_module(P_argmax, segments, rep)
        hard_out = cost_module(P_hard, segments, rep)
        b1_out = cost_module(P_b1, segments, rep)

    soft_c = sum_cost_components(soft_out)
    argmax_c = sum_cost_components(argmax_out)
    hard_c = sum_cost_components(hard_out)
    b1_c = sum_cost_components(b1_out)

    soft_move = soft_movement_mass(P_seq)
    argmax_move = assignment_movement(raw_argmax)
    hard_move = assignment_movement(hard)
    b1_move = assignment_movement(b1)

    burden = repair_burden(raw_argmax, hard)
    argmax_ov, argmax_viol_pct = argmax_overflow_stats(raw_argmax, caps, K)
    ent, ent_norm, maxp, margin, frac_low = sharpness_stats(P_seq, K)

    hardening_abs = hard_c["total"] - soft_c["total"]
    hardening_pct = 100.0 * hardening_abs / max(abs(soft_c["total"]), 1e-12)
    rounding_abs = argmax_c["total"] - soft_c["total"]
    repair_abs = hard_c["total"] - argmax_c["total"]

    row = {
        "N": N,
        "T": len(P_seq),

        "soft_total": soft_c["total"],
        "hard_total": hard_c["total"],
        "b1_total": b1_c["total"],
        "mosaic_minus_b1": hard_c["total"] - b1_c["total"],
        "mosaic_beats_b1": int(hard_c["total"] < b1_c["total"]),

        "hardening_gap_abs": hardening_abs,
        "hardening_gap_pct": hardening_pct,
        "argmax_total_infeasible_diag": argmax_c["total"],
        "rounding_gap_soft_to_argmax": rounding_abs,
        "repair_gap_argmax_to_feasible": repair_abs,

        "soft_move_mass_mean": soft_move,
        "argmax_movement_mean": argmax_move,
        "hard_movement_mean": hard_move,
        "b1_movement_mean": b1_move,
        "repair_burden_mean": burden,
        "argmax_overflow_mean": argmax_ov,
        "argmax_violating_layers_pct": argmax_viol_pct,

        "mean_entropy_nats": ent,
        "mean_entropy_normalized": ent_norm,
        "mean_max_prob": maxp,
        "mean_top1_top2_margin": margin,
        "frac_rows_maxprob_lt_0p60": frac_low,
    }

    # Cost components, all in the same EFCL additive units.
    for prefix, comp in (("soft", soft_c), ("hard", hard_c), ("b1", b1_c)):
        for k in ("exec", "idle", "comm", "move"):
            row[f"{prefix}_{k}"] = comp[k]

    # Useful paired component deltas: negative means MOSAIC hard is better than B1.
    for k in ("exec", "idle", "comm", "move"):
        row[f"hard_minus_b1_{k}"] = hard_c[k] - b1_c[k]

    return row


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

SUMMARY_KEYS = [
    "soft_total", "hard_total", "b1_total", "mosaic_minus_b1",
    "hardening_gap_abs", "hardening_gap_pct",
    "rounding_gap_soft_to_argmax", "repair_gap_argmax_to_feasible",
    "soft_move_mass_mean", "argmax_movement_mean", "hard_movement_mean",
    "repair_burden_mean", "argmax_overflow_mean", "argmax_violating_layers_pct",
    "mean_entropy_nats", "mean_entropy_normalized", "mean_max_prob",
    "mean_top1_top2_margin", "frac_rows_maxprob_lt_0p60",
    "soft_exec", "soft_idle", "soft_comm", "soft_move",
    "hard_exec", "hard_idle", "hard_comm", "hard_move",
    "b1_exec", "b1_idle", "b1_comm", "b1_move",
    "hard_minus_b1_exec", "hard_minus_b1_idle",
    "hard_minus_b1_comm", "hard_minus_b1_move",
]


def build_summary(rows: List[dict], args, art: dict) -> dict:
    wins = [r for r in rows if r["mosaic_beats_b1"] == 1]
    losses = [r for r in rows if r["mosaic_beats_b1"] == 0]

    summary = {
        "run_dir": args.run_dir,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "num_qubits": args.num_qubits,
        "n_circuits": len(rows),
        "tech_names": art["tech_names"],
        "capacities": [float(x) for x in art["caps"].tolist()],
        "mosaic_b1_wins": len(wins),
        "mosaic_b1_losses_or_ties": len(losses),
        "mosaic_b1_win_pct": 100.0 * len(wins) / max(len(rows), 1),
        "all": {
            "mean": mean_dict(rows, SUMMARY_KEYS),
            "std": std_dict(rows, SUMMARY_KEYS),
        },
        "mosaic_wins_vs_b1": {
            "n": len(wins),
            "mean": mean_dict(wins, SUMMARY_KEYS) if wins else {},
        },
        "b1_wins_or_ties": {
            "n": len(losses),
            "mean": mean_dict(losses, SUMMARY_KEYS) if losses else {},
        },
    }
    return summary


def render_summary(summary: dict) -> str:
    a = summary["all"]["mean"]
    w = summary["mosaic_wins_vs_b1"]["mean"]
    l = summary["b1_wins_or_ties"]["mean"]

    lines = []
    lines.append("=" * 78)
    lines.append("MOSAIC SOFT -> HARD DIAGNOSTIC")
    lines.append("=" * 78)
    lines.append(
        f"Circuits={summary['n_circuits']}  N={summary['num_qubits']}  "
        f"checkpoint={summary['checkpoint']}  seed={summary['seed']}"
    )
    lines.append(
        f"Techs={summary['tech_names']}  caps={summary['capacities']}"
    )
    lines.append(
        f"MOSAIC beats B1: {summary['mosaic_b1_wins']}/{summary['n_circuits']} "
        f"({summary['mosaic_b1_win_pct']:.1f}%)"
    )

    lines.append("\n[1] COST: SOFT -> HARD -> B1")
    lines.append(
        f"  soft TotalCost                 : {fmt(a['soft_total'])}"
    )
    lines.append(
        f"  feasible hard TotalCost        : {fmt(a['hard_total'])}"
    )
    lines.append(
        f"  B1 TotalCost                   : {fmt(a['b1_total'])}"
    )
    lines.append(
        f"  hardening gap (hard-soft)      : {fmt(a['hardening_gap_abs'])} "
        f"({a['hardening_gap_pct']:.2f}%)"
    )
    lines.append(
        f"    soft -> raw argmax gap       : {fmt(a['rounding_gap_soft_to_argmax'])}"
    )
    lines.append(
        f"    raw argmax -> repair gap     : {fmt(a['repair_gap_argmax_to_feasible'])}"
    )
    lines.append("  NOTE: raw-argmax cost may be capacity-infeasible; it is diagnostic only.")

    lines.append("\n[2] MOVEMENT: SOFT MASS VS DISCRETE FLIPS")
    lines.append(
        f"  soft moved mass / boundary     : {fmt(a['soft_move_mass_mean'])}"
    )
    lines.append(
        f"  raw-argmax flips / boundary    : {fmt(a['argmax_movement_mean'])}"
    )
    lines.append(
        f"  repaired-hard flips / boundary : {fmt(a['hard_movement_mean'])}"
    )
    lines.append(
        f"  repair burden / layer          : {fmt(a['repair_burden_mean'])}"
    )
    lines.append(
        f"  argmax overflow / layer        : {fmt(a['argmax_overflow_mean'])}"
    )
    lines.append(
        f"  argmax violating layers        : {a['argmax_violating_layers_pct']:.2f}%"
    )

    lines.append("\n[3] ASSIGNMENT SHARPNESS")
    lines.append(
        f"  mean entropy (nats)            : {fmt(a['mean_entropy_nats'])}"
    )
    lines.append(
        f"  normalized entropy             : {fmt(a['mean_entropy_normalized'])}"
    )
    lines.append(
        f"  mean max probability           : {fmt(a['mean_max_prob'])}"
    )
    lines.append(
        f"  mean top1-top2 margin          : {fmt(a['mean_top1_top2_margin'])}"
    )
    lines.append(
        f"  rows with max prob < 0.60      : {100*a['frac_rows_maxprob_lt_0p60']:.2f}%"
    )

    lines.append("\n[4] EFCL COMPONENTS (circuit totals)")
    lines.append("                         exec       idle       comm       move")
    lines.append(
        "  soft MOSAIC       " + "  ".join(
            f"{a[f'soft_{k}']:9.4f}" for k in ("exec", "idle", "comm", "move")
        )
    )
    lines.append(
        "  hard MOSAIC       " + "  ".join(
            f"{a[f'hard_{k}']:9.4f}" for k in ("exec", "idle", "comm", "move")
        )
    )
    lines.append(
        "  hard B1           " + "  ".join(
            f"{a[f'b1_{k}']:9.4f}" for k in ("exec", "idle", "comm", "move")
        )
    )
    lines.append(
        "  MOSAIC - B1       " + "  ".join(
            f"{a[f'hard_minus_b1_{k}']:9.4f}" for k in ("exec", "idle", "comm", "move")
        )
    )
    lines.append("  (negative MOSAIC-B1 component = MOSAIC is better on that component)")

    if w and l:
        lines.append("\n[5] WHAT DIFFERS WHEN MOSAIC WINS VS WHEN B1 WINS")
        lines.append(
            f"  {'metric':34s} {'MOSAIC wins':>14s} {'B1 wins/ties':>14s}"
        )
        split_keys = [
            ("hardening_gap_pct", "hardening gap %"),
            ("soft_move_mass_mean", "soft move mass/boundary"),
            ("argmax_movement_mean", "argmax flips/boundary"),
            ("hard_movement_mean", "hard flips/boundary"),
            ("repair_burden_mean", "repair burden/layer"),
            ("mean_entropy_normalized", "normalized entropy"),
            ("mean_max_prob", "mean max probability"),
            ("hard_minus_b1_exec", "M-B1 execution cost"),
            ("hard_minus_b1_idle", "M-B1 idle cost"),
            ("hard_minus_b1_comm", "M-B1 communication cost"),
            ("hard_minus_b1_move", "M-B1 movement cost"),
        ]
        for key, label in split_keys:
            lines.append(f"  {label:34s} {w[key]:14.5f} {l[key]:14.5f}")

    lines.append("\nINTERPRETATION GUIDE")
    lines.append("  A) Large hardening gap + small soft move mass but many argmax/hard flips:")
    lines.append("     soft relaxation is too blurry; temperature/sharpness is the first knob.")
    lines.append("  B) Small hardening gap but high soft AND hard movement on B1-loss circuits:")
    lines.append("     the policy itself is over-moving; temporal context/stickiness is more likely.")
    lines.append("  C) Repair gap is large while soft->argmax gap is small:")
    lines.append("     capacity rounding/repair is the main discretization problem.")
    lines.append("  D) Hard MOSAIC gains comm cost but loses more in idle+move cost vs B1:")
    lines.append("     model learned locality, but not when a static B1-like partition is good enough.")

    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    t0 = time.time()

    save_dir = args.save_dir or os.path.join(
        args.run_dir, f"diag_soft_hard_N{args.num_qubits}"
    )
    os.makedirs(save_dir, exist_ok=True)

    # Match eval_scheduler_v1.py exactly: it evaluates on CPU and reconstructs
    # the model/cost from the run snapshots.
    art = ev1.load_run_artifacts(args.run_dir, args.checkpoint, device="cpu")

    c_total = int(art["caps"].sum().item())
    if args.num_qubits > c_total:
        raise SystemExit(
            f"N={args.num_qubits} exceeds total capacity {c_total}; infeasible."
        )

    # Fresh provider, same fixed-N construction as eval_scheduler_v1.py.
    circuit_cfg = copy.deepcopy(art["circuit_source_cfg"])
    circuit_cfg.setdefault("kwargs", {})["num_qubits"] = args.num_qubits
    provider = build_provider(circuit_cfg, seed_base=args.seed)

    rows = []
    print("\nRunning soft/hard diagnostic...")
    print(f"  run_dir={args.run_dir}")
    print(f"  checkpoint={args.checkpoint}")
    print(f"  N={args.num_qubits}, circuits={args.n_circuits}, seed={args.seed}")
    print(f"  caps={art['caps'].tolist()}\n")

    for i in range(args.n_circuits):
        qc = provider.get(i)
        row = diagnose_one(qc, art)
        row = {"circuit_idx": i, "provider_seed": args.seed + i, **row}
        rows.append(row)

        if ((i + 1) % args.progress_every == 0 or i == 0 or
                (i + 1) == args.n_circuits):
            print(
                f"[{i+1:3d}/{args.n_circuits}] T={row['T']:3d} "
                f"soft={row['soft_total']:.3f} hard={row['hard_total']:.3f} "
                f"B1={row['b1_total']:.3f} gap={row['hardening_gap_pct']:+.2f}% "
                f"softMove={row['soft_move_mass_mean']:.2f} "
                f"hardMove={row['hard_movement_mean']:.2f} "
                f"maxP={row['mean_max_prob']:.3f}"
            )

    summary = build_summary(rows, args, art)
    text = render_summary(summary)

    csv_path = os.path.join(save_dir, "diagnostic_per_circuit.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(save_dir, "diagnostic_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    txt_path = os.path.join(save_dir, "diagnostic_summary.txt")
    with open(txt_path, "w") as f:
        f.write(text)

    print("\n" + text)
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
