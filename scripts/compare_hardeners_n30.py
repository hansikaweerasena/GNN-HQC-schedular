#!/usr/bin/env python3
"""
Compare MOSAIC's current confidence hardener with an EFCL-aware hardener.

No retraining. The same checkpoint, soft P_seq, circuits, TotalCost, and B1 are
used for both hardeners.

Recommended initial run:
    python scripts/compare_hardeners_n30.py \
        --run_dir results/pilot30_sinkhorn2 \
        --checkpoint best \
        --n_circuits 100 \
        --seed 99999 \
        --num_qubits 30 \
        --candidate_pool 8 \
        --save_dir results/pilot30_sinkhorn2/hardener_experiment_N30

For an exhaustive candidate repair (slower, N=30 only):
    ... --candidate_pool 0
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
from typing import Dict, List

import numpy as np
import torch

try:
    import eval_scheduler_v1 as ev1
except ImportError as exc:
    raise ImportError(
        "Put this script beside eval_scheduler_v1.py (normally scripts/) and "
        "run it from the project root."
    ) from exc

from utils.circuit_sources import build_provider
from utils.inference_utils import enforce_capacity_sequence
from baselines_tier1 import baseline_b1
from utils.cost_aware_hardening import cost_aware_enforce_capacity_sequence


def parse_args():
    p = argparse.ArgumentParser(description="MOSAIC cost-aware hardener experiment")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--checkpoint", default="best")
    p.add_argument("--n_circuits", type=int, default=100)
    p.add_argument("--seed", type=int, default=99999)
    p.add_argument("--num_qubits", type=int, default=30)
    p.add_argument(
        "--candidate_pool", type=int, default=8,
        help="EFCL-score this many lowest-soft-loss legal moves per repair step; <=0 = all"
    )
    p.add_argument("--progress_every", type=int, default=5)
    p.add_argument("--save_dir", default=None)
    return p.parse_args()


def one_hot_schedule(assignments: List[torch.Tensor], N: int, K: int) -> List[torch.Tensor]:
    out = []
    idx = torch.arange(N)
    for a in assignments:
        P = torch.zeros(N, K, dtype=torch.float32)
        P[idx, a.cpu().long()] = 1.0
        out.append(P)
    return out


def sum_components(out: dict) -> Dict[str, float]:
    return {
        "total": float(out["total_cost"].detach().cpu().item()),
        "exec": float(out["per_segment_exec"].detach().sum().cpu().item()),
        "idle": float(out["per_segment_idle"].detach().sum().cpu().item()),
        "comm": float(out["per_segment_comm"].detach().sum().cpu().item()),
        "move": float(out["per_segment_move"].detach().sum().cpu().item()),
    }


def score(cost_module, P, segments, rep, stats):
    with torch.no_grad():
        return sum_components(cost_module(P, segments, rep, precomp_stats=stats))


def movement(a: List[torch.Tensor]) -> float:
    if len(a) < 2:
        return 0.0
    return float(np.mean([
        float((x.cpu() != y.cpu()).sum().item()) for x, y in zip(a[:-1], a[1:])
    ]))


def burden(raw: List[torch.Tensor], hard: List[torch.Tensor]) -> float:
    return float(np.mean([
        float((x.cpu() != y.cpu()).sum().item()) for x, y in zip(raw, hard)
    ])) if raw else 0.0


def cut_rate(a: List[torch.Tensor], rep) -> float:
    total = 0
    cut = 0
    for t in range(min(len(a), len(rep.layers))):
        at = a[t].cpu()
        for _, qargs in rep.layers[t].gates:
            if len(qargs) == 2:
                u, v = qargs
                total += 1
                if int(at[u]) != int(at[v]):
                    cut += 1
    return cut / max(total, 1)


def idle_placement(a: List[torch.Tensor], rep, config, K) -> float:
    return float(ev1.compute_idle_decoherence_placement(a, rep, config, K))


def argmax_overflow(raw: List[torch.Tensor], caps: torch.Tensor, K: int):
    vals = []
    violating = 0
    caps = caps.cpu().long()
    for a in raw:
        c = torch.stack([(a.cpu() == k).sum() for k in range(K)]).long()
        ov = int(torch.relu(c - caps).sum().item())
        vals.append(ov)
        violating += int(ov > 0)
    return float(np.mean(vals)), 100.0 * violating / max(len(raw), 1)


def diagnose_one(qc, art, candidate_pool: int) -> dict:
    rep, segments, layer_data = ev1.preprocess_circuit(
        qc, art["dataset_cfg"], art["w_short"], art["w_long"]
    )
    N = rep.num_qubits
    K = art["K"]
    caps = art["caps"]
    cm = art["cost_module"]

    # One expensive structural-stat extraction per circuit. Every candidate
    # TotalCost call reuses this exact data.
    stats = cm.stats_extractor.compute_stats_cpu(segments, rep, N=N)

    P_seq = ev1.run_inference(art["evol_model"], art["cluster_module"], layer_data)
    raw = [P.detach().argmax(dim=1).cpu() for P in P_seq]

    # Existing production repair.
    current = [a.cpu() for a in enforce_capacity_sequence(P_seq, caps)]

    # Validation inside ASAP checks circuit-gate disjointness, not assignment.
    # Stats/circuit were already built once, so turn the repeated assertion off
    # for thousands of candidate scores. This does NOT change the cost formula.
    old_validate = getattr(cm, "asap_validate", False)
    cm.asap_validate = False
    try:
        t_ca = time.time()
        costaware, ca_diag = cost_aware_enforce_capacity_sequence(
            P_seq,
            caps,
            cost_module=cm,
            segments=segments,
            circuit=rep,
            precomp_stats=stats,
            candidate_pool=candidate_pool,
        )
        ca_seconds = time.time() - t_ca
    finally:
        cm.asap_validate = old_validate

    costaware = [a.cpu() for a in costaware]
    b1 = [a.cpu() for a in baseline_b1(rep, caps, art["config"], K)]

    P_soft = [p.cpu() for p in P_seq]
    P_raw = one_hot_schedule(raw, N, K)
    P_cur = one_hot_schedule(current, N, K)
    P_ca = one_hot_schedule(costaware, N, K)
    P_b1 = one_hot_schedule(b1, N, K)

    # Final exact scores. Precomputed stats are reused for all methods.
    s_soft = score(cm, P_soft, segments, rep, stats)
    s_raw = score(cm, P_raw, segments, rep, stats)
    s_cur = score(cm, P_cur, segments, rep, stats)
    s_ca = score(cm, P_ca, segments, rep, stats)
    s_b1 = score(cm, P_b1, segments, rep, stats)

    ov_mean, ov_pct = argmax_overflow(raw, caps, K)

    row = {
        "N": N,
        "T": len(P_seq),
        "soft_total": s_soft["total"],
        "raw_argmax_total_infeasible": s_raw["total"],
        "current_total": s_cur["total"],
        "costaware_total": s_ca["total"],
        "b1_total": s_b1["total"],
        "costaware_minus_current": s_ca["total"] - s_cur["total"],
        "costaware_improvement_pct": 100.0 * (s_cur["total"] - s_ca["total"]) / max(abs(s_cur["total"]), 1e-12),
        "current_minus_b1": s_cur["total"] - s_b1["total"],
        "costaware_minus_b1": s_ca["total"] - s_b1["total"],
        "current_beats_b1": int(s_cur["total"] < s_b1["total"]),
        "costaware_beats_b1": int(s_ca["total"] < s_b1["total"]),
        "rescued_vs_b1": int(s_cur["total"] >= s_b1["total"] and s_ca["total"] < s_b1["total"]),
        "hurt_vs_b1": int(s_cur["total"] < s_b1["total"] and s_ca["total"] >= s_b1["total"]),
        "current_repair_gap": s_cur["total"] - s_raw["total"],
        "costaware_repair_gap": s_ca["total"] - s_raw["total"],
        "current_burden": burden(raw, current),
        "costaware_burden": burden(raw, costaware),
        "hardener_disagreement": burden(current, costaware),
        "argmax_overflow_mean": ov_mean,
        "argmax_violating_layers_pct": ov_pct,
        "current_movement": movement(current),
        "costaware_movement": movement(costaware),
        "b1_movement": movement(b1),
        "current_cut_rate": cut_rate(current, rep),
        "costaware_cut_rate": cut_rate(costaware, rep),
        "b1_cut_rate": cut_rate(b1, rep),
        "current_idle_placement": idle_placement(current, rep, art["config"], K),
        "costaware_idle_placement": idle_placement(costaware, rep, art["config"], K),
        "b1_idle_placement": idle_placement(b1, rep, art["config"], K),
        "candidate_evaluations": int(ca_diag.candidate_evaluations),
        "costaware_repaired_layers": int(ca_diag.repaired_layers),
        "costaware_committed_moves": int(ca_diag.committed_moves),
        "costaware_max_initial_overflow": int(ca_diag.max_initial_overflow),
        "costaware_seconds": ca_seconds,
    }

    for prefix, comp in (("current", s_cur), ("costaware", s_ca), ("b1", s_b1), ("soft", s_soft)):
        for key in ("exec", "idle", "comm", "move"):
            row[f"{prefix}_{key}"] = comp[key]

    return row


MEAN_KEYS = [
    "soft_total", "raw_argmax_total_infeasible", "current_total", "costaware_total", "b1_total",
    "costaware_minus_current", "costaware_improvement_pct",
    "current_minus_b1", "costaware_minus_b1",
    "current_repair_gap", "costaware_repair_gap",
    "current_burden", "costaware_burden", "hardener_disagreement",
    "argmax_overflow_mean", "argmax_violating_layers_pct",
    "current_movement", "costaware_movement", "b1_movement",
    "current_cut_rate", "costaware_cut_rate", "b1_cut_rate",
    "current_idle_placement", "costaware_idle_placement", "b1_idle_placement",
    "candidate_evaluations", "costaware_repaired_layers", "costaware_committed_moves",
    "costaware_seconds",
    "current_exec", "current_idle", "current_comm", "current_move",
    "costaware_exec", "costaware_idle", "costaware_comm", "costaware_move",
    "b1_exec", "b1_idle", "b1_comm", "b1_move",
]


def avg(rows, key):
    return float(np.mean([float(r[key]) for r in rows]))


def std(rows, key):
    return float(np.std([float(r[key]) for r in rows]))


def make_summary(rows, args, art, elapsed):
    n = len(rows)
    cur_wins = sum(r["current_beats_b1"] for r in rows)
    ca_wins = sum(r["costaware_beats_b1"] for r in rows)
    rescued = sum(r["rescued_vs_b1"] for r in rows)
    hurt = sum(r["hurt_vs_b1"] for r in rows)
    ca_better = sum(r["costaware_total"] < r["current_total"] for r in rows)
    return {
        "run_dir": args.run_dir,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "num_qubits": args.num_qubits,
        "n_circuits": n,
        "candidate_pool": args.candidate_pool,
        "tech_names": art["tech_names"],
        "capacities": [float(x) for x in art["caps"].tolist()],
        "elapsed_seconds": elapsed,
        "costaware_lower_than_current_count": ca_better,
        "costaware_lower_than_current_pct": 100.0 * ca_better / max(n, 1),
        "current_wins_vs_b1": cur_wins,
        "current_win_pct_vs_b1": 100.0 * cur_wins / max(n, 1),
        "costaware_wins_vs_b1": ca_wins,
        "costaware_win_pct_vs_b1": 100.0 * ca_wins / max(n, 1),
        "rescued_vs_b1": rescued,
        "hurt_vs_b1": hurt,
        "mean": {k: avg(rows, k) for k in MEAN_KEYS},
        "std": {k: std(rows, k) for k in MEAN_KEYS},
    }


def render(summary):
    a = summary["mean"]
    lines = []
    lines += ["=" * 82, "MOSAIC COST-AWARE HARDENER EXPERIMENT", "=" * 82]
    lines.append(
        f"Circuits={summary['n_circuits']}  N={summary['num_qubits']}  "
        f"checkpoint={summary['checkpoint']}  seed={summary['seed']}  "
        f"candidate_pool={summary['candidate_pool']}"
    )
    lines.append(f"Techs={summary['tech_names']}  caps={summary['capacities']}")

    lines.append("\n[1] COST PATH")
    lines.append(f"  soft MOSAIC                         : {a['soft_total']:.5f}")
    lines.append(f"  raw argmax (infeasible diagnostic)  : {a['raw_argmax_total_infeasible']:.5f}")
    lines.append(f"  current confidence hardener          : {a['current_total']:.5f}")
    lines.append(f"  EFCL-aware hardener                  : {a['costaware_total']:.5f}")
    lines.append(f"  B1                                    : {a['b1_total']:.5f}")
    lines.append(f"  cost-aware - current                  : {a['costaware_minus_current']:+.5f}")
    lines.append(f"  current repair gap from raw argmax    : {a['current_repair_gap']:+.5f}")
    lines.append(f"  cost-aware repair gap from raw argmax : {a['costaware_repair_gap']:+.5f}")

    lines.append("\n[2] PAIRED OUTCOMES")
    lines.append(
        f"  cost-aware lower than current: {summary['costaware_lower_than_current_count']}/"
        f"{summary['n_circuits']} ({summary['costaware_lower_than_current_pct']:.1f}%)"
    )
    lines.append(
        f"  current MOSAIC beats B1       : {summary['current_wins_vs_b1']}/"
        f"{summary['n_circuits']} ({summary['current_win_pct_vs_b1']:.1f}%)"
    )
    lines.append(
        f"  cost-aware MOSAIC beats B1    : {summary['costaware_wins_vs_b1']}/"
        f"{summary['n_circuits']} ({summary['costaware_win_pct_vs_b1']:.1f}%)"
    )
    lines.append(f"  B1 losses rescued             : {summary['rescued_vs_b1']}")
    lines.append(f"  previous B1 wins lost         : {summary['hurt_vs_b1']}")

    lines.append("\n[3] SCHEDULE BEHAVIOUR")
    lines.append("                            current      cost-aware       B1")
    lines.append(
        f"  movement / boundary      {a['current_movement']:10.4f}  "
        f"{a['costaware_movement']:12.4f}  {a['b1_movement']:8.4f}"
    )
    lines.append(
        f"  remote 2Q cut rate       {100*a['current_cut_rate']:9.2f}%  "
        f"{100*a['costaware_cut_rate']:11.2f}%  {100*a['b1_cut_rate']:7.2f}%"
    )
    lines.append(
        f"  idle placement           {a['current_idle_placement']:10.4f}  "
        f"{a['costaware_idle_placement']:12.4f}  {a['b1_idle_placement']:8.4f}"
    )
    lines.append(
        f"  repair burden/layer      {a['current_burden']:10.4f}  "
        f"{a['costaware_burden']:12.4f}"
    )
    lines.append(f"  hardener disagreement                    {a['hardener_disagreement']:.4f} qubits/layer")

    lines.append("\n[4] HARD EFCL COMPONENTS")
    lines.append("                         exec       idle       comm       move")
    for label, pfx in (("current", "current"), ("cost-aware", "costaware"), ("B1", "b1")):
        lines.append(
            f"  {label:12s} " + " ".join(f"{a[f'{pfx}_{k}']:10.4f}" for k in ("exec", "idle", "comm", "move"))
        )

    lines.append("\n[5] COMPUTE")
    lines.append(f"  candidate TotalCost evaluations/circuit : {a['candidate_evaluations']:.1f}")
    lines.append(f"  repaired layers/circuit                 : {a['costaware_repaired_layers']:.2f}")
    lines.append(f"  committed repair moves/circuit          : {a['costaware_committed_moves']:.2f}")
    lines.append(f"  cost-aware repair time/circuit           : {a['costaware_seconds']:.3f}s")
    lines.append(f"  full experiment elapsed                  : {summary['elapsed_seconds']:.1f}s")

    lines.append("\nINTERPRETATION")
    lines.append("  * If cost-aware TotalCost falls materially and B1 win rate rises, the")
    lines.append("    confidence-only decoder is a real bottleneck; do not change the GNN yet.")
    lines.append("  * Especially look for idle cost falling without communication exploding.")
    lines.append("  * If cost-aware stays near the current hardener, the larger problem is the")
    lines.append("    soft relaxation / policy itself rather than which overflowing qubit is moved.")
    lines.append("  * This is still greedy repair: it does not modify already-feasible layers.")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    t0 = time.time()
    save_dir = args.save_dir or os.path.join(
        args.run_dir, f"hardener_experiment_N{args.num_qubits}"
    )
    os.makedirs(save_dir, exist_ok=True)

    art = ev1.load_run_artifacts(args.run_dir, args.checkpoint, device="cpu")
    if args.num_qubits > int(art["caps"].sum().item()):
        raise SystemExit("Requested N exceeds total hardware capacity.")

    cfg = copy.deepcopy(art["circuit_source_cfg"])
    cfg.setdefault("kwargs", {})["num_qubits"] = args.num_qubits
    provider = build_provider(cfg, seed_base=args.seed)

    rows = []
    print("\nRunning current-vs-cost-aware hardener experiment")
    print(f"  circuits={args.n_circuits}, N={args.num_qubits}, seed={args.seed}")
    print(f"  candidate_pool={args.candidate_pool} (<=0 means exhaustive)\n")

    for i in range(args.n_circuits):
        row = diagnose_one(provider.get(i), art, args.candidate_pool)
        row = {"circuit_idx": i, "provider_seed": args.seed + i, **row}
        rows.append(row)

        if i == 0 or (i + 1) % args.progress_every == 0 or i + 1 == args.n_circuits:
            print(
                f"[{i+1:3d}/{args.n_circuits}] T={row['T']:3d} "
                f"current={row['current_total']:.3f} "
                f"CA={row['costaware_total']:.3f} B1={row['b1_total']:.3f} "
                f"dCA={row['costaware_minus_current']:+.3f} "
                f"evals={row['candidate_evaluations']:4d} "
                f"time={row['costaware_seconds']:.2f}s"
            )

    elapsed = time.time() - t0
    summary = make_summary(rows, args, art, elapsed)
    text = render(summary)

    csv_path = os.path.join(save_dir, "hardener_per_circuit.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    json_path = os.path.join(save_dir, "hardener_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    txt_path = os.path.join(save_dir, "hardener_summary.txt")
    with open(txt_path, "w") as f:
        f.write(text)

    print("\n" + text)
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
