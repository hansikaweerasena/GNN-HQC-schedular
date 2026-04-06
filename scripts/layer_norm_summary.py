"""
layer_norm_summary.py  —  Layer-Normalized Hard Cost Summary

Post-processes results JSON from eval_scheduler_v1 (synthetic) or
eval_scheduler_v2 (real/MQT) to compute per-layer hard cost:

    LN_cost = hard_cost / T   per circuit per method

Then aggregates (mean ± std) across circuits.

For real mode: reproduces the full v2 comparison table (all 4 metrics,
win rates, per-algorithm breakdown) but with hard_cost replaced by
layer-normalized hard_cost. Supports excluding algorithms (e.g. --exclude bv).

Usage:
    # Synthetic (fixed-size or range)
    python layer_norm_summary.py \
        --json_file eval_syn_best/comparison_results_N30.json \
        --mode synthetic

    # Real / MQT Bench (exclude BV)
    python layer_norm_summary.py \
        --json_file eval_mqt_best/mqt_results.json \
        --mode real \
        --exclude bv
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np


METHOD_NAMES = ["MOSAIC", "B1", "B3", "B4", "B5"]

# Metrics config: (key, display_label, direction, fmt)
# hard_cost is replaced by ln_hard_cost in all outputs
METRICS_CFG = [
    ("ln_hard_cost",          "LN Hard Cost (cost/layer)",    "↓", ".4f"),
    ("remote_2q_cut_rate",    "Remote 2Q Cut Rate (%)",       "↓", ".2f"),
    ("mean_movement",         "Mean Temporal Movement",       "↓", ".3f"),
    ("idle_decoherence_rate", "Idle Decoherence Placement",   "↑", ".4f"),
]

SCALE_PCT = {"remote_2q_cut_rate"}


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Layer-Normalized Hard Cost Summary")
    p.add_argument("--json_file", type=str, required=True,
                   help="Path to results JSON from eval_scheduler_v1 or v2")
    p.add_argument("--mode", type=str, required=True, choices=["synthetic", "real"],
                   help="Processing mode: 'synthetic' (v1) or 'real' (v2/MQT)")
    p.add_argument("--exclude", type=str, default=None,
                   help="Comma-separated list of algorithms to exclude (e.g. 'bv,wstate')")
    return p.parse_args()


# =============================================================================
# Data loading and filtering
# =============================================================================

def load_and_filter(
    data:         dict,
    method_names: List[str],
    exclude_algos: Optional[List[str]] = None,
) -> Tuple[Dict[str, List[dict]], int]:
    """
    Load per-circuit data, add ln_hard_cost, optionally filter by algo.
    Returns (methods_dict, n_excluded).
    """
    methods = data["methods"]
    n_excluded = 0

    filtered: Dict[str, List[dict]] = {m: [] for m in method_names}

    # Use first method to determine indices to keep
    ref_circuits = methods[method_names[0]]["per_circuit"]

    keep_indices = []
    for i, entry in enumerate(ref_circuits):
        algo = entry.get("algo", "")
        if exclude_algos and algo in exclude_algos:
            n_excluded += 1
            continue
        keep_indices.append(i)

    for method in method_names:
        per_circuit = methods[method]["per_circuit"]
        for i in keep_indices:
            entry = dict(per_circuit[i])  # shallow copy
            T = entry.get("T", 0)
            hc = entry.get("hard_cost", 0.0)
            entry["ln_hard_cost"] = hc / T if T > 0 else 0.0
            filtered[method].append(entry)

    return filtered, n_excluded


# =============================================================================
# Formatting helpers
# =============================================================================

def _scale(key: str, val: float) -> float:
    return val * 100.0 if key in SCALE_PCT else val


def fmt_stats_simple(vals: List[float], fmt: str = ".4f") -> str:
    if not vals:
        return "N/A"
    return (f"{np.mean(vals):{fmt}} ± {np.std(vals):{fmt}}  "
            f"[min={np.min(vals):{fmt}}, max={np.max(vals):{fmt}}]")


# =============================================================================
# Synthetic mode tables
# =============================================================================

def build_synthetic_aggregate(
    methods: Dict[str, List[dict]],
    method_names: List[str],
) -> List[str]:
    lines = []
    lines.append("AGGREGATE LAYER-NORMALIZED HARD COST (hard_cost / T)")
    lines.append("=" * 72)
    lines.append("")

    col_w = 10
    val_w = 50
    lines.append(f"  {'Method':<{col_w}}  {'Mean ± Std  [Min, Max]':>{val_w}}")
    lines.append("  " + "-" * (col_w + val_w + 2))

    for method in method_names:
        vals = [e["ln_hard_cost"] for e in methods[method]]
        lines.append(f"  {method:<{col_w}}  {fmt_stats_simple(vals):>{val_w}}")

    lines.append("")
    return lines


def build_per_n_table(
    methods: Dict[str, List[dict]],
    method_names: List[str],
) -> List[str]:
    lines = []
    lines.append("PER-SIZE BREAKDOWN (layer-normalized hard cost by qubit count)")
    lines.append("=" * 72)
    lines.append("")

    n_values = sorted(set(
        e["N"] for e in methods[method_names[0]] if "N" in e))

    if not n_values:
        lines.append("  No per-size data available.")
        lines.append("")
        return lines

    col_w = 6
    val_w = 14
    header = f"  {'N':>{col_w}}"
    for method in method_names:
        header += f"  {method:^{val_w}}"
    lines.append(header)
    lines.append("  " + "-" * (col_w + (val_w + 2) * len(method_names)))

    for n_val in n_values:
        row = f"  {n_val:>{col_w}}"
        for method in method_names:
            vals = [e["ln_hard_cost"] for e in methods[method]
                    if e.get("N") == n_val]
            if vals:
                cell = f"{np.mean(vals):.4f}±{np.std(vals):.4f}"
            else:
                cell = "N/A"
            row += f"  {cell:^{val_w}}"
        lines.append(row)

    lines.append("")
    return lines


# =============================================================================
# Real mode tables (full comparison like v2 output)
# =============================================================================

def build_real_aggregate_table(
    methods:      Dict[str, List[dict]],
    method_names: List[str],
    tech_names:   Optional[List[str]] = None,
    K:            Optional[int] = None,
) -> List[str]:
    """Full aggregate comparison table with all 4 metrics."""
    lines = []
    lines.append("AGGREGATE COMPARISON TABLE (layer-normalized, filtered)")
    lines.append("=" * 72)
    lines.append("")

    n = len(methods[method_names[0]])
    lines.append(f"  Circuits evaluated : {n}")
    if tech_names and K:
        lines.append(f"  Technologies (K={K}): {', '.join(tech_names)}")
    lines.append("")

    col_w = 38
    val_w = 18

    header = f"  {'Metric':<{col_w}}"
    for m in method_names:
        header += f"  {m:^{val_w}}"
    lines.append(header)
    lines.append("  " + "-" * (col_w + (val_w + 2) * len(method_names)))

    for key, label, direction, fmt in METRICS_CFG:
        row = f"  {label + ' ' + direction:<{col_w}}"
        for method in method_names:
            vals = [_scale(key, e[key]) for e in methods[method]]
            cell = f"{np.mean(vals):{fmt}} ± {np.std(vals):{fmt}}"
            row += f"  {cell:^{val_w}}"
        lines.append(row)

    lines.append("")
    lines.append("  " + "-" * (col_w + (val_w + 2) * len(method_names)))
    for key, label, direction, fmt in METRICS_CFG:
        means = {m: np.mean([_scale(key, e[key]) for e in methods[m]])
                 for m in method_names}
        winner = (min if direction == "↓" else max)(means, key=lambda m: means[m])
        lines.append(f"  {'Best ' + label + ':':<{col_w + 2}}  {winner}")

    lines.append("")

    # Win rates (ln_hard_cost)
    baselines = [m for m in method_names if m != "MOSAIC"]
    lines.append("  Win Rates (MOSAIC ln_hard_cost < baseline):")
    for bl in baselines:
        wins = sum(
            1 for i in range(n)
            if methods["MOSAIC"][i]["ln_hard_cost"] < methods[bl][i]["ln_hard_cost"]
        )
        lines.append(f"    MOSAIC vs {bl}: {wins}/{n}  ({100.0 * wins / max(n, 1):.1f}%)")
    lines.append("")
    return lines


def build_real_per_algo_table(
    methods:      Dict[str, List[dict]],
    method_names: List[str],
) -> List[str]:
    """Per-algorithm breakdown: LN cost, best baseline, winner, MOSAIC cut%."""
    lines = []
    lines.append("PER-ALGORITHM BREAKDOWN (MOSAIC LN hard cost vs best baseline)")
    lines.append("=" * 72)
    lines.append("")

    algo_set = sorted(set(
        e["algo"] for e in methods[method_names[0]] if "algo" in e))

    if not algo_set:
        lines.append("  No per-algorithm data available.")
        lines.append("")
        return lines

    col_w = 18
    lines.append(
        f"  {'Algorithm':<{col_w}}  {'N_circ':>6}  "
        f"{'MOSAIC':>10}  {'BestBase':>10}  {'Winner':>10}  {'MOSAIC_cut%':>11}")
    lines.append("  " + "-" * 78)

    baselines = [m for m in method_names if m != "MOSAIC"]

    for algo in algo_set:
        # Indices for this algorithm
        idxs = [i for i, e in enumerate(methods["MOSAIC"])
                if e.get("algo") == algo]
        n_circ = len(idxs)

        mosaic_mean = float(np.mean([methods["MOSAIC"][i]["ln_hard_cost"]
                                      for i in idxs]))

        # Best baseline for this algo
        best_bl_name = None
        best_bl_mean = float("inf")
        for bl in baselines:
            bl_mean = float(np.mean([methods[bl][i]["ln_hard_cost"]
                                      for i in idxs]))
            if bl_mean < best_bl_mean:
                best_bl_mean = bl_mean
                best_bl_name = bl

        winner = "MOSAIC" if mosaic_mean < best_bl_mean else best_bl_name

        # MOSAIC cut rate for this algo
        cut_rates = [methods["MOSAIC"][i]["remote_2q_cut_rate"] * 100.0
                     for i in idxs]
        mosaic_cut = float(np.mean(cut_rates))

        lines.append(
            f"  {algo:<{col_w}}  {n_circ:>6}  "
            f"{mosaic_mean:>10.4f}  {best_bl_mean:>10.4f}  "
            f"{winner:>10}  {mosaic_cut:>10.1f}%")

    lines.append("")
    return lines


def build_real_per_algo_all_methods(
    methods:      Dict[str, List[dict]],
    method_names: List[str],
) -> List[str]:
    """Per-algorithm LN hard cost for ALL methods (detailed view)."""
    lines = []
    lines.append("PER-ALGORITHM LN HARD COST BY METHOD")
    lines.append("=" * 72)
    lines.append("")

    algo_set = sorted(set(
        e["algo"] for e in methods[method_names[0]] if "algo" in e))

    if not algo_set:
        return lines

    col_w = 18
    val_w = 14
    header = f"  {'Algorithm':<{col_w}}"
    for method in method_names:
        header += f"  {method:^{val_w}}"
    header += f"  {'count':>6}"
    lines.append(header)
    lines.append("  " + "-" * (col_w + (val_w + 2) * len(method_names) + 8))

    for algo in algo_set:
        row = f"  {algo:<{col_w}}"
        idxs = [i for i, e in enumerate(methods[method_names[0]])
                if e.get("algo") == algo]
        for method in method_names:
            vals = [methods[method][i]["ln_hard_cost"] for i in idxs]
            if vals:
                cell = f"{np.mean(vals):.4f}±{np.std(vals):.4f}"
            else:
                cell = "N/A"
            row += f"  {cell:^{val_w}}"
        row += f"  {len(idxs):>6}"
        lines.append(row)

    lines.append("")
    return lines


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # ---- Parse exclusions ----
    exclude_algos = None
    if args.exclude:
        exclude_algos = [a.strip().lower() for a in args.exclude.split(",")]

    # ---- Load JSON ----
    with open(args.json_file) as f:
        data = json.load(f)

    raw_methods = data.get("methods", {})
    if not raw_methods:
        print(f"ERROR: no 'methods' key in {args.json_file}", file=sys.stderr)
        sys.exit(1)

    method_names = [m for m in METHOD_NAMES if m in raw_methods]
    if not method_names:
        print(f"ERROR: no recognized methods in JSON", file=sys.stderr)
        sys.exit(1)

    # ---- Validate T exists ----
    sample = raw_methods[method_names[0]]["per_circuit"][0]
    if "T" not in sample:
        print(f"ERROR: per_circuit entries are missing 'T'. "
              f"Re-run eval_scheduler_v1/v2 with the updated version that "
              f"preserves T in the JSON.", file=sys.stderr)
        sys.exit(1)

    # ---- Filter and compute LN cost ----
    methods, n_excluded = load_and_filter(data, method_names, exclude_algos)
    n_kept = len(methods[method_names[0]])

    if n_kept == 0:
        print("ERROR: no circuits remaining after filtering.", file=sys.stderr)
        sys.exit(1)

    # ---- Build output lines ----
    output = []

    # Metadata header
    output.append(f"Source:   {os.path.basename(args.json_file)}")
    output.append(f"Mode:     {args.mode}")
    if exclude_algos:
        output.append(f"Excluded: {', '.join(exclude_algos)} ({n_excluded} circuits removed)")
    if "number_of_qubits" in data:
        if data.get("is_range"):
            qr = data.get("qubit_range", [data["number_of_qubits"], "?"])
            output.append(f"Qubits:   {qr[0]} – {qr[1]} (range)")
        else:
            output.append(f"Qubits:   {data['number_of_qubits']}")
    output.append(f"Circuits: {n_kept}")
    output.append("")

    tech_names = data.get("tech_names", None)
    K = len(tech_names) if tech_names else None

    if args.mode == "synthetic":
        output += build_synthetic_aggregate(methods, method_names)
        is_range = data.get("is_range", False)
        if is_range:
            output += build_per_n_table(methods, method_names)

    elif args.mode == "real":
        output += build_real_aggregate_table(methods, method_names, tech_names, K)
        output += build_real_per_algo_table(methods, method_names)
        output += build_real_per_algo_all_methods(methods, method_names)

    # ---- Save ----
    stem = os.path.splitext(args.json_file)[0]
    out_path = f"{stem}_LN_summary.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(output))

    for line in output:
        print(line)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
