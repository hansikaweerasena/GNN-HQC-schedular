"""
layer_norm_summary.py  —  Layer-Normalized Hard Cost Summary

Post-processes results JSON from eval_scheduler_v1 (synthetic) or
eval_scheduler_v2 (real/MQT) to compute per-layer hard cost:

    LN_cost = hard_cost / T   per circuit per method

Then aggregates (mean ± std) across circuits.

Usage:
    # Synthetic (fixed-size or range)
    python layer_norm_summary.py \
        --json_file eval_syn_best/comparison_results_N30.json \
        --mode synthetic

    # Real / MQT Bench
    python layer_norm_summary.py \
        --json_file eval_mqt_best/mqt_results.json \
        --mode real
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np


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
    return p.parse_args()


# =============================================================================
# Core computation
# =============================================================================

def compute_ln_costs(per_circuit: List[dict]) -> List[float]:
    """Compute hard_cost / T for each circuit entry."""
    ln_costs = []
    for entry in per_circuit:
        T = entry.get("T")
        hard_cost = entry.get("hard_cost")
        if T is None or hard_cost is None:
            print(f"  WARNING: missing T or hard_cost in entry, skipping: "
                  f"{entry}", file=sys.stderr)
            continue
        if T == 0:
            print(f"  WARNING: T=0 in entry, skipping", file=sys.stderr)
            continue
        ln_costs.append(hard_cost / T)
    return ln_costs


# =============================================================================
# Formatting helpers
# =============================================================================

def fmt_stats(vals: List[float], fmt: str = ".4f") -> str:
    """Format mean ± std [min, max]."""
    if not vals:
        return "N/A"
    return (f"{np.mean(vals):{fmt}} ± {np.std(vals):{fmt}}  "
            f"[min={np.min(vals):{fmt}}, max={np.max(vals):{fmt}}]")


def build_aggregate_table(
    methods: Dict[str, List[dict]],
    method_names: List[str],
) -> List[str]:
    """Build aggregate LN cost table: one row per method."""
    lines = []
    lines.append("AGGREGATE LAYER-NORMALIZED HARD COST (hard_cost / T)")
    lines.append("=" * 72)
    lines.append("")

    col_w = 10
    val_w = 50
    lines.append(f"  {'Method':<{col_w}}  {'Mean ± Std  [Min, Max]':>{val_w}}")
    lines.append("  " + "-" * (col_w + val_w + 2))

    for method in method_names:
        per_circuit = methods[method]["per_circuit"]
        ln_costs = compute_ln_costs(per_circuit)
        lines.append(f"  {method:<{col_w}}  {fmt_stats(ln_costs):>{val_w}}")

    lines.append("")
    return lines


def build_per_n_table(
    methods: Dict[str, List[dict]],
    method_names: List[str],
) -> List[str]:
    """Build per-N breakdown table for range mode."""
    lines = []
    lines.append("PER-SIZE BREAKDOWN (layer-normalized hard cost by qubit count)")
    lines.append("=" * 72)
    lines.append("")

    # Collect all unique N values from MOSAIC (representative)
    n_values = sorted(set(
        entry["N"] for entry in methods[method_names[0]]["per_circuit"]
        if "N" in entry
    ))

    if not n_values:
        lines.append("  No per-size data available (N missing from per_circuit).")
        lines.append("")
        return lines

    # Header
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
            per_circuit = methods[method]["per_circuit"]
            ln_costs = [
                entry["hard_cost"] / entry["T"]
                for entry in per_circuit
                if entry.get("N") == n_val and entry.get("T", 0) > 0
            ]
            if ln_costs:
                cell = f"{np.mean(ln_costs):.4f}±{np.std(ln_costs):.4f}"
            else:
                cell = "N/A"
            row += f"  {cell:^{val_w}}"
        lines.append(row)

    lines.append("")
    return lines


def build_per_algo_table(
    methods: Dict[str, List[dict]],
    method_names: List[str],
) -> List[str]:
    """Build per-algorithm breakdown table for real/MQT mode."""
    lines = []
    lines.append("PER-ALGORITHM BREAKDOWN (layer-normalized hard cost)")
    lines.append("=" * 72)
    lines.append("")

    # Collect all unique algorithms from MOSAIC (representative)
    algo_set = sorted(set(
        entry["algo"] for entry in methods[method_names[0]]["per_circuit"]
        if "algo" in entry
    ))

    if not algo_set:
        lines.append("  No per-algorithm data available (algo missing from per_circuit).")
        lines.append("")
        return lines

    # Header
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
        count = 0
        for method in method_names:
            per_circuit = methods[method]["per_circuit"]
            ln_costs = [
                entry["hard_cost"] / entry["T"]
                for entry in per_circuit
                if entry.get("algo") == algo and entry.get("T", 0) > 0
            ]
            count = max(count, len(ln_costs))
            if ln_costs:
                cell = f"{np.mean(ln_costs):.4f}±{np.std(ln_costs):.4f}"
            else:
                cell = "N/A"
            row += f"  {cell:^{val_w}}"
        row += f"  {count:>6}"
        lines.append(row)

    lines.append("")
    return lines


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # ---- Load JSON ----
    with open(args.json_file) as f:
        data = json.load(f)

    methods = data.get("methods", {})
    if not methods:
        print(f"ERROR: no 'methods' key in {args.json_file}", file=sys.stderr)
        sys.exit(1)

    method_names = [m for m in ["MOSAIC", "B1", "B3", "B4", "B5"] if m in methods]
    if not method_names:
        print(f"ERROR: no recognized methods in JSON", file=sys.stderr)
        sys.exit(1)

    # ---- Validate T exists ----
    sample = methods[method_names[0]]["per_circuit"][0]
    if "T" not in sample:
        print(f"ERROR: per_circuit entries are missing 'T'. "
              f"Re-run eval_scheduler_v1/v2 with the updated version that "
              f"preserves T in the JSON.", file=sys.stderr)
        sys.exit(1)

    # ---- Build output lines ----
    lines = []

    # Metadata header
    lines.append(f"Source: {os.path.basename(args.json_file)}")
    lines.append(f"Mode:   {args.mode}")
    if "number_of_qubits" in data:
        if data.get("is_range"):
            qr = data.get("qubit_range", [data["number_of_qubits"], "?"])
            lines.append(f"Qubits: {qr[0]} – {qr[1]} (range)")
        else:
            lines.append(f"Qubits: {data['number_of_qubits']}")
    n_circuits = data.get("n_circuits", len(methods[method_names[0]]["per_circuit"]))
    lines.append(f"Circuits: {n_circuits}")
    lines.append("")

    # Aggregate table (always)
    lines += build_aggregate_table(methods, method_names)

    # Mode-specific breakdown
    if args.mode == "synthetic":
        is_range = data.get("is_range", False)
        if is_range:
            lines += build_per_n_table(methods, method_names)
    elif args.mode == "real":
        lines += build_per_algo_table(methods, method_names)

    # ---- Save ----
    stem = os.path.splitext(args.json_file)[0]
    out_path = f"{stem}_LN_summary.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    # Also print to console
    for line in lines:
        print(line)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
