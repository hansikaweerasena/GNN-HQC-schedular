from __future__ import annotations

"""
roi_gen_trend.py

Trend diagnostics for ROI-based circuit generation across increasing qubit sizes.

Compared groups:
  - op1
  - op2a
  - op2b
  - op3
  - uniform_mix  (each sample picks one of the four options uniformly)

For each qubit size, this script generates N samples per group at a fixed layer count,
computes structural metrics, proxy/gamma metrics, and saves trend plots + run_config.json.

Uses the same generator defaults as roi_gen_diagnostics.py.
"""

import argparse
import ast
import contextlib
import io
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import networkx as nx

from roi_gen_diagnostics import ExperimentConfig, OPTIONS, compute_metrics, extract_layers
from src.circuit_generation import generate_roi_composed_circuit, ROI_LIBRARY  # type: ignore
from src.cost_function import SegmentStatsExtractor  # type: ignore


UNIFORM_GROUP = "uniform_mix"
GROUPS = list(OPTIONS) + [UNIFORM_GROUP]

GAMMA_MODE_MAP = {
    "edge_density": "edge_density",
    "twoq_per_layer": "twoq_per_layer",
    "pair_degree_pressure": "pair_degree_pressure",
    "pair_congestion": "pair_congestion",
    "pair_betweenness": "pair_betweenness",
    "pair_hybrid": "pair_hybrid",
}

PAIR_HYBRID_WEIGHTS = {"a0": 0.0, "a1": 0.33, "a2": 0.33, "a3": 0.33}

# One figure per metric, no hard-coded colors.
BASE_METRICS = [
    ("avg_2q_distance", "avg_dist"),
    ("frac_long_range_k3", "frac_long_k3"),
    ("unique_2q_pairs", "unique_pairs"),
    ("graph_density", "graph_density"),
    ("n2_per_layer_mean", "n2_per_layer"),
    ("active_qubits_per_layer_mean", "active_counts"),
    ("idle_streak_p90", "idle_p90"),
    ("mean_temporal_jaccard_2q", "temporal_jaccard"),
]

EXTRA_METRICS = [
    ("mean_2q_degree_active", "mean_2q_degree_active"),
    ("max_2q_degree", "max_2q_degree"),
    ("degree_gini", "degree_gini"),
    ("degree_cv", "degree_cv"),
    ("largest_cc_frac", "largest_cc_frac"),
    ("edge_multiplicity_mean", "edge_mult_mean"),
    ("edge_multiplicity_max", "edge_mult_max"),
    ("n2_per_layer_per_qubit_mean", "n2_per_layer_per_qubit"),
    ("active_qubits_per_layer_frac_mean", "active_counts_per_qubit"),
]


# ------------------------------
# Helpers
# ------------------------------

def _safe_mean(vals: Sequence[float]) -> float:
    arr = np.asarray([float(v) for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _safe_std(vals: Sequence[float]) -> float:
    arr = np.asarray([float(v) for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.std(arr))


def _gini_nonnegative(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    if np.allclose(x, 0.0):
        return 0.0
    x = np.sort(np.maximum(x, 0.0))
    n = x.size
    s = float(np.sum(x))
    if s <= 0.0:
        return 0.0
    idx = np.arange(1, n + 1, dtype=float)
    g = float((2.0 * np.sum(idx * x)) / (n * s) - (n + 1) / n)
    return max(0.0, g)


def _capture_debug_generate(kwargs: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        qc = generate_roi_composed_circuit(debug=False, roi_debug_prints=True, **kwargs)
    text = buf.getvalue()
    meta: Dict[str, Any] = {
        "chosen_rois": [],
        "roi_counts": {},
        "roi_area": {},
    }
    for line in text.splitlines():
        line = line.strip()
        if "chosen_rois=" in line:
            try:
                part = line.split("chosen_rois=", 1)[1].strip()
                meta["chosen_rois"] = list(ast.literal_eval(part))
            except Exception:
                pass
        elif line.startswith("ROI counts:"):
            try:
                meta["roi_counts"] = dict(ast.literal_eval(line.split(":", 1)[1].strip()))
            except Exception:
                pass
        elif line.startswith("ROI area (volume):"):
            try:
                meta["roi_area"] = dict(ast.literal_eval(line.split(":", 1)[1].strip()))
            except Exception:
                pass
    return qc, meta


def _build_gen_kwargs(cfg: ExperimentConfig, *, num_qubits: int, option: str, seed: int, use_barriers: bool) -> Dict[str, Any]:
    return dict(
        num_qubits=num_qubits,
        num_layers=cfg.num_layers,
        option=option,
        n_rois=cfg.n_rois,
        twoq_to_oneq_ratio=cfg.twoq_to_oneq_ratio,
        idle_density=cfg.idle_density,
        p_bridge_boundary=cfg.p_bridge_boundary,
        p_bridge_interior=cfg.p_bridge_interior,
        noise_1q_prob=cfg.noise_1q_prob,
        noise_2q_prob=cfg.noise_2q_prob,
        measure_frac=cfg.measure_frac,
        min_block_w=cfg.min_block_w,
        max_block_w=cfg.max_block_w,
        min_block_h=cfg.min_block_h,
        max_block_h=cfg.max_block_h,
        n_long=cfg.n_long,
        long_w_min=cfg.long_w_min,
        long_w_max=cfg.long_w_max,
        n_tall=cfg.n_tall,
        tall_h_min=cfg.tall_h_min,
        tall_h_max=cfg.tall_h_max,
        use_barriers=use_barriers,
        seed=seed,
    )


def _sample_option_for_group(group: str, seed: int) -> str:
    if group != UNIFORM_GROUP:
        return group
    rng = np.random.RandomState(seed ^ 0x5A5A1234)
    return str(rng.choice(list(OPTIONS)))


def _segment_edge_counts(qc, prefer_barriers: bool) -> Tuple[List[Dict[Tuple[int, int], int]], List[int], List[int]]:
    layers = extract_layers(qc, prefer_barriers=prefer_barriers)
    edge_counts_per_layer: List[Dict[Tuple[int, int], int]] = []
    n2_per_layer: List[int] = []
    active_per_layer: List[int] = []
    for ops in layers:
        ec: Dict[Tuple[int, int], int] = defaultdict(int)
        active = set()
        n2 = 0
        for name, qs in ops:
            if name in {"barrier", "measure", "meas", "m", "reset"}:
                continue
            for q in qs:
                active.add(int(q))
            if len(qs) == 2:
                a, b = int(qs[0]), int(qs[1])
                if a > b:
                    a, b = b, a
                ec[(a, b)] += 1
                n2 += 1
        edge_counts_per_layer.append(dict(ec))
        n2_per_layer.append(n2)
        active_per_layer.append(len(active))
    return edge_counts_per_layer, n2_per_layer, active_per_layer


def _compute_gamma_means(qc, num_qubits: int, prefer_barriers: bool) -> Dict[str, float]:
    """
    Compute layer-wise gamma trends.

    For pair_* modes, this matches the intended cost-model behavior for layer-wise
    segmentation with EWMA history enabled:
      1) update the history-smoothed multigraph with the current layer,
      2) compute pair gamma on the history graph,
      3) summarize the current layer by the weighted mean gamma across its edges.

    For scalar modes, gamma is computed layer-wise on the current layer graph.
    """
    edge_counts_per_layer, _, _ = _segment_edge_counts(qc, prefer_barriers=prefer_barriers)
    out: Dict[str, float] = {}

    for public_name, internal_mode in GAMMA_MODE_MAP.items():
        extractor = SegmentStatsExtractor({
            "connectivity_proxy": {
                "mode": internal_mode,
                "eps": 1e-12,
                "hyb_weights": PAIR_HYBRID_WEIGHTS,
                "temporal_graph": {
                    "mode": "window",

                    "history": {
                        "enabled": True,
                        "alpha": 0.85,
                        "cutoff": 0.25
                    },

                    "window": {
                        "enabled": True,
                        "radius": 2,
                        "decay": 0.6,
                        "normalize": False
                    }
                    }
            }
        })

        vals: List[float] = []
        hist_edge_counts: Dict[Tuple[int, int], float] = {}
        is_pair_mode = str(internal_mode).lower().startswith("pair_")

        for ec in edge_counts_per_layer:
            if is_pair_mode:
                extractor._ewma_update_and_prune(hist_edge_counts, ec)  # type: ignore[attr-defined]
                gamma_hist, gamma_map = extractor._compute_gamma_value(  # type: ignore[attr-defined]
                    N=num_qubits,
                    L_s=1,
                    edge_counts=hist_edge_counts,
                )
                if gamma_map is not None and len(ec) > 0:
                    gamma_cur = extractor._weighted_mean_over_edges(ec, gamma_map)  # type: ignore[attr-defined]
                else:
                    gamma_cur = float(gamma_hist)
                vals.append(float(gamma_cur))
            else:
                gamma, _ = extractor._compute_gamma_value(N=num_qubits, L_s=1, edge_counts=ec)  # type: ignore[attr-defined]
                vals.append(float(gamma))

        out[f"gamma_{public_name}_mean"] = _safe_mean(vals)
        out[f"gamma_{public_name}_p95"] = float(np.percentile(vals, 95)) if len(vals) > 0 else float("nan")
    return out


def _extended_metrics(qc, *, num_qubits: int, prefer_barriers: bool) -> Dict[str, Any]:
    edge_counts_per_layer, n2_per_layer, active_per_layer = _segment_edge_counts(qc, prefer_barriers=prefer_barriers)

    pair_counter: Counter = Counter()
    for ec in edge_counts_per_layer:
        pair_counter.update(ec)

    unique_pairs = list(pair_counter.keys())
    multiplicities = np.asarray(list(pair_counter.values()), dtype=float) if pair_counter else np.asarray([], dtype=float)

    G = nx.Graph() if nx is not None else None
    if G is not None:
        G.add_nodes_from(range(num_qubits))
        for (a, b), w in pair_counter.items():
            G.add_edge(a, b, weight=int(w))

    degrees = np.zeros(num_qubits, dtype=float)
    if G is not None and G.number_of_edges() > 0:
        for node, deg in G.degree():
            degrees[int(node)] = float(deg)

    active_2q = degrees[degrees > 0]
    mean_deg = float(np.mean(active_2q)) if active_2q.size > 0 else 0.0
    max_deg = float(np.max(active_2q)) if active_2q.size > 0 else 0.0
    degree_cv = float(np.std(active_2q) / (np.mean(active_2q) + 1e-12)) if active_2q.size > 0 else 0.0
    degree_gini = _gini_nonnegative(active_2q) if active_2q.size > 0 else 0.0

    largest_cc_frac = 0.0
    if G is not None and G.number_of_edges() > 0:
        largest_cc = max((len(c) for c in nx.connected_components(G)), default=0)
        largest_cc_frac = float(largest_cc / max(1, num_qubits))

    return {
        "mean_2q_degree_active": mean_deg,
        "max_2q_degree": max_deg,
        "degree_gini": degree_gini,
        "degree_cv": degree_cv,
        "largest_cc_frac": largest_cc_frac,
        "edge_multiplicity_mean": float(np.mean(multiplicities)) if multiplicities.size > 0 else 0.0,
        "edge_multiplicity_max": float(np.max(multiplicities)) if multiplicities.size > 0 else 0.0,
        "n2_per_layer_per_qubit_mean": float(np.mean(n2_per_layer) / max(1, num_qubits)) if len(n2_per_layer) > 0 else 0.0,
        "active_qubits_per_layer_frac_mean": float(np.mean(active_per_layer) / max(1, num_qubits)) if len(active_per_layer) > 0 else 0.0,
    }


def _aggregate_roi_meta(meta_rows: List[Dict[str, Any]], sizes: Sequence[int]) -> Dict[str, Dict[int, Dict[str, float]]]:
    out: Dict[str, Dict[int, Dict[str, float]]] = {}
    for size in sizes:
        sub = [r for r in meta_rows if int(r["num_qubits"]) == int(size)]
        total = len(sub)
        chosen_counts = Counter()
        rect_counts = Counter()
        area_counts = Counter()
        for r in sub:
            for roi in r.get("chosen_rois", []):
                chosen_counts[str(roi)] += 1
            for roi, c in (r.get("roi_counts") or {}).items():
                rect_counts[str(roi)] += float(c)
            for roi, a in (r.get("roi_area") or {}).items():
                area_counts[str(roi)] += float(a)
        out.setdefault("chosen_rate", {})[size] = {roi: float(chosen_counts.get(roi, 0) / max(1, total)) for roi in ROI_LIBRARY}
        out.setdefault("rect_fraction", {})[size] = {
            roi: float(rect_counts.get(roi, 0.0) / max(1.0, sum(rect_counts.values()))) for roi in ["idle", *ROI_LIBRARY]
        }
        out.setdefault("area_fraction", {})[size] = {
            roi: float(area_counts.get(roi, 0.0) / max(1.0, sum(area_counts.values()))) for roi in ["idle", *ROI_LIBRARY]
        }
    return out


def _save_trend_plot(rows: List[Dict[str, Any]], metric_key: str, metric_label: str, sizes: Sequence[int], outpath: Path) -> None:
    plt.figure(figsize=(10, 4.5))
    for group in GROUPS:
        ys = []
        yerr = []
        for size in sizes:
            vals = [float(r[metric_key]) for r in rows if r["group"] == group and int(r["num_qubits"]) == int(size) and not np.isnan(float(r[metric_key]))]
            ys.append(_safe_mean(vals))
            yerr.append(_safe_std(vals))
        plt.errorbar(list(sizes), ys, yerr=yerr, marker="o", capsize=3, label=group)
    plt.xlabel("num_qubits")
    plt.ylabel(metric_label)
    plt.title(f"{metric_label} vs num_qubits")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _save_gamma_panel(rows: List[Dict[str, Any]], sizes: Sequence[int], outdir: Path) -> None:
    for public_name in GAMMA_MODE_MAP.keys():
        key = f"gamma_{public_name}_mean"
        _save_trend_plot(rows, key, key, sizes, outdir / f"trend_{key}.png")
        p95_key = f"gamma_{public_name}_p95"
        _save_trend_plot(rows, p95_key, p95_key, sizes, outdir / f"trend_{p95_key}.png")


def _save_roi_heatmaps(meta_rows: List[Dict[str, Any]], sizes: Sequence[int], outdir: Path) -> None:
    for group in GROUPS:
        group_rows = [r for r in meta_rows if r["group"] == group]
        agg = _aggregate_roi_meta(group_rows, sizes)
        for view_name in ["chosen_rate", "rect_fraction", "area_fraction"]:
            rois = list(ROI_LIBRARY) if view_name == "chosen_rate" else ["idle", *ROI_LIBRARY]
            mat = np.zeros((len(rois), len(sizes)), dtype=float)
            for j, size in enumerate(sizes):
                table = agg.get(view_name, {}).get(size, {})
                for i, roi in enumerate(rois):
                    mat[i, j] = float(table.get(roi, 0.0))
            plt.figure(figsize=(11, max(5, 0.35 * len(rois))))
            im = plt.imshow(mat, aspect="auto")
            plt.colorbar(im)
            plt.xticks(range(len(sizes)), [str(s) for s in sizes])
            plt.yticks(range(len(rois)), rois)
            plt.xlabel("num_qubits")
            plt.ylabel("ROI")
            plt.title(f"{view_name} heatmap ({group})")
            plt.tight_layout()
            plt.savefig(outdir / f"heatmap_{group}_{view_name}.png", dpi=200)
            plt.close()


def _rows_to_summary(rows: List[Dict[str, Any]], sizes: Sequence[int], metric_keys: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for group in GROUPS:
        out[group] = {}
        for size in sizes:
            sub = [r for r in rows if r["group"] == group and int(r["num_qubits"]) == int(size)]
            out[group][str(size)] = {
                k: {
                    "mean": _safe_mean([float(r[k]) for r in sub]),
                    "std": _safe_std([float(r[k]) for r in sub]),
                }
                for k in metric_keys
            }
    return out


# ------------------------------
# Main
# ------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="roi_gen_trend")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed_base", type=int, default=0)
    ap.add_argument("--num_layers", type=int, default=80)
    ap.add_argument("--qubit_sizes", type=int, nargs="+", default=[15, 20, 25, 30, 35, 40, 45, 50])
    ap.add_argument("--use_barriers", action="store_true", default=True)
    ap.add_argument("--no_use_barriers", dest="use_barriers", action="store_false")
    args = ap.parse_args()

    cfg = ExperimentConfig(num_qubits=min(args.qubit_sizes), num_layers=args.num_layers, n=args.n, seed_base=args.seed_base)
    sizes = list(args.qubit_sizes)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    run_cfg = {
        "experiment_config_base": asdict(cfg),
        "groups": GROUPS,
        "gamma_modes": GAMMA_MODE_MAP,
        "num_layers": args.num_layers,
        "qubit_sizes": sizes,
        "n_per_group": args.n,
        "use_barriers": bool(args.use_barriers),
    }
    (outdir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    rows: List[Dict[str, Any]] = []
    meta_rows: List[Dict[str, Any]] = []

    total = len(sizes) * len(GROUPS) * args.n
    done = 0
    for s_i, num_qubits in enumerate(sizes):
        for g_i, group in enumerate(GROUPS):
            for i in range(args.n):
                seed = int(args.seed_base + s_i * 1_000_000 + g_i * 100_000 + i)
                option = _sample_option_for_group(group, seed)
                kwargs = _build_gen_kwargs(cfg, num_qubits=num_qubits, option=option, seed=seed, use_barriers=args.use_barriers)
                qc, meta = _capture_debug_generate(kwargs)

                base = compute_metrics(qc, num_qubits=num_qubits, prefer_barriers=args.use_barriers)
                extra = _extended_metrics(qc, num_qubits=num_qubits, prefer_barriers=args.use_barriers)
                gamma = _compute_gamma_means(qc, num_qubits=num_qubits, prefer_barriers=args.use_barriers)

                row = {
                    "group": group,
                    "option_used": option,
                    "num_qubits": num_qubits,
                    "seed": seed,
                    **base,
                    **extra,
                    **gamma,
                }
                rows.append(row)
                meta_rows.append({
                    "group": group,
                    "option_used": option,
                    "num_qubits": num_qubits,
                    "seed": seed,
                    **meta,
                })
                done += 1
                if done % 25 == 0 or done == total:
                    print(f"[{done}/{total}] generated")

    # Save raw summaries for later inspection.
    metric_keys = [k for k, _ in BASE_METRICS + EXTRA_METRICS] + [
        f"gamma_{name}_mean" for name in GAMMA_MODE_MAP.keys()
    ] + [
        f"gamma_{name}_p95" for name in GAMMA_MODE_MAP.keys()
    ]
    (outdir / "summary_stats.json").write_text(json.dumps(_rows_to_summary(rows, sizes, metric_keys), indent=2), encoding="utf-8")
    (outdir / "roi_meta_summary.json").write_text(json.dumps({
        group: _aggregate_roi_meta([r for r in meta_rows if r["group"] == group], sizes)
        for group in GROUPS
    }, indent=2), encoding="utf-8")

    # Trend plots.
    trend_dir = outdir / "trend_plots"
    trend_dir.mkdir(exist_ok=True)
    for metric_key, label in BASE_METRICS + EXTRA_METRICS:
        _save_trend_plot(rows, metric_key, label, sizes, trend_dir / f"trend_{metric_key}.png")

    gamma_dir = outdir / "gamma_trends"
    gamma_dir.mkdir(exist_ok=True)
    _save_gamma_panel(rows, sizes, gamma_dir)

    roi_dir = outdir / "roi_heatmaps"
    roi_dir.mkdir(exist_ok=True)
    _save_roi_heatmaps(meta_rows, sizes, roi_dir)

    print("Done. Outputs in:", outdir.resolve())


if __name__ == "__main__":
    main()
