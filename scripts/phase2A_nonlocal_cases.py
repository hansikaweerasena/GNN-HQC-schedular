#!/usr/bin/env python3
from __future__ import annotations

"""
Phase 2.2 / Phase 2A non-local classification harness.

What this script does
---------------------
1. Reuses the Phase 1B dense motifs and adds a few bridge / cross-community
   positive motifs for non-local analysis.
2. Builds a longer-horizon flat bidirectional window graph (default radius=6)
   for structural non-local classification.
3. Computes, for every 2Q gate in every motif:
      - pair-fraction edge betweenness
      - neighborhood dissimilarity
      - conservative non-local classifier decision
   using the finalized Phase 2.1 definitions.
4. Produces case-wise logs, CSV tables, and visualizations to support threshold
   selection and false-positive inspection.

Non-local classification used here
----------------------------------
Let G_nl(l) be the longer-horizon effective graph around layer l.
For target edge (u,v), we compute:

    B_btw_pairfrac = raw_edge_betweenness(u,v) / (|V| choose 2)
    D_nbr          = 1 - weighted_jaccard(N_u without {v}, N_v without {u})
    I_nl           = 1[ B_btw_pairfrac >= tau_btw AND D_nbr >= tau_nbr ]

This script is focused on the classifier metrics and their threshold behavior.
The final non-local score magnitude (detour-based Gamma_nonlocal) is left for a
later phase.
"""

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Import Phase 1B helpers / motifs so the cases stay consistent with user code.
# -----------------------------------------------------------------------------

P1B_PATH = Path("scripts/phase1B_dense_cases.py")
if not P1B_PATH.exists():
    raise FileNotFoundError(f"Expected Phase 1B script at {P1B_PATH}")

spec = importlib.util.spec_from_file_location("phase1b_dense", P1B_PATH)
phase1b = importlib.util.module_from_spec(spec)
sys.modules["phase1b_dense"] = phase1b
assert spec.loader is not None
spec.loader.exec_module(phase1b)

LayerSpec = phase1b.LayerSpec
MotifSpec = phase1b.MotifSpec
build_quantum_circuit = phase1b.build_quantum_circuit
layer_edge_counts = phase1b.layer_edge_counts
plot_circuit_timeline = phase1b.plot_circuit_timeline
plot_pair_layer_heatmap = phase1b.plot_pair_layer_heatmap
plot_graphs_by_layer = phase1b.plot_graphs_by_layer
write_text = phase1b.write_text
_sorted_pair = phase1b._sorted_pair
_ensure_dir = phase1b._ensure_dir


# -----------------------------------------------------------------------------
# Config / motif extensions
# -----------------------------------------------------------------------------


@dataclass
class NonlocalCaseConfig:
    window_radius_nl: int = 6
    window_weights_nl: Optional[List[float]] = None  # flat by default
    window_normalize: bool = False

    tau_btw: float = 0.12
    tau_nbr: float = 0.75
    eps: float = 1e-12

    outdir: str = "calibration/phase2A_nonlocal_cases"
    annotate_heatmaps: bool = True
    save_qiskit_text: bool = True


class NonlocalMotifFactory:
    def __init__(self) -> None:
        self._p1 = phase1b.MotifFactory()

    def all_names(self) -> List[str]:
        return self._p1.all_names() + [
            "true_bridge",
            "cross_community",
            "shortcut_bridge",
        ]

    def build(self, name: str) -> MotifSpec:
        name_l = name.strip().lower()
        if name_l in {n.lower() for n in self._p1.all_names()}:
            return self._p1.build(name_l)
        if name_l == "true_bridge":
            return self.true_bridge()
        if name_l == "cross_community":
            return self.cross_community()
        if name_l == "shortcut_bridge":
            return self.shortcut_bridge()
        raise ValueError(f"Unknown motif: {name}")

    def true_bridge(self) -> MotifSpec:
        # Two repeated local regions connected by one target bridge.
        layers = [
            LayerSpec(twoq=[(0, 1), (1, 2), (0, 2)], label="tb_left_cluster_0"),
            LayerSpec(twoq=[(4, 5), (5, 6), (4, 6)], label="tb_right_cluster_1"),
            LayerSpec(twoq=[(0, 1), (4, 5)], label="tb_local_repeat_2"),
            LayerSpec(twoq=[(2, 4)], label="tb_target_bridge_3"),
            LayerSpec(twoq=[(1, 2), (5, 6)], label="tb_local_repeat_4"),
            LayerSpec(twoq=[(0, 2), (4, 6)], label="tb_local_repeat_5"),
        ]
        return MotifSpec(
            name="true_bridge",
            num_qubits=7,
            layers=layers,
            target_layer=3,
            target_pair=(2, 4),
            notes="Positive non-local motif: strict bridge between two local communities.",
        )

    def cross_community(self) -> MotifSpec:
        # Cross-community edge with an alternate cross-link, so not a strict bridge.
        layers = [
            LayerSpec(twoq=[(0, 1), (1, 2), (0, 2)], label="cc_left_cluster_0"),
            LayerSpec(twoq=[(4, 5), (5, 6), (4, 6)], label="cc_right_cluster_1"),
            LayerSpec(twoq=[(1, 4)], label="cc_alt_cross_2"),
            LayerSpec(twoq=[(2, 5)], label="cc_target_cross_3"),
            LayerSpec(twoq=[(0, 2), (4, 6)], label="cc_local_repeat_4"),
            LayerSpec(twoq=[(1, 2), (5, 6)], label="cc_local_repeat_5"),
        ]
        return MotifSpec(
            name="cross_community",
            num_qubits=7,
            layers=layers,
            target_layer=3,
            target_pair=(2, 5),
            notes="Positive non-local motif: cross-community edge with moderate alternate connectivity.",
        )

    def shortcut_bridge(self) -> MotifSpec:
        # A local chain with a target shortcut edge linking separated sub-regions.
        layers = [
            LayerSpec(twoq=[(0, 1), (2, 3), (4, 5)], label="sb_even_0"),
            LayerSpec(twoq=[(1, 2), (3, 4)], label="sb_odd_1"),
            LayerSpec(twoq=[(2, 5)], label="sb_target_shortcut_2"),
            LayerSpec(twoq=[(1, 2), (3, 4)], label="sb_odd_3"),
            LayerSpec(twoq=[(0, 1), (2, 3), (4, 5)], label="sb_even_4"),
        ]
        return MotifSpec(
            name="shortcut_bridge",
            num_qubits=6,
            layers=layers,
            target_layer=2,
            target_pair=(2, 5),
            notes="Positive non-local motif: shortcut-like edge across an otherwise local chain backbone.",
        )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _flat_window_weights(radius: int) -> List[float]:
    return [1.0] * (2 * int(radius) + 1)


def build_window_effective_graphs(
    edge_counts_per_layer: Sequence[Dict[Tuple[int, int], float]],
    radius: int,
    weights: Optional[Sequence[float]],
    normalize: bool,
) -> List[Dict[Tuple[int, int], float]]:
    if weights is None:
        weights = _flat_window_weights(radius)
    if len(weights) != 2 * radius + 1:
        raise ValueError("window weights must have length 2*radius+1")

    out: List[Dict[Tuple[int, int], float]] = []
    n_layers = len(edge_counts_per_layer)
    for s in range(n_layers):
        acc: Dict[Tuple[int, int], float] = defaultdict(float)
        total_w = 0.0
        for off in range(-radius, radius + 1):
            j = s + off
            if j < 0 or j >= n_layers:
                continue
            w = float(weights[off + radius])
            total_w += w
            for pair, val in edge_counts_per_layer[j].items():
                acc[pair] += w * float(val)
        if normalize and total_w > 0.0:
            acc = defaultdict(float, {p: v / total_w for p, v in acc.items()})
        out.append(dict(acc))
    return out


def effective_to_graph(eff: Dict[Tuple[int, int], float], num_qubits: int, weighted: bool = True) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(num_qubits))
    for (u, v), w in eff.items():
        if weighted:
            g.add_edge(u, v, weight=float(w))
        else:
            g.add_edge(u, v)
    return g


def edge_betweenness_pair_fraction(eff: Dict[Tuple[int, int], float], num_qubits: int, eps: float) -> Dict[Tuple[int, int], float]:
    g = effective_to_graph(eff, num_qubits, weighted=False)
    if g.number_of_edges() == 0:
        return {}
    raw = nx.edge_betweenness_centrality(g, normalized=False)
    n = g.number_of_nodes()
    denom = max(1.0, math.comb(n, 2)) + eps
    return {_sorted_pair(u, v): float(val) / denom for (u, v), val in raw.items()}


def weighted_neighbor_profile_from_effective(
    eff: Dict[Tuple[int, int], float],
    endpoint: int,
    exclude_neighbor: Optional[int] = None,
) -> Dict[int, float]:
    out: Dict[int, float] = defaultdict(float)
    for (u, v), w in eff.items():
        if u == endpoint and v != exclude_neighbor:
            out[v] += float(w)
        elif v == endpoint and u != exclude_neighbor:
            out[u] += float(w)
    return dict(out)


def _weighted_jaccard(a: Dict[int, float], b: Dict[int, float], eps: float) -> float:
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 0.0
    num = sum(min(float(a.get(k, 0.0)), float(b.get(k, 0.0))) for k in keys)
    den = sum(max(float(a.get(k, 0.0)), float(b.get(k, 0.0))) for k in keys) + eps
    return float(num / den)


def neighborhood_dissimilarity(eff: Dict[Tuple[int, int], float], pair: Tuple[int, int], eps: float) -> float:
    u, v = pair
    N_u = weighted_neighbor_profile_from_effective(eff, u, exclude_neighbor=v)
    N_v = weighted_neighbor_profile_from_effective(eff, v, exclude_neighbor=u)
    j = _weighted_jaccard(N_u, N_v, eps=eps)
    return float(1.0 - j)


def compute_nonlocal_gate_rows(
    motif: MotifSpec,
    edge_counts_per_layer: Sequence[Dict[Tuple[int, int], float]],
    effective_graphs: Sequence[Dict[Tuple[int, int], float]],
    cfg: NonlocalCaseConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for s, layer in enumerate(motif.layers):
        eff = effective_graphs[s]
        btw_map = edge_betweenness_pair_fraction(eff, motif.num_qubits, cfg.eps)
        ordered_pairs = sorted((_sorted_pair(u, v) for (u, v) in layer.twoq), key=lambda p: (p[0], p[1]))
        for gate_idx, pair in enumerate(ordered_pairs, start=1):
            u, v = pair
            btw_pf = float(btw_map.get(pair, 0.0))
            d_nbr = float(neighborhood_dissimilarity(eff, pair, cfg.eps))
            is_nonlocal = bool((btw_pf >= cfg.tau_btw) and (d_nbr >= cfg.tau_nbr))
            is_target = bool(motif.target_layer == s and motif.target_pair is not None and _sorted_pair(*motif.target_pair) == pair)
            rows.append({
                "motif": motif.name,
                "layer": int(s),
                "layer_label": layer.label,
                "gate_id": int(gate_idx),
                "gate_label": f"L{s}_G{gate_idx}:{u}-{v}",
                "u": int(u),
                "v": int(v),
                "pair": f"({u},{v})",
                "B_btw_pairfrac": float(btw_pf),
                "D_nbr": float(d_nbr),
                "tau_btw": float(cfg.tau_btw),
                "tau_nbr": float(cfg.tau_nbr),
                "I_nonlocal": int(is_nonlocal),
                "is_target": bool(is_target),
            })
    return pd.DataFrame(rows)


def summarize_motif(df: pd.DataFrame, motif: MotifSpec, cfg: NonlocalCaseConfig) -> Dict[str, Any]:
    if df.empty:
        return {"motif": motif.name, "num_layers": 0, "num_gates": 0}
    target_df = df[df["is_target"] == True]
    target_btw = float(target_df["B_btw_pairfrac"].iloc[0]) if not target_df.empty else float("nan")
    target_nbr = float(target_df["D_nbr"].iloc[0]) if not target_df.empty else float("nan")
    target_cls = int(target_df["I_nonlocal"].iloc[0]) if not target_df.empty else -1
    return {
        "motif": motif.name,
        "window_radius_nl": int(cfg.window_radius_nl),
        "tau_btw": float(cfg.tau_btw),
        "tau_nbr": float(cfg.tau_nbr),
        "num_layers": int(len(motif.layers)),
        "num_gates": int(len(df)),
        "num_classified_nonlocal": int(df["I_nonlocal"].sum()),
        "target_B_btw_pairfrac": float(target_btw),
        "target_D_nbr": float(target_nbr),
        "target_I_nonlocal": int(target_cls),
        "mean_B_btw_pairfrac": float(df["B_btw_pairfrac"].mean()),
        "mean_D_nbr": float(df["D_nbr"].mean()),
        "notes": motif.notes,
    }


def concise_gate_log(df: pd.DataFrame) -> str:
    cols = ["layer", "gate_label", "B_btw_pairfrac", "D_nbr", "I_nonlocal", "is_target"]
    out = df.loc[:, cols].copy()
    for c in ["B_btw_pairfrac", "D_nbr"]:
        out[c] = out[c].map(lambda x: f"{float(x):.4f}")
    return out.to_string(index=False)


def concise_summary_log(summary: Dict[str, Any]) -> str:
    keys = [
        "motif",
        "window_radius_nl",
        "tau_btw",
        "tau_nbr",
        "num_layers",
        "num_gates",
        "num_classified_nonlocal",
        "target_B_btw_pairfrac",
        "target_D_nbr",
        "target_I_nonlocal",
        "mean_B_btw_pairfrac",
        "mean_D_nbr",
    ]
    lines = []
    for k in keys:
        v = summary.get(k)
        if isinstance(v, float):
            lines.append(f"- {k}: {v:.6f}")
        else:
            lines.append(f"- {k}: {v}")
    lines.append(f"- notes: {summary.get('notes', '')}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def plot_metrics_heatmap(df: pd.DataFrame, outpath: Path, annotate: bool = True) -> None:
    if df.empty:
        return
    pivot_cols = ["B_btw_pairfrac", "D_nbr", "I_nonlocal"]
    plot_df = df[["gate_label"] + pivot_cols].copy().set_index("gate_label")
    arr = plot_df.to_numpy(dtype=float)
    fig_h = max(2.8, 0.42 * len(plot_df.index))
    fig, ax = plt.subplots(figsize=(8.2, fig_h))
    im = ax.imshow(arr, aspect="auto")
    ax.set_xticks(range(len(pivot_cols)))
    ax.set_xticklabels(pivot_cols, rotation=20, ha="right")
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index)
    ax.set_title("Non-local classification metrics by gate")
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.ax.set_ylabel("value", rotation=90)
    if annotate:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_gate_metric_bars(df: pd.DataFrame, motif: MotifSpec, outpath: Path) -> None:
    if df.empty:
        return
    labels = df["gate_label"].tolist()
    x = np.arange(len(labels))
    btw = df["B_btw_pairfrac"].to_numpy(dtype=float)
    nbr = df["D_nbr"].to_numpy(dtype=float)
    target_mask = df["is_target"].to_numpy(dtype=bool)

    fig, ax = plt.subplots(figsize=(max(8.8, 0.55 * len(labels)), 4.4))
    w = 0.38
    ax.bar(x - w/2, btw, width=w, label="B_btw_pairfrac")
    ax.bar(x + w/2, nbr, width=w, label="D_nbr")
    for i, is_t in enumerate(target_mask):
        if is_t:
            ax.axvspan(i - 0.55, i + 0.55, color="gold", alpha=0.18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("metric value")
    ax.set_title(f"Per-gate non-local metrics: {motif.name}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_summary_bar(summary_df: pd.DataFrame, outpath: Path, value_col: str, title: str) -> None:
    if summary_df.empty:
        return
    sdf = summary_df.sort_values(value_col, ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(max(8.0, 0.75 * len(sdf)), 4.6))
    xpos = np.arange(len(sdf))
    ax.bar(xpos, sdf[value_col])
    ax.set_xticks(xpos)
    ax.set_xticklabels(sdf["motif"], rotation=45, ha="right")
    ax.set_ylabel(value_col)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_target_metric_scatter(summary_df: pd.DataFrame, outpath: Path, tau_btw: float, tau_nbr: float) -> None:
    if summary_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.8, 5.5))
    x = summary_df["target_B_btw_pairfrac"].to_numpy(dtype=float)
    y = summary_df["target_D_nbr"].to_numpy(dtype=float)
    colors = summary_df["target_I_nonlocal"].map(lambda z: "tab:red" if int(z) == 1 else "tab:blue").tolist()
    ax.scatter(x, y, s=55, c=colors)
    for _, row in summary_df.iterrows():
        ax.annotate(str(row["motif"]), (float(row["target_B_btw_pairfrac"]), float(row["target_D_nbr"])), fontsize=8, xytext=(4, 2), textcoords="offset points")
    ax.axvline(float(tau_btw), linestyle="--", linewidth=1)
    ax.axhline(float(tau_nbr), linestyle="--", linewidth=1)
    ax.set_xlabel("target pair-fraction betweenness")
    ax.set_ylabel("target neighborhood dissimilarity")
    ax.set_title("Target-edge metric plane for non-local threshold selection")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_threshold_grid(summary_df: pd.DataFrame, outpath: Path) -> None:
    if summary_df.empty:
        return
    btw_vals = np.linspace(0.0, max(0.25, float(summary_df["target_B_btw_pairfrac"].max()) + 0.02), 11)
    nbr_vals = np.linspace(0.3, 1.0, 15)
    arr = np.zeros((len(nbr_vals), len(btw_vals)), dtype=float)
    x = summary_df["target_B_btw_pairfrac"].to_numpy(dtype=float)
    y = summary_df["target_D_nbr"].to_numpy(dtype=float)
    for i, tn in enumerate(nbr_vals):
        for j, tb in enumerate(btw_vals):
            arr[i, j] = float(np.sum((x >= tb) & (y >= tn)))
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    im = ax.imshow(arr, origin="lower", aspect="auto",
                   extent=[btw_vals[0], btw_vals[-1], nbr_vals[0], nbr_vals[-1]])
    ax.set_xlabel("tau_btw")
    ax.set_ylabel("tau_nbr")
    ax.set_title("How many target edges would be classified non-local?")
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.ax.set_ylabel("# target positives", rotation=90)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Run harness
# -----------------------------------------------------------------------------


def run_one_case(motif: MotifSpec, cfg: NonlocalCaseConfig, base_outdir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    case_dir = base_outdir / motif.name
    _ensure_dir(case_dir)

    if cfg.save_qiskit_text:
        qc = build_quantum_circuit(motif)
        if qc is not None:
            try:
                text_repr = str(qc.draw(output="text"))
            except Exception:
                text_repr = repr(qc)
            write_text(case_dir / "circuit_text.txt", text_repr)

    plot_circuit_timeline(motif, case_dir / "timeline.png")
    plot_pair_layer_heatmap(motif.layers, case_dir / "pair_layer_heatmap.png")

    edge_counts = layer_edge_counts(motif.layers)
    weights = cfg.window_weights_nl or _flat_window_weights(cfg.window_radius_nl)
    effective_graphs = build_window_effective_graphs(edge_counts, cfg.window_radius_nl, weights, normalize=cfg.window_normalize)
    current_graphs = edge_counts
    plot_graphs_by_layer(motif, current_graphs, effective_graphs, "nonlocal_window", case_dir / "graphs_by_layer.png")

    df = compute_nonlocal_gate_rows(motif, edge_counts, effective_graphs, cfg)
    df.to_csv(case_dir / "gate_metrics.csv", index=False)

    summary = summarize_motif(df, motif, cfg)
    pd.DataFrame([summary]).to_csv(case_dir / "summary.csv", index=False)

    plot_metrics_heatmap(df, case_dir / "metrics_heatmap.png", annotate=cfg.annotate_heatmaps)
    plot_gate_metric_bars(df, motif, case_dir / "gate_metric_bars.png")

    write_text(case_dir / "gate_log.txt", concise_gate_log(df))
    write_text(case_dir / "summary_log.txt", concise_summary_log(summary))
    write_text(case_dir / "notes.txt", motif.notes)
    return df, summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2A non-local classification harness")
    p.add_argument("--motif", type=str, default="all", help="Motif name or 'all'")
    p.add_argument("--window-radius-nl", type=int, default=6, help="Non-local flat window radius")
    p.add_argument("--tau-btw", type=float, default=0.12, help="Pair-fraction betweenness threshold")
    p.add_argument("--tau-nbr", type=float, default=0.75, help="Neighborhood dissimilarity threshold")
    p.add_argument("--outdir", type=str, default="/mnt/data/phase2A_nonlocal_cases_out", help="Output directory")
    p.add_argument("--no-annotate", action="store_true", help="Disable heatmap annotations")
    p.add_argument("--no-qiskit-text", action="store_true", help="Skip circuit text dump")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = NonlocalCaseConfig(
        window_radius_nl=int(args.window_radius_nl),
        tau_btw=float(args.tau_btw),
        tau_nbr=float(args.tau_nbr),
        annotate_heatmaps=not bool(args.no_annotate),
        save_qiskit_text=not bool(args.no_qiskit_text),
        outdir=str(args.outdir),
    )
    base_outdir = Path(cfg.outdir)
    _ensure_dir(base_outdir)

    factory = NonlocalMotifFactory()
    motif_names = factory.all_names() if str(args.motif).lower() == "all" else [args.motif]

    all_gate_dfs: List[pd.DataFrame] = []
    summaries: List[Dict[str, Any]] = []
    for name in motif_names:
        motif = factory.build(name)
        df, summary = run_one_case(motif, cfg, base_outdir)
        all_gate_dfs.append(df)
        summaries.append(summary)

    full_df = pd.concat(all_gate_dfs, ignore_index=True) if all_gate_dfs else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)
    full_df.to_csv(base_outdir / "all_gate_metrics.csv", index=False)
    summary_df.to_csv(base_outdir / "all_summaries.csv", index=False)

    plot_summary_bar(summary_df, base_outdir / "summary_target_btw_bar.png", "target_B_btw_pairfrac", "Target pair-fraction betweenness by motif")
    plot_summary_bar(summary_df, base_outdir / "summary_target_nbr_bar.png", "target_D_nbr", "Target neighborhood dissimilarity by motif")
    plot_summary_bar(summary_df, base_outdir / "summary_positive_count_bar.png", "num_classified_nonlocal", "Number of classified non-local gates by motif")
    plot_target_metric_scatter(summary_df, base_outdir / "target_metric_scatter.png", cfg.tau_btw, cfg.tau_nbr)
    plot_threshold_grid(summary_df, base_outdir / "target_threshold_grid.png")

    manifest = {
        "config": asdict(cfg),
        "motifs": motif_names,
        "outputs": {
            "all_gate_metrics_csv": str(base_outdir / "all_gate_metrics.csv"),
            "all_summaries_csv": str(base_outdir / "all_summaries.csv"),
            "target_metric_scatter": str(base_outdir / "target_metric_scatter.png"),
            "target_threshold_grid": str(base_outdir / "target_threshold_grid.png"),
        },
    }
    write_text(base_outdir / "manifest.json", json.dumps(manifest, indent=2))
    print(summary_df[[
        "motif",
        "target_B_btw_pairfrac",
        "target_D_nbr",
        "target_I_nonlocal",
        "num_classified_nonlocal",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
