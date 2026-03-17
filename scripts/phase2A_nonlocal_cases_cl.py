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
      - max-normalized edge betweenness
      - neighborhood dissimilarity
      - conservative non-local classifier decision
   using the finalized Phase 2.1 definitions.
4. Produces case-wise logs, CSV tables, and visualizations to support threshold
   selection and false-positive inspection.

Non-local classification used here (4-stage pipeline)
------------------------------------------------------
Let G_nl(l) be the longer-horizon effective graph around layer l.
For target edge (u,v):

  Stage 1  betweenness pre-filter
           B_btw = max-normalized edge betweenness in G_nl
           pass if B_btw >= tau_btw  (cheap O(VE) filter)

  Stage 2  community size guard
           remove (u,v) from G_nl; let C_u, C_v be the resulting
           connected components containing u and v respectively.
           pass if |C_u| >= delta_community AND |C_v| >= delta_community
           (kills false positives from small/pendant neighborhoods)

  Stage 3  pair-reuse guard
           count how many times (u,v) appears in a small local window
           (±pair_reuse_radius layers around current layer).
           fail if count >= pair_reuse_threshold
           (kills false positives from repeated nearest-neighbor edges
            that happen to sit at community boundaries)

  Stage 4  detour threshold
           L_detour = shortest path in G_nl without (u,v)
           pass if L_detour >= tau_detour
           (L_detour - 1 is a SWAP-count proxy for routing burden)

  I_nl = 1[ stage1 AND stage2 AND stage3 AND stage4 ]

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

    # Stage 1 pre-filter: max-normalized betweenness threshold.
    # Scale is [0, 1] where 1.0 = most bridge-like edge in G_nl.
    # 0.8 means "at least 80% as bridge-like as the worst bottleneck".
    # Aggressive pre-filter; stages 2/3 handle the remaining precision.
    tau_btw: float = 0.6

    # Stage 2 community size guard: both components after edge removal must be >= delta
    delta_community: int = 3

    # Stage 3 pair-reuse guard: if edge appears >= pair_reuse_threshold times
    # in a small local window (±pair_reuse_radius layers), it is a temporally
    # stable local interaction, not a long-range bridge.
    pair_reuse_radius: int = 2
    pair_reuse_threshold: int = 2

    # Stage 4 detour threshold: L_detour >= tau_detour to classify as non-local
    tau_detour: int = 3

    # Technology connectivity capacity (e.g. 3 for heavy-hex).
    # detour_cap is derived as kappa + 1.
    kappa: int = 3

    eps: float = 1e-12

    outdir: str = "calibration/phase2A_nonlocal_cases"
    annotate_heatmaps: bool = True
    save_qiskit_text: bool = True

    @property
    def detour_cap(self) -> int:
        return self.kappa + 1


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


def edge_betweenness_max_normalized(eff: Dict[Tuple[int, int], float], num_qubits: int, eps: float) -> Dict[Tuple[int, int], float]:
    """
    Stage 1 pre-filter: max-normalized edge betweenness.  Cheap O(VE) pass.

    Normalization: divide each edge's raw betweenness by the maximum across
    all edges in the effective graph.  This makes the score motif-agnostic:
      - The most bridge-like edge always scores 1.0.
      - All others score relative to it.
      - In a uniformly connected graph every edge scores ~1.0 and passes
        stage 1, but stages 2/3 will kill them.

    Uses networkx normalized=False to get raw shortest-path counts, then
    divides by max.
    """
    g = effective_to_graph(eff, num_qubits, weighted=False)
    if g.number_of_edges() == 0:
        return {}
    raw = nx.edge_betweenness_centrality(g, normalized=False)
    max_btw = max(raw.values())
    if max_btw < eps:
        return {_sorted_pair(u, v): 0.0 for (u, v) in raw}
    return {_sorted_pair(u, v): float(val) / max_btw for (u, v), val in raw.items()}


def _bfs_component_and_distance(
    g: nx.Graph,
    src: int,
    dst: int,
    exclude_edge: Tuple[int, int],
) -> Tuple[float, int]:
    """
    BFS in g with exclude_edge removed.
    Returns (shortest_path_distance, component_size_of_src).
    distance = math.inf if dst unreachable.
    component_size counts all nodes reachable from src (not through exclude_edge).
    """
    eu, ev = exclude_edge

    def _skip(a: int, b: int) -> bool:
        return (a == eu and b == ev) or (a == ev and b == eu)

    visited: Dict[int, int] = {src: 0}
    queue = defaultdict(list)
    queue[0].append(src)
    dist_to_dst = math.inf
    frontier = [src]
    bfs_queue = [(src, 0)]
    visited_set = {src}
    from collections import deque as _deque
    q = _deque([(src, 0)])
    while q:
        node, d = q.popleft()
        for nbr in g.neighbors(node):
            if _skip(node, nbr):
                continue
            if nbr not in visited_set:
                visited_set.add(nbr)
                q.append((nbr, d + 1))
                if nbr == dst:
                    dist_to_dst = d + 1
    return dist_to_dst, len(visited_set)


def detour_metrics(
    eff: Dict[Tuple[int, int], float],
    num_qubits: int,
    pair: Tuple[int, int],
) -> Tuple[float, int, int]:
    """
    Stage 2+3: compute L_detour, |C_u|, |C_v| for pair (u,v).

    Returns:
        l_detour        : shortest path in G_nl \\ {(u,v)}, math.inf if bridge
        component_size_u: # nodes reachable from u after removing (u,v)
        component_size_v: # nodes reachable from v after removing (u,v)
    """
    g = effective_to_graph(eff, num_qubits, weighted=False)
    u, v = pair
    if not g.has_edge(u, v):
        return math.inf, 1, 1
    l_detour, cu = _bfs_component_and_distance(g, u, v, exclude_edge=(u, v))
    _, cv          = _bfs_component_and_distance(g, v, u, exclude_edge=(u, v))
    return l_detour, cu, cv


def pair_reuse_count(
    edge_counts_per_layer: Sequence[Dict[Tuple[int, int], float]],
    layer_idx: int,
    pair: Tuple[int, int],
    radius: int,
) -> int:
    """
    Stage 3 pair-reuse guard helper.

    Count how many layers within ±radius of layer_idx contain pair (u,v).
    A count >= threshold means the edge is a temporally stable local
    interaction, not a one-shot long-range bridge.
    """
    n_layers = len(edge_counts_per_layer)
    count = 0
    for off in range(-radius, radius + 1):
        j = layer_idx + off
        if j < 0 or j >= n_layers:
            continue
        if pair in edge_counts_per_layer[j]:
            count += 1
    return count


def compute_nonlocal_gate_rows(
    motif: MotifSpec,
    edge_counts_per_layer: Sequence[Dict[Tuple[int, int], float]],
    effective_graphs: Sequence[Dict[Tuple[int, int], float]],
    cfg: NonlocalCaseConfig,
) -> pd.DataFrame:
    """
    Four-stage non-local classifier per gate:

    Stage 1  betweenness pre-filter   B_btw >= tau_btw
             (cheap O(VE); kills trivially local edges fast)
    Stage 2  community size guard     min(|C_u|, |C_v|) >= delta_community
             (kills false positives from small/pendant neighborhoods)
    Stage 3  pair-reuse guard         reuse_count < pair_reuse_threshold
             (kills false positives from repeated nearest-neighbor edges)
    Stage 4  detour threshold         L_detour >= tau_detour
             (hop count = SWAP proxy; provides both classification and score)

    A gate is non-local only if ALL FOUR stages pass.
    The stage at which it fails is recorded for inspection.
    """
    detour_cap = cfg.detour_cap  # kappa + 1

    rows: List[Dict[str, Any]] = []
    for s, layer in enumerate(motif.layers):
        eff = effective_graphs[s]

        # Stage 1: betweenness for all gates in this layer (one pass, shared)
        btw_map = edge_betweenness_max_normalized(eff, motif.num_qubits, cfg.eps)

        ordered_pairs = sorted(
            (_sorted_pair(u, v) for (u, v) in layer.twoq),
            key=lambda p: (p[0], p[1]),
        )
        for gate_idx, pair in enumerate(ordered_pairs, start=1):
            u, v = pair

            # --- Stage 1: betweenness pre-filter ---
            btw = float(btw_map.get(pair, 0.0))
            pass_btw = btw >= cfg.tau_btw

            # --- Stages 2-4 (only if previous stages pass — saves cost) ---
            l_detour: float = float("nan")
            cu: int = 0
            cv: int = 0
            reuse: int = 0
            pass_community = False
            pass_pair_reuse = False
            pass_detour = False

            if pass_btw:
                # Stage 2: community size guard
                l_raw, cu, cv = detour_metrics(eff, motif.num_qubits, pair)
                l_detour = float(detour_cap) if math.isinf(l_raw) else float(l_raw)
                pass_community = (cu >= cfg.delta_community) and (cv >= cfg.delta_community)

                if pass_community:
                    # Stage 3: pair-reuse guard
                    reuse = pair_reuse_count(
                        edge_counts_per_layer, s, pair, cfg.pair_reuse_radius,
                    )
                    pass_pair_reuse = reuse < cfg.pair_reuse_threshold

                    if pass_pair_reuse:
                        # Stage 4: detour threshold
                        pass_detour = l_raw >= cfg.tau_detour  # raw inf counts as pass

            is_nonlocal = pass_btw and pass_community and pass_pair_reuse and pass_detour

            # Which stage killed it (for diagnosis)
            if is_nonlocal:
                fail_stage = "none"
            elif not pass_btw:
                fail_stage = "btw"
            elif not pass_community:
                fail_stage = "community"
            elif not pass_pair_reuse:
                fail_stage = "pair_reuse"
            else:
                fail_stage = "detour"

            is_target = bool(
                motif.target_layer == s
                and motif.target_pair is not None
                and _sorted_pair(*motif.target_pair) == pair
            )
            rows.append({
                "motif":          motif.name,
                "layer":          int(s),
                "layer_label":    layer.label,
                "gate_id":        int(gate_idx),
                "gate_label":     f"L{s}_G{gate_idx}:{u}-{v}",
                "u":              int(u),
                "v":              int(v),
                "pair":           f"({u},{v})",
                # metrics
                "B_btw":          float(btw),
                "C_u":            int(cu),
                "C_v":            int(cv),
                "reuse_count":    int(reuse),
                "L_detour":       float(l_detour),
                # thresholds used
                "tau_btw":        float(cfg.tau_btw),
                "delta_community":int(cfg.delta_community),
                "pair_reuse_threshold": int(cfg.pair_reuse_threshold),
                "tau_detour":     int(cfg.tau_detour),
                "kappa":          int(cfg.kappa),
                # outcome
                "pass_btw":       bool(pass_btw),
                "pass_community": bool(pass_community),
                "pass_pair_reuse":bool(pass_pair_reuse),
                "pass_detour":    bool(pass_detour),
                "fail_stage":     str(fail_stage),
                "I_nonlocal":     int(is_nonlocal),
                "is_target":      bool(is_target),
            })
    return pd.DataFrame(rows)


def summarize_motif(df: pd.DataFrame, motif: MotifSpec, cfg: NonlocalCaseConfig) -> Dict[str, Any]:
    if df.empty:
        return {"motif": motif.name, "num_layers": 0, "num_gates": 0}
    target_df = df[df["is_target"] == True]
    def _tgt(col: str, default: Any) -> Any:
        return target_df[col].iloc[0] if not target_df.empty else default
    fail_counts = df["fail_stage"].value_counts().to_dict()
    return {
        "motif":                  motif.name,
        "window_radius_nl":       int(cfg.window_radius_nl),
        "tau_btw":                float(cfg.tau_btw),
        "delta_community":        int(cfg.delta_community),
        "pair_reuse_threshold":   int(cfg.pair_reuse_threshold),
        "tau_detour":             int(cfg.tau_detour),
        "kappa":                  int(cfg.kappa),
        "num_layers":             int(len(motif.layers)),
        "num_gates":              int(len(df)),
        "num_classified_nonlocal":int(df["I_nonlocal"].sum()),
        # target gate breakdown
        "target_B_btw":           float(_tgt("B_btw", float("nan"))),
        "target_C_u":             int(_tgt("C_u", -1)),
        "target_C_v":             int(_tgt("C_v", -1)),
        "target_reuse_count":     int(_tgt("reuse_count", 0)),
        "target_L_detour":        float(_tgt("L_detour", float("nan"))),
        "target_pass_btw":        bool(_tgt("pass_btw", False)),
        "target_pass_community":  bool(_tgt("pass_community", False)),
        "target_pass_pair_reuse": bool(_tgt("pass_pair_reuse", False)),
        "target_pass_detour":     bool(_tgt("pass_detour", False)),
        "target_I_nonlocal":      int(_tgt("I_nonlocal", -1)),
        "target_fail_stage":      str(_tgt("fail_stage", "n/a")),
        # population stats
        "fail_btw":               int(fail_counts.get("btw", 0)),
        "fail_community":         int(fail_counts.get("community", 0)),
        "fail_pair_reuse":        int(fail_counts.get("pair_reuse", 0)),
        "fail_detour":            int(fail_counts.get("detour", 0)),
        "notes":                  motif.notes,
    }


def concise_gate_log(df: pd.DataFrame) -> str:
    cols = ["layer", "gate_label", "B_btw", "C_u", "C_v", "reuse_count", "L_detour",
            "pass_btw", "pass_community", "pass_pair_reuse", "pass_detour", "fail_stage",
            "I_nonlocal", "is_target"]
    out = df.loc[:, cols].copy()
    out["B_btw"]    = out["B_btw"].map(lambda x: f"{float(x):.4f}")
    out["L_detour"] = out["L_detour"].map(lambda x: f"{float(x):.1f}" if not math.isnan(float(x)) else "—")
    return out.to_string(index=False)


def concise_summary_log(summary: Dict[str, Any]) -> str:
    keys = [
        "motif", "window_radius_nl",
        "tau_btw", "delta_community", "pair_reuse_threshold", "tau_detour", "kappa",
        "num_layers", "num_gates", "num_classified_nonlocal",
        "target_B_btw", "target_C_u", "target_C_v",
        "target_reuse_count", "target_L_detour",
        "target_pass_btw", "target_pass_community",
        "target_pass_pair_reuse", "target_pass_detour",
        "target_I_nonlocal", "target_fail_stage",
        "fail_btw", "fail_community", "fail_pair_reuse", "fail_detour",
    ]
    lines = []
    for k in keys:
        v = summary.get(k)
        lines.append(f"- {k}: {v:.6f}" if isinstance(v, float) else f"- {k}: {v}")
    lines.append(f"- notes: {summary.get('notes', '')}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def plot_metrics_heatmap(df: pd.DataFrame, outpath: Path, annotate: bool = True) -> None:
    if df.empty:
        return
    pivot_cols = ["B_btw", "C_u", "C_v", "reuse_count", "L_detour", "I_nonlocal"]
    plot_df = df[["gate_label"] + pivot_cols].copy().set_index("gate_label")
    arr = plot_df.to_numpy(dtype=float)
    fig_h = max(2.8, 0.42 * len(plot_df.index))
    fig, ax = plt.subplots(figsize=(10.0, fig_h))
    im = ax.imshow(arr, aspect="auto")
    ax.set_xticks(range(len(pivot_cols)))
    ax.set_xticklabels(pivot_cols, rotation=20, ha="right")
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index)
    ax.set_title("Non-local classification metrics by gate (4-stage pipeline)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.ax.set_ylabel("value", rotation=90)
    # Integer columns: C_u(1), C_v(2), reuse_count(3), I_nonlocal(5)
    int_cols = {1, 2, 3, 5}
    if annotate:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                txt = "—" if math.isnan(v) else (f"{v:.0f}" if j in int_cols else f"{v:.2f}")
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_gate_metric_bars(df: pd.DataFrame, motif: MotifSpec, outpath: Path) -> None:
    if df.empty:
        return
    labels      = df["gate_label"].tolist()
    x           = np.arange(len(labels))
    btw         = df["B_btw"].to_numpy(dtype=float)
    l_det       = df["L_detour"].to_numpy(dtype=float)
    cu          = df["C_u"].to_numpy(dtype=float)
    cv          = df["C_v"].to_numpy(dtype=float)
    reuse       = df["reuse_count"].to_numpy(dtype=float)
    target_mask = df["is_target"].to_numpy(dtype=bool)
    fail_stage  = df["fail_stage"].tolist()

    fig, axes = plt.subplots(4, 1, figsize=(max(9.0, 0.55 * len(labels)), 12.0), sharex=True)

    # Stage 1: betweenness
    ax = axes[0]
    ax.bar(x, btw, color=["tab:orange" if f == "none" else
                           "tab:blue"   if f == "btw" else "tab:gray"
                           for f in fail_stage])
    ax.axhline(df["tau_btw"].iloc[0], linestyle="--", linewidth=1, label=f"tau_btw={df['tau_btw'].iloc[0]:.3f}")
    ax.set_ylabel("B_btw (max-norm.)")
    ax.set_title(f"Stage 1 — betweenness pre-filter: {motif.name}")
    ax.legend(loc="upper right", fontsize=8)

    # Stage 2: community sizes
    ax = axes[1]
    w = 0.38
    ax.bar(x - w/2, cu, width=w, label="|C_u|")
    ax.bar(x + w/2, cv, width=w, label="|C_v|")
    delta = df["delta_community"].iloc[0]
    ax.axhline(delta, linestyle="--", linewidth=1, label=f"delta_community={delta}")
    ax.set_ylabel("component size")
    ax.set_title("Stage 2 — community size guard")
    ax.legend(loc="upper right", fontsize=8)

    # Stage 3: pair-reuse guard
    ax = axes[2]
    reuse_thresh = df["pair_reuse_threshold"].iloc[0]
    ax.bar(x, reuse, color=["tab:purple" if f == "pair_reuse" else
                             "tab:orange" if f == "none" else "tab:gray"
                             for f in fail_stage])
    ax.axhline(reuse_thresh, linestyle="--", linewidth=1, label=f"pair_reuse_threshold={reuse_thresh}")
    ax.set_ylabel("reuse count")
    ax.set_title("Stage 3 — pair-reuse guard (fail if ≥ threshold)")
    ax.legend(loc="upper right", fontsize=8)

    # Stage 4: detour
    ax = axes[3]
    tau_d = df["tau_detour"].iloc[0]
    colors = ["tab:red" if f == "none" else "tab:gray" for f in fail_stage]
    ax.bar(x, l_det, color=colors)
    ax.axhline(tau_d, linestyle="--", linewidth=1, label=f"tau_detour={tau_d}")
    ax.set_ylabel("L_detour (hops)")
    ax.set_title("Stage 4 — detour hop count")
    ax.legend(loc="upper right", fontsize=8)

    for ax in axes:
        for i, is_t in enumerate(target_mask):
            if is_t:
                ax.axvspan(i - 0.55, i + 0.55, color="gold", alpha=0.18)
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(labels, rotation=60, ha="right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_summary_bar(summary_df: pd.DataFrame, outpath: Path, value_col: str, title: str) -> None:
    if summary_df.empty or value_col not in summary_df.columns:
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


def plot_target_metric_scatter(summary_df: pd.DataFrame, outpath: Path,
                                tau_detour: int, delta_community: int) -> None:
    """
    Scatter: target L_detour (x) vs min(C_u, C_v) (y).
    Each motif is one point. Dashed lines show the classification thresholds.
    Points are coloured by final I_nonlocal.
    """
    if summary_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.8, 5.5))
    x = summary_df["target_L_detour"].to_numpy(dtype=float)
    y = np.minimum(
        summary_df["target_C_u"].to_numpy(dtype=float),
        summary_df["target_C_v"].to_numpy(dtype=float),
    )
    colors = summary_df["target_I_nonlocal"].map(
        lambda z: "tab:red" if int(z) == 1 else "tab:blue"
    ).tolist()
    ax.scatter(x, y, s=60, c=colors)
    for _, row in summary_df.iterrows():
        ax.annotate(str(row["motif"]),
                    (float(row["target_L_detour"]),
                     min(float(row["target_C_u"]), float(row["target_C_v"]))),
                    fontsize=8, xytext=(4, 2), textcoords="offset points")
    ax.axvline(float(tau_detour),    linestyle="--", linewidth=1, label=f"tau_detour={tau_detour}")
    ax.axhline(float(delta_community), linestyle=":",  linewidth=1, label=f"delta_community={delta_community}")
    ax.set_xlabel("target L_detour (hops)")
    ax.set_ylabel("target min(|C_u|, |C_v|)")
    ax.set_title("Target edge: detour vs community size")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_fail_stage_breakdown(summary_df: pd.DataFrame, outpath: Path) -> None:
    """Stacked bar: how many gates fell out at each stage, per motif."""
    if summary_df.empty:
        return
    cols = ["fail_btw", "fail_community", "fail_pair_reuse", "fail_detour", "num_classified_nonlocal"]
    present = [c for c in cols if c in summary_df.columns]
    if not present:
        return
    sdf = summary_df.set_index("motif")[present]
    fig, ax = plt.subplots(figsize=(max(8.0, 0.75 * len(sdf)), 4.6))
    sdf.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("gate count")
    ax.set_title("Gate funnel: gates eliminated at each stage")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_threshold_grid(summary_df: pd.DataFrame, outpath: Path) -> None:
    """
    Grid: how many target edges pass for each (tau_detour, delta_community) pair.
    """
    if summary_df.empty:
        return
    detour_vals = list(range(1, 9))
    community_vals = list(range(1, 8))
    l_det = summary_df["target_L_detour"].to_numpy(dtype=float)
    cu    = summary_df["target_C_u"].to_numpy(dtype=float)
    cv    = summary_df["target_C_v"].to_numpy(dtype=float)
    min_c = np.minimum(cu, cv)

    arr = np.zeros((len(community_vals), len(detour_vals)), dtype=float)
    for i, dc in enumerate(community_vals):
        for j, td in enumerate(detour_vals):
            arr[i, j] = float(np.sum((l_det >= td) & (min_c >= dc)))

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    im = ax.imshow(arr, origin="lower", aspect="auto",
                   extent=[detour_vals[0] - 0.5, detour_vals[-1] + 0.5,
                           community_vals[0] - 0.5, community_vals[-1] + 0.5])
    ax.set_xlabel("tau_detour")
    ax.set_ylabel("delta_community")
    ax.set_title("Target positives vs (tau_detour, delta_community)")
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
    # All numeric defaults are pulled from the dataclass — single source of truth.
    _defaults = NonlocalCaseConfig()
    p = argparse.ArgumentParser(description="Phase 2A non-local classification harness (4-stage pipeline)")
    p.add_argument("--motif", type=str, default="all", help="Motif name or 'all'")
    p.add_argument("--window-radius-nl", type=int, default=_defaults.window_radius_nl, help="Non-local flat window radius")
    p.add_argument("--tau-btw", type=float, default=_defaults.tau_btw, help="Stage 1: max-normalized betweenness pre-filter threshold")
    p.add_argument("--delta-community", type=int, default=_defaults.delta_community, help="Stage 2: minimum component size after edge removal")
    p.add_argument("--pair-reuse-radius", type=int, default=_defaults.pair_reuse_radius, help="Stage 3: local window radius for pair-reuse count")
    p.add_argument("--pair-reuse-threshold", type=int, default=_defaults.pair_reuse_threshold, help="Stage 3: fail if pair appears >= this many times in local window")
    p.add_argument("--tau-detour", type=int, default=_defaults.tau_detour, help="Stage 4: minimum L_detour hop count")
    p.add_argument("--kappa", type=int, default=_defaults.kappa, help="Technology connectivity capacity (detour_cap = kappa + 1)")
    p.add_argument("--outdir", type=str, default=_defaults.outdir, help="Output directory")
    p.add_argument("--no-annotate", action="store_true", help="Disable heatmap annotations")
    p.add_argument("--no-qiskit-text", action="store_true", help="Skip circuit text dump")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = NonlocalCaseConfig(
        window_radius_nl=int(args.window_radius_nl),
        tau_btw=float(args.tau_btw),
        delta_community=int(args.delta_community),
        pair_reuse_radius=int(args.pair_reuse_radius),
        pair_reuse_threshold=int(args.pair_reuse_threshold),
        tau_detour=int(args.tau_detour),
        kappa=int(args.kappa),
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

    plot_summary_bar(summary_df, base_outdir / "summary_target_detour_bar.png", "target_L_detour", "Target L_detour by motif")
    plot_summary_bar(summary_df, base_outdir / "summary_positive_count_bar.png", "num_classified_nonlocal", "Number of classified non-local gates by motif")
    plot_target_metric_scatter(summary_df, base_outdir / "target_metric_scatter.png", cfg.tau_detour, cfg.delta_community)
    plot_threshold_grid(summary_df, base_outdir / "target_threshold_grid.png")
    plot_fail_stage_breakdown(summary_df, base_outdir / "fail_stage_breakdown.png")

    manifest = {
        "config": {
            "window_radius_nl": cfg.window_radius_nl,
            "tau_btw": cfg.tau_btw,
            "delta_community": cfg.delta_community,
            "pair_reuse_radius": cfg.pair_reuse_radius,
            "pair_reuse_threshold": cfg.pair_reuse_threshold,
            "tau_detour": cfg.tau_detour,
            "kappa": cfg.kappa,
            "detour_cap": cfg.detour_cap,
        },
        "motifs": motif_names,
        "pipeline": {
            "stage1": f"max-norm betweenness >= {cfg.tau_btw}",
            "stage2": f"min(|C_u|, |C_v|) >= {cfg.delta_community}",
            "stage3": f"pair_reuse_count < {cfg.pair_reuse_threshold} (radius={cfg.pair_reuse_radius})",
            "stage4": f"L_detour >= {cfg.tau_detour} (cap={cfg.detour_cap}, kappa={cfg.kappa})",
        },
        "outputs": {
            "all_gate_metrics_csv": str(base_outdir / "all_gate_metrics.csv"),
            "all_summaries_csv":    str(base_outdir / "all_summaries.csv"),
            "target_metric_scatter": str(base_outdir / "target_metric_scatter.png"),
            "target_threshold_grid": str(base_outdir / "target_threshold_grid.png"),
            "fail_stage_breakdown":  str(base_outdir / "fail_stage_breakdown.png"),
        },
    }
    write_text(base_outdir / "manifest.json", json.dumps(manifest, indent=2))
    print(summary_df[[
        "motif",
        "target_B_btw",
        "target_C_u",
        "target_C_v",
        "target_reuse_count",
        "target_L_detour",
        "target_I_nonlocal",
        "target_fail_stage",
        "num_classified_nonlocal",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
