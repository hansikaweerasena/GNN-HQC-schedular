#!/usr/bin/env python3
from __future__ import annotations

"""
Phase 1A dense-proxy calibration harness.

What this script does
---------------------
1. Generates small deterministic 10-qubit motif circuits in explicit layers.
2. Builds the current-layer graph and the effective temporal graph used for scoring
   (EWMA or bidirectional window).
3. Computes per-2Q-gate dense metrics:
      C_raw, C_hat, J_u, J_v, S_struct, R_pair, D_dense, Phi_dense, Gamma_dense
4. Saves concise logs, CSV tables, and visualizations into a calibration directory.

Design notes
------------
- Self-contained by design: useful logic is copied/adapted instead of importing the
  broader project pipeline.
- Segmentation is always layer-based.
- J_u / J_v exclude the current layer and use the same one-sided radius and decay as
  the window graph configuration.
- C normalization uses max-normalization inside the effective graph of each layer.
"""

import argparse
import json
import math
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
try:
    from qiskit import QuantumCircuit  # type: ignore
except Exception:  # pragma: no cover
    QuantumCircuit = None  # type: ignore


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class DenseConfig:
    # Dense proxy knobs
    lambda_decay: float = 0.80
    alpha: float = 0.80
    beta: float = 0.65
    c_high: float = 0.90
    eta: float = 0.25

    # Temporal graph mode for C / graph-based quantities
    mode: str = "both"  # ewma | window | both

    # EWMA graph settings
    ewma_alpha: float = 0.85
    ewma_cutoff: float = 0.25

    # Window graph settings
    window_radius: int = 2
    window_decay: float = 0.60
    window_normalize: bool = False
    window_weights: Optional[List[float]] = None

    # Misc
    eps: float = 1e-12
    num_qubits: int = 10
    seed: int = 7
    outdir: str = "calibration/phase1A"
    save_qiskit_text: bool = True
    annotate_heatmaps: bool = True

    # These are kept explicit so it is obvious where to change them.
    # They intentionally mirror the one-sided window settings used by J_u/J_v.
    @property
    def temporal_profile_radius(self) -> int:
        return int(self.window_radius)

    @property
    def temporal_profile_decay(self) -> float:
        return float(self.window_decay)


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass
class LayerSpec:
    twoq: List[Tuple[int, int]] = field(default_factory=list)
    oneq: List[int] = field(default_factory=list)
    label: str = ""


@dataclass
class MotifSpec:
    name: str
    layers: List[LayerSpec]
    notes: str = ""


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def _sorted_pair(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _window_weight(delta: int, radius: int, decay: float, weights: Optional[Sequence[float]] = None) -> float:
    d = abs(int(delta))
    if weights is not None:
        expected = 2 * radius + 1
        if len(weights) == expected:
            idx = int(delta) + radius
            if 0 <= idx < expected:
                return float(weights[idx])
    return float(decay) ** d


def _one_side_weight(delta: int, decay: float) -> float:
    # delta is 1..H for past/future profiles
    return float(decay) ** max(0, int(delta) - 1)


def _weighted_jaccard(a: Dict[int, float], b: Dict[int, float], eps: float) -> float:
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 0.0
    num = sum(min(float(a.get(k, 0.0)), float(b.get(k, 0.0))) for k in keys)
    den = sum(max(float(a.get(k, 0.0)), float(b.get(k, 0.0))) for k in keys) + eps
    return float(num / den)


def _max_normalize_map(raw_map: Dict[Tuple[int, int], float], eps: float) -> Dict[Tuple[int, int], float]:
    if not raw_map:
        return {}
    max_val = max(float(v) for v in raw_map.values())
    if max_val <= eps:
        return {k: 0.0 for k in raw_map}
    return {k: float(v) / max_val for k, v in raw_map.items()}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Deterministic motif generators
# -----------------------------------------------------------------------------


class MotifFactory:
    def __init__(self, n: int, seed: int):
        self.n = int(n)
        self.seed = int(seed)
        self.rng = np.random.RandomState(seed)

    def build(self, motif: str) -> MotifSpec:
        motif = motif.strip().lower()
        if motif == "repeated_local_chain":
            return self._repeated_local_chain()
        if motif == "brickwork_dense":
            return self._brickwork_dense()
        if motif == "hotspot_fanout":
            return self._hotspot_fanout()
        if motif == "alternating_hotspots":
            return self._alternating_hotspots()
        if motif == "mixed_dense_ratio1":
            return self._mixed_dense_ratio1()
        if motif == "extreme_dense_local":
            return self._extreme_dense_local()
        raise ValueError(f"Unknown motif: {motif}")

    def all_names(self) -> List[str]:
        return [
            "repeated_local_chain",
            "brickwork_dense",
            "hotspot_fanout",
            "alternating_hotspots",
            "mixed_dense_ratio1",
            "extreme_dense_local",
        ]

    def _repeated_local_chain(self) -> MotifSpec:
        # Repeats a compact local chain in the same region to test reuse + structural continuity.
        start = min(max(0, self.seed % max(1, self.n - 5)), max(0, self.n - 6))
        L0 = LayerSpec(twoq=[(start + 0, start + 1), (start + 2, start + 3), (start + 4, start + 5)], label="chain_even_0")
        L1 = LayerSpec(twoq=[(start + 1, start + 2), (start + 3, start + 4)], label="chain_odd_1")
        L2 = LayerSpec(twoq=[(start + 0, start + 1), (start + 2, start + 3), (start + 4, start + 5)], label="chain_even_2")
        L3 = LayerSpec(twoq=[(start + 1, start + 2), (start + 3, start + 4)], label="chain_odd_3")
        L4 = LayerSpec(twoq=[(start + 0, start + 1), (start + 2, start + 3), (start + 4, start + 5)], label="chain_even_4")
        return MotifSpec(
            name="repeated_local_chain",
            layers=[L0, L1, L2, L3, L4],
            notes="Compact repeated local chain; should show continuity and repeated-pair discount in a stable local zone.",
        )

    def _brickwork_dense(self) -> MotifSpec:
        # Dense nearest-neighbor brickwork over almost the full 10-qubit line.
        even_pairs = [(q, q + 1) for q in range(0, self.n - 1, 2)]
        odd_pairs = [(q, q + 1) for q in range(1, self.n - 1, 2)]
        layers = [
            LayerSpec(twoq=list(even_pairs), label="brick_even_0"),
            LayerSpec(twoq=list(odd_pairs), label="brick_odd_1"),
            LayerSpec(twoq=list(even_pairs), label="brick_even_2"),
            LayerSpec(twoq=list(odd_pairs), label="brick_odd_3"),
            LayerSpec(twoq=list(even_pairs), label="brick_even_4"),
        ]
        return MotifSpec(
            name="brickwork_dense",
            layers=layers,
            notes="Alternating nearest-neighbor brickwork over most qubits; dense local overlap with strong temporal structure.",
        )

    def _hotspot_fanout(self) -> MotifSpec:
        # One persistent hub interacts with changing neighbors across layers.
        c = min(max(2, self.n // 2), self.n - 3)
        layers = [
            LayerSpec(twoq=[(c, c - 1), (0, 1), (self.n - 2, self.n - 1)], label="fanout_0"),
            LayerSpec(twoq=[(c, c + 1), (1, 2), (self.n - 3, self.n - 2)], label="fanout_1"),
            LayerSpec(twoq=[(c, c - 2), (0, 1), (self.n - 2, self.n - 1)], label="fanout_2"),
            LayerSpec(twoq=[(c, c + 2), (1, 2), (self.n - 3, self.n - 2)], label="fanout_3"),
            LayerSpec(twoq=[(c, c - 1), (0, 1), (self.n - 2, self.n - 1)], label="fanout_4"),
        ]
        return MotifSpec(
            name="hotspot_fanout",
            layers=layers,
            notes="One hub repeatedly fans out to different neighbors; should create high endpoint competition around the hub.",
        )

    def _alternating_hotspots(self) -> MotifSpec:
        # Two hubs alternate, which should keep density but reduce structural continuity at a fixed endpoint.
        c1 = max(2, self.n // 2 - 2)
        c2 = min(self.n - 3, self.n // 2 + 1)
        layers = [
            LayerSpec(twoq=[(c1, c1 - 1), (0, 1), (8, 9)], label="alt_hotspot_c1_0"),
            LayerSpec(twoq=[(c2, c2 + 1), (1, 2), (7, 8)], label="alt_hotspot_c2_1"),
            LayerSpec(twoq=[(c1, c1 + 1), (0, 1), (8, 9)], label="alt_hotspot_c1_2"),
            LayerSpec(twoq=[(c2, c2 - 1), (1, 2), (7, 8)], label="alt_hotspot_c2_3"),
            LayerSpec(twoq=[(c1, c1 - 1), (0, 1), (8, 9)], label="alt_hotspot_c1_4"),
        ]
        return MotifSpec(
            name="alternating_hotspots",
            layers=layers,
            notes="Hotspot role alternates between two hubs; useful for checking dense stress without perfectly stable structure.",
        )

    def _mixed_dense_ratio1(self) -> MotifSpec:
        # Equal counts of 1Q and 2Q gates per layer (3 each here).
        layers = [
            LayerSpec(twoq=[(2, 3), (4, 5), (6, 7)], oneq=[0, 1, 8], label="mixed_0"),
            LayerSpec(twoq=[(3, 4), (5, 6), (7, 8)], oneq=[0, 2, 9], label="mixed_1"),
            LayerSpec(twoq=[(2, 3), (4, 5), (6, 7)], oneq=[1, 8, 9], label="mixed_2"),
            LayerSpec(twoq=[(3, 4), (5, 6), (7, 8)], oneq=[0, 2, 9], label="mixed_3"),
            LayerSpec(twoq=[(2, 3), (4, 5), (6, 7)], oneq=[1, 8, 9], label="mixed_4"),
        ]
        return MotifSpec(
            name="mixed_dense_ratio1",
            layers=layers,
            notes="Dense local 2Q region with matching 1Q count per layer (1Q:2Q ratio = 1 by gate count).",
        )

    def _extreme_dense_local(self) -> MotifSpec:
        # Dense interactions concentrated in a 6-qubit window across several layers.
        layers = [
            LayerSpec(twoq=[(2, 3), (4, 5), (6, 7)], label="extreme_0"),
            LayerSpec(twoq=[(2, 4), (3, 6), (5, 7)], label="extreme_1"),
            LayerSpec(twoq=[(2, 5), (3, 7), (4, 6)], label="extreme_2"),
            LayerSpec(twoq=[(2, 6), (3, 5), (4, 7)], label="extreme_3"),
            LayerSpec(twoq=[(2, 7), (3, 4), (5, 6)], label="extreme_4"),
            LayerSpec(twoq=[(2, 3), (4, 5), (6, 7)], label="extreme_5"),
        ]
        return MotifSpec(
            name="extreme_dense_local",
            layers=layers,
            notes="Very dense repeated local interaction zone over a 6-qubit block; intended as the strongest dense-stress case.",
        )


# -----------------------------------------------------------------------------
# Circuit construction from LayerSpec
# -----------------------------------------------------------------------------


ONEQ_PATTERN = ("h", "x", "z", "s")
TWOQ_PATTERN = ("cx", "cz")


def build_quantum_circuit(motif: MotifSpec, n: int):
    if QuantumCircuit is None:
        return None
    qc = QuantumCircuit(n, name=motif.name)
    for layer_idx, layer in enumerate(motif.layers):
        for i, q in enumerate(layer.oneq):
            gate = ONEQ_PATTERN[(layer_idx + q + i) % len(ONEQ_PATTERN)]
            if gate == "h":
                qc.h(q)
            elif gate == "x":
                qc.x(q)
            elif gate == "z":
                qc.z(q)
            else:
                qc.s(q)

        for i, (u, v) in enumerate(layer.twoq):
            gate = TWOQ_PATTERN[(layer_idx + i) % len(TWOQ_PATTERN)]
            if gate == "cx":
                qc.cx(u, v)
            else:
                qc.cz(u, v)
        qc.barrier(*range(n))
    return qc


def layer_edge_counts(layers: Sequence[LayerSpec]) -> List[Dict[Tuple[int, int], float]]:
    out: List[Dict[Tuple[int, int], float]] = []
    for layer in layers:
        counts: Dict[Tuple[int, int], float] = defaultdict(float)
        for u, v in layer.twoq:
            counts[_sorted_pair(int(u), int(v))] += 1.0
        out.append(dict(counts))
    return out


# -----------------------------------------------------------------------------
# Effective temporal graphs (copied/adjusted from the broader codebase)
# -----------------------------------------------------------------------------


def ewma_update_and_prune(hist: Dict[Tuple[int, int], float], cur: Dict[Tuple[int, int], float], alpha: float, cutoff: float) -> None:
    if not hist and not cur:
        return
    a = min(max(float(alpha), 0.0), 1.0)
    for e in list(hist.keys()):
        hist[e] = float(hist[e]) * a
    for e, n in cur.items():
        hist[e] = float(hist.get(e, 0.0)) + float(n)
    if cutoff > 0.0:
        cur_keys = set(cur.keys())
        for e in list(hist.keys()):
            if e in cur_keys:
                continue
            if float(hist[e]) < cutoff:
                del hist[e]


def build_ewma_effective_graphs(edge_counts_per_layer: Sequence[Dict[Tuple[int, int], float]], alpha: float, cutoff: float) -> List[Dict[Tuple[int, int], float]]:
    hist: Dict[Tuple[int, int], float] = {}
    out: List[Dict[Tuple[int, int], float]] = []
    for cur in edge_counts_per_layer:
        ewma_update_and_prune(hist, cur, alpha=alpha, cutoff=cutoff)
        out.append(dict(hist))
    return out


def build_window_effective_graphs(
    edge_counts_per_layer: Sequence[Dict[Tuple[int, int], float]],
    radius: int,
    decay: float,
    normalize: bool,
    weights: Optional[Sequence[float]] = None,
) -> List[Dict[Tuple[int, int], float]]:
    radius = int(radius)
    out_all: List[Dict[Tuple[int, int], float]] = []
    n_layers = len(edge_counts_per_layer)
    for center_idx in range(n_layers):
        out: Dict[Tuple[int, int], float] = defaultdict(float)
        total_weight = 0.0
        start = max(0, center_idx - radius)
        end = min(n_layers - 1, center_idx + radius)
        for j in range(start, end + 1):
            delta = j - center_idx
            w_layer = _window_weight(delta, radius=radius, decay=decay, weights=weights)
            if w_layer <= 0.0:
                continue
            total_weight += w_layer
            for e, n in edge_counts_per_layer[j].items():
                out[e] += float(w_layer) * float(n)
        if normalize and total_weight > 0.0:
            for e in list(out.keys()):
                out[e] = float(out[e]) / total_weight
        out_all.append(dict(out))
    return out_all


# -----------------------------------------------------------------------------
# Dense metrics
# -----------------------------------------------------------------------------


def segment_nodes_and_deg(edge_counts: Dict[Tuple[int, int], float]) -> Tuple[Dict[int, float], List[int]]:
    deg_s: Dict[int, float] = defaultdict(float)
    nodes = set()
    for (a, b), n in edge_counts.items():
        nn = float(n)
        deg_s[a] += nn
        deg_s[b] += nn
        nodes.add(a)
        nodes.add(b)
    return dict(deg_s), sorted(nodes)


def congestion_raw_map(edge_counts: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    deg_s, _ = segment_nodes_and_deg(edge_counts)
    out: Dict[Tuple[int, int], float] = {}
    for (a, b), n_uv in edge_counts.items():
        nn = float(n_uv)
        out[(a, b)] = max(0.0, (deg_s.get(a, 0.0) - nn) + (deg_s.get(b, 0.0) - nn))
    return out


def weighted_neighbor_profile(
    edge_counts_per_layer: Sequence[Dict[Tuple[int, int], float]],
    layer_idx: int,
    endpoint: int,
    direction: str,
    radius: int,
    decay: float,
) -> Dict[int, float]:
    out: Dict[int, float] = defaultdict(float)
    n_layers = len(edge_counts_per_layer)
    for delta in range(1, int(radius) + 1):
        j = layer_idx - delta if direction == "past" else layer_idx + delta
        if j < 0 or j >= n_layers:
            continue
        w = _one_side_weight(delta, decay=decay)
        if w <= 0.0:
            continue
        for (a, b), n in edge_counts_per_layer[j].items():
            if a == endpoint:
                out[b] += float(w) * float(n)
            elif b == endpoint:
                out[a] += float(w) * float(n)
    return dict(out)


def pair_reuse_score(
    edge_counts_per_layer: Sequence[Dict[Tuple[int, int], float]],
    layer_idx: int,
    pair: Tuple[int, int],
    radius: int,
    lambda_decay: float,
) -> float:
    num = 0.0
    den = 0.0
    for delta in range(1, int(radius) + 1):
        j = layer_idx - delta
        if j < 0:
            continue
        w = float(lambda_decay) ** max(0, delta - 1)
        den += w
        if float(edge_counts_per_layer[j].get(pair, 0.0)) > 0.0:
            num += w
    return float(num / den) if den > 0.0 else 0.0


def compute_dense_gate_rows(
    motif_name: str,
    layers: Sequence[LayerSpec],
    edge_counts_per_layer: Sequence[Dict[Tuple[int, int], float]],
    effective_graphs: Sequence[Dict[Tuple[int, int], float]],
    mode_name: str,
    cfg: DenseConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    radius = int(cfg.temporal_profile_radius)
    decay = float(cfg.temporal_profile_decay)

    for s, layer in enumerate(layers):
        eff = effective_graphs[s]
        c_raw_map = congestion_raw_map(eff)
        c_hat_map = _max_normalize_map(c_raw_map, eps=cfg.eps)
        ordered_pairs = sorted((_sorted_pair(u, v) for (u, v) in layer.twoq), key=lambda p: (p[0], p[1]))

        for gate_idx, pair in enumerate(ordered_pairs, start=1):
            u, v = pair
            P_u = weighted_neighbor_profile(edge_counts_per_layer, s, u, direction="past", radius=radius, decay=decay)
            F_u = weighted_neighbor_profile(edge_counts_per_layer, s, u, direction="future", radius=radius, decay=decay)
            P_v = weighted_neighbor_profile(edge_counts_per_layer, s, v, direction="past", radius=radius, decay=decay)
            F_v = weighted_neighbor_profile(edge_counts_per_layer, s, v, direction="future", radius=radius, decay=decay)

            J_u = _weighted_jaccard(P_u, F_u, eps=cfg.eps)
            J_v = _weighted_jaccard(P_v, F_v, eps=cfg.eps)
            S_struct = min(J_u, J_v)
            R_pair = pair_reuse_score(edge_counts_per_layer, s, pair, radius=radius, lambda_decay=cfg.lambda_decay)

            D_dense = max(cfg.alpha * S_struct, cfg.beta * R_pair)
            D_dense = min(max(D_dense, 0.0), 1.0)

            C_raw = float(c_raw_map.get(pair, 0.0))
            C_hat = float(c_hat_map.get(pair, 0.0))
            retained = 1.0 - D_dense
            if C_hat > cfg.c_high:
                Phi_dense = max(float(cfg.eta), retained)
            else:
                Phi_dense = retained
            Phi_dense = min(max(Phi_dense, 0.0), 1.0)
            Gamma_dense = C_hat * Phi_dense

            rows.append({
                "motif": motif_name,
                "mode": mode_name,
                "layer": int(s),
                "layer_label": layer.label,
                "gate_id": int(gate_idx),
                "gate_label": f"L{s}_G{gate_idx}:{pair[0]}-{pair[1]}",
                "u": int(u),
                "v": int(v),
                "pair": f"({u},{v})",
                "C_raw": float(C_raw),
                "C_hat": float(C_hat),
                "J_u": float(J_u),
                "J_v": float(J_v),
                "S_struct": float(S_struct),
                "R_pair": float(R_pair),
                "D_dense": float(D_dense),
                "Phi_dense": float(Phi_dense),
                "Gamma_dense": float(Gamma_dense),
            })
    return pd.DataFrame(rows)


def summarize_motif(df: pd.DataFrame, motif_name: str, mode_name: str) -> Dict[str, Any]:
    if df.empty:
        return {
            "motif": motif_name,
            "mode": mode_name,
            "num_layers": 0,
            "num_gates": 0,
            "mean_Gamma_dense": float("nan"),
        }

    return {
        "motif": motif_name,
        "mode": mode_name,
        "num_layers": int(df["layer"].nunique()),
        "num_gates": int(len(df)),
        "mean_C_raw": float(df["C_raw"].mean()),
        "mean_C_hat": float(df["C_hat"].mean()),
        "mean_J_u": float(df["J_u"].mean()),
        "mean_J_v": float(df["J_v"].mean()),
        "mean_S_struct": float(df["S_struct"].mean()),
        "mean_R_pair": float(df["R_pair"].mean()),
        "mean_D_dense": float(df["D_dense"].mean()),
        "mean_Phi_dense": float(df["Phi_dense"].mean()),
        "mean_Gamma_dense": float(df["Gamma_dense"].mean()),
        "max_Gamma_dense": float(df["Gamma_dense"].max()),
        "p90_Gamma_dense": float(np.percentile(df["Gamma_dense"], 90)),
    }


# -----------------------------------------------------------------------------
# Visualizations
# -----------------------------------------------------------------------------


def plot_circuit_timeline(motif: MotifSpec, n: int, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(motif.layers) * 1.5), max(5, n * 0.55)))
    x_positions = np.arange(len(motif.layers), dtype=float)

    for q in range(n):
        ax.hlines(q, -0.5, len(motif.layers) - 0.5, color="lightgray", linewidth=1.0)

    for s, layer in enumerate(motif.layers):
        for q in layer.oneq:
            ax.scatter([s], [q], marker="s", s=120, zorder=3)
        for u, v in layer.twoq:
            a, b = _sorted_pair(u, v)
            ax.vlines(s, a, b, linewidth=2.5, zorder=2)
            ax.scatter([s, s], [a, b], marker="o", s=60, zorder=4)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"L{s}" for s in range(len(motif.layers))])
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"q{q}" for q in range(n)])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Qubit")
    ax.set_title(f"Circuit timeline: {motif.name}")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pair_layer_heatmap(layers: Sequence[LayerSpec], outpath: Path) -> None:
    edge_counts_per_layer = layer_edge_counts(layers)
    pairs = sorted({e for ec in edge_counts_per_layer for e in ec.keys()})
    if not pairs:
        return
    mat = np.zeros((len(pairs), len(edge_counts_per_layer)), dtype=float)
    for j, ec in enumerate(edge_counts_per_layer):
        for i, pair in enumerate(pairs):
            mat[i, j] = float(ec.get(pair, 0.0))

    fig, ax = plt.subplots(figsize=(max(7, len(edge_counts_per_layer) * 1.4), max(5, len(pairs) * 0.45)))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(range(len(edge_counts_per_layer)))
    ax.set_xticklabels([f"L{i}" for i in range(len(edge_counts_per_layer))])
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([f"{p[0]}-{p[1]}" for p in pairs])
    ax.set_xlabel("Layer")
    ax.set_ylabel("2Q pair")
    ax.set_title("Pair activity by layer")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_graphs_by_layer(
    current_graphs: Sequence[Dict[Tuple[int, int], float]],
    effective_graphs: Sequence[Dict[Tuple[int, int], float]],
    n: int,
    mode_name: str,
    outpath: Path,
) -> None:
    n_layers = len(current_graphs)
    fig, axes = plt.subplots(n_layers, 2, figsize=(14, max(3.2 * n_layers, 4.0)))
    if n_layers == 1:
        axes = np.array([axes])

    pos = {q: (q, 0.0) for q in range(n)}

    def _draw_graph(ax: plt.Axes, edge_counts: Dict[Tuple[int, int], float], title: str) -> None:
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for (u, v), w in edge_counts.items():
            G.add_edge(u, v, weight=float(w))
        nx.draw_networkx_nodes(G, pos, node_size=350, ax=ax)
        nx.draw_networkx_labels(G, pos, labels={q: str(q) for q in range(n)}, font_size=8, ax=ax)
        if G.number_of_edges() > 0:
            widths = [1.2 + 2.5 * float(d["weight"]) / max(1.0, max(nx.get_edge_attributes(G, "weight").values())) for _, _, d in G.edges(data=True)]
            nx.draw_networkx_edges(G, pos, width=widths, ax=ax)
            edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)
        ax.set_title(title)
        ax.set_axis_off()

    for s in range(n_layers):
        _draw_graph(axes[s, 0], current_graphs[s], f"L{s}: current graph")
        _draw_graph(axes[s, 1], effective_graphs[s], f"L{s}: effective {mode_name} graph")

    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_heatmap(df: pd.DataFrame, outpath: Path, annotate: bool = True) -> None:
    if df.empty:
        return
    metrics = ["C_hat", "J_u", "J_v", "S_struct", "R_pair", "D_dense", "Phi_dense", "Gamma_dense"]
    mat = df[metrics].to_numpy(dtype=float)
    labels = df["gate_label"].tolist()

    fig, ax = plt.subplots(figsize=(10.5, max(4.0, 0.42 * len(labels) + 2.0)))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Per-gate dense metrics")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_summary_bar(summary_df: pd.DataFrame, outpath: Path) -> None:
    if summary_df.empty:
        return
    order = summary_df.sort_values("mean_Gamma_dense", ascending=False).reset_index(drop=True)
    labels = [f"{m}\n({mode})" for m, mode in zip(order["motif"], order["mode"])]
    vals = order["mean_Gamma_dense"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(8, 1.25 * len(labels)), 5))
    ax.bar(range(len(labels)), vals)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Mean Gamma_dense")
    ax.set_title("Motif ranking by mean Gamma_dense")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def concise_gate_log(df: pd.DataFrame) -> str:
    if df.empty:
        return "No 2Q gates found."
    cols = ["layer", "gate_id", "pair", "C_hat", "J_u", "J_v", "S_struct", "R_pair", "D_dense", "Gamma_dense"]
    show = df[cols].copy()
    for c in ["C_hat", "J_u", "J_v", "S_struct", "R_pair", "D_dense", "Gamma_dense"]:
        show[c] = show[c].map(lambda x: f"{float(x):.3f}")
    return show.to_string(index=False)


def concise_summary_log(summary: Dict[str, Any], notes: str) -> str:
    lines = [
        f"motif: {summary['motif']}",
        f"mode: {summary['mode']}",
        f"notes: {notes}",
        f"num_layers: {summary.get('num_layers', 0)}",
        f"num_gates: {summary.get('num_gates', 0)}",
    ]
    ordered_keys = [
        "mean_C_raw", "mean_C_hat", "mean_J_u", "mean_J_v", "mean_S_struct",
        "mean_R_pair", "mean_D_dense", "mean_Phi_dense", "mean_Gamma_dense",
        "max_Gamma_dense", "p90_Gamma_dense",
    ]
    for k in ordered_keys:
        if k in summary:
            lines.append(f"{k}: {float(summary[k]):.4f}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------


def run_one_case(motif: MotifSpec, cfg: DenseConfig, mode_name: str, case_dir: Path) -> Dict[str, Any]:
    _ensure_dir(case_dir)

    qc = build_quantum_circuit(motif, cfg.num_qubits)
    edge_counts_per_layer = layer_edge_counts(motif.layers)
    current_graphs = edge_counts_per_layer

    if mode_name == "ewma":
        effective_graphs = build_ewma_effective_graphs(
            edge_counts_per_layer,
            alpha=cfg.ewma_alpha,
            cutoff=cfg.ewma_cutoff,
        )
    elif mode_name == "window":
        effective_graphs = build_window_effective_graphs(
            edge_counts_per_layer,
            radius=cfg.window_radius,
            decay=cfg.window_decay,
            normalize=cfg.window_normalize,
            weights=cfg.window_weights,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode_name}")

    df = compute_dense_gate_rows(
        motif_name=motif.name,
        layers=motif.layers,
        edge_counts_per_layer=edge_counts_per_layer,
        effective_graphs=effective_graphs,
        mode_name=mode_name,
        cfg=cfg,
    )
    df.to_csv(case_dir / "gate_metrics.csv", index=False)

    summary = summarize_motif(df, motif_name=motif.name, mode_name=mode_name)
    pd.DataFrame([summary]).to_csv(case_dir / "motif_summary.csv", index=False)

    # Save graphs as CSV for easier inspection.
    cur_rows: List[Dict[str, Any]] = []
    eff_rows: List[Dict[str, Any]] = []
    for s, ec in enumerate(current_graphs):
        for (u, v), w in sorted(ec.items()):
            cur_rows.append({"layer": s, "u": u, "v": v, "weight": float(w)})
    for s, ec in enumerate(effective_graphs):
        for (u, v), w in sorted(ec.items()):
            eff_rows.append({"layer": s, "u": u, "v": v, "weight": float(w)})
    pd.DataFrame(cur_rows).to_csv(case_dir / "current_layer_graph_edges.csv", index=False)
    pd.DataFrame(eff_rows).to_csv(case_dir / f"effective_{mode_name}_graph_edges.csv", index=False)

    plot_circuit_timeline(motif, cfg.num_qubits, case_dir / "circuit_timeline.png")
    plot_pair_layer_heatmap(motif.layers, case_dir / "pair_layer_heatmap.png")
    plot_graphs_by_layer(current_graphs, effective_graphs, cfg.num_qubits, mode_name, case_dir / f"graphs_by_layer_{mode_name}.png")
    plot_metrics_heatmap(df, case_dir / "dense_metrics_heatmap.png", annotate=cfg.annotate_heatmaps)

    write_text(case_dir / "gate_metrics_concise.txt", concise_gate_log(df))
    write_text(case_dir / "summary.txt", concise_summary_log(summary, motif.notes))

    if cfg.save_qiskit_text:
        if qc is None:
            write_text(case_dir / "circuit_qiskit.txt", "Qiskit not available in this environment; timeline plot was saved instead.\n")
        else:
            try:
                write_text(case_dir / "circuit_qiskit.txt", str(qc.draw(output="text")))
            except Exception as exc:  # pragma: no cover
                write_text(case_dir / "circuit_qiskit.txt", f"Qiskit text drawing failed: {exc}\n")

    print(f"[{mode_name}] {motif.name}: mean_Gamma_dense={summary['mean_Gamma_dense']:.4f}, max_Gamma_dense={summary['max_Gamma_dense']:.4f}")
    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1A Gamma_dense calibration harness")
    parser.add_argument("--motif", type=str, default="all",
                        choices=["all", "repeated_local_chain", "brickwork_dense", "hotspot_fanout", "alternating_hotspots", "mixed_dense_ratio1", "extreme_dense_local"],
                        help="Which motif to run")
    parser.add_argument("--mode", type=str, default="both", choices=["ewma", "window", "both"], help="Effective graph mode")
    parser.add_argument("--num-qubits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--outdir", type=str, default="calibration/phase1A")

    # Dense knobs
    parser.add_argument("--lambda-decay", type=float, default=0.80)
    parser.add_argument("--alpha", type=float, default=0.80)
    parser.add_argument("--beta", type=float, default=0.65)
    parser.add_argument("--c-high", type=float, default=0.90)
    parser.add_argument("--eta", type=float, default=0.25)

    # EWMA knobs
    parser.add_argument("--ewma-alpha", type=float, default=0.85)
    parser.add_argument("--ewma-cutoff", type=float, default=0.25)

    # Window knobs
    parser.add_argument("--window-radius", type=int, default=2)
    parser.add_argument("--window-decay", type=float, default=0.60)
    parser.add_argument("--window-normalize", action="store_true")

    # Misc
    parser.add_argument("--no-annotate", action="store_true", help="Disable text annotations inside heatmaps")
    parser.add_argument("--no-qiskit-text", action="store_true", help="Do not save qiskit text circuit rendering")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DenseConfig(
        lambda_decay=args.lambda_decay,
        alpha=args.alpha,
        beta=args.beta,
        c_high=args.c_high,
        eta=args.eta,
        mode=args.mode,
        ewma_alpha=args.ewma_alpha,
        ewma_cutoff=args.ewma_cutoff,
        window_radius=args.window_radius,
        window_decay=args.window_decay,
        window_normalize=bool(args.window_normalize),
        num_qubits=args.num_qubits,
        seed=args.seed,
        outdir=args.outdir,
        save_qiskit_text=not bool(args.no_qiskit_text),
        annotate_heatmaps=not bool(args.no_annotate),
    )

    root = Path(cfg.outdir)
    _ensure_dir(root)
    write_text(root / "run_config.json", json.dumps(asdict(cfg), indent=2))

    factory = MotifFactory(n=cfg.num_qubits, seed=cfg.seed)
    motif_names = factory.all_names() if args.motif == "all" else [args.motif]
    mode_names = ["ewma", "window"] if cfg.mode == "both" else [cfg.mode]

    all_summaries: List[Dict[str, Any]] = []
    for motif_name in motif_names:
        motif = factory.build(motif_name)
        for mode_name in mode_names:
            case_dir = root / mode_name / motif_name
            summary = run_one_case(motif, cfg, mode_name, case_dir)
            all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)
    if not summary_df.empty:
        summary_df.to_csv(root / "all_motif_summaries.csv", index=False)
        plot_summary_bar(summary_df, root / "motif_ranking_mean_gamma_dense.png")
        write_text(root / "all_motif_summaries.txt", summary_df.to_string(index=False))

    print(f"Saved Phase 1A outputs under: {root.resolve()}")


if __name__ == "__main__":
    main()
