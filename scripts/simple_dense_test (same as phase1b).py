#!/usr/bin/env python3
from __future__ import annotations

"""
Phase 1.2 / Phase 1B dense-score calibration harness.

What this script does
---------------------
1. Builds hand-crafted motif circuits, including user cases A/B/C/D and several
   canonical dense-local motifs.
2. Computes the new dense score:
      C_raw, B_dense(kappa), J_u, J_v, S_struct, R_pair, D_dense, Gamma_dense
   using a bidirectional window graph by default.
3. Produces case-wise logs, CSV tables, and visualizations so the motifs can be
   inspected quickly.
4. Supports multiple kappa values and optional EWMA runs for comparison.

Dense score used here
---------------------
For fixed-topology technology capacity kappa,

    K_extra        = k - 1
    K_pair         = 2 * (k - 1)
    C_raw(u,v,l)   = (deg(u)-n_uv) + (deg(v)-n_uv)
    B_dense        = max(0, C_raw - K_pair) / (K_extra + eps)
    D_dense        = max(S_struct, R_pair)
    Gamma_dense    = B_dense * (1 - D_dense)

This is the simplified dense branch discussed in the paper draft.
"""

import argparse
import json
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
# Config and data structures
# -----------------------------------------------------------------------------


@dataclass
class DenseCaseConfig:
    mode: str = "both"  # window | ewma | both

    # Window graph (main technique)
    window_radius: int = 3
    window_weights: Optional[List[float]] = None  # symmetric length 2r+1
    window_normalize: bool = False

    # One-sided temporal profiles used in S_struct
    profile_radius: int = 3
    profile_weights: Optional[List[float]] = None  # length radius

    # Pair-reuse decay (kept separate intentionally)
    lambda_decay: float = 0.85

    # Optional EWMA comparison
    ewma_alpha: float = 0.85
    ewma_cutoff: float = 0.25

    # Technology capacity values
    kappas: List[float] = field(default_factory=lambda: [3.0])

    eps: float = 1e-12
    outdir: str = "calibration/phase1B_dense_cases"
    annotate_heatmaps: bool = True
    save_qiskit_text: bool = True


@dataclass
class LayerSpec:
    twoq: List[Tuple[int, int]] = field(default_factory=list)
    oneq: List[int] = field(default_factory=list)
    label: str = ""


@dataclass
class MotifSpec:
    name: str
    num_qubits: int
    layers: List[LayerSpec]
    target_layer: Optional[int] = None
    target_pair: Optional[Tuple[int, int]] = None
    notes: str = ""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _sorted_pair(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _symmetric_window_weights(radius: int) -> List[float]:
    # Default requested by user for radius=3: [0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]
    # base = {0: 1.0, 1: 0.75, 2: 0.50, 3: 0.25}
    # weights: List[float] = []
    # for d in range(radius, 0, -1):
    #     weights.append(base.get(d, max(0.0, 1.0 - 0.25 * d)))
    # weights.append(base.get(0, 1.0))
    # for d in range(1, radius + 1):
    #     weights.append(base.get(d, max(0.0, 1.0 - 0.25 * d)))
    return [1.0]* (2 * int(radius) + 1)
    # return weights


def _default_profile_weights(radius: int) -> List[float]:
    # Mirrors the side of the default window (nearest to farthest).
    side = {1: 0.75, 2: 0.50, 3: 0.25}
    return [side.get(d, max(0.0, 1.0 - 0.25 * d)) for d in range(1, radius + 1)]


def _window_weight(delta: int, radius: int, weights: Optional[Sequence[float]]) -> float:
    d = int(delta)
    if weights is None:
        weights = _symmetric_window_weights(radius)
    expected = 2 * radius + 1
    if len(weights) != expected:
        raise ValueError(f"window weights must have length {expected}, got {len(weights)}")
    idx = d + radius
    if not (0 <= idx < expected):
        return 0.0
    return float(weights[idx])



def _one_side_weight(delta: int, weights: Optional[Sequence[float]]) -> float:
    d = int(delta)
    if d <= 0:
        return 0.0
    if weights is None:
        raise ValueError("profile weights must not be None")
    if d > len(weights):
        return 0.0
    return float(weights[d - 1])



def _weighted_jaccard(a: Dict[int, float], b: Dict[int, float], eps: float) -> float:
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 0.0
    num = sum(min(float(a.get(k, 0.0)), float(b.get(k, 0.0))) for k in keys)
    den = sum(max(float(a.get(k, 0.0)), float(b.get(k, 0.0))) for k in keys) + eps
    return float(num / den)


# -----------------------------------------------------------------------------
# Motif factory
# -----------------------------------------------------------------------------


class MotifFactory:
    def all_names(self) -> List[str]:
        return [
            "case_A",
            "case_B",
            "case_C",
            "case_D",
            "chain",
            "star_hotspot",
            "brickwork_local",
            "repeated_pair_end",
            "alternating_local_neighborhoods",
        ]

    def build(self, name: str) -> MotifSpec:
        name = name.strip().lower()
        if name == "case_a":
            return self.case_a()
        if name == "case_b":
            return self.case_b()
        if name == "case_c":
            return self.case_c()
        if name == "case_d":
            return self.case_d()
        if name == "chain":
            return self.chain()
        if name == "star_hotspot":
            return self.star_hotspot()
        if name == "brickwork_local":
            return self.brickwork_local()
        if name == "repeated_pair_end":
            return self.repeated_pair_end()
        if name == "alternating_local_neighborhoods":
            return self.alternating_local_neighborhoods()
        raise ValueError(f"Unknown motif: {name}")

    def case_a(self) -> MotifSpec:
        # High raw congestion but very stable repeated local structure.
        layers = [
            LayerSpec(twoq=[(0, 1), (2, 3)], label="A_L0_parallel_outer"),
            LayerSpec(twoq=[(0, 1), (2, 3)], label="A_L1_parallel_outer"),
            LayerSpec(twoq=[(1, 2)], label="A_L2_target_middle"),
            LayerSpec(twoq=[(0, 1), (2, 3)], label="A_L3_parallel_outer"),
            LayerSpec(twoq=[(0, 1), (2, 3)], label="A_L4_parallel_outer"),
        ]
        return MotifSpec(
            name="case_A",
            num_qubits=4,
            layers=layers,
            target_layer=2,
            target_pair=(1, 2),
            notes="User case A: symmetric repeated local sandwich; target should be low / near-zero despite high raw congestion.",
        )

    def case_b(self) -> MotifSpec:
        # Medium case: one busy endpoint, some reuse around target, but weaker symmetry than A.
        layers = [
            LayerSpec(twoq=[(1, 2)], label="B_L0_repeat12"),
            LayerSpec(twoq=[(1, 2)], label="B_L1_repeat12"),
            LayerSpec(twoq=[(0, 2)], label="B_L2_fanout02"),
            LayerSpec(twoq=[(2, 3)], label="B_L3_target23"),
            LayerSpec(twoq=[(1, 2), (3, 4)], label="B_L4_repeat12_plus34"),
            LayerSpec(twoq=[(2, 4)], label="B_L5_fanout24"),
            LayerSpec(twoq=[(2, 5)], label="B_L6_fanout25"),
        ]
        return MotifSpec(
            name="case_B",
            num_qubits=6,
            layers=layers,
            target_layer=3,
            target_pair=(2, 3),
            notes="User case B: asymmetric local hinge / moving local chain; target should be medium.",
        )

    def case_c(self) -> MotifSpec:
        # Persistent hotspot around q3 with target q3-q5 in the middle.
        layers = [
            LayerSpec(twoq=[(0, 3)], label="C_L0_03"),
            LayerSpec(twoq=[(1, 3)], label="C_L1_13"),
            LayerSpec(twoq=[(2, 3), (4, 5)], label="C_L2_23_plus45"),
            LayerSpec(twoq=[(3, 4), (5, 7)], label="C_L3_34_plus57"),
            LayerSpec(twoq=[(3, 5)], label="C_L4_target35"),
            LayerSpec(twoq=[(3, 6), (4, 5)], label="C_L5_36_plus45"),
            LayerSpec(twoq=[(3, 7), (5, 6)], label="C_L6_37_plus56"),
        ]
        return MotifSpec(
            name="case_C",
            num_qubits=8,
            layers=layers,
            target_layer=4,
            target_pair=(3, 5),
            notes="User case C: persistent hotspot with weak temporal symmetry around target; target should be high.",
        )

    def case_d(self) -> MotifSpec:
        # Two busy endpoints with multi-branch competition; intended as another high case.
        layers = [
            LayerSpec(twoq=[(0, 3), (1, 6)], label="D_L0_03_plus16"),
            LayerSpec(twoq=[(1, 3), (2, 6)], label="D_L1_13_plus26"),
            LayerSpec(twoq=[(3, 4), (4, 6)], label="D_L2_34_plus46"),
            LayerSpec(twoq=[(0, 3), (5, 6)], label="D_L3_03_plus56"),
            LayerSpec(twoq=[(3, 6)], label="D_L4_target36"),
            LayerSpec(twoq=[(3, 7), (4, 6)], label="D_L5_37_plus46"),
            LayerSpec(twoq=[(1, 3), (2, 6)], label="D_L6_13_plus26"),
        ]
        return MotifSpec(
            name="case_D",
            num_qubits=8,
            layers=layers,
            target_layer=4,
            target_pair=(3, 6),
            notes="User case D: strong multi-branch local pressure on both endpoints; target should be high.",
        )

    def chain(self) -> MotifSpec:
        layers = [
            LayerSpec(twoq=[(0, 1), (2, 3), (4, 5)], label="chain_even_0"),
            LayerSpec(twoq=[(1, 2), (3, 4)], label="chain_odd_1"),
            LayerSpec(twoq=[(0, 1), (2, 3), (4, 5)], label="chain_even_2"),
            LayerSpec(twoq=[(1, 2), (3, 4)], label="chain_odd_3"),
        ]
        return MotifSpec(
            name="chain",
            num_qubits=6,
            layers=layers,
            target_layer=1,
            target_pair=(1, 2),
            notes="Repeated local chain; strong local structure and moderate congestion.",
        )

    def star_hotspot(self) -> MotifSpec:
        layers = [
            LayerSpec(twoq=[(3, 1)], label="star_31"),
            LayerSpec(twoq=[(3, 2)], label="star_32"),
            LayerSpec(twoq=[(4, 5)], label="star_45"),
            LayerSpec(twoq=[(3, 4)], label="star_target34"),
            LayerSpec(twoq=[(3, 5)], label="star_35"),
            LayerSpec(twoq=[(3, 6)], label="star_36"),
            LayerSpec(twoq=[(4, 6)], label="star_46"),
        ]
        return MotifSpec(
            name="star_hotspot",
            num_qubits=8,
            layers=layers,
            target_layer=3,
            target_pair=(3, 4),
            notes="Canonical hotspot / fanout motif; target should be high due to one highly contested endpoint.",
        )

    def brickwork_local(self) -> MotifSpec:
        layers = [
            LayerSpec(twoq=[(1, 2), (3, 4), (5, 6)], label="brick_even_0"),
            LayerSpec(twoq=[(2, 3), (4, 5)], label="brick_odd_1"),
            LayerSpec(twoq=[(1, 2), (3, 4), (5, 6)], label="brick_even_2"),
            LayerSpec(twoq=[(2, 3), (4, 5)], label="brick_odd_3"),
            LayerSpec(twoq=[(1, 2), (3, 4), (5, 6)], label="brick_even_4"),
        ]
        return MotifSpec(
            name="brickwork_local",
            num_qubits=8,
            layers=layers,
            target_layer=1,
            target_pair=(2, 3),
            notes="Local brickwork block; high local density with strong temporal regularity.",
        )

    def repeated_pair_end(self) -> MotifSpec:
        layers = [
            LayerSpec(twoq=[(1, 2), (4, 5)], label="rpe_0"),
            LayerSpec(twoq=[(2, 3), (4, 5)], label="rpe_1"),
            LayerSpec(twoq=[(1, 2), (3, 4)], label="rpe_2"),
            LayerSpec(twoq=[(2, 3)], label="rpe_target23"),
        ]
        return MotifSpec(
            name="repeated_pair_end",
            num_qubits=6,
            layers=layers,
            target_layer=3,
            target_pair=(2, 3),
            notes="Repeated pair appears again at the end of the motif; intended to test direct pair reuse discount.",
        )

    def alternating_local_neighborhoods(self) -> MotifSpec:
        layers = [
            LayerSpec(twoq=[(3, 1)], label="alt_31"),
            LayerSpec(twoq=[(3, 5)], label="alt_35"),
            LayerSpec(twoq=[(3, 2)], label="alt_target32"),
            LayerSpec(twoq=[(3, 6)], label="alt_36"),
            LayerSpec(twoq=[(3, 1)], label="alt_31_repeat"),
        ]
        return MotifSpec(
            name="alternating_local_neighborhoods",
            num_qubits=8,
            layers=layers,
            target_layer=2,
            target_pair=(2, 3),
            notes="Endpoint alternates between neighboring local regions; useful for separating congestion from stable structure.",
        )


# -----------------------------------------------------------------------------
# Circuit construction
# -----------------------------------------------------------------------------


ONEQ_PATTERN = ("h", "x", "z", "s")
TWOQ_PATTERN = ("cx", "cz")


def build_quantum_circuit(motif: MotifSpec):
    if QuantumCircuit is None:
        return None
    qc = QuantumCircuit(motif.num_qubits, name=motif.name)
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
        qc.barrier(*range(motif.num_qubits))
    return qc


# -----------------------------------------------------------------------------
# Effective graphs and metric computation
# -----------------------------------------------------------------------------


def layer_edge_counts(layers: Sequence[LayerSpec]) -> List[Dict[Tuple[int, int], float]]:
    out: List[Dict[Tuple[int, int], float]] = []
    for layer in layers:
        counts: Dict[Tuple[int, int], float] = defaultdict(float)
        for u, v in layer.twoq:
            counts[_sorted_pair(int(u), int(v))] += 1.0
        out.append(dict(counts))
    return out


def build_window_effective_graphs(
    edge_counts_per_layer: Sequence[Dict[Tuple[int, int], float]],
    radius: int,
    weights: Optional[Sequence[float]],
    normalize: bool,
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
            w_layer = _window_weight(delta, radius=radius, weights=weights)
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


def build_ewma_effective_graphs(
    edge_counts_per_layer: Sequence[Dict[Tuple[int, int], float]],
    alpha: float,
    cutoff: float,
) -> List[Dict[Tuple[int, int], float]]:
    hist: Dict[Tuple[int, int], float] = {}
    out: List[Dict[Tuple[int, int], float]] = []
    for cur in edge_counts_per_layer:
        ewma_update_and_prune(hist, cur, alpha=alpha, cutoff=cutoff)
        out.append(dict(hist))
    return out


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
    weights: Optional[Sequence[float]],
) -> Dict[int, float]:
    out: Dict[int, float] = defaultdict(float)
    n_layers = len(edge_counts_per_layer)
    for delta in range(1, int(radius) + 1):
        j = layer_idx - delta if direction == "past" else layer_idx + delta
        if j < 0 or j >= n_layers:
            continue
        w = _one_side_weight(delta, weights=weights)
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
    motif: MotifSpec,
    edge_counts_per_layer: Sequence[Dict[Tuple[int, int], float]],
    effective_graphs: Sequence[Dict[Tuple[int, int], float]],
    mode_name: str,
    cfg: DenseCaseConfig,
    kappa: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    radius = int(cfg.profile_radius)
    profile_weights = cfg.profile_weights or _default_profile_weights(radius)

    for s, layer in enumerate(motif.layers):
        eff = effective_graphs[s]
        c_raw_map = congestion_raw_map(eff)
        ordered_pairs = sorted((_sorted_pair(u, v) for (u, v) in layer.twoq), key=lambda p: (p[0], p[1]))

        for gate_idx, pair in enumerate(ordered_pairs, start=1):
            u, v = pair
            P_u = weighted_neighbor_profile(edge_counts_per_layer, s, u, direction="past", radius=radius, weights=profile_weights)
            F_u = weighted_neighbor_profile(edge_counts_per_layer, s, u, direction="future", radius=radius, weights=profile_weights)
            P_v = weighted_neighbor_profile(edge_counts_per_layer, s, v, direction="past", radius=radius, weights=profile_weights)
            F_v = weighted_neighbor_profile(edge_counts_per_layer, s, v, direction="future", radius=radius, weights=profile_weights)

            J_u = _weighted_jaccard(P_u, F_u, eps=cfg.eps)
            J_v = _weighted_jaccard(P_v, F_v, eps=cfg.eps)
            S_struct = min(J_u, J_v)
            R_pair = pair_reuse_score(edge_counts_per_layer, s, pair, radius=radius, lambda_decay=cfg.lambda_decay)
            D_dense = max(S_struct, R_pair)

            C_raw = float(c_raw_map.get(pair, 0.0))
            k_actual = float(kappa)              # input is actual technology connectivity k
            K_extra = max(0.0, k_actual - 1.0)   # per-endpoint extra connectivity
            K_pair = 2.0 * K_extra               # summed pair spare capacity threshold

            B_dense = max(0.0, C_raw - K_pair) / (K_extra + cfg.eps) if K_extra > 0.0 else 0.0
            Gamma_dense = B_dense * (1.0 - D_dense)

            is_target = bool(motif.target_layer == s and motif.target_pair is not None and _sorted_pair(*motif.target_pair) == pair)

            rows.append({
                "motif": motif.name,
                "mode": mode_name,
                "kappa": float(kappa),
                "k_actual": float(k_actual),
                "K_extra": float(K_extra),
                "K_pair": float(K_pair),
                "layer": int(s),
                "layer_label": layer.label,
                "gate_id": int(gate_idx),
                "gate_label": f"L{s}_G{gate_idx}:{pair[0]}-{pair[1]}",
                "u": int(u),
                "v": int(v),
                "pair": f"({u},{v})",
                "C_raw": float(C_raw),
                "B_dense": float(B_dense),
                "J_u": float(J_u),
                "J_v": float(J_v),
                "S_struct": float(S_struct),
                "R_pair": float(R_pair),
                "D_dense": float(D_dense),
                "Gamma_dense": float(Gamma_dense),
                "is_target": bool(is_target),
            })
    return pd.DataFrame(rows)


def summarize_motif(df: pd.DataFrame, motif: MotifSpec, mode_name: str, kappa: float) -> Dict[str, Any]:
    if df.empty:
        return {"motif": motif.name, "mode": mode_name, "kappa": float(kappa), "num_layers": 0, "num_gates": 0}
    target_df = df[df["is_target"] == True]
    target_gamma = float(target_df["Gamma_dense"].iloc[0]) if not target_df.empty else float("nan")
    return {
        "motif": motif.name,
        "mode": mode_name,
        "kappa": float(kappa),
        "num_layers": int(df["layer"].nunique()),
        "num_gates": int(len(df)),
        "mean_C_raw": float(df["C_raw"].mean()),
        "mean_B_dense": float(df["B_dense"].mean()),
        "mean_S_struct": float(df["S_struct"].mean()),
        "mean_R_pair": float(df["R_pair"].mean()),
        "mean_D_dense": float(df["D_dense"].mean()),
        "mean_Gamma_dense": float(df["Gamma_dense"].mean()),
        "max_Gamma_dense": float(df["Gamma_dense"].max()),
        "target_Gamma_dense": float(target_gamma),
        "target_minus_mean": float(target_gamma - float(df["Gamma_dense"].mean())) if not np.isnan(target_gamma) else float("nan"),
    }


# -----------------------------------------------------------------------------
# Visualization and logs
# -----------------------------------------------------------------------------


def plot_circuit_timeline(motif: MotifSpec, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(motif.layers) * 1.4), max(4.5, motif.num_qubits * 0.52)))
    x_positions = np.arange(len(motif.layers), dtype=float)
    for q in range(motif.num_qubits):
        ax.hlines(q, -0.5, len(motif.layers) - 0.5, color="lightgray", linewidth=1.0)

    for s, layer in enumerate(motif.layers):
        for q in layer.oneq:
            ax.scatter([s], [q], marker="s", s=120, zorder=3)
        for u, v in layer.twoq:
            a, b = _sorted_pair(u, v)
            is_target = bool(motif.target_layer == s and motif.target_pair is not None and _sorted_pair(*motif.target_pair) == (a, b))
            color = "crimson" if is_target else None
            lw = 3.2 if is_target else 2.4
            size = 85 if is_target else 60
            ax.vlines(s, a, b, linewidth=lw, zorder=2, color=color)
            ax.scatter([s, s], [a, b], marker="o", s=size, zorder=4, color=color)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"L{s}" for s in range(len(motif.layers))])
    ax.set_yticks(range(motif.num_qubits))
    ax.set_yticklabels([f"q{q}" for q in range(motif.num_qubits)])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Qubit")
    ax.set_title(f"Circuit timeline: {motif.name} (target highlighted)")
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
    fig, ax = plt.subplots(figsize=(max(7, len(edge_counts_per_layer) * 1.2), max(4.5, len(pairs) * 0.45)))
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
    motif: MotifSpec,
    current_graphs: Sequence[Dict[Tuple[int, int], float]],
    effective_graphs: Sequence[Dict[Tuple[int, int], float]],
    mode_name: str,
    outpath: Path,
) -> None:
    n_layers = len(current_graphs)
    fig, axes = plt.subplots(n_layers, 2, figsize=(14, max(3.2 * n_layers, 4.0)))
    if n_layers == 1:
        axes = np.array([axes])

    pos = {q: (q, 0.0) for q in range(motif.num_qubits)}

    def _draw_graph(ax: plt.Axes, edge_counts: Dict[Tuple[int, int], float], title: str) -> None:
        G = nx.Graph()
        G.add_nodes_from(range(motif.num_qubits))
        for (u, v), w in edge_counts.items():
            G.add_edge(u, v, weight=float(w))
        nx.draw_networkx_nodes(G, pos, node_size=320, ax=ax)
        nx.draw_networkx_labels(G, pos, labels={q: str(q) for q in range(motif.num_qubits)}, font_size=8, ax=ax)
        if G.number_of_edges() > 0:
            max_w = max(nx.get_edge_attributes(G, "weight").values())
            widths = [1.2 + 2.6 * float(d["weight"]) / max(1.0, max_w) for _, _, d in G.edges(data=True)]
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
    metrics = ["C_raw", "B_dense", "S_struct", "R_pair", "D_dense", "Gamma_dense"]
    mat = df[metrics].to_numpy(dtype=float)
    labels = [f"* {g}" if t else g for g, t in zip(df["gate_label"].tolist(), df["is_target"].tolist())]
    fig, ax = plt.subplots(figsize=(10.5, max(4.0, 0.42 * len(labels) + 2.0)))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Per-gate dense metrics (* = target)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)



def plot_gate_gamma_bar(df: pd.DataFrame, motif: MotifSpec, kappa: float, outpath: Path) -> None:
    if df.empty:
        return
    labels = df["gate_label"].tolist()
    vals = df["Gamma_dense"].to_numpy(dtype=float)
    colors = ["crimson" if bool(v) else "C0" for v in df["is_target"].tolist()]
    mean_val = float(np.mean(vals))
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(labels) + 4), 4.8))
    ax.bar(range(len(labels)), vals, color=colors)
    ax.axhline(mean_val, linestyle="--", linewidth=1.5, color="black", label=f"mean={mean_val:.3f}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Gamma_dense")
    ax.set_title(f"Per-gate Gamma_dense: {motif.name} (kappa={kappa:g})")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)



def plot_summary_bar(summary_df: pd.DataFrame, outpath: Path, value_col: str, title: str) -> None:
    if summary_df.empty:
        return
    order = summary_df.sort_values(value_col, ascending=False).reset_index(drop=True)
    labels = [f"{m}\n({mode}, k={k:g})" for m, mode, k in zip(order["motif"], order["mode"], order["kappa"])]
    vals = order[value_col].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(8, 1.15 * len(labels)), 5.2))
    ax.bar(range(len(labels)), vals)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(value_col)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)



def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")



def concise_gate_log(df: pd.DataFrame) -> str:
    if df.empty:
        return "No 2Q gates found."
    cols = ["layer", "gate_id", "pair", "C_raw", "B_dense", "S_struct", "R_pair", "D_dense", "Gamma_dense", "is_target"]
    show = df[cols].copy()
    for c in ["C_raw", "B_dense", "S_struct", "R_pair", "D_dense", "Gamma_dense"]:
        show[c] = show[c].map(lambda x: f"{float(x):.3f}")
    return show.to_string(index=False)



def concise_summary_log(summary: Dict[str, Any], notes: str) -> str:
    lines = [
        f"motif: {summary['motif']}",
        f"mode: {summary['mode']}",
        f"kappa: {summary['kappa']}",
        f"notes: {notes}",
        f"num_layers: {summary.get('num_layers', 0)}",
        f"num_gates: {summary.get('num_gates', 0)}",
    ]
    ordered_keys = [
        "mean_C_raw", "mean_B_dense", "mean_S_struct", "mean_R_pair",
        "mean_D_dense", "mean_Gamma_dense", "max_Gamma_dense",
        "target_Gamma_dense", "target_minus_mean",
    ]
    for k in ordered_keys:
        if k in summary:
            lines.append(f"{k}: {float(summary[k]):.4f}")
    return "\n".join(lines)



def ranking_text(summary_df: pd.DataFrame, mode_name: str, kappa: float) -> str:
    sub = summary_df[(summary_df["mode"] == mode_name) & (summary_df["kappa"] == float(kappa)) & (summary_df["motif"].isin(["case_A", "case_B", "case_C", "case_D"]))].copy()
    if sub.empty or sub.shape[0] < 4:
        return f"Not enough A/B/C/D rows for mode={mode_name}, kappa={kappa:g}."
    order = sub.sort_values("target_Gamma_dense", ascending=True)[["motif", "target_Gamma_dense"]]
    relation = " < ".join(order["motif"].tolist())
    detail = order.to_string(index=False)
    return f"Target ranking for mode={mode_name}, kappa={kappa:g}:\n{relation}\n\n{detail}\n"


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------


def run_one_case(motif: MotifSpec, cfg: DenseCaseConfig, mode_name: str, kappa: float, case_dir: Path) -> Dict[str, Any]:
    _ensure_dir(case_dir)
    qc = build_quantum_circuit(motif)
    edge_counts_per_layer = layer_edge_counts(motif.layers)
    current_graphs = edge_counts_per_layer

    if mode_name == "window":
        effective_graphs = build_window_effective_graphs(
            edge_counts_per_layer,
            radius=cfg.window_radius,
            weights=cfg.window_weights or _symmetric_window_weights(cfg.window_radius),
            normalize=cfg.window_normalize,
        )
    elif mode_name == "ewma":
        effective_graphs = build_ewma_effective_graphs(
            edge_counts_per_layer,
            alpha=cfg.ewma_alpha,
            cutoff=cfg.ewma_cutoff,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode_name}")

    df = compute_dense_gate_rows(motif, edge_counts_per_layer, effective_graphs, mode_name, cfg, kappa)
    df.to_csv(case_dir / "gate_metrics.csv", index=False)

    summary = summarize_motif(df, motif, mode_name, kappa)
    pd.DataFrame([summary]).to_csv(case_dir / "motif_summary.csv", index=False)

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

    plot_circuit_timeline(motif, case_dir / "circuit_timeline.png")
    plot_pair_layer_heatmap(motif.layers, case_dir / "pair_layer_heatmap.png")
    plot_graphs_by_layer(motif, current_graphs, effective_graphs, mode_name, case_dir / f"graphs_by_layer_{mode_name}.png")
    plot_metrics_heatmap(df, case_dir / "dense_metrics_heatmap.png", annotate=cfg.annotate_heatmaps)
    plot_gate_gamma_bar(df, motif, kappa, case_dir / "gate_gamma_bar.png")

    write_text(case_dir / "gate_metrics_concise.txt", concise_gate_log(df))
    write_text(case_dir / "summary.txt", concise_summary_log(summary, motif.notes))
    if cfg.save_qiskit_text:
        if qc is None:
            write_text(case_dir / "circuit_qiskit.txt", "Qiskit not available in this environment; timeline plot was saved instead.\n")
        else:
            try:
                write_text(case_dir / "circuit_qiskit.txt", str(qc.draw(output="text")))
            except Exception as exc:
                write_text(case_dir / "circuit_qiskit.txt", f"Qiskit text drawing failed: {exc}\n")

    print(f"[{mode_name}][kappa={kappa:g}] {motif.name}: mean={summary['mean_Gamma_dense']:.4f}, target={summary['target_Gamma_dense']:.4f}, max={summary['max_Gamma_dense']:.4f}")
    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_float_list(s: str) -> List[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("At least one kappa value is required")
    return vals



def parse_args() -> argparse.Namespace:
    factory = MotifFactory()
    parser = argparse.ArgumentParser(description="Phase 1B dense-score motif harness")
    parser.add_argument("--motif", type=str, default="all", choices=["all"] + factory.all_names(), help="Which motif to run")
    parser.add_argument("--mode", type=str, default="both", choices=["window", "ewma", "both"], help="Effective graph mode")
    parser.add_argument(
        "--kappas",
        type=str,
        default="3",
        help="Comma-separated actual connectivity values k, e.g. 2,3,4",
    )
    parser.add_argument("--outdir", type=str, default="calibration/phase1B_dense_cases")

    parser.add_argument("--window-radius", type=int, default=3)
    parser.add_argument("--window-normalize", action="store_true")
    parser.add_argument("--lambda-decay", type=float, default=0.85)
    parser.add_argument("--ewma-alpha", type=float, default=0.85)
    parser.add_argument("--ewma-cutoff", type=float, default=0.25)

    parser.add_argument("--no-annotate", action="store_true")
    parser.add_argument("--no-qiskit-text", action="store_true")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    kappas = parse_float_list(args.kappas)
    cfg = DenseCaseConfig(
        mode=args.mode,
        window_radius=int(args.window_radius),
        window_weights=_symmetric_window_weights(int(args.window_radius)),
        window_normalize=bool(args.window_normalize),
        profile_radius=int(args.window_radius),
        profile_weights=_default_profile_weights(int(args.window_radius)),
        lambda_decay=float(args.lambda_decay),
        ewma_alpha=float(args.ewma_alpha),
        ewma_cutoff=float(args.ewma_cutoff),
        kappas=kappas,
        outdir=args.outdir,
        annotate_heatmaps=not bool(args.no_annotate),
        save_qiskit_text=not bool(args.no_qiskit_text),
    )

    root = Path(cfg.outdir)
    _ensure_dir(root)
    write_text(root / "run_config.json", json.dumps(asdict(cfg), indent=2))

    factory = MotifFactory()
    motif_names = factory.all_names() if args.motif == "all" else [args.motif]
    mode_names = ["window", "ewma"] if cfg.mode == "both" else [cfg.mode]

    all_summaries: List[Dict[str, Any]] = []
    for kappa in cfg.kappas:
        kappa_dir = root / f"kappa_{kappa:g}"
        _ensure_dir(kappa_dir)
        for motif_name in motif_names:
            motif = factory.build(motif_name)
            for mode_name in mode_names:
                case_dir = kappa_dir / mode_name / motif.name
                summary = run_one_case(motif, cfg, mode_name, kappa, case_dir)
                all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)
    if not summary_df.empty:
        summary_df.to_csv(root / "all_motif_summaries.csv", index=False)
        plot_summary_bar(summary_df, root / "motif_ranking_mean_gamma_dense.png", "mean_Gamma_dense", "Motif ranking by mean Gamma_dense")
        plot_summary_bar(summary_df, root / "motif_ranking_target_gamma_dense.png", "target_Gamma_dense", "Motif ranking by target Gamma_dense")
        write_text(root / "all_motif_summaries.txt", summary_df.to_string(index=False))

        ranking_lines: List[str] = []
        for kappa in cfg.kappas:
            for mode_name in mode_names:
                ranking_lines.append(ranking_text(summary_df, mode_name, kappa))
        write_text(root / "ABCD_target_rankings.txt", "\n".join(ranking_lines))

    print(f"Saved Phase 1B outputs under: {root.resolve()}")


if __name__ == "__main__":
    main()
