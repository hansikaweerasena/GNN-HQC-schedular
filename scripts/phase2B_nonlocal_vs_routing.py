#!/usr/bin/env python3
from __future__ import annotations

"""
Phase 2B: combined dense + non-local scoring vs routed overhead.

Purpose
-------
For each motif (from phase2A_nonlocal_cases_large.py):
  1) Compute per-edge dense score (Gamma_dense) using Phase 1B machinery.
  2) Classify each edge as local or non-local (3-stage local-bridge pipeline).
  3) Compute per-edge non-local score (Gamma_nonlocal) for non-local edges.
  4) Assign combined edge-wise score: Gamma_nonlocal if non-local, else Gamma_dense.
  5) Route the circuit on a chosen sparse topology via Sabre.
  6) Compare predicted scores vs routing overhead (SWAP count, added 2Q depth).

Scoring formulas
----------------
Dense (local edges):
    Gamma_dense = B_dense * (1 - D_dense)
    (from Phase 1B)

Non-local (classified edges):
    L_capped = min(L_detour, floor(|V_active| / kappa) + 1)
    Gamma_nonlocal = min((L_capped - 1) / kappa, Gamma_max)

Combined per-edge:
    Gamma(u,v,l) = Gamma_nonlocal   if I_nl = 1
                   Gamma_dense      otherwise
"""

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


# =============================================================================
# Module imports
# =============================================================================


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# =============================================================================
# Unified configuration — single source of truth for ALL parameters
# =============================================================================


@dataclass
class EvalConfig:
    # --- paths ---
    phase1b_path: str = "scripts/phase1B_dense_cases.py"
    phase2a_path: str = "scripts/phase2A_nonlocal_cases_large.py"
    outdir: str = "phase2B_nonlocal_vs_routing_out"

    # --- technology ---
    kappa: float = 4                   # average connectivity (e.g. 3 for heavy-hex)
    topology: str = "grid"      # routing topology for Sabre

    # --- dense score parameters (Phase 1B) ---
    dense_window_radius: int = 4
    dense_window_normalize: bool = False
    dense_lambda_decay: float = 0.85
    dense_eps: float = 1e-12

    # --- non-local classification parameters (Phase 2A) ---
    nl_window_radius: int = 8
    nl_window_normalize: bool = False
    delta_community: int = 3
    pair_reuse_radius: int = nl_window_radius      
    pair_reuse_threshold: int = 2

    # --- non-local scoring ---
    gamma_max: float = 3           # safety cap on non-local score

    # --- routing ---
    router: str = "sabre"            # "sabre" or "custom" or "both"
    sabre_seed: int = 7
    sabre_trials: int = 5            # average over N Sabre runs

    # --- motifs ---
    motifs: str = "all"              # comma-separated or "all"

    @property
    def detour_cap_from_v(self) -> None:
        """L_max is computed per-motif from |V_active|, not stored here."""
        return None


# =============================================================================
# Topology construction (reused from Phase 1C)
# =============================================================================


def build_heavy_hex_27() -> Tuple[nx.Graph, Dict[int, Tuple[float, float]]]:
    """
    27-node heavy-hex graph (max degree 3), following IBM's pattern.

    Structure: data qubits in rows, bridge qubits between rows.
    Each bridge qubit has degree 2 (connects exactly 2 data qubits).
    Data qubits have degree 2-3.

    Row 0 (data):    0 — 1 — 2 — 3 — 4
    Bridges:         5       6       7
    Row 1 (data):    8 — 9 —10 —11 —12
    Bridges:            13      14
    Row 2 (data):   15 —16 —17 —18 —19
    Bridges:        20      21      22
    Row 3 (data):   23 —24 —25 —26
    """
    g = nx.Graph()
    for n in range(27):
        g.add_node(n)

    # Row 0: horizontal chain
    for i in range(4):
        g.add_edge(i, i + 1)
    # Row 1: horizontal chain
    for i in range(8, 12):
        g.add_edge(i, i + 1)
    # Row 2: horizontal chain
    for i in range(15, 19):
        g.add_edge(i, i + 1)
    # Row 3: horizontal chain
    for i in range(23, 26):
        g.add_edge(i, i + 1)

    # Bridges between row 0 and row 1 (offset columns)
    g.add_edge(0, 5);  g.add_edge(5, 8)    # bridge 5: col 0
    g.add_edge(2, 6);  g.add_edge(6, 10)   # bridge 6: col 2
    g.add_edge(4, 7);  g.add_edge(7, 12)   # bridge 7: col 4

    # Bridges between row 1 and row 2 (different offset)
    g.add_edge(9, 13);  g.add_edge(13, 16)   # bridge 13: col 1
    g.add_edge(11, 14); g.add_edge(14, 18)   # bridge 14: col 3

    # Bridges between row 2 and row 3
    g.add_edge(15, 20); g.add_edge(20, 23)   # bridge 20: col 0
    g.add_edge(17, 21); g.add_edge(21, 25)   # bridge 21: col 2
    g.add_edge(19, 22); g.add_edge(22, 26)   # bridge 22: col 4 (26 is last in row 3)

    # Layout coords
    coords: Dict[int, Tuple[float, float]] = {}
    for i in range(5):
        coords[i] = (float(i * 2), 6.0)       # row 0
    coords[5] = (0.0, 5.0); coords[6] = (4.0, 5.0); coords[7] = (8.0, 5.0)
    for i in range(5):
        coords[8 + i] = (float(i * 2), 4.0)   # row 1
    coords[13] = (2.0, 3.0); coords[14] = (6.0, 3.0)
    for i in range(5):
        coords[15 + i] = (float(i * 2), 2.0)  # row 2
    coords[20] = (0.0, 1.0); coords[21] = (4.0, 1.0); coords[22] = (8.0, 1.0)
    for i in range(4):
        coords[23 + i] = (float(i * 2), 0.0)  # row 3

    for n, xy in coords.items():
        g.nodes[n]["pos"] = xy

    return g, coords


def build_grid_6x6() -> Tuple[nx.Graph, Dict[int, Tuple[float, float]]]:
    """6×6 square grid with 36 nodes, interior degree 4."""
    raw = nx.grid_2d_graph(6, 6)
    mapping = {node: idx for idx, node in enumerate(sorted(raw.nodes()))}
    g = nx.relabel_nodes(raw, mapping)
    coords = {mapping[(r, c)]: (float(c), float(-r)) for r in range(6) for c in range(6)}
    for n, xy in coords.items():
        g.nodes[n]["pos"] = tuple(xy)
    return g, coords


def select_topology(name: str) -> Tuple[nx.Graph, Dict[int, Tuple[float, float]], str]:
    name = name.strip().lower()
    if name == "heavy_hex":
        g, c = build_heavy_hex_27()
        return g, c, "heavy_hex_27"
    if name == "grid":
        g, c = build_grid_6x6()
        return g, c, "grid_6x6"
    raise ValueError(f"Unsupported topology: {name}. Use 'heavy_hex' or 'grid'.")


# =============================================================================
# Dense score computation (delegates to Phase 1B)
# =============================================================================


def compute_dense_scores(phase1b, motif, cfg: EvalConfig) -> pd.DataFrame:
    """Compute per-edge Gamma_dense using Phase 1B machinery."""
    p1b_cfg = phase1b.DenseCaseConfig(
        window_radius=cfg.dense_window_radius,
        window_weights=[1.0] * (2 * cfg.dense_window_radius + 1),
        window_normalize=cfg.dense_window_normalize,
        lambda_decay=cfg.dense_lambda_decay,
        eps=cfg.dense_eps,
    )
    edge_counts = phase1b.layer_edge_counts(motif.layers)
    eff_graphs = phase1b.build_window_effective_graphs(
        edge_counts,
        radius=cfg.dense_window_radius,
        weights=p1b_cfg.window_weights,
        normalize=cfg.dense_window_normalize,
    )
    df = phase1b.compute_dense_gate_rows(
        motif, edge_counts, eff_graphs, "window", p1b_cfg, kappa=float(cfg.kappa),
    )
    return df


# =============================================================================
# Non-local classification and scoring (delegates to Phase 2A + new score)
# =============================================================================


def _sorted_pair(u: int, v: int) -> Tuple[int, int]:
    return (min(u, v), max(u, v))


def compute_nonlocal_scores(phase2a, motif, cfg: EvalConfig) -> pd.DataFrame:
    """
    Classify each edge and compute Gamma_nonlocal for non-local edges.

    Score formula:
        L_capped = min(L_detour, floor(|V_active| / kappa) + 1)
        Gamma_nonlocal = min((L_capped - 1) / kappa, gamma_max)
    """
    edge_counts = phase2a.layer_edge_counts(motif.layers)
    weights = [1.0] * (2 * cfg.nl_window_radius + 1)
    eff_graphs = phase2a.build_window_effective_graphs(
        edge_counts, cfg.nl_window_radius, weights, normalize=cfg.nl_window_normalize,
    )

    kappa = float(cfg.kappa)
    gamma_max = float(cfg.gamma_max)

    rows: List[Dict[str, Any]] = []
    for s, layer in enumerate(motif.layers):
        eff = eff_graphs[s]

        # Count active qubits in this window's effective graph
        v_active = set()
        for (a, b) in eff:
            v_active.add(a)
            v_active.add(b)
        n_active = len(v_active)
        l_max = int(n_active / kappa) + 1

        ordered_pairs = sorted(
            (_sorted_pair(u, v) for (u, v) in layer.twoq),
            key=lambda p: (p[0], p[1]),
        )
        for pair in ordered_pairs:
            u, v = pair

            # Stage 1: common-neighbor test
            has_cn = phase2a.has_common_neighbor(eff, pair)
            is_local_bridge = not has_cn

            l_detour = float("nan")
            cu, cv = 0, 0
            reuse = 0
            pass_community = False
            pass_pair_reuse = False
            gamma_nl = 0.0

            if is_local_bridge:
                # Stage 2: community size guard
                l_raw, cu, cv = phase2a.detour_metrics(eff, motif.num_qubits, pair)
                l_detour = float(l_max) if math.isinf(l_raw) else float(l_raw)
                pass_community = (cu >= cfg.delta_community) and (cv >= cfg.delta_community)

                if pass_community:
                    # Stage 3: pair-reuse guard
                    reuse = phase2a.pair_reuse_count(
                        edge_counts, s, pair, cfg.pair_reuse_radius,
                    )
                    pass_pair_reuse = reuse < cfg.pair_reuse_threshold

            is_nonlocal = is_local_bridge and pass_community and pass_pair_reuse

            # Compute score for classified non-local edges
            if is_nonlocal:
                l_capped = min(l_detour, float(l_max))
                gamma_nl = min((l_capped - 1.0) / kappa, gamma_max)
                gamma_nl = max(0.0, gamma_nl)

            rows.append({
                "layer": int(s),
                "u": int(u),
                "v": int(v),
                "pair": f"({u},{v})",
                "is_local_bridge": bool(is_local_bridge),
                "C_u": int(cu),
                "C_v": int(cv),
                "reuse_count": int(reuse),
                "L_detour": float(l_detour),
                "L_max": int(l_max),
                "V_active": int(n_active),
                "I_nonlocal": int(is_nonlocal),
                "Gamma_nonlocal": float(gamma_nl),
            })

    return pd.DataFrame(rows)


# =============================================================================
# Combined edge-wise scoring
# =============================================================================


def combine_scores(dense_df: pd.DataFrame, nl_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge dense and non-local scores per edge.
    Mutual exclusion: non-local edges get Gamma_nonlocal, local edges get Gamma_dense.
    """
    # Build lookup from non-local df
    nl_lookup: Dict[Tuple[int, str], Tuple[int, float]] = {}
    for _, row in nl_df.iterrows():
        key = (int(row["layer"]), str(row["pair"]))
        nl_lookup[key] = (int(row["I_nonlocal"]), float(row["Gamma_nonlocal"]))

    combined_rows: List[Dict[str, Any]] = []
    for _, row in dense_df.iterrows():
        key = (int(row["layer"]), str(row["pair"]))
        i_nl, gamma_nl = nl_lookup.get(key, (0, 0.0))
        gamma_dense = float(row["Gamma_dense"])

        # Mutual exclusion
        if i_nl:
            gamma_combined = gamma_nl
            score_source = "nonlocal"
        else:
            gamma_combined = gamma_dense
            score_source = "dense"

        combined_rows.append({
            "motif": str(row["motif"]),
            "layer": int(row["layer"]),
            "u": int(row["u"]),
            "v": int(row["v"]),
            "pair": str(row["pair"]),
            "Gamma_dense": gamma_dense,
            "I_nonlocal": int(i_nl),
            "Gamma_nonlocal": gamma_nl,
            "Gamma_combined": gamma_combined,
            "score_source": score_source,
        })

    return pd.DataFrame(combined_rows)


# =============================================================================
# Sabre routing
# =============================================================================


def try_import_qiskit():
    try:
        from qiskit import transpile
        from qiskit.transpiler import CouplingMap
        return {"transpile": transpile, "CouplingMap": CouplingMap}
    except Exception:
        return None


def original_twoq_depth(motif) -> int:
    return sum(1 for layer in motif.layers if len(layer.twoq) > 0)


def original_twoq_count(motif) -> int:
    return int(sum(len(layer.twoq) for layer in motif.layers))


def qiskit_twoq_depth(qc) -> int:
    busy_until: Dict[int, int] = defaultdict(int)
    depth = 0
    for inst, qargs, _cargs in qc.data:
        if str(inst.name) == "barrier":
            continue
        qidx = []
        for q in qargs:
            idx = getattr(q, "_index", None)
            if idx is None:
                try:
                    idx = qc.find_bit(q).index
                except Exception:
                    idx = None
            if idx is not None:
                qidx.append(int(idx))
        if len(qidx) != 2:
            continue
        layer = 1 + max((busy_until[q] for q in qidx), default=0)
        for q in qidx:
            busy_until[q] = layer
        depth = max(depth, layer)
    return int(depth)


def count_twoq_ops_qiskit(qc) -> int:
    total = 0
    for inst, qargs, _cargs in qc.data:
        if str(inst.name) == "barrier":
            continue
        if len(qargs) == 2:
            total += 1
    return int(total)


@dataclass
class SabreResult:
    motif: str
    topology: str
    swap_count: int
    routed_twoq_depth: int
    added_twoq_depth: int
    routed_twoq_ops: int
    added_twoq_ops: int


def route_motif_sabre(
    phase1b, motif, topo_g: nx.Graph, topology_name: str, cfg: EvalConfig,
) -> Optional[SabreResult]:
    """Route a motif using Qiskit Sabre, averaged over cfg.sabre_trials seeds."""
    qk = try_import_qiskit()
    if qk is None:
        return None

    qc = phase1b.build_quantum_circuit(motif)
    if qc is None:
        return None

    edge_list = [list(map(int, e)) for e in topo_g.edges()]
    cmap = qk["CouplingMap"](couplinglist=edge_list)

    swap_counts = []
    depth_counts = []
    ops_counts = []

    for trial in range(cfg.sabre_trials):
        tqc = qk["transpile"](
            qc,
            coupling_map=cmap,
            basis_gates=["cx", "cz", "swap", "h", "x", "z", "s"],
            routing_method="sabre",
            layout_method="sabre",
            optimization_level=0,
            seed_transpiler=cfg.sabre_seed + trial,
        )
        sc = int(tqc.count_ops().get("swap", 0))
        td = qiskit_twoq_depth(tqc)
        to = count_twoq_ops_qiskit(tqc)
        swap_counts.append(sc)
        depth_counts.append(td)
        ops_counts.append(to)

    # Use median for stability
    swap_med = int(np.median(swap_counts))
    depth_med = int(np.median(depth_counts))
    ops_med = int(np.median(ops_counts))

    return SabreResult(
        motif=motif.name,
        topology=topology_name,
        swap_count=swap_med,
        routed_twoq_depth=depth_med,
        added_twoq_depth=int(depth_med - original_twoq_depth(motif)),
        routed_twoq_ops=ops_med,
        added_twoq_ops=int(ops_med - original_twoq_count(motif)),
    )


# =============================================================================
# Plot helpers
# =============================================================================


def save_scatter(
    df: pd.DataFrame, xcol: str, ycol: str, title: str, outpath: Path,
    annotate: bool = True,
) -> None:
    plt.figure(figsize=(9, 6))
    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    plt.scatter(x, y, s=50, zorder=5)

    if annotate:
        for _, row in df.iterrows():
            plt.annotate(
                str(row["motif"]), (float(row[xcol]), float(row[ycol])),
                fontsize=7, xytext=(4, 3), textcoords="offset points",
            )

    if len(df) >= 3 and np.std(x) > 1e-12:
        coeff = np.polyfit(x, y, 1)
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        ys = coeff[0] * xs + coeff[1]
        plt.plot(xs, ys, linestyle="--", color="gray", alpha=0.7)
        corr = np.corrcoef(x, y)[0, 1]
        from scipy import stats
        try:
            spearman_r, spearman_p = stats.spearmanr(x, y)
            plt.title(f"{title}\nPearson r={corr:.3f}, Spearman ρ={spearman_r:.3f}")
        except Exception:
            plt.title(f"{title}\nPearson r={corr:.3f}")
    else:
        plt.title(title)

    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def save_comparison_scatter(
    df: pd.DataFrame, outpath: Path,
) -> None:
    """
    Three-panel scatter: dense-only vs SWAPs, combined vs SWAPs, NL-count vs SWAPs.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, xcol, xlabel in [
        (axes[0], "total_gamma_dense_only", "Σ Gamma_dense (all edges)"),
        (axes[1], "total_gamma_combined", "Σ Gamma_combined (dense + NL)"),
        (axes[2], "num_nonlocal_edges", "# non-local edges"),
    ]:
        x = df[xcol].to_numpy(dtype=float)
        y = df["swap_count"].to_numpy(dtype=float)
        ax.scatter(x, y, s=50, zorder=5)
        for _, row in df.iterrows():
            ax.annotate(
                str(row["motif"]), (float(row[xcol]), float(row["swap_count"])),
                fontsize=6, xytext=(3, 3), textcoords="offset points",
            )
        if len(df) >= 3 and np.std(x) > 1e-12:
            coeff = np.polyfit(x, y, 1)
            xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            ax.plot(xs, coeff[0] * xs + coeff[1], "--", color="gray", alpha=0.7)
            corr = np.corrcoef(x, y)[0, 1]
            ax.set_title(f"r={corr:.3f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Sabre SWAP count")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Predicted score vs Sabre SWAP count", fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


# =============================================================================
# Motif selection
# =============================================================================


EVAL_MOTIFS = [
    # Local-only baselines
    "scaled_chain",
    "scaled_brickwork",
    # Low non-local
    "scaled_bridge",
    "scaled_shortcut",
    # Medium non-local
    "scaled_cross_community",
    # High non-local
    "random_overlay",
    "dense_cross_community",
    "burst_unrelated_lr",
    "alternating_lr_local",
    # Real algorithm circuits
    "real_qft_10",
    "real_qaoa_maxcut",
    "real_qft_16",
]


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    _d = EvalConfig()
    p = argparse.ArgumentParser(
        description="Phase 2B: combined dense + non-local scoring vs routed overhead",
    )
    # paths
    p.add_argument("--phase1b", type=str, default=_d.phase1b_path)
    p.add_argument("--phase2a", type=str, default=_d.phase2a_path)
    p.add_argument("--outdir", type=str, default=_d.outdir)
    # technology
    p.add_argument("--kappa", type=int, default=_d.kappa)
    p.add_argument("--topology", type=str, default=_d.topology, choices=["heavy_hex", "grid"])
    # dense params
    p.add_argument("--dense-window-radius", type=int, default=_d.dense_window_radius)
    p.add_argument("--dense-lambda-decay", type=float, default=_d.dense_lambda_decay)
    # non-local classification params
    p.add_argument("--nl-window-radius", type=int, default=_d.nl_window_radius)
    p.add_argument("--delta-community", type=int, default=_d.delta_community)
    p.add_argument("--pair-reuse-radius", type=int, default=_d.pair_reuse_radius)
    p.add_argument("--pair-reuse-threshold", type=int, default=_d.pair_reuse_threshold)
    # non-local scoring
    p.add_argument("--gamma-max", type=float, default=_d.gamma_max)
    # routing
    p.add_argument("--router", type=str, default=_d.router, choices=["sabre", "custom", "both"])
    p.add_argument("--sabre-seed", type=int, default=_d.sabre_seed)
    p.add_argument("--sabre-trials", type=int, default=_d.sabre_trials)
    # motifs
    p.add_argument("--motifs", type=str, default=_d.motifs)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EvalConfig(
        phase1b_path=args.phase1b,
        phase2a_path=args.phase2a,
        outdir=args.outdir,
        kappa=args.kappa,
        topology=args.topology,
        dense_window_radius=args.dense_window_radius,
        dense_lambda_decay=args.dense_lambda_decay,
        nl_window_radius=args.nl_window_radius,
        delta_community=args.delta_community,
        pair_reuse_radius=args.pair_reuse_radius,
        pair_reuse_threshold=args.pair_reuse_threshold,
        gamma_max=args.gamma_max,
        router=args.router,
        sabre_seed=args.sabre_seed,
        sabre_trials=args.sabre_trials,
        motifs=args.motifs,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load modules
    phase1b = _load_module("phase1B_dense_cases", Path(cfg.phase1b_path))
    phase2a = _load_module("phase2A_nonlocal_cases_large", Path(cfg.phase2a_path))

    # Select topology
    topo_g, topo_coords, topo_name = select_topology(cfg.topology)

    # Select motifs
    factory = phase2a.NonlocalMotifFactory()
    if cfg.motifs.strip().lower() == "all":
        motif_names = EVAL_MOTIFS
    else:
        motif_names = [m.strip() for m in cfg.motifs.split(",") if m.strip()]

    # Check qubit count vs topology
    topo_nodes = topo_g.number_of_nodes()

    summary_rows: List[Dict[str, Any]] = []
    detail_dir = outdir / "motif_details"
    detail_dir.mkdir(exist_ok=True)

    for name in motif_names:
        print(f"Processing: {name} ...", end=" ", flush=True)
        try:
            motif = factory.build(name)
        except Exception as exc:
            print(f"SKIP (build failed: {exc})")
            continue

        if motif.num_qubits > topo_nodes:
            print(f"SKIP (needs {motif.num_qubits} qubits, topology has {topo_nodes})")
            continue

        # 1. Dense scores
        dense_df = compute_dense_scores(phase1b, motif, cfg)

        # 2. Non-local classification + scores
        nl_df = compute_nonlocal_scores(phase2a, motif, cfg)

        # 3. Combined scores
        combined_df = combine_scores(dense_df, nl_df)

        # Save per-motif details
        mdir = detail_dir / name
        mdir.mkdir(exist_ok=True)
        dense_df.to_csv(mdir / "dense_scores.csv", index=False)
        nl_df.to_csv(mdir / "nonlocal_scores.csv", index=False)
        combined_df.to_csv(mdir / "combined_scores.csv", index=False)

        # Aggregate scores
        total_gamma_dense_only = float(dense_df["Gamma_dense"].sum())
        total_gamma_nl_only = float(combined_df.loc[
            combined_df["score_source"] == "nonlocal", "Gamma_nonlocal"
        ].sum())
        total_gamma_combined = float(combined_df["Gamma_combined"].sum())
        num_nl = int(nl_df["I_nonlocal"].sum())
        num_gates = len(combined_df)

        row = {
            "motif": name,
            "num_qubits": motif.num_qubits,
            "num_layers": len(motif.layers),
            "num_2q_gates": num_gates,
            "num_nonlocal_edges": num_nl,
            "total_gamma_dense_only": total_gamma_dense_only,
            "total_gamma_nl_only": total_gamma_nl_only,
            "total_gamma_combined": total_gamma_combined,
        }

        # 4. Route via Sabre
        if cfg.router in {"sabre", "both"}:
            sabre_res = route_motif_sabre(phase1b, motif, topo_g, topo_name, cfg)
            if sabre_res is not None:
                row.update({
                    "swap_count": sabre_res.swap_count,
                    "added_twoq_depth": sabre_res.added_twoq_depth,
                    "added_twoq_ops": sabre_res.added_twoq_ops,
                    "routed_twoq_depth": sabre_res.routed_twoq_depth,
                })
                print(f"OK (NL={num_nl}, Γ_comb={total_gamma_combined:.2f}, SWAPs={sabre_res.swap_count})")
            else:
                row.update({"swap_count": -1, "added_twoq_depth": -1, "added_twoq_ops": -1, "routed_twoq_depth": -1})
                print(f"OK (NL={num_nl}, Γ_comb={total_gamma_combined:.2f}, Sabre unavailable)")
        else:
            print(f"OK (NL={num_nl}, Γ_comb={total_gamma_combined:.2f})")

        summary_rows.append(row)

    if not summary_rows:
        raise RuntimeError("No results produced.")

    # Build summary dataframe
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "summary.csv", index=False)

    # Plots (only if Sabre results available)
    has_sabre = "swap_count" in summary_df.columns and (summary_df["swap_count"] > 0).any()
    if has_sabre:
        valid = summary_df[summary_df["swap_count"] >= 0].copy()

        # Main comparison scatter (3-panel)
        save_comparison_scatter(valid, outdir / "comparison_scatter.png")

        # Individual scatters
        save_scatter(valid, "total_gamma_combined", "swap_count",
                     "Combined score vs Sabre SWAPs", outdir / "combined_vs_swaps.png")
        save_scatter(valid, "total_gamma_combined", "added_twoq_depth",
                     "Combined score vs added 2Q depth", outdir / "combined_vs_depth.png")
        save_scatter(valid, "total_gamma_dense_only", "swap_count",
                     "Dense-only score vs Sabre SWAPs (baseline)", outdir / "dense_only_vs_swaps.png")
        save_scatter(valid, "total_gamma_nl_only", "swap_count",
                     "NL-only score vs Sabre SWAPs", outdir / "nl_only_vs_swaps.png")
        save_scatter(valid, "num_nonlocal_edges", "swap_count",
                     "# NL edges vs Sabre SWAPs", outdir / "nl_count_vs_swaps.png")

    # Save config
    config_out = {
        "kappa": cfg.kappa,
        "topology": cfg.topology,
        "dense_window_radius": cfg.dense_window_radius,
        "dense_lambda_decay": cfg.dense_lambda_decay,
        "nl_window_radius": cfg.nl_window_radius,
        "delta_community": cfg.delta_community,
        "pair_reuse_radius": cfg.pair_reuse_radius,
        "pair_reuse_threshold": cfg.pair_reuse_threshold,
        "gamma_max": cfg.gamma_max,
        "sabre_seed": cfg.sabre_seed,
        "sabre_trials": cfg.sabre_trials,
        "motifs_evaluated": [r["motif"] for r in summary_rows],
    }
    with open(outdir / "config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("Phase 2B Summary")
    print("=" * 80)
    print_cols = ["motif", "num_nonlocal_edges", "total_gamma_dense_only",
                  "total_gamma_combined"]
    if has_sabre:
        print_cols += ["swap_count", "added_twoq_depth"]
    print(summary_df[print_cols].to_string(index=False))


if __name__ == "__main__":
    main()
