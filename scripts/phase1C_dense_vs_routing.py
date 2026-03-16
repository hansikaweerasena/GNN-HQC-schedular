#!/usr/bin/env python3
from __future__ import annotations

"""
Phase 1C: motif-level dense-burden vs routed-overhead comparison.

Purpose
-------
For each hand-crafted motif from phase1B_dense_cases.py:
  1) compute the total predicted dense burden by summing Gamma_dense over all 2Q gates,
  2) route the full motif on a chosen sparse topology,
  3) measure routed overhead (SWAP count, added 2Q depth),
  4) visualize burden-vs-overhead trends and a few routed examples.

Notes
-----
- The primary path is a simple custom shortest-path swap inserter.
- An optional Qiskit SABRE path is included when qiskit is available in the runtime.
- This script reuses the exact motif definitions and dense-score machinery from
  phase1B_dense_cases.py, so the Phase 1C validation stays aligned with Phase 1B.
- The custom router is intentionally simple and deterministic; its outputs should be
  interpreted as approximate routed overhead, not exact compilation results.
"""

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Dynamic import of Phase 1B harness
# -----------------------------------------------------------------------------


def load_phase1b_module(path: Path):
    spec = importlib.util.spec_from_file_location("phase1B_dense_cases", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# -----------------------------------------------------------------------------
# Optional Qiskit path
# -----------------------------------------------------------------------------


def try_import_qiskit():
    try:
        from qiskit import transpile  # type: ignore
        from qiskit.transpiler import CouplingMap  # type: ignore
        return {"transpile": transpile, "CouplingMap": CouplingMap}
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Topology construction
# -----------------------------------------------------------------------------


def build_heavy_hex_d3_fallback() -> Tuple[nx.Graph, Dict[int, Tuple[float, float]]]:
    """
    Fallback 19-node heavy-hex d=3-like graph.

    The coordinates below follow the small heavy-hex layout used in IBM's custom-backend
    documentation example; edges are inferred by unit-distance adjacency in that layout.
    This gives a connected max-degree-3 graph suitable for Phase 1C experiments even when
    qiskit/rustworkx is unavailable locally.
    """
    coords: Dict[int, Tuple[float, float]] = {
        0: (3, 1),
        1: (3, -1),
        2: (2, -2),
        3: (1, 1),
        4: (0, 0),
        5: (-1, -1),
        6: (-2, 2),
        7: (-3, 1),
        8: (-3, -1),
        9: (2, 1),
        10: (1, -1),
        11: (-1, 1),
        12: (-2, -1),
        13: (3, 0),
        14: (2, -1),
        15: (0, 1),
        16: (0, -1),
        17: (-2, 1),
        18: (-3, 0),
    }
    g = nx.Graph()
    for n, xy in coords.items():
        g.add_node(int(n), pos=tuple(xy))

    nodes = sorted(coords)
    for i, u in enumerate(nodes):
        x1, y1 = coords[u]
        for v in nodes[i + 1 :]:
            x2, y2 = coords[v]
            d = math.dist((x1, y1), (x2, y2))
            if abs(d - 1.0) < 1e-9:
                g.add_edge(u, v)

    return g, coords



def build_grid_3x3_fallback() -> Tuple[nx.Graph, Dict[int, Tuple[float, float]]]:
    """Simple 3x3 square grid with 9 nodes and interior degree 4."""
    raw = nx.grid_2d_graph(3, 3)
    mapping = {node: idx for idx, node in enumerate(sorted(raw.nodes()))}
    g = nx.relabel_nodes(raw, mapping)
    coords = {mapping[(r, c)]: (float(c), float(-r)) for r in range(3) for c in range(3)}
    for n, xy in coords.items():
        g.nodes[n]["pos"] = tuple(xy)
    return g, coords



def select_topology(topology: str) -> Tuple[nx.Graph, Dict[int, Tuple[float, float]], str]:
    topology = topology.strip().lower()
    if topology == "heavy_hex":
        g, coords = build_heavy_hex_d3_fallback()
        return g, coords, "heavy_hex_d3"
    if topology == "grid_3x3":
        g, coords = build_grid_3x3_fallback()
        return g, coords, "grid_3x3"
    raise ValueError(f"Unsupported topology: {topology}")


# -----------------------------------------------------------------------------
# Original-motif metrics
# -----------------------------------------------------------------------------


def original_twoq_depth(motif) -> int:
    return sum(1 for layer in motif.layers if len(layer.twoq) > 0)



def original_twoq_count(motif) -> int:
    return int(sum(len(layer.twoq) for layer in motif.layers))


# -----------------------------------------------------------------------------
# Custom deterministic shortest-path router
# -----------------------------------------------------------------------------


@dataclass
class RoutedLayer:
    layer_idx: int
    source_layer: int
    kind: str               # oneq | twoq | swap
    name: str               # gate name or 'swap'
    phys: Tuple[int, ...]
    logical: Tuple[int, ...]
    note: str = ""


@dataclass
class CustomRouteResult:
    router: str
    motif: str
    initial_layout: Dict[int, int]
    final_layout: Dict[int, int]
    swap_count: int
    routed_twoq_depth: int
    added_twoq_depth: int
    routed_twoq_ops: int
    added_twoq_ops: int
    routed_layers: List[RoutedLayer]
    swap_edges: Counter
    topology_name: str



def choose_initial_layout_nodes(g: nx.Graph, n_logical: int) -> List[int]:
    if n_logical > g.number_of_nodes():
        raise ValueError(f"Need {n_logical} logical qubits but topology only has {g.number_of_nodes()} nodes")

    center_candidates = nx.center(g)
    root = int(sorted(center_candidates)[0]) if center_candidates else int(sorted(g.nodes())[0])

    seen = {root}
    order = [root]
    q = deque([root])
    while q and len(order) < n_logical:
        u = q.popleft()
        for v in sorted(g.neighbors(u)):
            if v not in seen:
                seen.add(v)
                order.append(int(v))
                q.append(int(v))
                if len(order) >= n_logical:
                    break

    if len(order) < n_logical:
        raise RuntimeError("Failed to extract a connected initial layout subset")
    return order[:n_logical]



def update_swap_mapping(a: int, b: int, p2l: Dict[int, Optional[int]], l2p: Dict[int, int]) -> None:
    la = p2l.get(a, None)
    lb = p2l.get(b, None)
    p2l[a], p2l[b] = lb, la
    if la is not None:
        l2p[la] = b
    if lb is not None:
        l2p[lb] = a



def route_motif_custom(motif, topo_g: nx.Graph, topology_name: str) -> CustomRouteResult:
    phys_nodes = choose_initial_layout_nodes(topo_g, motif.num_qubits)
    l2p: Dict[int, int] = {q: phys_nodes[q] for q in range(motif.num_qubits)}
    p2l: Dict[int, Optional[int]] = {p: None for p in topo_g.nodes()}
    for lq, pq in l2p.items():
        p2l[pq] = lq

    routed_layers: List[RoutedLayer] = []
    swap_edges: Counter = Counter()
    swap_count = 0
    routed_twoq_depth = 0
    routed_twoq_ops = 0

    for src_layer_idx, layer in enumerate(motif.layers):
        if layer.oneq:
            phys = tuple(l2p[int(q)] for q in layer.oneq)
            logical = tuple(int(q) for q in layer.oneq)
            routed_layers.append(
                RoutedLayer(
                    layer_idx=len(routed_layers),
                    source_layer=src_layer_idx,
                    kind="oneq",
                    name="1q",
                    phys=phys,
                    logical=logical,
                    note=layer.label,
                )
            )

        for (u0, v0) in layer.twoq:
            u = int(u0)
            v = int(v0)
            while nx.shortest_path_length(topo_g, l2p[u], l2p[v]) > 1:
                path = nx.shortest_path(topo_g, l2p[u], l2p[v])
                a = int(path[0])
                b = int(path[1])
                la = p2l.get(a, None)
                lb = p2l.get(b, None)
                update_swap_mapping(a, b, p2l=p2l, l2p=l2p)
                swap_edges[tuple(sorted((a, b)))] += 1
                swap_count += 1
                routed_twoq_depth += 1
                routed_twoq_ops += 1
                routed_layers.append(
                    RoutedLayer(
                        layer_idx=len(routed_layers),
                        source_layer=src_layer_idx,
                        kind="swap",
                        name="swap",
                        phys=(a, b),
                        logical=(int(la) if la is not None else -1, int(lb) if lb is not None else -1),
                        note=f"route {u}-{v}",
                    )
                )

            pu = int(l2p[u])
            pv = int(l2p[v])
            routed_twoq_depth += 1
            routed_twoq_ops += 1
            routed_layers.append(
                RoutedLayer(
                    layer_idx=len(routed_layers),
                    source_layer=src_layer_idx,
                    kind="twoq",
                    name="2q",
                    phys=(pu, pv),
                    logical=(u, v),
                    note=layer.label,
                )
            )

    return CustomRouteResult(
        router="custom",
        motif=motif.name,
        initial_layout={int(k): int(v) for k, v in zip(range(motif.num_qubits), phys_nodes)},
        final_layout={int(k): int(v) for k, v in l2p.items()},
        swap_count=int(swap_count),
        routed_twoq_depth=int(routed_twoq_depth),
        added_twoq_depth=int(routed_twoq_depth - original_twoq_depth(motif)),
        routed_twoq_ops=int(routed_twoq_ops),
        added_twoq_ops=int(routed_twoq_ops - original_twoq_count(motif)),
        routed_layers=routed_layers,
        swap_edges=swap_edges,
        topology_name=topology_name,
    )


# -----------------------------------------------------------------------------
# Optional Qiskit SABRE routing
# -----------------------------------------------------------------------------


@dataclass
class LibraryRouteResult:
    router: str
    motif: str
    swap_count: int
    routed_twoq_depth: int
    added_twoq_depth: int
    routed_twoq_ops: int
    added_twoq_ops: int
    transpiled_depth: int
    topology_name: str
    extra: Dict[str, Any]



def qiskit_twoq_depth(qc) -> int:
    busy_until: Dict[int, int] = defaultdict(int)
    depth = 0
    for inst, qargs, _cargs in qc.data:
        name = str(inst.name)
        if name == "barrier":
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



def route_motif_qiskit_sabre(motif, topo_g: nx.Graph, topology_name: str) -> LibraryRouteResult:
    qk = try_import_qiskit()
    if qk is None:
        raise RuntimeError("qiskit is not available in this runtime")

    phase1b = load_phase1b_module(Path(__file__).with_name("phase1B_dense_cases.py"))
    qc = phase1b.build_quantum_circuit(motif)
    if qc is None:
        raise RuntimeError("qiskit is unavailable or build_quantum_circuit returned None")

    transpile = qk["transpile"]
    CouplingMap = qk["CouplingMap"]
    edge_list = [list(map(int, e)) for e in topo_g.edges()]
    cmap = CouplingMap(couplinglist=edge_list)

    tqc = transpile(
        qc,
        coupling_map=cmap,
        basis_gates=["cx", "cz", "swap", "h", "x", "z", "s"],
        routing_method="sabre",
        layout_method="sabre",
        optimization_level=0,
        seed_transpiler=7,
    )

    swap_count = int(tqc.count_ops().get("swap", 0))
    routed_twoq_ops = count_twoq_ops_qiskit(tqc)
    routed_twoq_depth = qiskit_twoq_depth(tqc)
    return LibraryRouteResult(
        router="sabre",
        motif=motif.name,
        swap_count=swap_count,
        routed_twoq_depth=routed_twoq_depth,
        added_twoq_depth=int(routed_twoq_depth - original_twoq_depth(motif)),
        routed_twoq_ops=routed_twoq_ops,
        added_twoq_ops=int(routed_twoq_ops - original_twoq_count(motif)),
        transpiled_depth=int(tqc.depth()),
        topology_name=topology_name,
        extra={"count_ops": dict(tqc.count_ops()), "text_draw": str(tqc.draw("text"))},
    )


# -----------------------------------------------------------------------------
# Dense-burden computation using Phase 1B code
# -----------------------------------------------------------------------------


@dataclass
class BurdenResult:
    motif: str
    predicted_total_gamma: float
    predicted_mean_gamma: float
    predicted_target_gamma: float
    num_2q_gates: int
    gamma_df: pd.DataFrame



def compute_motif_burden(phase1b, motif, k_actual: float) -> BurdenResult:
    cfg = phase1b.DenseCaseConfig(mode="window")
    edge_counts = phase1b.layer_edge_counts(motif.layers)
    eff_graphs = phase1b.build_window_effective_graphs(
        edge_counts,
        radius=int(cfg.window_radius),
        weights=cfg.window_weights,
        normalize=bool(cfg.window_normalize),
    )
    df = phase1b.compute_dense_gate_rows(motif, edge_counts, eff_graphs, "window", cfg, kappa=float(k_actual))
    total_gamma = float(df["Gamma_dense"].sum()) if not df.empty else 0.0
    mean_gamma = float(df["Gamma_dense"].mean()) if not df.empty else 0.0
    target_df = df[df["is_target"] == True]
    target_gamma = float(target_df["Gamma_dense"].iloc[0]) if not target_df.empty else float("nan")
    return BurdenResult(
        motif=motif.name,
        predicted_total_gamma=total_gamma,
        predicted_mean_gamma=mean_gamma,
        predicted_target_gamma=target_gamma,
        num_2q_gates=int(df.shape[0]),
        gamma_df=df,
    )


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------


def save_scatter(df: pd.DataFrame, xcol: str, ycol: str, title: str, outpath: Path) -> None:
    plt.figure(figsize=(8, 5))
    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    plt.scatter(x, y)
    for _, row in df.iterrows():
        label = str(row["motif"])
        if "topology" in row.index:
            label += f"\n[{row['topology']}]"
        plt.annotate(label, (float(row[xcol]), float(row[ycol])), fontsize=8, xytext=(4, 3), textcoords="offset points")
    if len(df) >= 2 and np.std(x) > 1e-12:
        coeff = np.polyfit(x, y, 1)
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        ys = coeff[0] * xs + coeff[1]
        plt.plot(xs, ys, linestyle="--")
        corr = np.corrcoef(x, y)[0, 1]
        plt.title(f"{title}\nPearson r={corr:.3f}")
    else:
        plt.title(title)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()



def save_binned_calibration(df: pd.DataFrame, xcol: str, ycol: str, title: str, outpath: Path, n_bins: int = 4) -> None:
    vals = df[[xcol, ycol]].dropna().sort_values(xcol)
    if len(vals) == 0:
        return
    bins = pd.qcut(vals[xcol], q=min(n_bins, len(vals)), duplicates="drop")
    grouped = vals.groupby(bins, observed=False).agg({xcol: "mean", ycol: ["mean", "std", "count"]})
    x_mean = grouped[(xcol, "mean")].to_numpy(dtype=float)
    y_mean = grouped[(ycol, "mean")].to_numpy(dtype=float)
    y_std = grouped[(ycol, "std")].fillna(0.0).to_numpy(dtype=float)

    plt.figure(figsize=(8, 5))
    plt.errorbar(x_mean, y_mean, yerr=y_std, marker="o", capsize=3)
    plt.xlabel(f"binned {xcol}")
    plt.ylabel(f"mean {ycol}")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()



def save_routed_timeline(route_layers: Sequence[RoutedLayer], motif_name: str, outpath: Path) -> None:
    if len(route_layers) == 0:
        return
    plt.figure(figsize=(10, 4.8))
    for step in route_layers:
        x = step.layer_idx
        if step.kind == "oneq":
            for p in step.phys:
                plt.scatter([x], [p], marker="s", s=24, alpha=0.6)
        elif len(step.phys) == 2:
            y0, y1 = step.phys
            color = "tab:red" if step.kind == "swap" else "tab:blue"
            lw = 2.2 if step.kind == "swap" else 1.8
            plt.plot([x, x], [y0, y1], color=color, linewidth=lw)
            plt.scatter([x, x], [y0, y1], color=color, s=20)
    plt.xlabel("routed step index")
    plt.ylabel("physical qubit")
    plt.title(f"Routed timeline: {motif_name} (red=SWAP, blue=2Q gate)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()



def save_swap_edge_overlay(g: nx.Graph, coords: Dict[int, Tuple[float, float]], swap_edges: Counter, used_phys: Sequence[int], motif_name: str, outpath: Path) -> None:
    plt.figure(figsize=(7.5, 4.8))
    pos = {int(n): (float(x), float(y)) for n, (x, y) in coords.items()}
    nx.draw_networkx_edges(g, pos=pos, width=1.0, alpha=0.35)
    nx.draw_networkx_nodes(g, pos=pos, nodelist=list(g.nodes()), node_size=160, alpha=0.25)
    if used_phys:
        nx.draw_networkx_nodes(g, pos=pos, nodelist=list(used_phys), node_size=220)

    if swap_edges:
        weighted_edges = list(swap_edges.items())
        edges = [e for e, _ in weighted_edges]
        widths = [1.5 + 1.5 * float(c) for _, c in weighted_edges]
        nx.draw_networkx_edges(g, pos=pos, edgelist=edges, width=widths, edge_color="tab:red")

    nx.draw_networkx_labels(g, pos=pos, font_size=8)
    plt.title(f"SWAP edge overlay: {motif_name}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1C: dense burden vs routed overhead")
    p.add_argument("--phase1b", type=str, default=str(Path(__file__).with_name("phase1B_dense_cases.py")), help="Path to phase1B_dense_cases.py")
    p.add_argument("--motifs", type=str, default="all", help="Comma-separated motif names or 'all'")
    p.add_argument("--k", type=float, default=3.0, help="Actual technology connectivity k (heavy hex default: 3; grid_3x3 interior degree: 4)")
    p.add_argument("--topology", type=str, default="heavy_hex", choices=["heavy_hex", "grid_3x3"], help="Sparse topology used for routing comparison")
    p.add_argument("--router", type=str, default="custom", choices=["custom", "sabre", "both"], help="Routing backend(s) to compare")
    p.add_argument("--outdir", type=str, default="phase1C_dense_vs_routing_out", help="Output directory")
    return p.parse_args()



def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    phase1b = load_phase1b_module(Path(args.phase1b))
    factory = phase1b.MotifFactory()
    motif_names = factory.all_names() if args.motifs.strip().lower() == "all" else [m.strip() for m in args.motifs.split(",") if m.strip()]
    motifs = [factory.build(name) for name in motif_names]

    topo_g, topo_coords, topology_name = select_topology(args.topology)

    rows: List[Dict[str, Any]] = []

    custom_dir = outdir / "custom_router"
    custom_dir.mkdir(exist_ok=True)
    motif_detail_dir = custom_dir / "motif_details"
    motif_detail_dir.mkdir(exist_ok=True)

    sabre_dir = outdir / "sabre_router"
    if args.router in {"sabre", "both"}:
        sabre_dir.mkdir(exist_ok=True)

    for motif in motifs:
        burden = compute_motif_burden(phase1b, motif, k_actual=float(args.k))
        detail_dir = motif_detail_dir / motif.name
        detail_dir.mkdir(exist_ok=True)
        burden.gamma_df.to_csv(detail_dir / "dense_gate_metrics.csv", index=False)

        if args.router in {"custom", "both"}:
            routed = route_motif_custom(motif, topo_g=topo_g, topology_name=topology_name)
            used_phys = sorted(set(routed.initial_layout.values()))
            save_routed_timeline(routed.routed_layers, motif.name, detail_dir / f"custom_routed_timeline_{topology_name}.png")
            save_swap_edge_overlay(topo_g, topo_coords, routed.swap_edges, used_phys, motif.name, detail_dir / f"custom_swap_overlay_{topology_name}.png")

            with open(detail_dir / f"custom_route_summary_{topology_name}.json", "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "motif": motif.name,
                        "predicted_total_gamma": burden.predicted_total_gamma,
                        "predicted_mean_gamma": burden.predicted_mean_gamma,
                        "predicted_target_gamma": burden.predicted_target_gamma,
                        **{k: v for k, v in asdict(routed).items() if k not in {"routed_layers", "swap_edges"}},
                        "swap_edges": {f"{a}-{b}": int(c) for (a, b), c in routed.swap_edges.items()},
                    },
                    fh,
                    indent=2,
                )

            rows.append(
                {
                    "motif": motif.name,
                    "topology": topology_name,
                    "router": "custom",
                    "predicted_total_gamma": burden.predicted_total_gamma,
                    "predicted_mean_gamma": burden.predicted_mean_gamma,
                    "predicted_target_gamma": burden.predicted_target_gamma,
                    "num_2q_gates": burden.num_2q_gates,
                    "swap_count": routed.swap_count,
                    "routed_twoq_depth": routed.routed_twoq_depth,
                    "added_twoq_depth": routed.added_twoq_depth,
                    "routed_twoq_ops": routed.routed_twoq_ops,
                    "added_twoq_ops": routed.added_twoq_ops,
                }
            )

        if args.router in {"sabre", "both"}:
            detail_sabre_dir = sabre_dir / motif.name
            detail_sabre_dir.mkdir(exist_ok=True)
            try:
                lib = route_motif_qiskit_sabre(motif, topo_g=topo_g, topology_name=topology_name)
                with open(detail_sabre_dir / f"sabre_route_summary_{topology_name}.json", "w", encoding="utf-8") as fh:
                    json.dump({"motif": motif.name, "predicted_total_gamma": burden.predicted_total_gamma, **asdict(lib)}, fh, indent=2)
                with open(detail_sabre_dir / f"sabre_circuit_{topology_name}.txt", "w", encoding="utf-8") as fh:
                    fh.write(str(lib.extra.get("text_draw", "")))
                rows.append(
                    {
                        "motif": motif.name,
                        "topology": topology_name,
                        "router": "sabre",
                        "predicted_total_gamma": burden.predicted_total_gamma,
                        "predicted_mean_gamma": burden.predicted_mean_gamma,
                        "predicted_target_gamma": burden.predicted_target_gamma,
                        "num_2q_gates": burden.num_2q_gates,
                        "swap_count": lib.swap_count,
                        "routed_twoq_depth": lib.routed_twoq_depth,
                        "added_twoq_depth": lib.added_twoq_depth,
                        "routed_twoq_ops": lib.routed_twoq_ops,
                        "added_twoq_ops": lib.added_twoq_ops,
                    }
                )
            except Exception as exc:
                with open(detail_sabre_dir / f"sabre_unavailable_{topology_name}.txt", "w", encoding="utf-8") as fh:
                    fh.write(f"SABRE path unavailable in this runtime:\n{exc}\n")

    if not rows:
        raise RuntimeError("No routing results were produced.")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "motif_burden_vs_routing.csv", index=False)

    summary_lines = []
    for (topology, router_name), gdf in df.groupby(["topology", "router"]):
        order = gdf.sort_values("predicted_total_gamma")["motif"].tolist()
        summary_lines.append(f"topology={topology}, router={router_name}: burden ranking = {' < '.join(order)}")
    with open(outdir / "rankings.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary_lines) + "\n")

    for (topology, router_name), gdf in df.groupby(["topology", "router"]):
        rdir = outdir / f"{router_name}_plots_{topology}"
        rdir.mkdir(exist_ok=True)
        save_scatter(gdf, "predicted_total_gamma", "swap_count", f"{router_name} [{topology}]: predicted burden vs SWAP count", rdir / "burden_vs_swap_count.png")
        save_scatter(gdf, "predicted_total_gamma", "added_twoq_depth", f"{router_name} [{topology}]: predicted burden vs added 2Q depth", rdir / "burden_vs_added_2q_depth.png")
        save_binned_calibration(gdf, "predicted_total_gamma", "swap_count", f"{router_name} [{topology}]: binned burden vs SWAP count", rdir / "binned_burden_vs_swap_count.png")
        save_binned_calibration(gdf, "predicted_total_gamma", "added_twoq_depth", f"{router_name} [{topology}]: binned burden vs added 2Q depth", rdir / "binned_burden_vs_added_2q_depth.png")

    run_cfg = {
        "phase1b": str(args.phase1b),
        "motifs": motif_names,
        "k": float(args.k),
        "topology": str(args.topology),
        "topology_name": topology_name,
        "router": args.router,
        "fallback_topology_num_nodes": topo_g.number_of_nodes(),
        "fallback_topology_num_edges": topo_g.number_of_edges(),
    }
    with open(outdir / "run_config.json", "w", encoding="utf-8") as fh:
        json.dump(run_cfg, fh, indent=2)

    print("\n=== Phase 1C complete ===")
    print(f"Output directory: {outdir}")
    print(df[["motif", "topology", "router", "predicted_total_gamma", "swap_count", "added_twoq_depth"]].to_string(index=False))


if __name__ == "__main__":
    main()
