
"""
roi_gen_diagnostics.py

High-level statistical diagnostics for ROI-based circuit generation (no cost-model evaluation).
NO CSVs are written (plots + brief console summaries + config JSON only).

Runs:
  1) Option comparison (op1/op2a/op2b/op3) with use_barriers=True (>=50 circuits per option).
  2) Option comparison with use_barriers=False (>=50 circuits per option).
  3) Parameter sweeps (each sweep value × each option, >=50 circuits per cell), use_barriers=True:
       - idle_density in {0.0, 0.1, 0.2, 0.4}
       - twoq_to_oneq_ratio in {0.5, 1.0, 2.0}
       - n_rois in {3, 4, 5, 6}
       - n_long in {0, 1, 5}
       - n_tall in {0, 1, 5}

Metrics/plots:
  (a) Core gate-count / mix
  (b) Temporal structure
  (c) Spatial structure

Usage:
  python roi_gen_diagnostics.py --outdir roi_gen_diagnostics_out --n 50 --seed_base 0 --num_qubits 20 --num_layers 60

Outputs:
  - PNG plots
  - run_config.json

Notes:
  - If barriers are present, temporal layers are extracted by splitting on barriers.
  - If barriers are absent, temporal layers are extracted via DAG parallel layers.
  - Community/modularity metrics require networkx; script will skip them if unavailable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, DefaultDict
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import os, sys

try:
    import networkx as nx
    from networkx.algorithms.community import greedy_modularity_communities
    from networkx.algorithms.community.quality import modularity as nx_modularity
    _HAS_NX = True
except Exception:
    _HAS_NX = False

from qiskit.converters import circuit_to_dag

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import generator
try:
    from src.circuit_generation import generate_roi_composed_circuit
except Exception:
    from circuit_generation import generate_roi_composed_circuit


OPTIONS = ["op1", "op2a", "op2b", "op3"]


def _qbit_index(qc, q) -> int:
    """
    Robustly map a Qubit object to its integer position in qc.qubits.
    Compatible across Qiskit versions where Qubit may not expose `.index`.
    """
    # Newer/older versions may have different attributes
    if hasattr(q, "index"):
        try:
            return int(q.index)  # type: ignore[attr-defined]
        except Exception:
            pass
    if hasattr(q, "_index"):
        try:
            return int(q._index)  # type: ignore[attr-defined]
        except Exception:
            pass
    # Fallback: QuantumCircuit.find_bit
    try:
        return int(qc.find_bit(q).index)
    except Exception:
        # Last resort: linear search (should be rare)
        for i, qq in enumerate(qc.qubits):
            if qq is q:
                return i
        raise



# ------------------------------
# Layer extraction
# ------------------------------

def _extract_layers_by_barrier(qc) -> List[List[Tuple[str, List[int]]]]:
    layers: List[List[Tuple[str, List[int]]]] = []
    cur: List[Tuple[str, List[int]]] = []
    for inst, qargs, _cargs in qc.data:
        name = inst.name
        if name == "barrier":
            layers.append(cur)
            cur = []
            continue
        qs = [_qbit_index(qc, q) for q in qargs]
        cur.append((name, qs))
    if cur:
        layers.append(cur)
    return layers


def _extract_layers_by_dag(qc) -> List[List[Tuple[str, List[int]]]]:
    dag = circuit_to_dag(qc)
    layers: List[List[Tuple[str, List[int]]]] = []
    for layer in dag.layers():
        ops = []
        g = layer["graph"]
        for node in g.op_nodes():
            name = node.op.name
            qs = [_qbit_index(qc, q) for q in node.qargs]
            ops.append((name, qs))
        layers.append(ops)
    return layers


def extract_layers(qc, prefer_barriers: bool) -> List[List[Tuple[str, List[int]]]]:
    has_barrier = any(inst.name == "barrier" for inst, *_ in qc.data)
    if prefer_barriers and has_barrier:
        return _extract_layers_by_barrier(qc)
    return _extract_layers_by_dag(qc)


# ------------------------------
# Metrics
# ------------------------------

def _is_measure(name: str) -> bool:
    return name in {"measure", "reset"}


def compute_metrics(qc, *, num_qubits: int, prefer_barriers: bool) -> Dict[str, Any]:
    n_barrier = 0
    n_meas = 0
    n_1q = 0
    n_2q = 0
    total_ops = 0

    twoq_pairs: List[Tuple[int, int]] = []
    oneq_per_qubit = np.zeros(num_qubits, dtype=int)
    twoq_per_qubit = np.zeros(num_qubits, dtype=int)

    for inst, qargs, _cargs in qc.data:
        name = inst.name
        if name == "barrier":
            n_barrier += 1
            total_ops += 1
            continue

        qs = [_qbit_index(qc, q) for q in qargs]
        if _is_measure(name):
            n_meas += 1
            total_ops += 1
            continue

        total_ops += 1
        if len(qs) == 1:
            n_1q += 1
            oneq_per_qubit[int(qs[0])] += 1
        elif len(qs) == 2:
            n_2q += 1
            a, b = int(qs[0]), int(qs[1])
            if a > b:
                a, b = b, a
            twoq_pairs.append((a, b))
            twoq_per_qubit[a] += 1
            twoq_per_qubit[b] += 1

    ratio_2q_1q = float(n_2q) / float(n_1q) if n_1q > 0 else np.nan

    layers = extract_layers(qc, prefer_barriers=prefer_barriers)
    T = len(layers)

    active_counts = np.zeros(T, dtype=int)
    n2_per_layer = np.zeros(T, dtype=int)
    edge_sets: List[set] = []
    active_mat = np.zeros((T, num_qubits), dtype=bool)

    for t, ops in enumerate(layers):
        active = set()
        edges_t = set()
        for name, qs in ops:
            if name == "barrier" or _is_measure(name):
                continue
            for q in qs:
                active.add(int(q))
            if len(qs) == 2:
                n2_per_layer[t] += 1
                a, b = int(qs[0]), int(qs[1])
                if a > b:
                    a, b = b, a
                edges_t.add((a, b))
        active_counts[t] = len(active)
        for q in active:
            active_mat[t, q] = True
        edge_sets.append(edges_t)

    # Temporal Jaccard on 2Q edge sets
    jacc = []
    for t in range(1, T):
        A = edge_sets[t - 1]
        B = edge_sets[t]
        if not A and not B:
            continue
        denom = len(A | B)
        if denom > 0:
            jacc.append(len(A & B) / denom)
    mean_jacc = float(np.mean(jacc)) if jacc else np.nan

    # Idle streaks
    streaks = []
    for q in range(num_qubits):
        run = 0
        for t in range(T):
            if not active_mat[t, q]:
                run += 1
            else:
                if run > 0:
                    streaks.append(run)
                run = 0
        if run > 0:
            streaks.append(run)

    idle_p90 = float(np.percentile(streaks, 90)) if streaks else 0.0

    # Spatial
    unique_pairs = len(set(twoq_pairs))
    denom_pairs = num_qubits * (num_qubits - 1) / 2
    graph_density = unique_pairs / denom_pairs if denom_pairs > 0 else np.nan

    if twoq_pairs:
        dists = [abs(a - b) for (a, b) in twoq_pairs]
        avg_dist = float(np.mean(dists))
        frac_long_k3 = float(np.mean([1.0 if d >= 3 else 0.0 for d in dists]))
    else:
        avg_dist = np.nan
        frac_long_k3 = np.nan

    modularity_Q = np.nan
    n_communities = np.nan
    if _HAS_NX and unique_pairs > 0:
        G = nx.Graph()
        G.add_nodes_from(range(num_qubits))
        weights = {}
        for a, b in twoq_pairs:
            weights[(a, b)] = weights.get((a, b), 0) + 1
        for (a, b), w in weights.items():
            G.add_edge(a, b, weight=w)
        try:
            comms = list(greedy_modularity_communities(G, weight="weight"))
            modularity_Q = float(nx_modularity(G, comms, weight="weight"))
            n_communities = float(len(comms))
        except Exception:
            pass

    return {
        # core
        "depth": int(qc.depth()),
        "size": int(qc.size()),
        "n_barrier": int(n_barrier),
        "n_1q": int(n_1q),
        "n_2q": int(n_2q),
        "ratio_2q_1q": ratio_2q_1q,
        "unique_2q_pairs": int(unique_pairs),

        "oneq_per_qubit_median": float(np.median(oneq_per_qubit)) if num_qubits > 0 else np.nan,
        "twoq_participation_median": float(np.median(twoq_per_qubit)) if num_qubits > 0 else np.nan,

        # temporal
        "num_extracted_layers": int(T),
        "active_qubits_per_layer_mean": float(active_counts.mean()) if T > 0 else np.nan,
        "n2_per_layer_mean": float(n2_per_layer.mean()) if T > 0 else np.nan,
        "n2_per_layer_cv": float(n2_per_layer.std() / (n2_per_layer.mean() + 1e-9)) if T > 0 else np.nan,
        "mean_temporal_jaccard_2q": float(mean_jacc),
        "idle_streak_p90": float(idle_p90),

        # spatial
        "graph_density": float(graph_density),
        "avg_2q_distance": float(avg_dist),
        "frac_long_range_k3": float(frac_long_k3),
        "modularity_Q": float(modularity_Q),
        "n_communities": float(n_communities),
    }


# ------------------------------
# Plot helpers (no seaborn, no fixed colors, no subplots)
# ------------------------------

def _values_by_group(rows: List[Dict[str, Any]], metric: str, group_key: str) -> Dict[Any, np.ndarray]:
    buckets: DefaultDict[Any, List[float]] = defaultdict(list)
    for r in rows:
        g = r[group_key]
        v = r.get(metric, np.nan)
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        buckets[g].append(float(v))
    return {k: np.asarray(v, dtype=float) for k, v in buckets.items()}


def save_boxplot(rows: List[Dict[str, Any]], metric: str, group_key: str, title: str, outpath: Path) -> None:
    grouped = _values_by_group(rows, metric, group_key)
    groups = sorted(grouped.keys(), key=lambda x: str(x))
    data = [grouped[g] for g in groups]

    plt.figure(figsize=(10, 4))
    plt.boxplot(data, labels=[str(g) for g in groups], showfliers=False)
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_hist(rows: List[Dict[str, Any]], metric: str, title: str, outpath: Path, bins: int = 30) -> None:
    vals = []
    for r in rows:
        v = r.get(metric, np.nan)
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        vals.append(float(v))
    vals = np.asarray(vals, dtype=float)

    plt.figure(figsize=(10, 4))
    plt.hist(vals, bins=bins)
    plt.xlabel(metric)
    plt.ylabel("count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_sweep_lineplot(summary: Dict[Any, Dict[Any, Tuple[float, float]]],
                        x_values: Sequence[Any],
                        title: str,
                        xlabel: str,
                        ylabel: str,
                        outpath: Path) -> None:
    """
    summary[option][x] = (mean, std)
    """
    plt.figure(figsize=(10, 4))
    for opt in sorted(summary.keys(), key=lambda x: str(x)):
        xs = list(x_values)
        ys = [summary[opt][x][0] for x in xs]
        yerr = [summary[opt][x][1] for x in xs]
        plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=str(opt))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _print_option_summary(rows: List[Dict[str, Any]], group_key: str, metrics: List[str], header: str) -> None:
    print("\n" + header)
    grouped = defaultdict(list)
    for r in rows:
        grouped[r[group_key]].append(r)
    for g in sorted(grouped.keys(), key=lambda x: str(x)):
        rs = grouped[g]
        parts = []
        for m in metrics:
            vals = [float(r[m]) for r in rs if r.get(m) is not None and not (isinstance(r.get(m), float) and np.isnan(r[m]))]
            if len(vals) == 0:
                parts.append(f"{m}=NA")
            else:
                parts.append(f"{m}={np.mean(vals):.2f}±{np.std(vals):.2f}")
        print(f"  {group_key}={g}: " + " | ".join(parts))


# ------------------------------
# Experiment runners
# ------------------------------

@dataclass
class ExperimentConfig:
    num_qubits: int = 20
    num_layers: int = 60
    n: int = 50
    seed_base: int = 0

    # Baseline params
    n_rois: int = 3
    twoq_to_oneq_ratio: float = 0.7
    idle_density: float = 0.2
    p_bridge_boundary: Tuple[float, float] = (0.10, 0.20)
    p_bridge_interior: Tuple[float, float] = (0.01, 0.05)
    noise_1q_prob: float = 0.02
    noise_2q_prob: float = 0.004
    measure_frac: float = 0.0

    min_block_w: int = 2
    max_block_w: int = 18
    min_block_h: int = 2
    max_block_h: int = 16

    n_long: Tuple[int, int] = (2, 5)
    long_w_min: int = 12
    long_w_max: int = 40

    n_tall: Tuple[int, int] = (1, 3)
    tall_h_min: int = 6
    tall_h_max: int = 10


def generate_one(cfg: ExperimentConfig, *, option: str, use_barriers: bool, seed: int, overrides: Dict[str, Any]) -> Any:
    kwargs = dict(
        num_qubits=cfg.num_qubits,
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
    kwargs.update(overrides)
    return generate_roi_composed_circuit(**kwargs)


def run_option_comparison(cfg: ExperimentConfig, outdir: Path, *, use_barriers: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    prefer_barriers = use_barriers

    for opt_i, opt in enumerate(OPTIONS):
        for i in range(cfg.n):
            seed = cfg.seed_base + opt_i * 100000 + i
            qc = generate_one(cfg, option=opt, use_barriers=use_barriers, seed=seed, overrides={})
            m = compute_metrics(qc, num_qubits=cfg.num_qubits, prefer_barriers=prefer_barriers)
            m.update({"option": opt, "use_barriers": use_barriers, "seed": seed})
            rows.append(m)

    # High-level console summary
    headline = ["n_2q", "n_1q", "ratio_2q_1q", "idle_streak_p90", "mean_temporal_jaccard_2q", "modularity_Q"]
    _print_option_summary(rows, "option", headline, header=f"[SUMMARY] Option comparison (use_barriers={use_barriers})")

    # Plots
    plots_dir = outdir / f"option_comparison_barriers_{int(use_barriers)}"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics_core = ["n_1q", "n_2q", "ratio_2q_1q", "unique_2q_pairs", "depth"]
    metrics_temp = ["active_qubits_per_layer_mean", "n2_per_layer_mean", "mean_temporal_jaccard_2q", "idle_streak_p90"]
    metrics_spat = ["graph_density", "avg_2q_distance", "frac_long_range_k3"]
    if _HAS_NX:
        metrics_spat += ["modularity_Q", "n_communities"]

    for metric in metrics_core + metrics_temp + metrics_spat:
        save_boxplot(rows, metric, "option", f"{metric} by option (use_barriers={use_barriers})",
                     plots_dir / f"box_{metric}.png")
        save_hist(rows, metric, f"{metric} pooled (use_barriers={use_barriers})",
                  plots_dir / f"hist_{metric}.png")

    return rows


def run_sweep(cfg: ExperimentConfig,
              outdir: Path,
              *,
              sweep_name: str,
              sweep_values: Sequence[Any],
              override_key: str,
              use_barriers: bool) -> None:
    prefer_barriers = use_barriers
    rows: List[Dict[str, Any]] = []

    for opt_i, opt in enumerate(OPTIONS):
        for v_i, v in enumerate(sweep_values):
            for i in range(cfg.n):
                seed = cfg.seed_base + opt_i * 100000 + v_i * 1000 + i + (abs(hash(sweep_name)) % 97)
                qc = generate_one(cfg, option=opt, use_barriers=use_barriers, seed=seed, overrides={override_key: v})
                m = compute_metrics(qc, num_qubits=cfg.num_qubits, prefer_barriers=prefer_barriers)
                m.update({"option": opt, "use_barriers": use_barriers, "seed": seed, sweep_name: v})
                rows.append(m)

    # High-level summary for one metric (quick console visibility)
    headline = ["n_2q", "ratio_2q_1q", "idle_streak_p90", "mean_temporal_jaccard_2q", "modularity_Q"]
    print(f"\n[SUMMARY] Sweep={sweep_name} (use_barriers={use_barriers})")
    for opt in OPTIONS:
        sub = [r for r in rows if r["option"] == opt]
        # print mean of ratio_2q_1q as a quick anchor
        vals = [float(r["ratio_2q_1q"]) for r in sub if not (isinstance(r["ratio_2q_1q"], float) and np.isnan(r["ratio_2q_1q"]))]
        print(f"  option={opt}: ratio_2q_1q mean={np.mean(vals):.3f} (n={len(vals)})")

    sweep_dir = outdir / f"sweep_{sweep_name}_barriers_{int(use_barriers)}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # For each metric: mean±std vs sweep value with separate lines per option
    metrics = {
        "core": ["n_1q", "n_2q", "ratio_2q_1q", "unique_2q_pairs", "depth"],
        "temporal": ["active_qubits_per_layer_mean", "n2_per_layer_mean", "mean_temporal_jaccard_2q", "idle_streak_p90"],
        "spatial": ["graph_density", "avg_2q_distance", "frac_long_range_k3"],
    }
    if _HAS_NX:
        metrics["spatial"] += ["modularity_Q", "n_communities"]

    for _cat, ms in metrics.items():
        for metric in ms:
            # summary[option][x] = (mean, std)
            summary: Dict[str, Dict[Any, Tuple[float, float]]] = {opt: {} for opt in OPTIONS}
            for opt in OPTIONS:
                for x in sweep_values:
                    vals = []
                    for r in rows:
                        if r["option"] == opt and r[sweep_name] == x:
                            v = r.get(metric, np.nan)
                            if v is None:
                                continue
                            if isinstance(v, float) and np.isnan(v):
                                continue
                            vals.append(float(v))
                    if len(vals) == 0:
                        summary[opt][x] = (np.nan, np.nan)
                    else:
                        summary[opt][x] = (float(np.mean(vals)), float(np.std(vals)))

            save_sweep_lineplot(
                summary=summary,
                x_values=sweep_values,
                title=f"{metric} vs {sweep_name} (use_barriers={use_barriers})",
                xlabel=sweep_name,
                ylabel=metric,
                outpath=sweep_dir / f"line_{metric}.png",
            )

    # Also save pooled histograms for a few headline metrics (quick glance)
    pooled_dir = sweep_dir / "pooled_hists"
    pooled_dir.mkdir(exist_ok=True)
    for metric in ["ratio_2q_1q", "idle_streak_p90", "mean_temporal_jaccard_2q", "avg_2q_distance"]:
        save_hist(rows, metric, f"{metric} pooled (sweep={sweep_name}, use_barriers={use_barriers})",
                  pooled_dir / f"hist_{metric}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="roi_gen_diagnostics_out")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed_base", type=int, default=0)
    ap.add_argument("--num_qubits", type=int, default=20)
    ap.add_argument("--num_layers", type=int, default=60)
    args = ap.parse_args()

    cfg = ExperimentConfig(
        num_qubits=args.num_qubits,
        num_layers=args.num_layers,
        n=args.n,
        seed_base=args.seed_base,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save run config JSON (no CSVs)
    (outdir / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    print("[1/3] Option comparison (use_barriers=True)")
    run_option_comparison(cfg, outdir, use_barriers=True)

    print("[2/3] Option comparison (use_barriers=False)")
    run_option_comparison(cfg, outdir, use_barriers=False)

    print("[3/3] Sweeps (use_barriers=True)")
    sweeps_dir = outdir / "sweeps_barriers_1"
    sweeps_dir.mkdir(parents=True, exist_ok=True)

    run_sweep(cfg, sweeps_dir, sweep_name="idle_density",
              sweep_values=[0.0, 0.1, 0.2, 0.4], override_key="idle_density", use_barriers=True)

    run_sweep(cfg, sweeps_dir, sweep_name="twoq_to_oneq_ratio",
              sweep_values=[0.5, 1.0, 2.0], override_key="twoq_to_oneq_ratio", use_barriers=True)

    run_sweep(cfg, sweeps_dir, sweep_name="n_rois",
              sweep_values=[3, 4, 5, 6], override_key="n_rois", use_barriers=True)

    run_sweep(cfg, sweeps_dir, sweep_name="n_long",
              sweep_values=[0, 1, 5], override_key="n_long", use_barriers=True)

    run_sweep(cfg, sweeps_dir, sweep_name="n_tall",
              sweep_values=[0, 1, 5], override_key="n_tall", use_barriers=True)

    print("Done. Plots in:", outdir.resolve())


if __name__ == "__main__":
    main()
