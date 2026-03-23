#!/usr/bin/env python3
from __future__ import annotations

"""
Phase 2D: random circuit scoring vs Sabre — large-scale validation.

Runs 4 scenarios:
  1. 20q on heavy-hex (Falcon 28q, κ=2.3)
  2. 20q on grid (6x6, κ=3.3)
  3. 30q on heavy-hex (Hummingbird 65q, κ=2.3)
  4. 30q on grid (6x6, κ=3.3)

Each scenario: 1000 random circuits, 50/50 1Q:2Q ratio.
Reports Pearson + Spearman per scenario.
Saves edge-level CSVs for first 10 circuits per scenario.
"""

import argparse, importlib.util, json, math, sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None: raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
    spec.loader.exec_module(mod); return mod


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EvalConfig:
    phase1b_path: str = "scripts/phase1B_dense_cases.py"
    phase2a_path: str = "scripts/phase2A_nonlocal_cases_large.py"
    circuit_gen_path: str = "src/circuit_generation.py"
    outdir: str = "phase2D_random_vs_routing_out"

    # Per-scenario settings (populated in main loop)
    kappa: float = 2.3
    dense_lambda_decay: float = 0.3
    dense_eps: float = 1e-12

    @property
    def dense_window_radius(self): return int(math.ceil(self.kappa))
    nl_window_normalize: bool = False
    delta_community: int = 3
    pair_reuse_threshold: int = 2
    @property
    def nl_window_radius(self): return 2 * int(math.ceil(self.kappa))
    @property
    def pair_reuse_radius(self): return self.nl_window_radius
    gamma_max: float = 2.5

    # Circuit generation
    num_circuits: int = 1000
    seed_base: int = 42
    depth: int = 60
    gate_density: float = 0.3
    two_qubit_ratio: float = 0.5

    # Sabre
    sabre_seed: int = 7
    sabre_trials: int = 3  # 3 trials for speed at 1000 circuits

    # Edge-level detail for first N circuits
    detail_count: int = 10


# =============================================================================
# Scenarios
# =============================================================================

SCENARIOS = [
    {"name": "20q_heavy_hex", "num_qubits": 20, "topology": "heavy_hex", "kappa": 2.3},
    {"name": "20q_grid",      "num_qubits": 20, "topology": "grid",      "kappa": 3.3},
    {"name": "30q_heavy_hex", "num_qubits": 30, "topology": "heavy_hex", "kappa": 2.3},
    {"name": "30q_grid",      "num_qubits": 30, "topology": "grid",      "kappa": 3.3},
]


# =============================================================================
# Topology
# =============================================================================

def build_heavy_hex(min_qubits=27):
    import networkx as nx
    IBM = [(4,5,28,"falcon"), (5,9,65,"hummingbird"), (11,8,127,"eagle")]
    chosen = None
    for c,n,t,nm in IBM:
        if t >= min_qubits: chosen = (c,n,t,nm); break
    if not chosen:
        raise ValueError(f"Need {min_qubits}q, max heavy-hex is 127. Use grid.")
    cols, ndr, _, chip = chosen
    g = nx.Graph(); nid = 0; dr = []
    for r in range(ndr):
        rn = list(range(nid, nid+cols))
        for n in rn: g.add_node(n)
        for i in range(cols-1): g.add_edge(rn[i], rn[i+1])
        dr.append(rn); nid += cols
    for gi in range(ndr-1):
        u=dr[gi]; l=dr[gi+1]
        if gi%2==0: bc=list(range(0,cols,2))
        else: bc=list(range(1,cols,2))
        for c in bc:
            bn=nid; g.add_node(bn); g.add_edge(u[c],bn); g.add_edge(bn,l[c]); nid+=1
    return g, g.number_of_nodes(), chip

def build_grid(min_qubits=20):
    import networkx as nx
    side = max(5, int(math.ceil(math.sqrt(min_qubits))))
    while side * side < min_qubits: side += 1
    raw = nx.grid_2d_graph(side, side)
    mapping = {node: idx for idx, node in enumerate(sorted(raw.nodes()))}
    g = nx.relabel_nodes(raw, mapping)
    return g, g.number_of_nodes(), f"grid_{side}x{side}"

def get_topology(name, min_qubits):
    if name == "heavy_hex":
        g, n, chip = build_heavy_hex(min_qubits)
        return g, f"heavy_hex_{chip}_{n}q"
    elif name == "grid":
        g, n, label = build_grid(min_qubits)
        return g, label
    raise ValueError(f"Unknown topology: {name}")


# =============================================================================
# QuantumCircuit → MotifSpec (Qiskit DAG layering)
# =============================================================================

def qc_to_motif_spec(phase1b, qc, name):
    from qiskit.converters import circuit_to_dag
    dag = circuit_to_dag(qc)
    layers_2q = []
    for dag_layer in dag.layers():
        layer_gates = []
        for node in dag_layer["graph"].op_nodes():
            if len(node.qargs) == 2:
                q0 = qc.find_bit(node.qargs[0]).index
                q1 = qc.find_bit(node.qargs[1]).index
                layer_gates.append((int(q0), int(q1)))
        if layer_gates: layers_2q.append(layer_gates)
    if not layers_2q:
        return None
    specs = [phase1b.LayerSpec(twoq=g, label=f"{name}_L{i}") for i,g in enumerate(layers_2q)]
    fp = specs[0].twoq[0] if specs[0].twoq else (0,1)
    return phase1b.MotifSpec(name=name, num_qubits=int(qc.num_qubits),
        layers=specs, target_layer=0, target_pair=fp, notes=f"Random: {name}")


# =============================================================================
# Scoring
# =============================================================================

def _sp(u, v): return (min(u,v), max(u,v))

def compute_dense_scores(phase1b, motif, cfg):
    p1b_cfg = phase1b.DenseCaseConfig(
        window_radius=cfg.dense_window_radius,
        window_weights=[1.0]*(2*cfg.dense_window_radius+1),
        window_normalize=False,
        lambda_decay=cfg.dense_lambda_decay, eps=cfg.dense_eps)
    ec = phase1b.layer_edge_counts(motif.layers)
    eg = phase1b.build_window_effective_graphs(ec, radius=cfg.dense_window_radius,
        weights=p1b_cfg.window_weights, normalize=False)
    return phase1b.compute_dense_gate_rows(motif, ec, eg, "window", p1b_cfg, kappa=float(cfg.kappa))

def compute_nonlocal_scores(phase2a, motif, cfg):
    ec = phase2a.layer_edge_counts(motif.layers)
    w = [1.0]*(2*cfg.nl_window_radius+1)
    eg = phase2a.build_window_effective_graphs(ec, cfg.nl_window_radius, w, normalize=cfg.nl_window_normalize)
    kappa, gmax = float(cfg.kappa), float(cfg.gamma_max)
    rows = []
    for s, layer in enumerate(motif.layers):
        eff = eg[s]; va = set()
        for (a,b) in eff: va.add(a); va.add(b)
        na = len(va); lm = int(na/kappa)+1
        for pair in sorted((_sp(u,v) for u,v in layer.twoq), key=lambda p:(p[0],p[1])):
            u,v = pair; hcn = phase2a.has_common_neighbor(eff, pair); ilb = not hcn
            ld, cu, cv, reuse, pc, ppr, gnl = float("nan"), 0, 0, 0, False, False, 0.0
            if ilb:
                lr, cu, cv = phase2a.detour_metrics(eff, motif.num_qubits, pair)
                ld = float(lm) if math.isinf(lr) else float(lr)
                pc = cu >= cfg.delta_community and cv >= cfg.delta_community
                if pc:
                    reuse = phase2a.pair_reuse_count(ec, s, pair, cfg.pair_reuse_radius)
                    ppr = reuse < cfg.pair_reuse_threshold
            inl = ilb and pc and ppr
            if inl: gnl = min(max(0.0, (min(ld, float(lm))-1.0)/kappa), gmax)
            rows.append({"layer":int(s),"u":int(u),"v":int(v),"pair":f"({u},{v})",
                        "I_nonlocal":int(inl),"Gamma_nonlocal":float(gnl)})
    return pd.DataFrame(rows)

def combine_scores(ddf, ndf):
    nl = {}
    for _, r in ndf.iterrows(): nl[(int(r["layer"]),str(r["pair"]))] = (int(r["I_nonlocal"]), float(r["Gamma_nonlocal"]))
    rows = []
    for _, r in ddf.iterrows():
        k = (int(r["layer"]), str(r["pair"])); inl, gnl = nl.get(k, (0,0.0))
        gd = float(r["Gamma_dense"]); gc = gnl if inl else gd
        rows.append({"layer":int(r["layer"]),"pair":str(r["pair"]),"Gamma_dense":gd,
                     "I_nonlocal":int(inl),"Gamma_nonlocal":gnl,"Gamma_combined":gc})
    return pd.DataFrame(rows)


# =============================================================================
# Sabre routing
# =============================================================================

def try_import_qiskit():
    try:
        from qiskit import transpile; from qiskit.transpiler import CouplingMap
        return {"transpile": transpile, "CouplingMap": CouplingMap}
    except: return None

def qiskit_twoq_depth(qc):
    bu = defaultdict(int); d = 0
    for inst, qargs, _ in qc.data:
        if str(inst.name)=="barrier": continue
        qi = []
        for q in qargs:
            idx = getattr(q,"_index",None)
            if idx is None:
                try: idx = qc.find_bit(q).index
                except: idx = None
            if idx is not None: qi.append(int(idx))
        if len(qi)!=2: continue
        ly = 1+max((bu[q] for q in qi), default=0)
        for q in qi: bu[q]=ly
        d = max(d,ly)
    return int(d)

def count_twoq_ops(qc):
    return sum(1 for inst, qargs, _ in qc.data if str(inst.name)!="barrier" and len(qargs)==2)

def route_sabre(qc, topo_g, cfg):
    qk = try_import_qiskit()
    if qk is None: return None
    el = [list(map(int,e)) for e in topo_g.edges()]
    cmap = qk["CouplingMap"](couplinglist=el)
    od, oo = qiskit_twoq_depth(qc), count_twoq_ops(qc)
    sc, dc, oc = [], [], []
    for t in range(cfg.sabre_trials):
        tqc = qk["transpile"](qc, coupling_map=cmap,
            basis_gates=["cx","cz","swap","h","x","y","z","s","t","cp","measure"],
            routing_method="sabre", layout_method="sabre",
            optimization_level=0, seed_transpiler=cfg.sabre_seed+t)
        sc.append(int(tqc.count_ops().get("swap",0)))
        dc.append(qiskit_twoq_depth(tqc)); oc.append(count_twoq_ops(tqc))
    return {
        "swap_count": int(np.median(sc)),
        "added_twoq_depth": int(np.median(dc)) - od,
        "added_twoq_ops": int(np.median(oc)) - oo,
    }


# =============================================================================
# Plots
# =============================================================================

def save_scatter(df, xcol, ycol, title, outpath):
    plt.figure(figsize=(9, 6))
    x = df[xcol].to_numpy(dtype=float); y = df[ycol].to_numpy(dtype=float)
    plt.scatter(x, y, s=12, alpha=0.4, zorder=5)
    if len(df) >= 3 and np.std(x) > 1e-12:
        c = np.polyfit(x, y, 1)
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        plt.plot(xs, c[0]*xs+c[1], "--", color="red", linewidth=1.5, alpha=0.8)
        corr = np.corrcoef(x, y)[0, 1]
        try:
            from scipy import stats; sp, _ = stats.spearmanr(x, y)
            plt.title(f"{title}\nPearson r={corr:.3f}, Spearman ρ={sp:.3f}, n={len(df)}")
        except: plt.title(f"{title}\nPearson r={corr:.3f}, n={len(df)}")
    else:
        plt.title(title)
    plt.xlabel(xcol); plt.ylabel(ycol); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(outpath, dpi=220); plt.close()


# =============================================================================
# Process one scenario
# =============================================================================

def run_scenario(
    scenario: Dict[str, Any],
    phase1b, phase2a, circuit_gen,
    base_cfg: EvalConfig,
    outdir: Path,
) -> pd.DataFrame:
    sname = scenario["name"]
    nq = scenario["num_qubits"]
    topo_name = scenario["topology"]
    kappa = scenario["kappa"]

    sdir = outdir / sname
    sdir.mkdir(parents=True, exist_ok=True)
    detail_dir = sdir / "edge_details"
    detail_dir.mkdir(exist_ok=True)

    # Build config for this scenario
    cfg = EvalConfig(
        phase1b_path=base_cfg.phase1b_path,
        phase2a_path=base_cfg.phase2a_path,
        circuit_gen_path=base_cfg.circuit_gen_path,
        kappa=kappa,
        num_circuits=base_cfg.num_circuits,
        seed_base=base_cfg.seed_base,
        depth=base_cfg.depth,
        gate_density=base_cfg.gate_density,
        two_qubit_ratio=base_cfg.two_qubit_ratio,
        sabre_seed=base_cfg.sabre_seed,
        sabre_trials=base_cfg.sabre_trials,
        detail_count=base_cfg.detail_count,
        gamma_max=base_cfg.gamma_max,
    )

    # Build topology
    topo_g, topo_label = get_topology(topo_name, min_qubits=nq)
    topo_nodes = topo_g.number_of_nodes()

    print(f"\n{'='*70}")
    print(f"Scenario: {sname}")
    print(f"  {nq} qubits, κ={kappa}, topology={topo_label} ({topo_nodes} nodes)")
    print(f"  dense_window={cfg.dense_window_radius}, nl_window={cfg.nl_window_radius}")
    print(f"  {cfg.num_circuits} circuits, depth={cfg.depth}, 2Q_ratio={cfg.two_qubit_ratio}")
    print(f"{'='*70}")

    if nq > topo_nodes:
        print(f"  ERROR: {nq}q > {topo_nodes} topology nodes. Skipping.")
        return pd.DataFrame()

    rows = []
    for idx in range(cfg.num_circuits):
        seed = cfg.seed_base + nq * 10000 + idx
        cid = f"{sname}_{idx:04d}"

        # Progress every 100
        if idx % 100 == 0:
            print(f"  [{idx}/{cfg.num_circuits}] ...", flush=True)

        # 1. Generate
        try:
            qc = circuit_gen.generate_random_circuit_custom(
                num_qubits=nq, depth=cfg.depth,
                gate_density=cfg.gate_density,
                two_qubit_ratio=cfg.two_qubit_ratio,
                use_barriers=False, seed=seed)
        except Exception as e:
            continue

        # 2. Convert
        try:
            motif = qc_to_motif_spec(phase1b, qc, cid)
        except Exception:
            continue
        if motif is None or not motif.layers:
            continue

        n2q = sum(len(l.twoq) for l in motif.layers)
        if n2q == 0:
            continue

        # 3. Score
        try:
            ddf = compute_dense_scores(phase1b, motif, cfg)
            ndf = compute_nonlocal_scores(phase2a, motif, cfg)
            cdf = combine_scores(ddf, ndf)
        except Exception:
            continue

        td = float(ddf["Gamma_dense"].sum())
        tnl = float(cdf.loc[cdf["I_nonlocal"]==1, "Gamma_nonlocal"].sum()) if "I_nonlocal" in cdf.columns else 0.0
        tc = float(cdf["Gamma_combined"].sum())
        nnl = int(ndf["I_nonlocal"].sum()) if "I_nonlocal" in ndf.columns else 0

        # Save edge-level CSV for first N circuits
        if idx < cfg.detail_count:
            cdf.to_csv(detail_dir / f"{cid}_edges.csv", index=False)

        # 4. Route
        sr = route_sabre(qc, topo_g, cfg)

        row = {
            "circuit_id": cid, "scenario": sname, "num_qubits": nq,
            "kappa": kappa, "topology": topo_label,
            "seed": seed, "num_2q_gates": n2q,
            "num_nonlocal_edges": nnl,
            "total_gamma_dense_only": td,
            "total_gamma_nl_only": tnl,
            "total_gamma_combined": tc,
        }
        if sr is not None:
            row.update(sr)
        else:
            row.update({"swap_count":-1, "added_twoq_depth":-1, "added_twoq_ops":-1})

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(sdir / "results.csv", index=False)

    # Per-scenario plots
    valid = df[df["swap_count"] >= 0].copy() if "swap_count" in df.columns else df
    if len(valid) >= 3 and "swap_count" in valid.columns:
        save_scatter(valid, "total_gamma_combined", "swap_count",
                    f"[{sname}] Combined Γ vs Sabre SWAPs", sdir / "combined_vs_swaps.png")
        save_scatter(valid, "total_gamma_combined", "added_twoq_depth",
                    f"[{sname}] Combined Γ vs added 2Q depth", sdir / "combined_vs_depth.png")

    # Per-scenario correlation
    if len(valid) >= 3 and "swap_count" in valid.columns:
        x = valid["total_gamma_combined"].to_numpy(dtype=float)
        y_sw = valid["swap_count"].to_numpy(dtype=float)
        y_dp = valid["added_twoq_depth"].to_numpy(dtype=float)
        if np.std(x) > 1e-12:
            pr_sw = np.corrcoef(x, y_sw)[0, 1]
            pr_dp = np.corrcoef(x, y_dp)[0, 1]
            try:
                from scipy import stats
                sp_sw, _ = stats.spearmanr(x, y_sw)
                sp_dp, _ = stats.spearmanr(x, y_dp)
            except:
                sp_sw, sp_dp = float("nan"), float("nan")

            corr_info = {
                "scenario": sname, "n_circuits": len(valid),
                "pearson_vs_swaps": pr_sw, "spearman_vs_swaps": sp_sw,
                "pearson_vs_depth": pr_dp, "spearman_vs_depth": sp_dp,
            }
            with open(sdir / "correlation.json", "w") as f:
                json.dump(corr_info, f, indent=2)

            print(f"\n  --- {sname} results ({len(valid)} circuits) ---")
            print(f"  Γ vs SWAPs:  Pearson r={pr_sw:.3f}, Spearman ρ={sp_sw:.3f}")
            print(f"  Γ vs depth:  Pearson r={pr_dp:.3f}, Spearman ρ={sp_dp:.3f}")

            # Calibration
            alpha, beta = np.polyfit(x, y_sw, 1)
            print(f"  Calibration: SWAPs ≈ {alpha:.2f} × Γ + {beta:.2f}")

    return df


# =============================================================================
# Main
# =============================================================================

def parse_args():
    _d = EvalConfig()
    p = argparse.ArgumentParser(description="Phase 2D: large-scale random circuit validation")
    p.add_argument("--phase1b", type=str, default=_d.phase1b_path)
    p.add_argument("--phase2a", type=str, default=_d.phase2a_path)
    p.add_argument("--circuit-gen", type=str, default=_d.circuit_gen_path)
    p.add_argument("--outdir", type=str, default=_d.outdir)
    p.add_argument("--num-circuits", type=int, default=_d.num_circuits)
    p.add_argument("--seed-base", type=int, default=_d.seed_base)
    p.add_argument("--depth", type=int, default=_d.depth)
    p.add_argument("--gate-density", type=float, default=_d.gate_density)
    p.add_argument("--two-qubit-ratio", type=float, default=_d.two_qubit_ratio)
    p.add_argument("--gamma-max", type=float, default=_d.gamma_max)
    p.add_argument("--sabre-seed", type=int, default=_d.sabre_seed)
    p.add_argument("--sabre-trials", type=int, default=_d.sabre_trials)
    p.add_argument("--detail-count", type=int, default=_d.detail_count)
    return p.parse_args()


def main():
    args = parse_args()
    base_cfg = EvalConfig(
        phase1b_path=args.phase1b, phase2a_path=args.phase2a,
        circuit_gen_path=args.circuit_gen, outdir=args.outdir,
        num_circuits=args.num_circuits, seed_base=args.seed_base,
        depth=args.depth, gate_density=args.gate_density,
        two_qubit_ratio=args.two_qubit_ratio, gamma_max=args.gamma_max,
        sabre_seed=args.sabre_seed, sabre_trials=args.sabre_trials,
        detail_count=args.detail_count,
    )
    outdir = Path(base_cfg.outdir); outdir.mkdir(parents=True, exist_ok=True)

    phase1b = _load_module("phase1B_dense_cases", Path(base_cfg.phase1b_path))
    phase2a = _load_module("phase2A_nonlocal_cases_large", Path(base_cfg.phase2a_path))
    circuit_gen = _load_module("circuit_generation", Path(base_cfg.circuit_gen_path))

    # Run all 4 scenarios
    all_dfs = []
    corr_rows = []

    for scenario in SCENARIOS:
        df = run_scenario(scenario, phase1b, phase2a, circuit_gen, base_cfg, outdir)
        if not df.empty:
            all_dfs.append(df)

    # Combine all scenarios
    if not all_dfs:
        print("No results."); return
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df.to_csv(outdir / "all_results.csv", index=False)

    # Cross-scenario summary table
    print("\n" + "=" * 90)
    print("Phase 2D — Cross-Scenario Summary")
    print("=" * 90)

    summary_rows = []
    for scenario in SCENARIOS:
        sname = scenario["name"]
        sdf = full_df[full_df["scenario"] == sname]
        valid = sdf[sdf["swap_count"] >= 0] if "swap_count" in sdf.columns else sdf

        if len(valid) < 3:
            continue

        x = valid["total_gamma_combined"].to_numpy(dtype=float)
        y_sw = valid["swap_count"].to_numpy(dtype=float)
        y_dp = valid["added_twoq_depth"].to_numpy(dtype=float)

        pr_sw = np.corrcoef(x, y_sw)[0, 1] if np.std(x) > 1e-12 else 0.0
        pr_dp = np.corrcoef(x, y_dp)[0, 1] if np.std(x) > 1e-12 else 0.0
        try:
            from scipy import stats
            sp_sw, _ = stats.spearmanr(x, y_sw)
            sp_dp, _ = stats.spearmanr(x, y_dp)
        except:
            sp_sw, sp_dp = 0.0, 0.0

        alpha, _ = np.polyfit(x, y_sw, 1) if np.std(x) > 1e-12 else (0.0, 0.0)

        summary_rows.append({
            "scenario": sname,
            "n_circuits": len(valid),
            "kappa": scenario["kappa"],
            "topology": scenario["topology"],
            "pearson_swaps": pr_sw,
            "spearman_swaps": sp_sw,
            "pearson_depth": pr_dp,
            "spearman_depth": sp_dp,
            "alpha_calibration": alpha,
            "mean_gamma": float(np.mean(x)),
            "mean_swaps": float(np.mean(y_sw)),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "scenario_summary.csv", index=False)
    print(summary_df.to_string(index=False))

    # Save config
    cfg_out = {
        "scenarios": SCENARIOS,
        "num_circuits": base_cfg.num_circuits,
        "depth": base_cfg.depth,
        "gate_density": base_cfg.gate_density,
        "two_qubit_ratio": base_cfg.two_qubit_ratio,
        "gamma_max": base_cfg.gamma_max,
        "sabre_trials": base_cfg.sabre_trials,
        "seed_base": base_cfg.seed_base,
        "detail_count": base_cfg.detail_count,
    }
    with open(outdir / "config.json", "w") as f:
        json.dump(cfg_out, f, indent=2)


if __name__ == "__main__":
    main()
