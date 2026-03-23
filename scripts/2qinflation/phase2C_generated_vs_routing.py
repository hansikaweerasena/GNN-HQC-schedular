#!/usr/bin/env python3
from __future__ import annotations

"""
Phase 2C: dense + non-local scoring vs Sabre on generated circuits.

Circuit generation uses CIRCUIT_SOURCE_CFG from scheduler_config.py
via GeneratedCircuitProvider from circuit_sources.py.

Window size relationships (derived from kappa):
  dense_window_radius  = kappa
  nl_window_radius     = 2 * kappa
  pair_reuse_radius    = nl_window_radius
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

@dataclass
class EvalConfig:
    phase1b_path: str = "scripts/phase1B_dense_cases.py"
    phase2a_path: str = "scripts/phase2A_nonlocal_cases_large.py"
    circuit_sources_path: str = "utils/circuit_sources.py"
    scheduler_config_path: str = "configs/scheduler_config.py"
    outdir: str = "phase2C_generated_vs_routing_out"
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
    gamma_max: float = 4
    num_circuits: int = 1000
    seed_base: int = 42
    num_qubits_override: Optional[int] = None
    num_layers_override: Optional[int] = None
    sabre_seed: int = 7
    sabre_trials: int = 5
    topology: str = "heavy_hex"

def build_heavy_hex(min_qubits: int = 27):
    """
    Build a heavy-hex topology at standard IBM chip sizes.
    Snaps UP to the nearest real IBM heavy-hex size:
      - 28  (Falcon-class,  cols=4, rows=5)
      - 65  (Hummingbird,   cols=5, rows=9)
      - 127 (Eagle,         cols=11, rows=8)

    Max degree 3. Bridge qubits degree 2, data qubits degree 2-3.
    """
    import networkx as nx

    # (cols, n_data_rows, total_qubits, chip_name)
    IBM_CONFIGS = [
        (4,  5,  28,  "falcon"),
        (5,  9,  65,  "hummingbird"),
        (11, 8,  127, "eagle"),
    ]

    # Snap to smallest config that fits
    chosen = None
    for cols, ndr, total, name in IBM_CONFIGS:
        if total >= min_qubits:
            chosen = (cols, ndr, total, name)
            break

    if chosen is None:
        raise ValueError(
            f"Need {min_qubits} qubits but largest heavy-hex is 127 (Eagle). "
            f"Use --topology grid for larger circuits."
        )

    cols, n_data_rows, expected_total, chip_name = chosen

    g = nx.Graph()
    node_id = 0

    data_rows: list = []
    for r in range(n_data_rows):
        row_nodes = list(range(node_id, node_id + cols))
        for n in row_nodes:
            g.add_node(n)
        for i in range(cols - 1):
            g.add_edge(row_nodes[i], row_nodes[i + 1])
        data_rows.append(row_nodes)
        node_id += cols

    for gap_idx in range(n_data_rows - 1):
        upper = data_rows[gap_idx]
        lower = data_rows[gap_idx + 1]
        if gap_idx % 2 == 0:
            bridge_cols = list(range(0, cols, 2))
        else:
            bridge_cols = list(range(1, cols, 2))
        for c in bridge_cols:
            bridge_node = node_id
            g.add_node(bridge_node)
            g.add_edge(upper[c], bridge_node)
            g.add_edge(bridge_node, lower[c])
            node_id += 1

    actual_total = g.number_of_nodes()
    return g, actual_total, chip_name

def build_grid(n=6):
    import networkx as nx
    raw = nx.grid_2d_graph(n, n)
    mapping = {node: idx for idx, node in enumerate(sorted(raw.nodes()))}
    return nx.relabel_nodes(raw, mapping)

def select_topology(name, min_nodes=27):
    name = name.strip().lower()
    if name == "heavy_hex":
        g, total, chip = build_heavy_hex(min_qubits=min_nodes)
        return g, f"heavy_hex_{chip}_{total}q"
    if name == "grid":
        side = max(6, int(np.ceil(np.sqrt(min_nodes + 1))))
        return build_grid(side), f"grid_{side}x{side}"
    raise ValueError(f"Unknown topology: {name}")

def qc_to_motif_spec(phase1b, qc, name):
    """
    Convert a Qiskit QuantumCircuit into a MotifSpec.

    Uses Qiskit's DAG-based layering (circuit_to_dag + dag.layers()) to
    extract proper parallel layers. This works correctly on barrier-free
    circuits — no greedy heuristics needed.
    """
    from qiskit.converters import circuit_to_dag

    dag = circuit_to_dag(qc)
    layers_2q: List[List[Tuple[int, int]]] = []

    for dag_layer in dag.layers():
        layer_gates: List[Tuple[int, int]] = []
        for node in dag_layer["graph"].op_nodes():
            if len(node.qargs) == 2:
                q0 = qc.find_bit(node.qargs[0]).index
                q1 = qc.find_bit(node.qargs[1]).index
                layer_gates.append((int(q0), int(q1)))
        if layer_gates:
            layers_2q.append(layer_gates)

    if not layers_2q:
        # Empty circuit — return minimal spec
        return phase1b.MotifSpec(
            name=name, num_qubits=int(qc.num_qubits),
            layers=[phase1b.LayerSpec(twoq=[], label=f"{name}_empty")],
            target_layer=0, target_pair=(0, 1), notes=f"Generated (empty): {name}",
        )

    specs = [
        phase1b.LayerSpec(twoq=gates, label=f"{name}_L{i}")
        for i, gates in enumerate(layers_2q)
    ]
    fp = specs[0].twoq[0] if specs[0].twoq else (0, 1)
    return phase1b.MotifSpec(
        name=name, num_qubits=int(qc.num_qubits), layers=specs,
        target_layer=0, target_pair=fp, notes=f"Generated: {name}",
    )

def _sp(u, v): return (min(u,v), max(u,v))

def compute_dense_scores(phase1b, motif, cfg):
    p1b_cfg = phase1b.DenseCaseConfig(window_radius=cfg.dense_window_radius, window_weights=[1.0]*(2*cfg.dense_window_radius+1), window_normalize=False, lambda_decay=cfg.dense_lambda_decay, eps=cfg.dense_eps)
    ec = phase1b.layer_edge_counts(motif.layers)
    eg = phase1b.build_window_effective_graphs(ec, radius=cfg.dense_window_radius, weights=p1b_cfg.window_weights, normalize=False)
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
            rows.append({"layer":int(s),"u":int(u),"v":int(v),"pair":f"({u},{v})","I_nonlocal":int(inl),"Gamma_nonlocal":float(gnl)})
    return pd.DataFrame(rows)

def combine_scores(ddf, ndf):
    nl = {}
    for _, r in ndf.iterrows(): nl[(int(r["layer"]),str(r["pair"]))] = (int(r["I_nonlocal"]), float(r["Gamma_nonlocal"]))
    rows = []
    for _, r in ddf.iterrows():
        k = (int(r["layer"]), str(r["pair"])); inl, gnl = nl.get(k, (0,0.0))
        gd = float(r["Gamma_dense"]); gc = gnl if inl else gd
        rows.append({"layer":int(r["layer"]),"pair":str(r["pair"]),"Gamma_dense":gd,"I_nonlocal":int(inl),"Gamma_nonlocal":gnl,"Gamma_combined":gc})
    return pd.DataFrame(rows)

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
        tqc = qk["transpile"](qc, coupling_map=cmap, basis_gates=["cx","cz","swap","h","x","y","z","s","t","cp","measure","rx","ry","rz"], routing_method="sabre", layout_method="sabre", optimization_level=0, seed_transpiler=cfg.sabre_seed+t)
        sc.append(int(tqc.count_ops().get("swap",0))); dc.append(qiskit_twoq_depth(tqc)); oc.append(count_twoq_ops(tqc))
    return {"swap_count":int(np.median(sc)),"added_twoq_depth":int(np.median(dc))-od,"added_twoq_ops":int(np.median(oc))-oo}

def save_scatter(df, xcol, ycol, title, outpath, annotate=True):
    plt.figure(figsize=(9,6)); x=df[xcol].to_numpy(dtype=float); y=df[ycol].to_numpy(dtype=float)
    plt.scatter(x,y,s=40,alpha=0.7,zorder=5)
    if annotate and len(df)<=30:
        for _,r in df.iterrows(): plt.annotate(str(r.get("circuit_id","")), (float(r[xcol]),float(r[ycol])), fontsize=6, xytext=(3,3), textcoords="offset points")
    if len(df)>=3 and np.std(x)>1e-12:
        c=np.polyfit(x,y,1); xs=np.linspace(float(np.min(x)),float(np.max(x)),100)
        plt.plot(xs,c[0]*xs+c[1],"--",color="gray",alpha=0.7); corr=np.corrcoef(x,y)[0,1]
        try:
            from scipy import stats; sp,_=stats.spearmanr(x,y)
            plt.title(f"{title}\nPearson r={corr:.3f}, Spearman ρ={sp:.3f}")
        except: plt.title(f"{title}\nPearson r={corr:.3f}")
    else: plt.title(title)
    plt.xlabel(xcol);plt.ylabel(ycol);plt.grid(True,alpha=0.3);plt.tight_layout();plt.savefig(outpath,dpi=220);plt.close()

def save_comparison_3panel(df, outpath):
    fig,axes=plt.subplots(1,3,figsize=(18,5.5))
    for ax,xc,xl in [(axes[0],"total_gamma_dense_only","Σ Gamma_dense"),(axes[1],"total_gamma_combined","Σ Gamma_combined"),(axes[2],"num_nonlocal_edges","# NL edges")]:
        x=df[xc].to_numpy(dtype=float);y=df["swap_count"].to_numpy(dtype=float);ax.scatter(x,y,s=30,alpha=0.7)
        if len(df)>=3 and np.std(x)>1e-12:
            c=np.polyfit(x,y,1);xs=np.linspace(float(np.min(x)),float(np.max(x)),100)
            ax.plot(xs,c[0]*xs+c[1],"--",color="gray",alpha=0.7);ax.set_title(f"r={np.corrcoef(x,y)[0,1]:.3f}")
        ax.set_xlabel(xl);ax.set_ylabel("Sabre SWAPs");ax.grid(True,alpha=0.3)
    fig.suptitle("Predicted vs Sabre SWAPs (generated circuits)",fontsize=13);fig.tight_layout();fig.savefig(outpath,dpi=220);plt.close(fig)

def parse_args():
    _d=EvalConfig()
    p=argparse.ArgumentParser(description="Phase 2C: scoring vs Sabre on generated circuits")
    p.add_argument("--phase1b",type=str,default=_d.phase1b_path)
    p.add_argument("--phase2a",type=str,default=_d.phase2a_path)
    p.add_argument("--circuit-sources",type=str,default=_d.circuit_sources_path)
    p.add_argument("--scheduler-config",type=str,default=_d.scheduler_config_path)
    p.add_argument("--outdir",type=str,default=_d.outdir)
    p.add_argument("--kappa",type=float,default=_d.kappa)
    p.add_argument("--topology",type=str,default=_d.topology,choices=["heavy_hex","grid"])
    p.add_argument("--dense-lambda-decay",type=float,default=_d.dense_lambda_decay)
    p.add_argument("--delta-community",type=int,default=_d.delta_community)
    p.add_argument("--pair-reuse-threshold",type=int,default=_d.pair_reuse_threshold)
    p.add_argument("--gamma-max",type=float,default=_d.gamma_max)
    p.add_argument("--num-circuits",type=int,default=_d.num_circuits)
    p.add_argument("--seed-base",type=int,default=_d.seed_base)
    p.add_argument("--num-qubits",type=int,default=None,help="Override num_qubits from scheduler_config")
    p.add_argument("--num-layers",type=int,default=None,help="Override num_layers from scheduler_config")
    p.add_argument("--sabre-seed",type=int,default=_d.sabre_seed)
    p.add_argument("--sabre-trials",type=int,default=_d.sabre_trials)
    return p.parse_args()

def main():
    args=parse_args()
    cfg=EvalConfig(phase1b_path=args.phase1b,phase2a_path=args.phase2a,circuit_sources_path=args.circuit_sources,scheduler_config_path=args.scheduler_config,outdir=args.outdir,kappa=args.kappa,topology=args.topology,dense_lambda_decay=args.dense_lambda_decay,delta_community=args.delta_community,pair_reuse_threshold=args.pair_reuse_threshold,gamma_max=args.gamma_max,num_circuits=args.num_circuits,seed_base=args.seed_base,num_qubits_override=args.num_qubits,num_layers_override=args.num_layers,sabre_seed=args.sabre_seed,sabre_trials=args.sabre_trials)
    outdir=Path(cfg.outdir);outdir.mkdir(parents=True,exist_ok=True)

    print(f"kappa = {cfg.kappa}")
    print(f"dense_window_radius = {cfg.dense_window_radius} (= kappa)")
    print(f"nl_window_radius    = {cfg.nl_window_radius} (= 2*kappa)")
    print(f"pair_reuse_radius   = {cfg.pair_reuse_radius} (= nl_window_radius)")

    phase1b=_load_module("phase1B_dense_cases",Path(cfg.phase1b_path))
    phase2a=_load_module("phase2A_nonlocal_cases_large",Path(cfg.phase2a_path))
    sched_cfg_mod=_load_module("scheduler_config",Path(cfg.scheduler_config_path))
    circuit_sources_mod=_load_module("circuit_sources",Path(cfg.circuit_sources_path))

    CIRCUIT_SOURCE_CFG=sched_cfg_mod.CIRCUIT_SOURCE_CFG
    source_kwargs=dict(CIRCUIT_SOURCE_CFG.get("kwargs",{}))

    # Force use_barriers=False (realistic — no real circuit has barriers)
    source_kwargs["use_barriers"] = False

    if cfg.num_qubits_override is not None: source_kwargs["num_qubits"]=cfg.num_qubits_override
    if cfg.num_layers_override is not None:
        if "num_layers" in source_kwargs: source_kwargs["num_layers"]=cfg.num_layers_override
        elif "depth" in source_kwargs: source_kwargs["depth"]=cfg.num_layers_override
    nq=source_kwargs.get("num_qubits",20)
    print(f"Source: {CIRCUIT_SOURCE_CFG['name']}, num_qubits={nq}")
    print(f"use_barriers=False (layering via Qiskit DAG)")

    # Force 1:1:1:1 option mix for balanced evaluation across all ROI options
    forced_sampled_kwargs = {
        "option_mix": {
            "op1": 0.25,
            "op2a": 0.25,
            "op2b": 0.25,
            "op3": 0.25,
        }
    }

    provider=circuit_sources_mod.GeneratedCircuitProvider(
        source_name=CIRCUIT_SOURCE_CFG["name"],
        source_kwargs=source_kwargs,
        seed_base=cfg.seed_base,
        two_qubit_bounds=CIRCUIT_SOURCE_CFG.get("two_qubit_bounds",None),
        sampled_kwargs=forced_sampled_kwargs,
    )

    topo_g,topo_name=select_topology(cfg.topology,min_nodes=nq)
    print(f"Topology: {topo_name} ({topo_g.number_of_nodes()} nodes)")
    if nq>topo_g.number_of_nodes():
        print(f"ERROR: {nq} qubits > {topo_g.number_of_nodes()} topology nodes. Use --num-qubits or --topology grid."); return

    summary_rows=[];detail_dir=outdir/"circuit_details";detail_dir.mkdir(exist_ok=True)

    # Pre-compute which ROI option each circuit will use (replicate provider's sampling)
    options_list = ["op1", "op2a", "op2b", "op3"]
    def get_option_for_idx(idx):
        """Replicate provider's sampling to know which option was used."""
        seed = cfg.seed_base + idx
        rng = np.random.RandomState(seed + 10000)
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        return options_list[int(rng.choice(len(options_list), p=probs))]

    for idx in range(cfg.num_circuits):
        cid=f"circ_{idx:03d}"
        variant = get_option_for_idx(idx)
        print(f"  [{cid}] ({variant}) ...",end=" ",flush=True)
        try: qc=provider.get(idx)
        except Exception as e: print(f"FAIL (gen: {e})"); continue
        try: motif=qc_to_motif_spec(phase1b,qc,cid)
        except Exception as e: print(f"FAIL (conv: {e})"); continue
        if not motif.layers or all(len(l.twoq)==0 for l in motif.layers): print("SKIP (no 2Q)"); continue

        n2q=sum(len(l.twoq) for l in motif.layers)
        try: ddf=compute_dense_scores(phase1b,motif,cfg)
        except Exception as e: print(f"FAIL (dense: {e})"); continue
        try: ndf=compute_nonlocal_scores(phase2a,motif,cfg)
        except Exception as e: print(f"FAIL (nl: {e})"); continue

        cdf=combine_scores(ddf,ndf)
        td=float(ddf["Gamma_dense"].sum())
        tnl=float(cdf.loc[cdf["I_nonlocal"]==1,"Gamma_nonlocal"].sum()) if "I_nonlocal" in cdf.columns else 0.0
        tc=float(cdf["Gamma_combined"].sum())
        nnl=int(ndf["I_nonlocal"].sum()) if "I_nonlocal" in ndf.columns else 0

        cd=detail_dir/cid;cd.mkdir(exist_ok=True);cdf.to_csv(cd/"combined_scores.csv",index=False)
        sr=route_sabre(qc,topo_g,cfg)
        row={"circuit_id":cid,"source":CIRCUIT_SOURCE_CFG["name"],"variant":variant,"seed":cfg.seed_base+idx,"num_qubits":motif.num_qubits,"num_layers":len(motif.layers),"num_2q_gates":n2q,"num_nonlocal_edges":nnl,"total_gamma_dense_only":td,"total_gamma_nl_only":tnl,"total_gamma_combined":tc}
        if sr is not None:
            row.update(sr);print(f"OK ({n2q} 2Q, NL={nnl}, Γ={tc:.1f}, SWAPs={sr['swap_count']})")
        else:
            row.update({"swap_count":-1,"added_twoq_depth":-1,"added_twoq_ops":-1});print(f"OK ({n2q} 2Q, NL={nnl}, Γ={tc:.1f}, no Sabre)")
        summary_rows.append(row)

    if not summary_rows: print("No results."); return
    df=pd.DataFrame(summary_rows);df.to_csv(outdir/"summary.csv",index=False)

    has_sabre="swap_count" in df.columns and (df["swap_count"]>0).any()
    if has_sabre:
        v=df[df["swap_count"]>=0].copy()
        save_comparison_3panel(v,outdir/"comparison_3panel.png")
        save_scatter(v,"total_gamma_combined","swap_count","Combined vs SWAPs",outdir/"combined_vs_swaps.png")
        save_scatter(v,"total_gamma_combined","added_twoq_depth","Combined vs added 2Q depth",outdir/"combined_vs_depth.png")
        save_scatter(v,"total_gamma_dense_only","swap_count","Dense-only vs SWAPs",outdir/"dense_only_vs_swaps.png")
        save_scatter(v,"num_2q_gates","swap_count","# 2Q gates vs SWAPs",outdir/"gate_count_vs_swaps.png")

    cfg_out={"kappa":cfg.kappa,"dense_window_radius":cfg.dense_window_radius,"nl_window_radius":cfg.nl_window_radius,"pair_reuse_radius":cfg.pair_reuse_radius,"delta_community":cfg.delta_community,"pair_reuse_threshold":cfg.pair_reuse_threshold,"gamma_max":cfg.gamma_max,"topology":topo_name,"source":CIRCUIT_SOURCE_CFG["name"],"num_circuits":cfg.num_circuits,"seed_base":cfg.seed_base,"sabre_trials":cfg.sabre_trials}
    with open(outdir/"config.json","w") as f: json.dump(cfg_out,f,indent=2)

    print("\n"+"="*90);print("Phase 2C Summary");print("="*90)
    pc=["circuit_id","variant","num_2q_gates","num_nonlocal_edges","total_gamma_combined"]
    if has_sabre: pc+=["swap_count","added_twoq_depth"]
    print(df[pc].to_string(index=False))
    if has_sabre and len(v)>=3:
        x=v["total_gamma_combined"].to_numpy(dtype=float);y=v["swap_count"].to_numpy(dtype=float)
        if np.std(x)>1e-12:
            pr=np.corrcoef(x,y)[0,1]
            try:
                from scipy import stats;sp,_=stats.spearmanr(x,y);print(f"\nCorrelation: Pearson r={pr:.3f}, Spearman ρ={sp:.3f}")
            except: print(f"\nCorrelation: Pearson r={pr:.3f}")

if __name__=="__main__": main()
