"""M1 -- static enumeration: does heterogeneous SC+NA beat homogeneous DQC?

For each circuit, enumerate every balanced 4/4 partition on 2xSC, 2xNA and SC+NA,
Aer-score all of them, and take each machine's best. The scheduler is held fixed
(one static assignment for the whole circuit), so nothing a scheduler does can be
credited to the hardware -- this is hardware-vs-hardware.

Symmetry: on a homogeneous machine the two modules are interchangeable, so only
C(8,4)/2 = 35 partitions are distinct. Enumerating 70 would score every one twice.
Cost per circuit is therefore 35 + 35 + 70 = 140 Aer runs, not 210.

Run
---
    # everything, one process (~2-3 h at depth 20)
    python -m mosaic_aer.exp_m1 --seeds 30 --out m1.json

    # one chunk, for SLURM array jobs
    python -m mosaic_aer.exp_m1 --seed0 $SLURM_ARRAY_TASK_ID --seeds 1 \
        --out results/m1_$SLURM_ARRAY_TASK_ID.json

    # merge chunks, write one CSV, print the report
    python -m mosaic_aer.exp_m1 --report 'results/m1_*.json' --csv m1_all.csv

Outputs
-------
JSON   full records (nested dicts), one file per chunk
CSV    one row per circuit with every raw number -- fidelity, makespan, comm
       count and chosen partition for each of the three machines, plus n_hot,
       p_cold, duty spread and the Pareto flags. Plot from this.
console summary split by n_hot (active set vs SC capacity) and by storage
       activity, with makespan reported beside fidelity.
"""

import argparse
import glob
import itertools
import json
import os
import time

import numpy as np

from .families_o import generate, generate_pair, duty_threshold, P_COLD
from .hardware import Module, TECHS, COMM
from .scoring import score

MACHINES = {"2xSC": ("sc", "sc"), "2xNA": ("na", "na"), "SC+NA": ("sc", "na")}


def _partitions(n=8, cap=4, symmetric=False):
    """All balanced partitions; half of them when the two modules are interchangeable."""
    out = []
    for grp in itertools.combinations(range(n), cap):
        if symmetric and 0 not in grp:      # canonical rep: qubit 0 always in module 0
            continue
        out.append(grp)
    return out


def best_static(lc, layout, cap=4):
    n = lc.n_qubits
    mods = [Module(0, layout[0], tuple(range(cap))),
            Module(1, layout[1], tuple(range(cap, n)))]
    best = None
    for grp in _partitions(n, cap, symmetric=(layout[0] == layout[1])):
        sch = [{q: (0 if q in grp else 1) for q in range(n)}] * lc.depth
        s = score(lc, sch, mods)
        if s and (best is None or s["fidelity"] > best["fidelity"]):
            best = dict(s, partition=list(grp))
    return best


def run_circuit(lc, meta, prof):
    res = {name: best_static(lc, layout) for name, layout in MACHINES.items()}
    homo_name = max(("2xSC", "2xNA"), key=lambda k: res[k]["fidelity"])
    homo = res[homo_name]["fidelity"]
    het = res["SC+NA"]["fidelity"]
    rel = ((1 - homo) - (1 - het)) / (1 - homo) * 100 if homo < 1 else 0.0
    # M1's gate is a fidelity win that is NOT bought with a much longer makespan:
    # a Pareto-dominated point is not a win even if its fidelity is higher.
    mk_het = res["SC+NA"]["makespan"]
    mk_homo = res[homo_name]["makespan"]
    mk_dominated = (het > homo) and (mk_het > mk_homo)
    return dict(
        seed=meta["seed"], family=meta["family"], depth=lc.depth,
        rel=rel, winner=max(res, key=lambda k: res[k]["fidelity"]),
        best_homo=homo_name,
        f={k: v["fidelity"] for k, v in res.items()},
        makespan={k: v["makespan"] for k, v in res.items()},
        mk_ratio=mk_het / mk_homo if mk_homo else float("nan"),
        mk_dominated=bool(mk_dominated),
        pareto_win=bool(het > homo and not mk_dominated),
        partition={k: v["partition"] for k, v in res.items()},
        comm={k: v["comm_count"] for k, v in res.items()},
        n_hot=meta.get("n_hot"), p_cold=meta.get("p_cold"),
        duty_spread=prof.duty_spread, mean_duty=prof.mean_duty,
        contested=prof.contested, n_2q=lc.n_2q, cut=prof.n_cross_best)


def run(seeds, seed0, depth, families, out=None, csv=None, verbose=True):
    rows, t0 = [], time.time()
    for s in range(seed0, seed0 + seeds):
        insts = []
        if "uniform" in families:
            (h, hm, hp), (u, um, up) = generate_pair(s, depth=depth)
            if "hotcore" in families:
                insts.append((h, hm, hp))
            insts.append((u, um, up))
        else:
            insts.append(generate("hotcore", s, depth=depth))

        for lc, meta, prof in insts:
            r = run_circuit(lc, meta, prof)
            rows.append(r)
            if verbose:
                print(f"  s{s:<3} {r['family']:<8} n_hot={r['n_hot'] or '-'} "
                      f"p_cold={r['p_cold'] or 0:.3f} "
                      f"F: 2xSC={r['f']['2xSC']:.4f} 2xNA={r['f']['2xNA']:.4f} "
                      f"SC+NA={r['f']['SC+NA']:.4f} ->{r['rel']:+6.1f}%  "
                      f"mk(ns): {r['makespan']['2xSC']:.0f}/{r['makespan']['2xNA']:.0f}/"
                      f"{r['makespan']['SC+NA']:.0f}  win={r['winner']}"
                      f"{'' if not r['mk_dominated'] else '  [MK-DOMINATED]'}",
                      flush=True)

    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        json.dump(dict(config=_config(depth), rows=rows), open(out, "w"), indent=1)
        csv_path = csv or (os.path.splitext(out)[0] + ".csv")
        to_csv(rows, csv_path)
        if verbose:
            print(f"\nwrote {out} and {csv_path} "
                  f"({len(rows)} rows, {time.time()-t0:.0f}s)")
    elif csv:
        to_csv(rows, csv)
        if verbose:
            print(f"\nwrote {csv} ({len(rows)} rows, {time.time()-t0:.0f}s)")
    return rows


def _config(depth):
    return dict(
        frozen_id="techs_v3", depth=depth, p_cold_range=list(P_COLD),
        sc_t2q=TECHS["sc"].t2q, sc_T2=TECHS["sc"].T2, sc_f2q=TECHS["sc"].f2q,
        na_t2q=TECHS["na"].t2q, na_T2=TECHS["na"].T2, na_f2q=TECHS["na"].f2q,
        f_comm=COMM["f_comm"], f_move=COMM["f_move"],
        duty_threshold=duty_threshold())


CSV_COLUMNS = [
    "seed", "family", "depth", "n_hot", "p_cold", "duty_spread", "mean_duty",
    "contested", "n_2q", "cut",
    "f_2xSC", "f_2xNA", "f_SCNA",
    "mk_2xSC", "mk_2xNA", "mk_SCNA",
    "comm_2xSC", "comm_2xNA", "comm_SCNA",
    "part_2xSC", "part_2xNA", "part_SCNA",
    "best_homo", "f_best_homo", "mk_best_homo",
    "rel_gain_pct", "mk_ratio", "mk_dominated", "pareto_win", "winner",
]


def to_csv(rows, path):
    """One row per circuit, every raw number. This is the artefact to plot from --
    the console summary is for reading, the CSV is for analysis."""
    import csv as _csv

    key = {"2xSC": "2xSC", "2xNA": "2xNA", "SC+NA": "SCNA"}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            bh = r["best_homo"]
            d = dict(
                seed=r["seed"], family=r["family"], depth=r["depth"],
                n_hot=r.get("n_hot"), p_cold=r.get("p_cold"),
                duty_spread=round(r["duty_spread"], 4),
                mean_duty=round(r["mean_duty"], 4),
                contested=int(bool(r["contested"])), n_2q=r["n_2q"], cut=r["cut"],
                best_homo=bh,
                f_best_homo=round(r["f"][bh], 6),
                mk_best_homo=round(r["makespan"][bh], 1),
                rel_gain_pct=round(r["rel"], 3),
                mk_ratio=round(r.get("mk_ratio", float("nan")), 4),
                mk_dominated=int(bool(r.get("mk_dominated", False))),
                pareto_win=int(bool(r.get("pareto_win", False))),
                winner=r["winner"])
            for m, tag in key.items():
                d[f"f_{tag}"] = round(r["f"][m], 6)
                d[f"mk_{tag}"] = round(r["makespan"][m], 1)
                d[f"comm_{tag}"] = r["comm"][m]
                d[f"part_{tag}"] = "|".join(str(q) for q in r["partition"][m])
            w.writerow(d)
    return path


def _group_stats(R, label):
    """One summary line for a subset of circuits."""
    rel = np.array([r["rel"] for r in R])
    wins = sum(1 for r in R if r["winner"] == "SC+NA")
    pareto = sum(1 for r in R if r.get("pareto_win"))
    se = rel.std(ddof=1) / np.sqrt(len(rel)) if len(rel) > 1 else 0.0
    mk = np.mean([r.get("mk_ratio", np.nan) for r in R])
    return (f"    {label:<16} n={len(R):<3} mean {rel.mean():+6.1f}% +/-{se:4.1f}  "
            f"median {np.median(rel):+6.1f}%  wins {wins}/{len(R)}  "
            f"pareto {pareto}/{len(R)}  mk x{mk:.2f}")


def report(rows, cfg=None):
    if cfg:
        print(f"config: frozen={cfg.get('frozen_id')} depth={cfg['depth']} "
              f"sc.t2q={cfg['sc_t2q']:g} sc.T2={cfg['sc_T2']:g} "
              f"na.f2q={cfg['na_f2q']} na.t2q={cfg['na_t2q']:g} "
              f"f_comm={cfg['f_comm']} p_cold~U{tuple(cfg['p_cold_range'])} "
              f"duty_thr={cfg['duty_threshold']*100:.0f}%")

    for fam in sorted({r["family"] for r in rows}):
        R = [r for r in rows if r["family"] == fam]
        rel = np.array([r["rel"] for r in R])
        wins = sum(1 for r in R if r["winner"] == "SC+NA")
        pareto = sum(1 for r in R if r.get("pareto_win"))
        se = rel.std(ddof=1) / np.sqrt(len(rel)) if len(rel) > 1 else 0.0

        print(f"\n{'='*78}\n{fam}  n={len(R)}\n{'='*78}")
        print(f"  SC+NA wins on fidelity   {wins}/{len(R)} ({100*wins/len(R):.0f}%)")
        print(f"  ... and NOT makespan-dominated (Pareto win)  "
              f"{pareto}/{len(R)} ({100*pareto/len(R):.0f}%)")
        print(f"  rel infidelity gain      mean {rel.mean():+.1f}% +/- {se:.1f} (SE)  "
              f"median {np.median(rel):+.1f}%  [{rel.min():+.1f}, {rel.max():+.1f}]")

        print(f"\n  {'machine':<8} {'mean best F':>12} {'mean makespan (ns)':>20}")
        for m in ("2xSC", "2xNA", "SC+NA"):
            print(f"  {m:<8} {np.mean([r['f'][m] for r in R]):>12.4f} "
                  f"{np.mean([r['makespan'][m] for r in R]):>20.0f}")
        mkr = np.array([r.get("mk_ratio", np.nan) for r in R], dtype=float)
        print(f"  SC+NA makespan / best-homogeneous:  mean x{np.nanmean(mkr):.2f}  "
              f"median x{np.nanmedian(mkr):.2f}  max x{np.nanmax(mkr):.2f}")
        bh = [r["best_homo"] for r in R]
        print(f"  stronger homogeneous baseline: 2xNA {bh.count('2xNA')}/{len(R)}, "
              f"2xSC {bh.count('2xSC')}/{len(R)}")

        # --- BY n_hot: the three regimes are qualitatively different and pooling
        # them describes none of them. Active set vs module capacity (4).
        nh = [r.get("n_hot") for r in R]
        if any(v is not None for v in nh):
            print("\n  by active-set size vs SC capacity (cap=4):")
            for k in sorted({v for v in nh if v is not None}):
                G = [r for r in R if r.get("n_hot") == k]
                tag = "under" if k < 4 else ("matched" if k == 4 else "OVER capacity")
                print(_group_stats(G, f"n_hot={k} ({tag})"))

        # --- BY storage activity: the boundary, reported rather than hidden
        pc = np.array([r.get("p_cold") or 0.0 for r in R])
        if np.ptp(pc) > 1e-9:
            print("\n  by storage-region activity:")
            for lo, hi in ((0.0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.25)):
                G = [r for r, v in zip(R, pc) if lo <= v < hi]
                if G:
                    print(_group_stats(G, f"p_cold[{lo:.2f},{hi:.2f})"))

        # --- the two cuts crossed, since they are not independent
        if any(v is not None for v in nh) and np.ptp(pc) > 1e-9:
            print("\n  n_hot x storage activity (mean rel gain %, n):")
            ks = sorted({v for v in nh if v is not None})
            bins = ((0.0, 0.07), (0.07, 0.14), (0.14, 0.25))
            print("    " + "n_hot".ljust(8)
                  + "".join(f"p_cold<{hi:.2f}".rjust(16) for _, hi in bins))
            for k in ks:
                cells = []
                for lo, hi in bins:
                    G = [r for r, v in zip(R, pc)
                         if r.get("n_hot") == k and lo <= v < hi]
                    cells.append(f"{np.mean([g['rel'] for g in G]):+.1f} (n={len(G)})"
                                 .rjust(16) if G else "-".rjust(16))
                print("    " + str(k).ljust(8) + "".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--families", default="hotcore",
                    help="comma-separated: hotcore,uniform")
    ap.add_argument("--out", default=None,
                    help="JSON path; a .csv with the same stem is written too")
    ap.add_argument("--csv", default=None,
                    help="explicit CSV path (also works with --report)")
    ap.add_argument("--report", nargs="*", default=None,
                    help="merge these JSON files and print the report")
    a = ap.parse_args()

    if a.report is not None:
        files = [f for pat in (a.report or ["m1_*.json"]) for f in sorted(glob.glob(pat))]
        rows, cfg = [], None
        for f in files:
            d = json.load(open(f))
            rows += d["rows"]
            cfg = cfg or d.get("config")
        print(f"merged {len(files)} file(s), {len(rows)} rows\n")
        if a.csv:
            to_csv(rows, a.csv)
            print(f"wrote {a.csv}\n")
        report(rows, cfg)
        return

    fams = [x.strip() for x in a.families.split(",") if x.strip()]
    print(f"M1: seeds {a.seed0}..{a.seed0+a.seeds-1}, depth {a.depth}, families {fams}")
    print(f"    {_config(a.depth)}\n")
    rows = run(a.seeds, a.seed0, a.depth, fams, a.out, a.csv)
    print()
    report(rows, _config(a.depth))


if __name__ == "__main__":
    main()
