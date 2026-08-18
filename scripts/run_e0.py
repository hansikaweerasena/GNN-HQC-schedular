"""E0 -- can EFCL rank candidate schedules the way Qiskit Aer does?

Design (as agreed):
    100 phased circuits, 8 qubits x 20 layers, SC+NA at capacity 4+4.
    16 distinct capacity-feasible schedules per circuit:
        4  baselines            paper B1..B4  =  code b1, b3, b4, b5
        2  migration-preserving perturbations of each of B1, B2, B3  (6 total)
        6  balanced random schedules, 2 each at p_switch in {0, 0.1, 0.5}
    Score every schedule with BOTH scorers, rank within each circuit, and ask
    whether the two orderings agree.

Output:
    e0_raw.csv        1600 rows -- one per (circuit, schedule), raw values + ranks
    e0_per_circuit.csv 100 rows -- per-circuit rho, top-1/top-3, fidelity gaps
    e0_heatmap.png     16x16 EFCL-rank vs Aer-rank counts
    e0_heatmap.csv     the same matrix as numbers
    console            median Spearman rho + IQR, top-1 / top-3, dF_miss

Nothing here is filtered on outcome. Circuits are skipped only for two
STRUCTURAL reasons, both disclosed in the console output:
    - the family's own fragility validator rejects them (idle time invisible);
    - they contain a fully idle layer, which CircuitRepresentation drops,
      desyncing the EFCL layer grid from the Aer one.

Run:  python3 run_e0.py            (~5-10 min single core)
"""

import argparse
import collections
import csv
import json
import os
import random
import time

import numpy as np
import torch
from qiskit import QuantumCircuit
from scipy.stats import spearmanr

from mosaic_aer import generate, score, Module
from mosaic_aer.configs import drift_check
from mosaic_aer.hardware import TECHS, COMM
from mosaic_aer.lowering import segment_blocks

from src.circuit_representation import CircuitRepresentation
from src.cost_function import TotalCost
from src.baselines_tier1 import baseline_b1, baseline_b3
from src.baselines_tier2 import baseline_b4, baseline_b5

# ---------------------------------------------------------------------------
# Frozen experiment constants
# ---------------------------------------------------------------------------

N_QUBITS = 8
DEPTH = 20
CAP = 4                    # per module. Load-bearing: SABRE event accounting in
                           # lowering.py is only order-independent at cap<=4.
K = 2                      # techs: index 0 = sc, index 1 = na
N_CIRCUITS = 100
N_SCHEDULES = 16
FAMILY = "phased"

# Candidate-set composition. Held FIXED across circuits: dedup replaces within
# category so every circuit contributes the same mix of difficulty.
#   4 baselines + 2 perturbations of each of B1/B2/B3 + 6 random = 16
# Perturbing three anchors rather than one covers three structural regimes:
# static (B1, ~1.0 residency blocks), highly dynamic (B2, ~15.3) and
# low-movement dynamic (B3, ~2.3).
N_BASE, N_RAND = 4, 6
PERT_ANCHORS = ["B1", "B2", "B3"]
N_PERT_EACH = 2

# Stickiness ladder: exactly 2 random schedules at each of 3 rates --
# static diversity / sticky / deliberately high-movement. p=0.5 now anchors the
# low-quality end of the range, since 10 of the 16 candidates sit at or beside a
# baseline. Duplicates are resampled AT THE SAME RATE so the 2/2/2 mix holds.
P_SWITCH = [0.0, 0.0, 0.1, 0.1, 0.5, 0.5]

# DIAGNOSTIC ONLY -- never used in ranking. Ranks stay strict permutations of
# 1..16 and the heatmap contains no ties. This threshold is used in exactly one
# place: to report what fraction of top-1 disagreements involve a fidelity gap
# too small to be physically meaningful. Fixed A PRIORI, before seeing any
# result: SC's per-2Q-gate failure cost is -log(0.999) ~ 1e-3, so 1e-4 sits an
# order of magnitude below the smallest physical event the noise model has.
NEAR_TIE_EPS = 1e-4

# Perturbation operators. Hamming distance in qubit-layer assignments is the
# WRONG notion of "small" here: both scorers price movement per moved qubit, so
# a one-layer blip that splits a residency block adds migration events and
# leaves the anchor's structural class. The two operators below are chosen to
# preserve the anchor's movement structure instead:
#
#   Op B  boundary shift -- move an existing migration boundary by +-1 layer.
#         Changes WHEN the anchor migrates, not how much. dN_move = 0 except
#         when the shift merges two adjacent boundaries and cancels moves,
#         which is why dN_move is measured per schedule rather than assumed.
#   Op C  global pair swap -- exchange the full trajectories of one SC-resident
#         and one NA-resident qubit. This permutes the multiset of trajectories,
#         so dN_move = 0 identically. For a static anchor it is exactly the
#         nearest neighbour in partition space.
#
# A suffix-swap fallback was considered and dropped: it can add, remove or
# reshape movers depending on later boundaries, so it would inject a small
# subpopulation with different dN_move into an otherwise uniform near-anchor
# set. Op B u Op C offers up to ~30 candidates per anchor, far more than the 2
# needed, so the fallback is unnecessary. Circuits where fewer than 2 distinct
# perturbations exist for some anchor are skipped structurally.
OPS = ("B", "C")

CFG_PATH = "cost_config_v3.json"
OUT_DIR = "e0_out"


# ---------------------------------------------------------------------------
# Circuit / representation plumbing
# ---------------------------------------------------------------------------

def to_qc(lc):
    """LayeredCircuit -> QuantumCircuit with a full-width barrier after every
    layer. The barriers are load-bearing: without them circuit_to_dag re-layers
    ASAP and silently merges the idle structure the family exists to create."""
    qc = QuantumCircuit(lc.n_qubits)
    for lay in lc.layers:
        for g in lay:
            if g[0] == "1q":
                qc.append(g[2], [g[1]])
            else:
                qc.append(g[3], [g[1], g[2]])
        qc.barrier()
    return qc


def build_circuit_set(n_wanted, cfg, caps, depth=DEPTH, verbose=True):
    """First `n_wanted` seeds that survive both structural skips.

    Baselines are computed HERE, once, and cached on the record: the main loop
    reuses them, so the preflight distinctness assert below costs nothing extra.
    """
    recs, skip_frag, skip_idle, skip_bcol, s = [], [], [], [], 0
    while len(recs) < n_wanted:
        try:
            lc, meta, prof = generate(FAMILY, s, depth=depth)
        except ValueError:
            skip_frag.append(s); s += 1; continue

        if any(len(lay) == 0 for lay in lc.layers):
            # CircuitRepresentation drops gate-less layers, so EFCL would see
            # depth-1 layers while Aer sees depth. Skipped rather than patched.
            skip_idle.append(s); s += 1; continue

        rep = CircuitRepresentation(to_qc(lc))
        assert len(rep.layers) == lc.depth, (
            f"seed {s}: EFCL sees {len(rep.layers)} layers, Aer sees {lc.depth}. "
            "The two scorers must be given the identical layer grid.")

        base = {}
        for label, fn in [("B1", baseline_b1), ("B2", baseline_b3),
                          ("B3", baseline_b4), ("B4", baseline_b5)]:
            sch = tensors_to_sched(fn(rep, caps, cfg, K))
            if not caps_ok(sch):
                raise RuntimeError(f"seed {s}: {label} violated 4/4 capacity")
            base[label] = sch

        # -- Third structural skip. The fixed 4/4/8 composition needs four
        #    DISTINCT baselines, and there is no way to invent a replacement
        #    baseline, so a circuit on which two of B1-B4 coincide cannot fill
        #    the candidate set. Skipping is a property of the circuit and the
        #    baseline algorithms only -- neither scorer has been consulted at
        #    this point -- so it cannot bias the ranking result.
        labels = ["B1", "B2", "B3", "B4"]
        dup = [(a, b) for i, a in enumerate(labels) for b in labels[i + 1:]
               if key_of(base[a]) == key_of(base[b])]
        if dup:
            skip_bcol.append((s, dup[0])); s += 1; continue

        recs.append(dict(seed=s, lc=lc, meta=meta, prof=prof, rep=rep,
                         baselines=base))
        s += 1

    if verbose:
        print(f"  {len(recs)} circuits from seeds 0..{s-1}")
        print(f"  skipped, fragility validator : {skip_frag}")
        print(f"  skipped, all-idle layer      : {skip_idle}")
        print(f"  skipped, B1-B4 collision     : {[x[0] for x in skip_bcol]}"
              + (f"  (pairs: {[x[1] for x in skip_bcol]})" if skip_bcol else ""))
        print(f"  all {len(recs)} accepted circuits have 4 distinct baselines")
    return recs, skip_frag, skip_idle, skip_bcol


def key_of(sched):
    """Canonical hashable form, for dedup."""
    return tuple(tuple(d[q] for q in range(N_QUBITS)) for d in sched)


def encode(sched):
    """Compact CSV form: one digit per qubit, layers joined by '|'."""
    return "|".join("".join(str(d[q]) for q in range(N_QUBITS)) for d in sched)


def caps_ok(sched):
    return all(sum(1 for q in d if d[q] == 0) == CAP for d in sched)


def tensors_to_sched(tlist):
    """Baseline output (list of [N] int tensors) -> list of {q: module} dicts."""
    return [{q: int(t[q]) for q in range(N_QUBITS)} for t in tlist]


def random_balanced(depth, p_switch, rng):
    """Balanced 4/4 start; at each switch event swap one SC qubit with one NA
    qubit. Capacity is preserved by construction, so no repair step is needed --
    and no repair step means nothing in the sampler can bias toward good
    schedules."""
    qs = list(range(N_QUBITS))
    rng.shuffle(qs)
    cur = {q: (0 if q in qs[:CAP] else 1) for q in range(N_QUBITS)}
    out = [dict(cur)]
    for _ in range(1, depth):
        if rng.random() < p_switch:
            a = rng.choice([q for q in cur if cur[q] == 0])
            b = rng.choice([q for q in cur if cur[q] == 1])
            cur[a], cur[b] = 1, 0
        out.append(dict(cur))
    return out


def move_count(sched):
    """Number of MOVED QUBITS, not boundaries. Both scorers price movement per
    qubit, so a boundary at which four qubits swap is four events. This is the
    quantity the perturbation operators are designed to preserve."""
    return sum(1 for t in range(1, len(sched)) for q in range(N_QUBITS)
               if sched[t][q] != sched[t - 1][q])


def boundaries(sched):
    """Layer indices t where at least one qubit changes module between t-1, t."""
    return [t for t in range(1, len(sched))
            if any(sched[t][q] != sched[t - 1][q] for q in range(N_QUBITS))]


def op_boundary_shift(anchor, rng):
    """Op B -- move one existing migration boundary by +-1 layer.

    Implemented by copying the assignment of the layer on one side of the
    boundary onto the layer on the other, which slides the transition without
    inventing a new one. Returns None when the anchor has no boundary to shift
    (a static schedule) or the shift is out of range."""
    bs = boundaries(anchor)
    if not bs:
        return None
    t = bs[rng.randrange(len(bs))]
    out = [dict(d) for d in anchor]
    if rng.random() < 0.5:                 # shift earlier: layer t-1 adopts t
        if t - 1 < 0:
            return None
        out[t - 1] = dict(anchor[t])
    else:                                  # shift later: layer t adopts t-1
        out[t] = dict(anchor[t - 1])
    return out


def op_pair_swap(anchor, rng):
    """Op C -- exchange the entire trajectories of one SC-resident and one
    NA-resident qubit. Permuting the multiset of trajectories leaves the total
    moved-qubit count identically unchanged, and capacity is preserved at every
    layer by construction. Returns None if no differing pair exists."""
    cands = [(a, b) for a in range(N_QUBITS) for b in range(N_QUBITS)
             if a < b and any(anchor[t][a] != anchor[t][b]
                              for t in range(len(anchor)))]
    if not cands:
        return None
    a, b = cands[rng.randrange(len(cands))]
    out = []
    for d in anchor:
        e = dict(d)
        e[a], e[b] = d[b], d[a]
        out.append(e)
    return out


def perturb(anchor, rng, order):
    """Try the operators in `order` and return (schedule, op_label).

    B1 is static so Op B does not exist for it and the ladder falls straight to
    Op C; B3 has only ~2 boundaries so it may exhaust Op B and fall back too.
    Selection is entirely in schedule space -- no scorer is consulted."""
    for op in order:
        fn = op_boundary_shift if op == "B" else op_pair_swap
        cand = fn(anchor, rng)
        if cand is not None and caps_ok(cand) and key_of(cand) != key_of(anchor):
            return cand, op
    return None, None


# Per-anchor operator ladder. B1 is static, so Op B has nothing to shift and
# the ladder starts at Op C. B2 has ~15 boundaries, so Op B is plentiful. B3 has
# ~2, so it may exhaust Op B and fall through to Op C.
ANCHOR_OPS = {"B1": ("C",), "B2": ("B", "C"), "B3": ("B", "C")}


def build_schedules(rec, rng):
    """The 16 candidates for one circuit, deduplicated WITHIN CATEGORY so the
    4 / 2+2+2 / 6 composition -- and therefore the difficulty of the ranking
    task -- is identical across circuits.

    Returns (candidates, ok). ok is False when some anchor could not yield
    N_PERT_EACH distinct perturbations, in which case the circuit is skipped
    structurally by the caller. Neither scorer has run at this point."""
    out, seen = [], set()

    def add(sched, cat, label, **extra):
        k = key_of(sched)
        if k in seen or not caps_ok(sched):
            return False
        seen.add(k)
        out.append(dict(sched=sched, category=cat, label=label, **extra))
        return True

    # -- 4 baselines, computed in build_circuit_set and checked distinct there.
    #    Paper naming: B1=b1, B2=b3 (sticky), B3=b4, B4=b5. code baseline_b2
    #    (fully myopic) is unused -- it carries no paper label.
    for label in ("B1", "B2", "B3", "B4"):
        if not add(rec["baselines"][label], "baseline", label,
                   anchor="", operator="",
                   anchor_moves=move_count(rec["baselines"][label]),
                   moves=move_count(rec["baselines"][label]), d_moves=0):
            raise RuntimeError(f"seed {rec['seed']}: baseline {label} collided "
                               "despite the preflight check")

    # -- 2 migration-preserving perturbations of each of B1, B2, B3
    for anc in PERT_ANCHORS:
        base = rec["baselines"][anc]
        nb = move_count(base)
        got, tries = 0, 0
        while got < N_PERT_EACH and tries < 400:
            tries += 1
            cand, op = perturb(base, rng, ANCHOR_OPS[anc])
            if cand is None:
                continue
            if add(cand, "perturb", f"{anc}p{got}", anchor=anc, operator=op,
                   anchor_moves=nb, moves=move_count(cand),
                   d_moves=move_count(cand) - nb):
                got += 1
        if got < N_PERT_EACH:
            return None, False

    # -- 6 random along the stickiness ladder. Resample AT THE SAME p until the
    #    slot succeeds, so the mix is exactly 2/2/2 across the three rates
    #    rather than drifting whenever a draw happens to duplicate.
    for p in P_SWITCH:
        tries = 0
        while True:
            cand = random_balanced(rec["lc"].depth, p, rng)
            if add(cand, "random", f"R{p}", anchor="", operator="",
                   anchor_moves="", moves=move_count(cand), d_moves=""):
                break
            tries += 1
            if tries > 200:
                raise RuntimeError(
                    f"seed {rec['seed']}: cannot find a distinct random schedule "
                    f"at p_switch={p}")

    if len(out) != N_SCHEDULES:
        raise RuntimeError(
            f"seed {rec['seed']}: got {len(out)} distinct schedules, need {N_SCHEDULES}")
    return out, True


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def verify_config(cfg):
    """E0 is meaningless if the two scorers use different hardware numbers, so
    this is asserted at startup rather than trusted. It does NOT replace the
    standalone pre-checks -- the T1 one-hot/zero-migration ASAP comparison
    against lowering.py is run separately."""
    drift_check(verbose=False)                       # hardware.py vs techs.json
    by = {t["name"].lower(): t for t in cfg["techs"]}
    bad = []
    for nm in ("sc", "na"):
        for sec, key, attr in [("gate_fidelity", "f1q", "f1q"),
                               ("gate_fidelity", "f2q", "f2q"),
                               ("gate_time", "t1q", "t1q"),
                               ("gate_time", "t2q", "t2q"),
                               ("coherence", "T2", "T2")]:
            got, want = by[nm][sec][key], getattr(TECHS[nm], attr)
            if not np.isclose(got, want):
                bad.append(f"{nm}.{key}: EFCL={got} frozen={want}")
    for key in ("f_comm", "f_move"):
        got, want = cfg["comm"][key], COMM[key]
        if not np.isclose(got, want):
            bad.append(f"{key}: EFCL={got} frozen={want}")
    if bad:
        raise RuntimeError("EFCL config disagrees with the frozen table on "
                           + "; ".join(bad) + ". E0 cannot run.")
    if cfg["timing_model"]["mode"] != "asap":
        raise RuntimeError(f"timing_model.mode is "
                           f"{cfg['timing_model']['mode']!r}, E0 requires 'asap'")
    print("  config OK: EFCL and Aer agree on all SC/NA hardware numbers; "
          "timing_model=asap")


def efcl_cost(tc, rep, sched):
    P = []
    for d in sched:
        p = torch.zeros(N_QUBITS, K)
        for q in range(N_QUBITS):
            p[q, d[q]] = 1.0
        P.append(p)
    with torch.no_grad():
        return tc(P, [[i] for i in range(len(rep.layers))], rep)


def rank_asc(vals):
    """1 = best. Ties broken by index, deterministically -- Aer's density-matrix
    fidelity is deterministic, so exact ties are essentially measure-zero, but
    the rule keeps the rank vector a strict permutation of 1..16 either way."""
    order = sorted(range(len(vals)), key=lambda i: (vals[i], i))
    r = [0] * len(vals)
    for pos, i in enumerate(order):
        r[i] = pos + 1
    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_circuits", type=int, default=N_CIRCUITS)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--seed", type=int, default=0, help="sampler seed base")
    ap.add_argument("--config", default=CFG_PATH)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--boundary_sync", default=None,
                    choices=["off", "hard", "transfer"],
                    help="ASAP block-boundary synchronisation. 'hard' charges "
                         "non-movers' pre-migration wait; exact for one-hot P.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    print("=" * 74)
    print("E0 -- EFCL vs Aer schedule-ranking agreement")
    print("=" * 74)
    print(f"  {args.n_circuits} phased circuits, {N_QUBITS}q x {args.depth} layers, "
          f"SC+NA cap {CAP}+{CAP}, {N_SCHEDULES} schedules each")

    cfg = json.load(open(args.config))
    # E0 runs a 4+4 machine; the shipped config declares capacity 15. TotalCost
    # never reads capacity, so this affects the BASELINES only -- without it they
    # emit schedules that cannot be mapped onto the Aer machine at all.
    # The config file is the source of truth. A CLI flag overrides it only when
    # explicitly supplied, so `python run_e0.py` reproduces exactly what the
    # frozen config declares rather than silently falling back to "off".
    if args.boundary_sync is not None:
        cfg["timing_model"]["boundary_sync"] = args.boundary_sync
    boundary_sync = cfg["timing_model"].get("boundary_sync", "off")
    print(f"  boundary_sync = {boundary_sync}"
          + ("  (from --boundary_sync)" if args.boundary_sync is not None
             else "  (from config)"))
    for t in cfg["techs"]:
        t["capacity"]["max_qubits"] = CAP
    names = [t["name"] for t in cfg["techs"]]
    assert names == ["sc", "na"], f"tech order must be [sc, na], got {names}"
    caps = torch.tensor([float(CAP)] * K)

    verify_config(cfg)
    tc = TotalCost(cfg)
    tc.eval()
    mods = [Module(0, "sc", tuple(range(CAP))),
            Module(1, "na", tuple(range(CAP, N_QUBITS)))]

    print("\n-- building circuit set --")
    recs, skip_frag, skip_idle, skip_bcol = build_circuit_set(
        args.n_circuits, cfg, caps, args.depth)

    raw_rows, per_circ = [], []
    heat = np.zeros((N_SCHEDULES, N_SCHEDULES), dtype=int)
    skip_pert = []

    print("\n-- scoring --")
    for n, rec in enumerate(recs):
        rng = random.Random(args.seed * 1_000_003 + rec["seed"])
        cands, ok = build_schedules(rec, rng)
        if not ok:
            # Fourth structural skip: some anchor could not yield 2 distinct
            # migration-preserving perturbations, so the fixed composition
            # cannot be filled. Decided before either scorer runs.
            skip_pert.append(rec["seed"])
            continue

        C, F, rows = [], [], []
        for j, c in enumerate(cands):
            a = score(rec["lc"], c["sched"], mods)
            if a is None:
                raise RuntimeError(f"seed {rec['seed']} schedule {j}: Aer infeasible")
            e = efcl_cost(tc, rec["rep"], c["sched"])
            C.append(float(e["total_cost"]))
            F.append(a["fidelity"])
            rows.append(dict(
                seed=rec["seed"], boundary_sync=boundary_sync,
                schedule_id=j, category=c["category"],
                label=c["label"],
                efcl_cost=float(e["total_cost"]),
                aer_fidelity=a["fidelity"],
                neglog_aer_fidelity=-np.log(max(a["fidelity"], 1e-300)),
                efcl_makespan=float(e["makespan"]),
                aer_makespan=a["makespan"],
                efcl_exec=float(e["per_segment_exec"].sum()),
                efcl_idle=float(e["per_segment_idle"].sum()),
                efcl_comm=float(e["per_segment_comm"].sum()),
                efcl_move=float(e["per_segment_move"].sum()),
                aer_comm_count=a["comm_count"], aer_move_count=a["move_count"],
                aer_swap_count=a["swap_count"], aer_n_blocks=a["n_blocks"],
                anchor=c["anchor"], operator=c["operator"],
                anchor_moves=c["anchor_moves"], sched_moves=c["moves"],
                d_moves=c["d_moves"],
                schedule=encode(c["sched"])))

        # rank 1 = best: lowest EFCL cost, highest Aer fidelity
        r_efcl = rank_asc(C)
        r_aer = rank_asc([-f for f in F])
        for row, re_, ra in zip(rows, r_efcl, r_aer):
            row["efcl_rank"] = re_
            row["aer_rank"] = ra
            heat[re_ - 1, ra - 1] += 1
        raw_rows.extend(rows)

        rho = spearmanr(r_efcl, r_aer).correlation
        pick = r_efcl.index(1)                 # EFCL's chosen schedule
        best = int(np.argmax(F))               # Aer's actual best
        gap = F[best] - F[pick]
        per_circ.append(dict(
            seed=rec["seed"], rho=rho,
            top1=int(r_aer[pick] == 1), top3=int(r_aer[pick] <= 3),
            aer_rank_of_efcl_pick=r_aer[pick],
            dF_miss=(np.nan if r_aer[pick] == 1 else gap),
            dF_spread=max(F) - min(F),
            efcl_pick_category=rows[pick]["category"],
            aer_best_category=rows[best]["category"],
            n_hot=rec["meta"].get("n_hot"),
            n_boundaries=rec["meta"].get("n_boundaries")))

        if (n + 1) % 10 == 0:
            print(f"    {n+1:3d}/{len(recs)}  running median rho = "
                  f"{np.median([p['rho'] for p in per_circ]):.3f}  "
                  f"({time.time()-t0:.0f}s)")

    # ---- write ----
    raw_p = os.path.join(args.out, "e0_raw.csv")
    with open(raw_p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(raw_rows[0].keys()))
        w.writeheader(); w.writerows(raw_rows)

    pc_p = os.path.join(args.out, "e0_per_circuit.csv")
    with open(pc_p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_circ[0].keys()))
        w.writeheader(); w.writerows(per_circ)

    hm_p = os.path.join(args.out, "e0_heatmap.csv")
    np.savetxt(hm_p, heat, fmt="%d", delimiter=",")

    # ---- headline ----
    rhos = np.array([p["rho"] for p in per_circ], dtype=float)
    q1, med, q3 = np.percentile(rhos, [25, 50, 75])
    top1 = 100 * np.mean([p["top1"] for p in per_circ])
    top3 = 100 * np.mean([p["top3"] for p in per_circ])
    miss = np.array([p["dF_miss"] for p in per_circ], dtype=float)
    miss = miss[~np.isnan(miss)]
    spread = np.array([p["dF_spread"] for p in per_circ], dtype=float)

    print("\n" + "=" * 74)
    print("E0 RESULTS")
    print("=" * 74)
    print(f"  circuits {len(per_circ)}   schedules/circuit {N_SCHEDULES}   "
          f"rows {len(raw_rows)}")
    print(f"\n  HEADLINE")
    print(f"    median per-circuit Spearman rho   {med:.3f}   [IQR {q1:.3f}-{q3:.3f}]")
    print(f"    top-1 agreement                   {top1:.0f}%")
    print(f"    top-3 agreement                   {top3:.0f}%")
    if len(miss):
        n_tied = int((miss < NEAR_TIE_EPS).sum())
        print(f"    when EFCL's pick is not Aer-rank-1 ({len(miss)} circuits):")
        print(f"      median Aer fidelity gap         {np.median(miss):.2e}")
        print(f"      mean   Aer fidelity gap         {miss.mean():.2e}")
        print(f"      below the a-priori tie threshold ({NEAR_TIE_EPS:g}): "
              f"{n_tied}/{len(miss)}")
    else:
        print("    EFCL selected Aer's best schedule on every circuit")

    print(f"\n  CONTEXT (not headline)")
    print(f"    achievable fidelity spread per circuit: "
          f"median {np.median(spread):.2e}  min {spread.min():.2e}")
    print(f"    rho 10th pct {np.percentile(rhos,10):.3f}   "
          f"min {rhos.min():.3f}   circuits with rho<0.5: {(rhos<0.5).sum()}")
    print(f"    null expectation per heatmap cell: "
          f"{len(per_circ)/N_SCHEDULES:.2f}   diagonal mean: {np.mean(np.diag(heat)):.1f}")
    print("    perturbations, by anchor (dN_move should be 0 by construction):")
    for anc in PERT_ANCHORS:
        rs = [r for r in raw_rows if r["anchor"] == anc]
        if not rs:
            continue
        dm = np.array([r["d_moves"] for r in rs], dtype=float)
        ops = collections.Counter(r["operator"] for r in rs)
        rk = np.mean([r["aer_rank"] for r in rs])
        arank = np.mean([r["aer_rank"] for r in raw_rows if r["label"] == anc])
        print(f"      {anc}: ops {dict(ops)}  mean dN_move {dm.mean():+.2f}  "
              f"nonzero {int((dm != 0).sum())}/{len(dm)}  "
              f"mean Aer rank {rk:.1f} vs anchor {arank:.1f}")
    print(f"    skipped seeds -- fragility {len(skip_frag)}, "
          f"all-idle layer {len(skip_idle)}, baseline collision {len(skip_bcol)}, "
          f"perturbation exhaustion {len(skip_pert)}")

    make_heatmap(heat, med, q1, q3, len(per_circ), os.path.join(args.out, "e0_heatmap.png"))

    print(f"\n  wrote {raw_p}")
    print(f"        {pc_p}")
    print(f"        {hm_p}")
    print(f"  total {time.time()-t0:.0f}s")

    print("\n  Paper sentence:")
    print(f'    "Across {len(per_circ)} circuits, EFCL and Aer exhibit a median '
          f'per-circuit Spearman rank correlation of {med:.2f} '
          f'[IQR {q1:.2f}-{q3:.2f}]. The schedule ranked first by EFCL is also '
          f'Aer\'s highest-ranked schedule in {top1:.0f}% of circuits and lies '
          f'within Aer\'s top three in {top3:.0f}%."')


def make_heatmap(heat, med, q1, q3, n_circ, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  [matplotlib missing -- heatmap PNG skipped, CSV still written]")
        return

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(heat, origin="upper", cmap="viridis")
    ax.set_xlabel("Aer rank (1 = highest fidelity)")
    ax.set_ylabel("EFCL rank (1 = lowest cost)")
    ticks = np.arange(0, N_SCHEDULES, 2)
    ax.set_xticks(ticks); ax.set_xticklabels(ticks + 1)
    ax.set_yticks(ticks); ax.set_yticklabels(ticks + 1)
    ax.set_title(f"EFCL vs Aer schedule ranking\n"
                 f"median Spearman $\\rho$ = {med:.2f} [IQR {q1:.2f}-{q3:.2f}]",
                 fontsize=10)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(f"count (uniform-ranking expectation "
                 f"= {n_circ/N_SCHEDULES:.2f} per cell)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"\n  wrote {path}")


if __name__ == "__main__":
    main()
