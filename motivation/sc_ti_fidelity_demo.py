#!/usr/bin/env python3
# =============================================================================
# sc_ti_fidelity_demo.py  -- STANDALONE, no repo imports.
#
# GOAL: empirically show that in an SC+TI heterogeneity setting (2 qubits on a
# superconducting QPU, 2 on a trapped-ion QPU), the Aer density-matrix state
# fidelity of small circuits is CATASTROPHICALLY low -- because under a
# block-based ASAP timing model, superconducting qubits (T2=80us) sit idle and
# decohere while the trapped-ion side grinds through its ~100us-scale gates.
#
# Runs 10 random 4-qubit circuits, 3 schedules each (B1 static, B2 sticky,
# B2 non-sticky), reports Aer fidelity (primary) and an EFCL-style expected-
# failure-cost proxy (secondary, computed analytically -- NOT from Aer).
#
# Only needs: numpy, qiskit, qiskit-aer.
#
# NOTE ON THE MODEL: this reproduces the conventions of the Phase-0 harness
# (NB1-NB4) in compact standalone form:
#   * idle dephasing  = thermal_relaxation_error(t1=inf, t2=T2, time=t)  -> exp(-t/T2)
#   * gate infidelity = depolarizing_error with lambda=(1-F)*d/(d-1)     -> avg gate fid = F
#   * remote 2Q gate  = aggregate fidelity f_comm (replaces f2q), duration = slower
#                       tech's 2Q time; both participants advance by that duration
#   * movement        = one f_move channel on the moved qubit (fidelity only)
#   * timing          = pure per-qubit ASAP; every qubit is finally synced to the
#                       makespan (busy+idle == makespan), so SC qubits decohere
#                       while TI finishes.
# =============================================================================

import numpy as np
from itertools import combinations
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import thermal_relaxation_error, depolarizing_error

# ----------------------------------------------------------------------------
# HARD-CODED CONFIG  (SC+TI heterogeneity; 2 logical qubits per QPU)
# ----------------------------------------------------------------------------
TECH = {
    "sc": dict(f1q=0.9999, f2q=0.9990, fm=0.9900, T2=80000.0,
               t1q=20.0,    t2q=200.0,    tm=300.0),
    "ti": dict(f1q=0.9999, f2q=0.9997, fm=0.9950, T2=2000000.0,
               t1q=10000.0, t2q=100000.0, tm=5000.0),
}
F_COMM   = 0.95          # aggregate cross-QPU 2Q fidelity
F_MOVE   = 0.99          # per technology-switch fidelity
MODULE_TECH = ["sc", "ti"]   # module 0 = SC QPU, module 1 = TI QPU
CAP      = 2             # logical-qubit capacity per QPU
N        = 4             # qubits per circuit (2 SC + 2 TI)

N_CIRCUITS = 10
DEPTH      = 8           # < 10 layers
DENSITY    = 0.75        # per-qubit activation prob per layer
TWO_Q      = 0.60        # fraction of active pairs made into 2Q gates
SEED0      = 20260722

ONEQ_GATES = ["h", "x", "y", "z", "s", "t"]
TWOQ_GATES = ["cx", "cz", "swap"]

BIG_T1 = 1e15            # ~infinite T1 -> pure dephasing


# ----------------------------------------------------------------------------
# noise channels
# ----------------------------------------------------------------------------
def dephasing(T2, t):
    if t <= 0:
        return None
    return thermal_relaxation_error(BIG_T1, T2, t)

def gate_chan(F, n):
    d = 2 ** n
    lam = (1.0 - F) * d / (d - 1)
    lam = min(max(lam, 0.0), 1.0)
    return depolarizing_error(lam, n)

def apply_gate(qc, op):
    if op[0] == "1q":
        _, q, name = op
        getattr(qc, name)(q)
    else:
        _, u, v, name = op
        {"cx": qc.cx, "cz": qc.cz, "swap": qc.swap}[name](u, v)


# ----------------------------------------------------------------------------
# random circuit as a list of layers
# ----------------------------------------------------------------------------
def gen_layers(n, depth, rng):
    layers = []
    for _ in range(depth):
        active = [q for q in range(n) if rng.random() < DENSITY]
        rng.shuffle(active)
        ops, i = [], 0
        while i < len(active):
            if i + 1 < len(active) and rng.random() < TWO_Q:
                ops.append(("2q", active[i], active[i + 1],
                            rng.choice(TWOQ_GATES)))
                i += 2
            else:
                ops.append(("1q", active[i], rng.choice(ONEQ_GATES)))
                i += 1
        if ops:
            layers.append(ops)
    return layers


# ----------------------------------------------------------------------------
# schedules  (each returns list[dict{qubit -> module}] , one dict per layer,
#             always exactly 2 qubits per QPU)
# ----------------------------------------------------------------------------
def _activity(layers, n):
    act = [0] * n
    for L in layers:
        for op in L:
            if op[0] == "2q":
                act[op[1]] += 2; act[op[2]] += 2
            else:
                act[op[1]] += 1
    return act

def schedule_b1(layers, n):
    """Static heterogeneity-aware: 2 least-active qubits -> TI (coherence
    protection), 2 most-active -> SC (fast gates). One fixed mapping."""
    act = _activity(layers, n)
    order = sorted(range(n), key=lambda q: act[q])   # ascending activity
    ti = set(order[:CAP])                            # least active -> TI
    assign = {q: (1 if q in ti else 0) for q in range(n)}
    return [dict(assign) for _ in layers]

def _best_layer_assignment(layer, n, prev, sticky):
    active2q = [(op[1], op[2]) for op in layer if op[0] == "2q"]
    activeq = set()
    for op in layer:
        activeq.update(op[1:3] if op[0] == "2q" else [op[1]])
    best = None
    for sc_set in combinations(range(n), CAP):       # choose the 2 SC qubits
        sc = set(sc_set)
        assign = {q: (0 if q in sc else 1) for q in range(n)}
        cost = 0.0
        for (u, v) in active2q:                       # cut vs local
            if assign[u] != assign[v]:
                cost += -np.log(F_COMM)
            else:
                cost += -np.log(TECH[MODULE_TECH[assign[u]]]["f2q"])
        for q in range(n):                            # nudge idle qubits to TI
            if q not in activeq and assign[q] == 0:
                cost += 0.02
        if sticky and prev is not None:               # discourage reassignment
            for q in range(n):
                if prev[q] != assign[q]:
                    cost += -np.log(F_MOVE)
        if best is None or cost < best[0]:
            best = (cost, assign)
    return best[1]

def schedule_b2(layers, n, sticky):
    out, prev = [], None
    for L in layers:
        a = _best_layer_assignment(L, n, prev, sticky)
        out.append(a); prev = a
    return out


# ----------------------------------------------------------------------------
# Aer fidelity under block-based ASAP timing
# ----------------------------------------------------------------------------
def aer_fidelity(layers, schedule, n=N, include_idle=True):
    """Aer state fidelity under block-based ASAP timing.
    include_idle=False drops all idle + final-sync dephasing (gate/comm/move
    errors only) -> the 'if timing didn't bite' upper bound. The gap between
    the two is the pure heterogeneity-timing decoherence tax."""
    # ideal reference
    ideal = QuantumCircuit(n)
    for L in layers:
        for op in L:
            apply_gate(ideal, op)
    ideal_dm = DensityMatrix(Statevector.from_instruction(ideal))

    qc = QuantumCircuit(n)
    t_avail = [0.0] * n
    for li, L in enumerate(layers):
        sched = schedule[li]
        if li > 0:                                    # movement
            for q in range(n):
                if schedule[li][q] != schedule[li - 1][q]:
                    qc.append(gate_chan(F_MOVE, 1).to_instruction(), [q])
        for op in L:
            if op[0] == "1q":
                _, q, name = op
                tech = MODULE_TECH[sched[q]]
                apply_gate(qc, op)
                qc.append(gate_chan(TECH[tech]["f1q"], 1).to_instruction(), [q])
                t_avail[q] += TECH[tech]["t1q"]
            else:
                _, u, v, name = op
                mu, mv = sched[u], sched[v]
                start = max(t_avail[u], t_avail[v])
                if include_idle:
                    for q in (u, v):                  # idle each to gate start
                        idle = start - t_avail[q]
                        d = dephasing(TECH[MODULE_TECH[sched[q]]]["T2"], idle)
                        if d:
                            qc.append(d.to_instruction(), [q])
                if mu == mv:
                    tech = MODULE_TECH[mu]
                    dur, F = TECH[tech]["t2q"], TECH[tech]["f2q"]
                else:                                  # remote cross-QPU
                    dur = max(TECH[MODULE_TECH[mu]]["t2q"],
                              TECH[MODULE_TECH[mv]]["t2q"])
                    F = F_COMM
                apply_gate(qc, op)
                qc.append(gate_chan(F, 2).to_instruction(), [u, v])
                t_avail[u] = t_avail[v] = start + dur

    makespan = max(t_avail)                            # final sync (ST14)
    if include_idle:
        for q in range(n):
            idle = makespan - t_avail[q]
            d = dephasing(TECH[MODULE_TECH[schedule[-1][q]]]["T2"], idle)
            if d:
                qc.append(d.to_instruction(), [q])

    qc.save_density_matrix()
    noisy = DensityMatrix(
        AerSimulator(method="density_matrix").run(qc).result().data(0)["density_matrix"])
    return float(state_fidelity(noisy, ideal_dm)), makespan


# ----------------------------------------------------------------------------
# EFCL-style expected-failure-cost proxy (analytic; independent of Aer)
# ----------------------------------------------------------------------------
def efcl_proxy(layers, schedule, n=N):
    t_avail = [0.0] * n
    cost = 0.0
    for li, L in enumerate(layers):
        sched = schedule[li]
        if li > 0:
            for q in range(n):
                if schedule[li][q] != schedule[li - 1][q]:
                    cost += -np.log(F_MOVE)
        for op in L:
            if op[0] == "1q":
                _, q, name = op
                tech = MODULE_TECH[sched[q]]
                cost += -np.log(TECH[tech]["f1q"])
                t_avail[q] += TECH[tech]["t1q"]
            else:
                _, u, v, name = op
                mu, mv = sched[u], sched[v]
                start = max(t_avail[u], t_avail[v])
                for q in (u, v):
                    idle = start - t_avail[q]
                    if idle > 0:
                        cost += idle / TECH[MODULE_TECH[sched[q]]]["T2"]
                if mu == mv:
                    tech = MODULE_TECH[mu]; dur = TECH[tech]["t2q"]
                    cost += -np.log(TECH[tech]["f2q"])
                else:
                    dur = max(TECH[MODULE_TECH[mu]]["t2q"],
                              TECH[MODULE_TECH[mv]]["t2q"])
                    cost += -np.log(F_COMM)
                t_avail[u] = t_avail[v] = start + dur
    makespan = max(t_avail)
    for q in range(n):
        idle = makespan - t_avail[q]
        if idle > 0:
            cost += idle / TECH[MODULE_TECH[schedule[-1][q]]]["T2"]
    return cost / max(len(layers), 1)


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
SCHEDULERS = [
    ("B1_static",     lambda layers, n: schedule_b1(layers, n)),
    ("B2_sticky",     lambda layers, n: schedule_b2(layers, n, sticky=True)),
    ("B2_nonsticky",  lambda layers, n: schedule_b2(layers, n, sticky=False)),
]

def main():
    rows = []
    print(f"\nSC+TI heterogeneity  (2 qubits SC / 2 qubits TI, cap {CAP} each)")
    print(f"{N_CIRCUITS} random 4-qubit circuits, depth {DEPTH}, "
          f"f_comm={F_COMM}, f_move={F_MOVE}\n")
    header = (f"{'circ':>4} {'lyr':>3} {'schedule':>13} "
              f"{'Aer_fid':>8} {'no-idle':>8} {'drop':>6} {'EFCL':>7} {'mkspan(us)':>10}")
    print(header); print("-" * len(header))
    for c in range(N_CIRCUITS):
        rng = np.random.RandomState(SEED0 + c)
        layers = gen_layers(N, DEPTH, rng)
        for sname, sfn in SCHEDULERS:
            sched = sfn(layers, N)
            fid, mk   = aer_fidelity(layers, sched, include_idle=True)
            ref, _    = aer_fidelity(layers, sched, include_idle=False)
            efcl      = efcl_proxy(layers, sched)
            rows.append((c, len(layers), sname, fid, ref, efcl, mk))
            print(f"{c:>4} {len(layers):>3} {sname:>13} "
                  f"{fid:>8.4f} {ref:>8.4f} {ref-fid:>6.2f} {efcl:>7.3f} {mk/1000:>10.1f}")
        print("-" * len(header))

    print("\nPER-SCHEDULE MEANS  (Aer_fid is the headline):")
    for sname, _ in SCHEDULERS:
        fids  = [r[3] for r in rows if r[2] == sname]
        refs  = [r[4] for r in rows if r[2] == sname]
        efcls = [r[5] for r in rows if r[2] == sname]
        print(f"  {sname:>13}:  Aer_fid = {np.mean(fids):.3f} "
              f"(min {np.min(fids):.3f})   no-idle ref = {np.mean(refs):.3f}   "
              f"decoherence drop = {np.mean(refs)-np.mean(fids):.3f}   EFCL = {np.mean(efcls):.3f}")
    allf, allr = [r[3] for r in rows], [r[4] for r in rows]
    print(f"\n  OVERALL: SC+TI Aer fidelity = {np.mean(allf):.3f}   "
          f"vs no-idle reference = {np.mean(allr):.3f}")
    print(f"  => the {np.mean(allr)-np.mean(allf):.3f} gap is PURE heterogeneity-timing "
          f"decoherence: SC qubits idling while TI runs its ~100us gates.")

    try:
        import csv
        with open("sc_ti_fidelity_demo.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["circuit", "layers", "schedule", "aer_fidelity",
                        "aer_fidelity_no_idle", "efcl_proxy", "makespan_ns"])
            w.writerows(rows)
        print("\nwrote sc_ti_fidelity_demo.csv")
    except Exception as e:
        print("csv skipped:", e)


if __name__ == "__main__":
    main()
