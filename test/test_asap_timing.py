"""Step 4 correctness tests for the movement-aware ASAP timing surrogate.

Run:
    python test_asap_timing.py                 # standalone, no pytest needed
    pytest -v test_asap_timing.py              # or under pytest

T0  moved mass          relu form, timing AND fidelity agree, no phantom movement
T1  vs lowering.py      zero-migration exactness limit  (needs mosaic_aer; skips if absent)
T2  unequal wait        the earlier operand is charged exactly its real wait
T3  directional move    t_move(i,j) = t2q_i, asymmetric
T4  remote duration     t_comm(i,j) = max(t2q_i, t2q_j)
T5  autograd            the ONLY test that exercises the soft path

Plus two regressions: the barrier path still runs, and lambda_cap stays finite
(the delta-poisoning bug).

T1 is the load-bearing one. With one-hot P and zero migrations there are no
block boundaries, so the surrogate and the lowerer must agree EXACTLY -- not
approximately. It uses all-to-all technologies only (na, ti) so that SABRE
inserts no SWAPs and the Gamma routing proxy is identically zero; that is the
regime where the two models are supposed to coincide. Introducing sc would make
T1 fail for routing reasons that have nothing to do with timing.
"""

from __future__ import annotations

import json
import math
import os
import sys
import copy

import torch

# --- make the repo importable regardless of where this file is run from ------
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from src.cost_function import TotalCost, CapacityPenalty
except ImportError:
    from cost_function import TotalCost, CapacityPenalty  # type: ignore


# ===========================================================================
# Config
# ===========================================================================

CONFIG_PATH = os.environ.get(
    "MOSAIC_COST_CONFIG",
    os.path.join(_HERE, "configs", "cost_config_v3.json"),
)

# Frozen operating point, techs.json / techs_v3. The tests assert against these
# numbers directly, so if the config on disk disagrees the tests fail loudly --
# which is the point: they double as a Step 0 reconciliation check.
T2Q = {"sc": 200.0, "na": 1000.0, "ti": 100000.0}
T1Q = {"sc": 20.0, "na": 500.0, "ti": 10000.0}
TT2 = {"sc": 100000.0, "na": 2000000.0, "ti": 2000000.0}

# float32 buffers: makespans reach ~1e5-1e6 ns, so ~0.1 ns of representation
# error is expected. "Exact" in T1 means exact up to float32, not to 1e-9.
EXACT_RTOL, EXACT_ATOL = 1e-5, 1e-3


def load_cfg(techs=("sc", "na")):
    """Load the cost config and hard-select which two technologies to use."""
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    by_name = {t["name"]: t for t in cfg["techs"]}
    missing = [t for t in techs if t not in by_name]
    if missing:
        raise RuntimeError(
            f"config {CONFIG_PATH} has no tech(s) {missing}; "
            f"available: {sorted(by_name)}"
        )
    cfg = copy.deepcopy(cfg)
    cfg["techs"] = [by_name[t] for t in techs]
    assert_config_matches_techs_json(cfg)
    return cfg


def assert_config_matches_techs_json(cfg):
    """Step 0 reconciliation guard.

    The tests assert against the frozen techs.json operating point directly. If
    the cost config still holds pre-reconciliation values (e.g. na t2q=2000,
    na T2=200000, sc T2=80000, f_comm=0.95) every downstream test fails with a
    confusing number. Fail here instead, naming the field.
    """
    bad = []
    for t in cfg["techs"]:
        n = t["name"]
        if n not in T2Q:
            continue
        for field, table, path in (
            ("t1q", T1Q, ("gate_time", "t1q")),
            ("t2q", T2Q, ("gate_time", "t2q")),
            ("T2", TT2, ("coherence", "T2")),
        ):
            got = t[path[0]][path[1]]
            if not close(got, table[n], rtol=1e-9):
                bad.append(f"{n}.{field}: config={got} techs.json={table[n]}")
    f_comm = (cfg.get("comm", {}) or {}).get("f_comm")
    if f_comm is not None and not close(f_comm, 0.97, rtol=1e-9):
        bad.append(f"comm.f_comm: config={f_comm} techs.json=0.97")
    if bad:
        raise AssertionError(
            "Step 0 NOT applied -- cost config disagrees with techs.json:\n  "
            + "\n  ".join(bad)
        )


def build(cfg, mode="asap", validate=True):
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("timing_model", {})
    cfg["timing_model"]["mode"] = mode
    cfg["timing_model"]["validate"] = validate
    return TotalCost(cfg)


# ===========================================================================
# Minimal circuit containers (duck-typed for SegmentStatsExtractor)
# ===========================================================================

class _Layer:
    def __init__(self, gates):
        self.gates = gates          # [(name, [qubits]), ...]


class _Circuit:
    def __init__(self, layers):
        self.layers = layers


class _Seg:
    def __init__(self, layer_ids):
        self.layers = layer_ids


def circ_from(layer_gates):
    """layer_gates: list of list of (name, [qubits]). One segment per layer."""
    circ = _Circuit([_Layer(g) for g in layer_gates])
    segs = [_Seg([i]) for i in range(len(layer_gates))]
    return circ, segs


def onehot(rows, techs):
    """rows: e.g. ['sc','na','sc'] -> [N,K] one-hot tensor."""
    idx = {t: i for i, t in enumerate(techs)}
    K = len(techs)
    out = torch.zeros((len(rows), K))
    for u, t in enumerate(rows):
        out[u, idx[t]] = 1.0
    return out


# ===========================================================================
# Assertion helpers
# ===========================================================================

_FAILS = []
_SKIPS = []


def close(a, b, rtol=1e-6, atol=1e-9):
    return abs(float(a) - float(b)) <= atol + rtol * abs(float(b))


def check(name, got, want, rtol=1e-6, atol=1e-9):
    ok = close(got, want, rtol, atol)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: got {float(got):.10g}  want {float(want):.10g}")
    if not ok:
        _FAILS.append(name)
    return ok


def check_true(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        _FAILS.append(name)
    return bool(cond)


# ===========================================================================
# T0 -- moved mass
# ===========================================================================

def test_T0_moved_mass():
    print("\nT0  moved mass (relu / min-coupling)")
    techs = ("sc", "na")
    cfg = load_cfg(techs)
    tc = build(cfg)
    cmove = float(tc.cmove)

    # An empty 2-layer circuit isolates movement from everything else.
    circ, segs = circ_from([[], []])

    def run(P0, P1):
        return tc([P0, P1], segs, circ, debug=True)

    # (a) constant SOFT assignment must cost nothing -- this is the phantom
    #     movement bug the product form had.
    P = torch.full((2, 2), 0.5)
    o = run(P, P)
    check("T0a soft-constant move TIME", o["asap_move_time_total"], 0.0)
    check("T0a soft-constant move COST", o["per_segment_move"].sum(), 0.0)

    # (b) one-hot, no migration
    P = onehot(["sc", "na"], techs)
    o = run(P, P)
    check("T0b one-hot no-move TIME", o["asap_move_time_total"], 0.0)
    check("T0b one-hot no-move COST", o["per_segment_move"].sum(), 0.0)

    # (c) one-hot migration: exactly one qubit moves -> fidelity cost = cmove
    o = run(onehot(["sc", "na"], techs), onehot(["na", "na"], techs))
    check("T0c one migration COST", o["per_segment_move"].sum(), cmove)

    # (d) partial drift: 0.1 mass leaves sc -> 0.1 * cmove
    P0 = torch.tensor([[0.6, 0.4]])
    P1 = torch.tensor([[0.5, 0.5]])
    circ1, segs1 = circ_from([[], []])
    o = tc([P0, P1], segs1, circ1, debug=True)
    check("T0d partial-drift COST", o["per_segment_move"].sum(), 0.1 * cmove)
    check("T0d partial-drift TIME", o["asap_move_time_total"], 0.1 * T2Q["sc"])

    # (e) timing and fidelity agree on HOW MUCH moved: cost/cmove == mass, and
    #     time == mass-weighted t2q of the SOURCE.
    P0 = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    P1 = torch.tensor([[0.3, 0.7], [0.7, 0.3]])
    circ2, segs2 = circ_from([[], []])
    o = tc([P0, P1], segs2, circ2, debug=True)
    mass = float(torch.relu(P0 - P1).sum())
    t_expect = float((torch.relu(P0 - P1) * torch.tensor([T2Q["sc"], T2Q["na"]])).sum())
    check("T0e mass consistency", o["per_segment_move"].sum() / cmove, mass)
    check("T0e time consistency", o["asap_move_time_total"], t_expect)


# ===========================================================================
# T2 -- unequal-duration wait
# ===========================================================================

def test_T2_unequal_wait():
    print("\nT2  unequal-duration wait")
    techs = ("sc", "na")
    cfg = load_cfg(techs)
    tc = build(cfg)
    P = onehot(["sc", "na"], techs)

    # layer 0: one 1Q gate each. q0 (sc) ready at 20ns, q1 (na) at 500ns.
    # layer 1: a 2Q gate -> q0 must wait exactly 480ns, q1 waits 0.
    circ, segs = circ_from([
        [("h", [0]), ("h", [1])],
        [("cx", [0, 1])],
    ])
    o = tc([P, P], segs, circ, debug=True)

    wait = T1Q["na"] - T1Q["sc"]                       # 480 ns
    check("T2 gate-idle COST", o["C_idle_gate"], wait / TT2["sc"])

    pq = o["asap_idle_time_per_qubit"]
    # q0 waits 480 then both finish together -> no tail for either
    check("T2 q0 idle TIME", pq[0], wait)
    check("T2 q1 idle TIME", pq[1], 0.0)
    check("T2 tail", o["C_tail"], 0.0)

    # makespan = 500 (na 1q) + 1000 (remote 2q = max(200,1000))
    check("T2 makespan", o["makespan"], T1Q["na"] + T2Q["na"])


# ===========================================================================
# T3 -- directional movement
# ===========================================================================

def test_T3_directional_move():
    print("\nT3  directional movement (t_move = t2q_source)")
    techs = ("sc", "na")
    cfg = load_cfg(techs)
    tc = build(cfg)
    circ, segs = circ_from([[], []])

    o = tc([onehot(["sc"], techs), onehot(["na"], techs)], segs, circ, debug=True)
    check("T3 sc->na", o["asap_move_time_total"], T2Q["sc"])

    o = tc([onehot(["na"], techs), onehot(["sc"], techs)], segs, circ, debug=True)
    check("T3 na->sc", o["asap_move_time_total"], T2Q["na"])

    check_true(
        "T3 asymmetry (sc->na != na->sc)",
        not close(T2Q["sc"], T2Q["na"]),
        "t2q_sc != t2q_na, so a symmetric rule would be caught here",
    )

    # ti has a very different t2q -- confirms the rule scales, not a coincidence
    techs3 = ("sc", "ti")
    tc3 = build(load_cfg(techs3))
    o = tc3([onehot(["ti"], techs3), onehot(["sc"], techs3)], segs, circ, debug=True)
    check("T3 ti->sc", o["asap_move_time_total"], T2Q["ti"])


# ===========================================================================
# T4 -- remote gate duration
# ===========================================================================

def test_T4_remote_duration():
    print("\nT4  2Q duration matrix (t_comm = max)")
    techs = ("sc", "na")
    cfg = load_cfg(techs)
    tc = build(cfg)
    circ, segs = circ_from([[("cx", [0, 1])]])

    cases = [
        (["sc", "sc"], T2Q["sc"], "local sc"),
        (["na", "na"], T2Q["na"], "local na"),
        (["sc", "na"], max(T2Q["sc"], T2Q["na"]), "remote sc-na"),
        (["na", "sc"], max(T2Q["sc"], T2Q["na"]), "remote na-sc (symmetric)"),
    ]
    for rows, want, label in cases:
        o = tc([onehot(rows, techs)], segs, circ, debug=True)
        check(f"T4 {label}", o["makespan"], want)

    # sc-ti: the asymmetry between t_comm (max, symmetric) and t_move
    # (source-side, asymmetric) is most visible here.
    techs3 = ("sc", "ti")
    tc3 = build(load_cfg(techs3))
    o = tc3([onehot(["sc", "ti"], techs3)], segs, circ, debug=True)
    check("T4 remote sc-ti", o["makespan"], max(T2Q["sc"], T2Q["ti"]))


# ===========================================================================
# T5 -- autograd (the only soft-path test)
# ===========================================================================

def test_T5_autograd():
    print("\nT5  autograd on the SOFT path")
    techs = ("sc", "na")
    cfg = load_cfg(techs)
    tc = build(cfg)

    # One circuit containing all four ingredients:
    #   layer 0: 1Q gates            (1Q timing)
    #   layer 1: 2Q gate             (a wait + a remote gate)
    #   layer 2: 2Q gate on a subset (tail idle for the untouched qubit)
    # and P changes between segments, so movement is live too.
    circ, segs = circ_from([
        [("h", [0]), ("h", [1]), ("h", [2])],
        [("cx", [0, 1])],
        [("cx", [1, 2])],
    ])
    P_seq = [
        torch.tensor([[0.7, 0.3], [0.2, 0.8], [0.5, 0.5]], requires_grad=True),
        torch.tensor([[0.4, 0.6], [0.3, 0.7], [0.6, 0.4]], requires_grad=True),
        torch.tensor([[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]], requires_grad=True),
    ]
    out = tc(P_seq, segs, circ, debug=True)
    out["total_cost"].backward()

    all_present = all(p.grad is not None for p in P_seq)
    check_true("T5 grads exist", all_present)
    if not all_present:
        return
    all_finite = all(bool(torch.isfinite(p.grad).all()) for p in P_seq)
    check_true("T5 grads finite", all_finite)

    norms = [float(p.grad.norm()) for p in P_seq]
    print(f"        per-segment grad norms: {[round(n, 8) for n in norms]}")
    check_true("T5 total grad nonzero", sum(norms) > 0)
    # Segment 0 is the one a detached ready-time or an in-place bug kills first,
    # because its only downstream path is through the ready-time recursion.
    check_true("T5 segment 0 grad nonzero", norms[0] > 0,
               "a detached `ready` or an in-place scatter would zero this")
    check_true("T5 every segment grad nonzero", all(n > 0 for n in norms))


# ===========================================================================
# Regressions
# ===========================================================================

def test_R1_barrier_path():
    print("\nR1  barrier path still runs (E4 ablation)")
    techs = ("sc", "na")
    cfg = load_cfg(techs)
    circ, segs = circ_from([
        [("h", [0]), ("h", [1])],
        [("cx", [0, 1])],
    ])
    P = [torch.full((2, 2), 0.5, requires_grad=True) for _ in range(2)]

    b = build(cfg, mode="smooth_max")
    ob = b(P, segs, circ, debug=True)
    check_true("R1 barrier total finite", bool(torch.isfinite(ob["total_cost"])))
    check("R1 barrier delta intact", b.delta, 500.0, rtol=1e-3)

    tau0 = float(b.tau)
    b.set_epoch(5)
    check_true("R1 barrier tau anneals", float(b.tau) < tau0,
               f"{tau0:.1f} -> {float(b.tau):.1f}")

    a = build(cfg, mode="asap")
    tau_a = float(a.tau)
    a.set_epoch(5)
    check_true("R1 asap set_epoch is a no-op", close(float(a.tau), tau_a))

    try:
        build(cfg, mode="asapp")
        check_true("R1 bad mode rejected", False, "no exception raised")
    except ValueError:
        check_true("R1 bad mode rejected", True)


def test_R2_lambda_cap_finite():
    print("\nR2  lambda_cap finite on BOTH paths (delta-poisoning regression)")
    techs = ("sc", "na")
    cfg = load_cfg(techs)
    for mode in ("asap", "smooth_max"):
        c = copy.deepcopy(cfg)
        c.setdefault("timing_model", {})["mode"] = mode
        tc = TotalCost(c)
        cap = CapacityPenalty(tc, c)
        ok = bool(torch.isfinite(cap.lambda_cap))
        check_true(f"R2 lambda_cap finite [{mode}]", ok,
                   f"lambda_cap={float(cap.lambda_cap):.4f}, delta={float(tc.delta):.1f}")


def test_R3_disjointness_guard():
    print("\nR3  per-layer disjointness guard (validate=True)")
    techs = ("sc", "na")
    cfg = load_cfg(techs)
    tc = build(cfg, validate=True)
    # q0 appears in both a 1Q and a 2Q op in the same layer -- illegal.
    circ, segs = circ_from([[("h", [0]), ("cx", [0, 1])]])
    try:
        tc([onehot(["sc", "na"], techs)], segs, circ)
        check_true("R3 guard fires", False, "no exception raised")
    except ValueError as e:
        check_true("R3 guard fires", "twice" in str(e), str(e)[:70])


# ===========================================================================
# T1 -- zero-migration exactness vs lowering.py
# ===========================================================================

def test_T1_vs_lowering():
    print("\nT1  zero-migration exactness vs lowering.py")
    try:
        from mosaic_aer.lowering import lower
        from mosaic_aer.hardware import Module, TECHS
        from qiskit.circuit.library import CXGate, HGate, RZGate
    except ImportError as e:
        _SKIPS.append(f"T1 (mosaic_aer not importable: {e})")
        print(f"  [SKIP] mosaic_aer not importable ({e}). "
              f"Run from the repo root, or set PYTHONPATH.")
        return

    # All-to-all technologies ONLY. sc would inject SABRE SWAPs that the Gamma
    # proxy does not reproduce, and T1 would fail for routing reasons.
    techs = ("na", "ti")
    for t in techs:
        if not TECHS[t].all_to_all:
            _SKIPS.append(f"T1 ({t} is not all-to-all)")
            print(f"  [SKIP] {t} is not all-to-all in this hardware table; "
                  f"T1 needs Gamma == 0.")
            return

    # 4 logical qubits, module 0 = na (q0,q1), module 1 = ti (q2,q3).
    # No measurement (lower() takes no measure ops), no migration (one block).
    modules = [Module(0, "na", (0, 1)), Module(1, "ti", (2, 3))]
    assign = {0: 0, 1: 0, 2: 1, 3: 1}
    tech_of = ["na", "na", "ti", "ti"]

    # NOTE: lower() appends g[-1] straight into a QuantumCircuit, so the gate
    # must be a Qiskit Gate OBJECT, not a name string. validate_layers() in
    # circuits.py enforces this too.
    H, CX, RZ = HGate(), CXGate(), RZGate(0.3)
    aer_layers = [
        [("1q", 0, H), ("1q", 1, H), ("1q", 2, H), ("1q", 3, H)],
        [("2q", 0, 1, CX), ("2q", 2, 3, CX)],
        [("2q", 1, 2, CX)],                         # cross-module -> remote
        [("1q", 0, RZ), ("2q", 1, 3, CX)],          # cross-module -> remote
        [("2q", 0, 1, CX)],
    ]
    efcl_layers = [
        [("h", [0]), ("h", [1]), ("h", [2]), ("h", [3])],
        [("cx", [0, 1]), ("cx", [2, 3])],
        [("cx", [1, 2])],
        [("rz", [0]), ("cx", [1, 3])],
        [("cx", [0, 1])],
    ]
    schedule = [dict(assign) for _ in aer_layers]    # identical every layer

    qc, l2w, diag = lower(aer_layers, schedule, modules)
    if not diag.feasible:
        check_true("T1 aer schedule feasible", False)
        return
    check_true("T1 zero migrations", diag.move_count == 0,
               f"move_count={diag.move_count}, n_blocks={diag.n_blocks}")
    check_true("T1 zero SABRE swaps", diag.swap_count == 0,
               f"swap_count={diag.swap_count} (must be 0 for all-to-all)")

    cfg = load_cfg(techs)
    tc = build(cfg, validate=True)
    P = onehot(tech_of, techs)
    circ, segs = circ_from(efcl_layers)
    out = tc([P] * len(efcl_layers), segs, circ, debug=True)

    # (1) makespan -- exact
    check("T1 makespan", out["makespan"], diag.makespan,
          rtol=EXACT_RTOL, atol=EXACT_ATOL)

    # (2) per-qubit idle TIME -- exact, entry by entry. A summed comparison
    #     hides errors that cancel between qubits.
    pq = out["asap_idle_time_per_qubit"]
    for q in range(4):
        check(f"T1 idle_time[q{q}] ({tech_of[q]})",
              pq[q], diag.idle_time.get(q, 0.0),
              rtol=EXACT_RTOL, atol=EXACT_ATOL)

    # (3) idle COST -- what actually enters the loss
    invT2 = {t: 1.0 / TTT for t, TTT in TT2.items()}
    want_cost = sum(diag.idle_time.get(q, 0.0) * invT2[tech_of[q]] for q in range(4))
    check("T1 idle cost", out["per_segment_idle"].sum(), want_cost,
          rtol=EXACT_RTOL, atol=1e-9)

    # (4) ST14 analogue: idle + busy == makespan, per qubit.
    #     NOTE `ready[q]` is NOT busy time -- it already contains the waits, so
    #     idle + ready double-counts them. Busy is the complement of idle:
    #         busy = makespan - idle
    #     Checking that against lowering's own busy_time closes the loop: it
    #     verifies EFCL splits the timeline into the same two buckets Aer does,
    #     not merely that the buckets happen to sum correctly.
    mk = float(out["makespan"])
    for q in range(4):
        check(f"T1 ST14 busy[q{q}]",
              mk - float(pq[q]), diag.busy_time.get(q, 0.0),
              rtol=EXACT_RTOL, atol=EXACT_ATOL)
        check_true(f"T1 ST14 sum q{q}",
                   close(float(pq[q]) + diag.busy_time.get(q, 0.0), mk,
                         rtol=EXACT_RTOL, atol=EXACT_ATOL),
                   f"idle={float(pq[q]):.1f} + busy={diag.busy_time.get(q, 0.0):.1f} "
                   f"== makespan={mk:.1f}")


# ===========================================================================

def main():
    print("=" * 74)
    print("Step 4 -- ASAP timing correctness tests")
    print(f"config: {CONFIG_PATH}")
    print("=" * 74)

    for fn in (test_T0_moved_mass,
               test_T2_unequal_wait,
               test_T3_directional_move,
               test_T4_remote_duration,
               test_T5_autograd,
               test_R1_barrier_path,
               test_R2_lambda_cap_finite,
               test_R3_disjointness_guard,
               test_T1_vs_lowering):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  [ERROR] {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            _FAILS.append(fn.__name__)

    print("\n" + "=" * 74)
    if _FAILS:
        print(f"FAILED ({len(_FAILS)}): {_FAILS}")
        return 1
    if _SKIPS:
        # A skip must NOT read as a pass. T1 is the exactness limit -- the whole
        # point of the suite -- and skipping it verifies nothing.
        print(f"INCOMPLETE -- {len(_SKIPS)} test(s) SKIPPED, nothing verified there:")
        for sk in _SKIPS:
            print(f"  - {sk}")
        print("Everything that RAN passed. Re-run from the repo root so T1 executes.")
        return 2
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
