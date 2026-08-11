"""Single-window SABRE router  (extracted verbatim from NB3).

Does exactly one thing: route a single residency-stable window onto a fixed
coupling map by inserting SWAPs, and report what was inserted plus where every
qubit ended up. This is the ONLY place SWAPs are computed. No noise, no
fidelity -- it builds the physically-honest circuit that `lowering` schedules
and `scoring` scores.

The five locked rules
---------------------
1. All-to-all passthrough. ``coupling_map is None`` (NA/TI) -> return the
   circuit unchanged, 0 swaps, identity layout. No routing entered.
2. SABRE routing-only, best-of-N. ``SabreSwap(cm, heuristic="decay", seed=s)``
   for each seed; keep the fewest-swaps result (ties -> lowest seed). Never
   strawman SC with a bad seed.
3. Insert swaps only. Bare ``PassManager([SabreSwap])`` -- no basis
   translation, no gate cancellation, no optimisation. ``swap`` gates stay as
   ``swap`` (NOT decomposed to 3 CX): `lowering` applies the f2q**3 / 3*t2q
   channel per swap directly.
4. Return the final permutation. ``final_layout[i]`` = physical wire where
   virtual qubit i ends up. `scoring` uses it to un-permute before comparing to
   the ideal -- the one bookkeeping step that, if dropped, silently produces
   low fidelities on routed circuits only.
5. Correctness invariant. Routed circuit == original up to ``final_layout``.

Scope boundary (NOT this module): windowing, persistent placement across
windows, residency changes, move channels, non-identity initial layout. Those
belong to `lowering`. The ``initial_layout`` argument is a Phase-2 stub and
raises.
"""

from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import SabreSwap
from qiskit.circuit.library import PermutationGate

from .hardware import SABRE_SEEDS

__all__ = ["RouteResult", "route", "un_permute"]


@dataclass
class RouteResult:
    routed_circuit: QuantumCircuit
    swap_count: int
    final_layout: list        # final_layout[i] = physical wire where virtual qubit i ends


def _final_perm(prop_set, n):
    """Normalise SabreSwap's final_layout to a plain list [virtual i -> physical wire]."""
    fl = prop_set.get("final_layout")
    if fl is None:
        return list(range(n))                       # no swaps -> identity
    v2p = {}
    for q, p in fl.get_virtual_bits().items():
        idx = getattr(q, "_index", None)
        if idx is None:                             # qiskit >= 1.x Qubit without _index
            idx = q.index if hasattr(q, "index") else None
        v2p[idx] = p
    return [v2p[i] for i in range(n)]


def route(circuit, coupling_map, seeds=SABRE_SEEDS, initial_layout=None):
    """Route one residency-stable window. See module docstring for the five rules."""
    if initial_layout is not None:
        raise NotImplementedError(
            "non-identity initial_layout is a Phase-2 / lowering concern (carried "
            "placement across windows); this router handles identity-initial windows only.")
    if any(inst.operation.name in ("measure", "reset") for inst in circuit.data):
        raise ValueError("strip terminal measurements/resets before routing")

    n = circuit.num_qubits

    # Rule 1: all-to-all -> passthrough
    if coupling_map is None:
        return RouteResult(circuit, 0, list(range(n)))

    # Rules 2 + 3: bare SabreSwap, best-of-N by min swaps (ties -> lowest seed)
    best = None
    for s in seeds:
        pm = PassManager([SabreSwap(coupling_map, heuristic="decay", seed=s)])
        tqc = pm.run(circuit)
        nsw = tqc.count_ops().get("swap", 0)
        if best is None or nsw < best[0]:
            best = (nsw, tqc, _final_perm(pm.property_set, n))
    nsw, tqc, perm = best
    return RouteResult(tqc, nsw, perm)


def un_permute(state, final_layout):
    """Undo the routing permutation so a routed output aligns with the ideal
    qubit order. Works for Statevector and DensityMatrix."""
    return state.evolve(PermutationGate(final_layout))
