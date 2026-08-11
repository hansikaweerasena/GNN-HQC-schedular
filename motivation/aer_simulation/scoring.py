"""The scorer -- public interface  (extracted verbatim from NB5).

Runs the lowered circuit on Aer's `density_matrix` backend, reduces to the
wires holding the logical qubits, un-permutes them to logical order (the
routing permutation), and compares to the noiseless ideal. Infeasible schedules
return None.

`score()` is the experiment entry point: it returns both axes, because Phase 1
reports both. Fidelity already internalises makespan through T2 decay, so
makespan and fidelity are NOT independent evidence -- say so once in the paper,
then plot both.

`tts = makespan / fidelity**2` is a defensible single-number collapse (an
expectation-value estimate needs shots scaling as 1/F**2); report raw fidelity
as primary and tts as secondary.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import PermutationGate
from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity, partial_trace

from .lowering import lower

__all__ = ["aer_fidelity", "score", "pareto_front", "ideal_state"]


def ideal_state(layers, allq):
    """Noiseless reference: the logical circuit with no hardware model at all."""
    idx = {q: i for i, q in enumerate(allq)}
    qc = QuantumCircuit(len(allq))
    for lay in layers:
        for g in lay:
            if g[0] == '1q':
                qc.append(g[2], [idx[g[1]]])
            else:
                qc.append(g[3], [idx[g[1]], idx[g[2]]])
    return Statevector(qc)


def aer_fidelity(layers, schedule, modules, **kw):
    """(fidelity, diagnostics). Returns (None, diagnostics) if the schedule is infeasible."""
    qc, l2w, diag = lower(layers, schedule, modules, **kw)
    if not diag.feasible:
        return None, diag
    dm = DensityMatrix(
        AerSimulator(method="density_matrix").run(qc).result().data(0)["density_matrix"])
    allq = sorted(l2w)
    wires = [l2w[q] for q in allq]
    red = partial_trace(dm, [w for w in range(qc.num_qubits) if w not in wires])
    order = list(np.argsort(np.argsort(wires)))
    if order != list(range(len(order))):
        red = red.evolve(PermutationGate(order))
    return state_fidelity(ideal_state(layers, allq), red), diag


def score(layers, schedule, modules, **kw):
    """Both axes plus the diagnostics an experiment needs. None if infeasible."""
    f, d = aer_fidelity(layers, schedule, modules, **kw)
    if f is None:
        return None
    return {"fidelity": f, "makespan": d.makespan, "tts": d.makespan / (f ** 2),
            "comm_count": d.comm_count, "comm_time": d.comm_time,
            "move_count": d.move_count, "move_time": d.move_time,
            "swap_count": d.swap_count,
            "n_blocks": d.n_blocks, "block_makespans": d.block_makespans}


def pareto_front(points):
    """points: list of dicts with 'makespan' (minimise) and 'fidelity' (maximise).
    Returns the indices of the non-dominated points."""
    keep = []
    for i, p in enumerate(points):
        dominated = any(
            (q["makespan"] <= p["makespan"] and q["fidelity"] >= p["fidelity"]) and
            (q["makespan"] < p["makespan"] or q["fidelity"] > p["fidelity"])
            for j, q in enumerate(points) if j != i)
        if not dominated:
            keep.append(i)
    return keep
