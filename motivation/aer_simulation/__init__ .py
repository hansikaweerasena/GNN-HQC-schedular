"""mosaic_aer -- the Phase-0 Aer harness, extracted from notebooks NB1-NB6.

One trusted function ``assignment -> Aer fidelity``, physics-grounded and
independent of EFCL. This package is the *judge*; EFCL is the training
surrogate. Never let a script redefine a noise channel, a tech spec, or a
timing rule -- import from here.

Layout
------
    noise      NB1   dephasing_channel, gate_infidelity_channel
    hardware   NB2   TECHS, COMM, t_comm, HW, Module, Machine builders
    routing    NB3   route, un_permute
    lowering   NB4   segment_blocks, lower  (block-segmented ASAP)
    scoring    NB5   aer_fidelity, score, pareto_front
    circuits         make_layers (build directly), circuit_to_layers / from_qasm
                     (convert existing circuits), to_cx_basis (normalise the basis)
    families         randomized circuit families + profiler (P0.4)
    configs          techs_v1.json (frozen table) + drift_check

Quick start
-----------
    from mosaic_aer import score, heterogeneous_machine, H, CX

    machine = heterogeneous_machine([("sc", 4), ("na", 4)])
    layers  = [[('2q', 0, 1, CX)], [('1q', 2, H)]]
    sched   = [{q: (0 if q < 4 else 1) for q in range(8)}] * 2
    print(score(layers, sched, machine))

Gate tuple format: ``('1q', q, gate)`` or ``('2q', qa, qb, gate)`` where `gate`
is a Qiskit ``Gate`` instance. `schedule` is one ``{logical_qubit: module_id}``
dict per layer. Strip terminal measurements before scoring -- the scorer
compares pre-measurement states, so readout never enters.
"""

from qiskit.circuit.library import HGate, CXGate

from .noise import (
    dephasing_channel,
    gate_infidelity_channel,
    plus_coherence,
    idle_coherence_on_aer,
)
from .hardware import (
    TechSpec, TECHS, COMM, t_comm, t_move,
    HardwareModel, HW, SABRE_SEEDS,
    Module, Machine, homogeneous_machine, heterogeneous_machine,
    noiseless_techs, noiseless_comm, movement_mode,
)
from .routing import RouteResult, route, un_permute
from .lowering import Diagnostics, segment_blocks, lower
from .scoring import aer_fidelity, score, pareto_front, ideal_state
from .circuits import (
    LayeredCircuit, make_layers, circuit_to_layers, from_qasm, layers_to_circuit,
    to_cx_basis, validate_layers,
)
from .families import (
    FamilySpec, CircuitProfile, profile, fragility, generate, generate_pair,
    generate_family, FAMILIES, validate_structure, duty_threshold,
)
from .configs import drift_check, efcl_deltas, load_frozen, find_config, FROZEN_PATH

__version__ = "0.1.0"

# Convenience gate singletons -- every experiment needs these two.
H = HGate()
CX = CXGate()


def mod(mid, tech, qubits):
    """Terse Module constructor, matching the notebooks' `mod(0,'sc',[0,1])`."""
    return Module(mid, tech, tuple(qubits))


__all__ = [
    "dephasing_channel", "gate_infidelity_channel", "plus_coherence",
    "idle_coherence_on_aer",
    "TechSpec", "TECHS", "COMM", "t_comm", "t_move", "HardwareModel", "HW", "SABRE_SEEDS",
    "Module", "Machine", "homogeneous_machine", "heterogeneous_machine",
    "noiseless_techs", "noiseless_comm", "movement_mode", "mod",
    "RouteResult", "route", "un_permute",
    "Diagnostics", "segment_blocks", "lower",
    "aer_fidelity", "score", "pareto_front", "ideal_state",
    "LayeredCircuit", "make_layers", "circuit_to_layers", "from_qasm",
    "layers_to_circuit", "to_cx_basis", "validate_layers",
    "FamilySpec", "CircuitProfile", "profile", "fragility", "generate",
    "generate_pair", "generate_family", "FAMILIES", "validate_structure",
    "duty_threshold",
    "drift_check", "efcl_deltas", "load_frozen", "find_config", "FROZEN_PATH",
    "H", "CX", "__version__",
]
