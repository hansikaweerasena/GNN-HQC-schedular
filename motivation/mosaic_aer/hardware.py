"""Per-technology hardware + communication model  (extracted from NB2, v4).

Pure hardware description: no noise applied, no fidelity numbers produced.
`routing` and `lowering` consume it.

Produces
--------
TECHS            SC / NA / TI specs, hardcoded from the cost configs, field cited in-line.
COMM, t_comm()   the communication model (see below).
HW               coupling maps, SWAP cost, average degree.
Module, Machine  distributed machine builders (homogeneous references + heterogeneous pools).
SABRE_SEEDS      routing seed set, pinned in one place.

The v4 communication model
--------------------------
`t_remote` is gone. It was ONE scalar used for two physically distinct
quantities -- the duration of a remote 2Q gate, and the latency of a
block-boundary state transfer. At 0.0 the conflation was invisible.

  t_comm(a, b)            duration of a remote 2Q gate      max(t2q_a, t2q_b)
  t_move(a, b)            latency of a state transfer       max(t2q_a, t2q_b)
  COMM["f_comm"]          aggregate fidelity of the whole teleported-gate primitive   0.95
  COMM["f_move"]          aggregate fidelity of the whole state-transfer primitive    0.99

Why t_comm > 0: a remote gate sits on the dependency path and cannot complete
faster than the slower endpoint's native 2Q gate. Zero duration made a remote
gate temporally *cheaper* than a local one and hid its dominant cost -- the
decoherence it inflicts on spectator qubits waiting behind it.

Why t_comm is per-pair, not a uniform experiment-level constant: a uniform value
would force the homogeneous 2xSC baseline to pay NA's or TI's gate time per
cross-module gate while its own endpoints run at 200 ns -- a 10x/500x handicap
on the very baseline the experiment exists to beat. The substrate is held
constant through f_comm; the endpoints are allowed to differ.

v5 -- movement latency is now EXPOSED, not overlapped
-----------------------------------------------------
Through v4, ``COMM["t_move_visible"] = 0.0``: state transfer was assumed fully
overlapped with the tail of the preceding block. That was never free -- a mover
is dephased at its PRE-MOVE T2 from its own t_avail up to the block makespan
before it may leave (ST7) -- but "movement costs no critical-path time" is a
line in the assumption ledger that reads as a subsidy to dynamic schedules, and
it is cheap to remove.

Default is now ``COMM["t_move_derived"] = True``: the boundary advances by
``max`` over movers of ``t_move(from, to)``, and every qubit synchronised at
that boundary pays for the wait. Set ``t_move_derived = False`` (with
``t_move_visible = 0.0``) to recover the v4 overlapped model exactly -- kept as
the sensitivity foil, not deleted. Use the ``movement_mode`` context manager.

Who pays during the transfer interval:
  movers      NOTHING extra. Their transfer infidelity is already aggregated
              into f_move, so charging T2 dephasing on top would double-count.
              The interval is booked as `busy` (an operation occupies them), not
              as idle, which keeps the ST14 invariant intact.
  non-movers  Real idle at their own, unchanged T2 -- they are sitting in their
  in sync_q   modules waiting for the boundary to clear. Bucketed as `sync_idle`.
  everyone    Untouched under sync_scope="module": an unaffected module keeps
  else        running (ST8).

Why f_comm / f_move are uniform across module pairs: under heralded
entanglement distribution, channel loss and transduction inefficiency reduce
the entanglement generation *rate*, not the fidelity of a successfully heralded
Bell pair. The heterogeneity penalty is carried entirely by t_comm. This is a
forward-looking assumption and belongs in the limitations paragraph.

t_comm is an OPTIMISTIC LOWER BOUND: it neglects the endpoint basis
measurements, the classical round trip, and the conditional Pauli corrections,
all of which are absorbed into f_comm on the fidelity side and dropped on the
timing side. State it as such in the paper.
"""

from dataclasses import dataclass, replace
from qiskit.transpiler import CouplingMap

__all__ = [
    "TechSpec", "TECHS", "COMM", "t_comm", "t_move",
    "HardwareModel", "HW", "SABRE_SEEDS",
    "Module", "Machine", "homogeneous_machine", "heterogeneous_machine",
    "noiseless_techs", "noiseless_comm", "movement_mode",
]

# Best-of-N SABRE: take min-SWAP over these seeds so an unlucky seed never
# strawmans the SC baseline.
SABRE_SEEDS = list(range(10))


# ---------------------------------------------------------------------------
# 1. Technology specs
# ---------------------------------------------------------------------------
# Hardcoded from cost_config_v3.json (SC, NA) and cost_config_tp2n_99.json
# (SC, TI). SC is byte-identical across both configs, so it is defined once.
# All times in NANOSECONDS.
#
# fm / tm are intentionally omitted. Data-qubit measurement is terminal and
# stripped before lowering, so readout never enters the score.

@dataclass(frozen=True)
class TechSpec:
    name: str
    f1q: float            # gate_fidelity.f1q
    f2q: float            # gate_fidelity.f2q
    T2: float             # coherence.T2   (ns) -- T1 -> inf (pure dephasing)
    t1q: float            # gate_time.t1q  (ns)
    t2q: float            # gate_time.t2q  (ns)
    kappa: float          # routing.kappa  (coarse connectivity descriptor in EFCL)
    all_to_all: bool      # routing.all_to_all
    max_qubits: int = 20  # capacity.max_qubits (device ceiling, NOT experiment capacity)


TECHS = {
    "sc": TechSpec(name="sc",
                   f1q=0.9999, f2q=0.999,          # v3 / tp2n gate_fidelity
                   T2=80_000.0,                    # coherence.T2
                   t1q=20.0, t2q=200.0,            # gate_time
                   kappa=2.3, all_to_all=False),   # routing.kappa
    "na": TechSpec(name="na",
                   f1q=0.9995, f2q=0.997,          # v3 gate_fidelity
                   T2=200_000.0,                   # coherence.T2
                   t1q=200.0, t2q=2_000.0,         # gate_time
                   kappa=0.0, all_to_all=True),    # routing.all_to_all
    "ti": TechSpec(name="ti",
                   f1q=0.9999, f2q=0.9997,         # tp2n gate_fidelity
                   T2=2_000_000.0,                 # coherence.T2
                   t1q=10_000.0, t2q=100_000.0,    # gate_time
                   kappa=0.0, all_to_all=True),    # routing.all_to_all
}


# ---------------------------------------------------------------------------
# 2. Communication model.  ONE source of truth, consumed by `lowering`.
# ---------------------------------------------------------------------------
COMM = {
    "f_comm":          0.95,   # aggregate fidelity of the WHOLE teleported-gate primitive,
                               # INCLUDING its local endpoint operations. Not an add-on
                               # channel: a remote gate pays f_comm and NOT f2q.
    "f_move":          0.99,   # aggregate fidelity of the WHOLE state-transfer primitive.
                               # Movement happens at a scheduled boundary and may consume a
                               # pre-purified Bell pair; remote gates fire on demand and
                               # consume raw pairs. Hence f_move > f_comm.
    "t_move_derived":  True,   # DEFAULT (v5): transfer latency is exposed on the critical
                               # path and derived per pair by t_move(a, b). See below.
    "t_move_visible":  0.0,    # legacy overlapped-transfer scalar. Used ONLY when
                               # t_move_derived is False. Kept as the sensitivity foil.
}


def t_comm(tech_a, tech_b):
    """Duration of a remote 2Q gate between modules of technology `tech_a`/`tech_b`.

    Gate teleportation applies a local CNOT against a Bell-pair half at each
    endpoint; the two run in parallel, so the operation cannot complete before
    the slower endpoint's native 2Q gate.

    Keyed on the TECHNOLOGY of the two endpoints. `lowering` decides *whether* a
    gate is remote by MODULE, never by technology.
    """
    return max(TECHS[tech_a].t2q, TECHS[tech_b].t2q)


def t_move(tech_from, tech_to):
    """Critical-path latency of a state transfer from `tech_from` to `tech_to`.

    Same max rule as `t_comm`, and deliberately so: state teleportation consumes
    a Bell pair through a local Bell-basis measurement at the source and a
    conditional Pauli correction at the destination. The measurement is a 2Q
    operation on the source side, so the transfer cannot complete faster than
    the source's native 2Q gate; the max rule additionally refuses to let a slow
    destination look free.

    Relative to a source-only rule (`t2q_from`) this is CONSERVATIVE -- it never
    under-charges. Relative to physical reality it remains an OPTIMISTIC LOWER
    BOUND, because it still drops the classical round trip and the correction
    latency, exactly as `t_comm` does. Declare both in the limitations paragraph.

    Kept as a SEPARATE function from `t_comm` on purpose. These were one scalar
    (`t_remote`) until v4 and the conflation was harmless only because it was 0.
    Never merge them again.
    """
    return max(TECHS[tech_from].t2q, TECHS[tech_to].t2q)


class noiseless_comm:
    """Context manager: temporarily set f_comm = f_move = 1.

    Combined with `noiseless_techs`, this makes every channel in the harness the
    identity, so any fidelity below 1 is a BOOKKEEPING error -- a gate applied to
    the wrong wire, a permutation not undone, a state left behind by a migration.
    This is the only configuration in which such bugs are visible at all.
    """

    def __enter__(self):
        self._backup = dict(COMM)
        COMM["f_comm"] = 1.0
        COMM["f_move"] = 1.0
        return COMM

    def __exit__(self, *exc):
        COMM.clear()
        COMM.update(self._backup)
        return False


class movement_mode:
    """Context manager to switch the movement-latency model for a sensitivity sweep.

        with movement_mode(derived=False):   # legacy fully-overlapped transfer
            ...

        with movement_mode(derived=False, visible=5000.0):  # uniform 5 us scalar
            ...

    `lowering` reads COMM at call time, so this mutates in place and never
    rebinds. Restores on exit.
    """

    def __init__(self, derived=True, visible=0.0):
        self.derived, self.visible = derived, visible

    def __enter__(self):
        self._backup = dict(COMM)
        COMM["t_move_derived"] = self.derived
        COMM["t_move_visible"] = self.visible
        return COMM

    def __exit__(self, *exc):
        COMM.clear()
        COMM.update(self._backup)
        return False


# ---------------------------------------------------------------------------
# 3. Connectivity, SWAP cost
# ---------------------------------------------------------------------------

class HardwareModel:
    """Coupling maps as a function of capacity, plus SWAP cost.

    SC is a cycle on its qubits: cap 4 -> the 2x2 ring (0-1-2-3-0), cap 2 -> a
    single edge (no routing), cap 1 -> an isolated node. NA/TI return None =
    all-to-all (the router reads None as "no routing, no SWAPs").
    """

    def __init__(self, techs, comm):
        self.techs = techs
        self.comm = comm

    def spec(self, tech):
        return self.techs[tech]

    def coupling_map(self, tech, n_qubits):
        """Local coupling map on 0..n-1. None => all-to-all (no routing)."""
        t = self.techs[tech]
        if t.all_to_all:
            return None
        if n_qubits <= 1:
            return CouplingMap([])                    # single node, no edges
        if n_qubits == 2:
            return CouplingMap([[0, 1], [1, 0]])      # single edge, never routes
        edges = []
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            edges += [[i, j], [j, i]]
        return CouplingMap(edges)

    def avg_degree(self, tech, n_qubits):
        """Average undirected degree of the local topology (all-to-all -> n-1).

        This is what you set EFCL's kappa to for consistency in Phase 2.
        """
        t = self.techs[tech]
        if t.all_to_all:
            return float(n_qubits - 1)
        cm = self.coupling_map(tech, n_qubits)
        und = {frozenset(e) for e in cm.get_edges() if e[0] != e[1]}
        return 2.0 * len(und) / n_qubits if n_qubits else 0.0

    # ---- SWAP cost (SC only; all-to-all techs never SWAP) ----
    def swap_fidelity(self, tech):
        """SC has no native SWAP: 3 CX -> f2q**3."""
        return self.techs[tech].f2q ** 3

    def swap_duration(self, tech):
        return 3.0 * self.techs[tech].t2q


HW = HardwareModel(TECHS, COMM)


# ---------------------------------------------------------------------------
# 4. Distributed machines
# ---------------------------------------------------------------------------
# A Machine is a list of Modules with GLOBAL qubit indexing; cross-module 2Q
# interactions are remote (charged f_comm + t_comm in `lowering` -- there is no
# coupling edge between modules).
#
#   homogeneous_machine("sc", 2, 4)                 -> the 2xSC reference (8q, distributed DQC)
#   heterogeneous_machine([("sc", 4), ("na", 4)])   -> the mixed pool for 1A
#
# NB2 and NB4 defined two different `Module` dataclasses (`module_id`+`coupling_map`
# vs `mid`+`cap`). They are unified here: `mid` is canonical, `module_id` is an
# alias, and `coupling_map` is derived rather than stored so it can never desync
# from `tech`/`cap`.

@dataclass(frozen=True)
class Module:
    mid: int
    tech: str
    qubits: tuple          # global qubit indices owned by this module

    @property
    def cap(self):
        return len(self.qubits)

    @property
    def n_qubits(self):
        return len(self.qubits)

    @property
    def module_id(self):
        return self.mid

    @property
    def coupling_map(self):
        return HW.coupling_map(self.tech, self.cap)


@dataclass
class Machine:
    name: str
    modules: list

    @property
    def n_qubits(self):
        return sum(m.n_qubits for m in self.modules)

    def module_of(self, global_qubit):
        for m in self.modules:
            if global_qubit in m.qubits:
                return m
        raise KeyError(global_qubit)

    def summary(self):
        parts = [f"{m.tech}[{m.qubits[0]}..{m.qubits[-1]}]" for m in self.modules]
        return f"{self.name}: {self.n_qubits}q = " + " | ".join(parts)


def _build(name, layout):
    """layout: list of (tech, cap). Assigns contiguous global indices per module."""
    modules, base = [], 0
    for mid, (tech, cap) in enumerate(layout):
        modules.append(Module(mid, tech, tuple(range(base, base + cap))))
        base += cap
    return Machine(name, modules)


def homogeneous_machine(tech, n_modules, cap_per_module):
    """The distributed homogeneous reference. NOT a monolithic machine: its
    cross-module gates pay f_comm AND t_comm, exactly like the heterogeneous
    pool's. If they did not, the baseline would win trivially."""
    return _build(f"{n_modules}x{tech.upper()}", [(tech, cap_per_module)] * n_modules)


def heterogeneous_machine(layout):
    tag = "+".join(t.upper() for t, _ in layout)
    return _build(tag, layout)


# ---------------------------------------------------------------------------
# 5. Test utility
# ---------------------------------------------------------------------------

class noiseless_techs:
    """Context manager: temporarily set every tech to f=1, T2=inf, keeping gate
    TIMES intact.

    `lowering` reads the module-level TECHS dict at call time, so this mutates
    in place and never rebinds. Used by the certificate to isolate a single
    error source (ST1, ST12a).

        with noiseless_techs():
            ...   # only the channel under test contributes infidelity
    """

    def __enter__(self):
        self._backup = dict(TECHS)
        for k, spec in list(TECHS.items()):
            TECHS[k] = replace(spec, f1q=1.0, f2q=1.0, T2=1e18)
        return TECHS

    def __exit__(self, *exc):
        TECHS.clear()
        TECHS.update(self._backup)
        return False
