"""Circuit -> layers adapter, and direct construction of the layer format.

The harness's layer format is:

    [ [('2q', qa, qb, gate), ('1q', q, gate), ...],   # layer 0
      [...],                                          # layer 1
      ... ]

Two ways in
-----------
`make_layers(spec)`   Build the layer grid DIRECTLY. Use this for synthetic
                      circuits. Nothing is re-layered, so the active/idle pattern
                      you write is the one that gets scored.

`circuit_to_layers(c)` Convert an existing QuantumCircuit (MQT Bench, QASM, ...).
                      Re-layers ASAP, honouring barriers as alignment points.

Why `make_layers` is the right default for M1/M2
------------------------------------------------
`circuit_to_layers` re-derives layers from the gate dependency graph. Two gates
on disjoint qubits are independent, so ASAP will pack them into the same layer
whatever the source circuit looked like:

    intended                        after ASAP re-layering
    L0:  q0 gate,  q1 idle          L0:  q0 gate, q1 gate
    L1:  q0 idle,  q1 gate          (the intended idleness is gone)

For M1/M2 the whole point is circuits where different qubits have different
active/idle behaviour, so silently compressing that structure would mean
scoring a circuit family other than the one you generated. Barriers now prevent
this (see below), but building the grid directly avoids the round trip entirely.

Barriers, delays and identity gates
-----------------------------------
`barrier`   Alignment point: every qubit it spans is advanced to the latest
            frontier among them, so no later gate on those wires can be packed
            back before it. A full-width barrier therefore preserves an intended
            layer grid exactly. Barriers emit no gate and cost nothing.
`delay`/`id` Advance that qubit's frontier by one layer without emitting a gate,
            which preserves an intended idle slot. They cost nothing directly:
            in this harness idle time comes from the block clock in `lowering`,
            not from the source circuit.

Measurements and resets
-----------------------
Terminal `measure`/`reset` are stripped -- the scorer compares PRE-measurement
states, so readout never enters the model, and generated circuits should simply
assume a terminal measurement that is dropped. A MID-CIRCUIT measure or reset is
a hard error rather than a silent drop: removing one changes what the circuit
computes. Benchmarks needing them are out of scope (decided in NB2).

Logical SWAP gates
------------------
The hardware model prices a SWAP as three native 2Q gates (`f2q**3`, `3*t2q`),
and `lowering` charges routing-inserted SWAPs that way. A SWAP already present
in the LOGICAL source circuit would otherwise arrive here as an ordinary 2Q gate
and be charged a single `f2q`/`t2q` -- a 3x under-price, and an inconsistency
between two SWAPs that are physically identical.

So `swap_policy="raise"` is the default. Use `to_cx_basis()` (or
`swap_policy="expand"`) to decompose logical SWAPs into 3 CX before scoring.
Normalising every circuit to a CX + 1Q basis is the recommended path for MQT
Bench too: it removes this class of mismatch for all composite gates at once.
"""

from dataclasses import dataclass, field

from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, HGate

__all__ = [
    "LayeredCircuit", "make_layers", "circuit_to_layers", "from_qasm",
    "layers_to_circuit", "to_cx_basis", "validate_layers",
]

_DROPPABLE = {"barrier", "delay", "id"}
_TERMINAL_ONLY = {"measure", "reset"}
_ADVANCING = {"delay", "id"}      # dropped, but still consume a layer slot


@dataclass
class LayeredCircuit:
    """A layered circuit. Behaves like the plain `layers` list, so it can be passed
    straight to `lower`, `aer_fidelity` and `score`."""
    layers: list
    n_qubits: int
    name: str = ""
    dropped: dict = field(default_factory=dict)   # {op_name: count} removed during conversion

    # -- list protocol, so `layers` and LayeredCircuit are interchangeable --
    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __getitem__(self, i):
        return self.layers[i]

    # -- descriptors the experiment generators need --
    @property
    def depth(self):
        return len(self.layers)

    @property
    def n_2q(self):
        return sum(1 for lay in self.layers for g in lay if g[0] == '2q')

    @property
    def n_1q(self):
        return sum(1 for lay in self.layers for g in lay if g[0] == '1q')

    def active_qubits(self):
        out = set()
        for lay in self.layers:
            for g in lay:
                out.update(g[1:3] if g[0] == '2q' else [g[1]])
        return sorted(out)

    def interaction_pairs(self):
        """{(qa, qb): count} over 2Q gates, qa < qb. The input a partitioner needs."""
        pairs = {}
        for lay in self.layers:
            for g in lay:
                if g[0] == '2q':
                    k = (min(g[1], g[2]), max(g[1], g[2]))
                    pairs[k] = pairs.get(k, 0) + 1
        return pairs

    def activity(self):
        """{qubit: number of layers in which it holds a gate}. The asymmetric-activity
        check for M1: heterogeneity can only pay when some qubits are hot and others cold."""
        act = {q: 0 for q in range(self.n_qubits)}
        for lay in self.layers:
            for g in lay:
                for q in (g[1:3] if g[0] == '2q' else [g[1]]):
                    act[q] = act.get(q, 0) + 1
        return act

    def idle_fraction(self):
        """Fraction of (qubit, layer) slots in which a qubit has no gate.

        A coarse, hardware-free proxy for how much idle exposure a circuit offers.
        It is NOT the physics -- real idle time is set by gate durations and the
        block clock -- but it is a cheap filter when generating circuit families.
        """
        if not self.layers or not self.n_qubits:
            return 0.0
        busy = sum(len(g[1:3]) if g[0] == '2q' else 1 for lay in self.layers for g in lay)
        return 1.0 - busy / (self.n_qubits * len(self.layers))

    def summary(self):
        return (f"{self.name or 'circuit'}: {self.n_qubits}q depth={self.depth} "
                f"2q={self.n_2q} 1q={self.n_1q} idle_frac={self.idle_fraction():.2f}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_layers(layers, n_qubits=None):
    """Check the layer format is well formed. Raises on the failure modes that would
    otherwise produce a plausible-looking but wrong fidelity."""
    for li, lay in enumerate(layers):
        seen = set()
        for g in lay:
            if g[0] == '1q':
                qs = [g[1]]
            elif g[0] == '2q':
                qs = [g[1], g[2]]
                if g[1] == g[2]:
                    raise ValueError(f"layer {li}: 2Q gate on a single qubit {g[1]}")
            else:
                raise ValueError(f"layer {li}: unknown gate kind {g[0]!r}")
            gate = g[-1]
            if not hasattr(gate, "num_qubits"):
                raise ValueError(f"layer {li}: last element must be a Gate, got {gate!r}")
            if gate.num_qubits != len(qs):
                raise ValueError(
                    f"layer {li}: gate '{gate.name}' acts on {gate.num_qubits} qubits but "
                    f"was placed on {len(qs)}")
            if gate.name == "swap":
                raise ValueError(
                    "logical SWAP in the layer list: the hardware model prices a SWAP as "
                    "3 native 2Q gates, but a source-level SWAP here would be charged one. "
                    "Decompose it (to_cx_basis / swap_policy='expand').")
            for q in qs:
                if q in seen:
                    raise ValueError(
                        f"layer {li}: qubit {q} appears in two gates of the same layer -- "
                        "a layer must be a set of simultaneous, disjoint operations")
                seen.add(q)
            if n_qubits is not None and any(q >= n_qubits or q < 0 for q in qs):
                raise ValueError(f"layer {li}: qubit index out of range for {n_qubits} qubits")
    return True


# ---------------------------------------------------------------------------
# Direct construction -- the path synthetic generators should use
# ---------------------------------------------------------------------------

def make_layers(spec, n_qubits=None, gate_2q=None, gate_1q=None, name=""):
    """Build a LayeredCircuit DIRECTLY from an explicit layer grid. Nothing is
    re-layered, so the active/idle pattern in `spec` is exactly what gets scored.

    `spec` is a list of layers. Each layer is a list whose entries may be:

        (a, b)                   a 2Q gate on qubits a, b     -> uses `gate_2q` (default CX)
        a                        a 1Q gate on qubit a         -> uses `gate_1q` (default H)
        ('2q', a, b, gate)       fully explicit, passed through
        ('1q', a, gate)          fully explicit, passed through
        ('2q', a, b) / ('1q', a) explicit kind, default gate

    An empty layer is legal and means "every qubit idles here" -- which is often
    exactly what an idle-heavy family needs.

        make_layers([[(0, 1), (2, 3)],
                     [(1, 2)],
                     [],                      # everyone idles
                     [(0, 1)]], n_qubits=8)
    """
    g2 = gate_2q if gate_2q is not None else CXGate()
    g1 = gate_1q if gate_1q is not None else HGate()

    layers = []
    for li, lay in enumerate(spec):
        out = []
        for item in lay:
            if isinstance(item, int):
                out.append(('1q', item, g1))
            elif isinstance(item, (tuple, list)) and len(item) == 2 and \
                    all(isinstance(x, int) for x in item):
                out.append(('2q', item[0], item[1], g2))
            elif isinstance(item, (tuple, list)) and item and item[0] == '1q':
                out.append(('1q', item[1], item[2] if len(item) > 2 else g1))
            elif isinstance(item, (tuple, list)) and item and item[0] == '2q':
                out.append(('2q', item[1], item[2], item[3] if len(item) > 3 else g2))
            else:
                raise ValueError(f"layer {li}: cannot interpret entry {item!r}")
        layers.append(out)

    if n_qubits is None:
        used = {q for lay in layers for g in lay
                for q in (g[1:3] if g[0] == '2q' else [g[1]])}
        n_qubits = (max(used) + 1) if used else 0

    validate_layers(layers, n_qubits)
    return LayeredCircuit(layers=layers, n_qubits=n_qubits, name=name)


# ---------------------------------------------------------------------------
# Basis normalisation
# ---------------------------------------------------------------------------

def to_cx_basis(circuit, optimization_level=0):
    """Transpile to a {CX + 1Q} basis, with no routing and no coupling map.

    Removes the whole class of composite-gate mispricing at once: logical SWAPs
    become 3 CX (matching how the hardware model prices a SWAP), and CCX, CSWAP,
    iSWAP etc. all reduce to operations the model actually knows how to charge.
    Recommended for every MQT Bench circuit before `circuit_to_layers`.

    `optimization_level=0` by default so the gate count you asked for is the gate
    count you get -- raise it only if you deliberately want Qiskit to simplify.
    """
    from qiskit import transpile
    return transpile(circuit, basis_gates=["cx", "u"], coupling_map=None,
                     optimization_level=optimization_level)


# ---------------------------------------------------------------------------
# Conversion from a QuantumCircuit
# ---------------------------------------------------------------------------

def _flatten(circuit, max_depth=6):
    """Yield (op, [qubit_indices]) with every operation reduced to <= 2 qubits."""
    def walk(circ, mapping, depth):
        for inst in circ.data:
            idx = [mapping[circ.find_bit(b).index] for b in inst.qubits]
            op = inst.operation
            # barrier/delay/measure carry no definition and may be wider than 2 qubits;
            # let them through untouched and let the caller drop or reject them.
            if op.name in _DROPPABLE or op.name in _TERMINAL_ONLY:
                yield op, idx
                continue
            if op.num_qubits <= 2:
                yield op, idx
                continue
            if depth >= max_depth:
                raise ValueError(
                    f"gate '{op.name}' acts on {op.num_qubits} qubits and could not be "
                    f"reduced to 1Q/2Q within {max_depth} levels of definition")
            defn = getattr(op, "definition", None)
            if defn is None:
                raise ValueError(
                    f"gate '{op.name}' acts on {op.num_qubits} qubits and has no "
                    "definition to decompose; run to_cx_basis() first")
            yield from walk(defn, idx, depth + 1)

    yield from walk(circuit, list(range(circuit.num_qubits)), 0)


def circuit_to_layers(circuit, name=None, strict=False, swap_policy="raise"):
    """Convert a QuantumCircuit into a `LayeredCircuit` by ASAP layering.

    Barriers act as alignment points, so an intended layer grid survives the round
    trip. Even so, prefer `make_layers` for synthetic circuits -- see module docs.

    Parameters
    ----------
    circuit     : QuantumCircuit
    name        : label carried into the result (defaults to `circuit.name`)
    strict      : raise if ANY operation is dropped, instead of recording it. Use for
                  benchmark circuits where a silent drop would change the depth.
    swap_policy : what to do with a logical SWAP in the source circuit.
                  "raise"  (default) -- refuse; it would be charged 1x f2q instead of
                           the f2q**3 the hardware model prices a SWAP at.
                  "expand" -- rewrite as 3 CX, matching the hardware model.
                  "as_gate" -- charge it as a single 2Q gate. Knowingly under-priced;
                           for sensitivity analysis only.
    """
    if swap_policy not in ("raise", "expand", "as_gate"):
        raise ValueError(f"swap_policy must be raise/expand/as_gate, got {swap_policy!r}")

    n = circuit.num_qubits
    ops = list(_flatten(circuit))

    # Terminal measure/reset may be stripped; mid-circuit ones may not. Find the last
    # operation on each wire that is NOT a measure/reset -- anything of that kind
    # occurring earlier is mid-circuit.
    last_real = {}
    for i, (op, idx) in enumerate(ops):
        if op.name not in _TERMINAL_ONLY and op.name not in _DROPPABLE:
            for q in idx:
                last_real[q] = i

    dropped = {}
    layers, frontier = [], [0] * n

    def place(entry, idx):
        L = max(frontier[q] for q in idx)
        while len(layers) <= L:
            layers.append([])
        layers[L].append(entry)
        for q in idx:
            frontier[q] = L + 1

    for i, (op, idx) in enumerate(ops):
        if getattr(op, "condition", None) is not None:
            raise ValueError(
                f"classically-conditioned operation '{op.name}' is not supported: the "
                "harness lowers a fixed gate list with no classical control flow")

        if op.name in _TERMINAL_ONLY:
            if any(i < last_real.get(q, -1) for q in idx):
                raise ValueError(
                    f"mid-circuit '{op.name}' on qubit(s) {idx}: dropping it would change "
                    "what the circuit computes. Such circuits are out of scope.")
            dropped[op.name] = dropped.get(op.name, 0) + 1
            continue

        if op.name == "barrier":
            # Alignment point: nothing after it on these wires may be packed before it.
            # A full-width barrier therefore preserves an intended layer grid exactly.
            dropped["barrier"] = dropped.get("barrier", 0) + 1
            if idx:
                b = max(frontier[q] for q in idx)
                for q in idx:
                    frontier[q] = b
            continue

        if op.name in _ADVANCING:
            # Emits no gate, but consumes a layer slot so an intended idle survives.
            dropped[op.name] = dropped.get(op.name, 0) + 1
            for q in idx:
                frontier[q] += 1
            continue

        if op.name == "swap" and op.num_qubits == 2:
            if swap_policy == "raise":
                raise ValueError(
                    "logical SWAP in the source circuit. The hardware model prices a SWAP "
                    "as 3 native 2Q gates (f2q**3, 3*t2q) and charges routing-inserted "
                    "SWAPs that way, but a source-level SWAP would be charged a single "
                    "f2q/t2q -- a 3x under-price. Run to_cx_basis(circuit) first, or pass "
                    "swap_policy='expand'.")
            if swap_policy == "expand":
                a, b = idx
                for ctl, tgt in ((a, b), (b, a), (a, b)):
                    place(('2q', ctl, tgt, CXGate()), [ctl, tgt])
                continue
            # "as_gate": fall through, knowingly under-priced

        if op.num_qubits == 1:
            entry = ('1q', idx[0], op)
        elif op.num_qubits == 2:
            if idx[0] == idx[1]:
                raise ValueError(f"2Q gate '{op.name}' on a single qubit {idx[0]}")
            entry = ('2q', idx[0], idx[1], op)
        else:                                   # unreachable: _flatten guarantees <= 2
            raise ValueError(f"gate '{op.name}' still acts on {op.num_qubits} qubits")

        place(entry, idx)

    if strict and dropped:
        raise ValueError(f"strict=True and operations were dropped: {dropped}")

    return LayeredCircuit(layers=layers, n_qubits=n,
                          name=name if name is not None else (circuit.name or ""),
                          dropped=dropped)


def from_qasm(source, name=None, strict=False, swap_policy="raise"):
    """Load OpenQASM from a file path or a raw string, then layer it.

    Tries QASM3 first, then QASM2, so MQT Bench exports in either dialect work.
    QASM3 needs `pip install qiskit-qasm3-import`.
    """
    import os

    text = source
    if isinstance(source, str) and os.path.exists(source):
        with open(source) as fh:
            text = fh.read()
        if name is None:
            name = os.path.splitext(os.path.basename(source))[0]

    errors = []
    for loader in ("qasm3", "qasm2"):
        try:
            if loader == "qasm3":
                from qiskit import qasm3
                qc = qasm3.loads(text)
            else:
                from qiskit import qasm2
                qc = qasm2.loads(text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
            return circuit_to_layers(qc, name=name, strict=strict, swap_policy=swap_policy)
        except ValueError:
            raise                                          # our own diagnostics, not a parse failure
        except Exception as exc:                           # noqa: BLE001 - report both dialects
            errors.append(f"{loader}: {exc}")
    raise ValueError("could not parse as OpenQASM 3 or 2 --\n  " + "\n  ".join(errors))


def layers_to_circuit(layers, n_qubits=None):
    """Inverse of `circuit_to_layers`: rebuild a flat QuantumCircuit.

    Used to check the adapter round-trips, and to hand a layered circuit to any
    Qiskit tool that expects a circuit.
    """
    if n_qubits is None:
        n_qubits = getattr(layers, "n_qubits", None)
    if n_qubits is None:
        used = {q for lay in layers for g in lay
                for q in (g[1:3] if g[0] == '2q' else [g[1]])}
        n_qubits = (max(used) + 1) if used else 0
    qc = QuantumCircuit(n_qubits)
    for lay in layers:
        for g in lay:
            if g[0] == '1q':
                qc.append(g[2], [g[1]])
            else:
                qc.append(g[3], [g[1], g[2]])
    return qc

