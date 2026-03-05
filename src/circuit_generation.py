"""circuit_generation.py

Circuit generators used by the HQC scheduler pipeline.

This module focuses on *logical* circuit structure (who interacts with whom,
roughly when). It does **not** attempt to respect a physical connectivity map.

Key generators:
  - generate_random_circuit_custom: sparse random circuits (baseline / QV-like).
  - generate_roi_composed_circuit: ROI-based spatiotemporal generator with 4
    tiling options (OP1/OP2A/OP2B/OP3) over an (atomic_layer × qubit) canvas.

Notes:
  - The ROI generator can insert barriers at every atomic layer
    (use_barriers=True) so the intended time grid is preserved.
  - "Soft layer guarantees": within an atomic layer we may place multiple ops
    on the same qubit; this is treated as a realism/noise feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit


# -----------------------------------------------------------------------------
# Small baseline: QFT
# -----------------------------------------------------------------------------

def generate_qft_circuit(num_qubits: int) -> QuantumCircuit:
    """Generate a Quantum Fourier Transform circuit (logical, no measurements)."""
    qc = QuantumCircuit(num_qubits, name="qft")

    for j in range(num_qubits):
        qc.h(j)
        for k in range(j + 1, num_qubits):
            angle = 2 * np.pi / (2 ** (k - j + 1))
            qc.cp(angle, k, j)

    for i in range(num_qubits // 2):
        qc.swap(i, num_qubits - i - 1)

    return qc


# -----------------------------------------------------------------------------
# Random gate helpers
# -----------------------------------------------------------------------------

ONE_QUBIT_GATES: Sequence[str] = ("h", "x", "y", "z", "s", "t")
TWO_QUBIT_GATES: Sequence[str] = ("cx", "cz", "swap")


def apply_random_1q_gate(qc: QuantumCircuit, q: int, rng: np.random.RandomState) -> None:
    gate_type = rng.choice(ONE_QUBIT_GATES)
    if gate_type == "h":
        qc.h(q)
    elif gate_type == "x":
        qc.x(q)
    elif gate_type == "y":
        qc.y(q)
    elif gate_type == "z":
        qc.z(q)
    elif gate_type == "s":
        qc.s(q)
    else:  # 't'
        qc.t(q)


def apply_random_2q_gate(qc: QuantumCircuit, q1: int, q2: int, rng: np.random.RandomState) -> None:
    if q1 == q2:
        return
    gate_type = rng.choice(TWO_QUBIT_GATES)
    if gate_type == "cx":
        qc.cx(q1, q2)
    elif gate_type == "cz":
        qc.cz(q1, q2)
    else:  # 'swap'
        qc.swap(q1, q2)


# -----------------------------------------------------------------------------
# Random sparse circuits (baseline / QV-like)
# -----------------------------------------------------------------------------

def generate_random_circuit_custom(
    num_qubits: int = 10,
    depth: int = 20,
    gate_density: float = 0.3,
    seed: Optional[int] = None,
    two_qubit_ratio: float = 0.5,
    use_barriers: bool = True,
) -> QuantumCircuit:
    """Generate a sparse random circuit with controllable density."""
    rng = np.random.RandomState(seed)
    qc = QuantumCircuit(num_qubits)

    for _ in range(depth):
        active = [q for q in range(num_qubits) if rng.rand() < gate_density]
        rng.shuffle(active)

        i = 0
        while i < len(active):
            if i + 1 < len(active) and rng.rand() < two_qubit_ratio:
                apply_random_2q_gate(qc, active[i], active[i + 1], rng)
                i += 2
            else:
                apply_random_1q_gate(qc, active[i], rng)
                i += 1

        if use_barriers:
            qc.barrier(*range(num_qubits))

    return qc


# -----------------------------------------------------------------------------
# ROI-based spatiotemporal generator
# -----------------------------------------------------------------------------

Number = Union[int, float]
Range = Tuple[Number, Number]


def _as_int(x: Union[int, Range], rng: np.random.RandomState) -> int:
    if isinstance(x, (tuple, list)):
        lo, hi = int(x[0]), int(x[1])
        if lo == hi:
            return lo
        return int(rng.randint(lo, hi + 1))
    return int(x)


def _as_float(x: Union[float, Range], rng: np.random.RandomState) -> float:
    if isinstance(x, (tuple, list)):
        lo, hi = float(x[0]), float(x[1])
        if lo == hi:
            return lo
        return float(rng.uniform(lo, hi))
    return float(x)


def _ratio_to_p2(twoq_to_oneq_ratio: float) -> float:
    """Convert a 2Q:1Q ratio into a probability of choosing 2Q over 1Q."""
    r = max(0.0, float(twoq_to_oneq_ratio))
    return r / (1.0 + r) if r > 0 else 0.0


@dataclass(frozen=True)
class Rect:
    """A non-overlapping rectangle on the (time × qubit) canvas."""

    t0: int
    t1: int
    q0: int
    q1: int
    roi: str = ""  # filled after assignment

    @property
    def area(self) -> int:
        return (self.t1 - self.t0) * (self.q1 - self.q0)

    def qubits(self) -> List[int]:
        return list(range(self.q0, self.q1))


def _partition_1d(
    total: int,
    n_big: int,
    big_min: int,
    big_max: int,
    small_min: int,
    small_max: int,
    rng: np.random.RandomState,
    shuffle: bool = True,
    max_tries: int = 50,
) -> List[int]:
    """Partition an integer length into segments.

    Allocates `n_big` large segments first (long/tall blocks), then fills the rest
    with smaller segments. Final segment is clamped to match `total` exactly.
    """
    total = int(total)
    if total <= 0:
        return []

    small_min = max(1, int(small_min))
    small_max = max(small_min, int(small_max))
    big_min = max(small_min, int(big_min))
    big_max = max(big_min, int(big_max))

    n_big = max(0, int(n_big))
    n_big = min(n_big, total // big_min) if big_min > 0 else 0

    for _ in range(max_tries):
        big_sizes = [int(rng.randint(big_min, big_max + 1)) for _ in range(n_big)]
        remaining = total - sum(big_sizes)
        if remaining < 0:
            continue

        segs: List[int] = list(big_sizes)

        while remaining > 0:
            if remaining <= small_max:
                segs.append(max(small_min, remaining))
                remaining = total - sum(segs)
                continue

            s = int(rng.randint(small_min, small_max + 1))
            if s > remaining:
                s = remaining
            segs.append(s)
            remaining -= s

        diff = total - sum(segs)
        if diff != 0:
            segs[-1] += diff

        if sum(segs) == total and all(v > 0 for v in segs):
            if shuffle:
                rng.shuffle(segs)
            return segs

    return [total]


def _segments_to_bounds(sizes: List[int]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    cur = 0
    for s in sizes:
        out.append((cur, cur + s))
        cur += s
    return out


# ROI library (non-idle). A circuit samples a subset of these (n_rois).
ROI_LIBRARY: Tuple[str, ...] = (
    "1q_heavy",
    "2q_dense_short",
    "2q_sparse_long",
    "streaming",
    "brickwork_entangler",
    "swap_network",
)


def _fill_roi_layer(
    qc: QuantumCircuit,
    roi: str,
    qubits: List[int],
    t_local: int,
    rng: np.random.RandomState,
    p2_default: float,
) -> None:
    """Emit operations for a single ROI for one atomic layer.

    Soft layering: we do not guarantee disjointness within a layer.
    """
    if not qubits:
        return

    n = len(qubits)

    if roi == "idle":
        return

    if roi == "1q_heavy":
        for q in qubits:
            if rng.rand() < 0.75:
                apply_random_1q_gate(qc, q, rng)
        if n >= 2 and rng.rand() < 0.5 * p2_default:
            q1, q2 = rng.choice(qubits, size=2, replace=False)
            apply_random_2q_gate(qc, int(q1), int(q2), rng)
        return

    if roi == "2q_dense_short":
        offset = 0 if (t_local % 2 == 0) else 1
        for i in range(offset, n - 1, 2):
            if rng.rand() < 0.85:
                apply_random_2q_gate(qc, qubits[i], qubits[i + 1], rng)
        for q in qubits:
            if rng.rand() < 0.15:
                apply_random_1q_gate(qc, q, rng)
        return

    if roi == "2q_sparse_long":
        max_pairs = 1 if n < 8 else 2
        dist_thr = max(2, n // 2)
        pairs_added = 0
        for _ in range(6):
            if pairs_added >= max_pairs:
                break
            q1, q2 = rng.choice(qubits, size=2, replace=False)
            if abs(int(q1) - int(q2)) >= dist_thr:
                apply_random_2q_gate(qc, int(q1), int(q2), rng)
                pairs_added += 1
        for q in qubits:
            if rng.rand() < 0.25:
                apply_random_1q_gate(qc, q, rng)
        return

    if roi == "brickwork_entangler":
        offset = 0 if (t_local % 2 == 0) else 1
        for i in range(offset, n - 1, 2):
            apply_random_2q_gate(qc, qubits[i], qubits[i + 1], rng)
        for q in qubits:
            if rng.rand() < 0.20:
                apply_random_1q_gate(qc, q, rng)
        return

    if roi == "swap_network":
        offset = 0 if (t_local % 2 == 0) else 1
        for i in range(offset, n - 1, 2):
            qc.swap(qubits[i], qubits[i + 1])
        for i in range(offset, n - 1, 2):
            if rng.rand() < 0.35:
                apply_random_2q_gate(qc, qubits[i], qubits[i + 1], rng)
        return

    if roi == "streaming":
        max_active = min(n, 1 + t_local // 2)
        eligible = qubits[:max_active]
        active = [q for q in eligible if rng.rand() < 0.55]
        if not active and eligible and rng.rand() < 0.25:
            active = [int(rng.choice(eligible))]
        for q in active:
            if rng.rand() < 0.85:
                apply_random_1q_gate(qc, q, rng)
        if len(active) >= 2 and rng.rand() < 0.6 * p2_default:
            q1, q2 = rng.choice(active, size=2, replace=False)
            apply_random_2q_gate(qc, int(q1), int(q2), rng)
        return

    for q in qubits:
        if rng.rand() < 0.5:
            apply_random_1q_gate(qc, q, rng)


def _sprinkle_block_noise(
    qc: QuantumCircuit,
    qubits: List[int],
    rng: np.random.RandomState,
    noise_1q_prob: float,
    noise_2q_prob: float,
) -> None:
    """Block-local noise (primary mechanism to break layer guarantees)."""
    if not qubits:
        return

    for q in qubits:
        if rng.rand() < noise_1q_prob:
            apply_random_1q_gate(qc, q, rng)

    if len(qubits) >= 2:
        attempts = max(1, len(qubits) // 4)
        for _ in range(attempts):
            if rng.rand() < noise_2q_prob:
                q1, q2 = rng.choice(qubits, size=2, replace=False)
                apply_random_2q_gate(qc, int(q1), int(q2), rng)


def generate_roi_composed_circuit(
    num_qubits: int,
    num_layers: int,
    option: str = "op2a",
    n_rois: int = 3,
    twoq_to_oneq_ratio: float = 0.6,
    idle_density: float = 0.2,
    p_bridge_boundary: Union[float, Range] = (0.10, 0.20),
    p_bridge_interior: Union[float, Range] = (0.01, 0.05),
    noise_1q_prob: float = 0.02,
    noise_2q_prob: float = 0.004,
    measure_frac: float = 0.0,
    # Rectangle bounds (used for "small" segments during tiling)
    min_block_w: int = 2,
    max_block_w: int = 18,
    min_block_h: int = 2,
    max_block_h: int = 16,
    # Long/tall block allocation (modularity proxies)
    n_long: Union[int, Range] = (2, 5),
    long_w_min: int = 12,
    long_w_max: int = 40,
    n_tall: Union[int, Range] = (1, 3),
    tall_h_min: int = 10,
    tall_h_max: int = 30,
    use_barriers: bool = True,
    seed: Optional[int] = None,
    debug: bool = False,
) -> QuantumCircuit:
    """Generate an ROI-composed circuit by tiling the (layer × qubit) canvas.

    Four tiling options over an (atomic_layer × qubit) canvas:
      - OP1 : fixed spatial bands + fixed time slices
      - OP2A: fixed spatial bands + varied time slices (per band)
      - OP2B: varied spatial bands (per time slice) + fixed time slices
      - OP3 : varied spatial (per time slice) + varied time (global)

    Per-circuit behavior:
      1) Sample `n_rois` ROI types from ROI_LIBRARY (excluding "idle"). Idle
         is handled separately via `idle_density`.
      2) Assign enough rectangles to "idle" until idle volume
         >= idle_density * num_qubits * num_layers (overshoot allowed).
      3) Assign remaining rectangles uniformly from the sampled ROI subset.
      4) Fill each atomic layer by applying ROI patterns inside each active
         rectangle, then sprinkle block-local noise.
      5) Add cross-rectangle 2Q bridges with probabilities sampled from
         (min,max) for boundary vs interior layers.
      6) Optionally measure a fraction of qubits at the end.

    Returns:
        Qiskit QuantumCircuit.
    """

    if num_qubits <= 0 or num_layers <= 0:
        raise ValueError("num_qubits and num_layers must be positive")

    opt = option.lower()
    if opt not in {"op1", "op2a", "op2b", "op3"}:
        raise ValueError(f"Unknown option={option!r}. Expected op1/op2a/op2b/op3")

    rng = np.random.RandomState(seed)

    p_bdry = _as_float(p_bridge_boundary, rng)
    p_int = _as_float(p_bridge_interior, rng)

    p2_default = _ratio_to_p2(twoq_to_oneq_ratio)

    # Choose per-circuit ROI subset (excluding idle).
    n_rois = int(max(1, min(len(ROI_LIBRARY), n_rois)))
    chosen_rois = list(rng.choice(list(ROI_LIBRARY), size=n_rois, replace=False))

    # Long/tall modularity knobs per circuit.
    n_long_i = _as_int(n_long, rng)
    n_tall_i = _as_int(n_tall, rng)

    rects: List[Rect] = []

    # Fixed partitions when needed.
    if opt in {"op1", "op2a"}:
        h_sizes = _partition_1d(
            total=num_qubits,
            n_big=n_tall_i,
            big_min=tall_h_min,
            big_max=tall_h_max,
            small_min=min_block_h,
            small_max=max_block_h,
            rng=rng,
        )
        h_bounds = _segments_to_bounds(h_sizes)

    if opt in {"op1", "op2b", "op3"}:
        w_sizes = _partition_1d(
            total=num_layers,
            n_big=n_long_i,
            big_min=long_w_min,
            big_max=long_w_max,
            small_min=min_block_w,
            small_max=max_block_w,
            rng=rng,
        )
        w_bounds = _segments_to_bounds(w_sizes)

    # Tiling by option
    if opt == "op1":
        for (t0, t1) in w_bounds:
            for (q0, q1) in h_bounds:
                rects.append(Rect(t0=t0, t1=t1, q0=q0, q1=q1))

    elif opt == "op2a":
        for (q0, q1) in h_bounds:
            w_sizes_i = _partition_1d(
                total=num_layers,
                n_big=n_long_i,
                big_min=long_w_min,
                big_max=long_w_max,
                small_min=min_block_w,
                small_max=max_block_w,
                rng=rng,
            )
            for (t0, t1) in _segments_to_bounds(w_sizes_i):
                rects.append(Rect(t0=t0, t1=t1, q0=q0, q1=q1))

    elif opt == "op2b":
        for (t0, t1) in w_bounds:
            h_sizes_j = _partition_1d(
                total=num_qubits,
                n_big=n_tall_i,
                big_min=tall_h_min,
                big_max=tall_h_max,
                small_min=min_block_h,
                small_max=max_block_h,
                rng=rng,
            )
            for (q0, q1) in _segments_to_bounds(h_sizes_j):
                rects.append(Rect(t0=t0, t1=t1, q0=q0, q1=q1))

    else:  # op3
        for (t0, t1) in w_bounds:
            h_sizes_j = _partition_1d(
                total=num_qubits,
                n_big=n_tall_i,
                big_min=tall_h_min,
                big_max=tall_h_max,
                small_min=min_block_h,
                small_max=max_block_h,
                rng=rng,
            )
            for (q0, q1) in _segments_to_bounds(h_sizes_j):
                rects.append(Rect(t0=t0, t1=t1, q0=q0, q1=q1))

    if not rects:
        raise RuntimeError("Failed to tile canvas into rectangles")

    # Step 3: idle-first assignment until budget reached (overshoot allowed).
    total_volume = float(num_qubits * num_layers)
    idle_target = float(idle_density) * total_volume
    idle_target = max(0.0, idle_target)

    order = list(range(len(rects)))
    rng.shuffle(order)

    idle_volume = 0.0
    is_idle = [False] * len(rects)
    for idx in order:
        if idle_volume >= idle_target:
            break
        is_idle[idx] = True
        idle_volume += rects[idx].area

    assigned: List[Rect] = []
    for i, r in enumerate(rects):
        if is_idle[i]:
            assigned.append(Rect(r.t0, r.t1, r.q0, r.q1, roi="idle"))
        else:
            assigned.append(Rect(r.t0, r.t1, r.q0, r.q1, roi=str(rng.choice(chosen_rois))))
    rects = assigned

    # Per-layer lookups
    rects_by_layer: List[List[int]] = [[] for _ in range(num_layers)]
    boundary_layer = np.zeros(num_layers, dtype=bool)

    for ridx, r in enumerate(rects):
        for t in range(r.t0, r.t1):
            rects_by_layer[t].append(ridx)
        if 0 <= r.t0 < num_layers:
            boundary_layer[r.t0] = True
        if 0 <= (r.t1 - 1) < num_layers:
            boundary_layer[r.t1 - 1] = True

    # Allocate classical bits only if measuring.
    m = int(np.ceil(float(measure_frac) * num_qubits)) if measure_frac > 0 else 0
    qc = QuantumCircuit(num_qubits, m) if m > 0 else QuantumCircuit(num_qubits)

    # Fill layers
    for t in range(num_layers):
        active_rects = rects_by_layer[t]

        for ridx in active_rects:
            r = rects[ridx]
            qs = r.qubits()
            t_local = t - r.t0
            _fill_roi_layer(qc, r.roi, qs, t_local, rng, p2_default)
            _sprinkle_block_noise(qc, qs, rng, noise_1q_prob, noise_2q_prob)

        # Bridges
        if len(active_rects) >= 2:
            p_bridge = p_bdry if boundary_layer[t] else p_int
            if rng.rand() < p_bridge:
                # Add 1–2 bridges (fixed behavior; not a config parameter).
                for _ in range(2):
                    ra, rb = rng.choice(active_rects, size=2, replace=False)
                    qa = int(rng.choice(rects[ra].qubits()))
                    qb = int(rng.choice(rects[rb].qubits()))
                    apply_random_2q_gate(qc, qa, qb, rng)
                    if rng.rand() < 0.5:
                        break

        if use_barriers:
            qc.barrier(*range(num_qubits))

    # End-only measurement
    if m > 0:
        meas_qubits = list(range(num_qubits))
        rng.shuffle(meas_qubits)
        meas_qubits = meas_qubits[:m]
        for ci, q in enumerate(meas_qubits):
            qc.measure(q, ci)

    return qc
