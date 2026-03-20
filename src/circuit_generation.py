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
from collections import Counter


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

        # Fill the remainder with "small" segments while respecting small_min.
        # We allow the *last* segment to be < small_min only when total < small_min.
        while remaining > 0:
            # If what's left is smaller than small_min, merge it into the previous segment.
            if remaining < small_min:
                if segs:
                    segs[-1] += remaining
                    remaining = 0
                    break
                # Degenerate case: total itself is smaller than small_min.
                segs.append(remaining)
                remaining = 0
                break

            # If the remainder fits into one segment, finish.
            if remaining <= small_max:
                segs.append(remaining)
                remaining = 0
                break

            # Sample a segment size, but leave enough room for a final segment >= small_min.
            s = int(rng.randint(small_min, small_max + 1))
            if remaining - s < small_min:
                s = remaining - small_min
            s = max(small_min, min(s, remaining))
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


# ROI library split into buckets for balanced sampling.
#
# Bucket structure:
#   ROI_1Q_BUCKET   : 1Q-dominant motifs (single-qubit heavy)
#   ROI_2Q_BUCKET   : 2Q-dominant motifs (two-qubit heavy, local or long-range)
#   ROI_SPECIAL     : rare / structural motifs sampled with a separate low probability
#
# The full library is the union of all three, preserved for legacy reference.
ROI_1Q_BUCKET: Tuple[str, ...] = (
    "1q_dense",             # 80% per-qubit 1Q gate density, rare 2Q sprinkle
    "1q_sparse",            # 30-40% per-qubit 1Q density, zero 2Q — low-activity phase
    "streaming",            # gradual qubit activation, 1Q-only (state prep model)
    "mixed_sparse",         # sparse, balanced 1Q/2Q — sits at the 1Q/2Q boundary
    "parametric_rotation",  # every qubit gets an Rx/Ry/Rz each layer, zero 2Q
                            # models VQE/QAOA rotation layers
)

ROI_2Q_BUCKET: Tuple[str, ...] = (
    # Density/range corners
    "2q_dense_short",       # short-range, moderate density
    "2q_sparse_long",       # long-range, sparse
    "2q_dense_long",        # long-range, dense (cross-half matching)
    # Structured local patterns
    "brickwork_entangler",  # alternating-offset nearest-neighbour pairs
    "swap_network",         # dense SWAP chain
    # Structured motifs
    "star_entangler",       # GHZ / hub-and-spoke (parallel fan)
    "nn_ladder",            # moving nearest-neighbour rung
    "ripple_carry",         # chain propagation (long rectangles)
    "enc_dec",              # encode/decode bookend (long rectangles)
    "qft_like",             # QFT-like long-range structured interactions
    "ghz_chain",            # sequential GHZ: CNOT(0,1), CNOT(1,2), ... one step/layer
    "trotter_zz",           # Trotterized ZZ Hamiltonian: fixed qubit pairs repeat each layer
)

ROI_SPECIAL: Tuple[str, ...] = (
    "bridge_burst",   # burst of extra cross-ROI bridges; kept rare
)

# Full library union (for legacy / debug use)
ROI_LIBRARY: Tuple[str, ...] = ROI_1Q_BUCKET + ROI_2Q_BUCKET + ROI_SPECIAL


def _sample_roi_subset(
    n_rois: int,
    rng: np.random.RandomState,
    p_special: float = 0.05,
    twoq_frac_lo: float = 0.30,
    twoq_frac_hi: float = 0.80,
) -> List[str]:
    """Sample a balanced subset of ROI types for one circuit.

    Algorithm
    ---------
    1. Special slot: with probability ``p_special`` reserve one slot for a
       randomly chosen ROI from ROI_SPECIAL (e.g. bridge_burst).  This keeps
       burst-communication motifs rare but present in the dataset.
    2. Remaining slots are split between 2Q-heavy and 1Q-heavy buckets using a
       ratio r ~ Uniform[twoq_frac_lo, twoq_frac_hi].  This guarantees a floor
       on 1Q representation regardless of total n_rois, unlike the old uniform
       draw which could easily produce all-2Q subsets.
    3. Within each bucket, sampling is without replacement (clamped to bucket
       size if necessary).

    The result is a list of ``n_rois`` unique ROI names.
    """
    n_rois = int(max(1, n_rois))
    chosen: List[str] = []

    # Step 1: optional special slot
    remaining = n_rois
    if remaining > 0 and len(ROI_SPECIAL) > 0 and rng.rand() < p_special:
        special_roi = str(rng.choice(list(ROI_SPECIAL)))
        chosen.append(special_roi)
        remaining -= 1

    if remaining == 0:
        return chosen

    # Step 2: split remaining between 2Q and 1Q buckets
    r1 = float(rng.uniform(twoq_frac_lo, twoq_frac_hi))
    n_2q = int(round(remaining * r1))
    n_1q = remaining - n_2q

    # Clamp to bucket sizes and redistribute overflow
    n_2q = min(n_2q, len(ROI_2Q_BUCKET))
    n_1q = min(n_1q, len(ROI_1Q_BUCKET))

    # If clamping created a shortfall, fill from the other bucket
    shortfall = remaining - (n_2q + n_1q)
    if shortfall > 0:
        extra_2q = min(shortfall, len(ROI_2Q_BUCKET) - n_2q)
        n_2q += extra_2q
        shortfall -= extra_2q
    if shortfall > 0:
        extra_1q = min(shortfall, len(ROI_1Q_BUCKET) - n_1q)
        n_1q += extra_1q

    # Step 3: sample without replacement from each bucket
    if n_2q > 0:
        pool_2q = [r for r in ROI_2Q_BUCKET if r not in chosen]
        drawn = list(rng.choice(pool_2q, size=min(n_2q, len(pool_2q)), replace=False))
        chosen.extend(drawn)

    if n_1q > 0:
        pool_1q = [r for r in ROI_1Q_BUCKET if r not in chosen]
        drawn = list(rng.choice(pool_1q, size=min(n_1q, len(pool_1q)), replace=False))
        chosen.extend(drawn)

    rng.shuffle(chosen)
    return chosen


def _fill_roi_layer(
    qc: QuantumCircuit,
    roi: str,
    qubits: List[int],
    t_local: int,
    t_len: int,
    rng: np.random.RandomState,
    p2_default: float,
    dist_thr_long: int,
    rect_key: int,
) -> None:
    """Emit operations for a single ROI for one atomic layer.

    Soft layering: we do not guarantee disjointness within a layer.
    `t_local` is the layer index within the rectangle; `t_len` is rectangle duration.
    """
    if not qubits:
        return

    n = len(qubits)

    if roi == "idle":
        return

    # ----------------------------
    # 1Q-heavy ROI
    # ----------------------------
    if roi == "1q_dense":
        for q in qubits:
            if rng.rand() < 0.80:
                apply_random_1q_gate(qc, q, rng)
        # Occasional 2Q (biased by default p2) just to avoid being perfectly 1Q-only.
        if n >= 2 and rng.rand() < 0.10:
            q1, q2 = rng.choice(qubits, size=2, replace=False)
            apply_random_2q_gate(qc, int(q1), int(q2), rng)
        return

    # ----------------------------
    # Mixed sparse (random 1Q or 2Q with no strong bias)
    # ----------------------------
    if roi == "mixed_sparse":
        if n == 1:
            if rng.rand() < 0.5:
                apply_random_1q_gate(qc, qubits[0], rng)
            return

        frac = float(rng.uniform(0.25, 0.45))
        m = max(1, min(n, int(np.ceil(frac * n))))
        active = sorted(int(x) for x in rng.choice(qubits, size=m, replace=False))

        i = 0
        while i < len(active):
            if i + 1 < len(active) and rng.rand() < 0.5:
                apply_random_2q_gate(qc, active[i], active[i + 1], rng)
                i += 2
            else:
                apply_random_1q_gate(qc, active[i], rng)
                i += 1
        return

    # ----------------------------
    # Short-range (1q or 2q apart) 2Q
    # (dense+short is covered by brickwork_entangler and swap_network)
    # ----------------------------
    if roi in {"2q_dense_short", "2q_short_range"}:
        if n >= 2 and rng.rand() < 0.90:
            if n == 2:
                # only adjacent is possible
                i = int((rect_key + t_local) % (n - 1))
                apply_random_2q_gate(qc, qubits[i], qubits[i + 1], rng)
            else:
                # choose span 1 or 2 with equal probability
                span = 1 if rng.rand() < 0.5 else 2

                # if span=2 is not possible for this block, fall back to span=1
                if span == 2 and n < 3:
                    span = 1

                i = int((rect_key + t_local) % (n - span))
                apply_random_2q_gate(qc, qubits[i], qubits[i + span], rng)
        # Light 1Q sprinkle
        # for q in qubits:
        #     if rng.rand() < 0.04:
        #         apply_random_1q_gate(qc, q, rng)
        return

    # ----------------------------
    # Sparse long-range 2Q
    # ----------------------------
    if roi == "2q_sparse_long":
        if n < 2:
            # Can't form 2Q pairs; fall back to a few 1Q gates.
            for q in qubits:
                if rng.rand() < 0.35:
                    apply_random_1q_gate(qc, q, rng)
            return

        dist_thr = int(max(2, dist_thr_long))
        dist_thr = min(dist_thr, n - 1)  # clamp to ROI size

        # At most 1-2 long-range pairs per layer (sparse)
        max_pairs = 1 if n < 10 else 2
        pairs_added = 0
        for _ in range(10):
            if pairs_added >= max_pairs:
                break
            q1, q2 = rng.choice(qubits, size=2, replace=False)
            if abs(int(q1) - int(q2)) >= dist_thr:
                apply_random_2q_gate(qc, int(q1), int(q2), rng)
                pairs_added += 1

        # for q in qubits:
        #     if rng.rand() < 0.04:
        #         apply_random_1q_gate(qc, q, rng)
        return

    # ----------------------------
    # Dense long-range 2Q
    # (random long-range matching across the band)
    # ----------------------------
    if roi == "2q_dense_long":
        if n < 2:
            for q in qubits:
                if rng.rand() < 0.4:
                    apply_random_1q_gate(qc, q, rng)
            return

        dist_thr = int(max(2, dist_thr_long))
        dist_thr = min(dist_thr, n - 1)

        # Pair across halves to encourage large |i-j|
        qs_sorted = sorted(qubits)
        half = n // 2
        left = qs_sorted[:half]
        right = qs_sorted[half:]
        rng.shuffle(left)
        rng.shuffle(right)

        # Apply as many cross-half pairs as possible (dense)
        k = min(len(left), len(right))
        for i in range(k):
            a, b = int(left[i]), int(right[i])
            if abs(a - b) >= dist_thr or rng.rand() < 0.15:
                apply_random_2q_gate(qc, a, b, rng)

        # # Small 1Q sprinkle
        # for q in qubits:
        #     if rng.rand() < 0.05:
        #         apply_random_1q_gate(qc, q, rng)
        return

    # ----------------------------
    # Brickwork entangler (dense, structured, local)
    # ----------------------------
    if roi == "brickwork_entangler":
        offset = 0 if (t_local % 2 == 0) else 1
        for i in range(offset, n - 1, 2):
            apply_random_2q_gate(qc, qubits[i], qubits[i + 1], rng)
        # for q in qubits:
        #     if rng.rand() < 0.05:
        #         apply_random_1q_gate(qc, q, rng)
        return

    # ----------------------------
    # Swap network (dense, routing-heavy)
    # ----------------------------
    if roi == "swap_network":
        offset = 0 if (t_local % 2 == 0) else 1
        for i in range(offset, n - 1, 2):
            qc.swap(qubits[i], qubits[i + 1])
        # for i in range(offset, n - 1, 2):
        #     if rng.rand() < 0.35:
        #         apply_random_2q_gate(qc, qubits[i], qubits[i + 1], rng)
        return

    # ----------------------------
    # Streaming (gradually activates more qubits over time — state prep model)
    # 1Q-dominant: 2Q gates are intentionally rare (5% fixed, independent of p2_default)
    # ----------------------------
    if roi == "streaming":
        max_active = min(n, 1 + t_local // 2)
        eligible = qubits[:max_active]
        active = [q for q in eligible if rng.rand() < 0.55]
        if not active and eligible and rng.rand() < 0.25:
            active = [int(rng.choice(eligible))]
        for q in active:
            if rng.rand() < 0.85:
                apply_random_1q_gate(qc, q, rng)
        # Very occasional 2Q — fixed low probability, NOT scaled by p2_default
        if len(active) >= 2 and rng.rand() < 0.05:
            q1, q2 = rng.choice(active, size=2, replace=False)
            apply_random_2q_gate(qc, int(q1), int(q2), rng)
        return

    # ----------------------------
    # Star / GHZ-like entangler (hub interacts with many)
    # ----------------------------
    if roi == "star_entangler":
        if n < 2:
            return
        hub = qubits[int(rect_key % n)]
        others = [q for q in qubits if q != hub]
        if not others:
            return
        p1 = others[(t_local + (rect_key // 7)) % len(others)]
        apply_random_2q_gate(qc, int(hub), int(p1), rng)
        if len(others) >= 2 and rng.rand() < 0.35:
            p2 = others[(t_local + (rect_key // 13) + 1) % len(others)]
            if p2 != p1:
                apply_random_2q_gate(qc, int(hub), int(p2), rng)
        # if rng.rand() < 0.25:
        #     apply_random_1q_gate(qc, int(hub), rng)
        return

    # ----------------------------
    # Nearest-neighbor ladder (moving rung)
    # ----------------------------
    if roi == "nn_ladder":
        if n < 2:
            return
        i = int((rect_key + t_local) % (n - 1))
        apply_random_2q_gate(qc, qubits[i], qubits[i + 1], rng)
        # if rng.rand() < 0.25:
        #     apply_random_1q_gate(qc, qubits[i], rng)
        # if rng.rand() < 0.25:
        #     apply_random_1q_gate(qc, qubits[i + 1], rng)
        return

    # ----------------------------
    # Ripple-carry-like (chain propagation; meaningful on long rectangles)
    # ----------------------------
    if roi == "ripple_carry":
        if n < 2:
            return
        if t_len < max(6, n):
            i = int((rect_key + t_local) % (n - 1))
            apply_random_2q_gate(qc, qubits[i], qubits[i + 1], rng)
            return
        i = int((t_local + (rect_key // 5)) % (n - 1))
        apply_random_2q_gate(qc, qubits[i], qubits[i + 1], rng)
        # if (t_local % 3 == 2) and (i + 2 < n):
        #     apply_random_2q_gate(qc, qubits[i + 1], qubits[i + 2], rng)
        return

    # ----------------------------
    # Encoding/decoding motif (bookends on long rectangles)
    # ----------------------------
    if roi == "enc_dec":
        if n < 2:
            return
        if t_len < max(10, n + 4):
            i = int((rect_key + t_local) % (n - 1))
            apply_random_2q_gate(qc, qubits[i], qubits[i + 1], rng)
            return
        span = max(2, min(6, t_len // 5))
        if t_local < span:
            i = int((t_local + (rect_key // 11)) % (n - 1))
            apply_random_2q_gate(qc, qubits[i], qubits[i + 1], rng)
        elif t_local >= (t_len - span):
            i = int((t_len - 1 - t_local + (rect_key // 11)) % (n - 1))
            j = (n - 2 - i)
            j = max(0, min(j, n - 2))
            apply_random_2q_gate(qc, qubits[j], qubits[j + 1], rng)
        else:
            for q in qubits:
                if rng.rand() < 0.35:
                    apply_random_1q_gate(qc, q, rng)
        return

    # ----------------------------
    # QFT-like structured long-range interactions (long rectangles)
    # ----------------------------
    if roi == "qft_like":
        if n < 2:
            return
        if t_len < max(8, n):
            dist_thr = int(max(2, dist_thr_long))
            dist_thr = min(dist_thr, n - 1)
            for _ in range(6):
                q1, q2 = rng.choice(qubits, size=2, replace=False)
                if abs(int(q1) - int(q2)) >= dist_thr:
                    apply_random_2q_gate(qc, int(q1), int(q2), rng)
                    break
            return

        ctrl_idx = int((rect_key + t_local) % n)
        ctrl = qubits[ctrl_idx]
        targets = []
        for k in range(1, min(4, n)):
            tidx = (ctrl_idx + (n // 2) + k) % n
            if tidx != ctrl_idx:
                targets.append(qubits[tidx])
        for tgt in targets:
            qc.cz(int(ctrl), int(tgt))
        # if rng.rand() < 0.35:
        #     qc.h(int(ctrl))
        return

    # ----------------------------
    # Parametric rotation layer (VQE / QAOA ansatz rotation step)
    # Every qubit gets exactly one Rx/Ry/Rz per layer. No 2Q gates.
    # Models the single-qubit rotation layers that dominate VQE circuits.
    # ----------------------------
    if roi == "parametric_rotation":
        ROTATION_GATES = ("rx", "ry", "rz")
        for q in qubits:
            gate = ROTATION_GATES[int(rng.randint(len(ROTATION_GATES)))]
            angle = float(rng.uniform(0, 2 * np.pi))
            if gate == "rx":
                qc.rx(angle, q)
            elif gate == "ry":
                qc.ry(angle, q)
            else:
                qc.rz(angle, q)
        return

    # ----------------------------
    # Sparse 1Q (low-activity idle-like phase)
    # 30-40% per-qubit probability, no 2Q gates.
    # Models ancilla layers, reset windows, or low-activity classical feedback phases.
    # ----------------------------
    if roi == "1q_sparse":
        for q in qubits:
            if rng.rand() < float(rng.uniform(0.30, 0.40)):
                apply_random_1q_gate(qc, q, rng)
        return

    # ----------------------------
    # GHZ chain (sequential entanglement propagation)
    # Each layer applies one CNOT step along the chain: CNOT(i, i+1) where i
    # advances with t_local. Models sequential GHZ state preparation or
    # linear entanglement propagation (distinct from star_entangler fan-out).
    # ----------------------------
    if roi == "ghz_chain":
        if n < 2:
            return
        # Step through the chain: layer 0 applies CNOT(0,1), layer 1 CNOT(1,2), etc.
        # After reaching the end, restart from 0 (allows long rectangles to cycle).
        i = int(t_local % (n - 1))
        qc.cx(qubits[i], qubits[i + 1])
        # Hadamard on the control at the first step to seed superposition
        if t_local == 0:
            qc.h(qubits[0])
        return

    # ----------------------------
    # Trotter ZZ (Trotterized Hamiltonian simulation)
    # Fixed qubit pairs determined by rect_key interact with ZZ (Rzz) gates each layer.
    # The same pairs repeat across layers, matching the structure of Trotterized
    # simulation where the Hamiltonian graph is fixed. Even layers apply one set of
    # pairs, odd layers apply the complementary set (checkerboard Trotter step).
    # ----------------------------
    if roi == "trotter_zz":
        if n < 2:
            return
        # Build a fixed pairing from rect_key: even/odd offset alternates each layer
        # to approximate a 2-coloring Trotter decomposition.
        offset = (t_local + int(rect_key % 2)) % 2
        pairs_applied = 0
        for i in range(offset, n - 1, 2):
            a, b = qubits[i], qubits[i + 1]
            angle = float(np.pi / 4)  # canonical Trotter step angle
            # Rzz(θ) = exp(-i θ/2 ZZ): decomposed as CNOT - Rz - CNOT
            qc.cx(int(a), int(b))
            qc.rz(angle, int(b))
            qc.cx(int(a), int(b))
            pairs_applied += 1
            if pairs_applied >= max(1, n // 4):
                # Keep density moderate — not every pair every layer
                break
        return

    # ----------------------------
    # Bridge-burst ROI: handled at generator-level (adds extra inter-ROI bridges)
    # Still add a bit of local structure here so it's not empty.
    # ----------------------------
    if roi == "bridge_burst":
        for q in qubits:
            if rng.rand() < 0.10:
                apply_random_1q_gate(qc, q, rng)
        if n >= 2 and rng.rand() < 0.10:
            q1, q2 = rng.choice(qubits, size=2, replace=False)
            apply_random_2q_gate(qc, int(q1), int(q2), rng)
        return

    # Fallback: mild 1Q activity
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


def _debug_plot_rects(rects, num_layers, num_qubits):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle as MplRect

    fig, ax = plt.subplots(figsize=(max(8, num_layers/6), max(3, num_qubits/3)))
    ax.set_xlim(0, num_layers)
    ax.set_ylim(0, num_qubits)
    ax.invert_yaxis()
    ax.set_xlabel("layer")
    ax.set_ylabel("qubit")

    for r in rects:
        w = r.t1 - r.t0
        h = r.q1 - r.q0
        ax.add_patch(MplRect((r.t0, r.q0), w, h, fill=False, linewidth=1.5))

        # label only if rectangle is reasonably large
        if w >= 6 and h >= 3:
            ax.text(r.t0 + w/2, r.q0 + h/2, r.roi,
                    ha="center", va="center", fontsize=8)

    fig.tight_layout()
    plt.show()

def generate_roi_composed_circuit(
    num_qubits: int,
    num_layers: int,
    option: str = "op2a",
    n_rois: int = 3,
    twoq_to_oneq_ratio: float = 0.6,
    idle_density: Union[float, Range] = (0.20, 0.35),
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
    roi_debug_prints: Optional[bool] = None,
) -> QuantumCircuit:
    """Generate an ROI-composed circuit by tiling the (layer × qubit) canvas.

    Args:
        roi_debug_prints: Controls ROI metadata/debug printing independently of
            other debug behavior. If None, inherits from ``debug``.

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
    # TODO: make this apercentage based instead a count
    dist_thr_long = 5

    if num_qubits <= 0 or num_layers <= 0:
        raise ValueError("num_qubits and num_layers must be positive")

    opt = option.lower()
    roi_debug_prints = debug if roi_debug_prints is None else bool(roi_debug_prints)
    if opt not in {"op1", "op2a", "op2b", "op3"}:
        raise ValueError(f"Unknown option={option!r}. Expected op1/op2a/op2b/op3")

    rng = np.random.RandomState(seed)

    p_bdry = _as_float(p_bridge_boundary, rng)
    p_int = _as_float(p_bridge_interior, rng)
    idle_density_f = _as_float(idle_density, rng)

    p2_default = _ratio_to_p2(twoq_to_oneq_ratio)

    # Choose per-circuit ROI subset using bucketed balanced sampling.
    # bridge_burst is kept rare (p_special=0.10); 1Q ROIs always get a
    # proportional floor (30–60% of slots) via the twoq_frac split.
    n_rois = int(max(1, min(len(ROI_LIBRARY), n_rois)))
    chosen_rois = _sample_roi_subset(n_rois, rng)
    if debug:
        print(f"\n[ROI-GEN DEBUG] chosen_rois={chosen_rois} (p2_default={p2_default:.3f})")

    # Long/tall modularity knobs per circuit.
    n_long_i = _as_int(n_long, rng)
    n_tall_i = _as_int(n_tall, rng)

    if roi_debug_prints:
        print("\n[ROI-GEN DEBUG] ===== per-circuit sampled knobs =====")
        print(f"num_qubits={num_qubits} num_layers={num_layers} option={opt}")
        print(f"seed={seed}")
        print(f"n_rois={n_rois} chosen_rois={chosen_rois}")
        print(f"twoq_to_oneq_ratio={twoq_to_oneq_ratio} -> p2_default={p2_default:.3f}")
        print(f"idle_density={idle_density} -> resolved={idle_density_f:.3f} idle_target={idle_density_f*num_qubits*num_layers:.1f}")
        print(f"p_bridge_boundary={p_bdry:.4f} p_bridge_interior={p_int:.4f}")
        print(f"noise_1q_prob={noise_1q_prob} noise_2q_prob={noise_2q_prob}")
        print(f"n_long_i={n_long_i} (long_w_min={long_w_min}, long_w_max={long_w_max})")
        print(f"n_tall_i={n_tall_i} (tall_h_min={tall_h_min}, tall_h_max={tall_h_max})")

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
    
    if roi_debug_prints:
        print("\n[ROI-GEN DEBUG] ===== tiling summary =====")
        print(f"#rects={len(rects)}  coverage={sum(r.area for r in rects)} "
            f"(expected {num_qubits*num_layers})")

        # Basic sanity: detect degenerate tiling
        uniq_t_starts = sorted({r.t0 for r in rects})
        uniq_t_ends   = sorted({r.t1 for r in rects})
        uniq_q_starts = sorted({r.q0 for r in rects})
        uniq_q_ends   = sorted({r.q1 for r in rects})
        print(f"unique time starts={len(uniq_t_starts)} unique time ends={len(uniq_t_ends)}")
        print(f"unique qubit starts={len(uniq_q_starts)} unique qubit ends={len(uniq_q_ends)}")

        # Largest rectangles (often reveals the issue)
        top = sorted(rects, key=lambda r: r.area, reverse=True)[:5]
        for i, r in enumerate(top):
            print(f"  top[{i}] area={r.area}  t=[{r.t0},{r.t1})  q=[{r.q0},{r.q1})")

    # Step 3: idle-first assignment until budget reached (overshoot allowed).
    total_volume = float(num_qubits * num_layers)
    idle_target = float(idle_density_f) * total_volume
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

    non_idle_idxs = [i for i in range(len(rects)) if not is_idle[i]]

    # --- Force-assign some eligible rectangles to 2q_sparse_long (if present in chosen_rois) ---
    # Only fires 50% of the time so the dataset has variety in how tall rects are used.
    forced_sparse = set()
    if "2q_sparse_long" in chosen_rois and len(non_idle_idxs) > 0 and rng.rand() < 0.5:
        eligible = [
            i for i in non_idle_idxs
            if (rects[i].q1 - rects[i].q0) >= int(dist_thr_long) + 1
        ]
        rng.shuffle(eligible)

        # Assign to floor(blocks/rois), where blocks ≈ non-idle rectangles.
        n_force = int(len(non_idle_idxs) // max(1, len(chosen_rois)))
        if n_force > 0 and len(eligible) > 0:
            forced_sparse = set(eligible[: min(n_force, len(eligible))])

    # Long-rectangle eligibility thresholds (relative to whole circuit length).
    ripple_min_w = max(8, int(np.ceil(0.8 * max_block_w)))
    encdec_min_w = max(8, int(np.ceil(0.8 * max_block_w)))
    qft_min_w    = max(8, int(np.ceil(0.8 * max_block_w)))

    def _roi_eligible_for_rect(roi_name: str, r: Rect) -> bool:
        w = int(r.t1 - r.t0)
        if roi_name == "ripple_carry":
            return w >= ripple_min_w
        if roi_name == "enc_dec":
            return w >= encdec_min_w
        if roi_name == "qft_like":
            return w >= qft_min_w
        return True

    def _sample_roi_for_rect(r: Rect) -> str:
        # Try a few times to respect eligibility; otherwise fall back to a safe ROI.
        for _ in range(12):
            cand = str(rng.choice(chosen_rois))
            if _roi_eligible_for_rect(cand, r):
                return cand

        # Fallback preference: 1Q-safe ROIs first, then 2Q structured ones.
        # This matters when many chosen_rois are long-rect-only (ripple_carry etc.)
        # and the current rect is too short/narrow to host them.
        for pref in ("1q_dense", "1q_sparse", "parametric_rotation", "streaming",
                     "mixed_sparse", "brickwork_entangler", "swap_network", "2q_dense_short"):
            if pref in chosen_rois:
                return pref
        return str(chosen_rois[0])

    assigned: List[Rect] = []
    for i, r in enumerate(rects):
        if is_idle[i]:
            assigned.append(Rect(r.t0, r.t1, r.q0, r.q1, roi="idle"))
        elif i in forced_sparse:
            assigned.append(Rect(r.t0, r.t1, r.q0, r.q1, roi="2q_sparse_long"))
        else:
            assigned.append(Rect(r.t0, r.t1, r.q0, r.q1, roi=_sample_roi_for_rect(r)))

    rects = assigned

    # Per-rectangle stable keys and RNGs.
    # These support ROIs that require cross-layer stickiness (e.g., star hub),
    # without introducing global hidden state.
    base_seed = int(seed) if seed is not None else 0
    rect_keys: List[int] = []
    rect_rngs: List[np.random.RandomState] = []
    for ridx, r in enumerate(rects):
        key = (
            (base_seed * 1315423911)
            + (ridx * 2654435761)
            + (r.t0 * 97)
            + (r.t1 * 193)
            + (r.q0 * 389)
            + (r.q1 * 911)
        ) & 0xFFFFFFFF
        rect_keys.append(int(key))
        rect_rngs.append(np.random.RandomState(int(key)))

    if roi_debug_prints:
        print("\n[ROI-GEN DEBUG] ===== ROI assignment =====")
        roi_counts = Counter(r.roi for r in rects)
        roi_area = Counter()
        for r in rects:
            roi_area[r.roi] += r.area

        print("ROI counts:", dict(roi_counts))
        print("ROI area (volume):", dict(roi_area))
        print(f"idle_volume={roi_area.get('idle', 0)}  "
            f"target={idle_density_f*num_qubits*num_layers:.1f}")

        # Flag the problematic case explicitly
        non_idle_area = sum(v for k, v in roi_area.items() if k != "idle")
        if non_idle_area == 0:
            print("!!! WARNING: all rectangles are idle -> circuit will have no gates "
                "(except barriers / bridges / noise).")

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

    if debug:
        active_counts = [len(lst) for lst in rects_by_layer]
        print("\n[ROI-GEN DEBUG] ===== per-layer activity =====")
        print(f"active rects per layer: min={min(active_counts)} "
            f"max={max(active_counts)} avg={sum(active_counts)/len(active_counts):.2f}")

        # Bridges impossible if <2 active rects
        layers_lt2 = sum(1 for c in active_counts if c < 2)
        print(f"layers with <2 active rects (no bridging possible): {layers_lt2}/{num_layers}")
        _debug_plot_rects(rects, num_layers, num_qubits)

    # Allocate classical bits only if measuring.
    m = int(np.ceil(float(measure_frac) * num_qubits)) if measure_frac > 0 else 0
    qc = QuantumCircuit(num_qubits, m) if m > 0 else QuantumCircuit(num_qubits)

    # Fill layers
    for t in range(num_layers):
        active_rects = rects_by_layer[t]
        non_idle_active_rects = [ridx for ridx in active_rects if rects[ridx].roi != "idle"]

        for ridx in active_rects:
            r = rects[ridx]
            qs = r.qubits()
            t_local = t - r.t0
            _fill_roi_layer(qc, r.roi, qs, t_local, (r.t1 - r.t0), rect_rngs[ridx], p2_default, dist_thr_long, rect_keys[ridx])
            _sprinkle_block_noise(qc, qs, rect_rngs[ridx], noise_1q_prob, noise_2q_prob)

        # Bridges
        if len(non_idle_active_rects) >= 2:
            p_bridge = p_bdry if boundary_layer[t] else p_int
            if rng.rand() < p_bridge:
                # Add 1–2 bridges (fixed behavior; not a config parameter).
                for _ in range(2):
                    ra, rb = rng.choice(non_idle_active_rects, size=2, replace=False)
                    qa = int(rng.choice(rects[ra].qubits()))
                    qb = int(rng.choice(rects[rb].qubits()))
                    apply_random_2q_gate(qc, qa, qb, rng)
                    if rng.rand() < 0.5:
                        break
        # Bridge-burst ROI: burst edges must involve a burst rectangle, and never idle
        burst_rects = []
        other_non_idle_rects = []

        for rr_idx in non_idle_active_rects:
            rr = rects[rr_idx]
            if rr.roi == "bridge_burst":
                tl = t - rr.t0
                w = rr.t1 - rr.t0
                if tl < 3 or tl >= (w - 3):
                    burst_rects.append(rr_idx)
            else:
                other_non_idle_rects.append(rr_idx)

        if burst_rects and other_non_idle_rects:
            burst_edges = min(3 * len(burst_rects), 10)
            for _ in range(burst_edges):
                ra = int(rng.choice(burst_rects))
                rb = int(rng.choice(other_non_idle_rects))
                qa = int(rng.choice(rects[ra].qubits()))
                qb = int(rng.choice(rects[rb].qubits()))
                apply_random_2q_gate(qc, qa, qb, rng)


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
