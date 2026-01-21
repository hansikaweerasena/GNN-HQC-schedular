"""
circuit_generation.py

Purpose:
    Provide simple utilities to generate benchmark quantum circuits
    (starting with QFT) using Qiskit.

Main functions:
    - generate_qft_circuit(num_qubits) -> QuantumCircuit

Inputs:
    - num_qubits (int): number of logical qubits in the circuit.

Outputs:
    - QuantumCircuit object that can be passed to other modules for
      DAG extraction, layering, segmentation, etc.

Usage:
    from src.circuit_generation import generate_qft_circuit
    qc = generate_qft_circuit(5)
"""

import numpy as np
from qiskit import QuantumCircuit

def generate_qft_circuit(num_qubits: int) -> QuantumCircuit:
    """Generate Quantum Fourier Transform circuit."""
    qc = QuantumCircuit(num_qubits, name="qft")
    
    for j in range(num_qubits):
        qc.h(j)
        for k in range(j + 1, num_qubits):
            angle = 2 * np.pi / (2 ** (k - j + 1))
            qc.cp(angle, k, j)
    
    for i in range(num_qubits // 2):
        qc.swap(i, num_qubits - i - 1)
    
    return qc

ONE_QUBIT_GATES = ['h', 'x', 'y', 'z', 's', 't']
TWO_QUBIT_GATES = ['cx', 'cz', 'swap']

def apply_random_1q_gate(qc: QuantumCircuit, q: int):
    gate_type = np.random.choice(ONE_QUBIT_GATES)
    if gate_type == 'h':
        qc.h(q)
    elif gate_type == 'x':
        qc.x(q)
    elif gate_type == 'y':
        qc.y(q)
    elif gate_type == 'z':
        qc.z(q)
    elif gate_type == 's':
        qc.s(q)
    else:  # 't'
        qc.t(q)


def apply_random_2q_gate(qc: QuantumCircuit, q1: int, q2: int):
    gate_type = np.random.choice(TWO_QUBIT_GATES)
    if gate_type == 'cx':
        qc.cx(q1, q2)
    elif gate_type == 'cz':
        qc.cz(q1, q2)
    else:  # 'swap'
        qc.swap(q1, q2)


def generate_random_circuit_custom(n_qubits=10, depth=20, gate_density=0.3, seed=None, two_qubit_ratio=0.5,):
    """
    Generate a custom random circuit with controllable sparsity.
    - This creates more realistic circuits where not all qubits are active every layer.
    
    Args:
        n_qubits: Number of qubits
        depth: Number of layers
        gate_density: Probability of gate on each qubit per layer (0.0 to 1.0)
        seed: Random seed
        
    Returns:
        QuantumCircuit
    """
    if seed is not None:
        np.random.seed(seed)
    
    qc = QuantumCircuit(n_qubits)
    
    for layer_idx in range(depth):
        # Randomly select qubits to be active this layer
        active_qubits = []
        for q in range(n_qubits):
            if np.random.random() < gate_density:
                active_qubits.append(q)
        
        # Add gates to active qubits
                np.random.shuffle(active_qubits)
        i = 0
        while i < len(active_qubits):
            if i + 1 < len(active_qubits):
                q1, q2 = active_qubits[i], active_qubits[i + 1]
                use_two_qubit = np.random.rand() < two_qubit_ratio

                if use_two_qubit:
                    apply_random_2q_gate(qc, q1, q2)
                else:
                    apply_random_1q_gate(qc, q1)
                    apply_random_1q_gate(qc, q2)
                i += 2
            else:
                q = active_qubits[i]
                apply_random_1q_gate(qc, q)
                i += 1
         # Add barrier to force layer separation
        if active_qubits:
            qc.barrier(*active_qubits)
            
    return qc


def make_1q_burst_segment(num_qubits: int,
                          depth: int,
                          rng: np.random.RandomState) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    for _ in range(depth):
        for q in range(num_qubits):
            if rng.rand() < 0.8:      # 80% chance use 1q gate
                apply_random_1q_gate(qc, q)
            # else: idle this layer on this qubit
        qc.barrier(*range(num_qubits))
    return qc


def make_2q_burst_segment(num_qubits: int,
                          depth: int,
                          rng: np.random.RandomState) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    for _ in range(depth):
        # Optional 1q rotations (small cost, just activity)
        for q in range(num_qubits):
            if rng.rand() < 0.3:
                apply_random_1q_gate(qc, q)

        # Random pairing for 2q gates
        qubits = list(range(num_qubits))
        rng.shuffle(qubits)
        i = 0
        while i + 1 < len(qubits):
            q1, q2 = qubits[i], qubits[i+1]
            if rng.rand() < 0.8:      # 80% chance to use 2q gate on this pair
                apply_random_2q_gate(qc, q1, q2)
            i += 2
        qc.barrier(*range(num_qubits))
    return qc



def make_idle_heavy_segment(num_qubits: int,
                            depth: int,
                            rng: np.random.RandomState) -> QuantumCircuit:
    """
    Idle-heavy segment:
      - Most qubits idle most of the time.
      - Each layer activates only a small contiguous block of qubits (size 2–3),
        others remain idle.
    """
    qc = QuantumCircuit(num_qubits)

    for _ in range(depth):
        # choose a small active window: size 2 or 3
        window_size = rng.choice([2, 3])
        if num_qubits <= window_size:
            start = 0
        else:
            start = rng.randint(0, num_qubits - window_size + 1)
        active_qubits = list(range(start, start + window_size))

        # with high prob, apply 1q gates on this small block
        for q in active_qubits:
            if rng.rand() < 0.7:
                apply_random_1q_gate(qc, q)

        # occasionally add a 2q gate inside the window
        if len(active_qubits) >= 2 and rng.rand() < 0.4:
            q1, q2 = rng.choice(active_qubits, size=2, replace=False)
            apply_random_2q_gate(qc, int(q1), int(q2))

        # everyone shares the same time step
        qc.barrier(*range(num_qubits))

    return qc


def make_idle_heavy_segment(num_qubits: int,
                            depth: int,
                            rng: np.random.RandomState) -> QuantumCircuit:
    """
    Idle-heavy segment:
      - Only a fixed subset of qubits (e.g. first half) ever receive gates.
      - Other qubits are completely idle over the whole segment.
      - Within the active subset, each layer activates only a small contiguous
        window (size 2–3), so even active qubits are idle most of the time.
    """
    qc = QuantumCircuit(num_qubits)

    # Choose which qubits are EVER allowed to be active
    active_band_size = max(2, num_qubits // 2)       # e.g. 5 if num_qubits=10
    active_pool = list(range(active_band_size))      # e.g. qubits 0..4
    idle_forever = list(range(active_band_size, num_qubits))  # e.g. 5..9

    for _ in range(depth):
        # Pick a small window inside the active_pool: size 2 or 3
        window_size = rng.choice([2, 3])
        if len(active_pool) <= window_size:
            start_idx = 0
        else:
            start_idx = rng.randint(0, len(active_pool) - window_size + 1)
        window = active_pool[start_idx:start_idx + window_size]

        # Apply 1q gates with high prob on this small window
        for q in window:
            if rng.rand() < 0.7:
                apply_random_1q_gate(qc, q)

        # Occasionally add a 2q gate inside the window
        if len(window) >= 2 and rng.rand() < 0.4:
            q1, q2 = rng.choice(window, size=2, replace=False)
            apply_random_2q_gate(qc, int(q1), int(q2))

        # All qubits share the same time step; idle_forever stay idle
        qc.barrier(*range(num_qubits))

    return qc


def make_streaming_segment(num_qubits: int,
                           depth: int,
                           rng: np.random.RandomState) -> QuantumCircuit:
    """
    Streaming segment:
      - Qubits become active gradually over time.
      - At time t, only qubits 0..k_t may get gates, where k_t grows with t.
      - Encourages the scheduler to move qubits to 'good exec' tech
        when they start doing work, while others remain idle.
    """
    qc = QuantumCircuit(num_qubits)

    for t in range(depth):
        # how many qubits are 'eligible' to be active at this time step
        max_active_index = min(num_qubits, 1 + t // 2)  # grow every 2 layers
        eligible = list(range(max_active_index))

        # choose a subset of eligible qubits to actually use this layer
        # (so even eligible qubits are not always active)
        if eligible:
            layer_active = []
            for q in eligible:
                if rng.rand() < 0.5:   # 50% chance this eligible qubit is active
                    layer_active.append(q)

            # ensure at least one active qubit occasionally
            if not layer_active and rng.rand() < 0.3:
                layer_active = [rng.choice(eligible)]

            # apply gates on layer_active
            #  - mostly 1q, sometimes 2q among active ones
            for q in layer_active:
                if rng.rand() < 0.8:
                    apply_random_1q_gate(qc, q)

            if len(layer_active) >= 2 and rng.rand() < 0.5:
                q1, q2 = rng.choice(layer_active, size=2, replace=False)
                apply_random_2q_gate(qc, int(q1), int(q2))

        qc.barrier(*range(num_qubits))

    return qc


SEGMENT_GENERATORS = {
    "1q_burst": make_1q_burst_segment,
    "2q_burst": make_2q_burst_segment,
    "idle": make_idle_heavy_segment,
    "streaming": make_streaming_segment,
}


def generate_roi_composed_circuit(
    num_qubits: int,
    num_segments: int = 5,
    segment_depth: int = 10,
    seed = None,
) -> QuantumCircuit:
    """
    Build a circuit as a random composition of ROI segments.

    Segments are chosen uniformly from:
      - '1q_burst', '2q_burst', 'idle', 'streaming'

    Args:
        num_qubits: total qubits.
        num_segments: how many ROI segments to concatenate.
        segment_depth: per‑segment depth parameter passed to each generator.
        seed: random seed for reproducibility.

    Returns:
        QuantumCircuit with num_qubits, composed in time from ROI segments.
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random

    qc = QuantumCircuit(num_qubits)
    pattern_names = list(SEGMENT_GENERATORS.keys())

    for _ in range(num_segments):
        seg_name = rng.choice(pattern_names)
        seg_fn = SEGMENT_GENERATORS[seg_name]

        # each segment gets its own small depth, same rng
        seg_circ = seg_fn(num_qubits=num_qubits,
                          depth=segment_depth,
                          rng=rng)

        # append in time: just compose the circuits
        qc.compose(seg_circ, inplace=True)

    return qc


