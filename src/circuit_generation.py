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