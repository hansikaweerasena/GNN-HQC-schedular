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
