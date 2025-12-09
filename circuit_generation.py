"""Generate simple quantum circuits."""
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
