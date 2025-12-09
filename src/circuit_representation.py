"""
circuit_representation.py

Purpose:
    Convert a Qiskit QuantumCircuit into a convenient internal
    representation with:
      - DAG (directed acyclic graph)
      - list of layers (parallel gates per time step)
      - active qubits per layer.

Main classes:
    - CircuitLayer:
        Holds:
          * layer_idx (int)
          * gates: list of (gate_name, (qubit indices,))
          * active_qubits: set of qubit indices active in this layer

    - CircuitRepresentation:
        Wraps a QuantumCircuit and builds:
          * self.dag       : Qiskit DAGCircuit
          * self.layers    : List[CircuitLayer]
          * self.num_qubits: int

Inputs:
    - QuantumCircuit

Outputs:
    - CircuitRepresentation instance with:
        .layers   → structured list of layers
        .summary() → dict with num_qubits, num_layers, total_gates, depth

Usage:
    from src.circuit_representation import CircuitRepresentation
    rep = CircuitRepresentation(qc)
    print(rep.layers[0].active_qubits)
"""


from dataclasses import dataclass, field
from typing import List, Set, Tuple

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag


@dataclass
class CircuitLayer:
    layer_idx: int
    gates: List[Tuple[str, Tuple[int, ...]]] = field(default_factory=list)
    active_qubits: Set[int] = field(default_factory=set)


class CircuitRepresentation:
    """Wraps a QuantumCircuit with DAG + layer info."""

    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit
        self.dag = circuit_to_dag(circuit)
        self.num_qubits = circuit.num_qubits
        self.layers: List[CircuitLayer] = []
        self._extract_layers()

    def _extract_layers(self):
        qubit_index = {qb: i for i, qb in enumerate(self.circuit.qubits)}
        for layer_idx, layer in enumerate(self.dag.layers()):
            subdag = layer["graph"]
            cl = CircuitLayer(layer_idx=layer_idx)

            for node in subdag.op_nodes():
                gate_name = node.op.name
                qargs = tuple(qubit_index[qb] for qb in node.qargs)
                cl.gates.append((gate_name, qargs))
                cl.active_qubits.update(qargs)

            self.layers.append(cl)

    def summary(self):
        return {
            "num_qubits": self.num_qubits,
            "num_layers": len(self.layers),
            "total_gates": sum(len(l.gates) for l in self.layers),
            "depth": self.circuit.depth(),
        }
