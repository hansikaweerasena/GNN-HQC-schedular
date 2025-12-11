"""
qubit_graph.py

Purpose:
    Build a qubit interaction MultiGraph from a CircuitRepresentation:
      - Nodes = qubits
      - Edges = 2-qubit gate interactions (multi-edges allowed)
    Attach node/edge features used later by the GNN.

Inputs:
    - CircuitRepresentation (has .layers, .num_qubits, etc.)

Outputs:
    - NetworkX MultiGraph with:
        node attributes: first_layer, gate_count
        edge attributes: gate_type, interaction_layers, interaction_count
"""

from typing import List
import networkx as nx

from .circuit_representation import CircuitRepresentation, CircuitLayer

