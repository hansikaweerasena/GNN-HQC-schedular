"""
circuit_visualization.py

Purpose:
    Provide simple plotting utilities to:
      - visualize a QuantumCircuit
      - visualize per-layer qubit activity as a heatmap
      - overlay temporal segment boundaries on the heatmap.

Main functions:
    - visualize_circuit(circuit):
        Input:  QuantumCircuit
        Effect: Draws the circuit using Qiskit's matplotlib drawer.

    - visualize_layer_activity(layers, num_qubits) -> np.ndarray:
        Inputs:
          * layers: List[CircuitLayer]
          * num_qubits: int
        Output:
          * activity: 2D numpy array of shape (num_qubits, num_layers)
            where activity[q, l] = 1 if qubit q is active in layer l.
        Effect:
          * Displays a heatmap (qubits vs layers).

    - visualize_segmentation(activity, segments, title_suffix=""):
        Inputs:
          * activity: 2D array from visualize_layer_activity
          * segments: List[Segment] (from circuit_segmentation)
          * title_suffix: optional string for the plot title
        Effect:
          * Plots activity heatmap and draws red vertical lines at
            segment boundaries between layers.

Usage:
    activity = visualize_layer_activity(rep.layers, rep.num_qubits)
    visualize_segmentation(activity, segments, "(threshold=0.3)")
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List

from qiskit import QuantumCircuit

from .circuit_representation import CircuitLayer


def visualize_circuit(circuit: QuantumCircuit):
    circuit.draw(output="mpl", fold=-1)
    plt.show()


def visualize_layer_activity(layers: List[CircuitLayer], num_qubits: int):
    num_layers = len(layers)
    activity = np.zeros((num_qubits, num_layers), dtype=int)
    for l_idx, layer in enumerate(layers):
        for q in layer.active_qubits:
            activity[q, l_idx] = 1

    plt.figure(figsize=(8, 4))
    plt.imshow(activity, aspect="auto", cmap="Blues", interpolation="nearest")
    plt.colorbar(label="Active (1) / Inactive (0)")
    plt.xlabel("Layer")
    plt.ylabel("Qubit")
    plt.yticks(range(num_qubits))
    plt.xticks(range(num_layers))
    plt.title("Layer Activity (Qubit vs Layer)")
    plt.tight_layout()
    plt.show()
    return activity


def visualize_segmentation(activity: np.ndarray, segments, title_suffix=""):
    num_qubits, num_layers = activity.shape

    plt.figure(figsize=(8, 4))
    plt.imshow(activity, aspect="auto", cmap="Blues", interpolation="nearest")
    plt.colorbar(label="Active (1) / Inactive (0)")
    plt.xlabel("Layer")
    plt.ylabel("Qubit")
    plt.yticks(range(num_qubits))
    plt.xticks(range(num_layers))
    plt.title(f"Layer Activity with Segments {title_suffix}")

    for seg in segments[:-1]:
        _, end_layer = seg.layer_range
        plt.axvline(x=end_layer + 0.5, color="red", linestyle="--", linewidth=2)

    plt.tight_layout()
    plt.show()
