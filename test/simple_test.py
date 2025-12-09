"""
simple_test.py

Purpose:
    End-to-end sanity check for Phase 1:
      1) Generate a QFT circuit
      2) Visualize the circuit diagram
      3) Build CircuitRepresentation (DAG + layers)
      4) Visualize qubit activity per layer
      5) Run Jaccard-based temporal segmentation
      6) Visualize segment boundaries on the activity heatmap

Inputs:
    - None (circuit parameters are hardcoded for now: 5-qubit QFT).

Outputs:
    - Printed summary of the circuit and segments.
    - Matplotlib figures: circuit diagram, layer activity, and segmentation.

Usage:
    python simple_test.py
"""

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.circuit_generation import generate_qft_circuit
from src.circuit_representation import CircuitRepresentation
from src.circuit_segmentation import segment_circuit, analyze_segmentation
from src.circuit_visualization import (
    visualize_circuit,
    visualize_layer_activity,
    visualize_segmentation,
)


def main():
    # Generate circuit
    qc = generate_qft_circuit(5)
    print("✓ Circuit generated")
    print(f"  Depth: {qc.depth()}, Gates: {qc.size()}")

    # Visualize circuit
    visualize_circuit(qc)

    # Representation
    rep = CircuitRepresentation(qc)
    print("\n✓ Circuit representation")
    print(rep.summary())

    # Visualize layer activity
    activity = visualize_layer_activity(rep.layers, rep.num_qubits)

    # Segment
    threshold = 0.8
    segments, seg_ids = segment_circuit(rep.layers, threshold=threshold)
    stats = analyze_segmentation(segments, rep.num_qubits)

    print(f"\n✓ Segmentation (threshold={threshold})")
    print(f"  Num segments: {stats['num_segments']}")
    for seg in segments:
        print(seg)

    # Visualize segmentation
    visualize_segmentation(activity, segments, title_suffix=f"(threshold={threshold})")


if __name__ == "__main__":
    main()
