"""
test_thresholds_visual.py

Purpose:
    For a fixed circuit, show how different Jaccard thresholds
    segment the same layers differently, using heatmaps with
    segment boundaries drawn.

Thresholds tested:
    [0.1, 0.3, 0.5, 0.7, 0.9]
"""

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt

from src.circuit_generation import generate_qft_circuit
from src.circuit_representation import CircuitRepresentation
from src.circuit_segmentation import segment_circuit
from utils.circuit_visualization import visualize_layer_activity, visualize_segmentation


def main():
    # 1) Generate fixed circuit
    qc = generate_qft_circuit(5)
    rep = CircuitRepresentation(qc)
    print("Circuit summary:")
    print(f"  Qubits     : {rep.num_qubits}")
    print(f"  Num layers : {len(rep.layers)}")
    print(f"  Depth      : {rep.circuit.depth()}")

    # 2) Compute activity matrix once
    activity = visualize_layer_activity(rep.layers, rep.num_qubits)

    # 3) Try different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

    for th in thresholds:
        print(f"\n=== Threshold = {th:.1f} ===")
        segments, _ = segment_circuit(rep.layers, threshold=th)
        print(f"  Num segments: {len(segments)}")
        for seg in segments:
            s, e = seg.layer_range
            print(f"    Segment {seg.segment_idx}: layers [{s}..{e}], "
                  f"active={sorted(seg.active_qubits)}")

        # 4) Visualize segmentation for this threshold
        visualize_segmentation(activity, segments, title_suffix=f"(threshold={th:.1f})")


if __name__ == "__main__":
    main()
