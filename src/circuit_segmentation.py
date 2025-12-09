"""
circuit_segmentation.py

Purpose:
    Perform simple temporal segmentation of a circuit based on how
    the set of active qubits changes across layers, using Jaccard
    similarity between consecutive layers.

Key concepts:
    - When the active qubit set changes a lot (Jaccard < threshold),
      we start a new time segment.

Main components:
    - Segment dataclass:
        Holds:
          * segment_idx   (int)
          * layer_range   (start_layer, end_layer)
          * layers        [layer indices in this segment]
          * active_qubits [set of qubits active in this segment]

    - jaccard_similarity(a, b):
        Computes |A∩B| / |A∪B| for two sets.

    - segment_circuit(layers, threshold):
        Input:
          * layers: List[CircuitLayer] from CircuitRepresentation
          * threshold (float): Jaccard threshold for boundary detection
        Output:
          * segments: List[Segment]
          * segment_ids: list of segment index per layer (same length as layers)

    - analyze_segmentation(segments, num_qubits):
        Computes basic statistics (num_segments, avg size, etc.)

Usage:
    from src.circuit_segmentation import segment_circuit, analyze_segmentation
    segments, seg_ids = segment_circuit(rep.layers, threshold=0.3)
    stats = analyze_segmentation(segments, rep.num_qubits)
"""

from dataclasses import dataclass, field
from typing import List, Set, Tuple

import numpy as np

from .circuit_representation import CircuitLayer


@dataclass
class Segment:
    segment_idx: int
    layer_range: Tuple[int, int]
    layers: List[int] = field(default_factory=list)
    active_qubits: Set[int] = field(default_factory=set)


def jaccard_similarity(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def segment_circuit(layers: List[CircuitLayer], threshold: float = 0.3):
    if not layers:
        return [], []

    segments: List[Segment] = []
    segment_ids: List[int] = []

    current_seg = 0
    current_start = 0
    current_layers = []
    prev_active = None

    for l_idx, layer in enumerate(layers):
        act = layer.active_qubits
        if prev_active is not None:
            j = jaccard_similarity(prev_active, act)
            if j < threshold:
                seg = Segment(segment_idx=current_seg,
                              layer_range=(current_start, l_idx - 1),
                              layers=current_layers.copy())
                for li in current_layers:
                    seg.active_qubits.update(layers[li].active_qubits)
                segments.append(seg)
                current_seg += 1
                current_start = l_idx
                current_layers = []

        current_layers.append(l_idx)
        segment_ids.append(current_seg)
        prev_active = act

    if current_layers:
        seg = Segment(segment_idx=current_seg,
                      layer_range=(current_start, len(layers) - 1),
                      layers=current_layers.copy())
        for li in current_layers:
            seg.active_qubits.update(layers[li].active_qubits)
        segments.append(seg)

    return segments, segment_ids


def analyze_segmentation(segments: List[Segment], num_qubits: int) -> dict:
    sizes = [len(seg.layers) for seg in segments]
    active_counts = [len(seg.active_qubits) for seg in segments]
    return {
        "num_segments": len(segments),
        "avg_segment_size": float(np.mean(sizes)),
        "min_segment_size": int(np.min(sizes)),
        "max_segment_size": int(np.max(sizes)),
        "avg_active_qubits": float(np.mean(active_counts)),
        "min_active_qubits": int(np.min(active_counts)),
        "max_active_qubits": int(np.max(active_counts)),
    }
