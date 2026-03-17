#!/usr/bin/env python3
from __future__ import annotations

"""
Phase 2.2 / Phase 2A non-local classification harness.

What this script does
---------------------
1. Reuses the Phase 1B dense motifs and adds a few bridge / cross-community
   positive motifs for non-local analysis.
2. Builds a longer-horizon flat bidirectional window graph (default radius=6)
   for structural non-local classification.
3. Computes, for every 2Q gate in every motif:
      - local bridge classification (Granovetter common-neighbor test)
      - community size guard
      - pair-reuse guard
   to identify structurally non-local edges.
4. Produces case-wise logs, CSV tables, and visualizations to support
   inspection and validation.

Non-local classification used here (3-stage local-bridge pipeline)
-------------------------------------------------------------------
Let G_nl(l) be the longer-horizon effective graph around layer l.
For target edge (u,v):

  Stage 1  local bridge test (Granovetter, 1973)
           Check whether u and v share a common neighbor in G_nl.
           If yes → edge is locally embedded (triangle exists) → local.
           If no  → edge is a local bridge → candidate non-local.
           Equivalent to L_detour >= 3. Cost: O(deg(u) + deg(v)) per edge.

  Stage 2  community size guard
           remove (u,v) from G_nl; let C_u, C_v be the resulting
           connected components containing u and v respectively.
           pass if |C_u| >= delta_community AND |C_v| >= delta_community
           (kills false positives from small/pendant neighborhoods)

  Stage 3  pair-reuse guard
           count how many times (u,v) appears in a small local window
           (±pair_reuse_radius layers around current layer).
           fail if count >= pair_reuse_threshold
           (kills false positives from repeated nearest-neighbor edges
            that happen to sit at community boundaries)

  I_nl = 1[ stage1 AND stage2 AND stage3 ]

  Burden for classified non-local edges: L_detour - 1  (SWAP-count proxy)
  L_detour comes from BFS in stage 2 (free with component computation).

This script is focused on the classifier metrics and their threshold behavior.
The final non-local score magnitude (detour-based Gamma_nonlocal) is left for a
later phase.
"""

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Import Phase 1B helpers / motifs so the cases stay consistent with user code.
# -----------------------------------------------------------------------------

P1B_PATH = Path("scripts/phase1B_dense_cases.py")
if not P1B_PATH.exists():
    raise FileNotFoundError(f"Expected Phase 1B script at {P1B_PATH}")

spec = importlib.util.spec_from_file_location("phase1b_dense", P1B_PATH)
phase1b = importlib.util.module_from_spec(spec)
sys.modules["phase1b_dense"] = phase1b
assert spec.loader is not None
spec.loader.exec_module(phase1b)

LayerSpec = phase1b.LayerSpec
MotifSpec = phase1b.MotifSpec
build_quantum_circuit = phase1b.build_quantum_circuit
layer_edge_counts = phase1b.layer_edge_counts
plot_circuit_timeline = phase1b.plot_circuit_timeline
plot_pair_layer_heatmap = phase1b.plot_pair_layer_heatmap
plot_graphs_by_layer = phase1b.plot_graphs_by_layer
write_text = phase1b.write_text
_sorted_pair = phase1b._sorted_pair
_ensure_dir = phase1b._ensure_dir


# -----------------------------------------------------------------------------
# Config / motif extensions
# -----------------------------------------------------------------------------


@dataclass
class NonlocalCaseConfig:
    window_radius_nl: int = 6
    window_weights_nl: Optional[List[float]] = None  # flat by default
    window_normalize: bool = False

    # Stage 1: local bridge test has no tunable threshold.
    # An edge is a local bridge iff endpoints share no common neighbor.

    # Stage 2 community size guard: both components after edge removal must be >= delta
    delta_community: int = 3

    # Stage 3 pair-reuse guard: if edge appears >= pair_reuse_threshold times
    # in a small local window (±pair_reuse_radius layers), it is a temporally
    # stable local interaction, not a long-range bridge.
    pair_reuse_radius: int = window_radius_nl
    pair_reuse_threshold: int = 2

    # Technology connectivity capacity (e.g. 3 for heavy-hex).
    # detour_cap is derived as kappa + 1.
    kappa: int = 3

    eps: float = 1e-12

    outdir: str = "calibration/phase2A_nonlocal_cases"
    annotate_heatmaps: bool = True
    save_qiskit_text: bool = True

    @property
    def detour_cap(self) -> int:
        return self.kappa + 1


class NonlocalMotifFactory:
    def __init__(self) -> None:
        self._p1 = phase1b.MotifFactory()

    def all_names(self) -> List[str]:
        return self._p1.all_names() + [
            "true_bridge",
            "cross_community",
            "shortcut_bridge",
            # --- scaled motifs (20 qubits, 20 layers) ---
            "scaled_chain",
            "scaled_brickwork",
            "scaled_bridge",
            "scaled_cross_community",
            "scaled_shortcut",
            "multi_community",
            "long_range_probe",
            "temporal_bridge",
            "false_bridge_accumulation",
            "heavy_hex_local",
            # --- many-long-range motifs ---
            "dense_cross_community",
            "qft_like",
            "scattered_bridges",
            "random_overlay",
        ]

    def build(self, name: str) -> MotifSpec:
        name_l = name.strip().lower()
        if name_l in {n.lower() for n in self._p1.all_names()}:
            return self._p1.build(name_l)
        builder = {
            "true_bridge":               self.true_bridge,
            "cross_community":           self.cross_community,
            "shortcut_bridge":           self.shortcut_bridge,
            "scaled_chain":              self.scaled_chain,
            "scaled_brickwork":          self.scaled_brickwork,
            "scaled_bridge":             self.scaled_bridge,
            "scaled_cross_community":    self.scaled_cross_community,
            "scaled_shortcut":           self.scaled_shortcut,
            "multi_community":           self.multi_community,
            "long_range_probe":          self.long_range_probe,
            "temporal_bridge":           self.temporal_bridge,
            "false_bridge_accumulation": self.false_bridge_accumulation,
            "heavy_hex_local":           self.heavy_hex_local,
            "dense_cross_community":     self.dense_cross_community,
            "qft_like":                  self.qft_like,
            "scattered_bridges":         self.scattered_bridges,
            "random_overlay":            self.random_overlay,
        }
        if name_l not in builder:
            raise ValueError(f"Unknown motif: {name}")
        return builder[name_l]()

    # ----- helpers for generating dense cluster layer types -----

    @staticmethod
    def _dense_cluster_layer_types(a: int, n: int) -> List[List[Tuple[int, int]]]:
        """
        Generate 6 non-overlapping layer types for a dense cluster of n qubits
        starting at qubit a.  Pattern: 2 NN layers + 4 skip-one layers.
        Creates triangle (a+i, a+i+1, a+i+2) for every valid i, ensuring that
        every NN edge has at least one common neighbor.
        """
        even_nn = [(a + i, a + i + 1) for i in range(0, n - 1, 2)]
        odd_nn  = [(a + i, a + i + 1) for i in range(1, n - 1, 2)]
        skip_0  = [(a + i, a + i + 2) for i in range(0, n - 2, 4)]
        skip_1  = [(a + i, a + i + 2) for i in range(2, n - 2, 4)]
        skip_2  = [(a + i, a + i + 2) for i in range(1, n - 2, 4)]
        skip_3  = [(a + i, a + i + 2) for i in range(3, n - 2, 4)]
        return [even_nn, odd_nn, skip_0, skip_1, skip_2, skip_3]

    @staticmethod
    def _cycle_layers(
        layer_types: List[List[Tuple[int, int]]],
        total: int,
        prefix: str,
    ) -> List[LayerSpec]:
        """Cycle through layer_types to produce `total` LayerSpec objects."""
        k = len(layer_types)
        return [
            LayerSpec(twoq=list(layer_types[i % k]), label=f"{prefix}_{i}")
            for i in range(total)
        ]

    @staticmethod
    def _merge_layer_types(
        *clusters: List[List[Tuple[int, int]]],
    ) -> List[List[Tuple[int, int]]]:
        """
        Merge layer types from multiple clusters by pairing them round-robin.
        Each cluster contributes its layer types in sequence;
        the merged result has max(len(c)) layer types, with
        shorter clusters wrapping.
        """
        max_len = max(len(c) for c in clusters)
        merged: List[List[Tuple[int, int]]] = []
        for i in range(max_len):
            edges: List[Tuple[int, int]] = []
            for c in clusters:
                edges.extend(c[i % len(c)])
            merged.append(edges)
        return merged

    # ----- original small motifs (kept unchanged) -----

    def true_bridge(self) -> MotifSpec:
        layers = [
            LayerSpec(twoq=[(0, 1), (1, 2), (0, 2)], label="tb_left_cluster_0"),
            LayerSpec(twoq=[(4, 5), (5, 6), (4, 6)], label="tb_right_cluster_1"),
            LayerSpec(twoq=[(0, 1), (4, 5)], label="tb_local_repeat_2"),
            LayerSpec(twoq=[(2, 4)], label="tb_target_bridge_3"),
            LayerSpec(twoq=[(1, 2), (5, 6)], label="tb_local_repeat_4"),
            LayerSpec(twoq=[(0, 2), (4, 6)], label="tb_local_repeat_5"),
        ]
        return MotifSpec(
            name="true_bridge",
            num_qubits=7,
            layers=layers,
            target_layer=3,
            target_pair=(2, 4),
            notes="Positive non-local motif: strict bridge between two local communities.",
        )

    def cross_community(self) -> MotifSpec:
        layers = [
            LayerSpec(twoq=[(0, 1), (1, 2), (0, 2)], label="cc_left_cluster_0"),
            LayerSpec(twoq=[(4, 5), (5, 6), (4, 6)], label="cc_right_cluster_1"),
            LayerSpec(twoq=[(1, 4)], label="cc_alt_cross_2"),
            LayerSpec(twoq=[(2, 5)], label="cc_target_cross_3"),
            LayerSpec(twoq=[(0, 2), (4, 6)], label="cc_local_repeat_4"),
            LayerSpec(twoq=[(1, 2), (5, 6)], label="cc_local_repeat_5"),
        ]
        return MotifSpec(
            name="cross_community",
            num_qubits=7,
            layers=layers,
            target_layer=3,
            target_pair=(2, 5),
            notes="Positive non-local motif: cross-community edge with moderate alternate connectivity.",
        )

    def shortcut_bridge(self) -> MotifSpec:
        layers = [
            LayerSpec(twoq=[(0, 1), (2, 3), (4, 5)], label="sb_even_0"),
            LayerSpec(twoq=[(1, 2), (3, 4)], label="sb_odd_1"),
            LayerSpec(twoq=[(2, 5)], label="sb_target_shortcut_2"),
            LayerSpec(twoq=[(1, 2), (3, 4)], label="sb_odd_3"),
            LayerSpec(twoq=[(0, 1), (2, 3), (4, 5)], label="sb_even_4"),
        ]
        return MotifSpec(
            name="shortcut_bridge",
            num_qubits=6,
            layers=layers,
            target_layer=2,
            target_pair=(2, 5),
            notes="Positive non-local motif: shortcut-like edge across an otherwise local chain backbone.",
        )

    # ----- scaled motifs (~20 qubits, ~20 layers) -----

    def scaled_chain(self) -> MotifSpec:
        """
        20-qubit chain with skip-one edges for triangles.
        Every consecutive triple (i, i+1, i+2) forms a triangle.
        All edges are local — expect 0 non-local classifications.
        """
        ltypes = self._dense_cluster_layer_types(0, 20)
        layers = self._cycle_layers(ltypes, 20, "sc")
        return MotifSpec(
            name="scaled_chain",
            num_qubits=20,
            layers=layers,
            target_layer=1,
            target_pair=(9, 10),
            notes="Negative scaled motif: 20-qubit chain with skip-one triangles. Expect 0 non-local.",
        )

    def scaled_brickwork(self) -> MotifSpec:
        """
        20 qubits as 2 rows of 10 with horizontal, vertical, and diagonal edges.
        Each interior qubit has 4-6 neighbors — very dense local connectivity.
        Expect 0 non-local.

        Layout:
          Row 0:  0  1  2  3  4  5  6  7  8  9
          Row 1: 10 11 12 13 14 15 16 17 18 19
        """
        # Horizontal NN within each row
        h_even = [(i, i + 1) for i in range(0, 9, 2)] + \
                 [(10 + i, 10 + i + 1) for i in range(0, 9, 2)]
        h_odd  = [(i, i + 1) for i in range(1, 9, 2)] + \
                 [(10 + i, 10 + i + 1) for i in range(1, 9, 2)]
        # Vertical (same column)
        v_even = [(i, i + 10) for i in range(0, 10, 2)]
        v_odd  = [(i, i + 10) for i in range(1, 10, 2)]
        # Forward diagonal: (row0,col_c) → (row1,col_{c+1})
        d_fwd_even = [(i, i + 11) for i in range(0, 9, 2)]
        d_fwd_odd  = [(i, i + 11) for i in range(1, 9, 2)]
        # Backward diagonal: (row0,col_c) → (row1,col_{c-1})
        d_bwd_odd  = [(i, i + 9) for i in range(1, 10, 2)]
        d_bwd_even = [(i, i + 9) for i in range(2, 10, 2)]

        ltypes = [h_even, h_odd, v_even, v_odd,
                  d_fwd_even, d_fwd_odd, d_bwd_odd, d_bwd_even]
        layers = self._cycle_layers(ltypes, 20, "bw")
        return MotifSpec(
            name="scaled_brickwork",
            num_qubits=20,
            layers=layers,
            target_layer=0,
            target_pair=(5, 6),
            notes="Negative scaled motif: 2x10 grid with diagonals. Expect 0 non-local.",
        )

    def scaled_bridge(self) -> MotifSpec:
        """
        Two dense 10-qubit clusters (0-9 and 10-19) connected by one bridge.
        Cluster interiors have full triangle coverage via skip-one edges.
        Bridge (9,10) appears once at layer 10.
        Expect 1 non-local (the bridge).
        """
        ca = self._dense_cluster_layer_types(0, 10)
        cb = self._dense_cluster_layer_types(10, 10)
        merged = self._merge_layer_types(ca, cb)
        layers = self._cycle_layers(merged, 20, "sbr")
        # Insert bridge edge at layer 10
        layers[10] = LayerSpec(
            twoq=layers[10].twoq + [(9, 10)],
            label="sbr_bridge_10",
        )
        return MotifSpec(
            name="scaled_bridge",
            num_qubits=20,
            layers=layers,
            target_layer=10,
            target_pair=(9, 10),
            notes="Positive scaled motif: two 10-node clusters, one bridge. Expect 1 non-local.",
        )

    def scaled_cross_community(self) -> MotifSpec:
        """
        Two dense 10-qubit clusters (0-9 and 10-19) connected by 4 cross-links
        at different layers. Each cross-link connects different qubit pairs
        across the clusters.
        Expect 3-4 non-local (the cross-community edges).
        """
        ca = self._dense_cluster_layer_types(0, 10)
        cb = self._dense_cluster_layer_types(10, 10)
        merged = self._merge_layer_types(ca, cb)
        layers = self._cycle_layers(merged, 20, "scc")
        # Insert 4 cross-community edges at spread-out layers
        for lyr, edge, lbl in [
            (5,  (4, 15), "scc_cross_5"),
            (9,  (8, 11), "scc_cross_9"),
            (10, (9, 10), "scc_cross_10"),
            (15, (3, 16), "scc_cross_15"),
        ]:
            layers[lyr] = LayerSpec(
                twoq=layers[lyr].twoq + [edge],
                label=lbl,
            )
        return MotifSpec(
            name="scaled_cross_community",
            num_qubits=20,
            layers=layers,
            target_layer=10,
            target_pair=(9, 10),
            notes="Positive scaled motif: two 10-node clusters, 4 cross-links. Expect 3-4 non-local.",
        )

    def scaled_shortcut(self) -> MotifSpec:
        """
        20-qubit chain with skip-one triangles plus one long-range shortcut
        from qubit 3 to qubit 17 at layer 10. The shortcut spans most of the
        chain and has no common neighbor with either endpoint.
        Expect 1 non-local (the shortcut).
        """
        ltypes = self._dense_cluster_layer_types(0, 20)
        layers = self._cycle_layers(ltypes, 20, "ss")
        # Insert shortcut
        layers[10] = LayerSpec(
            twoq=layers[10].twoq + [(3, 17)],
            label="ss_shortcut_10",
        )
        return MotifSpec(
            name="scaled_shortcut",
            num_qubits=20,
            layers=layers,
            target_layer=10,
            target_pair=(3, 17),
            notes="Positive scaled motif: 20-qubit chain + one long shortcut (3,17). Expect 1 non-local.",
        )

    def multi_community(self) -> MotifSpec:
        """
        Three 7-qubit dense clusters (0-6, 7-13, 14-20) connected by two
        bridge edges: (6,7) at layer 7 and (13,14) at layer 14.
        Tests scaling to 3+ communities.
        Expect 2 non-local (both bridges).
        """
        ca = self._dense_cluster_layer_types(0, 7)
        cb = self._dense_cluster_layer_types(7, 7)
        cc = self._dense_cluster_layer_types(14, 7)
        merged = self._merge_layer_types(ca, cb, cc)
        layers = self._cycle_layers(merged, 21, "mc")
        # Insert two bridge edges
        layers[7] = LayerSpec(
            twoq=layers[7].twoq + [(6, 7)],
            label="mc_bridge_ab_7",
        )
        layers[14] = LayerSpec(
            twoq=layers[14].twoq + [(13, 14)],
            label="mc_bridge_bc_14",
        )
        return MotifSpec(
            name="multi_community",
            num_qubits=21,
            layers=layers,
            target_layer=7,
            target_pair=(6, 7),
            notes="Positive scaled motif: three 7-node clusters, 2 bridges. Expect 2 non-local.",
        )

    def long_range_probe(self) -> MotifSpec:
        """
        Dense cluster (qubits 0-14) with a small separate chain (15-19).
        One probe edge (7,18) connects the cluster interior to the chain
        interior at layer 10.
        Expect 1 non-local (the probe).
        """
        c_main = self._dense_cluster_layer_types(0, 15)
        c_side = self._dense_cluster_layer_types(15, 5)
        merged = self._merge_layer_types(c_main, c_side)
        layers = self._cycle_layers(merged, 20, "lrp")
        # Insert probe edge
        layers[10] = LayerSpec(
            twoq=layers[10].twoq + [(7, 18)],
            label="lrp_probe_10",
        )
        return MotifSpec(
            name="long_range_probe",
            num_qubits=20,
            layers=layers,
            target_layer=10,
            target_pair=(7, 18),
            notes="Positive scaled motif: 15-node dense cluster + 5-node chain, one probe. Expect 1 non-local.",
        )

    def temporal_bridge(self) -> MotifSpec:
        """
        Two dense 10-qubit clusters (0-9 and 10-19).
        Cluster A active layers 0-14, Cluster B active layers 5-19.
        Bridge (9,10) appears once at layer 10.
        The bridge is temporally isolated within ongoing cluster activity.
        Expect 1 non-local.
        """
        ca = self._dense_cluster_layer_types(0, 10)
        cb = self._dense_cluster_layer_types(10, 10)
        layers: List[LayerSpec] = []
        for i in range(20):
            edges: List[Tuple[int, int]] = []
            # Cluster A active in layers 0-14
            if i <= 14:
                edges.extend(ca[i % len(ca)])
            # Cluster B active in layers 5-19
            if i >= 5:
                edges.extend(cb[(i - 5) % len(cb)])
            lbl = f"tb_{i}"
            # Insert bridge at layer 10
            if i == 10:
                edges.append((9, 10))
                lbl = "tb_bridge_10"
            layers.append(LayerSpec(twoq=edges, label=lbl))
        return MotifSpec(
            name="temporal_bridge",
            num_qubits=20,
            layers=layers,
            target_layer=10,
            target_pair=(9, 10),
            notes="Positive scaled motif: two clusters with overlapping activity, one bridge. Expect 1 non-local.",
        )

    def false_bridge_accumulation(self) -> MotifSpec:
        """
        Temporal aliasing stress test. Qubits 0-19 in two overlapping groups:
          Group A (qubits 0-12): active in layers 0-12
          Group B (qubits 8-19): active in layers 8-19
        Overlap zone: qubits 8-12 participate in BOTH phases.
        Under window accumulation, overlap qubits appear to bridge two
        communities. But they have common neighbors from both phases,
        so no edge should be classified as non-local.
        Expect 0 non-local.
        """
        ca = self._dense_cluster_layer_types(0, 13)   # qubits 0-12
        cb = self._dense_cluster_layer_types(8, 12)    # qubits 8-19
        layers: List[LayerSpec] = []
        for i in range(20):
            edges: List[Tuple[int, int]] = []
            # Group A active layers 0-12
            if i <= 12:
                edges.extend(ca[i % len(ca)])
            # Group B active layers 8-19
            if i >= 8:
                edges.extend(cb[(i - 8) % len(cb)])
            layers.append(LayerSpec(twoq=edges, label=f"fba_{i}"))
        return MotifSpec(
            name="false_bridge_accumulation",
            num_qubits=20,
            layers=layers,
            target_layer=10,
            target_pair=(10, 11),
            notes="Negative scaled motif: temporal aliasing stress test. Overlap qubits have common neighbors from both phases. Expect 0 non-local.",
        )

    def heavy_hex_local(self) -> MotifSpec:
        """
        20 qubits wired in a heavy-hex-like pattern (degree 2-3) with one
        added long-range edge (2,17). The heavy-hex backbone has triangles
        from the hexagonal structure. Only the added edge should be non-local.

        Heavy-hex backbone (simplified 20-qubit section):
          Row A (data):    0   2   4   6   8
          Row B (bridge): 1   3   5   7   9
          Row C (data):   10  12  14  16  18
          Row D (bridge): 11  13  15  17  19

        Edges follow heavy-hex pattern:
          Vertical:  (0,1),(1,2),(2,3),(3,4),...  alternating data-bridge
          Horizontal data-bridge connections.
        Expect 1 non-local (the added long-range edge).
        """
        # Heavy-hex connections: data qubits connect to bridge qubits.
        # Data row A: 0,2,4,6,8  Bridge row B: 1,3,5,7,9
        # Data row C: 10,12,14,16,18  Bridge row D: 11,13,15,17,19
        # Vertical: data-bridge pairs
        vert_ab = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
        vert_ba = [(1, 2), (3, 4), (5, 6), (7, 8)]  # bridge connects adjacent data
        vert_cd = [(10, 11), (12, 13), (14, 15), (16, 17), (18, 19)]
        vert_dc = [(11, 12), (13, 14), (15, 16), (17, 18)]
        # Cross-row: bridge qubits connect across rows
        cross_bc = [(1, 10), (3, 12), (5, 14), (7, 16), (9, 18)]
        cross_bc2 = [(1, 12), (3, 14), (5, 16), (7, 18)]
        # Data-data skip within rows (for triangles)
        skip_a = [(0, 2), (4, 6)]
        skip_a2 = [(2, 4), (6, 8)]
        skip_c = [(10, 12), (14, 16)]
        skip_c2 = [(12, 14), (16, 18)]

        ltypes = [
            vert_ab + vert_cd,
            vert_ba + vert_dc,
            cross_bc,
            cross_bc2,
            skip_a + skip_c,
            skip_a2 + skip_c2,
        ]
        layers = self._cycle_layers(ltypes, 20, "hh")
        # Insert long-range edge at layer 10
        layers[10] = LayerSpec(
            twoq=layers[10].twoq + [(2, 17)],
            label="hh_longrange_10",
        )
        return MotifSpec(
            name="heavy_hex_local",
            num_qubits=20,
            layers=layers,
            target_layer=10,
            target_pair=(2, 17),
            notes="Positive scaled motif: heavy-hex-like local + one long-range edge (2,17). Expect 1 non-local.",
        )

    # ----- many-long-range motifs -----

    def dense_cross_community(self) -> MotifSpec:
        """
        Two 10-qubit dense clusters (0-9 and 10-19) connected by 8
        cross-community edges spread across layers.
        Tests: does the classifier hold up when there are MANY cross-links?
        At 8 cross-links between 10-node clusters, the community boundary
        is heavily bridged but the clusters still have far more internal
        edges than cross-edges, so community structure persists.
        Expect 6-8 non-local (the cross-community edges).
        """
        ca = self._dense_cluster_layer_types(0, 10)
        cb = self._dense_cluster_layer_types(10, 10)
        merged = self._merge_layer_types(ca, cb)
        layers = self._cycle_layers(merged, 20, "dcc")
        # Insert 8 cross-community edges at different layers, different qubit pairs
        cross_edges = [
            (2,  (0, 19), "dcc_cross_2"),
            (4,  (4, 15), "dcc_cross_4"),
            (6,  (8, 11), "dcc_cross_6"),
            (8,  (2, 17), "dcc_cross_8"),
            (10, (9, 10), "dcc_cross_10"),
            (12, (5, 14), "dcc_cross_12"),
            (14, (3, 16), "dcc_cross_14"),
            (16, (7, 12), "dcc_cross_16"),
        ]
        for lyr, edge, lbl in cross_edges:
            layers[lyr] = LayerSpec(
                twoq=layers[lyr].twoq + [edge],
                label=lbl,
            )
        return MotifSpec(
            name="dense_cross_community",
            num_qubits=20,
            layers=layers,
            target_layer=10,
            target_pair=(9, 10),
            notes="Positive scaled motif: two 10-node clusters, 8 cross-links. "
                  "Tests many-NL regime. Expect 8 non-local.",
        )

    def qft_like(self) -> MotifSpec:
        """
        20 qubits. First 10 layers are local brickwork within two 10-qubit
        clusters. Layers 10-19 inject QFT-like long-range gates: qubit i
        interacts with qubit i+10, i+7, etc — distant pairs that span
        across the two clusters.
        Tests: many non-local edges in the second half, zero in the first.
        Expect 8 non-local (the QFT-like long-range gates).
        """
        ca = self._dense_cluster_layer_types(0, 10)
        cb = self._dense_cluster_layer_types(10, 10)
        merged = self._merge_layer_types(ca, cb)

        # First 10 layers: pure local
        layers = self._cycle_layers(merged, 20, "qft")

        # Layers 10-19: inject one long-range gate per layer
        qft_edges = [
            (10, (0, 10)),
            (11, (1, 18)),
            (12, (2, 15)),
            (13, (3, 17)),
            (14, (4, 19)),
            (15, (5, 16)),
            (16, (6, 13)),
            (17, (8, 11)),
        ]
        for lyr, edge in qft_edges:
            layers[lyr] = LayerSpec(
                twoq=layers[lyr].twoq + [edge],
                label=f"qft_lr_{lyr}",
            )
        return MotifSpec(
            name="qft_like",
            num_qubits=20,
            layers=layers,
            target_layer=10,
            target_pair=(0, 10),
            notes="Positive scaled motif: local brickwork + QFT-like long-range "
                  "overlay in second half. Expect 8 non-local.",
        )

    def scattered_bridges(self) -> MotifSpec:
        """
        One dense 8-qubit core (0-7). Four satellite clusters of 3 qubits:
          Sat A: (8,9,10), Sat B: (11,12,13),
          Sat C: (14,15,16), Sat D: (17,18,19)
        Each satellite has internal edges (repeated, with triangles) and one
        bridge to the core. Bridges connect at different layers.
        Satellite size = 3 satisfies delta_community = 3.
        Tests: 4 simultaneous strict bridges radiating from one community.
        Expect 4 non-local (the bridge edges).
        """
        # Core cluster + 4 satellite clusters, all with triangles
        c_core = self._dense_cluster_layer_types(0, 8)
        c_sat_a = self._dense_cluster_layer_types(8, 3)
        c_sat_b = self._dense_cluster_layer_types(11, 3)
        c_sat_c = self._dense_cluster_layer_types(14, 3)
        c_sat_d = self._dense_cluster_layer_types(17, 3)

        merged = self._merge_layer_types(c_core, c_sat_a, c_sat_b, c_sat_c, c_sat_d)
        layers = self._cycle_layers(merged, 20, "sb")

        # Insert 4 bridge edges at spread-out layers (one per satellite)
        bridges = [
            (3,  (2, 8),  "sb_bridge_satA_3"),    # core qubit 2 → satellite A
            (7,  (5, 11), "sb_bridge_satB_7"),    # core qubit 5 → satellite B
            (11, (7, 14), "sb_bridge_satC_11"),   # core qubit 7 → satellite C
            (15, (3, 17), "sb_bridge_satD_15"),   # core qubit 3 → satellite D
        ]
        for lyr, edge, lbl in bridges:
            layers[lyr] = LayerSpec(
                twoq=layers[lyr].twoq + [edge],
                label=lbl,
            )
        return MotifSpec(
            name="scattered_bridges",
            num_qubits=20,
            layers=layers,
            target_layer=3,
            target_pair=(2, 8),
            notes="Positive scaled motif: dense 8-node core + 4 satellite clusters "
                  "(3 qubits each), each with one bridge. Expect 4 non-local.",
        )

    def random_overlay(self) -> MotifSpec:
        """
        20-qubit chain with skip-one triangles (local backbone), plus 8
        random long-range edges spanning ≥5 qubits injected at different
        layers. Mimics a messy real circuit with no clean community
        structure — just a local backbone with random long-range
        interactions scattered throughout.
        Expect 8 non-local (the long-range overlay edges).
        """
        ltypes = self._dense_cluster_layer_types(0, 20)
        layers = self._cycle_layers(ltypes, 20, "ro")

        # 8 long-range edges, each spanning ≥5 qubits, at different layers
        overlay = [
            (1,  (0, 14)),
            (3,  (2, 18)),
            (5,  (4, 12)),
            (7,  (1, 16)),
            (9,  (6, 19)),
            (11, (3, 15)),
            (14, (5, 17)),
            (17, (7, 13)),
        ]
        for lyr, edge in overlay:
            layers[lyr] = LayerSpec(
                twoq=layers[lyr].twoq + [edge],
                label=f"ro_lr_{lyr}",
            )
        return MotifSpec(
            name="random_overlay",
            num_qubits=20,
            layers=layers,
            target_layer=1,
            target_pair=(0, 14),
            notes="Positive scaled motif: 20-qubit local chain + 8 random long-range "
                  "edges (span ≥5). Tests messy real-circuit regime. Expect 8 non-local.",
        )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _flat_window_weights(radius: int) -> List[float]:
    return [1.0] * (2 * int(radius) + 1)


def build_window_effective_graphs(
    edge_counts_per_layer: Sequence[Dict[Tuple[int, int], float]],
    radius: int,
    weights: Optional[Sequence[float]],
    normalize: bool,
) -> List[Dict[Tuple[int, int], float]]:
    if weights is None:
        weights = _flat_window_weights(radius)
    if len(weights) != 2 * radius + 1:
        raise ValueError("window weights must have length 2*radius+1")

    out: List[Dict[Tuple[int, int], float]] = []
    n_layers = len(edge_counts_per_layer)
    for s in range(n_layers):
        acc: Dict[Tuple[int, int], float] = defaultdict(float)
        total_w = 0.0
        for off in range(-radius, radius + 1):
            j = s + off
            if j < 0 or j >= n_layers:
                continue
            w = float(weights[off + radius])
            total_w += w
            for pair, val in edge_counts_per_layer[j].items():
                acc[pair] += w * float(val)
        if normalize and total_w > 0.0:
            acc = defaultdict(float, {p: v / total_w for p, v in acc.items()})
        out.append(dict(acc))
    return out


def effective_to_graph(eff: Dict[Tuple[int, int], float], num_qubits: int, weighted: bool = True) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(num_qubits))
    for (u, v), w in eff.items():
        if weighted:
            g.add_edge(u, v, weight=float(w))
        else:
            g.add_edge(u, v)
    return g


def has_common_neighbor(eff: Dict[Tuple[int, int], float], pair: Tuple[int, int]) -> bool:
    """
    Stage 1: Local bridge test (Granovetter, 1973).

    Returns True if u and v share at least one common neighbor in the
    effective graph — meaning the edge is locally embedded (triangle exists)
    and is NOT a local bridge.

    Returns False if no common neighbor exists — meaning the edge IS a local
    bridge (L_detour >= 3 guaranteed).

    Cost: O(deg(u) + deg(v)) per edge — much cheaper than betweenness O(VE).
    """
    u, v = pair
    # Build adjacency sets from the effective graph
    neighbors_u: set = set()
    neighbors_v: set = set()
    for (a, b) in eff:
        if a == u and b != v:
            neighbors_u.add(b)
        elif b == u and a != v:
            neighbors_u.add(a)
        if a == v and b != u:
            neighbors_v.add(b)
        elif b == v and a != u:
            neighbors_v.add(a)
    # Common neighbor exists iff intersection is non-empty
    return len(neighbors_u & neighbors_v) > 0


def _bfs_component_and_distance(
    g: nx.Graph,
    src: int,
    dst: int,
    exclude_edge: Tuple[int, int],
) -> Tuple[float, int]:
    """
    BFS in g with exclude_edge removed.
    Returns (shortest_path_distance, component_size_of_src).
    distance = math.inf if dst unreachable.
    component_size counts all nodes reachable from src (not through exclude_edge).
    """
    eu, ev = exclude_edge

    def _skip(a: int, b: int) -> bool:
        return (a == eu and b == ev) or (a == ev and b == eu)

    visited: Dict[int, int] = {src: 0}
    queue = defaultdict(list)
    queue[0].append(src)
    dist_to_dst = math.inf
    frontier = [src]
    bfs_queue = [(src, 0)]
    visited_set = {src}
    from collections import deque as _deque
    q = _deque([(src, 0)])
    while q:
        node, d = q.popleft()
        for nbr in g.neighbors(node):
            if _skip(node, nbr):
                continue
            if nbr not in visited_set:
                visited_set.add(nbr)
                q.append((nbr, d + 1))
                if nbr == dst:
                    dist_to_dst = d + 1
    return dist_to_dst, len(visited_set)


def detour_metrics(
    eff: Dict[Tuple[int, int], float],
    num_qubits: int,
    pair: Tuple[int, int],
) -> Tuple[float, int, int]:
    """
    Stage 2+3: compute L_detour, |C_u|, |C_v| for pair (u,v).

    Returns:
        l_detour        : shortest path in G_nl \\ {(u,v)}, math.inf if bridge
        component_size_u: # nodes reachable from u after removing (u,v)
        component_size_v: # nodes reachable from v after removing (u,v)
    """
    g = effective_to_graph(eff, num_qubits, weighted=False)
    u, v = pair
    if not g.has_edge(u, v):
        return math.inf, 1, 1
    l_detour, cu = _bfs_component_and_distance(g, u, v, exclude_edge=(u, v))
    _, cv          = _bfs_component_and_distance(g, v, u, exclude_edge=(u, v))
    return l_detour, cu, cv


def pair_reuse_count(
    edge_counts_per_layer: Sequence[Dict[Tuple[int, int], float]],
    layer_idx: int,
    pair: Tuple[int, int],
    radius: int,
) -> int:
    """
    Stage 3 pair-reuse guard helper.

    Count how many layers within ±radius of layer_idx contain pair (u,v).
    A count >= threshold means the edge is a temporally stable local
    interaction, not a one-shot long-range bridge.
    """
    n_layers = len(edge_counts_per_layer)
    count = 0
    for off in range(-radius, radius + 1):
        j = layer_idx + off
        if j < 0 or j >= n_layers:
            continue
        if pair in edge_counts_per_layer[j]:
            count += 1
    return count


def compute_nonlocal_gate_rows(
    motif: MotifSpec,
    edge_counts_per_layer: Sequence[Dict[Tuple[int, int], float]],
    effective_graphs: Sequence[Dict[Tuple[int, int], float]],
    cfg: NonlocalCaseConfig,
) -> pd.DataFrame:
    """
    Three-stage local-bridge classifier per gate:

    Stage 1  local bridge test         no common neighbor in G_nl
             (O(deg) per edge; kills all locally-embedded edges)
    Stage 2  community size guard      min(|C_u|, |C_v|) >= delta_community
             (kills false positives from small/pendant neighborhoods)
    Stage 3  pair-reuse guard          reuse_count < pair_reuse_threshold
             (kills false positives from repeated nearest-neighbor edges)

    A gate is non-local only if ALL THREE stages pass.
    The stage at which it fails is recorded for inspection.
    L_detour is computed in stage 2 (free with BFS) and used for burden scoring.
    """
    detour_cap = cfg.detour_cap  # kappa + 1

    rows: List[Dict[str, Any]] = []
    for s, layer in enumerate(motif.layers):
        eff = effective_graphs[s]

        ordered_pairs = sorted(
            (_sorted_pair(u, v) for (u, v) in layer.twoq),
            key=lambda p: (p[0], p[1]),
        )
        for gate_idx, pair in enumerate(ordered_pairs, start=1):
            u, v = pair

            # --- Stage 1: local bridge test (common-neighbor check) ---
            has_cn = has_common_neighbor(eff, pair)
            is_local_bridge = not has_cn  # no common neighbor → local bridge

            # --- Stages 2-3 (only if stage 1 passes — saves BFS cost) ---
            l_detour: float = float("nan")
            cu: int = 0
            cv: int = 0
            reuse: int = 0
            pass_community = False
            pass_pair_reuse = False

            if is_local_bridge:
                # Stage 2: community size guard (BFS gives L_detour for free)
                l_raw, cu, cv = detour_metrics(eff, motif.num_qubits, pair)
                l_detour = float(detour_cap) if math.isinf(l_raw) else float(l_raw)
                pass_community = (cu >= cfg.delta_community) and (cv >= cfg.delta_community)

                if pass_community:
                    # Stage 3: pair-reuse guard
                    reuse = pair_reuse_count(
                        edge_counts_per_layer, s, pair, cfg.pair_reuse_radius,
                    )
                    pass_pair_reuse = reuse < cfg.pair_reuse_threshold

            is_nonlocal = is_local_bridge and pass_community and pass_pair_reuse

            # Which stage killed it (for diagnosis)
            if is_nonlocal:
                fail_stage = "none"
            elif not is_local_bridge:
                fail_stage = "common_neighbor"
            elif not pass_community:
                fail_stage = "community"
            else:
                fail_stage = "pair_reuse"

            is_target = bool(
                motif.target_layer == s
                and motif.target_pair is not None
                and _sorted_pair(*motif.target_pair) == pair
            )
            rows.append({
                "motif":          motif.name,
                "layer":          int(s),
                "layer_label":    layer.label,
                "gate_id":        int(gate_idx),
                "gate_label":     f"L{s}_G{gate_idx}:{u}-{v}",
                "u":              int(u),
                "v":              int(v),
                "pair":           f"({u},{v})",
                # metrics
                "is_local_bridge":bool(is_local_bridge),
                "C_u":            int(cu),
                "C_v":            int(cv),
                "reuse_count":    int(reuse),
                "L_detour":       float(l_detour),
                # thresholds used
                "delta_community":int(cfg.delta_community),
                "pair_reuse_threshold": int(cfg.pair_reuse_threshold),
                "kappa":          int(cfg.kappa),
                # outcome
                "pass_community": bool(pass_community),
                "pass_pair_reuse":bool(pass_pair_reuse),
                "fail_stage":     str(fail_stage),
                "I_nonlocal":     int(is_nonlocal),
                "is_target":      bool(is_target),
            })
    return pd.DataFrame(rows)


def summarize_motif(df: pd.DataFrame, motif: MotifSpec, cfg: NonlocalCaseConfig) -> Dict[str, Any]:
    if df.empty:
        return {"motif": motif.name, "num_layers": 0, "num_gates": 0}
    target_df = df[df["is_target"] == True]
    def _tgt(col: str, default: Any) -> Any:
        return target_df[col].iloc[0] if not target_df.empty else default
    fail_counts = df["fail_stage"].value_counts().to_dict()
    return {
        "motif":                  motif.name,
        "window_radius_nl":       int(cfg.window_radius_nl),
        "delta_community":        int(cfg.delta_community),
        "pair_reuse_threshold":   int(cfg.pair_reuse_threshold),
        "kappa":                  int(cfg.kappa),
        "num_layers":             int(len(motif.layers)),
        "num_gates":              int(len(df)),
        "num_classified_nonlocal":int(df["I_nonlocal"].sum()),
        # target gate breakdown
        "target_is_local_bridge": bool(_tgt("is_local_bridge", False)),
        "target_C_u":             int(_tgt("C_u", -1)),
        "target_C_v":             int(_tgt("C_v", -1)),
        "target_reuse_count":     int(_tgt("reuse_count", 0)),
        "target_L_detour":        float(_tgt("L_detour", float("nan"))),
        "target_pass_community":  bool(_tgt("pass_community", False)),
        "target_pass_pair_reuse": bool(_tgt("pass_pair_reuse", False)),
        "target_I_nonlocal":      int(_tgt("I_nonlocal", -1)),
        "target_fail_stage":      str(_tgt("fail_stage", "n/a")),
        # population stats
        "fail_common_neighbor":   int(fail_counts.get("common_neighbor", 0)),
        "fail_community":         int(fail_counts.get("community", 0)),
        "fail_pair_reuse":        int(fail_counts.get("pair_reuse", 0)),
        "notes":                  motif.notes,
    }


def concise_gate_log(df: pd.DataFrame) -> str:
    cols = ["layer", "gate_label", "is_local_bridge", "C_u", "C_v", "reuse_count", "L_detour",
            "pass_community", "pass_pair_reuse", "fail_stage",
            "I_nonlocal", "is_target"]
    out = df.loc[:, cols].copy()
    out["L_detour"] = out["L_detour"].map(lambda x: f"{float(x):.1f}" if not math.isnan(float(x)) else "—")
    return out.to_string(index=False)


def concise_summary_log(summary: Dict[str, Any]) -> str:
    keys = [
        "motif", "window_radius_nl",
        "delta_community", "pair_reuse_threshold", "kappa",
        "num_layers", "num_gates", "num_classified_nonlocal",
        "target_is_local_bridge", "target_C_u", "target_C_v",
        "target_reuse_count", "target_L_detour",
        "target_pass_community", "target_pass_pair_reuse",
        "target_I_nonlocal", "target_fail_stage",
        "fail_common_neighbor", "fail_community", "fail_pair_reuse",
    ]
    lines = []
    for k in keys:
        v = summary.get(k)
        lines.append(f"- {k}: {v:.6f}" if isinstance(v, float) else f"- {k}: {v}")
    lines.append(f"- notes: {summary.get('notes', '')}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def plot_metrics_heatmap(df: pd.DataFrame, outpath: Path, annotate: bool = True) -> None:
    if df.empty:
        return
    pivot_cols = ["is_local_bridge", "C_u", "C_v", "reuse_count", "L_detour", "I_nonlocal"]
    plot_df = df[["gate_label"] + pivot_cols].copy().set_index("gate_label")
    arr = plot_df.to_numpy(dtype=float)
    fig_h = max(2.8, 0.42 * len(plot_df.index))
    fig, ax = plt.subplots(figsize=(10.0, fig_h))
    im = ax.imshow(arr, aspect="auto")
    ax.set_xticks(range(len(pivot_cols)))
    ax.set_xticklabels(pivot_cols, rotation=20, ha="right")
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index)
    ax.set_title("Non-local classification metrics by gate (local-bridge pipeline)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.ax.set_ylabel("value", rotation=90)
    # All columns are integer-like except L_detour(4)
    int_cols = {0, 1, 2, 3, 5}
    if annotate:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                txt = "—" if math.isnan(v) else (f"{v:.0f}" if j in int_cols else f"{v:.1f}")
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_gate_metric_bars(df: pd.DataFrame, motif: MotifSpec, outpath: Path) -> None:
    if df.empty:
        return
    labels      = df["gate_label"].tolist()
    x           = np.arange(len(labels))
    is_lb       = df["is_local_bridge"].to_numpy(dtype=float)
    l_det       = df["L_detour"].to_numpy(dtype=float)
    cu          = df["C_u"].to_numpy(dtype=float)
    cv          = df["C_v"].to_numpy(dtype=float)
    reuse       = df["reuse_count"].to_numpy(dtype=float)
    target_mask = df["is_target"].to_numpy(dtype=bool)
    fail_stage  = df["fail_stage"].tolist()

    fig, axes = plt.subplots(4, 1, figsize=(max(9.0, 0.55 * len(labels)), 12.0), sharex=True)

    # Stage 1: local bridge test (boolean)
    ax = axes[0]
    ax.bar(x, is_lb, color=["tab:red" if lb else "tab:blue" for lb in is_lb])
    ax.set_ylabel("is_local_bridge")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["has common nbr", "local bridge"])
    ax.set_title(f"Stage 1 — local bridge test (no common neighbor): {motif.name}")

    # Stage 2: community sizes
    ax = axes[1]
    w = 0.38
    ax.bar(x - w/2, cu, width=w, label="|C_u|")
    ax.bar(x + w/2, cv, width=w, label="|C_v|")
    delta = df["delta_community"].iloc[0]
    ax.axhline(delta, linestyle="--", linewidth=1, label=f"delta_community={delta}")
    ax.set_ylabel("component size")
    ax.set_title("Stage 2 — community size guard")
    ax.legend(loc="upper right", fontsize=8)

    # Stage 3: pair-reuse guard
    ax = axes[2]
    reuse_thresh = df["pair_reuse_threshold"].iloc[0]
    ax.bar(x, reuse, color=["tab:purple" if f == "pair_reuse" else
                             "tab:orange" if f == "none" else "tab:gray"
                             for f in fail_stage])
    ax.axhline(reuse_thresh, linestyle="--", linewidth=1, label=f"pair_reuse_threshold={reuse_thresh}")
    ax.set_ylabel("reuse count")
    ax.set_title("Stage 3 — pair-reuse guard (fail if ≥ threshold)")
    ax.legend(loc="upper right", fontsize=8)

    # Info: L_detour (span) for classified gates
    ax = axes[3]
    colors = ["tab:red" if f == "none" else "tab:gray" for f in fail_stage]
    ax.bar(x, np.nan_to_num(l_det, nan=0.0), color=colors)
    ax.set_ylabel("L_detour (span)")
    ax.set_title("Span of classified non-local edges (burden = L_detour − 1)")

    for ax in axes:
        for i, is_t in enumerate(target_mask):
            if is_t:
                ax.axvspan(i - 0.55, i + 0.55, color="gold", alpha=0.18)
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(labels, rotation=60, ha="right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_summary_bar(summary_df: pd.DataFrame, outpath: Path, value_col: str, title: str) -> None:
    if summary_df.empty or value_col not in summary_df.columns:
        return
    sdf = summary_df.sort_values(value_col, ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(max(8.0, 0.75 * len(sdf)), 4.6))
    xpos = np.arange(len(sdf))
    ax.bar(xpos, sdf[value_col])
    ax.set_xticks(xpos)
    ax.set_xticklabels(sdf["motif"], rotation=45, ha="right")
    ax.set_ylabel(value_col)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_fail_stage_breakdown(summary_df: pd.DataFrame, outpath: Path) -> None:
    """Stacked bar: how many gates fell out at each stage, per motif."""
    if summary_df.empty:
        return
    cols = ["fail_common_neighbor", "fail_community", "fail_pair_reuse", "num_classified_nonlocal"]
    present = [c for c in cols if c in summary_df.columns]
    if not present:
        return
    sdf = summary_df.set_index("motif")[present]
    fig, ax = plt.subplots(figsize=(max(8.0, 0.75 * len(sdf)), 4.6))
    sdf.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("gate count")
    ax.set_title("Gate funnel: gates eliminated at each stage")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Run harness
# -----------------------------------------------------------------------------


def run_one_case(motif: MotifSpec, cfg: NonlocalCaseConfig, base_outdir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    case_dir = base_outdir / motif.name
    _ensure_dir(case_dir)

    if cfg.save_qiskit_text:
        qc = build_quantum_circuit(motif)
        if qc is not None:
            try:
                text_repr = str(qc.draw(output="text"))
            except Exception:
                text_repr = repr(qc)
            write_text(case_dir / "circuit_text.txt", text_repr)

    plot_circuit_timeline(motif, case_dir / "timeline.png")
    plot_pair_layer_heatmap(motif.layers, case_dir / "pair_layer_heatmap.png")

    edge_counts = layer_edge_counts(motif.layers)
    weights = cfg.window_weights_nl or _flat_window_weights(cfg.window_radius_nl)
    effective_graphs = build_window_effective_graphs(edge_counts, cfg.window_radius_nl, weights, normalize=cfg.window_normalize)
    current_graphs = edge_counts
    plot_graphs_by_layer(motif, current_graphs, effective_graphs, "nonlocal_window", case_dir / "graphs_by_layer.png")

    df = compute_nonlocal_gate_rows(motif, edge_counts, effective_graphs, cfg)
    df.to_csv(case_dir / "gate_metrics.csv", index=False)

    summary = summarize_motif(df, motif, cfg)
    pd.DataFrame([summary]).to_csv(case_dir / "summary.csv", index=False)

    plot_metrics_heatmap(df, case_dir / "metrics_heatmap.png", annotate=cfg.annotate_heatmaps)
    plot_gate_metric_bars(df, motif, case_dir / "gate_metric_bars.png")

    write_text(case_dir / "gate_log.txt", concise_gate_log(df))
    write_text(case_dir / "summary_log.txt", concise_summary_log(summary))
    write_text(case_dir / "notes.txt", motif.notes)
    return df, summary


def parse_args() -> argparse.Namespace:
    # All numeric defaults are pulled from the dataclass — single source of truth.
    _defaults = NonlocalCaseConfig()
    p = argparse.ArgumentParser(description="Phase 2A non-local classification harness (local-bridge pipeline)")
    p.add_argument("--motif", type=str, default="all", help="Motif name or 'all'")
    p.add_argument("--window-radius-nl", type=int, default=_defaults.window_radius_nl, help="Non-local flat window radius")
    p.add_argument("--delta-community", type=int, default=_defaults.delta_community, help="Stage 2: minimum component size after edge removal")
    p.add_argument("--pair-reuse-radius", type=int, default=_defaults.pair_reuse_radius, help="Stage 3: local window radius for pair-reuse count")
    p.add_argument("--pair-reuse-threshold", type=int, default=_defaults.pair_reuse_threshold, help="Stage 3: fail if pair appears >= this many times in local window")
    p.add_argument("--kappa", type=int, default=_defaults.kappa, help="Technology connectivity capacity (detour_cap = kappa + 1)")
    p.add_argument("--outdir", type=str, default=_defaults.outdir, help="Output directory")
    p.add_argument("--no-annotate", action="store_true", help="Disable heatmap annotations")
    p.add_argument("--no-qiskit-text", action="store_true", help="Skip circuit text dump")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = NonlocalCaseConfig(
        window_radius_nl=int(args.window_radius_nl),
        delta_community=int(args.delta_community),
        pair_reuse_radius=int(args.pair_reuse_radius),
        pair_reuse_threshold=int(args.pair_reuse_threshold),
        kappa=int(args.kappa),
        annotate_heatmaps=not bool(args.no_annotate),
        save_qiskit_text=not bool(args.no_qiskit_text),
        outdir=str(args.outdir),
    )
    base_outdir = Path(cfg.outdir)
    _ensure_dir(base_outdir)

    factory = NonlocalMotifFactory()
    motif_names = factory.all_names() if str(args.motif).lower() == "all" else [args.motif]

    all_gate_dfs: List[pd.DataFrame] = []
    summaries: List[Dict[str, Any]] = []
    for name in motif_names:
        motif = factory.build(name)
        df, summary = run_one_case(motif, cfg, base_outdir)
        all_gate_dfs.append(df)
        summaries.append(summary)

    full_df = pd.concat(all_gate_dfs, ignore_index=True) if all_gate_dfs else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)
    full_df.to_csv(base_outdir / "all_gate_metrics.csv", index=False)
    summary_df.to_csv(base_outdir / "all_summaries.csv", index=False)

    plot_summary_bar(summary_df, base_outdir / "summary_target_detour_bar.png", "target_L_detour", "Target L_detour by motif")
    plot_summary_bar(summary_df, base_outdir / "summary_positive_count_bar.png", "num_classified_nonlocal", "Number of classified non-local gates by motif")
    plot_fail_stage_breakdown(summary_df, base_outdir / "fail_stage_breakdown.png")

    manifest = {
        "config": {
            "window_radius_nl": cfg.window_radius_nl,
            "delta_community": cfg.delta_community,
            "pair_reuse_radius": cfg.pair_reuse_radius,
            "pair_reuse_threshold": cfg.pair_reuse_threshold,
            "kappa": cfg.kappa,
            "detour_cap": cfg.detour_cap,
        },
        "motifs": motif_names,
        "pipeline": {
            "stage1": "local bridge test (no common neighbor)",
            "stage2": f"min(|C_u|, |C_v|) >= {cfg.delta_community}",
            "stage3": f"pair_reuse_count < {cfg.pair_reuse_threshold} (radius={cfg.pair_reuse_radius})",
        },
        "outputs": {
            "all_gate_metrics_csv": str(base_outdir / "all_gate_metrics.csv"),
            "all_summaries_csv":    str(base_outdir / "all_summaries.csv"),
            "fail_stage_breakdown":  str(base_outdir / "fail_stage_breakdown.png"),
        },
    }
    write_text(base_outdir / "manifest.json", json.dumps(manifest, indent=2))
    print(summary_df[[
        "motif",
        "target_is_local_bridge",
        "target_C_u",
        "target_C_v",
        "target_reuse_count",
        "target_L_detour",
        "target_I_nonlocal",
        "target_fail_stage",
        "num_classified_nonlocal",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
