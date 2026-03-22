"""
qubit_interaction_graph.py

Builds per-layer qubit interaction graphs for the MOSAIC scheduler GNN input.

Design:
  - One graph per circuit layer (no segmentation)
  - Backbone graph at layer t: edge (u,v) exists iff pair has >= 1 interaction
    within the symmetric window [t - W_long, t + W_long]
  - Node features [N, NODE_FEAT_DIM=16]: local activity booleans + windowed
    interaction rates + positional encoding
  - Edge features [E_t, EDGE_FEAT_DIM=5]: current-layer activity flag +
    short/long windowed interaction rates (past and future)

Window sizes are derived from the maximum kappa across all non-all-to-all
technologies — the same scales used by gamma_scoring.py:
    W_short = ceil(max_kappa)
    W_long  = 2 * ceil(max_kappa)
This means window sizes are NOT hyperparameters; they follow from the
cost model's topology parameters.

Node feature layout  (NODE_FEAT_DIM = 16):
  [0]  is_idle_now           no gate of any kind in layer t
  [1]  is_1q_now             has >= 1 one-qubit gate in layer t
  [2]  is_2q_now             participates in >= 1 two-qubit gate in layer t
  [3]  rate_1q_past_short    1Q gates in [t-W_short, t-1] / W_short
  [4]  rate_1q_future_short  1Q gates in [t+1, t+W_short] / W_short
  [5]  rate_1q_past_long     1Q gates in [t-W_long,  t-1] / W_long
  [6]  rate_1q_future_long   1Q gates in [t+1, t+W_long]  / W_long
  [7]  rate_idle_past_short  idle layers in [t-W_short, t-1] / W_short
  [8]  rate_idle_future_short
  [9]  rate_idle_past_long
  [10] rate_idle_future_long
  [11] rate_2q_past_short    2Q participations in [t-W_short, t-1] / W_short
  [12] rate_2q_future_short
  [13] rate_2q_past_long
  [14] rate_2q_future_long
  [15] layer_position        t / (T-1), in [0, 1]

Edge feature layout  (EDGE_FEAT_DIM = 5):
  [0] active_now         1 if pair has a gate in layer t, else 0
  [1] rate_past_short    interactions in [t-W_short, t-1] / W_short
  [2] rate_future_short  interactions in [t+1, t+W_short] / W_short
  [3] rate_past_long     interactions in [t-W_long,  t-1] / W_long
  [4] rate_future_long   interactions in [t+1, t+W_long]  / W_long

Main API:
    build_layer_graph_arrays(circuit, w_short, w_long)
        -> List of T tuples: (x [N,16], edge_index [2,E], edge_attr [E,5])

    compute_window_sizes(tech_configs)
        -> (w_short, w_long)  derived from GammaTechConfig list

    compute_window_sizes_from_config(cost_config_dict)
        -> (w_short, w_long)  derived from raw JSON config dict
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

# Public dimension constants — import these wherever you build models
NODE_FEAT_DIM: int = 16
EDGE_FEAT_DIM: int = 5


# =============================================================================
# Window size helpers
# =============================================================================


def compute_window_sizes(tech_configs) -> Tuple[int, int]:
    """
    Derive W_short and W_long from a list of GammaTechConfig objects.

    Only non-all-to-all technologies contribute (all-to-all have kappa <= 0).
    Falls back to kappa=3 if every technology is all-to-all.

    Returns:
        (w_short, w_long) where w_short = ceil(max_kappa), w_long = 2*w_short
    """
    max_kappa = max(
        (cfg.kappa for cfg in tech_configs if not cfg.is_all_to_all),
        default=3.0,
    )
    w_short = max(1, int(math.ceil(max_kappa)))
    w_long  = 2 * w_short
    return w_short, w_long


def compute_window_sizes_from_config(cost_config: dict) -> Tuple[int, int]:
    """
    Derive W_short and W_long directly from a raw cost_config JSON dict
    (as loaded by load_cost_config), without constructing GammaTechConfig objects.

    Returns:
        (w_short, w_long)
    """
    techs = cost_config.get("techs", [])
    max_kappa = 3.0  # fallback
    found = False
    for tech in techs:
        routing = tech.get("routing", {})
        a2a = bool(routing.get("all_to_all", False))
        kappa = float(routing.get("kappa", 0.0))
        if not a2a and kappa > 0:
            if not found:
                max_kappa = kappa
                found = True
            else:
                max_kappa = max(max_kappa, kappa)
    w_short = max(1, int(math.ceil(max_kappa)))
    w_long  = 2 * w_short
    return w_short, w_long


# =============================================================================
# Main builder
# =============================================================================


def build_layer_graph_arrays(
    circuit,       # CircuitRepresentation
    w_short: int,
    w_long: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build per-layer graph arrays ready for conversion to PyG Data objects.

    For each layer t (0 .. T-1):
      - x          [N, NODE_FEAT_DIM]   node feature matrix
      - edge_index [2, E_t]             backbone edge connectivity (undirected;
                                        both directions included for GATv2)
      - edge_attr  [E_t, EDGE_FEAT_DIM] edge features

    Backbone rule: edge (u,v) is included in layer t iff the pair has at least
    one 2Q interaction anywhere in [t - W_long, t + W_long].

    Layers where no backbone edges are active return empty graphs:
      edge_index shape [2, 0], edge_attr shape [0, 5].

    Args:
        circuit:  CircuitRepresentation with .layers (List[CircuitLayer])
                  and .num_qubits (int)
        w_short:  short window radius = ceil(max_kappa)
        w_long:   long  window radius = 2 * w_short

    Returns:
        List of T tuples [(x_0, ei_0, ea_0), ..., (x_{T-1}, ei_{T-1}, ea_{T-1})]
    """
    T = len(circuit.layers)
    N = circuit.num_qubits

    if T == 0:
        return []

    # ------------------------------------------------------------------
    # Pass 1: collect per-layer per-qubit counts and per-pair interactions
    # ------------------------------------------------------------------
    q_1q  = np.zeros((T, N), dtype=np.float32)
    q_2q  = np.zeros((T, N), dtype=np.float32)
    pair_counts: List[Dict[Tuple[int, int], float]] = [{} for _ in range(T)]

    for t, layer in enumerate(circuit.layers):
        for gate_name, qubits in layer.gates:
            n = len(qubits)
            if n == 1:
                q = int(qubits[0])
                if 0 <= q < N:
                    q_1q[t, q] += 1.0
            elif n == 2:
                u, v = int(qubits[0]), int(qubits[1])
                if 0 <= u < N and 0 <= v < N and u != v:
                    q_2q[t, u] += 1.0
                    q_2q[t, v] += 1.0
                    pair = (min(u, v), max(u, v))
                    pair_counts[t][pair] = pair_counts[t].get(pair, 0.0) + 1.0

    # Idle: qubit has no gate of any kind in layer t
    q_idle = ((q_1q + q_2q) == 0.0).astype(np.float32)  # [T, N]

    # ------------------------------------------------------------------
    # Prefix sums for O(1) windowed node queries
    # prefix_X[t+1, :] = cumulative sum of X[0..t, :]
    # ------------------------------------------------------------------
    prefix_1q   = np.zeros((T + 1, N), dtype=np.float32)
    prefix_2q   = np.zeros((T + 1, N), dtype=np.float32)
    prefix_idle = np.zeros((T + 1, N), dtype=np.float32)
    for t in range(T):
        prefix_1q[t + 1]   = prefix_1q[t]   + q_1q[t]
        prefix_2q[t + 1]   = prefix_2q[t]   + q_2q[t]
        prefix_idle[t + 1] = prefix_idle[t] + q_idle[t]

    def node_range_sum(prefix: np.ndarray, t_lo: int, t_hi: int) -> np.ndarray:
        """
        Windowed sum over prefix[lo..hi], clamped to [0, T-1].
        Returns [N] — raw count (caller divides by actual window size).
        """
        t_lo = max(0, t_lo)
        t_hi = min(T - 1, t_hi)
        if t_lo > t_hi:
            return np.zeros(N, dtype=np.float32)
        return prefix[t_hi + 1] - prefix[t_lo]

    def node_rate(prefix: np.ndarray, t_lo: int, t_hi: int) -> np.ndarray:
        """
        Windowed rate: sum / actual_window_size, clamped to [0, T-1].
        Divides by the number of layers actually in the clamped window,
        not the nominal window size — avoids boundary deflation at the
        start and end of the circuit.
        Returns [N] in [0, 1].
        """
        t_lo_c = max(0, t_lo)
        t_hi_c = min(T - 1, t_hi)
        if t_lo_c > t_hi_c:
            return np.zeros(N, dtype=np.float32)
        actual_w = float(t_hi_c - t_lo_c + 1)
        return (prefix[t_hi_c + 1] - prefix[t_lo_c]) / actual_w

    # ------------------------------------------------------------------
    # Prefix sums for O(1) windowed edge queries
    # ------------------------------------------------------------------
    all_pairs: List[Tuple[int, int]] = sorted(
        {pair for pc in pair_counts for pair in pc}
    )
    P = len(all_pairs)
    pair_to_idx: Dict[Tuple[int, int], int] = {p: i for i, p in enumerate(all_pairs)}

    if P > 0:
        pair_arr = np.zeros((T, P), dtype=np.float32)
        for t in range(T):
            for pair, cnt in pair_counts[t].items():
                pair_arr[t, pair_to_idx[pair]] = float(cnt)

        pair_prefix = np.zeros((T + 1, P), dtype=np.float32)
        for t in range(T):
            pair_prefix[t + 1] = pair_prefix[t] + pair_arr[t]

        def edge_range_sum(t_lo: int, t_hi: int) -> np.ndarray:
            """Sum pair_prefix[lo..hi] inclusive, clamped. Returns [P]."""
            t_lo = max(0, t_lo)
            t_hi = min(T - 1, t_hi)
            if t_lo > t_hi:
                return np.zeros(P, dtype=np.float32)
            return pair_prefix[t_hi + 1] - pair_prefix[t_lo]

        def edge_rate(t_lo: int, t_hi: int) -> np.ndarray:
            """
            Windowed edge rate: sum / actual_window_size, clamped to [0, T-1].
            Divides by actual layers in window to avoid boundary deflation.
            Returns [P] in [0, 1].
            """
            t_lo_c = max(0, t_lo)
            t_hi_c = min(T - 1, t_hi)
            if t_lo_c > t_hi_c:
                return np.zeros(P, dtype=np.float32)
            actual_w = float(t_hi_c - t_lo_c + 1)
            return (pair_prefix[t_hi_c + 1] - pair_prefix[t_lo_c]) / actual_w
    else:
        pair_arr = np.zeros((T, 0), dtype=np.float32)

        def edge_range_sum(t_lo: int, t_hi: int) -> np.ndarray:
            return np.zeros(0, dtype=np.float32)

        def edge_rate(t_lo: int, t_hi: int) -> np.ndarray:
            return np.zeros(0, dtype=np.float32)

    # ------------------------------------------------------------------
    # Pass 2: assemble (x, edge_index, edge_attr) for each layer
    # ------------------------------------------------------------------
    result: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    T_norm = float(max(T - 1, 1))  # normalizer for layer_position

    for t in range(T):

        # ---- Node features [N, 16] ----
        is_idle_now = q_idle[t]
        is_1q_now   = (q_1q[t] > 0.0).astype(np.float32)
        is_2q_now   = (q_2q[t] > 0.0).astype(np.float32)

        # 1Q windowed rates (exclude current layer t)
        r_1q_ps = node_rate(prefix_1q, t - w_short, t - 1)
        r_1q_fs = node_rate(prefix_1q, t + 1, t + w_short)
        r_1q_pl = node_rate(prefix_1q, t - w_long,  t - 1)
        r_1q_fl = node_rate(prefix_1q, t + 1, t + w_long)

        # Idle windowed rates
        r_idle_ps = node_rate(prefix_idle, t - w_short, t - 1)
        r_idle_fs = node_rate(prefix_idle, t + 1, t + w_short)
        r_idle_pl = node_rate(prefix_idle, t - w_long,  t - 1)
        r_idle_fl = node_rate(prefix_idle, t + 1, t + w_long)

        # 2Q windowed rates
        r_2q_ps = node_rate(prefix_2q, t - w_short, t - 1)
        r_2q_fs = node_rate(prefix_2q, t + 1, t + w_short)
        r_2q_pl = node_rate(prefix_2q, t - w_long,  t - 1)
        r_2q_fl = node_rate(prefix_2q, t + 1, t + w_long)

        layer_pos = np.full(N, t / T_norm, dtype=np.float32)

        x = np.stack([
            is_idle_now, is_1q_now, is_2q_now,
            r_1q_ps, r_1q_fs, r_1q_pl, r_1q_fl,
            r_idle_ps, r_idle_fs, r_idle_pl, r_idle_fl,
            r_2q_ps, r_2q_fs, r_2q_pl, r_2q_fl,
            layer_pos,
        ], axis=1)  # [N, 16]

        # ---- Backbone edges ----
        if P == 0:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr  = np.zeros((0, EDGE_FEAT_DIM), dtype=np.float32)
            result.append((x, edge_index, edge_attr))
            continue

        # Pairs with any interaction in [t - W_long, t + W_long]
        backbone_counts = edge_range_sum(t - w_long, t + w_long)  # [P]
        active_mask     = backbone_counts > 0.0
        active_idx      = np.where(active_mask)[0]

        if len(active_idx) == 0:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr  = np.zeros((0, EDGE_FEAT_DIM), dtype=np.float32)
            result.append((x, edge_index, edge_attr))
            continue

        # Edge features [E, 5]
        active_now_vals = (pair_arr[t, active_idx] > 0.0).astype(np.float32)  # binary flag: 1 iff pair has any gate at layer t
        e_ps = edge_rate(t - w_short, t - 1)[active_idx]
        e_fs = edge_rate(t + 1, t + w_short)[active_idx]
        e_pl = edge_rate(t - w_long,  t - 1)[active_idx]
        e_fl = edge_rate(t + 1, t + w_long) [active_idx]

        ea_half = np.stack([active_now_vals, e_ps, e_fs, e_pl, e_fl], axis=1)  # [E, 5]

        # Undirected: add both (u→v) and (v→u)
        us = np.array([all_pairs[i][0] for i in active_idx], dtype=np.int64)
        vs = np.array([all_pairs[i][1] for i in active_idx], dtype=np.int64)

        edge_index = np.stack([
            np.concatenate([us, vs]),
            np.concatenate([vs, us]),
        ], axis=0)                                          # [2, 2E]
        edge_attr = np.concatenate([ea_half, ea_half], axis=0)  # [2E, 5]

        result.append((x, edge_index, edge_attr))

    return result
