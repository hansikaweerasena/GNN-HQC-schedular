"""
gamma_scoring.py — Edge-wise routing difficulty estimation.

Computes per-edge, per-technology routing inflation scores (Gamma) for
use in the heterogeneous quantum scheduler's cost model.

Two scoring channels:
  - Dense (local congestion): for edges embedded in dense local neighborhoods
  - Non-local (bridge detection): for edges connecting structurally disjoint regions

Combined via mutual exclusion: non-local edges get Gamma_nonlocal,
local edges get Gamma_dense.

All window sizes and thresholds derive from a single technology parameter:
kappa (average connectivity of the target topology).

API:
  compute_edge_gamma(edge_counts_per_layer, num_qubits, tech_configs)
    -> Dict[tech_name, List[Dict[(u,v) -> gamma]]]

  Each entry is per-layer, per-edge gamma for one technology.
  All-to-all technologies return 0.0 for all edges.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx


# =============================================================================
# Types
# =============================================================================

Pair = Tuple[int, int]
EdgeCounts = Dict[Pair, float]
EffGraph = Dict[Pair, float]


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class GammaTechConfig:
    """Per-technology scoring configuration. Derives window sizes from kappa."""
    name: str
    kappa: float            # average connectivity (0 or negative = all-to-all)
    all_to_all: bool = False

    # Configurable params with strong defaults
    gamma_max: float = 2.5
    delta_community: int = 3
    pair_reuse_threshold: int = 2
    dense_lambda_decay: float = 0.85
    eps: float = 1e-12

    @property
    def is_all_to_all(self) -> bool:
        return self.all_to_all or self.kappa <= 0

    @property
    def dense_window_radius(self) -> int:
        return int(math.ceil(self.kappa)) if not self.is_all_to_all else 0

    @property
    def nl_window_radius(self) -> int:
        return 2 * int(math.ceil(self.kappa)) if not self.is_all_to_all else 0

    @property
    def pair_reuse_radius(self) -> int:
        return self.nl_window_radius


def config_from_tech_dict(tech: Dict[str, Any], proxy_cfg: Dict[str, Any]) -> GammaTechConfig:
    """Build GammaTechConfig from cost_config_v3.json tech entry + connectivity_proxy block."""
    routing = tech.get("routing", {})
    kappa = float(routing.get("kappa", 0.0))
    a2a = bool(routing.get("all_to_all", False))

    return GammaTechConfig(
        name=str(tech.get("name", "unknown")),
        kappa=kappa,
        all_to_all=a2a,
        gamma_max=float(proxy_cfg.get("gamma_max", 2.5)),
        delta_community=int(proxy_cfg.get("delta_community", 3)),
        pair_reuse_threshold=int(proxy_cfg.get("pair_reuse_threshold", 2)),
        dense_lambda_decay=float(proxy_cfg.get("dense_lambda_decay", 0.85)),
        eps=float(proxy_cfg.get("eps", 1e-12)),
    )


# =============================================================================
# Helpers
# =============================================================================


def _sorted_pair(u: int, v: int) -> Pair:
    return (u, v) if u < v else (v, u)


# =============================================================================
# Edge counting from raw gate lists
# =============================================================================


def layer_edge_counts_from_gates(
    layers: Sequence[Sequence[Tuple[int, int]]],
) -> List[EdgeCounts]:
    """
    Convert raw gate lists to per-layer edge count dicts.

    Args:
        layers: list of layers, each layer is a list of (u, v) qubit pairs.

    Returns:
        List of {(a,b): count} dicts, one per layer.
    """
    out: List[EdgeCounts] = []
    for layer_gates in layers:
        counts: EdgeCounts = defaultdict(float)
        for u, v in layer_gates:
            counts[_sorted_pair(int(u), int(v))] += 1.0
        out.append(dict(counts))
    return out


# =============================================================================
# Window effective graph construction
# =============================================================================


def build_flat_window_graphs(
    edge_counts_per_layer: Sequence[EdgeCounts],
    radius: int,
) -> List[EffGraph]:
    """
    Build flat (uniform weight) bidirectional window effective graphs.

    For each layer index, accumulates all edge interactions within
    ±radius layers with equal weight. No decay.
    """
    radius = max(0, int(radius))
    n_layers = len(edge_counts_per_layer)
    out: List[EffGraph] = []

    for center in range(n_layers):
        acc: EffGraph = defaultdict(float)
        start = max(0, center - radius)
        end = min(n_layers - 1, center + radius)
        for j in range(start, end + 1):
            for e, n in edge_counts_per_layer[j].items():
                acc[e] += float(n)
        out.append(dict(acc))

    return out


# =============================================================================
# Dense score components
# =============================================================================


def _segment_degrees(edge_counts: EdgeCounts) -> Dict[int, float]:
    """Compute degree of each node in the interaction graph."""
    deg: Dict[int, float] = defaultdict(float)
    for (a, b), n in edge_counts.items():
        deg[a] += float(n)
        deg[b] += float(n)
    return dict(deg)


def _congestion_raw(
    edge_counts: EdgeCounts,
) -> Dict[Pair, float]:
    """
    Raw congestion C_raw per edge: sum of endpoint degrees minus their
    mutual interaction count, representing competing demand.
    """
    deg = _segment_degrees(edge_counts)
    out: Dict[Pair, float] = {}
    for (a, b), n_uv in edge_counts.items():
        nn = float(n_uv)
        out[(a, b)] = max(0.0, (deg.get(a, 0.0) - nn) + (deg.get(b, 0.0) - nn))
    return out


def _profile_weights(radius: int) -> List[float]:
    """One-sided temporal profile weights (nearest to farthest)."""
    side = {1: 0.75, 2: 0.50, 3: 0.25}
    return [side.get(d, max(0.0, 1.0 - 0.25 * d)) for d in range(1, radius + 1)]


def _one_side_weight(delta: int, weights: Sequence[float]) -> float:
    d = int(delta)
    if d <= 0 or d > len(weights):
        return 0.0
    return float(weights[d - 1])


def _weighted_neighbor_profile(
    edge_counts_per_layer: Sequence[EdgeCounts],
    layer_idx: int,
    endpoint: int,
    direction: str,
    radius: int,
    weights: Sequence[float],
) -> Dict[int, float]:
    """Build weighted neighbor profile looking past or future from layer_idx."""
    out: Dict[int, float] = defaultdict(float)
    n_layers = len(edge_counts_per_layer)
    for delta in range(1, radius + 1):
        j = layer_idx - delta if direction == "past" else layer_idx + delta
        if j < 0 or j >= n_layers:
            continue
        w = _one_side_weight(delta, weights)
        if w <= 0.0:
            continue
        for (a, b), n in edge_counts_per_layer[j].items():
            if a == endpoint:
                out[b] += w * float(n)
            elif b == endpoint:
                out[a] += w * float(n)
    return dict(out)


def _weighted_jaccard(a: Dict[int, float], b: Dict[int, float], eps: float) -> float:
    """Weighted Jaccard similarity between two neighbor profiles."""
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 0.0
    num = sum(min(float(a.get(k, 0.0)), float(b.get(k, 0.0))) for k in keys)
    den = sum(max(float(a.get(k, 0.0)), float(b.get(k, 0.0))) for k in keys) + eps
    return float(num / den)


def _pair_reuse_score(
    edge_counts_per_layer: Sequence[EdgeCounts],
    layer_idx: int,
    pair: Pair,
    radius: int,
    lambda_decay: float,
) -> float:
    """Exponentially decayed reuse score looking backward."""
    num, den = 0.0, 0.0
    for delta in range(1, radius + 1):
        j = layer_idx - delta
        if j < 0:
            continue
        w = lambda_decay ** max(0, delta - 1)
        den += w
        if float(edge_counts_per_layer[j].get(pair, 0.0)) > 0.0:
            num += w
    return float(num / den) if den > 0.0 else 0.0


def compute_dense_gamma(
    edge_counts_per_layer: Sequence[EdgeCounts],
    dense_eff: EffGraph,
    layer_idx: int,
    pair: Pair,
    cfg: GammaTechConfig,
) -> float:
    """
    Compute dense score Gamma_dense for one edge at one layer.

    Gamma_dense = B_dense × (1 - D_dense)

    where:
        B_dense = max(0, C_raw - K_pair) / K_pair
        D_dense = max(S_struct, R_pair)
        S_struct = min(J_u, J_v)  (past/future structural stability)
        R_pair = pair reuse score (backward-looking)
    """
    kappa = float(cfg.kappa)
    K_extra = max(0.0, kappa - 1.0)
    K_pair = 2.0 * K_extra

    # Congestion from the dense effective graph
    c_raw_map = _congestion_raw(dense_eff)
    C_raw = float(c_raw_map.get(pair, 0.0))

    if K_pair <= 0.0:
        B_dense = 0.0
    else:
        B_dense = max(0.0, C_raw - K_pair) / (K_pair + cfg.eps)

    # Structural stability discount
    radius = cfg.dense_window_radius
    pw = _profile_weights(radius)

    u, v = pair
    P_u = _weighted_neighbor_profile(edge_counts_per_layer, layer_idx, u, "past", radius, pw)
    F_u = _weighted_neighbor_profile(edge_counts_per_layer, layer_idx, u, "future", radius, pw)
    P_v = _weighted_neighbor_profile(edge_counts_per_layer, layer_idx, v, "past", radius, pw)
    F_v = _weighted_neighbor_profile(edge_counts_per_layer, layer_idx, v, "future", radius, pw)

    J_u = _weighted_jaccard(P_u, F_u, cfg.eps)
    J_v = _weighted_jaccard(P_v, F_v, cfg.eps)
    S_struct = min(J_u, J_v)

    R_pair = _pair_reuse_score(
        edge_counts_per_layer, layer_idx, pair,
        radius=radius, lambda_decay=cfg.dense_lambda_decay,
    )
    D_dense = max(S_struct, R_pair)

    return B_dense * (1.0 - D_dense)


# =============================================================================
# Non-local classification and scoring
# =============================================================================


def _eff_to_nx(eff: EffGraph, num_qubits: int) -> nx.Graph:
    """Convert effective graph dict to networkx Graph."""
    g = nx.Graph()
    g.add_nodes_from(range(num_qubits))
    for (u, v), w in eff.items():
        g.add_edge(u, v)
    return g


def has_common_neighbor(eff: EffGraph, pair: Pair) -> bool:
    """
    Stage 1: Local bridge test (Granovetter, 1973).
    Returns True if endpoints share a common neighbor (locally embedded).
    Returns False if no common neighbor (local bridge candidate).
    Cost: O(deg(u) + deg(v)).
    """
    u, v = pair
    nbrs_u: set = set()
    nbrs_v: set = set()
    for (a, b) in eff:
        if a == u and b != v:
            nbrs_u.add(b)
        elif b == u and a != v:
            nbrs_u.add(a)
        if a == v and b != u:
            nbrs_v.add(b)
        elif b == v and a != u:
            nbrs_v.add(a)
    return len(nbrs_u & nbrs_v) > 0


def _bfs_detour(g: nx.Graph, src: int, dst: int) -> Tuple[float, int, int]:
    """
    BFS from src with edge (src, dst) removed.
    Returns (distance_to_dst, component_size_of_src, component_size_of_dst).
    """
    eu, ev = min(src, dst), max(src, dst)

    def skip(a: int, b: int) -> bool:
        return (min(a, b) == eu and max(a, b) == ev)

    # BFS from src
    visited_src = {src}
    q = deque([(src, 0)])
    dist = math.inf
    while q:
        node, d = q.popleft()
        for nbr in g.neighbors(node):
            if skip(node, nbr):
                continue
            if nbr not in visited_src:
                visited_src.add(nbr)
                q.append((nbr, d + 1))
                if nbr == dst:
                    dist = d + 1

    # BFS from dst (for component size only)
    visited_dst = {dst}
    q2 = deque([dst])
    while q2:
        node = q2.popleft()
        for nbr in g.neighbors(node):
            if skip(node, nbr):
                continue
            if nbr not in visited_dst:
                visited_dst.add(nbr)
                q2.append(nbr)

    return dist, len(visited_src), len(visited_dst)


def detour_metrics(eff: EffGraph, num_qubits: int, pair: Pair) -> Tuple[float, int, int]:
    """
    Compute L_detour, |C_u|, |C_v| for pair (u,v) in the effective graph.
    Returns (l_detour, component_u, component_v).
    """
    g = _eff_to_nx(eff, num_qubits)
    u, v = pair
    if not g.has_edge(u, v):
        return math.inf, 1, 1
    return _bfs_detour(g, u, v)


def pair_reuse_count(
    edge_counts_per_layer: Sequence[EdgeCounts],
    layer_idx: int,
    pair: Pair,
    radius: int,
) -> int:
    """Count how many layers within ±radius contain this edge."""
    n_layers = len(edge_counts_per_layer)
    count = 0
    for off in range(-radius, radius + 1):
        j = layer_idx + off
        if 0 <= j < n_layers and pair in edge_counts_per_layer[j]:
            count += 1
    return count


def classify_and_score_nonlocal(
    edge_counts_per_layer: Sequence[EdgeCounts],
    nl_eff: EffGraph,
    num_qubits: int,
    layer_idx: int,
    pair: Pair,
    cfg: GammaTechConfig,
) -> Tuple[bool, float]:
    """
    3-stage local-bridge classifier + non-local score.

    Returns:
        (is_nonlocal, gamma_nonlocal)

    Stages:
        1. Common-neighbor test (no common neighbor → local bridge)
        2. Community size guard (both sides >= delta_community)
        3. Pair-reuse guard (< pair_reuse_threshold occurrences in window)

    Score for classified non-local edges:
        L_capped = min(L_detour, floor(|V_active| / kappa) + 1)
        Gamma_nl = min((L_capped - 1) / kappa, gamma_max)
    """
    kappa = float(cfg.kappa)

    # Stage 1: common-neighbor test
    if has_common_neighbor(nl_eff, pair):
        return False, 0.0

    # Stage 2: community size guard (BFS gives L_detour for free)
    l_raw, cu, cv = detour_metrics(nl_eff, num_qubits, pair)
    if cu < cfg.delta_community or cv < cfg.delta_community:
        return False, 0.0

    # Stage 3: pair-reuse guard
    reuse = pair_reuse_count(
        edge_counts_per_layer, layer_idx, pair, cfg.pair_reuse_radius,
    )
    if reuse >= cfg.pair_reuse_threshold:
        return False, 0.0

    # Classified as non-local — compute score
    v_active = set()
    for (a, b) in nl_eff:
        v_active.add(a)
        v_active.add(b)
    l_max = int(len(v_active) / kappa) + 1

    l_detour = float(l_max) if math.isinf(l_raw) else float(l_raw)
    l_capped = min(l_detour, float(l_max))
    gamma_nl = min(max(0.0, (l_capped - 1.0) / kappa), cfg.gamma_max)

    return True, gamma_nl


# =============================================================================
# Combined per-edge scoring for one technology
# =============================================================================


def compute_edge_gamma_for_tech(
    edge_counts_per_layer: Sequence[EdgeCounts],
    num_qubits: int,
    cfg: GammaTechConfig,
) -> List[Dict[Pair, float]]:
    """
    Compute per-edge gamma for all layers for one technology.

    Returns:
        List of {(u,v): gamma} dicts, one per layer.
        For all-to-all technologies, returns all zeros.
    """
    n_layers = len(edge_counts_per_layer)

    # All-to-all: no routing overhead
    if cfg.is_all_to_all:
        return [
            {pair: 0.0 for pair in layer_ec}
            for layer_ec in edge_counts_per_layer
        ]

    # Build effective graphs for this technology
    dense_effs = build_flat_window_graphs(edge_counts_per_layer, cfg.dense_window_radius)
    nl_effs = build_flat_window_graphs(edge_counts_per_layer, cfg.nl_window_radius)

    result: List[Dict[Pair, float]] = []

    for s in range(n_layers):
        layer_ec = edge_counts_per_layer[s]
        dense_eff = dense_effs[s]
        nl_eff = nl_effs[s]
        layer_gamma: Dict[Pair, float] = {}

        for pair in layer_ec:
            # Try non-local classification first
            is_nl, gamma_nl = classify_and_score_nonlocal(
                edge_counts_per_layer, nl_eff, num_qubits, s, pair, cfg,
            )

            if is_nl:
                layer_gamma[pair] = gamma_nl
            else:
                # Dense score
                layer_gamma[pair] = compute_dense_gamma(
                    edge_counts_per_layer, dense_eff, s, pair, cfg,
                )

        result.append(layer_gamma)

    return result


# =============================================================================
# Multi-technology top-level API
# =============================================================================


def compute_all_tech_gamma(
    edge_counts_per_layer: Sequence[EdgeCounts],
    num_qubits: int,
    tech_configs: Sequence[GammaTechConfig],
) -> Dict[str, List[Dict[Pair, float]]]:
    """
    Compute per-edge gamma for all layers, for all technologies.

    Args:
        edge_counts_per_layer: per-layer edge count dicts from the circuit.
        num_qubits: total qubit count.
        tech_configs: list of GammaTechConfig, one per technology.

    Returns:
        {tech_name: List[{(u,v): gamma}]} — one list per tech,
        each list has one dict per layer.
    """
    return {
        cfg.name: compute_edge_gamma_for_tech(
            edge_counts_per_layer, num_qubits, cfg,
        )
        for cfg in tech_configs
    }


def compute_edge_gamma_tensor_data(
    edge_counts_per_layer: Sequence[EdgeCounts],
    num_qubits: int,
    tech_configs: Sequence[GammaTechConfig],
) -> List[Dict[Pair, List[float]]]:
    """
    Compute per-edge gamma as [K]-length lists for direct tensor conversion.

    Returns:
        List (per layer) of {(u,v): [gamma_tech0, gamma_tech1, ...]}
        Ready for torch.tensor conversion in cost_function.py.
    """
    K = len(tech_configs)
    n_layers = len(edge_counts_per_layer)

    # Compute per-tech gamma
    per_tech = compute_all_tech_gamma(edge_counts_per_layer, num_qubits, tech_configs)

    # Merge into [K]-length vectors per edge
    result: List[Dict[Pair, List[float]]] = []
    for s in range(n_layers):
        layer_ec = edge_counts_per_layer[s]
        layer_gamma: Dict[Pair, List[float]] = {}
        for pair in layer_ec:
            gamma_vec = []
            for cfg in tech_configs:
                tech_gamma = per_tech[cfg.name][s]
                gamma_vec.append(tech_gamma.get(pair, 0.0))
            layer_gamma[pair] = gamma_vec
        result.append(layer_gamma)

    return result
