"""
baselines_tier2.py — Tier 2 communication-centric DQC baseline for MOSAIC comparison.

    B4: Wu-style time-aware beam search partitioner
        Implements the beam search algorithm from:
          "Efficient Time-Aware Partitioning of Quantum Circuits for
           Distributed Quantum Computing", Wu et al., arXiv:2603.04126, 2026.

        Minimizes a communication cost composed of:
            C_gate  — 2Q gates whose endpoints land on different QPUs
            C_state — qubits that change QPU between adjacent layers
            C(S) = w_gate * C_gate + w_state * C_state

        The algorithm incrementally builds a schedule layer by layer. At each
        layer it expands a beam of beta partial schedules using four candidate
        generation strategies (Preservation, Mitigation, Swaps, Diversification),
        scores each candidate by cumulative cost, and retains the top beta.

        No heterogeneity modelling — all QPUs are treated as identical. This
        makes it a strong communication-aware baseline that still cannot reason
        about HQC-specific tradeoffs (gate fidelity, decoherence differences).

Uniform interface (same as baselines_tier1.py):

    baseline_b4(rep, caps, config, K, ...) -> List[torch.Tensor]

    Args:
        rep    : CircuitRepresentation
        caps   : torch.Tensor [K]
        config : dict  (kept for interface consistency; not used internally)
        K      : int

    Returns:
        List[T] of LongTensor [N] — capacity-feasible hard assignment per layer.

Search parameters:
    beta_factor   : beam width  = beta_factor * N   (paper recommends 8N; default 4N)
    swaps_factor  : Gamma_swaps = swaps_factor * N  (paper recommends 4N; default 2N)
    random_factor : Gamma_random = random_factor * N (paper recommends 2N; default 1N)

    Reduced defaults cut runtime ~4x vs. paper recommendations while preserving
    the algorithm's core character. For 300-circuit eval at N~20-40 this gives
    roughly 1-3 seconds per circuit on CPU.

    w_state = w_gate = 1.0 following the paper's experimental setup.
"""

import random as _random
import torch
import numpy as np
from typing import Dict, List, Set, Tuple

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.circuit_representation import CircuitRepresentation


# =============================================================================
# Internal helpers
# =============================================================================

def _build_distance_matrix(K: int) -> np.ndarray:
    """
    Build a K x K QPU distance matrix.

    For K=2: D = [[0,1],[1,0]] — binary, no topology needed.
    For K>2: fully-connected topology (all distances = 1). This is the
    least-biased assumption when no physical topology is specified.
    """
    D = np.ones((K, K), dtype=np.float32)
    np.fill_diagonal(D, 0.0)
    return D


def _random_valid_assignment(N: int, caps_int: List[int], K: int,
                              rng: _random.Random) -> List[int]:
    """
    Generate one random capacity-feasible assignment of N qubits to K QPUs.
    Shuffles qubit indices and fills QPUs in order up to their capacities.
    If N > sum(caps), remaining qubits go to the least-loaded QPU (overflow).
    """
    order = list(range(N))
    rng.shuffle(order)
    assignment = [-1] * N
    counts = [0] * K

    # Fill slots in QPU order for shuffled qubits
    pool = list(range(K))
    k_idx = 0
    for q in order:
        # Advance to next QPU with remaining capacity
        while k_idx < K and counts[pool[k_idx]] >= caps_int[pool[k_idx]]:
            k_idx += 1
        if k_idx < K:
            k = pool[k_idx]
            assignment[q] = k
            counts[k] += 1
        else:
            # Overflow: least loaded
            k = min(range(K), key=lambda kk: counts[kk])
            assignment[q] = k
            counts[k] += 1

    return assignment


def _incremental_cost(
    prev: List[int],
    curr: List[int],
    twoq_edges: List[Tuple[int, int]],
    D: np.ndarray,
    w_state: float,
    w_gate: float,
) -> float:
    """
    Compute the incremental communication cost of moving from prev to curr.

    C_state: sum of D[prev[q], curr[q]] for all qubits q that changed QPU.
    C_gate:  sum of D[curr[u], curr[v]] for all 2Q gates (u,v) in this layer.
    """
    c_state = 0.0
    for q in range(len(prev)):
        if prev[q] != curr[q]:
            c_state += D[prev[q], curr[q]]

    c_gate = 0.0
    for u, v in twoq_edges:
        c_gate += D[curr[u], curr[v]]

    return w_state * c_state + w_gate * c_gate


def _is_valid(assignment: List[int], caps_int: List[int], K: int) -> bool:
    """Check capacity feasibility of an assignment."""
    counts = [0] * K
    for k in assignment:
        if k < 0 or k >= K:
            return False
        counts[k] += 1
    return all(counts[k] <= caps_int[k] for k in range(K))


# =============================================================================
# Candidate generation strategies
# =============================================================================

def _preservation(prev: List[int]) -> List[List[int]]:
    """Strategy 1: keep previous assignment unchanged."""
    return [list(prev)]


def _mitigation(
    prev: List[int],
    twoq_edges: List[Tuple[int, int]],
    caps_int: List[int],
    K: int,
) -> List[List[int]]:
    """
    Strategy 2: for each 2Q gate (u,v) whose endpoints are on different QPUs,
    generate candidates that co-locate them.

    K=2: two candidates (u→QPU_v, v→QPU_u) — original behaviour.
    K>2: additionally try:
        - move u alone to any third QPU C
        - move v alone to any third QPU C
        - move both u and v to the same third QPU C (full co-location on C)
    This ensures the beam considers all K techs as potential co-location targets,
    not just the two techs the gate's qubits already occupy.
    """
    candidates = []
    for u, v in twoq_edges:
        if prev[u] == prev[v]:
            continue  # already co-located, nothing to mitigate

        qpu_u, qpu_v = prev[u], prev[v]

        # Candidate: move u -> QPU of v
        cand = list(prev)
        cand[u] = qpu_v
        if _is_valid(cand, caps_int, K):
            candidates.append(cand)

        # Candidate: move v -> QPU of u
        cand = list(prev)
        cand[v] = qpu_u
        if _is_valid(cand, caps_int, K):
            candidates.append(cand)

        # K>2: third-tech candidates
        if K > 2:
            for C in range(K):
                if C == qpu_u or C == qpu_v:
                    continue

                # Move u alone to C
                cand = list(prev)
                cand[u] = C
                if _is_valid(cand, caps_int, K):
                    candidates.append(cand)

                # Move v alone to C
                cand = list(prev)
                cand[v] = C
                if _is_valid(cand, caps_int, K):
                    candidates.append(cand)

                # Move both u and v to C (co-locate on third tech)
                cand = list(prev)
                cand[u] = C
                cand[v] = C
                if _is_valid(cand, caps_int, K):
                    candidates.append(cand)

    return candidates


def _swaps(
    prev: List[int],
    n_swaps: int,
    caps_int: List[int],
    K: int,
    rng: _random.Random,
) -> List[List[int]]:
    """
    Strategy 3: generate n_swaps candidates by randomly swapping the QPU
    assignments of two distinct qubits. Swaps preserve capacity automatically
    (same qubit count per QPU before and after).

    Same-QPU pairs produce no-op candidates identical to Preservation and
    waste beam slots. We retry up to 8 times to find a cross-QPU pair;
    if none is found (e.g. all qubits already on one QPU), skip that swap.
    """
    N = len(prev)
    candidates = []
    for _ in range(n_swaps):
        if N < 2:
            break
        # Retry to find a cross-QPU pair; skip swap if none found
        for _attempt in range(8):
            i, j = rng.sample(range(N), 2)
            if prev[i] != prev[j]:
                break
        else:
            continue   # all attempts produced same-QPU pair; skip this swap
        cand = list(prev)
        cand[i], cand[j] = cand[j], cand[i]
        candidates.append(cand)
    return candidates


def _diversification(
    N: int,
    n_random: int,
    caps_int: List[int],
    K: int,
    rng: _random.Random,
) -> List[List[int]]:
    """Strategy 4: generate n_random fully random valid assignments."""
    return [_random_valid_assignment(N, caps_int, K, rng)
            for _ in range(n_random)]


def _migration(
    prev: List[int],
    n_migrations: int,
    caps_int: List[int],
    K: int,
    rng: _random.Random,
) -> List[List[int]]:
    """
    Strategy 5 (K>2 only): single-qubit relocation to any alternative tech.

    For K=2 this is equivalent to a swap with a random qubit on the other QPU
    and is already covered by _swaps. For K>2 it generates moves that cannot
    be produced by _mitigation (which only acts on cross-QPU gate pairs) or
    _swaps (which exchanges two qubits but cannot change the total count per
    QPU for the third tech).

    Specifically: sample n_migrations qubits at random, try moving each to
    every alternative QPU k != prev[q]. Keep valid moves.

    This ensures the beam includes candidates where individual qubits are
    relocated to techs they are not currently on and are not pulled there
    by a co-located gate partner — important when TI is the right choice
    for an idle qubit even if no gate partner is already on TI.
    """
    if K <= 2:
        return []  # covered by _swaps for K=2

    N = len(prev)
    candidates = []
    sample_size = min(N, n_migrations)
    qubits = rng.sample(range(N), sample_size)

    for q in qubits:
        src = prev[q]
        for dst in range(K):
            if dst == src:
                continue
            cand = list(prev)
            cand[q] = dst
            if _is_valid(cand, caps_int, K):
                candidates.append(cand)

    return candidates


# =============================================================================
# B4: Wu-style time-aware beam search partitioner
# =============================================================================

def baseline_b4(
    rep:          CircuitRepresentation,
    caps:         torch.Tensor,
    config:       dict,
    K:            int,
    beta_factor:  int   = 4,
    swaps_factor: int   = 2,
    random_factor: int  = 1,
    w_state:      float = 1.0,
    w_gate:       float = 1.0,
    seed:         int   = 0,
) -> List[torch.Tensor]:
    """
    Wu-style beam search circuit partitioner (B4).

    Builds a capacity-feasible qubit assignment schedule that minimises
    a communication cost:
        C = w_gate * (cross-QPU 2Q gates) + w_state * (qubit QPU changes)

    Parameters
    ----------
    beta_factor   : beam width = beta_factor * N
    swaps_factor  : Gamma_swaps = swaps_factor * N per beam member
    random_factor : Gamma_random = random_factor * N per beam member
    w_state       : weight for state teleportation (qubit moves)
    w_gate        : weight for gate teleportation (cross-QPU 2Q gates)
    seed          : RNG seed for reproducibility

    Returns
    -------
    List[T] of LongTensor [N] — same format as enforce_capacity_sequence.
    """
    T = len(rep.layers)
    N = rep.num_qubits
    caps_int = [int(c.item()) for c in caps]

    rng = _random.Random(seed)

    D = _build_distance_matrix(K)

    beta         = max(1, beta_factor   * N)
    n_swaps      = max(1, swaps_factor  * N)
    n_random     = max(1, random_factor * N)
    # For K>2: sample N qubits for migration (K-1 destinations each = N*(K-1) candidates)
    # For K=2: _migration returns [] immediately, no overhead
    n_migrations = N

    # Extract 2Q edges per layer once
    twoq_per_layer: List[List[Tuple[int, int]]] = []
    for layer in rep.layers:
        edges = []
        for gate_name, qargs in layer.gates:
            if len(qargs) == 2:
                edges.append((qargs[0], qargs[1]))
        twoq_per_layer.append(edges)

    # -------------------------------------------------------------------------
    # Initialise beam at t=0
    # beam: list of (cumulative_cost, schedule)
    #   schedule: List[List[int]] of length (layers processed so far)
    #   each element is a List[int] of length N (QPU index per qubit)
    # -------------------------------------------------------------------------
    # Generate beta random valid assignments for t=0
    # Cost at t=0 has no C_state (no previous layer), only C_gate
    init_assignments = [
        _random_valid_assignment(N, caps_int, K, rng)
        for _ in range(beta * 4)    # oversample then trim
    ]
    # Deduplicate
    seen = set()
    unique_inits = []
    for a in init_assignments:
        key = tuple(a)
        if key not in seen:
            seen.add(key)
            unique_inits.append(a)

    # Score each init assignment by C_gate only (no prev layer)
    scored_inits = []
    for a in unique_inits:
        c_gate = sum(D[a[u], a[v]] for u, v in twoq_per_layer[0])
        scored_inits.append((w_gate * c_gate, [a]))

    scored_inits.sort(key=lambda x: x[0])
    beam = scored_inits[:beta]

    # If fewer valid inits than beta, pad with duplicates of the best
    while len(beam) < beta:
        beam.append((beam[0][0], list(beam[0][1])))

    # -------------------------------------------------------------------------
    # Beam search over layers t = 1 .. T-1
    # -------------------------------------------------------------------------
    for t in range(1, T):
        edges_t = twoq_per_layer[t]
        candidates: List[Tuple[float, List[List[int]]]] = []

        for cum_cost, sched in beam:
            prev = sched[-1]   # assignment at t-1

            # Generate candidates for this layer
            cands_t: List[List[int]] = []
            cands_t.extend(_preservation(prev))
            cands_t.extend(_mitigation(prev, edges_t, caps_int, K))
            cands_t.extend(_swaps(prev, n_swaps, caps_int, K, rng))
            cands_t.extend(_diversification(N, n_random, caps_int, K, rng))
            cands_t.extend(_migration(prev, n_migrations, caps_int, K, rng))

            # Deduplicate within this expansion (set membership on tuple)
            seen_t: Set[tuple] = set()
            unique_cands: List[List[int]] = []
            for c in cands_t:
                key = tuple(c)
                if key not in seen_t:
                    seen_t.add(key)
                    unique_cands.append(c)

            # Score each candidate
            for curr in unique_cands:
                inc = _incremental_cost(prev, curr, edges_t, D, w_state, w_gate)
                new_cost = cum_cost + inc
                # Append layer t assignment to a copy of the partial schedule
                new_sched = sched + [curr]
                candidates.append((new_cost, new_sched))

        # Keep top beta by cumulative cost
        candidates.sort(key=lambda x: x[0])
        beam = candidates[:beta]

        # Safety: if beam is empty (shouldn't happen), reinitialise
        if not beam:
            fallback = _random_valid_assignment(N, caps_int, K, rng)
            beam = [(0.0, [list(fallback) for _ in range(t + 1)])]

    # -------------------------------------------------------------------------
    # Extract the best schedule (lowest cumulative cost)
    # -------------------------------------------------------------------------
    best_cost, best_sched = beam[0]

    # Convert to List[torch.Tensor]
    hard_assignments = [
        torch.tensor(best_sched[t], dtype=torch.long)
        for t in range(T)
    ]
    return hard_assignments


# =============================================================================
# B5: Burt-style FM + gate-grouping hypergraph partitioner
# =============================================================================
"""
B5: Reduced Burt-style FM + gate-grouping baseline.

Implements the core of the multilevel framework from:
    "A Multilevel Framework for Partitioning Quantum Circuits",
    Burt, Chen, Leung, Quantum 2026 (arXiv:2503.19082).

What is implemented (defines B5's character):
    - Temporal hypergraph: N*T nodes, state edges + gate edges
    - Gate grouping (Algorithm 1): greedy grouping of compatible 2Q gates
      into hyper-edges with root/receiver set structure
    - Entanglement cost (Eq. 18/19): root/receiver partition set difference
    - Fiduccia-Mattheyses refinement (Algorithm 2): gain buckets, per-pass
      locking, rollback to best cumulative gain
    - Greedy static initialization (all qubits fixed across all layers)
    - Per-time-step capacity constraints

What is NOT implemented (acceptable reductions for a comparison baseline):
    - Multilevel coarsening hierarchy (Algorithms 3/4/8/9) — the paper's
      full strongest pipeline; omitting it makes this a reduced Burt-style
      FM+grouping baseline, not the full MLFM-R method
    - Exploratory FM passes (standard FM only)
    - Circuit extraction (not needed; we only need the assignment schedule)

Gate grouping note:
    ROI circuits use mixed gate types. We assign min(u,v) as root for all
    2Q gates, matching Burt's symmetric-gate convention (Eq. 14). Groups
    are closed when a non-diagonal 1Q gate appears on the root qubit.
    Since we cannot inspect 1Q gate parameters from CircuitRepresentation,
    we conservatively close groups at every 1Q gate on the root qubit.
    This is correct and conservative — it loses some grouping opportunities
    vs Burt's full CP(theta) formulation but is faithful to the core idea.

Root set completeness:
    Per the paper, the root set of a group contains ALL nodes (q_root, t)
    from the first to last gate time step — not just endpoints. This ensures
    the Eq. 18 cost correctly accounts for nested state teleportation via
    intermediate root qubit movements.

Capacity:
    Enforced per time step via a counts[t][k] array (T x K).
    Moving node (q,t) changes only counts[t][src] and counts[t][dst].
"""


def _node_id(q: int, t: int, T: int) -> int:
    """Encode (qubit, layer) as a single integer node id."""
    return q * T + t


def _node_qt(node_id: int, T: int) -> Tuple[int, int]:
    """Decode node id back to (qubit, layer)."""
    return node_id // T, node_id % T


def _build_hypergraph(
    rep: CircuitRepresentation,
    use_grouping: bool,
) -> Tuple[int, int, List[Tuple[List[int], List[int]]]]:
    """
    Build the temporal hypergraph for a circuit.

    Returns:
        N          : number of qubits
        T          : number of layers
        edges      : List of (root_nodes, receiver_nodes) — each is a list
                     of node_ids. Both state edges and gate/hyper-edges are
                     included in this flat list.

    State edges: root = [(q, t)], receiver = [(q, t+1)]
    Gate edges (ungrouped): root = [(min_qubit, t)], receiver = [(max_qubit, t)]
    Hyper-edges (grouped): root = [(q_root, t_first)..(q_root, t_last)],
                           receiver = [(q_recv_i, t_i) for each gate in group]
    """
    N = rep.num_qubits
    T = len(rep.layers)
    edges: List[Tuple[List[int], List[int]]] = []

    # --- State edges: connect (q,t) -> (q,t+1) for all q, t in [0,T-2] ---
    for q in range(N):
        for t in range(T - 1):
            root_n     = _node_id(q, t,     T)
            receiver_n = _node_id(q, t + 1, T)
            edges.append(([root_n], [receiver_n]))

    # --- Gate edges / hyper-edges ---
    if not use_grouping:
        # Ungrouped: one hyper-edge per 2Q gate
        for t, layer in enumerate(rep.layers):
            for gate_name, qargs in layer.gates:
                if len(qargs) == 2:
                    u, v = qargs
                    root_q = min(u, v)
                    recv_q = max(u, v)
                    edges.append(
                        ([_node_id(root_q, t, T)],
                         [_node_id(recv_q, t, T)])
                    )
    else:
        # Gate grouping (Algorithm 1 — conservative asymmetric version).
        # active_group[q] = (root_q, first_t, last_t, receiver_nodes)
        # A group rooted on q is open when active_group[q] is not None.
        active_group: Dict[int, Tuple[int, int, int, List[int]]] = {}

        def close_group(q: int):
            if q not in active_group:
                return
            root_q, first_t, last_t, recv_nodes = active_group.pop(q)
            # Root set: all (root_q, t) from first_t to last_t inclusive
            root_nodes = [_node_id(root_q, t_r, T)
                          for t_r in range(first_t, last_t + 1)]
            edges.append((root_nodes, recv_nodes))

        for t, layer in enumerate(rep.layers):
            # --- Process 2Q gates at this layer ---
            twoq_at_t: List[Tuple[int, int]] = []
            for gate_name, qargs in layer.gates:
                if len(qargs) == 2:
                    u, v = qargs
                    twoq_at_t.append((min(u, v), max(u, v)))

            for root_q, recv_q in twoq_at_t:
                recv_node = _node_id(recv_q, t, T)
                if root_q in active_group:
                    # Extend existing group
                    rq, first_t, _last_t, recv_nodes = active_group[root_q]
                    recv_nodes.append(recv_node)
                    active_group[root_q] = (rq, first_t, t, recv_nodes)
                else:
                    # Start new group
                    active_group[root_q] = (root_q, t, t, [recv_node])

                # The recv_q may be root of another group — adding it as
                # receiver closes any open group rooted on recv_q
                # (it is now occupied as a receiver at this time step)
                close_group(recv_q)

            # --- Process 1Q gates: conservatively close groups ---
            for gate_name, qargs in layer.gates:
                if len(qargs) == 1:
                    q_1q = qargs[0]
                    close_group(q_1q)

        # Close all remaining open groups at end of circuit
        for q in list(active_group.keys()):
            close_group(q)

    return N, T, edges


def _init_assignment_greedy(N: int, T: int, caps_int: List[int], K: int) -> List[List[int]]:
    """
    Greedy static initialization: fill QPUs in order for qubit 0..N-1,
    then replicate the same assignment across all T layers.
    This is the paper's stated starting point (all qubits fixed in same
    partition for full circuit depth).

    Returns:
        phi[t][q] = QPU index  (T outer, N inner)
    """
    assignment_t0 = [-1] * N
    counts = [0] * K
    k = 0
    for q in range(N):
        while k < K and counts[k] >= caps_int[k]:
            k += 1
        if k < K:
            assignment_t0[q] = k
            counts[k] += 1
        else:
            # Overflow: least loaded
            k_min = min(range(K), key=lambda kk: counts[kk])
            assignment_t0[q] = k_min
            counts[k_min] += 1

    return [list(assignment_t0) for _ in range(T)]


def _compute_edge_cost(
    root_nodes:     List[int],
    receiver_nodes: List[int],
    phi_flat:       List[int],  # phi_flat[node_id] = QPU
) -> int:
    """
    Eq. 18: e-bit cost = number of receiver partitions NOT present in root partitions.
    """
    root_parts     = {phi_flat[n] for n in root_nodes}
    receiver_parts = {phi_flat[n] for n in receiver_nodes}
    return len(receiver_parts - root_parts)


def _total_cost(
    edges:    List[Tuple[List[int], List[int]]],
    phi_flat: List[int],
) -> int:
    """Total entanglement cost (Eq. 19): sum of Eq. 18 over all edges."""
    return sum(_compute_edge_cost(r, recv, phi_flat) for r, recv in edges)


def _build_node_adjacency(
    edges: List[Tuple[List[int], List[int]]],
    N:     int,
    T:     int,
) -> Dict[int, List[int]]:
    """
    For each node_id, list the edge indices in which it participates.
    Used to find affected edges when a node moves.
    """
    adj: Dict[int, List[int]] = {nid: [] for nid in range(N * T)}
    for eidx, (root_nodes, receiver_nodes) in enumerate(edges):
        for n in root_nodes:
            adj[n].append(eidx)
        for n in receiver_nodes:
            adj[n].append(eidx)
    return adj


def _fm_pass(
    edges:     List[Tuple[List[int], List[int]]],
    phi_flat:  List[int],           # mutable, updated in place
    adj:       Dict[int, List[int]],
    counts:    List[List[int]],     # counts[t][k] — mutable
    caps_int:  List[int],
    N:         int,
    T:         int,
    K:         int,
) -> int:
    """
    Single FM pass (Algorithm 2) — multi-destination version.

    Fixes vs. original single-destination version:
      Bug 1: All (node, dst) pairs are stored in buckets, not just the
             best destination per node. When a destination fills up, other
             destinations for the same node remain in the buckets and are
             naturally tried next.
      Bug 2: All (node, dst) pairs are stored unconditionally, including
             currently-infeasible destinations. Capacity is checked solely at
             selection time. If a move frees space in a partition, all same-
             layer nodes — adjacent or not — can immediately use that
             destination because their entry is already in the buckets.
             No separate same-layer recomputation pass is needed.

    Gain updates after a move:
      - Exact gain values change only for hypergraph-adjacent nodes (edge
        cost depends on partition membership, which only changes for nodes
        sharing an edge with the moved node).
      - Capacity feasibility changes for same-layer nodes are handled
        implicitly at selection time: all (node, dst) pairs are stored
        unconditionally, so newly-feasible destinations are already present
        in the buckets without any additional recomputation pass.

    node_entries[n] = {dst: gain} tracks all current bucket entries for n,
    enabling O(K) removal when n's gains need updating.

    Returns the net gain achieved (negative = improvement = cost decrease).
    """
    from collections import defaultdict

    total_nodes = N * T

    def gain_for_move(n: int, dst: int) -> int:
        """Cost change (negative = improvement) of moving n to dst."""
        src = phi_flat[n]
        if src == dst:
            return 0
        delta = 0
        for eidx in adj[n]:
            root_nodes, recv_nodes = edges[eidx]
            old_c = _compute_edge_cost(root_nodes, recv_nodes, phi_flat)
            phi_flat[n] = dst
            new_c = _compute_edge_cost(root_nodes, recv_nodes, phi_flat)
            phi_flat[n] = src  # restore
            delta += new_c - old_c
        return delta

    # --- Build buckets with ALL (node, dst) pairs ---
    # buckets[gain] = set of (node, dst)
    # node_entries[n] = {dst: gain}  — inverse index for fast removal
    buckets: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
    node_entries: Dict[int, Dict[int, int]] = {n: {} for n in range(total_nodes)}

    for n in range(total_nodes):
        src = phi_flat[n]
        for dst in range(K):
            if dst == src:
                continue
            # All destinations stored unconditionally — including currently
            # infeasible ones. Capacity feasibility is checked exclusively at
            # selection time so that capacity openings from later moves are
            # always visible to all nodes in the same layer.
            g = gain_for_move(n, dst)
            buckets[g].add((n, dst))
            node_entries[n][dst] = g

    def _remove_node_entries(n: int):
        """Remove all bucket entries for node n."""
        for dst, g in node_entries[n].items():
            buckets.get(g, set()).discard((n, dst))
        node_entries[n].clear()

    def _add_node_entries(n: int):
        """Recompute and insert all bucket entries for unlocked node n."""
        src = phi_flat[n]
        for dst in range(K):
            if dst == src:
                continue
            # All destinations stored unconditionally — capacity checked at
            # selection time only, so openings are always visible.
            g = gain_for_move(n, dst)
            buckets[g].add((n, dst))
            node_entries[n][dst] = g

    locked: Set[int] = set()
    move_sequence: List[Tuple[int, int, int]] = []  # (node, src, dst)
    cumulative_gains: List[int] = [0]
    cumulative = 0

    for _ in range(total_nodes):
        if not buckets:
            break

        # --- Select best admissible move ---
        # Scan gain buckets from lowest (most improving) upward.
        # Skip locked nodes and capacity-infeasible destinations.
        chosen = None
        chosen_gain = None
        for g in sorted(buckets.keys()):
            stale: Set[Tuple[int, int]] = set()
            for (n, dst) in list(buckets[g]):
                if n in locked:
                    stale.add((n, dst))
                    continue
                q, t = _node_qt(n, T)
                if counts[t][dst] >= caps_int[dst]:
                    # Capacity became infeasible after a prior move; skip but
                    # do not remove — the entry is still valid if space opens.
                    continue
                chosen = (n, dst)
                chosen_gain = g
                break
            # Clean up locked-node entries (they will never be chosen again)
            for item in stale:
                buckets[g].discard(item)
            if not buckets[g]:
                del buckets[g]
            if chosen is not None:
                break

        if chosen is None:
            break

        n_move, dst_move = chosen
        src_move = phi_flat[n_move]
        q_move, t_move = _node_qt(n_move, T)

        # Apply move
        phi_flat[n_move] = dst_move
        counts[t_move][src_move] -= 1
        counts[t_move][dst_move] += 1

        cumulative += chosen_gain
        move_sequence.append((n_move, src_move, dst_move))
        cumulative_gains.append(cumulative)
        locked.add(n_move)

        # Remove all entries for the moved node (it is now locked)
        _remove_node_entries(n_move)

        # --- Update gain values for hypergraph-adjacent unlocked nodes ---
        # Their edge costs have changed because phi_flat[n_move] changed.
        # Capacity feasibility changes for same-layer nodes are handled
        # implicitly at selection time — no extra recomputation needed.
        affected_nodes: Set[int] = set()
        for eidx in adj[n_move]:
            root_nodes, recv_nodes = edges[eidx]
            for nn in root_nodes:
                if nn != n_move and nn not in locked:
                    affected_nodes.add(nn)
            for nn in recv_nodes:
                if nn != n_move and nn not in locked:
                    affected_nodes.add(nn)

        for nn in affected_nodes:
            _remove_node_entries(nn)
            _add_node_entries(nn)

    # --- Rollback to best cumulative gain state ---
    if not cumulative_gains:
        return 0

    best_iter = int(np.argmin(cumulative_gains))
    net_gain  = cumulative_gains[best_iter]

    for i in range(len(move_sequence) - 1, best_iter - 1, -1):
        n_rb, src_rb, dst_rb = move_sequence[i]
        q_rb, t_rb = _node_qt(n_rb, T)
        phi_flat[n_rb] = src_rb
        counts[t_rb][dst_rb] -= 1
        counts[t_rb][src_rb] += 1

    return net_gain


def baseline_b5(
    rep:          CircuitRepresentation,
    caps:         torch.Tensor,
    config:       dict,
    K:            int,
    n_passes:     int  = 10,
    use_grouping: bool = True,
) -> List[torch.Tensor]:
    """
    Reduced Burt-style FM + gate-grouping hypergraph partitioner (B5).

    Minimises an entanglement cost:
        C = sum over hyper-edges of |recv_partitions - root_partitions|  (Eq. 18/19)

    No heterogeneity modelling — all QPUs treated as identical.
    This is a communication-aware dynamic baseline that reasons about
    circuit structure through gate grouping and FM refinement, but cannot
    reason about HQC-specific tradeoffs (fidelity, decoherence).

    Parameters
    ----------
    n_passes     : number of FM passes (default 10)
    use_grouping : enable greedy gate grouping into hyper-edges (default True)

    Returns
    -------
    List[T] of LongTensor [N] — same format as enforce_capacity_sequence.
    """
    T = len(rep.layers)
    N = rep.num_qubits
    caps_int = [int(c.item()) for c in caps]

    # --- Build hypergraph ---
    _, _, edges = _build_hypergraph(rep, use_grouping)

    # --- Build node adjacency index ---
    adj = _build_node_adjacency(edges, N, T)

    # --- Greedy static initialization ---
    # phi[t][q] = QPU index
    phi_2d = _init_assignment_greedy(N, T, caps_int, K)

    # Flat phi: phi_flat[node_id] = QPU
    phi_flat = [phi_2d[t][q] for q in range(N) for t in range(T)]
    # Note: node_id = q*T + t, so phi_flat[q*T + t] = phi_2d[t][q]

    # --- Per-time-step counts: counts[t][k] ---
    counts: List[List[int]] = [[0] * K for _ in range(T)]
    for q in range(N):
        for t in range(T):
            k = phi_flat[_node_id(q, t, T)]
            counts[t][k] += 1

    # --- FM passes ---
    for _ in range(n_passes):
        net_gain = _fm_pass(edges, phi_flat, adj, counts, caps_int, N, T, K)
        if net_gain >= 0:
            break  # No improvement; stop early

    # --- Extract schedule: phi_2d[t][q] from phi_flat ---
    hard_assignments = []
    for t in range(T):
        assignment = [phi_flat[_node_id(q, t, T)] for q in range(N)]
        hard_assignments.append(torch.tensor(assignment, dtype=torch.long))

    return hard_assignments
