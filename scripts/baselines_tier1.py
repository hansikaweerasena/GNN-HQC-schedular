"""
baselines_tier1.py — Tier 1 HQC-aware greedy baselines for MOSAIC comparison.

Three simple hardware-aware heuristics that capture single aspects of
heterogeneity, used to show that greedy one-criterion matching is insufficient
for full HQC scheduling.

    B1: Static idle-first hardware-aware greedy
        One fixed mapping for the entire circuit.

    B2: Layer-wise idle-first hardware-aware greedy
        Recomputes assignments independently at each layer (fully myopic).

    B3: Sticky layer-wise idle-first hardware-aware greedy
        Same as B2 but qubits only move if the preferred tech differs clearly
        from the previous layer AND has remaining capacity (temporal stability).

Uniform interface for all baselines:

    baseline_bX(rep, caps, config, K) -> List[torch.Tensor]

    Args:
        rep    : CircuitRepresentation
        caps   : torch.Tensor [K] — capacity per technology (int or float)
        config : dict from load_cost_config  (raw cost config with "techs" key)
        K      : int — number of technologies

    Returns:
        List[T] of LongTensor [N] — hard assignment per layer.
        Same format as enforce_capacity_sequence output.
        All assignments are guaranteed capacity-feasible (sum per tech <= cap).
        In rare overflow edge cases (N > sum(caps)), qubits spill to least-loaded.

Bug fixes applied vs. initial version:
    Bug 1 (B1): Three-stage pass structure now correctly operates on the
        *unassigned* set at each stage. Pass 2 can be a no-op when
        best_T2 == best_f2q (e.g. TP2/SC+TI where TI dominates both
        criteria) -- this is correct behaviour, not a bug to work around.
    Bug 2 (all): Spillover is now criterion-aware. _greedy_fill accepts
        pref_ranks: List[List[int]] (full ranked tech list per group) instead
        of a single preferred tech index. Spillover tries techs in ranked
        quality order, not arbitrary index order. B3's mover fallback uses
        the same ranked order. Correct for any K >= 2.
"""

import torch
import numpy as np
from typing import Dict, List, Set, Tuple

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.circuit_representation import CircuitRepresentation


# =============================================================================
# Technology ranking helper
# =============================================================================

def rank_techs_by(config: dict, criterion: str) -> List[int]:
    """
    Return tech indices sorted best-first for the given criterion.

    criterion:
        'T2'  -- longer coherence time is better
        'f2q' -- higher 2Q gate fidelity is better
        'f1q' -- higher 1Q gate fidelity is better

    Works for any K >= 2 and is automatically correct for any TP config.
    """
    techs = config["techs"]
    if criterion == "T2":
        scores = [t["coherence"]["T2"] for t in techs]
    elif criterion == "f2q":
        scores = [t["gate_fidelity"]["f2q"] for t in techs]
    elif criterion == "f1q":
        scores = [t["gate_fidelity"]["f1q"] for t in techs]
    else:
        raise ValueError(f"rank_techs_by: unknown criterion '{criterion}'. "
                         f"Use 'T2', 'f2q', or 'f1q'.")
    return sorted(range(len(techs)), key=lambda k: scores[k], reverse=True)


# =============================================================================
# Shared helpers
# =============================================================================

def _qubit_activity(
    rep: CircuitRepresentation,
) -> Tuple[List[Set[int]], List[Set[int]], List[Set[int]]]:
    """
    Classify each qubit at each layer as exactly one of: idle, 2Q-active, 1Q-only.

    Returns:
        idle_per_layer  : List[T] of sets -- qubits not in any gate
        twoq_per_layer  : List[T] of sets -- qubits in at least one 2Q gate
        oneq_per_layer  : List[T] of sets -- qubits in 1Q gates only (no 2Q gate)

    A qubit in both a 1Q and 2Q gate in the same layer is counted as 2Q-active.
    """
    idle_list = []
    twoq_list = []
    oneq_list = []
    N = rep.num_qubits

    for layer in rep.layers:
        twoq: Set[int] = set()
        oneq: Set[int] = set()
        for gate_name, qargs in layer.gates:
            if len(qargs) == 2:
                twoq.add(qargs[0])
                twoq.add(qargs[1])
            elif len(qargs) == 1:
                oneq.add(qargs[0])
        oneq_only = oneq - twoq
        active    = twoq | oneq
        idle      = set(range(N)) - active

        idle_list.append(idle)
        twoq_list.append(twoq)
        oneq_list.append(oneq_only)

    return idle_list, twoq_list, oneq_list


def _greedy_fill(
    groups:     List[List[int]],
    pref_ranks: List[List[int]],
    caps_int:   List[int],
    K:          int,
) -> Dict[int, int]:
    """
    Greedy capacity-aware assignment.

    Processes groups in order. Within each group, tries techs in the given
    ranked order (best criterion first) until a tech with remaining capacity
    is found. Last-resort overflow assigns to the least-loaded tech.

    pref_ranks[i] is the full ranked list of tech indices for group i,
    e.g. rank_techs_by(config, 'T2'). Spillover tries techs in that order,
    not in arbitrary index order -- correct for any K >= 2.

    Returns: dict {qubit_index: tech_index} covering every qubit in all groups.
    """
    tech_counts = [0] * K
    assignment: Dict[int, int] = {}

    for group_qubits, ranks in zip(groups, pref_ranks):
        for q in group_qubits:
            placed = False
            for k in ranks:
                if tech_counts[k] < caps_int[k]:
                    assignment[q] = k
                    tech_counts[k] += 1
                    placed = True
                    break
            if not placed:
                # True overflow (N > sum(caps)): assign to least loaded
                k = min(range(K), key=lambda kk: tech_counts[kk])
                assignment[q] = k
                tech_counts[k] += 1

    return assignment


# =============================================================================
# B1: Static idle-first hardware-aware greedy
# =============================================================================

def baseline_b1(
    rep:    CircuitRepresentation,
    caps:   torch.Tensor,
    config: dict,
    K:      int,
) -> List[torch.Tensor]:
    """
    Single fixed mapping for the entire circuit. Three sequential passes, each
    operating only on qubits not yet assigned by a previous pass.

    Pass 1 -- idle-heavy qubits -> best-T2 technology (fill to capacity).
    Pass 2 -- remaining qubits, sorted by 2Q count -> best-f2q technology.
              No-op when best_T2 == best_f2q (that tech is already full); the
              remaining qubits then fall through to Pass 3. This is correct:
              when one technology dominates both criteria, Pass 2 being empty
              is the right answer.
    Pass 3 -- remaining qubits, sorted by 1Q count -> best-f1q, with
              criterion-aware spillover through rank_techs_by('f1q').

    The same assignment vector is repeated for all T layers.
    """
    T = len(rep.layers)
    N = rep.num_qubits
    caps_int = [int(c.item()) for c in caps]

    ranks_T2  = rank_techs_by(config, "T2")
    ranks_f2q = rank_techs_by(config, "f2q")
    ranks_f1q = rank_techs_by(config, "f1q")

    best_T2  = ranks_T2[0]
    best_f2q = ranks_f2q[0]

    # Accumulate activity counts per qubit over all layers
    idle_count = np.zeros(N, dtype=int)
    twoq_count = np.zeros(N, dtype=int)
    oneq_count = np.zeros(N, dtype=int)

    idle_per, twoq_per, oneq_per = _qubit_activity(rep)
    for t in range(T):
        for q in idle_per[t]:  idle_count[q] += 1
        for q in twoq_per[t]:  twoq_count[q] += 1
        for q in oneq_per[t]:  oneq_count[q] += 1

    tech_counts = [0] * K
    assignment  = [-1] * N

    # --- Pass 1: most-idle qubits -> best_T2 until that tech is full ---
    idle_sorted = sorted(range(N), key=lambda q: idle_count[q], reverse=True)
    for q in idle_sorted:
        if tech_counts[best_T2] < caps_int[best_T2]:
            assignment[q] = best_T2
            tech_counts[best_T2] += 1

    # --- Pass 2: remaining qubits sorted by 2Q count -> best_f2q ---
    # Only operates on qubits not assigned in Pass 1.
    # No-op when best_T2 == best_f2q (tech already full from Pass 1).
    unassigned = [q for q in range(N) if assignment[q] == -1]
    twoq_sorted = sorted(unassigned, key=lambda q: twoq_count[q], reverse=True)
    for q in twoq_sorted:
        if tech_counts[best_f2q] < caps_int[best_f2q]:
            assignment[q] = best_f2q
            tech_counts[best_f2q] += 1

    # --- Pass 3: remaining qubits sorted by 1Q count -> best_f1q with spillover ---
    # Spillover uses criterion-ranked order, not arbitrary index order.
    unassigned = [q for q in range(N) if assignment[q] == -1]
    oneq_sorted = sorted(unassigned, key=lambda q: oneq_count[q], reverse=True)
    for q in oneq_sorted:
        placed = False
        for k in ranks_f1q:
            if tech_counts[k] < caps_int[k]:
                assignment[q] = k
                tech_counts[k] += 1
                placed = True
                break
        if not placed:
            k = min(range(K), key=lambda kk: tech_counts[kk])
            assignment[q] = k
            tech_counts[k] += 1

    ha = torch.tensor(assignment, dtype=torch.long)
    return [ha.clone() for _ in range(T)]


# =============================================================================
# B2: Layer-wise idle-first hardware-aware greedy
# =============================================================================

def baseline_b2(
    rep:    CircuitRepresentation,
    caps:   torch.Tensor,
    config: dict,
    K:      int,
) -> List[torch.Tensor]:
    """
    Fully myopic dynamic scheduler: recomputes assignments independently at
    each layer with no memory of previous layers.

    Per-layer rule (applied greedily, in priority order):
      1. Idle qubits      -> best-T2 technology
      2. 2Q-active qubits -> best-f2q technology
      3. 1Q-only qubits   -> best-f1q technology
    Spillover tries techs in criterion-ranked order if the preferred is full.
    """
    T = len(rep.layers)
    N = rep.num_qubits
    caps_int = [int(c.item()) for c in caps]

    ranks_T2  = rank_techs_by(config, "T2")
    ranks_f2q = rank_techs_by(config, "f2q")
    ranks_f1q = rank_techs_by(config, "f1q")

    idle_per, twoq_per, oneq_per = _qubit_activity(rep)

    hard_assignments = []
    for t in range(T):
        idle_q = sorted(idle_per[t])
        twoq_q = sorted(twoq_per[t])
        oneq_q = sorted(oneq_per[t])

        assign_dict = _greedy_fill(
            groups     = [idle_q, twoq_q, oneq_q],
            pref_ranks = [ranks_T2, ranks_f2q, ranks_f1q],
            caps_int   = caps_int,
            K          = K,
        )
        assignment = [assign_dict[q] for q in range(N)]
        hard_assignments.append(torch.tensor(assignment, dtype=torch.long))

    return hard_assignments


# =============================================================================
# B3: Sticky layer-wise idle-first hardware-aware greedy
# =============================================================================

def baseline_b3(
    rep:    CircuitRepresentation,
    caps:   torch.Tensor,
    config: dict,
    K:      int,
) -> List[torch.Tensor]:
    """
    Sticky variant of B2: adds temporal persistence to avoid unnecessary moves.

    Layer 0: plain B2 (no previous assignment available).

    Subsequent layers:
      - Stayers (preferred == previous): locked in place first, consuming
        capacity before movers are processed.
      - Movers (preferred != previous):
          1. Try preferred tech (best by criterion).
          2. Try staying in previous tech (stickiness).
          3. Try remaining techs in criterion-ranked order.
          4. Overflow: least-loaded tech.
      Fallback uses criterion-ranked order, not arbitrary index or load order.
    """
    T = len(rep.layers)
    N = rep.num_qubits
    caps_int = [int(c.item()) for c in caps]

    ranks_T2  = rank_techs_by(config, "T2")
    ranks_f2q = rank_techs_by(config, "f2q")
    ranks_f1q = rank_techs_by(config, "f1q")

    best_T2  = ranks_T2[0]
    best_f2q = ranks_f2q[0]
    best_f1q = ranks_f1q[0]

    idle_per, twoq_per, oneq_per = _qubit_activity(rep)

    def preferred_tech_at(q: int, t: int) -> int:
        if q in idle_per[t]: return best_T2
        if q in twoq_per[t]: return best_f2q
        return best_f1q

    def criterion_ranks_at(q: int, t: int) -> List[int]:
        """Full ranked tech list for qubit q at layer t based on its activity."""
        if q in idle_per[t]: return ranks_T2
        if q in twoq_per[t]: return ranks_f2q
        return ranks_f1q

    hard_assignments: List[torch.Tensor] = []
    prev_ha: List[int] = []

    for t in range(T):
        if t == 0:
            # Layer 0: plain B2
            idle_q = sorted(idle_per[t])
            twoq_q = sorted(twoq_per[t])
            oneq_q = sorted(oneq_per[t])
            assign_dict = _greedy_fill(
                groups     = [idle_q, twoq_q, oneq_q],
                pref_ranks = [ranks_T2, ranks_f2q, ranks_f1q],
                caps_int   = caps_int,
                K          = K,
            )
            assignment = [assign_dict[q] for q in range(N)]
        else:
            tech_counts = [0] * K
            assignment  = [-1] * N

            pref = [preferred_tech_at(q, t) for q in range(N)]
            prev = prev_ha

            stayers = [q for q in range(N) if pref[q] == prev[q]]
            movers  = [q for q in range(N) if pref[q] != prev[q]]

            # Lock stayers first so movers cannot displace them
            for q in stayers:
                k = prev[q]
                assignment[q] = k
                tech_counts[k] += 1

            # Movers: preferred -> stay-in-prev -> criterion-ranked fallback -> overflow
            for q in sorted(movers):
                k_pref  = pref[q]
                k_prev  = prev[q]
                q_ranks = criterion_ranks_at(q, t)

                # Build try-order: preferred first, then stay, then rest of ranks
                try_order: List[int] = [k_pref, k_prev]
                for k in q_ranks:
                    if k not in try_order:
                        try_order.append(k)

                placed = False
                for k in try_order:
                    if tech_counts[k] < caps_int[k]:
                        assignment[q] = k
                        tech_counts[k] += 1
                        placed = True
                        break
                if not placed:
                    k = min(range(K), key=lambda kk: tech_counts[kk])
                    assignment[q] = k
                    tech_counts[k] += 1

        ha = torch.tensor(assignment, dtype=torch.long)
        hard_assignments.append(ha)
        prev_ha = assignment

    return hard_assignments
