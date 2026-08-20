"""
Experimental EFCL-aware capacity hardening for MOSAIC.

This module is intentionally separate from utils/inference_utils.py so the
production confidence-based hardener remains untouched during the experiment.

Algorithm
---------
1. Start from row-wise argmax assignments for every layer (same starting point
   as the current hardener).
2. Visit only layers whose argmax assignment violates a technology capacity.
3. While a layer is infeasible, enumerate legal single-qubit moves that reduce
   overflow by one. Optionally restrict the enumeration to the `candidate_pool`
   moves with the smallest soft preference loss.
4. For each candidate, score the ENTIRE schedule with the actual TotalCost.
   Only the current layer differs, so this automatically accounts for:
      - local execution / idle / communication effects,
      - movement from the previous layer,
      - movement into the next layer,
      - and any downstream ASAP-timing consequences.
5. Commit the lowest-EFCL move and repeat until the layer is feasible.

This is a greedy repair, not a general schedule optimiser: it only performs
moves required to restore capacity feasibility and never changes an already
feasible raw-argmax layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class CostAwareRepairDiagnostics:
    candidate_evaluations: int = 0
    repaired_layers: int = 0
    committed_moves: int = 0
    max_initial_overflow: int = 0


def _one_hot(a: torch.Tensor, K: int, *, dtype=torch.float32) -> torch.Tensor:
    a = a.long()
    N = int(a.numel())
    out = torch.zeros(N, K, dtype=dtype, device=a.device)
    out[torch.arange(N, device=a.device), a] = 1.0
    return out


def _counts(a: torch.Tensor, K: int) -> torch.Tensor:
    return torch.stack([(a == k).sum() for k in range(K)]).long()


def _total_overflow(counts: torch.Tensor, caps: torch.Tensor) -> int:
    return int(torch.relu(counts - caps).sum().item())


def _assert_feasible(assignments: List[torch.Tensor], caps: torch.Tensor, K: int) -> None:
    for t, a in enumerate(assignments):
        c = _counts(a.cpu(), K)
        if bool((c > caps.cpu()).any()):
            raise RuntimeError(
                f"Cost-aware hardener failed at layer {t}: counts={c.tolist()}, "
                f"caps={caps.cpu().tolist()}"
            )


def _schedule_cost(
    P_schedule: List[torch.Tensor],
    cost_module,
    segments,
    circuit,
    precomp_stats: Optional[Dict[str, Any]],
) -> float:
    with torch.no_grad():
        out = cost_module(
            P_schedule,
            segments,
            circuit,
            precomp_stats=precomp_stats,
        )
    return float(out["total_cost"].detach().cpu().item())


def cost_aware_enforce_capacity_sequence(
    P_seq: List[torch.Tensor],
    capacities: torch.Tensor,
    *,
    cost_module,
    segments,
    circuit,
    precomp_stats: Optional[Dict[str, Any]] = None,
    candidate_pool: int = 8,
) -> Tuple[List[torch.Tensor], CostAwareRepairDiagnostics]:
    """
    Greedily repair raw argmax assignments using TotalCost as the decision rule.

    Parameters
    ----------
    P_seq:
        Soft assignments, list of [N,K] tensors.
    capacities:
        Per-technology hard capacities [K].
    cost_module / segments / circuit:
        The exact evaluation TotalCost and corresponding circuit objects.
    precomp_stats:
        Optional SegmentStatsExtractor.compute_stats_cpu() result. Strongly
        recommended; it makes repeated candidate scoring much cheaper.
    candidate_pool:
        Maximum number of candidate single-qubit moves scored at each repair
        step. Candidates are pre-ranked by soft preference loss
            P[u,src] - P[u,dst]
        and the smallest losses are retained. Set <=0 to score ALL legal moves.

    Returns
    -------
    (assignments, diagnostics)
        assignments is List[T] of capacity-feasible LongTensor[N].
    """
    if not P_seq:
        return [], CostAwareRepairDiagnostics()

    device = P_seq[0].device
    dtype = P_seq[0].dtype
    N, K = P_seq[0].shape

    caps = capacities.detach().long().to(device)
    if int(caps.sum().item()) < N:
        raise ValueError(
            f"Total capacity {int(caps.sum())} < N={N}; no feasible assignment exists."
        )
    if int(caps.numel()) != K:
        raise ValueError(f"capacities has {caps.numel()} entries but K={K}.")

    # Same starting point as the existing hardener.
    assignments: List[torch.Tensor] = [P.detach().argmax(dim=1).long().clone() for P in P_seq]
    hard_P: List[torch.Tensor] = [_one_hot(a, K, dtype=dtype) for a in assignments]

    diag = CostAwareRepairDiagnostics()

    # Sequential layer order means the cost of a repair at t is evaluated with
    # already-repaired past layers and raw/future layers not yet repaired.
    for t in range(len(assignments)):
        a_t = assignments[t]
        counts = _counts(a_t, K).to(device)
        initial_overflow = _total_overflow(counts, caps)
        if initial_overflow == 0:
            continue

        diag.repaired_layers += 1
        diag.max_initial_overflow = max(diag.max_initial_overflow, initial_overflow)

        safety = 0
        while _total_overflow(counts, caps) > 0:
            safety += 1
            if safety > K * N:
                raise RuntimeError(f"Repair did not converge at layer {t}.")

            legal: List[Tuple[float, int, int, int]] = []
            # tuple: (soft_preference_loss, q, src, dst)
            for src in range(K):
                if int(counts[src].item()) <= int(caps[src].item()):
                    continue

                q_src = (a_t == src).nonzero(as_tuple=True)[0]
                for dst in range(K):
                    if dst == src:
                        continue
                    if int(counts[dst].item()) >= int(caps[dst].item()):
                        continue
                    for q_tensor in q_src:
                        q = int(q_tensor.item())
                        # Raw argmax means this is normally >=0. It is only a
                        # pre-filter/tie-breaker; TotalCost chooses the winner.
                        pref_loss = float(
                            (P_seq[t][q, src] - P_seq[t][q, dst]).detach().cpu().item()
                        )
                        legal.append((pref_loss, q, src, dst))

            if not legal:
                raise RuntimeError(
                    f"No legal overflow-reducing move at layer {t}; "
                    f"counts={counts.tolist()}, caps={caps.tolist()}"
                )

            legal.sort(key=lambda x: (x[0], x[1], x[3]))
            if candidate_pool > 0:
                legal = legal[:candidate_pool]

            best = None
            for pref_loss, q, src, dst in legal:
                cand_a = a_t.clone()
                cand_a[q] = dst
                cand_P_t = _one_hot(cand_a, K, dtype=dtype)

                # Shallow copy is enough: all layers are immutable tensors for
                # this score, and only layer t is replaced.
                cand_schedule = list(hard_P)
                cand_schedule[t] = cand_P_t

                score = _schedule_cost(
                    cand_schedule,
                    cost_module,
                    segments,
                    circuit,
                    precomp_stats,
                )
                diag.candidate_evaluations += 1

                key = (score, pref_loss, q, dst)
                if best is None or key < best[0]:
                    best = (key, cand_a, cand_P_t, src, dst, q)

            assert best is not None
            _, a_t, P_t_best, src, dst, q = best
            assignments[t] = a_t
            hard_P[t] = P_t_best
            counts[src] -= 1
            counts[dst] += 1
            diag.committed_moves += 1

    _assert_feasible(assignments, caps, K)
    return assignments, diag
