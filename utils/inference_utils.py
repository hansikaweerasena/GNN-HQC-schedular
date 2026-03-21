# src/inference_utils.py

"""
Inference-time utilities for hardening soft scheduler assignments
into discrete, capacity-feasible technology assignments.

Used in benchmarking/evaluation scripts (not during training).
"""

from typing import List
import torch


def enforce_capacity(
    P_ell: torch.Tensor,
    capacities: torch.Tensor,
) -> torch.Tensor:
    """Harden soft assignments into discrete, capacity-feasible assignments.

    Algorithm:
      1. Argmax to get initial hard assignments.
      2. For each over-capacity technology k:
         - Sort its assigned qubits by P_ell[u, k] ascending (least confident first).
         - Reassign the lowest-confidence qubits to the best technology that
           still has remaining capacity, updating counts after each move.
      3. Repeat until all capacities are satisfied.

    Args:
        P_ell: [N, K] soft assignment probabilities for one layer.
        capacities: [K] integer capacities per technology.

    Returns:
        assignments: [N] integer tensor of technology indices, with
                     |{u : assignments[u] == k}| <= capacities[k] for all k.
    """
    N, K = P_ell.shape
    device = P_ell.device

    # Total capacity must accommodate all qubits
    total_cap = int(capacities.sum().item())
    if total_cap < N:
        raise ValueError(
            f"Total capacity ({total_cap}) < number of qubits ({N}). "
            f"No feasible assignment exists."
        )

    # Work on CPU for the greedy loop (small N, not performance-critical)
    probs = P_ell.detach().float().cpu()       # [N, K]
    caps = capacities.detach().long().cpu()     # [K]
    assignments = probs.argmax(dim=1)           # [N]

    # Running counts — kept in sync throughout the inner loop
    counts = torch.zeros(K, dtype=torch.long)
    for k in range(K):
        counts[k] = (assignments == k).sum()

    # Iterative repair: keep fixing until feasible
    max_iters = K * N  # safety bound, should converge much faster
    for _ in range(max_iters):
        # Find a tech that exceeds capacity
        violations = (counts > caps).nonzero(as_tuple=True)[0]
        if len(violations) == 0:
            break  # all feasible

        k_over = int(violations[0].item())
        excess = int(counts[k_over].item() - caps[k_over].item())

        # Qubits assigned to k_over, sorted by confidence ascending
        mask = (assignments == k_over)
        qubit_indices = mask.nonzero(as_tuple=True)[0]
        confidences = probs[qubit_indices, k_over]
        sorted_order = confidences.argsort()  # ascending
        to_reassign = qubit_indices[sorted_order[:excess]]

        # Reassign each to the best tech with remaining capacity
        for u_idx in to_reassign:
            u = int(u_idx.item())
            p_u = probs[u].clone()

            # Mask out the over-capacity tech
            p_u[k_over] = -float("inf")

            # Also mask out any other tech already at capacity
            for k in range(K):
                if k != k_over and counts[k] >= caps[k]:
                    p_u[k] = -float("inf")

            runner_up = int(p_u.argmax().item())
            assignments[u] = runner_up

            # Update running counts immediately
            counts[k_over] -= 1
            counts[runner_up] += 1

    return assignments.to(device)


def enforce_capacity_sequence(
    P_seq: List[torch.Tensor],
    capacities: torch.Tensor,
) -> List[torch.Tensor]:
    """Apply enforce_capacity to every layer in a sequence.

    Args:
        P_seq: list of [N, K] soft assignment tensors (one per layer).
        capacities: [K] integer capacities per technology.

    Returns:
        List of [N] integer tensors with hard, capacity-feasible assignments.
    """
    return [enforce_capacity(P_ell, capacities) for P_ell in P_seq]
