# src/pilot_metrics.py

"""
Gate diagnostics for the fixed-N=30 S-vs-R pilot.

Everything here is computed from P_seq under no_grad, once per evaluation pass.
It answers the five pilot gate questions with numbers rather than impressions:

  1. numerically stable          -> sinkhorn residuals (from the head)
  2. hardener burden ~ 0 late    -> `hardener_burden`
  3. EFCL converges              -> logged by the training loop
  4. matches or beats arm R      -> EFCL, compared across arms
  5. schedules dynamic and       -> `transition_frac`, `mean_moved`,
     circuit-dependent              `occupancy` (aggregated across circuits)

Two notes on interpretation.

Hardener burden is meaningful for BOTH arms and is the headline ablation number.
Sinkhorn makes the *soft* P capacity-feasible; it does not make argmax(P)
feasible. Arm R has no structural guarantee at all. The gap between the two
burdens is the quantitative form of "capacity moved from penalised to
structural".

Entropy is NOT comparable to pre-Sinkhorn runs. Where the column constraint
binds, Sinkhorn rows are structurally less sharp than softmax rows at the same
temperature, so old reference values would read a healthy model as collapsed.
It is logged here only for within-arm trends; collapse is judged from the
behavioural metrics.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from utils.inference_utils import enforce_capacity_sequence


@torch.no_grad()
def circuit_diagnostics(
    P_seq: Sequence[torch.Tensor],   # L tensors of [N, K]
    caps: torch.Tensor,              # [K]
) -> Dict[str, float]:
    """
    Diagnostics for one circuit's layer sequence. Returns Python floats: this
    runs once per evaluation pass, not on the training hot path.

    Everything runs on CPU (see the transfer note below), so the Python-float
    reads at the end are free rather than one synchronisation each.

    Remaining cost note: the hardener's own Python loop is O(L*N*K) per circuit
    and is NOT removed by moving to CPU -- it is inherent to the greedy
    confidence-ordered repair. Budget roughly 10 min per training run at 300
    test circuits with eval_every=5. Vectorising it would change the repair
    semantics and therefore the reported burden, so it is left alone.
    """
    L = len(P_seq)
    N, K = P_seq[0].shape

    # Move the whole layer sequence to CPU ONCE, then do everything below on
    # CPU. Two reasons.
    #
    # First, enforce_capacity_sequence moves each layer to CPU internally and
    # its result back to device, so calling it on device tensors costs 2L
    # transfers per circuit (~160 for an 80-layer circuit, ~1.3M per training
    # run at 300 test circuits and eval_every=5). Handing it CPU tensors makes
    # those internal .cpu() calls no-ops. The hardener itself is untouched, so
    # the reported burden is still the real inference-path number.
    #
    # Second, the ~11 float() reads at the end of this function are each a
    # synchronisation on device tensors and are free on CPU.
    P = torch.stack(list(P_seq)).detach().float().cpu()      # [L, N, K]
    caps = caps.detach().float().cpu()

    # --- capacity, before any repair -----------------------------------
    soft_occ = P.sum(dim=1)                            # [L, K]
    soft_overflow = torch.relu(soft_occ - caps).sum(dim=1)     # [L]

    argmax_assign = P.argmax(dim=-1)                   # [L, N]
    counts = torch.zeros(L, K, device=P.device, dtype=P.dtype)
    counts.scatter_add_(1, argmax_assign, torch.ones_like(argmax_assign, dtype=P.dtype))
    argmax_overflow = torch.relu(counts - caps).sum(dim=1)     # [L] qubits over cap

    # --- hardener burden ------------------------------------------------
    # The real inference-time hardener, so the number reported is the one the
    # paper's inference path would actually incur. Called on CPU tensors; the
    # trailing .cpu() is defensive in case the implementation restores a
    # captured device rather than the input's.
    hard = torch.stack([
        a.cpu() for a in enforce_capacity_sequence([P[l] for l in range(L)], caps)
    ])                                                                 # [L, N]
    moved = (hard != argmax_assign).sum(dim=1).to(P.dtype)            # [L]

    # --- temporal dynamics (collapse detection) --------------------------
    if L > 1:
        changed = (hard[1:] != hard[:-1])                              # [L-1, N]
        mean_moved = changed.sum(dim=1).to(P.dtype).mean()
        transition_frac = (changed.any(dim=1)).to(P.dtype).mean()
    else:
        mean_moved = torch.zeros((), device=P.device, dtype=P.dtype)
        transition_frac = torch.zeros((), device=P.device, dtype=P.dtype)

    # --- sharpness (within-arm trend only) -------------------------------
    row_entropy = -(P.clamp_min(1e-12) * P.clamp_min(1e-12).log()).sum(-1).mean()
    top1 = P.max(dim=-1).values
    frac_confident = (top1 > 0.8).to(P.dtype).mean()

    # --- per-technology occupancy after hardening ------------------------
    hard_counts = torch.zeros(L, K, device=P.device, dtype=P.dtype)
    hard_counts.scatter_add_(1, hard, torch.ones_like(hard, dtype=P.dtype))
    occ = (hard_counts / N).mean(dim=0)                                # [K]

    out = {
        "soft_overflow":     float(soft_overflow.mean()),
        "argmax_overflow":   float(argmax_overflow.mean()),
        "hardener_burden":   float(moved.mean()),          # qubits moved per layer
        "hardener_burden_frac": float(moved.mean()) / N,
        "mean_moved":        float(mean_moved),            # qubits changing tech per transition
        "transition_frac":   float(transition_frac),
        "row_entropy":       float(row_entropy),
        "frac_confident":    float(frac_confident),
    }
    for k in range(K):
        out[f"occ_{k}"] = float(occ[k])
    return out


def aggregate_diagnostics(per_circuit: List[Dict[str, float]], K: int) -> Dict[str, float]:
    """
    Mean over circuits, plus the cross-circuit occupancy spread.

    `occ_std` is the collapse test that the temporal metrics cannot provide: a
    policy that varies across layers but produces the same partition for every
    circuit is still collapsed. Near-zero occ_std with healthy transition_frac
    means the model learned a circuit-independent rule.
    """
    if not per_circuit:
        return {}
    keys = per_circuit[0].keys()
    agg = {k: float(sum(d[k] for d in per_circuit) / len(per_circuit)) for k in keys}

    occ = torch.tensor([[d[f"occ_{k}"] for k in range(K)] for d in per_circuit])
    agg["occ_std"] = float(occ.std(dim=0, unbiased=False).mean()) if len(per_circuit) > 1 else 0.0
    agg["n_circuits"] = len(per_circuit)
    return agg


METRIC_COLUMNS = [
    "epoch", "arm", "seed",
    "train_efcl", "test_efcl", "train_loss", "test_loss",
    "cap_penalty_train", "cap_penalty_test",
    "T", "cost_tau", "lr", "grad_norm", "alpha", "proto_norm",
    "sinkhorn_row_res", "sinkhorn_col_res",
    "soft_overflow", "argmax_overflow",
    "hardener_burden", "hardener_burden_frac",
    "mean_moved", "transition_frac",
    "row_entropy", "frac_confident", "occ_std", "occ_0", "occ_1",
    "epoch_seconds",
]
