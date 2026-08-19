# utils/train_utils.py

from typing import Dict, List, Tuple, Optional
import torch
from torch_geometric.data import Data


# =============================================================================
# Device transfer helper
# (Defined here — not in the training script — so batch_train_step can use it
#  without a circular import. train_hipergator.py imports this from here.)
# =============================================================================


def transfer_layer_data_list(layer_data_list: List[Data], device: torch.device) -> List[Data]:
    """Move a list of CPU Data objects to device (non-blocking)."""
    if device.type == "cpu":
        return layer_data_list
    return [
        Data(
            x          = d.x.to(device, non_blocking=True),
            edge_index = d.edge_index.to(device, non_blocking=True),
            edge_attr  = d.edge_attr.to(device, non_blocking=True),
        )
        for d in layer_data_list
    ]


# =============================================================================
# Single-circuit forward step
# Kept for evaluation, inference, and debugging — not used in the batched
# training loop. See batch_train_step below for the training path.
# =============================================================================


def train_step(
    evol_model,
    cluster_module,
    total_cost_module,
    layer_data_list: List[Data],
    segments,
    circuit,
    training: bool = True,
    optimizer=None,          # accepted but unused — caller manages optimizer
    capacity_penalty=None,
    precomp_stats=None,      # pre-computed CPU stats from dataset cache
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    One forward pass on a single circuit. Does NOT call optimizer.step().

    Args:
        evol_model:        EvolvingGNN
        cluster_module:    SegmentClustering
        total_cost_module: TotalCost
        layer_data_list:   list of PyG Data objects, one per layer (on device)
        segments:          segment objects expected by TotalCost
        circuit:           CircuitRepresentation
        training:          if True sets train mode, else eval mode
        optimizer:         accepted for backward compat; unused here
        capacity_penalty:  optional CapacityPenalty module
        precomp_stats:     CPU stats dict from SegmentStatsExtractor.compute_stats_cpu().
                           When provided, passed through to TotalCost.forward() to
                           skip re-extraction. Used by evaluate_model().

    Returns:
        (loss_tensor, per_segment_total, cap_penalty_value)
    """
    if training:
        evol_model.train()
        cluster_module.train()
    else:
        evol_model.eval()
        cluster_module.eval()

    h_seq, _ = evol_model(layer_data_list)
    P_seq    = cluster_module(h_seq, graphs=layer_data_list)

    cost_out = total_cost_module(P_seq, segments, circuit, precomp_stats=precomp_stats)
    loss = cost_out["total_cost"]

    cap_penalty_val = 0.0
    if capacity_penalty is not None:
        cap_out = capacity_penalty(P_seq)
        loss = loss + cap_out["penalty"]
        cap_penalty_val = float(cap_out["penalty"].item())   # single circuit: 1 sync

    return loss, cost_out["per_segment_total"].detach(), cap_penalty_val


# =============================================================================
# Batched forward step — used in the main training loop
# GNN + clustering head run in parallel via masked-max across B circuits.
# Cost is computed per-circuit after the batched pass (cheap: tensor arithmetic
# only, no GNN kernel launches).
# =============================================================================


def batch_train_step(
    evol_model,
    cluster_module,
    total_cost_module,
    batch,                   # List of (layer_data_list_cpu, segments, rep, stats_cpu)
    device: torch.device,
    capacity_penalty=None,
    training: bool = True,
    return_P: bool = False,
) -> Tuple[torch.Tensor, float, float]:
    """
    Batched forward over one mini-batch.

    The GNN (MLP + GATv2 + GRU) and the clustering head are processed together
    in one masked-max loop: at each layer step, alive circuits are collated into
    a disjoint PyG batch and processed with a single set of kernel launches.
    Ended circuits are frozen — no ghost updates.

    Cost computation remains per-circuit (it is cheap: simple tensor arithmetic
    on pre-computed stats with no GNN involvement).

    Args:
        evol_model:        EvolvingGNN
        cluster_module:    SegmentClustering (its .head is passed to batch_forward)
        total_cost_module: TotalCost
        batch:             collated list from BucketBatchSampler DataLoader
        device:            target compute device
        capacity_penalty:  optional CapacityPenalty module. MUST be None when
                           the head runs in capacity_mode="sinkhorn" -- capacity
                           is enforced structurally there and must not also
                           appear in the loss.
        training:          sets train/eval mode on both models
        return_P:          also return the per-circuit P_seq list (evaluation
                           only -- holding every P alive costs memory).

    Returns:
        (avg_loss_tensor, avg_cap_penalty_float, avg_efcl_float)
        and, if return_P, a fourth element: List[List[Tensor]] of P_seq.

        avg_loss_tensor is differentiable; caller calls .backward() on it.

        EFCL is returned SEPARATELY from the loss because the two arms optimise
        different objectives: arm R's loss is EFCL + R_cap, arm S's loss is
        EFCL alone. Comparing losses across arms would compare different
        quantities. Every cross-arm comparison -- the gate, the checkpoint
        selection, the plots -- must use EFCL.
    """
    if training:
        evol_model.train()
        cluster_module.train()
    else:
        evol_model.eval()
        cluster_module.eval()

    # Transfer all circuits to device and record true layer counts
    layer_lists  = [transfer_layer_data_list(item[0], device) for item in batch]
    true_lengths = [len(ll) for ll in layer_lists]

    # --- Batched GNN + clustering head (masked-max) ---
    # cluster_module.head is the inner ClusteringHead — passed directly so
    # batch_forward can call it on the [alive*N, H] batched tensor using the
    # shared disjoint edge_index without the SegmentClustering wrapper's
    # per-element Python loop.
    results = evol_model.batch_forward(
        layer_lists,
        true_lengths,
        cluster_head=cluster_module.head,
    )

    # --- Per-circuit cost (cheap: no GNN, just tensor arithmetic) ---
    # Accumulate the logging scalars as DEVICE tensors, not Python floats.
    # Calling .item() per circuit would force one CPU<->GPU synchronisation per
    # circuit per batch (32 for arm S, 64 for arm R at batch_size=32), which
    # serialises exactly the batched execution this loop exists to exploit.
    # One .item() after the loop gives the same number.
    #
    # float64 accumulation so the reported value matches the previous
    # Python-float summation to full precision; the tensors are scalars, so the
    # fp64 cost is nil.
    total_loss: Optional[torch.Tensor] = None
    total_cap_t: Optional[torch.Tensor] = None
    total_efcl_t: Optional[torch.Tensor] = None
    all_P: List[List[torch.Tensor]] = []

    for b, item in enumerate(batch):
        _, segments, rep, stats_cpu = item
        P_seq = results[b]["P_seq"]

        # Pass pre-computed CPU stats so TotalCost skips re-extraction.
        # stats_cpu was built by SegmentStatsExtractor.compute_stats_cpu()
        # during dataset __getitem__ and cached there. TotalCost.forward()
        # transfers it to device via _transfer_stats_to_device() — a cheap
        # tensor .to() call — instead of re-running the full NetworkX/gamma
        # pipeline on every training step.
        cost_out = total_cost_module(P_seq, segments, rep, precomp_stats=stats_cpu)
        loss     = cost_out["total_cost"]

        e = loss.detach().double()
        total_efcl_t = e if total_efcl_t is None else total_efcl_t + e

        if capacity_penalty is not None:
            cap_out  = capacity_penalty(P_seq)
            penalty  = cap_out["penalty"]
            loss     = loss + penalty
            c = penalty.detach().double()
            total_cap_t = c if total_cap_t is None else total_cap_t + c

        total_loss = loss if total_loss is None else total_loss + loss

        if return_P:
            all_P.append([p.detach() for p in P_seq])

    B = len(batch)
    # Exactly two synchronisations per batch (one if there is no penalty),
    # regardless of batch size.
    avg_efcl = float(total_efcl_t.item()) / B if total_efcl_t is not None else 0.0
    avg_cap  = float(total_cap_t.item()) / B if total_cap_t is not None else 0.0

    if return_P:
        return total_loss / B, avg_cap, avg_efcl, all_P
    return total_loss / B, avg_cap, avg_efcl
