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
        cap_penalty_val = float(cap_out["penalty"].item())

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
) -> Tuple[torch.Tensor, float]:
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
        capacity_penalty:  optional CapacityPenalty module
        training:          sets train/eval mode on both models

    Returns:
        (avg_loss_tensor, avg_cap_penalty_float)
        avg_loss_tensor is differentiable; caller calls .backward() on it.
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
    total_loss: Optional[torch.Tensor] = None
    total_cap  = 0.0

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

        if capacity_penalty is not None:
            cap_out  = capacity_penalty(P_seq)
            loss     = loss + cap_out["penalty"]
            total_cap += float(cap_out["penalty"].item())

        total_loss = loss if total_loss is None else total_loss + loss

    B = len(batch)
    return total_loss / B, total_cap / B
