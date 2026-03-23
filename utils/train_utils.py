# utils/train_utils.py

from typing import List, Tuple, Optional
import torch
from torch_geometric.data import Data


def train_step(
    evol_model,
    cluster_module,
    total_cost_module,
    layer_data_list: List[Data],
    segments,
    circuit,
    training: bool = True,
    optimizer=None,   # kept for backward compatibility — not used in this function
    capacity_penalty=None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    One forward pass on a single circuit.

    Does NOT call optimizer.zero_grad / backward / optimizer.step.
    The caller accumulates loss tensors across circuits in a batch and steps once.

    Args:
        evol_model:        EvolvingGNN — produces h_seq from layer_data_list
        cluster_module:    SegmentClustering — maps h_seq -> P_seq
        total_cost_module: TotalCost — differentiable cost given P_seq
        layer_data_list:   list of PyG Data objects, one per layer (on correct device)
        segments:          segment/layer objects expected by TotalCost
        circuit:           CircuitRepresentation
        training:          True  -> set both modules to train mode
                           False -> set both modules to eval mode
                           Caller controls mode explicitly — no implicit inference
                           from optimizer presence.
        optimizer:         Unused. Accepted for backward compatibility with old call
                           sites that pass optimizer=None for eval. Ignored here.
        capacity_penalty:  optional CapacityPenalty module

    Returns:
        (loss_tensor, per_segment_total, cap_penalty_value)
        loss_tensor: differentiable scalar tensor (caller calls .backward())
    """
    if training:
        evol_model.train()
        cluster_module.train()
    else:
        evol_model.eval()
        cluster_module.eval()

    # 1) Spatial + temporal encoding
    h_seq, z_seq = evol_model(layer_data_list)   # lists of [N, gru_hidden_dim]

    # 2) Soft technology assignments
    P_seq = cluster_module(h_seq, graphs=layer_data_list)  # list of [N, K]

    # 3) Differentiable cost
    cost_out = total_cost_module(P_seq, segments, circuit)
    loss = cost_out["total_cost"]

    # 4) Optional capacity regulariser
    cap_penalty_val = 0.0
    if capacity_penalty is not None:
        cap_out = capacity_penalty(P_seq)
        loss = loss + cap_out["penalty"]
        cap_penalty_val = float(cap_out["penalty"].item())

    return loss, cost_out["per_segment_total"].detach(), cap_penalty_val
