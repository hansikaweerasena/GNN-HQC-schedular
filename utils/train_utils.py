# src/train_utils.py

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
    optimizer=None,
    capacity_penalty=None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    One forward pass on a single circuit. Does NOT call optimizer.step().

    The caller is responsible for accumulating loss tensors across circuits in
    a batch and calling backward() + optimizer.step() once per batch. This
    ensures a true batch gradient update rather than a per-circuit update.

    For evaluation, pass optimizer=None and wrap in torch.no_grad(). The
    returned loss will be a scalar tensor with no grad_fn in that case.

    Args:
        evol_model:        EvolvingGNN — produces h_seq from layer_data_list
        cluster_module:    SegmentClustering — maps h_seq -> P_seq
        total_cost_module: TotalCost — differentiable cost given P_seq
        layer_data_list:   list of PyG Data objects, one per layer
        segments:          segment/layer objects expected by TotalCost
        circuit:           CircuitRepresentation
        optimizer:         if None, sets eval mode; if provided, sets train mode.
                           The actual zero_grad/backward/step are NOT performed
                           here — the caller handles those after accumulation.
        capacity_penalty:  optional CapacityPenalty module

    Returns:
        (loss_tensor, per_segment_total, cap_penalty_value)
        loss_tensor: differentiable scalar tensor (caller calls .backward())
    """
    if optimizer is not None:
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
