# src/train_utils.py

from typing import List, Tuple
import torch
from torch_geometric.data import Data

def train_step(
    evol_model,
    cluster_module,
    total_cost_module,
    segment_data_list: List[Data],
    segments,
    circuit,
    optimizer=None,  # ← Make optional
) -> Tuple[float, torch.Tensor]:
    """
    One training step on a single circuit (all its segments).
    If optimizer=None, just forward pass (for evaluation).
    """
    evol_model.train()
    cluster_module.train()

    # 1) Forward through evolving GNN
    h_seq, z_seq = evol_model(segment_data_list)  # lists of [N,H] and [N,Z]

    # 2) Clustering: soft assignments per segment
    P_seq = cluster_module(h_seq)                 # list of [N,K]

    # 3) Cost
    cost_out = total_cost_module(P_seq, segments, circuit)
    loss = cost_out["total_cost"]

    # 4) Backprop + update (only if training)
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return float(loss.item()), cost_out["per_segment_total"].detach()

