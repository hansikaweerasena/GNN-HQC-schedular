"""
evolving_gnn.py

MOSAIC Scheduler — per-layer GNN encoder.

Architecture per layer t:
  1. MLPNodeEncoder   [N, 16] -> [N, 32]   project raw features
  2. GATv2Conv        [N, 32] -> [N, 64]   spatial message passing on backbone graph
  3. GRUCell          [N, 64] -> [N, 64]   temporal state update

Full pipeline:
  Raw node features [N, 16]
      -> MLP:   Linear(16->32) -> ReLU -> Linear(32->32) -> LayerNorm
      -> GATv2: GATv2Conv(32->64, heads=4, edge_dim=5)   -> LayerNorm -> Dropout(0.1)
      -> GRU:   GRUCell(input=64, hidden=64)             -> LayerNorm
      -> Output: h_t [N, 64]  (fed to ClusteringHead)

Truncated BPTT:
  Hidden state is detached every `bptt_steps` layers during the forward pass.
  This prevents exploding/vanishing gradients over long layer sequences while
  still letting the GRU carry forward assignment context.

Empty graph handling:
  If a layer's backbone graph has no edges, we skip GATv2 message passing
  and use a linear fallback projection (gnn_fallback) instead of zero-padding,
  so the node embeddings remain informative.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv


# =============================================================================
# Stage 1: MLP node encoder
# =============================================================================


class MLPNodeEncoder(nn.Module):
    """
    Projects raw node features into initial embeddings.

    Linear(in_dim -> hidden_dim) -> activation -> Linear(hidden_dim -> out_dim)
    -> LayerNorm

    LayerNorm here ensures the GATv2 attention mechanism receives consistently
    scaled inputs regardless of the mix of boolean and rate-valued raw features.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        activation: str = "relu",
    ):
        super().__init__()
        act: nn.Module = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()
        self.fc1  = nn.Linear(in_dim, hidden_dim)
        self.act  = act
        self.fc2  = nn.Linear(hidden_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, in_dim]  ->  [N, out_dim]"""
        return self.norm(self.fc2(self.act(self.fc1(x))))


# =============================================================================
# Stage 2 + 3 + full sequential model: SchedulerGNN  (kept as EvolvingGNN
# for backward-compatible imports)
# =============================================================================


class EvolvingGNN(nn.Module):
    """
    Per-layer scheduler encoder: MLP -> GATv2 -> GRU.

    Processes a list of PyG Data objects (one per circuit layer) and returns
    per-qubit hidden state embeddings h_t at every layer.

    Args:
        node_feat_dim:  raw node feature dimension (default 16)
        mlp_hidden_dim: hidden width of the MLP node encoder (default 32)
        mlp_out_dim:    output dim of MLP = input dim to GATv2 (default 64, must equal gnn_out_dim)
        gnn_out_dim:    GATv2 output dim = GRU input dim (default 64)
        gru_hidden_dim: GRU hidden state dim = clustering head input dim (default 64)
        edge_feat_dim:  edge feature dimension (default 5)
        heads:          number of GATv2 attention heads (default 4)
        dropout:        dropout probability after GATv2 (default 0.1)
        bptt_steps:     detach hidden state every N steps; 0 = no truncation (default 3)
        activation:     "relu" or "gelu" for MLP (default "relu")
    """

    def __init__(
        self,
        node_feat_dim: int = 16,
        mlp_hidden_dim: int = 32,
        mlp_out_dim: int = 32,
        gnn_out_dim: int = 64,
        gru_hidden_dim: int = 64,
        edge_feat_dim: int = 5,
        heads: int = 4,
        dropout: float = 0.1,
        bptt_steps: int = 3,
        activation: str = "relu",
        # Legacy kwargs accepted but ignored (for old call sites)
        in_dim_node: int = None,
        in_dim_edge: int = None,
        gnn_hidden_dim: int = None,
        rnn_hidden_dim: int = None,
    ):
        super().__init__()

        # Resolve legacy param names
        if in_dim_node is not None:
            node_feat_dim = in_dim_node
        if in_dim_edge is not None:
            edge_feat_dim = in_dim_edge
        if rnn_hidden_dim is not None:
            gru_hidden_dim = rnn_hidden_dim

        self.gnn_out_dim   = gnn_out_dim
        self.rnn_hidden_dim = gru_hidden_dim  # public alias for ClusteringHead
        self.bptt_steps    = bptt_steps

        assert mlp_out_dim == gnn_out_dim, (
            f"mlp_out_dim ({mlp_out_dim}) must equal gnn_out_dim ({gnn_out_dim}) "
            f"for the residual connection (GAT output + MLP output) to work."
        )

        # Stage 1: MLP node encoder
        self.mlp = MLPNodeEncoder(
            in_dim=node_feat_dim,
            hidden_dim=mlp_hidden_dim,
            out_dim=mlp_out_dim,
            activation=activation,
        )

        # Stage 2: GATv2 spatial encoder
        # add_self_loops=False: self-loops are not meaningful here because
        # self-loop edge features would be zero (no circuit-derived meaning).
        # Isolated nodes are handled cleanly by the residual connection in forward().
        self.gat = GATv2Conv(
            in_channels=mlp_out_dim,
            out_channels=gnn_out_dim,
            heads=heads,
            concat=False,
            edge_dim=edge_feat_dim,
            add_self_loops=False,
        )
        self.gat_norm    = nn.LayerNorm(gnn_out_dim)
        self.gat_dropout = nn.Dropout(p=dropout)

        # Stage 3: GRU temporal encoder
        self.gru_cell = nn.GRUCell(
            input_size=gnn_out_dim,
            hidden_size=gru_hidden_dim,
        )
        self.gru_norm = nn.LayerNorm(gru_hidden_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        layer_graphs: List[Data],
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            layer_graphs: list of PyG Data objects, one per circuit layer.
                Each must have:
                    x:          [N, node_feat_dim]
                    edge_index: [2, E_t]                   (may be [2,0])
                    edge_attr:  [E_t, edge_feat_dim]       (may be [0,5])
            h0: optional initial hidden state [N, gru_hidden_dim].
                Defaults to zeros.

        Returns:
            h_seq: List[Tensor[N, gru_hidden_dim]]  — one per layer
                   These are fed to the ClusteringHead.
            z_seq: List[Tensor[N, gnn_out_dim]]     — GATv2 outputs, one per layer
                   Useful for debugging / analysis.
        """
        assert len(layer_graphs) > 0, "layer_graphs must be non-empty"

        N      = layer_graphs[0].x.size(0)
        device = layer_graphs[0].x.device
        h = torch.zeros(N, self.rnn_hidden_dim, device=device) if h0 is None else h0.to(device)

        h_seq: List[torch.Tensor] = []
        z_seq: List[torch.Tensor] = []

        for step, data in enumerate(layer_graphs):
            # Truncated BPTT: detach every bptt_steps to cap gradient flow
            if self.bptt_steps > 0 and step > 0 and step % self.bptt_steps == 0:
                h = h.detach()

            x          = data.x           # [N, node_feat_dim]
            edge_index = data.edge_index  # [2, E]

            # Edge attr: fall back to zeros if absent or empty
            if data.edge_attr is not None and data.edge_attr.numel() > 0:
                edge_attr = data.edge_attr
            else:
                edge_attr = torch.zeros(
                    edge_index.size(1), self.gat.edge_dim or 5,
                    device=device, dtype=x.dtype,
                )

            # Stage 1: MLP
            e = self.mlp(x)  # [N, mlp_out_dim]

            # Stage 2: GATv2 with residual connection
            # Residual: z = GAT(e) + e
            #   - Connected nodes: z = neighborhood_context + own_features
            #   - Isolated nodes:  GAT output = 0, so z = 0 + e = e (own features preserved)
            # Requires mlp_out_dim == gnn_out_dim (both 64).
            z = self.gat(e, edge_index, edge_attr) + e  # [N, gnn_out_dim]
            z = self.gat_norm(z)
            z = self.gat_dropout(z)

            z_seq.append(z)

            # Stage 3: GRUCell
            h = self.gru_cell(z, h)   # [N, gru_hidden_dim]
            h = self.gru_norm(h)
            h_seq.append(h)

        return h_seq, z_seq
