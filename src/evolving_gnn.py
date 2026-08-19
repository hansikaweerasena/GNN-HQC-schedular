"""
evolving_gnn.py

MOSAIC Scheduler — per-layer GNN encoder.

Architecture per layer t:
  1. MLPNodeEncoder   [N, 16] -> [N, 64]   project raw features
  2. GATv2Conv        [N, 64] -> [N, 64]   spatial message passing on backbone graph
                                            + residual connection from MLP output
  3. GRUCell          [N, 64] -> [N, 64]   temporal state update

Full pipeline:
  Raw node features [N, 16]
      -> MLP:   Linear(16->32) -> ReLU -> Linear(32->64) -> LayerNorm
      -> GATv2: GATv2Conv(64->64, heads=4, edge_dim=5) + residual -> LayerNorm -> Dropout(0.1)
      -> GRU:   GRUCell(input=64, hidden=64)             -> LayerNorm
      -> Output: h_t [N, 64]  (fed to ClusteringHead)

Residual connection:
  z = GAT(e, edges) + e
  - Connected nodes: z = neighborhood_context + own_features
  - Isolated nodes:  GAT output = 0, so z = e (own features fully preserved)
  Requires mlp_out_dim == gnn_out_dim (both 64).

Truncated BPTT:
  Hidden state is detached every `bptt_steps` layers during the forward pass.
  This prevents exploding/vanishing gradients over long layer sequences while
  still letting the GRU carry forward assignment context.

Batched forward (batch_forward):
  Processes B circuits in parallel using masked-max over layers.
  At each step t, only circuits with t < T_b are active (alive). Ended circuits
  are frozen — their GRU hidden state is not updated after their final real layer.
  If a ClusteringHead is provided, it is run batched at each step on the alive
  subset using the disjoint graph edge_index, so neighbor coordination never
  crosses circuit boundaries. P_seq is then split back per circuit and returned.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.data import Batch as PyGBatch
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
        mlp_out_dim: int = 64,   # must equal gnn_out_dim for residual connection
        gnn_out_dim: int = 64,
        gru_hidden_dim: int = 64,
        edge_feat_dim: int = 5,
        heads: int = 4,
        dropout: float = 0.1,
        bptt_steps: int = 3,
        activation: str = "relu",
        # Legacy kwargs accepted but remapped (for old call sites)
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

        self.gnn_out_dim    = gnn_out_dim
        self.rnn_hidden_dim = gru_hidden_dim  # public alias used by ClusteringHead
        self.bptt_steps     = bptt_steps
        self.edge_feat_dim  = edge_feat_dim   # stored explicitly — avoids fragile PyG attr access

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
    # Single-circuit forward (unchanged — used for eval / inference)
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

            # Edge attr: fall back to zeros if absent or empty.
            # Use self.edge_feat_dim (stored at init) — avoids fragile access
            # to self.gat.edge_dim which is not stable across PyG versions.
            if data.edge_attr is not None and data.edge_attr.numel() > 0:
                edge_attr = data.edge_attr
            else:
                edge_attr = torch.zeros(
                    edge_index.size(1), self.edge_feat_dim,
                    device=device, dtype=x.dtype,
                )

            # Stage 1: MLP
            e = self.mlp(x)  # [N, mlp_out_dim]

            # Stage 2: GATv2 with residual connection
            z = self.gat(e, edge_index, edge_attr) + e  # [N, gnn_out_dim]
            z = self.gat_norm(z)
            z = self.gat_dropout(z)

            z_seq.append(z)

            # Stage 3: GRUCell
            h = self.gru_cell(z, h)   # [N, gru_hidden_dim]
            h = self.gru_norm(h)
            h_seq.append(h)

        return h_seq, z_seq

    # ------------------------------------------------------------------
    # Batched forward — masked-max over B circuits with optional
    # batched clustering head
    # ------------------------------------------------------------------

    def batch_forward(
        self,
        batch_layer_graphs: List[List[Data]],
        true_lengths: List[int],
        cluster_head=None,
    ) -> List[Dict]:
        """
        Masked-max batched forward over B circuits.

        At each step t, only circuits with t < true_lengths[b] are active.
        Ended circuits are frozen — their GRU hidden state is not updated
        after their final real layer. No ghost updates, no dropped layers.

        If cluster_head (a ClusteringHead instance) is provided, it is run
        batched at each step using the same disjoint graph edge_index as the
        GNN step. The disjoint graph guarantees neighbor coordination never
        crosses circuit boundaries. P_seq is split back per circuit.

        Args:
            batch_layer_graphs: List[List[Data]] — B circuits, each with T_b layers
            true_lengths:       List[int]        — T_b for each circuit b
            cluster_head:       ClusteringHead instance (not SegmentClustering wrapper)
                                Pass cluster_module.head from the training script.

        Returns:
            List of B dicts, each containing:
                "h_seq": List[Tensor[N, H]]  — one tensor per real layer
                "z_seq": List[Tensor[N, D]]  — GATv2 outputs, one per real layer
                "P_seq": List[Tensor[N, K]]  — soft assignments (if cluster_head given)
                         None if cluster_head is None
        """
        B    = len(batch_layer_graphs)
        Lmax = max(true_lengths)
        dev  = batch_layer_graphs[0][0].x.device
        N    = batch_layer_graphs[0][0].x.size(0)   # fixed N across all circuits

        # Per-circuit current hidden state (Python list avoids in-place buffer issues)
        h_current: List[torch.Tensor] = [
            torch.zeros(N, self.rnn_hidden_dim, device=dev) for _ in range(B)
        ]

        # Per-circuit output accumulators
        h_seqs: List[List[torch.Tensor]] = [[] for _ in range(B)]
        z_seqs: List[List[torch.Tensor]] = [[] for _ in range(B)]
        P_seqs: Optional[List[List[torch.Tensor]]] = (
            [[] for _ in range(B)] if cluster_head is not None else None
        )

        for step in range(Lmax):
            # Determine alive circuits: those that still have a real layer at this step
            alive = [b for b in range(B) if step < true_lengths[b]]
            if not alive:
                break  # all circuits finished

            # Truncated BPTT: detach at window boundaries before using h as input.
            # Applied on the global step index (all circuits aligned to step 0),
            # matching the single-circuit forward() behaviour exactly.
            if self.bptt_steps > 0 and step > 0 and step % self.bptt_steps == 0:
                for b in alive:
                    h_current[b] = h_current[b].detach()

            # --- Build disjoint batched graph for alive circuits only ---
            # Batch.from_data_list creates a disjoint union: node indices are
            # offset per graph so edge_index spans [0, |alive|*N). This means
            # GATv2 and ClusteringHead neighbor ops are strictly within each
            # circuit's node range — no cross-circuit mixing is possible.
            alive_graphs = [batch_layer_graphs[b][step] for b in alive]
            batched      = PyGBatch.from_data_list(alive_graphs)

            # Edge attr fallback for empty layers
            if batched.edge_attr is None or batched.edge_attr.numel() == 0:
                batched.edge_attr = torch.zeros(
                    batched.edge_index.size(1), self.edge_feat_dim,
                    device=dev, dtype=batched.x.dtype,
                )

            # Gather hidden states for alive circuits → [|alive|*N, H]
            # torch.stack creates a new tensor from the list; autograd flows
            # back through h_current[b] tensors correctly.
            h_alive = torch.stack([h_current[b] for b in alive]).reshape(
                len(alive) * N, self.rnn_hidden_dim
            )

            # --- Stage 1: MLP ---
            e = self.mlp(batched.x)                                    # [|alive|*N, mlp_out_dim]

            # --- Stage 2: GATv2 + residual ---
            z = self.gat(e, batched.edge_index, batched.edge_attr) + e
            z = self.gat_norm(z)
            z = self.gat_dropout(z)                                    # [|alive|*N, gnn_out_dim]

            # --- Stage 3: GRUCell ---
            h_new = self.gru_cell(z, h_alive)
            h_new = self.gru_norm(h_new)                               # [|alive|*N, H]

            # --- Optional: batched clustering head ---
            # h_new and batched.edge_index share the same |alive|*N node space.
            # Disjoint edge_index guarantees per-circuit neighbor coordination.
            if cluster_head is not None:
                # n_qubits=N is required by the Sinkhorn path: capacity is a
                # per-circuit constraint, so the flat [|alive|*N, K] logit
                # tensor must be reshaped to [|alive|, N, K] before projection.
                # Omitting it would pool capacity across circuits in the batch.
                P_batched = cluster_head(
                    h_new, edge_index=batched.edge_index, n_qubits=N
                )                                                       # [|alive|*N, K]
                P_split = P_batched.reshape(len(alive), N, -1)         # [|alive|, N, K]

            # --- Reshape and write back per circuit ---
            h_split = h_new.reshape(len(alive), N, self.rnn_hidden_dim)
            z_split = z.reshape(len(alive), N, self.gnn_out_dim)

            for i, b in enumerate(alive):
                h_current[b] = h_split[i]          # update hidden state for next step
                h_seqs[b].append(h_split[i])
                z_seqs[b].append(z_split[i])
                if P_seqs is not None:
                    P_seqs[b].append(P_split[i])

        return [
            {
                "h_seq": h_seqs[b],
                "z_seq": z_seqs[b],
                "P_seq": P_seqs[b] if P_seqs is not None else None,
            }
            for b in range(B)
        ]
