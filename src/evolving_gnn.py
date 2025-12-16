# src/evolving_gnn.py

from typing import List, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv


class SegmentGNN(nn.Module):
    """
    Spatial encoder applied to a single segment graph.
    Input:  x_s, edge_index_s, edge_attr_s
    Output: z_s (per-qubit embedding for that segment)
    """

    def __init__(
        self,
        in_dim_node: int,
        in_dim_edge: int,
        hidden_dim: int = 32,
        out_dim: int = 32,
        heads: int = 4,
    ):
        super().__init__()
        self.conv1 = GATv2Conv(
            in_channels=in_dim_node,
            out_channels=hidden_dim,
            heads=heads,
            concat=False,
            edge_dim=in_dim_edge,
        )
        self.relu = nn.ReLU()
        self.conv2 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=out_dim,
            heads=1,
            concat=False,
            edge_dim=in_dim_edge,
        )

    def forward(self, x, edge_index, edge_attr):
        h = self.conv1(x, edge_index, edge_attr)
        h = self.relu(h)
        z = self.conv2(h, edge_index, edge_attr)
        return z  # [num_nodes, out_dim]


class EvolvingGNN(nn.Module):
    """
    Evolving GNN: per-segment GNN + per-qubit GRU over time.

    For each time step t (segment t):
      - run SegmentGNN on that segment's graph to get z_{:,t}
      - update per-qubit hidden state h_{:,t} = GRU(z_{:,t}, h_{:,t-1})

    Assumes:
      - same number of qubits (nodes) in every segment graph
      - node ordering is consistent (0..num_qubits-1)
    """

    def __init__(
        self,
        in_dim_node: int,
        in_dim_edge: int,
        gnn_hidden_dim: int = 32,
        gnn_out_dim: int = 32,
        rnn_hidden_dim: int = 32,
        heads: int = 4,
    ):
        super().__init__()
        self.gnn = SegmentGNN(
            in_dim_node=in_dim_node,
            in_dim_edge=in_dim_edge,
            hidden_dim=gnn_hidden_dim,
            out_dim=gnn_out_dim,
            heads=heads,
        )
        self.rnn_hidden_dim = rnn_hidden_dim
        # GRUCell operates per qubit, input dim = gnn_out_dim, hidden dim = rnn_hidden_dim
        self.gru_cell = nn.GRUCell(input_size=gnn_out_dim, hidden_size=rnn_hidden_dim)

    def forward(
        self,
        segment_graphs: List[Data],
        h0: Optional[torch.Tensor] = None,
    ):
        """
        segment_graphs: list of PyG Data objects, one per segment.
            Each Data_t must have:
                x: [num_qubits, in_dim_node]
                edge_index: [2, num_edges_t]
                edge_attr: [num_edges_t, in_dim_edge]

        h0: initial hidden state [num_qubits, rnn_hidden_dim] (optional)
            If None, initialized to zeros.

        Returns:
            h_seq: list of [num_qubits, rnn_hidden_dim], one per segment t
            z_seq: list of [num_qubits, gnn_out_dim], one per segment t
        """
        assert len(segment_graphs) > 0, "segment_graphs must be non-empty"

        num_qubits = segment_graphs[0].x.size(0)
        device = segment_graphs[0].x.device

        if h0 is None:
            h = torch.zeros(num_qubits, self.rnn_hidden_dim, device=device)
        else:
            h = h0.to(device)

        h_seq = []
        z_seq = []

        for data in segment_graphs:
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr

            # Spatial embedding for this segment
            z = self.gnn(x, edge_index, edge_attr)  # [num_qubits, gnn_out_dim]
            z_seq.append(z)

            # Temporal update per qubit
            h = self.gru_cell(z, h)  # still [num_qubits, rnn_hidden_dim]
            h_seq.append(h)

        return h_seq, z_seq
