import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

class QubitGNNEncoder(nn.Module):
    def __init__(self, in_dim_node: int, in_dim_edge: int,
                 hidden_dim: int = 32, out_dim: int = 32, heads: int = 4):
        super().__init__()
        # First GATv2 layer: from input node dim -> hidden_dim
        self.conv1 = GATv2Conv(
            in_channels=in_dim_node,
            out_channels=hidden_dim,
            heads=heads,
            concat=False,      # keep hidden_dim, not hidden_dim * heads
            edge_dim=in_dim_edge,
        )
        self.relu = nn.ReLU()
        # Second GATv2 layer: from hidden_dim -> out_dim (embedding size)
        self.conv2 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=out_dim,
            heads=1,
            concat=False,
            edge_dim=in_dim_edge,
        )

    def forward(self, x, edge_index, edge_attr):
        """
        x: [num_nodes, in_dim_node]
        edge_index: [2, num_edges]
        edge_attr: [num_edges, in_dim_edge]
        returns: z [num_nodes, out_dim]
        """
        h = self.conv1(x, edge_index, edge_attr)
        h = self.relu(h)
        z = self.conv2(h, edge_index, edge_attr)
        return z
