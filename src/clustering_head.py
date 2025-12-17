# src/clustering_head.py

from typing import List, Tuple
import torch
import torch.nn as nn


class ClusteringHead(nn.Module):
    """
    Soft clustering of per-qubit embeddings into K clusters.

    Input per segment t: h_t [num_qubits, H]
    Output per segment t: P_t [num_qubits, K] (soft assignments)
    """

    def __init__(self, hidden_dim: int, num_clusters: int, temperature: float = 2.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters

        # Learnable cluster prototypes: [K, H]
        self.cluster_prototypes = nn.Parameter(
            torch.randn(num_clusters, hidden_dim)* 0.1
        )
        self.temperature = temperature

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        """
        h_t: [num_qubits, H]
        returns:
            P_t: [num_qubits, K] soft assignments per qubit
        """
        # Per-qubit normalization
        h_mean = h_t.mean(dim=-1, keepdim=True)
        h_std  = h_t.std(dim=-1, keepdim=True) + 1e-6
        h_norm = (h_t - h_mean) / h_std  # [N, H]

        # Similarity: [num_qubits, K]
        scores = h_norm @ self.cluster_prototypes.t()  # [N, K]

        # Temperature-scaled softmax (T > 1 -> softer, less saturated)
        P_t = torch.softmax(scores / self.temperature, dim=-1)
        return P_t  # [num_qubits, K]


class SegmentClustering(nn.Module):
    """
    Convenience wrapper: apply ClusteringHead to a whole sequence of segments.

    Input:  h_seq: list of [num_qubits, H]
    Output: P_seq: list of [num_qubits, K]
    """

    def __init__(self, hidden_dim: int, num_clusters: int, temperature: float = 2.0):
        super().__init__()
        self.head = ClusteringHead(hidden_dim, num_clusters, temperature)

    def forward(self, h_seq: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.head(h_t) for h_t in h_seq]
