# src/clustering_head.py

"""
MOSAIC Scheduler — Clustering Head.

Converts per-qubit GNN embeddings h_t [N, H] into soft technology
assignment probabilities P_t [N, K].

Architecture per layer t:
  1. Per-qubit MLP projection  [N, H] -> [N, H]
     Learns a task-specific nonlinear feature space for the assignment
     decision, separate from the GNN's general-purpose representation.

  2. Cosine similarity against L2-normalised prototypes  [N, H] x [K, H] -> [N, K]
     Decouples prototype direction (what each technology "looks like")
     from assignment sharpness (controlled by temperature alone).

  3. Sparse neighbor-logit coordination
     One round of edge-restricted message passing on the K-dimensional
     logits. Each qubit's logits are refined by averaging its graph
     neighbors' logits and adding a learned fraction (alpha). This gives
     interacting qubits a direct short-circuit path to coordinate
     co-assignment, complementing the longer GNN gradient path.

  4. Temperature-scaled softmax with epoch annealing
     T_init (exploratory) -> T_min (sharp) via exponential decay.

Design choices:
  - No instance normalisation: the GRU's LayerNorm output is already
    well-scaled. A second normalisation would destroy magnitude information
    and the GRU LayerNorm's learned affine transform (see issue analysis).
  - Prototypes L2-normalised in forward(): scores are pure cosine similarity,
    gradients are projected onto the tangent plane automatically.
  - Orthogonal prototype initialisation on the unit sphere ensures maximal
    initial separation and non-uniform starting logits.
  - Neighbor mixing weight alpha is passed through sigmoid to stay in (0, 1).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


# =============================================================================
# Core: ClusteringHead
# =============================================================================


class ClusteringHead(nn.Module):
    """
    Soft clustering of per-qubit embeddings into K technology clusters.

    Input per layer t:
        h_t:        [N, H]  — GRU hidden states from EvolvingGNN
        edge_index: [2, E]  — backbone graph edges (optional, for neighbor coordination)

    Output per layer t:
        P_t:        [N, K]  — soft technology assignment probabilities
    """

    def __init__(
        self,
        hidden_dim: int,
        num_clusters: int,
        proj_hidden_dim: Optional[int] = None,
        temperature_init: float = 3.0,
        temperature_min: float = 0.5,
        temperature_gamma: float = 0.95,
        neighbor_alpha_init: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters

        # --- Stage 1: Per-qubit MLP projection ---
        # Learns nonlinear decision boundaries for tech assignment
        # (e.g. "high density AND non-local -> all-to-all technology").
        # Applied pointwise — same weights for every qubit, no cross-qubit mixing.
        proj_h = proj_hidden_dim if proj_hidden_dim is not None else hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, proj_h),
            nn.GELU(),
            nn.Linear(proj_h, hidden_dim),
        )

        # --- Stage 2: Learnable prototypes on unit sphere ---
        # Initialised as orthogonal unit vectors for maximal separation.
        # L2-normalised every forward pass so scores = pure cosine similarity.
        proto = torch.empty(num_clusters, hidden_dim)
        if num_clusters <= hidden_dim:
            # Orthogonal init: sample orthogonal matrix, take first K rows
            nn.init.orthogonal_(proto)
        else:
            # More clusters than dims (rare): fall back to random unit vectors
            nn.init.normal_(proto)
        # Normalise to unit sphere
        with torch.no_grad():
            proto = F.normalize(proto, dim=-1)
        self.cluster_prototypes = nn.Parameter(proto)

        # --- Stage 3: Neighbor coordination ---
        # Learned mixing weight, stored as raw logit; sigmoid applied in forward.
        # Init at 0.0 -> sigmoid(0) = 0.5, giving moderate initial mixing.
        self._alpha_logit = nn.Parameter(torch.tensor(float(neighbor_alpha_init)))

        # --- Stage 4: Temperature annealing ---
        self._temperature_init = temperature_init
        self._temperature_min = temperature_min
        self._temperature_gamma = temperature_gamma
        self.register_buffer("temperature", torch.tensor(temperature_init))

    @property
    def alpha(self) -> torch.Tensor:
        """Neighbor mixing weight, bounded in (0, 1)."""
        return torch.sigmoid(self._alpha_logit)

    def forward(
        self,
        h_t: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            h_t:        [N, H] per-qubit embeddings from GRU (already LayerNorm'd)
            edge_index: [2, E] graph edges for this layer (optional)

        Returns:
            P_t: [N, K] soft assignment probabilities
        """
        N = h_t.size(0)

        # Stage 1: Per-qubit nonlinear projection
        z = self.proj(h_t)                             # [N, H]

        # Stage 2: Cosine similarity with L2-normalised prototypes
        z_norm = F.normalize(z, dim=-1)                # [N, H]  unit sphere
        proto_norm = F.normalize(self.cluster_prototypes, dim=-1)  # [K, H]
        logits = z_norm @ proto_norm.t()               # [N, K]  cosine similarity

        # Stage 3: Sparse neighbor-logit coordination
        if edge_index is not None and edge_index.numel() > 0 and edge_index.size(1) > 0:
            logits = self._neighbor_coordinate(logits, edge_index, N)

        # Stage 4: Temperature-scaled softmax
        P_t = torch.softmax(logits / self.temperature, dim=-1)  # [N, K]
        return P_t

    def _neighbor_coordinate(
        self,
        logits: torch.Tensor,
        edge_index: torch.Tensor,
        N: int,
    ) -> torch.Tensor:
        """
        One round of sparse logit averaging over graph neighbors.

        For each qubit u with neighbors {v1, v2, ...}:
            neighbor_avg[u] = mean(logits[vi])
            logits[u] += alpha * neighbor_avg[u]

        Only qubits with at least one neighbor are affected.
        Isolated qubits keep their original logits unchanged.

        Args:
            logits:     [N, K]
            edge_index: [2, E]  (may be directed; we symmetrise)
            N:          number of qubits

        Returns:
            refined logits [N, K]
        """
        K = logits.size(1)
        device = logits.device

        src, dst = edge_index[0], edge_index[1]

        # Accumulate neighbor logits (symmetrise: both directions)
        neighbor_sum = torch.zeros(N, K, device=device, dtype=logits.dtype)
        neighbor_count = torch.zeros(N, 1, device=device, dtype=logits.dtype)
        ones = torch.ones(src.size(0), 1, device=device, dtype=logits.dtype)

        # dst <- src direction
        neighbor_sum.index_add_(0, dst, logits[src])
        neighbor_count.index_add_(0, dst, ones)

        # src <- dst direction (symmetrise)
        neighbor_sum.index_add_(0, src, logits[dst])
        neighbor_count.index_add_(0, src, ones)

        # Average (isolated qubits: count=0 -> no change)
        has_neighbors = neighbor_count > 0                         # [N, 1]
        neighbor_avg = neighbor_sum / neighbor_count.clamp(min=1)  # [N, K]

        # Mix: only add where qubit has neighbors
        alpha = self.alpha  # scalar in (0, 1)
        refinement = alpha * neighbor_avg * has_neighbors.float()
        return logits + refinement

    def set_epoch(self, epoch: int) -> None:
        """
        Update annealed temperature.

        Exponential decay:
            T(e) = max(T_min, T_init * gamma^e)

        Early training: high T -> soft exploratory assignments.
        Late training:  low T -> sharp decisive assignments.
        """
        e = int(epoch)
        with torch.no_grad():
            t_new = max(
                self._temperature_min,
                self._temperature_init * (self._temperature_gamma ** e),
            )
            self.temperature.fill_(float(t_new))


# =============================================================================
# Convenience wrapper: apply ClusteringHead to a whole sequence of layers
# =============================================================================


class SegmentClustering(nn.Module):
    """
    Apply ClusteringHead to each layer in a sequence.

    Input:
        h_seq:      list of [N, H]  — GRU embeddings per layer
        graphs:     list of PyG Data (optional) — for edge_index per layer

    Output:
        P_seq:      list of [N, K]  — soft assignments per layer
    """

    def __init__(
        self,
        hidden_dim: int,
        num_clusters: int,
        proj_hidden_dim: Optional[int] = None,
        temperature_init: float = 3.0,
        temperature_min: float = 0.5,
        temperature_gamma: float = 0.95,
        neighbor_alpha_init: float = 0.0,
    ):
        super().__init__()
        self.head = ClusteringHead(
            hidden_dim=hidden_dim,
            num_clusters=num_clusters,
            proj_hidden_dim=proj_hidden_dim,
            temperature_init=temperature_init,
            temperature_min=temperature_min,
            temperature_gamma=temperature_gamma,
            neighbor_alpha_init=neighbor_alpha_init,
        )

    def forward(
        self,
        h_seq: List[torch.Tensor],
        graphs: Optional[List[Data]] = None,
    ) -> List[torch.Tensor]:
        """
        Args:
            h_seq:  list of T tensors, each [N, H]
            graphs: list of T PyG Data objects (optional).
                    If provided, edge_index from each is passed to ClusteringHead
                    for neighbor-logit coordination.

        Returns:
            P_seq: list of T tensors, each [N, K]
        """
        if graphs is not None:
            assert len(graphs) == len(h_seq), (
                f"graphs ({len(graphs)}) and h_seq ({len(h_seq)}) length mismatch"
            )
            return [
                self.head(h_t, edge_index=g.edge_index)
                for h_t, g in zip(h_seq, graphs)
            ]
        else:
            # Fallback: no neighbor coordination (e.g. during standalone testing)
            return [self.head(h_t) for h_t in h_seq]

    def set_epoch(self, epoch: int) -> None:
        """Forward epoch update to the head for temperature annealing."""
        self.head.set_epoch(epoch)
