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
     logits using a convex blend:
         refined = (1 - alpha) * self_logits + alpha * neighbor_avg
     Alpha is a learned parameter bounded in (0, 1) via sigmoid. The convex
     form preserves logit scale (cosine values stay in [-1, 1]) so
     temperature controls sharpness uniformly across the circuit.

  4. Normalization (capacity_mode) -- the only stage C5 changes:
       "sinkhorn" (default) -- balanced entropic-OT projection onto the
            capacity polytope with (C_total - N) zero-logit dummy rows.
            Capacity becomes structural, so it leaves the loss entirely and
            no longer competes with the physics gradient.
       "softmax" (legacy)   -- temperature-scaled softmax; capacity enforced
            by the CapacityPenalty regularizer in the loss. Retained as the
            matched ablation arm (arm R).
     Both modes consume the SAME annealed temperature T: in sinkhorn mode T
     is the entropic regularisation parameter of the transport problem. Only
     one mode is ever active, so the schedule is never applied twice.

     Stages 1-3 are byte-identical between the two modes. That is the point of
     the compute_logits() / normalize() split: the ablation isolates the
     constraint mechanism and nothing else.

Design choices:
  - No instance normalisation: the GRU's LayerNorm output is already
    well-scaled. A second normalisation would destroy magnitude information
    and the GRU LayerNorm's learned affine transform (see issue analysis).
  - Prototypes L2-normalised in forward(): scores are pure cosine similarity,
    gradients are projected onto the tangent plane automatically.
  - Orthogonal prototype initialisation on the unit sphere ensures maximal
    initial separation and non-uniform starting logits.
  - Neighbor mixing weight alpha stored as logit, sigmoid-bounded to (0, 1).
    The caller specifies the desired initial alpha (e.g. 0.3 = 30% neighbor
    influence); internally converted to logit = log(alpha / (1 - alpha)).
  - The cosine + convex-blend design bounds the logits to [-1, 1]. The Sinkhorn
    iteration budget is chosen against that bound at T_min, so introducing a
    learnable similarity scale here would invalidate it and require rerunning
    the Step 2 convergence smoke test.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from src.sinkhorn import CapacitySinkhorn


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
        neighbor_alpha_init: float = 0.1,
        capacity_mode: str = "sinkhorn",
        caps=None,
        sinkhorn_iters: int = 30,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters

        capacity_mode = str(capacity_mode).lower()
        if capacity_mode not in ("sinkhorn", "softmax"):
            raise ValueError(
                f"capacity_mode must be 'sinkhorn' or 'softmax', got {capacity_mode!r}"
            )
        self.capacity_mode = capacity_mode

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
        # Caller specifies initial alpha in (0, 1) — the fraction of a qubit's
        # logits that comes from its neighbors.  Stored internally as a logit
        # (inverse-sigmoid) so that sigmoid bounds the learned value to (0, 1).
        # Clamp to avoid inf at boundaries.
        alpha_clamped = max(1e-4, min(1.0 - 1e-4, float(neighbor_alpha_init)))
        alpha_logit = torch.log(torch.tensor(alpha_clamped / (1.0 - alpha_clamped)))
        self._alpha_logit = nn.Parameter(alpha_logit)

        # --- Stage 4: temperature annealing (shared by both modes) ---
        self._temperature_init = temperature_init
        self._temperature_min = temperature_min
        self._temperature_gamma = temperature_gamma
        if temperature_init <= 0 or temperature_min <= 0:
            raise ValueError(
                f"temperatures must be strictly positive (T divides the logits): "
                f"init={temperature_init}, min={temperature_min}"
            )
        self.register_buffer("temperature", torch.tensor(temperature_init))
        # Python-float mirror of the buffer. The Sinkhorn path reads T once per
        # layer; reading the device buffer instead would force a CPU<->GPU sync
        # ~400k times over a training run. set_epoch() and load_state_dict()
        # are the only writers.
        self._T: float = float(temperature_init)

        # --- Stage 4: capacity projection ---
        if capacity_mode == "sinkhorn":
            if caps is None:
                raise ValueError("capacity_mode='sinkhorn' requires caps=[K]")
            self.sinkhorn = CapacitySinkhorn(caps, n_iters=sinkhorn_iters)
            if self.sinkhorn.caps.numel() != num_clusters:
                raise ValueError(
                    f"caps has {self.sinkhorn.caps.numel()} entries but "
                    f"num_clusters={num_clusters}"
                )
        else:
            self.sinkhorn = None

    @property
    def alpha(self) -> torch.Tensor:
        """Neighbor mixing weight, bounded in (0, 1)."""
        return torch.sigmoid(self._alpha_logit)

    # ------------------------------------------------------------------
    # Stages 1-3: logits.  Identical in both capacity modes.
    # ------------------------------------------------------------------
    def compute_logits(
        self,
        h_t: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            h_t:        [M, H] per-qubit embeddings from GRU (already LayerNorm'd).
                        M is a flat node count: N for one circuit, or B*N for a
                        fixed-N mini-batch in PyG disjoint-union order.
            edge_index: [2, E] graph edges for this layer (optional). For a
                        batched call this is the disjoint edge_index, so
                        neighbor coordination never crosses circuit boundaries.

        Returns:
            Z: [M, K] assignment logits, bounded to [-1, 1].
        """
        M = h_t.size(0)

        # Stage 1: Per-qubit nonlinear projection
        z = self.proj(h_t)                             # [M, H]

        # Stage 2: Cosine similarity with L2-normalised prototypes
        z_norm = F.normalize(z, dim=-1)                # [M, H]  unit sphere
        proto_norm = F.normalize(self.cluster_prototypes, dim=-1)  # [K, H]
        logits = z_norm @ proto_norm.t()               # [M, K]  cosine similarity

        # Stage 3: Sparse neighbor-logit coordination
        if edge_index is not None and edge_index.numel() > 0 and edge_index.size(1) > 0:
            logits = self._neighbor_coordinate(logits, edge_index, M)

        return logits

    # ------------------------------------------------------------------
    # Stage 4: normalization.  The only stage that differs between modes.
    # ------------------------------------------------------------------
    def normalize(
        self,
        logits: torch.Tensor,
        n_qubits: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits:   [M, K] from compute_logits().
            n_qubits: qubits per circuit. REQUIRED in sinkhorn mode when the
                      call is batched (M = B*N): capacity is a per-circuit
                      constraint, so the flat tensor must be reshaped to
                      [B, N, K] before projection. Without it, capacity mass
                      would be pooled across circuits. None => the whole
                      tensor is one circuit.

        Returns:
            P: [M, K]
        """
        if self.capacity_mode == "softmax":
            return torch.softmax(logits / self.temperature, dim=-1)

        M, K = logits.shape
        if n_qubits is None:
            n_qubits = M
        if M % n_qubits != 0:
            raise ValueError(f"{M} rows is not a multiple of n_qubits={n_qubits}")

        # [M, K] -> [B, N, K] -> project -> [M, K].
        # capacity_sinkhorn operates on the last two dims, so the leading
        # batch dim is handled without a per-circuit Python loop.
        Z = logits.view(-1, n_qubits, K)
        P = self.sinkhorn(Z, T=self._T)   # Python float: no device sync
        return P.reshape(M, K)

    def forward(
        self,
        h_t: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        n_qubits: Optional[int] = None,
    ) -> torch.Tensor:
        """compute_logits() followed by normalize(). See both for arguments."""
        logits = self.compute_logits(h_t, edge_index=edge_index)
        return self.normalize(logits, n_qubits=n_qubits)

    def _neighbor_coordinate(
        self,
        logits: torch.Tensor,
        edge_index: torch.Tensor,
        N: int,
    ) -> torch.Tensor:
        """
        One round of sparse convex blending over graph neighbors.

        For each qubit u with neighbors {v1, v2, ...}:
            neighbor_avg[u] = mean(logits[vi])
            refined[u] = (1 - alpha) * logits[u] + alpha * neighbor_avg[u]

        Convex blend preserves logit scale: if cosine inputs are in [-1, 1],
        the output stays in [-1, 1]. This avoids fighting the temperature
        schedule with implicit scale inflation.

        Isolated qubits (no neighbors) keep their original logits unchanged.

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

        # Average (isolated qubits: count=0 -> skip via mask)
        has_neighbors = (neighbor_count > 0).float()               # [N, 1]
        neighbor_avg = neighbor_sum / neighbor_count.clamp(min=1)  # [N, K]

        # Convex blend: preserves logit scale in [-1, 1]
        # Isolated qubits: mask ensures they keep original logits (alpha term zeroed)
        alpha = self.alpha  # scalar in (0, 1)
        blended = (1.0 - alpha) * logits + alpha * neighbor_avg
        return torch.where(has_neighbors.bool().expand_as(logits), blended, logits)

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
        self._T = float(t_new)

    def _load_from_state_dict(self, *args, **kwargs):
        # NOT load_state_dict(): when this head is loaded as a child of
        # SegmentClustering, PyTorch recurses through _load_from_state_dict and
        # a load_state_dict override on the child is never called. Refreshing
        # the float mirror here matters because an eval-only load never calls
        # set_epoch(), so the mirror would otherwise stay at T_init while the
        # buffer holds the annealed T_min -- silently evaluating the checkpoint
        # at the wrong temperature.
        super()._load_from_state_dict(*args, **kwargs)
        self._T = float(self.temperature)

    def reset_diagnostics(self) -> None:
        """Clear accumulated Sinkhorn residuals. Call at the start of each epoch."""
        if self.sinkhorn is not None:
            self.sinkhorn.reset_diagnostics()

    @property
    def diagnostics(self) -> dict:
        """
        Running-max Sinkhorn residuals since the last reset_diagnostics().
        Syncs the device once -- call when logging, not per layer.
        """
        if self.sinkhorn is None:
            return {}
        row_res, col_res = self.sinkhorn.residuals()
        return {"T": self._T, "row_residual": row_res, "col_residual": col_res}


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
        neighbor_alpha_init: float = 0.1,
        capacity_mode: str = "sinkhorn",
        caps=None,
        sinkhorn_iters: int = 30,
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
            capacity_mode=capacity_mode,
            caps=caps,
            sinkhorn_iters=sinkhorn_iters,
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
        # Single-circuit path: n_qubits is left None, so the head treats the
        # whole tensor as one circuit -- correct here, since h_seq entries are
        # [N, H] for a single circuit. The batched path goes through
        # EvolvingGNN.batch_forward, which passes n_qubits explicitly.
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

    def reset_diagnostics(self) -> None:
        """Clear accumulated Sinkhorn residuals. Call at the start of each epoch."""
        self.head.reset_diagnostics()

    @property
    def diagnostics(self) -> dict:
        """Running-max Sinkhorn residuals since the last reset; empty in softmax mode."""
        return self.head.diagnostics
