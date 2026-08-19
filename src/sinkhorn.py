# src/sinkhorn.py

"""
MOSAIC Scheduler — capacity Sinkhorn (C5).

Replaces the capacity *regularizer* with a capacity *structure*. Instead of
penalising overflow in the loss (whose gradient competes with the physics
gradient), the assignment matrix is projected onto the capacity-constrained
transportation polytope inside the head, so every P_l the cost model sees is
capacity-feasible by construction and the loss reduces to L = C_EFCL.

--------------------------------------------------------------------------
Formulation — balanced transport with dummy rows
--------------------------------------------------------------------------
Per layer, for a circuit with N logical qubits and K technologies of capacity
C_1..C_K, let C_total = sum_k C_k and let Z in R^{N x K} be the head's logits.

Append (C_total - N) dummy rows with zero logits:

    Z_aug in R^{C_total x K}

and solve the balanced entropic OT problem

    Q = argmin_{Q in Pi(1, c)}  <Q, -Z_aug> - T * H(Q)
    row marginals  r_u = 1        for all C_total rows
    col marginals  c_k = C_k

Then discard the dummy rows:  P = Q[:N, :].

This gives exactly the intended semantics:

    sum_k P[u,k] = 1     for every real qubit u
    sum_u P[u,k] <= C_k  for every technology k

with the difference C_k - (real occupancy) taken up by dummy mass. Crucially it
does NOT force a fixed proportional split: with C = [20,20] and N = 30, real
occupancy 18/12 (dummy 2/8) and 15/15 (dummy 5/5) are both reachable. Naively
renormalising the column marginals down to total mass N would force 15/15 and
destroy the scheduling decision.

Both marginals are unit-row / capacity-column, so sum(r) = sum(c) = C_total and
the problem is balanced — plain Sinkhorn converges without any unbalanced-OT
machinery.

Requires C_total >= N. Requires integer-valued capacities (they determine the
dummy row count).

--------------------------------------------------------------------------
T is the entropic regularisation parameter
--------------------------------------------------------------------------
The log-kernel is Z/T, so the head's existing temperature schedule plays the
role of eps. Lower T sharpens assignments AND makes the transport problem more
ill-conditioned, so it increases the number of iterations needed to satisfy the
marginals. The iteration count must therefore be chosen at T_min, not at T_init
-- see the Step 2 smoke test. This is also why only one schedule is ever active:
in "sinkhorn" mode T anneals the OT problem, in "softmax" mode it anneals the
softmax; running both would apply it twice.

Note: the bound Z in [-1, 1] (cosine similarity against L2-normalised
prototypes, preserved by the convex neighbour blend) is what keeps Z/T in a mild
numerical range at T_min = 0.5. Introducing a learnable similarity scale would
break that bound and invalidate the iteration budget.

--------------------------------------------------------------------------
Shape convention
--------------------------------------------------------------------------
The operator works on the LAST TWO dimensions: Z is [..., N, K]. A single
circuit is the no-leading-dims case [N, K]; a fixed-N mini-batch is [B, N, K].
The operator therefore contains no batching assumption of its own, and nothing
inside it changes if ragged mixed-N batching is adopted later — only the caller
changes. Vectorising over the leading dim rather than looping per circuit
matters: these matrices are far too small to be FLOP-bound, so a Python loop
costs ~16x in dispatch overhead at B=32.

--------------------------------------------------------------------------
Soft feasibility is not argmax feasibility
--------------------------------------------------------------------------
Sinkhorn makes the *differentiable* assignment used during training
capacity-feasible by construction. It does NOT guarantee that a row-wise argmax
of P is a capacity-feasible discrete assignment. Example: with C = [20, 20],
N = 30 and every qubit sharing an identical preference for technology 0, the
soft occupancy is [19.66, 10.34] -- feasible -- yet every row reads
[0.655, 0.345], so all 30 argmax to technology 0 and the hardener must move 10.

Discrete inference therefore still requires the capacity-feasible rounding step,
and hardener burden is an empirical quantity to be measured, not assumed.

--------------------------------------------------------------------------
Device synchronisation
--------------------------------------------------------------------------
This function is called once per circuit layer (~80x per forward, ~400k times
over a 140-epoch run), so it must not force CPU<->GPU synchronisation. Anything
that converts a device tensor to a Python scalar -- .item(), float(), bool(),
torch.allclose -- drains the CUDA queue and serialises execution.

All such work is therefore hoisted to construction time: capacity validation,
C_total, and log(caps) are computed once in CapacitySinkhorn.__init__ and passed
in via `precomputed`. The temperature arrives as a Python float (the head keeps
a float mirror of its buffer). Residuals stay as detached device tensors and are
only converted to Python values when actually logged.

`validate=True` is the default for the standalone/test path; the training path
passes validate=False with precomputed values.

--------------------------------------------------------------------------
Numerics
--------------------------------------------------------------------------
All updates are in the log domain. The final update is a row update: rows feed
EFCL as probability distributions, so a row summing to 0.9999 would
systematically under-count expected cost, whereas a converged column residual is
inert. This is an ordering detail, not a tolerance trade — the iteration count
is chosen so that BOTH residuals are negligible.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def validate_caps(caps: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """
    Validate capacities once and snap them to exact integers.

    Returns (caps_rounded, c_total). Call at construction, never per layer:
    every check here forces a device synchronisation.

    Snapping matters. A tolerance-based integer check alone would accept
    caps = 20.0001, after which log(caps) would use 20.0001 while the dummy row
    count uses round(sum) = 40 -- so total row mass and total column mass would
    differ and the transport problem would be silently unbalanced. Validate
    against the rounded values, then USE the rounded values.
    """
    caps = caps.detach().reshape(-1).float()
    rounded = caps.round()
    if not torch.allclose(caps, rounded, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"caps must be integer-valued (they set the dummy row count): {caps.tolist()}"
        )
    if bool((rounded <= 0).any()):
        raise ValueError(
            f"every technology capacity must be strictly positive: {rounded.tolist()}"
        )
    return rounded, int(rounded.sum().item())


def capacity_sinkhorn(
    Z: torch.Tensor,        # [..., N, K] real-qubit logits
    caps: torch.Tensor,     # [K] integer-valued, strictly positive capacities
    T: float,               # entropic regularisation == head temperature
    n_iters: int,
    return_residuals: bool = False,
    validate: bool = True,
    c_total: Optional[int] = None,
    log_caps: Optional[torch.Tensor] = None,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Balanced log-domain Sinkhorn with (C_total - N) zero-logit dummy rows.

    Args:
        Z:       [..., N, K] assignment logits for the real qubits.
        caps:    [K] per-technology capacity. Must be integer-valued and
                 sum to >= N.
        T:       entropic regularisation parameter (the head temperature).
        n_iters: fixed number of row/column update pairs. No Python
                 convergence loop -- the count is set once by the Step 2
                 smoke test at T_min.
        return_residuals: also return (max_row_residual, max_col_residual) as
                 DETACHED DEVICE TENSORS (never Python floats -- see the
                 synchronisation note in the module docstring), computed over
                 the FULL augmented matrix including dummy rows.
        validate: run capacity validation. Costs a device sync; leave True for
                 standalone/test use, pass False on the training hot path and
                 supply c_total / log_caps from `validate_caps`.
        c_total:  precomputed int(sum(caps)). Required when validate=False.
        log_caps: precomputed log(caps). Optional; recomputed if omitted.

    Returns:
        P: [..., N, K] with rows summing to 1 and columns summing to <= caps.
    """
    if Z.dim() < 2:
        raise ValueError(f"Z must be at least 2-D [..., N, K], got shape {tuple(Z.shape)}")

    if T <= 0:
        raise ValueError(f"T must be strictly positive (it divides the logits), got {T}")
    if n_iters <= 0:
        raise ValueError(
            f"n_iters must be >= 1, got {n_iters}. With n_iters=0 only the final "
            f"row update runs and the column marginals are left unconstrained."
        )

    N, K = Z.shape[-2], Z.shape[-1]
    device, dtype = Z.device, Z.dtype
    caps = caps.to(device=device, dtype=dtype).reshape(-1)

    if caps.numel() != K:
        raise ValueError(f"caps has {caps.numel()} entries but Z has K={K}")

    if validate:
        caps, c_total = validate_caps(caps)
        caps = caps.to(device=device, dtype=dtype)
    elif c_total is None:
        raise ValueError("c_total must be supplied when validate=False")

    if c_total < N:
        raise ValueError(
            f"Infeasible capacity: sum(caps)={c_total} < N={N}. "
            f"No assignment can satisfy per-technology capacity."
        )

    lead = Z.shape[:-2]
    n_dummy = c_total - N

    # --- Augment with dummy rows (zero logits) -> [..., C_total, K] ---
    if n_dummy > 0:
        dummy = torch.zeros(*lead, n_dummy, K, device=device, dtype=dtype)
        Z_aug = torch.cat([Z, dummy], dim=-2)
    else:
        Z_aug = Z

    log_k = Z_aug / T                                        # [..., R, K]

    # Every row (real or dummy) carries unit mass -> log r = 0 everywhere.
    # Column marginals are the capacities. sum(r) = sum(c) = C_total: balanced.
    if log_caps is None:
        log_caps = torch.log(caps)
    log_c = log_caps.to(device=device, dtype=dtype).expand(*lead, K)  # [..., K]

    f = torch.zeros(*lead, c_total, device=device, dtype=dtype)   # row potentials
    g = torch.zeros(*lead, K, device=device, dtype=dtype)         # col potentials

    for _ in range(n_iters):
        f = -torch.logsumexp(log_k + g.unsqueeze(-2), dim=-1)
        g = log_c - torch.logsumexp(log_k + f.unsqueeze(-1), dim=-2)

    # Final row update (see module docstring: ordering, not a tolerance trade).
    f = -torch.logsumexp(log_k + g.unsqueeze(-2), dim=-1)

    Q = torch.exp(log_k + f.unsqueeze(-1) + g.unsqueeze(-2))      # [..., R, K]
    P = Q[..., :N, :]

    if not return_residuals:
        return P

    # Detached device tensors, NOT Python floats: converting here would sync
    # the CUDA queue once per layer.
    with torch.no_grad():
        row_res = (Q.sum(dim=-1) - 1.0).abs().amax()
        col_res = (Q.sum(dim=-2) - caps).abs().amax()
    return P, row_res, col_res


class CapacitySinkhorn(nn.Module):
    """
    Thin nn.Module wrapper around `capacity_sinkhorn`.

    Holds the capacity vector and the iteration count as buffers (both are data,
    not learned parameters, and both must travel with the checkpoint). All
    validation and all derived quantities (C_total, log(caps)) are computed once
    at construction or at load, so the per-layer forward performs no device
    synchronisation. Temperature is passed in by the caller as a Python float so
    that a single schedule lives in the head and reading it costs nothing.

    Residuals accumulate as a running MAXIMUM over every forward since the last
    `reset_diagnostics()`, held as detached device tensors. A per-layer overwrite
    would report only the final layer of the final circuit, which is not the
    quantity the pilot needs to log.
    """

    def __init__(self, caps, n_iters: int = 30):
        super().__init__()
        if int(n_iters) <= 0:
            raise ValueError(f"n_iters must be >= 1, got {n_iters}")
        caps_t, c_total = validate_caps(torch.as_tensor(caps, dtype=torch.float32))
        self.register_buffer("caps", caps_t)
        self.register_buffer("log_caps", torch.log(caps_t))
        # n_iters is a buffer so it travels with the checkpoint. Anything the
        # forward depends on that is NOT in state_dict is a silent-wrong-answer
        # waiting to happen when an eval script reconstructs the module with
        # different defaults -- which is exactly how the _c_total bug arose.
        self.register_buffer("n_iters_buf", torch.tensor(int(n_iters), dtype=torch.long))
        # Python mirrors of derived / buffered state, refreshed on load.
        self._c_total = c_total
        self.n_iters = int(n_iters)
        self._row_res_max: Optional[torch.Tensor] = None
        self._col_res_max: Optional[torch.Tensor] = None

    def _load_from_state_dict(self, *args, **kwargs):
        """
        Refresh everything derived from the loaded buffers.

        `caps` and `log_caps` are buffers and load correctly, but `_c_total` and
        `n_iters` are Python ints used on the hot path and are NOT restored by
        the default machinery. Loading [20,20] into a module constructed with
        [15,15] would leave _c_total = 30, so Sinkhorn would build a 30-row
        problem while demanding 40 units of column mass -- a ~5-qubit column
        residual, with no error raised.

        `.item()` here is fine: this runs once at load, not per layer. The
        loaded capacities are re-validated for the same reason -- a checkpoint
        is an untrusted source of caps just as a config file is.
        """
        super()._load_from_state_dict(*args, **kwargs)
        caps_t, c_total = validate_caps(self.caps)
        self.caps.copy_(caps_t.to(self.caps.device))
        self.log_caps.copy_(torch.log(self.caps))
        self._c_total = c_total
        self.n_iters = int(self.n_iters_buf.item())

    @property
    def c_total(self) -> int:
        return self._c_total

    def set_caps(self, caps) -> None:
        """Replace the capacity vector (hardware-setting reconfiguration)."""
        caps_t, c_total = validate_caps(torch.as_tensor(caps, dtype=torch.float32))
        if caps_t.numel() != self.caps.numel():
            raise ValueError(f"caps must have {self.caps.numel()} entries")
        dev = self.caps.device
        self.caps = caps_t.to(dev)
        self.log_caps = torch.log(caps_t).to(dev)
        self._c_total = c_total

    def reset_diagnostics(self) -> None:
        """Clear the running residual maxima. Call at the start of each epoch."""
        self._row_res_max = None
        self._col_res_max = None

    def residuals(self) -> Tuple[float, float]:
        """
        Running max (row, col) residual since the last reset, as Python floats.
        This is the ONLY place a device sync occurs -- call it when logging, not
        per layer.
        """
        if self._row_res_max is None:
            return 0.0, 0.0
        return float(self._row_res_max.item()), float(self._col_res_max.item())

    def forward(self, Z: torch.Tensor, T: float) -> torch.Tensor:
        """
        Z: [..., N, K] -> P: [..., N, K]

        T must be a Python float. Passing the head's temperature *buffer* would
        force a device sync on every layer.
        """
        P, row_res, col_res = capacity_sinkhorn(
            Z, self.caps, T=T, n_iters=self.n_iters,
            return_residuals=True, validate=False,
            c_total=self._c_total, log_caps=self.log_caps,
        )
        # Running max, accumulated on device -- no sync.
        if self._row_res_max is None:
            self._row_res_max, self._col_res_max = row_res, col_res
        else:
            self._row_res_max = torch.maximum(self._row_res_max, row_res)
            self._col_res_max = torch.maximum(self._col_res_max, col_res)
        return P


@torch.no_grad()
def argmax_violation(P: torch.Tensor, caps: torch.Tensor) -> torch.Tensor:
    """
    Number of qubits a capacity-feasible rounding must move away from their
    argmax choice. This is the hardener burden, and it is an EMPIRICAL quantity:
    Sinkhorn guarantees the soft P is feasible, not that its argmax is.

    P: [..., N, K] -> [...] counts. Returns a device tensor; do not .item() it
    inside the training loop.
    """
    K = P.shape[-1]
    am = P.argmax(dim=-1)
    counts = torch.zeros(*P.shape[:-2], K, device=P.device, dtype=P.dtype)
    counts.scatter_add_(-1, am, torch.ones_like(am, dtype=P.dtype))
    return torch.relu(counts - caps.to(P.device, P.dtype)).sum(dim=-1)
