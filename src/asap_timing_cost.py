# Paste into src/cost_function.py, after SegmentTimeV3.
# SegmentTimeV3 and IdleCostV3 stay UNTOUCHED — they are the mode="barrier" path (E4 ablation).

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn


class ASAPTimingCost(nn.Module):
    """Movement-aware ASAP timing surrogate (EFCL).

    Replaces SegmentTimeV3 + IdleCostV3(idle_only) on the mode="asap" path.
    Computes per-qubit ready times and the decoherence cost of *actual* waiting,
    in one pass. Stateless: all hardware buffers are passed in by TotalCost,
    matching the ExecCostV3 / IdleCostV3 / CommMoveCostV3 convention.

    Model
    -----
    Each logical qubit u carries its own clock r[u].

      boundary (segment s-1 -> s)
          Moved probability mass leaves each technology:
              leaving = relu(P_{s-1} - P_s)                      [N,K]
              d_move  = sum_k leaving[u,k] * t2q[k]              [N]
          r[u] += d_move[u]. Movement is BUSY time: no T2 charge, because
          f_move already aggregates the state-transfer failure (matches
          lowering.py, which books movers as busy).

          This relu form is the exact min-coupling optimal-transport cost of the
          hard movement rule, NOT a heuristic. It holds ONLY because
          t_move(i,j) = t2q_i depends on the SOURCE index alone. If t_move ever
          becomes symmetric (e.g. max(t2q_i, t2q_j)), relu stops being the
          transport cost and this breaks SILENTLY. See _assert_source_side_rule.

      1Q gate on u
          r[u] += sum_k P_s[u,k] * t1q[k].    No wait, no idle.

      2Q gate on (u,v)
          start = max(r[u], r[v])                       (exact max, not LSE)
          the earlier operand waits; charge that wait at its expected 1/T2:
              C_idle += (start - r[u]) * <P_s[u], 1/T2>
                      + (start - r[v]) * <P_s[v], 1/T2>
          duration from a [K,K] matrix, local and remote in one expression:
              B[i,i] = t2q_i                  local
              B[i,j] = max(t2q_i, t2q_j)      remote (t_comm: teleported gate
                                              fires a local CNOT at BOTH
                                              endpoints in parallel, so it waits
                                              for the slower one; symmetric)
              d = sum_ij P_u(i) P_v(j) B[i,j]
                + sum_k  P_u(k) P_v(k) Gamma[k] t2q[k]     <- routing inflation,
                                                              gives (1+Gamma)t2q
                                                              on the diagonal
          r[u] = r[v] = start + d

      tail
          T_max = max_u r[u];  every early finisher holds its state:
              C_tail = sum_u (T_max - r[u]) * <P_last[u], 1/T2>

    Why walking segments/layers in index order IS ASAP
    --------------------------------------------------
    r[u] advances ONLY when u is touched by an operation, and the layering is a
    valid topological sort of the circuit DAG. So processing layers in index
    order reproduces per-qubit as-soon-as-possible issue. Qubits on independent
    dependency chains drift apart freely; nothing resynchronises them. This is
    the correctness argument and it is not obvious from the loop — do not
    "optimise" it into a per-layer clock.

    What this deliberately does NOT model
    -------------------------------------
    Block-boundary synchronisation. lowering.py synchronises the modules whose
    occupancy changes, so that decision-layer occupancy and wall-clock occupancy
    cannot diverge. Reproducing that under soft P would require differentiable
    block detection. It is omitted on purpose: capacity remains enforced in the
    scheduler's DECISION space, and the final hard schedule is judged by Aer.
    The omission drops only non-movers' synchronisation wait, which is
    nonnegative, so it is an optimistic timing bias.

    Measurement timing is not modelled — readout is stripped from the input
    circuits. Measurement FIDELITY (cm) is untouched, in ExecCostV3.

    Exactness limit: with one-hot P and zero migrations there are no boundaries,
    so this must agree with lowering.py exactly (test T1).
    """

    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------
    @staticmethod
    def _assert_source_side_rule(t_move_matrix: Optional[torch.Tensor]) -> None:
        """Guard: the moved-mass relu is only the transport cost for a
        source-side t_move. If someone later introduces a [K,K] move matrix,
        every row must be constant off-diagonal or this module is wrong.

        Not called on the hot path — invoke from tests / TotalCost.__init__ if a
        move matrix is ever added to the config.
        """
        if t_move_matrix is None:
            return
        K = t_move_matrix.shape[0]
        for i in range(K):
            off = [t_move_matrix[i, j] for j in range(K) if j != i]
            if len(off) > 1:
                ref = off[0]
                for x in off[1:]:
                    if not torch.allclose(x, ref):
                        raise ValueError(
                            "ASAPTimingCost: t_move is not source-side (row "
                            f"{i} varies with destination). The moved-mass relu "
                            "is no longer the optimal-transport cost; movement "
                            "time must be recomputed with a real transport step."
                        )

    @staticmethod
    def _assert_layer_disjoint(layer_d: Dict[str, torch.Tensor], s: int, ell: int) -> None:
        """A circuit layer must act on disjoint qubits. If it does not, the
        scatter below silently keeps one writer and drops the other.
        Test/debug only — not on the hot path.
        """
        parts = [layer_d[k] for k in ("oneq_u", "twoq_u", "twoq_v") if layer_d[k].numel() > 0]
        if not parts:
            return
        allq = torch.cat(parts, dim=0)
        if int(torch.unique(allq).numel()) != int(allq.numel()):
            raise ValueError(
                f"ASAPTimingCost: segment {s} layer {ell} touches a qubit twice; "
                "layers must be sets of parallel, disjoint operations."
            )

    # ------------------------------------------------------------------
    def forward(
        self,
        P_seq: List[torch.Tensor],          # list of [N,K], one per segment
        stats: Dict[str, Any],              # from SegmentStatsExtractor
        *,
        t1q: torch.Tensor,                  # [K]
        t2q: torch.Tensor,                  # [K]
        T2: torch.Tensor,                   # [K]
        use_routing_inflation_time: bool = True,
        validate: bool = False,
        debug: bool = False,
    ) -> Dict[str, Any]:

        device = P_seq[0].device
        dtype = P_seq[0].dtype
        S = len(P_seq)
        N = int(P_seq[0].shape[0])

        t1q = t1q.to(device=device, dtype=dtype)
        t2q = t2q.to(device=device, dtype=dtype)
        invT2 = 1.0 / torch.clamp(T2.to(device=device, dtype=dtype), min=1e-12)  # [K]

        # B[i,j] = expected 2Q duration for technologies (i, j).
        # torch.maximum already yields t2q_i on the diagonal, so local and
        # remote fall out of one expression with no diagonal fix-up.
        B = torch.maximum(t2q[:, None], t2q[None, :])  # [K,K]

        ready = torch.zeros((N,), device=device, dtype=dtype)
        seg_idle: List[torch.Tensor] = []
        zero = torch.zeros((), device=device, dtype=dtype)

        move_time_total = zero
        n_2q_seen = 0

        for s in range(S):
            P = P_seq[s]                    # [N,K]
            invT = P @ invT2                # [N]  post-move technology (see docstring)

            # ---- boundary: movement busy time -------------------------------
            if s > 0:
                leaving = torch.relu(P_seq[s - 1] - P)        # [N,K]
                d_move = (leaving * t2q).sum(dim=1)           # [N]
                ready = ready + d_move
                if debug:
                    move_time_total = move_time_total + d_move.sum().detach()

            idle_s = zero

            for ell, layer_d in enumerate(stats["layer_ops"][s]):
                if validate:
                    self._assert_layer_disjoint(layer_d, s, ell)

                # ---- 1Q: advance, no wait, no idle -------------------------
                oneq_u = layer_d["oneq_u"]
                if oneq_u.numel() > 0:
                    d1 = P[oneq_u] @ t1q                       # [G1]
                    ready = ready.index_add(0, oneq_u, d1)     # out-of-place

                # ---- 2Q: the only operation that produces waiting ----------
                tu = layer_d["twoq_u"]
                if tu.numel() > 0:
                    tv = layer_d["twoq_v"]

                    ru = ready[tu]                             # [G]
                    rv = ready[tv]                             # [G]
                    start = torch.maximum(ru, rv)              # exact max

                    # Gradient reaches BOTH operands: the later one through
                    # `start`, the earlier one through its own wait term below.
                    idle_s = idle_s + (
                        (start - ru) * invT[tu] + (start - rv) * invT[tv]
                    ).sum()

                    Pu = P[tu]                                 # [G,K]
                    Pv = P[tv]                                 # [G,K]
                    d = torch.einsum("gi,gj,ij->g", Pu, Pv, B)

                    if use_routing_inflation_time:
                        Ge = layer_d.get("twoq_gamma", None)   # [G,K]
                        if Ge is not None:
                            d = d + ((Pu * Pv) * Ge.to(dtype) * t2q).sum(dim=1)

                    new = start + d                            # [G]
                    ready = ready.scatter(
                        0,
                        torch.cat([tu, tv], dim=0),
                        torch.cat([new, new], dim=0),
                    )

                    if debug:
                        n_2q_seen += int(tu.numel())

            seg_idle.append(idle_s)

        # ---- tail: early finishers still hold their state -------------------
        makespan = ready.max()
        invT_last = P_seq[-1] @ invT2                          # [N]
        C_tail = ((makespan - ready) * invT_last).sum()

        per_segment_idle = torch.stack(seg_idle)               # [S]
        C_idle_gate = per_segment_idle.sum()

        # Attribute the tail to the last segment so per_segment_total keeps its
        # shape and nothing downstream changes.
        per_segment_idle = per_segment_idle.clone()
        per_segment_idle[-1] = per_segment_idle[-1] + C_tail

        out: Dict[str, Any] = {
            "per_segment_idle": per_segment_idle,              # [S]
            "C_idle_gate": C_idle_gate,
            "C_tail": C_tail,
            "makespan": makespan,
            "ready": ready,
        }

        if debug:
            out["asap_makespan"] = makespan.detach()
            out["asap_move_time_total"] = move_time_total
            out["asap_idle_gate"] = C_idle_gate.detach()
            out["asap_idle_tail"] = C_tail.detach()
            out["asap_n_2q"] = torch.tensor(float(n_2q_seen), device=device, dtype=dtype)

        return out
