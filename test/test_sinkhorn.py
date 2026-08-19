# tests/test_sinkhorn.py
"""
Step 2 — Sinkhorn numerical smoke test.

Four checks only:
  1. row / column marginal correctness
  2. finite gradients at T_min
  3. infeasible capacity raises
  4. dummy-row / slack behaviour for N < C_total

Plus the iteration-count selection sweep, which is the actual deliverable of
Step 2: one conservative n_iters that makes both residuals negligible at
T_min = 0.5 for balanced AND strongly imbalanced capacities, under both random
and adversarial (all-qubits-prefer-one-technology) logits.

NOT tested here, deliberately: any claim that caps >> N reduces to row-wise
softmax. Sinkhorn still chooses column potentials to meet the prescribed
marginals and the dummy rows interact with them, so no exact equivalence should
be asserted. The reviewer's "could it place everything on one technology when
every technology fits the whole circuit" is an evaluation-time empirical
question, answered by running the trained scheduler with caps >= N and
inspecting the hardened schedule.
"""

import torch

from src.sinkhorn import capacity_sinkhorn, CapacitySinkhorn

torch.manual_seed(0)

T_MIN = 0.5
T_INIT = 3.0
N = 30
K = 2
BALANCED = torch.tensor([20.0, 20.0])
IMBALANCED = torch.tensor([36.0, 4.0])

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def logits_random(n=N, k=K, scale=1.0):
    # Real head logits are cosine similarities in [-1, 1].
    return torch.empty(n, k).uniform_(-scale, scale)


def logits_adversarial(n=N, k=K):
    # Every qubit maximally prefers technology 0 -> the constraint binds as
    # hard as the head's bounded logits allow.
    z = torch.full((n, k), -1.0)
    z[:, 0] = 1.0
    return z


# ---------------------------------------------------------------- 1. marginals
def t_marginals():
    for label, caps in (("balanced", BALANCED), ("imbalanced", IMBALANCED)):
        for kind, Z in (("random", logits_random()), ("adversarial", logits_adversarial())):
            P = capacity_sinkhorn(Z, caps, T=T_MIN, n_iters=200)
            row_ok = torch.allclose(P.sum(-1), torch.ones(N), atol=1e-5)
            col_ok = bool((P.sum(-2) <= caps + 1e-4).all())
            mass_ok = abs(float(P.sum()) - N) < 1e-4
            check(f"1. marginals [{label}/{kind}]", row_ok and col_ok and mass_ok,
                  f"cols {[round(float(c), 3) for c in P.sum(-2)]}")

    # batched leading dim must match the per-circuit result exactly
    Zb = torch.stack([logits_random(), logits_random(), logits_adversarial()])
    Pb = capacity_sinkhorn(Zb, BALANCED, T=T_MIN, n_iters=200)
    dev = max(float((Pb[i] - capacity_sinkhorn(Zb[i], BALANCED, T=T_MIN, n_iters=200)).abs().max())
              for i in range(3))
    check("1. batched == per-circuit", dev < 1e-6, f"max dev {dev:.2e}")


# ---------------------------------------------------------------- 2. gradients
def t_gradients():
    for label, caps in (("balanced", BALANCED), ("imbalanced", IMBALANCED)):
        Z = logits_random().requires_grad_(True)
        P = capacity_sinkhorn(Z, caps, T=T_MIN, n_iters=50)
        (P * torch.randn(N, K)).sum().backward()
        g = Z.grad
        check(f"2. finite non-zero gradient at T_min [{label}]",
              bool(torch.isfinite(g).all()) and float(g.abs().max()) > 0,
              f"gnorm {float(g.norm()):.3e}")


# ---------------------------------------------------------------- 3. infeasible
def t_infeasible():
    raised = False
    try:
        capacity_sinkhorn(logits_random(), torch.tensor([10.0, 10.0]), T=T_MIN, n_iters=10)
    except ValueError:
        raised = True
    check("3. sum(caps) < N raises", raised)

    raised = False
    try:
        capacity_sinkhorn(logits_random(), torch.tensor([20.5, 19.5]), T=T_MIN, n_iters=10)
    except ValueError:
        raised = True
    check("3. non-integer caps raise", raised)


# ---------------------------------------------------------------- 4. slack
def t_slack():
    """
    N < C_total: dummy rows must absorb the surplus, and occupancy must NOT be
    pinned to the capacity-proportional split. This is the property that
    distinguishes dummy rows from naively renormalising column marginals down
    to total mass N (which would force 15/15 at C=[20,20], N=30).
    """
    caps = BALANCED
    Z = logits_adversarial()
    P = capacity_sinkhorn(Z, caps, T=T_MIN, n_iters=200)
    occ = P.sum(-2)
    proportional = caps / caps.sum() * N          # would be [15, 15]

    check("4. slack: occupancy not pinned to proportional split",
          float((occ - proportional).abs().max()) > 1.0,
          f"occ {[round(float(c), 2) for c in occ]} vs proportional "
          f"{[round(float(c), 2) for c in proportional]}")
    # Note: occupancy approaches but does not reach the cap. With logits bounded
    # to [-1, 1], the maximum logit gap at T_min = 0.5 is exp(2/0.5) ~ 55:1, so
    # the entropy term legitimately retains a little mass on the disfavoured
    # technology. Asserting exact saturation would be wrong.
    check("4. slack: occupancy driven toward the preferred technology's cap",
          float(occ[0]) > 0.95 * float(caps[0]),
          f"occ[0]={float(occ[0]):.4f}, cap[0]={float(caps[0]):.1f}")
    check("4. slack: dummy mass = C_total - N",
          abs((float(caps.sum()) - float(P.sum())) - (float(caps.sum()) - N)) < 1e-4)

    # zero-slack edge case: C_total == N is still well-posed (the split size is
    # forced, but which qubits go where is not). No dummy rows are exercised.
    P0 = capacity_sinkhorn(Z, torch.tensor([15.0, 15.0]), T=T_MIN, n_iters=200)
    check("4. zero-slack (C_total == N) is finite and exact",
          bool(torch.isfinite(P0).all()) and
          torch.allclose(P0.sum(-2), torch.tensor([15.0, 15.0]), atol=1e-3),
          f"cols {[round(float(c), 4) for c in P0.sum(-2)]}")


# ------------------------------------------------- iteration-count selection
def sweep():
    print("\n--- Step 2 deliverable: iteration count vs residuals ---")
    cases = [
        (f"T={T_MIN} balanced   adversarial", T_MIN, BALANCED, logits_adversarial()),
        (f"T={T_MIN} balanced   random     ", T_MIN, BALANCED, logits_random()),
        (f"T={T_MIN} imbalanced adversarial", T_MIN, IMBALANCED, logits_adversarial()),
        (f"T={T_MIN} imbalanced random     ", T_MIN, IMBALANCED, logits_random()),
        (f"T={T_INIT} balanced   adversarial", T_INIT, BALANCED, logits_adversarial()),
        (f"T={T_INIT} imbalanced adversarial", T_INIT, IMBALANCED, logits_adversarial()),
    ]
    iters_grid = [5, 10, 20, 30, 50, 75, 100]
    print(f"{'case':<36} " + " ".join(f"{i:>9}" for i in iters_grid))
    worst = {i: 0.0 for i in iters_grid}
    for label, T, caps, Z in cases:
        row = []
        for it in iters_grid:
            _, rr, cr = capacity_sinkhorn(Z, caps, T=T, n_iters=it, return_residuals=True)
            res = max(float(rr), float(cr))
            worst[it] = max(worst[it], res)
            row.append(f"{res:9.1e}")
        print(f"{label:<36} " + " ".join(row))
    print(f"{'WORST OVER ALL CASES':<36} " + " ".join(f"{worst[i]:9.1e}" for i in iters_grid))

    # 1e-5 rather than something tighter: the float32 residual floor is ~4e-6
    # (confirmed below by rerunning in float64), so a tighter tolerance would be
    # measuring precision, not convergence. 1e-5 is physically negligible --
    # four orders below the hardener's granularity of one qubit.
    tol = 1e-5
    ok = [i for i in iters_grid if worst[i] < tol]
    rec = ok[0] if ok else None
    print(f"\nsmallest n_iters with max(row,col) residual < {tol:g} across all cases: {rec}")
    if rec is not None:
        headroom = next((i for i in iters_grid if i >= 1.5 * rec), iters_grid[-1])
        print(f"recommended sinkhorn_iters (>=1.5x headroom): {headroom}")

    # Confirm the plateau is float32 precision, not stalled convergence.
    Z64 = logits_adversarial().double()
    print("\nfloat64 control (T=0.5, balanced, adversarial):")
    for it in (20, 50, 100):
        _, rr, cr = capacity_sinkhorn(Z64, BALANCED.double(), T=T_MIN,
                                      n_iters=it, return_residuals=True)
        print(f"  n_iters={it:<4} residual {max(float(rr), float(cr)):.2e}")
    return rec


# ------------------------------------------------- post-review robustness
def t_guards():
    """Guards added after review: T, n_iters, capacity sign, integer snapping."""
    Z = logits_random()
    for label, fn in (
        ("T = 0 raises",        lambda: capacity_sinkhorn(Z, BALANCED, T=0.0, n_iters=10)),
        ("T < 0 raises",        lambda: capacity_sinkhorn(Z, BALANCED, T=-0.5, n_iters=10)),
        ("n_iters = 0 raises",  lambda: capacity_sinkhorn(Z, BALANCED, T=T_MIN, n_iters=0)),
        ("negative cap raises", lambda: capacity_sinkhorn(Z, torch.tensor([-1.0, 41.0]), T=T_MIN, n_iters=10)),
        ("zero cap raises",     lambda: capacity_sinkhorn(Z, torch.tensor([0.0, 40.0]), T=T_MIN, n_iters=10)),
        ("near-integer cap raises", lambda: capacity_sinkhorn(Z, torch.tensor([20.0001, 19.9999]), T=T_MIN, n_iters=10)),
    ):
        raised = False
        try:
            fn()
        except ValueError:
            raised = True
        check(f"5. {label}", raised)


def t_no_sync_hot_path():
    """
    The module forward must not convert any device tensor to a Python scalar.
    Patched here on CPU tensors, which still catches every offending call.
    """
    from src.sinkhorn import CapacitySinkhorn
    mod = CapacitySinkhorn(BALANCED, n_iters=20)
    calls = []
    orig_item, orig_allclose = torch.Tensor.item, torch.allclose
    torch.Tensor.item = lambda self: (calls.append("item"), orig_item(self))[1]
    torch.allclose = lambda *a, **k: (calls.append("allclose"), orig_allclose(*a, **k))[1]
    try:
        for _ in range(3):
            mod(logits_random().unsqueeze(0), T=T_MIN)
    finally:
        torch.Tensor.item, torch.allclose = orig_item, orig_allclose
    check("5. forward performs no device sync", len(calls) == 0, f"sync calls: {calls}")

    # residuals() is where the sync is allowed to happen
    r, c = mod.residuals()
    check("5. residuals() returns accumulated running max", r >= 0 and c >= 0,
          f"row {r:.2e} col {c:.2e}")


def t_running_max():
    """Diagnostics must report the max over forwards, not the last forward."""
    from src.sinkhorn import CapacitySinkhorn
    mod = CapacitySinkhorn(BALANCED, n_iters=2)          # deliberately under-converged
    mod(logits_adversarial().unsqueeze(0), T=T_MIN)      # large residual
    hi_row, hi_col = mod.residuals()
    mod(torch.zeros(1, N, K), T=T_INIT)                  # tiny residual
    row, col = mod.residuals()
    check("5. residual is a running max, not last-forward",
          row >= hi_row - 1e-12 and col >= hi_col - 1e-12,
          f"after easy forward: ({row:.2e}, {col:.2e}); peak was ({hi_row:.2e}, {hi_col:.2e})")
    mod.reset_diagnostics()
    check("5. reset_diagnostics clears", mod.residuals() == (0.0, 0.0))


def t_checkpoint_roundtrip():
    """
    Everything the forward depends on must survive a checkpoint round-trip.

    `caps` and `log_caps` are buffers and load correctly; `_c_total` and
    `n_iters` are Python ints on the hot path and are NOT restored by the
    default machinery. Loading [20,20] into a module built with [15,15] would
    otherwise leave a 30-row problem demanding 40 units of column mass -- a
    ~5-qubit column residual, silently.
    """
    from src.sinkhorn import CapacitySinkhorn

    saved = CapacitySinkhorn(torch.tensor([20.0, 20.0]), n_iters=30)
    fresh = CapacitySinkhorn(torch.tensor([15.0, 15.0]), n_iters=7)
    check("6. pre-load state differs", fresh.c_total == 30 and fresh.n_iters == 7)

    fresh.load_state_dict(saved.state_dict())
    check("6. caps restored", torch.allclose(fresh.caps, torch.tensor([20.0, 20.0])))
    check("6. c_total recomputed after load", fresh.c_total == 40,
          f"c_total={fresh.c_total}, expected 40")
    check("6. n_iters restored from buffer", fresh.n_iters == 30,
          f"n_iters={fresh.n_iters}, expected 30")

    fresh.reset_diagnostics()
    P = fresh(logits_random().unsqueeze(0), T=T_MIN)
    row_res, col_res = fresh.residuals()
    check("6. residuals small after load (the bug produced ~5)",
          max(row_res, col_res) < 1e-5, f"max residual {max(row_res, col_res):.2e}")
    check("6. loaded module respects the LOADED caps",
          bool((P.sum(-2) <= torch.tensor([20.0, 20.0]) + 1e-4).all()),
          f"cols {[round(float(c), 3) for c in P.sum(-2)[0]]}")

    # a checkpoint is an untrusted source of caps, same as a config file
    bad = saved.state_dict()
    bad["caps"] = torch.tensor([-1.0, 41.0])
    raised = False
    try:
        CapacitySinkhorn(torch.tensor([20.0, 20.0]), n_iters=30).load_state_dict(bad)
    except ValueError:
        raised = True
    check("6. invalid caps in a checkpoint raise on load", raised)


def t_argmax_violation():
    """
    Soft feasibility does not imply argmax feasibility. Documents the
    worst case so the pilot's hardener-burden metric is interpretable.
    """
    from src.sinkhorn import argmax_violation
    Z = logits_adversarial()
    P = capacity_sinkhorn(Z, BALANCED, T=T_MIN, n_iters=30)
    soft_ok = bool((P.sum(-2) <= BALANCED + 1e-4).all())
    moved = float(argmax_violation(P, BALANCED))
    check("5. soft feasible yet argmax infeasible (documents hardener need)",
          soft_ok and moved > 0,
          f"occupancy {[round(float(c), 2) for c in P.sum(-2)]}, argmax must move {moved:.0f}/{N}")


if __name__ == "__main__":
    t_marginals()
    t_gradients()
    t_infeasible()
    t_slack()
    t_guards()
    t_no_sync_hot_path()
    t_running_max()
    t_checkpoint_roundtrip()
    t_argmax_violation()
    sweep()
    nf = sum(1 for _, ok in RESULTS if not ok)
    print(f"\n{len(RESULTS) - nf}/{len(RESULTS)} checks passed")
    raise SystemExit(1 if nf else 0)
