"""
Test suite for CapacityPenalty and enforce_capacity.

Tests:
  1. Lambda_cap derivation — verify Delta_max components against hand-computed values
  2. Sharpening — the soft-count underestimation problem (5 qubits, all 0.6)
  3. No penalty when within capacity
  4. Penalty fires when over capacity
  5. Penalty grows quadratically with violation size
  6. Gradient flow — penalty gradients reach P_seq entries
  7. enforce_capacity — hard assignments respect capacity
  8. enforce_capacity — runner-up reassignment logic
  9. Multi-layer sequence test
"""

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import math

from src.cost_function import TotalCost, CapacityPenalty
from utils.inference_utils import enforce_capacity, enforce_capacity_sequence
from utils.cost_config_reader import load_cost_config


def make_test_config(cap_sc=127, cap_na=256, beta=4.0, safety_factor=5.0):
    """Build a config dict matching cost_config_v3.json structure with custom capacities."""
    return {
        "techs": [
            {
                "name": "sc",
                "gate_fidelity": {"f1q": 0.9999, "f2q": 0.9990, "fm": 0.9900},
                "coherence": {"T2": 80000.0},
                "routing": {"kappa": 2.3},
                "gate_time": {"t1q": 20.0, "t2q": 200.0, "tm": 300.0},
                "capacity": {"max_qubits": cap_sc},
            },
            {
                "name": "na",
                "gate_fidelity": {"f1q": 0.9995, "f2q": 0.9970, "fm": 0.9800},
                "coherence": {"T2": 200000.0},
                "routing": {"kappa": 0, "all_to_all": True},
                "gate_time": {"t1q": 200.0, "t2q": 2000.0, "tm": 500.0},
                "capacity": {"max_qubits": cap_na},
            },
        ],
        "comm": {"f_comm": 0.95, "f_move": 0.99, "t_remote": 0.0},
        "timing": {"delta": 500.0},
        "timing_model": {
            "mode": "smooth_max",
            "tau0": 500.0, "tau_min": 25.0, "tau_gamma": 0.98,
            "lambda0": 0.2, "lambda_max": 1.0, "lambda_gamma": 1.05,
            "use_routing_inflation_time": True,
        },
        "decoherence_model": {"mode": "idle_only"},
        "gate_names": {"measure": ["measure", "meas", "m"]},
        "connectivity_proxy": {
            "gamma_max": 2.5,
            "delta_community": 3,
            "pair_reuse_threshold": 2,
            "dense_lambda_decay": 0.85,
        },
        "capacity_penalty": {
            "beta": beta,
            "safety_factor": safety_factor,
        },
    }


# =====================================================================
# Test 1: Lambda_cap derivation
# =====================================================================
def test_lambda_cap_derivation():
    print("\n" + "=" * 60)
    print("TEST 1: Lambda_cap derivation from cost-model units")
    print("=" * 60)

    config = make_test_config(safety_factor=5.0)
    cost_module = TotalCost(config)
    cap_module = CapacityPenalty(cost_module, config)

    # Hand-compute each Delta component
    c1q_sc = -math.log(0.9999)   # ~0.00010
    c1q_na = -math.log(0.9995)   # ~0.00050
    delta_1q = abs(c1q_na - c1q_sc)

    c2q_sc = -math.log(0.9990)   # ~0.00100
    c2q_na = -math.log(0.9970)   # ~0.00300
    c_comm = -math.log(0.95)     # ~0.05129
    gamma_max = 2.5
    worst_local = (1 + gamma_max) * c2q_sc
    delta_2q = max(0.0, c_comm - worst_local)

    T2_sc, T2_na = 80000.0, 200000.0
    delta_t = 500.0
    delta_idle = delta_t * (1.0 / T2_sc - 1.0 / T2_na)

    c_move = -math.log(0.99)
    delta_move = c_move

    delta_max = delta_1q + delta_2q + delta_idle + delta_move
    expected_lambda = 5.0 * delta_max

    actual_lambda = cap_module.lambda_cap.item()

    print(f"  Delta_1q   = {delta_1q:.6f}")
    print(f"  Delta_2q   = {delta_2q:.6f}  (dominant: comm avoidance)")
    print(f"  Delta_idle = {delta_idle:.6f}")
    print(f"  Delta_move = {delta_move:.6f}")
    print(f"  Delta_max  = {delta_max:.6f}")
    print(f"  Expected lambda_cap = {expected_lambda:.6f}")
    print(f"  Actual   lambda_cap = {actual_lambda:.6f}")

    assert abs(actual_lambda - expected_lambda) < 1e-5, \
        f"Lambda mismatch: {actual_lambda} vs {expected_lambda}"
    print("  ✓ PASSED")


# =====================================================================
# Test 2: Sharpening catches the soft-count underestimation
# =====================================================================
def test_sharpening_underestimation():
    print("\n" + "=" * 60)
    print("TEST 2: Sharpening catches soft-count underestimation")
    print("  Scenario: 5 qubits, all P[u,k0]=0.6, capacity=4")
    print("  Naive soft count = 3.0 (no penalty)")
    print("  Sharpened count should exceed 4 (penalty fires)")
    print("=" * 60)

    config = make_test_config(cap_sc=4, cap_na=4, beta=4.0)
    cost_module = TotalCost(config)
    cap_module = CapacityPenalty(cost_module, config)

    N, K = 5, 2
    # All qubits lean toward tech 0 at 0.6
    P_ell = torch.full((N, K), 0.4)
    P_ell[:, 0] = 0.6
    P_seq = [P_ell]

    # Naive soft count
    naive_count_k0 = P_ell[:, 0].sum().item()
    print(f"  Naive soft count (tech 0): {naive_count_k0:.1f}")

    # Sharpened count (manual check)
    beta = 4.0
    P_sharp = P_ell.pow(beta)
    P_sharp = P_sharp / P_sharp.sum(dim=1, keepdim=True)
    sharp_count_k0 = P_sharp[:, 0].sum().item()
    print(f"  Sharpened count (tech 0):  {sharp_count_k0:.3f}")

    # Penalty should fire
    cap_out = cap_module(P_seq)
    penalty = cap_out["penalty"].item()
    excess = cap_out["per_layer_excess"][0].item()

    print(f"  Excess (tech 0): {sharp_count_k0 - 4:.3f}")
    print(f"  Per-layer excess sum: {excess:.4f}")
    print(f"  Penalty: {penalty:.6f}")

    assert sharp_count_k0 > 4.0, \
        f"Sharpened count {sharp_count_k0} should exceed capacity 4"
    assert penalty > 0.0, \
        f"Penalty should be > 0 but got {penalty}"
    assert naive_count_k0 <= 4.0, \
        f"Naive count should be <= 4 (showing the problem)"
    print("  ✓ PASSED — sharpening detects violation that naive count misses")


# =====================================================================
# Test 3: No penalty when within capacity
# =====================================================================
def test_no_penalty_within_capacity():
    print("\n" + "=" * 60)
    print("TEST 3: No penalty when assignments are within capacity")
    print("=" * 60)

    config = make_test_config(cap_sc=127, cap_na=256)
    cost_module = TotalCost(config)
    cap_module = CapacityPenalty(cost_module, config)

    N, K = 10, 2
    # Even split: 5 qubits strongly on each tech
    P_ell = torch.zeros(N, K)
    P_ell[:5, 0] = 0.95   # qubits 0-4 → sc
    P_ell[:5, 1] = 0.05
    P_ell[5:, 0] = 0.05   # qubits 5-9 → na
    P_ell[5:, 1] = 0.95
    P_seq = [P_ell]

    cap_out = cap_module(P_seq)
    penalty = cap_out["penalty"].item()
    excess = cap_out["per_layer_excess"][0].item()

    print(f"  10 qubits, caps=[127, 256], near-hard split 5/5")
    print(f"  Penalty: {penalty:.10f}")
    print(f"  Excess:  {excess:.10f}")

    assert penalty < 1e-10, f"Penalty should be ~zero, got {penalty}"
    print("  ✓ PASSED")


# =====================================================================
# Test 4: Penalty fires when over capacity
# =====================================================================
def test_penalty_fires_over_capacity():
    print("\n" + "=" * 60)
    print("TEST 4: Penalty fires when tech exceeds capacity")
    print("=" * 60)

    # Tight capacity: only 3 qubits per tech, but 8 qubits lean to tech 0
    config = make_test_config(cap_sc=3, cap_na=7)
    cost_module = TotalCost(config)
    cap_module = CapacityPenalty(cost_module, config)

    N, K = 8, 2
    P_ell = torch.zeros(N, K)
    P_ell[:, 0] = 0.9   # all 8 qubits strongly prefer tech 0
    P_ell[:, 1] = 0.1
    P_seq = [P_ell]

    # Sharpened count for tech 0 should be close to 8
    beta = cap_module.beta
    P_sharp = P_ell.pow(beta)
    P_sharp = P_sharp / P_sharp.sum(dim=1, keepdim=True)
    n_tilde_0 = P_sharp[:, 0].sum().item()

    cap_out = cap_module(P_seq)
    penalty = cap_out["penalty"].item()

    print(f"  8 qubits, all P[:,0]=0.9, caps=[3, 7]")
    print(f"  Sharpened count tech 0: {n_tilde_0:.3f} (cap=3)")
    print(f"  Penalty: {penalty:.6f}")

    assert penalty > 0.0, f"Penalty should be > 0"
    assert n_tilde_0 > 3.0, f"Sharpened count should exceed cap=3"
    print("  ✓ PASSED")


# =====================================================================
# Test 5: Penalty grows quadratically with violation size
# =====================================================================
def test_penalty_quadratic_growth():
    print("\n" + "=" * 60)
    print("TEST 5: Penalty grows quadratically with violation size")
    print("=" * 60)

    K = 2
    penalties = []

    # Increase N from 4 to 10 with cap=3 on tech 0 → growing violation
    for N in [4, 6, 8, 10]:
        config = make_test_config(cap_sc=3, cap_na=N)
        cost_module = TotalCost(config)
        cap_module = CapacityPenalty(cost_module, config)

        P_ell = torch.zeros(N, K)
        P_ell[:, 0] = 0.95  # all strongly prefer tech 0
        P_ell[:, 1] = 0.05
        P_seq = [P_ell]

        cap_out = cap_module(P_seq)
        pen = cap_out["penalty"].item()
        penalties.append((N, pen))
        print(f"  N={N:2d}, penalty={pen:.6f}")

    # Check monotonicity
    for i in range(1, len(penalties)):
        assert penalties[i][1] >= penalties[i - 1][1], \
            f"Penalty should grow: N={penalties[i][0]} ({penalties[i][1]}) < N={penalties[i-1][0]} ({penalties[i-1][1]})"

    # Check super-linear growth (quadratic): ratio should increase
    if penalties[0][1] > 0 and penalties[-1][1] > 0:
        ratio = penalties[-1][1] / penalties[0][1]
        n_ratio = penalties[-1][0] / penalties[0][0]
        print(f"  Penalty ratio (N={penalties[-1][0]}/N={penalties[0][0]}): {ratio:.2f}")
        print(f"  N ratio: {n_ratio:.2f}")
        assert ratio > n_ratio, "Should grow faster than linearly"
        print("  ✓ PASSED — penalty grows super-linearly")
    else:
        print("  ✓ PASSED — monotonically increasing (first penalty is zero, no ratio check)")


# =====================================================================
# Test 6: Gradient flow
# =====================================================================
def test_gradient_flow():
    print("\n" + "=" * 60)
    print("TEST 6: Gradient flows from penalty to P_seq entries")
    print("=" * 60)

    config = make_test_config(cap_sc=3, cap_na=10)
    cost_module = TotalCost(config)
    cap_module = CapacityPenalty(cost_module, config)

    N, K = 8, 2
    # Use softmax to ensure valid probabilities with gradients
    logits = torch.randn(N, K, requires_grad=True)
    P_ell = torch.softmax(logits, dim=1)
    P_seq = [P_ell]

    cap_out = cap_module(P_seq)
    penalty = cap_out["penalty"]

    print(f"  Penalty value: {penalty.item():.6f}")

    penalty.backward()
    grad = logits.grad

    print(f"  Logits grad shape: {grad.shape}")
    print(f"  Grad norm: {grad.norm().item():.6f}")
    print(f"  Grad max abs: {grad.abs().max().item():.6f}")
    print(f"  Any non-zero: {(grad.abs() > 1e-12).any().item()}")

    assert grad is not None, "Gradient should not be None"
    assert (grad.abs() > 1e-12).any(), "At least some gradients should be non-zero"
    print("  ✓ PASSED")


# =====================================================================
# Test 7: enforce_capacity — basic feasibility
# =====================================================================
def test_enforce_capacity_basic():
    print("\n" + "=" * 60)
    print("TEST 7: enforce_capacity produces feasible hard assignments")
    print("=" * 60)

    N, K = 10, 2
    capacities = torch.tensor([4, 6])

    # All qubits prefer tech 0
    P_ell = torch.zeros(N, K)
    P_ell[:, 0] = 0.8
    P_ell[:, 1] = 0.2

    assignments = enforce_capacity(P_ell, capacities)

    count_0 = (assignments == 0).sum().item()
    count_1 = (assignments == 1).sum().item()

    print(f"  Input: all P[:,0]=0.8, caps=[4, 6]")
    print(f"  Assignments: {assignments.tolist()}")
    print(f"  Count tech 0: {count_0} (cap=4), tech 1: {count_1} (cap=6)")

    assert count_0 <= 4, f"Tech 0 count {count_0} exceeds cap 4"
    assert count_1 <= 6, f"Tech 1 count {count_1} exceeds cap 6"
    assert count_0 + count_1 == N, f"Total {count_0 + count_1} != N={N}"
    print("  ✓ PASSED")


# =====================================================================
# Test 8: enforce_capacity — runner-up reassignment
# =====================================================================
def test_enforce_capacity_runner_up():
    print("\n" + "=" * 60)
    print("TEST 8: enforce_capacity reassigns lowest-confidence qubits")
    print("=" * 60)

    N, K = 6, 2
    capacities = torch.tensor([3, 3])

    # Varying confidence: qubits 0-2 strongly prefer tech 0, qubits 3-5 weakly
    P_ell = torch.zeros(N, K)
    P_ell[0, :] = torch.tensor([0.95, 0.05])  # strong
    P_ell[1, :] = torch.tensor([0.90, 0.10])  # strong
    P_ell[2, :] = torch.tensor([0.85, 0.15])  # medium
    P_ell[3, :] = torch.tensor([0.60, 0.40])  # weak
    P_ell[4, :] = torch.tensor([0.55, 0.45])  # weakest
    P_ell[5, :] = torch.tensor([0.70, 0.30])  # medium-weak

    assignments = enforce_capacity(P_ell, capacities)

    count_0 = (assignments == 0).sum().item()
    count_1 = (assignments == 1).sum().item()

    print(f"  Probabilities for tech 0: {P_ell[:, 0].tolist()}")
    print(f"  Assignments: {assignments.tolist()}")
    print(f"  Count tech 0: {count_0} (cap=3), tech 1: {count_1} (cap=3)")

    assert count_0 <= 3, f"Tech 0 count {count_0} exceeds cap 3"
    assert count_1 <= 3, f"Tech 1 count {count_1} exceeds cap 3"

    # The 3 qubits kept on tech 0 should be the most confident ones (0, 1, 2)
    on_tech0 = set(i for i in range(N) if assignments[i].item() == 0)
    print(f"  Qubits on tech 0: {on_tech0}")

    # Qubits 0, 1, 2 have highest P[:,0] — they should be the ones kept
    assert on_tech0 == {0, 1, 2}, \
        f"Expected most confident qubits {{0,1,2}} on tech 0, got {on_tech0}"
    print("  ✓ PASSED — lowest-confidence qubits reassigned first")


# =====================================================================
# Test 9: enforce_capacity_sequence
# =====================================================================
def test_enforce_capacity_sequence():
    print("\n" + "=" * 60)
    print("TEST 9: enforce_capacity_sequence over multiple layers")
    print("=" * 60)

    N, K = 8, 2
    capacities = torch.tensor([4, 4])
    L = 5

    P_seq = []
    for ell in range(L):
        P_ell = torch.softmax(torch.randn(N, K), dim=1)
        P_seq.append(P_ell)

    assignments_seq = enforce_capacity_sequence(P_seq, capacities)

    print(f"  {L} layers, {N} qubits, caps=[4, 4]")
    all_feasible = True
    for ell, a in enumerate(assignments_seq):
        c0 = (a == 0).sum().item()
        c1 = (a == 1).sum().item()
        feasible = c0 <= 4 and c1 <= 4
        all_feasible = all_feasible and feasible
        print(f"  Layer {ell}: tech0={c0}, tech1={c1} {'✓' if feasible else '✗'}")

    assert all_feasible, "Some layers violated capacity"
    print("  ✓ PASSED")


# =====================================================================
# Test 10: 4-tech generalization
# =====================================================================
def test_four_techs():
    print("\n" + "=" * 60)
    print("TEST 10: Capacity penalty generalizes to K=4 technologies")
    print("=" * 60)

    config = {
        "techs": [
            {
                "name": f"tech{i}",
                "gate_fidelity": {"f1q": 0.9999 - i * 0.0002,
                                  "f2q": 0.999 - i * 0.001,
                                  "fm": 0.99 - i * 0.005},
                "coherence": {"T2": 80000.0 + i * 40000.0},
                "routing": {"kappa": 2.3 if i == 0 else 0, "all_to_all": i > 0},
                "gate_time": {"t1q": 20.0 + i * 60.0,
                              "t2q": 200.0 + i * 600.0,
                              "tm": 300.0 + i * 100.0},
                "capacity": {"max_qubits": 5},
            }
            for i in range(4)
        ],
        "comm": {"f_comm": 0.95, "f_move": 0.99, "t_remote": 0.0},
        "timing": {"delta": 500.0},
        "timing_model": {
            "mode": "smooth_max",
            "tau0": 500.0, "tau_min": 25.0, "tau_gamma": 0.98,
            "lambda0": 0.2, "lambda_max": 1.0, "lambda_gamma": 1.05,
            "use_routing_inflation_time": True,
        },
        "decoherence_model": {"mode": "idle_only"},
        "gate_names": {"measure": ["measure", "meas", "m"]},
        "connectivity_proxy": {
            "gamma_max": 2.5,
            "delta_community": 3,
            "pair_reuse_threshold": 2,
            "dense_lambda_decay": 0.85,
        },
        "capacity_penalty": {"beta": 4, "safety_factor": 5.0},
    }

    cost_module = TotalCost(config)
    cap_module = CapacityPenalty(cost_module, config)

    N, K = 20, 4
    print(f"  K={K} techs, N={N} qubits, caps=[5,5,5,5]")
    print(f"  lambda_cap={cap_module.lambda_cap.item():.6f}")

    # All qubits prefer tech 0 → violates cap=5
    P_ell = torch.zeros(N, K)
    P_ell[:, 0] = 0.7
    P_ell[:, 1] = 0.1
    P_ell[:, 2] = 0.1
    P_ell[:, 3] = 0.1
    P_seq = [P_ell]

    cap_out = cap_module(P_seq)
    penalty = cap_out["penalty"].item()
    print(f"  All prefer tech0: penalty={penalty:.6f}")
    assert penalty > 0.0, "Should have penalty for 20 qubits on cap=5 tech"

    # Even split: 5 qubits each → no violation
    P_even = torch.zeros(N, K)
    for i in range(N):
        k = i % K
        P_even[i, k] = 0.95
        for j in range(K):
            if j != k:
                P_even[i, j] = 0.05 / (K - 1)
    P_seq_even = [P_even]

    cap_out_even = cap_module(P_seq_even)
    penalty_even = cap_out_even["penalty"].item()
    print(f"  Even 5/5/5/5 split: penalty={penalty_even:.10f}")
    assert penalty_even < 1e-10, f"Even split should have ~zero penalty, got {penalty_even}"

    # enforce_capacity with K=4
    capacities = torch.tensor([5, 5, 5, 5])
    assignments = enforce_capacity(P_ell, capacities)
    counts = [(assignments == k).sum().item() for k in range(K)]
    print(f"  enforce_capacity counts: {counts}")
    for k in range(K):
        assert counts[k] <= 5, f"Tech {k} count {counts[k]} exceeds cap 5"

    print("  ✓ PASSED")


# =====================================================================
# Test 11: Total capacity insufficient → enforce_capacity raises error
# =====================================================================
def test_enforce_capacity_infeasible():
    print("\n" + "=" * 60)
    print("TEST 11: enforce_capacity raises on infeasible total capacity")
    print("=" * 60)

    N, K = 10, 2
    capacities = torch.tensor([3, 3])  # total=6 < N=10

    P_ell = torch.softmax(torch.randn(N, K), dim=1)

    try:
        enforce_capacity(P_ell, capacities)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Correctly raised: {e}")
        print("  ✓ PASSED")


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Capacity Constraint Test Suite")
    print("=" * 60)

    test_lambda_cap_derivation()
    test_sharpening_underestimation()
    test_no_penalty_within_capacity()
    test_penalty_fires_over_capacity()
    test_penalty_quadratic_growth()
    test_gradient_flow()
    test_enforce_capacity_basic()
    test_enforce_capacity_runner_up()
    test_enforce_capacity_sequence()
    test_four_techs()
    test_enforce_capacity_infeasible()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
