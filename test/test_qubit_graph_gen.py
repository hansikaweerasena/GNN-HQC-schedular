"""
test_qubit_graph_gen.py

Tests for the new per-layer backbone graph input pipeline.
(qubit_interaction_graph.py)

Checks:
  1. Output structure — T tuples for T layers, correct array shapes
  2. Node feature values — all in [0,1], correct dim (NODE_FEAT_DIM=16)
  3. Edge feature values — all in [0,1], correct dim (EDGE_FEAT_DIM=5)
  4. Backbone edge presence — edge (u,v) exists iff pair active in [t-W_long, t+W_long]
  5. Backbone edge absence — no edge for pairs outside the long window
  6. active_now correctness — edge_attr[:,0] == 1 iff pair has gate in layer t
  7. Empty layer handling — layers with no 2Q gates in window have empty edge_index
  8. Window size derivation — correct W_short/W_long from cost config kappa
  9. Isolated qubit — qubit with no gates anywhere has is_idle_now=1 for all layers
 10. Layer position — monotonically increases from 0 to 1
"""

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import math

from src.circuit_generation import generate_random_circuit_custom
from src.circuit_representation import CircuitRepresentation
from src.qubit_interaction_graph import (
    build_layer_graph_arrays,
    compute_window_sizes_from_config,
    NODE_FEAT_DIM,
    EDGE_FEAT_DIM,
)


# =============================================================================
# Helpers
# =============================================================================

PASS = "  ✓"
FAIL = "  ✗"

def check(condition: bool, msg: str):
    tag = PASS if condition else FAIL
    print(f"{tag}  {msg}")
    if not condition:
        raise AssertionError(f"FAILED: {msg}")


def make_circuit(n_qubits=10, depth=20, seed=42):
    qc  = generate_random_circuit_custom(num_qubits=n_qubits, depth=depth,
                                         gate_density=0.5, seed=seed)
    rep = CircuitRepresentation(qc)
    return rep


def minimal_cost_config(kappa=3.0):
    """Minimal dict matching cost_config_v3.json structure."""
    return {
        "techs": [
            {"name": "sc", "routing": {"kappa": kappa}},
            {"name": "na", "routing": {"kappa": 0, "all_to_all": True}},
        ]
    }


# =============================================================================
# Test 1: Output structure
# =============================================================================

def test_output_structure():
    print("\n--- Test 1: Output structure ---")
    rep = make_circuit()
    T   = len(rep.layers)
    N   = rep.num_qubits
    arrays = build_layer_graph_arrays(rep, w_short=3, w_long=6)

    check(len(arrays) == T, f"len(arrays)={len(arrays)} == T={T}")

    for t, (x, ei, ea) in enumerate(arrays):
        check(x.shape == (N, NODE_FEAT_DIM),
              f"layer {t}: x.shape={x.shape} == ({N},{NODE_FEAT_DIM})")
        check(ei.shape[0] == 2,
              f"layer {t}: edge_index has 2 rows, got {ei.shape[0]}")
        E = ei.shape[1]
        check(ea.shape == (E, EDGE_FEAT_DIM),
              f"layer {t}: edge_attr.shape={ea.shape} == ({E},{EDGE_FEAT_DIM})")


# =============================================================================
# Test 2: Node feature range
# =============================================================================

def test_node_feature_range():
    print("\n--- Test 2: Node feature values in [0,1] ---")
    rep    = make_circuit()
    arrays = build_layer_graph_arrays(rep, w_short=3, w_long=6)

    for t, (x, _, _) in enumerate(arrays):
        lo, hi = float(x.min()), float(x.max())
        check(lo >= -1e-6 and hi <= 1.0 + 1e-6,
              f"layer {t}: node features in [0,1] (min={lo:.4f}, max={hi:.4f})")


# =============================================================================
# Test 3: Edge feature range
# =============================================================================

def test_edge_feature_range():
    print("\n--- Test 3: Edge feature values in [0,1] ---")
    rep    = make_circuit()
    arrays = build_layer_graph_arrays(rep, w_short=3, w_long=6)

    for t, (_, ei, ea) in enumerate(arrays):
        if ea.shape[0] == 0:
            continue
        lo, hi = float(ea.min()), float(ea.max())
        check(lo >= -1e-6 and hi <= 1.0 + 1e-6,
              f"layer {t}: edge features in [0,1] (min={lo:.4f}, max={hi:.4f})")


# =============================================================================
# Test 4: Backbone edge presence — pair in window => edge exists
# =============================================================================

def test_backbone_presence():
    print("\n--- Test 4: Backbone edge presence ---")
    from qiskit import QuantumCircuit

    # Craft a small circuit with a single 2Q gate at layer 0 only
    # Use Qiskit directly for precise control
    qc = QuantumCircuit(4)
    qc.cx(0, 1)   # layer 0 — only 2Q gate
    qc.x(2)       # layer 0 — 1Q on qubit 2 (keeps layer non-empty)
    # layers 1..5: only 1Q gates, no 2Q
    for _ in range(6):
        qc.x(0)

    rep    = CircuitRepresentation(qc)
    w_long = 3
    arrays = build_layer_graph_arrays(rep, w_short=1, w_long=w_long)

    T = len(rep.layers)
    for t, (_, ei, ea) in enumerate(arrays):
        pair_in_window = (t <= w_long)   # gate at t=0, visible for t in [0, w_long]
        has_edge = ei.shape[1] > 0
        if pair_in_window:
            check(has_edge,
                  f"layer {t}: pair (0,1) in window => edge must exist")
        else:
            check(not has_edge,
                  f"layer {t}: pair (0,1) outside window => no edge")


# =============================================================================
# Test 5: active_now correctness
# =============================================================================

def test_active_now():
    print("\n--- Test 5: active_now edge feature ---")
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(4)
    qc.cx(0, 1)   # layer 0: pair (0,1) active
    qc.x(0)       # layer 1: no 2Q gate
    qc.cx(0, 1)   # layer 2: pair (0,1) active again

    rep    = CircuitRepresentation(qc)
    arrays = build_layer_graph_arrays(rep, w_short=1, w_long=2)

    T = len(rep.layers)
    active_layers = {0, 2}

    for t, (_, ei, ea) in enumerate(arrays):
        if ei.shape[1] == 0:
            continue
        # Find the edge for pair (0,1) — both directions present
        us, vs = ei[0], ei[1]
        pair_mask = ((us == 0) & (vs == 1)) | ((us == 1) & (vs == 0))
        if not pair_mask.any():
            continue
        active_now_val = float(ea[pair_mask, 0].mean())
        should_be_active = t in active_layers
        if should_be_active:
            check(active_now_val > 0.5,
                  f"layer {t}: active_now should be 1 (got {active_now_val:.2f})")
        else:
            check(active_now_val < 0.5,
                  f"layer {t}: active_now should be 0 (got {active_now_val:.2f})")


# =============================================================================
# Test 6: Isolated qubit is always idle
# =============================================================================

def test_isolated_qubit():
    print("\n--- Test 6: Isolated qubit always idle ---")
    from qiskit import QuantumCircuit

    # qubit 3 has no gates at all
    qc = QuantumCircuit(4)
    for _ in range(5):
        qc.cx(0, 1)
        qc.x(2)

    rep    = CircuitRepresentation(qc)
    arrays = build_layer_graph_arrays(rep, w_short=2, w_long=4)

    for t, (x, _, _) in enumerate(arrays):
        # qubit 3 = index 3
        is_idle_now = float(x[3, 0])
        is_1q_now   = float(x[3, 1])
        is_2q_now   = float(x[3, 2])
        check(is_idle_now == 1.0 and is_1q_now == 0.0 and is_2q_now == 0.0,
              f"layer {t}: qubit 3 isolated => is_idle=1, is_1q=0, is_2q=0")


# =============================================================================
# Test 7: Layer position monotonically increases
# =============================================================================

def test_layer_position():
    print("\n--- Test 7: layer_position monotonically increases ---")
    rep    = make_circuit(depth=15)
    arrays = build_layer_graph_arrays(rep, w_short=3, w_long=6)
    T      = len(arrays)

    positions = [float(x[0, 15]) for x, _, _ in arrays]  # feature index 15
    check(abs(positions[0]) < 1e-6,
          f"layer_position[0] == 0.0 (got {positions[0]:.4f})")
    check(abs(positions[-1] - 1.0) < 1e-6,
          f"layer_position[T-1] == 1.0 (got {positions[-1]:.4f})")

    for i in range(1, T):
        check(positions[i] >= positions[i - 1],
              f"layer_position monotone at t={i}")


# =============================================================================
# Test 8: Window size derivation from config
# =============================================================================

def test_window_sizes():
    print("\n--- Test 8: Window size derivation ---")

    cfg = minimal_cost_config(kappa=2.3)
    ws, wl = compute_window_sizes_from_config(cfg)
    check(ws == math.ceil(2.3), f"W_short={ws} == ceil(2.3)={math.ceil(2.3)}")
    check(wl == 2 * ws,         f"W_long={wl} == 2*W_short={2*ws}")

    # All-to-all only => fallback kappa=3
    cfg_a2a = {"techs": [{"name": "na", "routing": {"kappa": 0, "all_to_all": True}}]}
    ws2, wl2 = compute_window_sizes_from_config(cfg_a2a)
    check(ws2 == 3 and wl2 == 6,
          f"All-to-all fallback => W_short=3, W_long=6 (got {ws2},{wl2})")

    # Multiple techs — takes max kappa
    cfg_multi = minimal_cost_config(kappa=5.0)
    ws3, wl3 = compute_window_sizes_from_config(cfg_multi)
    check(ws3 == 5 and wl3 == 10,
          f"kappa=5 => W_short=5, W_long=10 (got {ws3},{wl3})")


# =============================================================================
# Test 9: Undirected edges — both directions present
# =============================================================================

def test_undirected_edges():
    print("\n--- Test 9: Both edge directions present for GATv2 ---")
    rep    = make_circuit()
    arrays = build_layer_graph_arrays(rep, w_short=3, w_long=6)

    for t, (_, ei, ea) in enumerate(arrays):
        if ei.shape[1] == 0:
            continue
        E_half = ei.shape[1] // 2
        # First half is u->v, second half is v->u
        us_fwd = ei[0, :E_half]
        vs_fwd = ei[1, :E_half]
        us_rev = ei[0, E_half:]
        vs_rev = ei[1, E_half:]
        check(
            np.array_equal(us_fwd, vs_rev) and np.array_equal(vs_fwd, us_rev),
            f"layer {t}: reverse edges match forward edges"
        )
        # Edge features must be identical for both directions
        check(
            np.allclose(ea[:E_half], ea[E_half:]),
            f"layer {t}: edge_attr same for both directions"
        )
        break  # checking one layer with edges is sufficient


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  test_qubit_graph_gen.py — new per-layer backbone pipeline")
    print("=" * 60)

    tests = [
        test_output_structure,
        test_node_feature_range,
        test_edge_feature_range,
        test_backbone_presence,
        test_active_now,
        test_isolated_qubit,
        test_layer_position,
        test_window_sizes,
        test_undirected_edges,
    ]

    passed, failed = 0, 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {fn.__name__}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
