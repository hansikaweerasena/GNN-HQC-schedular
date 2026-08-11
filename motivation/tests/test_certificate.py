"""The Phase-0 certificate, ported from the NB1-NB3 checkpoints and NB6/NB6b.

This is the instrument's calibration record. If any test here fails, every
number the harness produces is suspect. Run it before every experiment batch:

    pytest tests/ -q

Structure
---------
NB1 / NB2 / NB3 checkpoints  -- the per-notebook go/no-go asserts
ST1-ST8                      -- the v3 regression. Hand-calibrated under
                                t_remote = 0, and none of them asserts a
                                remote-gate DURATION, so under v4 they must be
                                BIT-IDENTICAL. Any movement here means the
                                refactor changed physics, not just names.
ST9-ST15                     -- the v4 semantics: nonzero t_comm, spectator
                                dephasing, module-keyed remote branch,
                                fidelity-only movement, block invariants,
                                clock/decoherence consistency, destination-tech
                                timing.
NB6b                         -- the two routing guards: silent on Phase-1
                                shapes, loud where the bug is real.
"""

import random

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit.quantum_info import (
    DensityMatrix, Statevector, state_fidelity, partial_trace,
    average_gate_fidelity, random_statevector,
)

from mosaic_aer import (
    dephasing_channel, gate_infidelity_channel, plus_coherence, idle_coherence_on_aer,
    TECHS, COMM, t_comm, t_move, HW, Module, noiseless_techs, noiseless_comm, movement_mode,
    homogeneous_machine, heterogeneous_machine,
    route, un_permute, segment_blocks, lower, aer_fidelity, score, pareto_front,
    drift_check, H, CX, mod,
    circuit_to_layers, from_qasm, layers_to_circuit, ideal_state,
    make_layers, to_cx_basis, validate_layers,
)

ATOL = 1e-4


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def coh_of(qc, wire):
    """Coherence |rho01| / 0.5 of one wire of a lowered circuit."""
    dm = DensityMatrix(
        AerSimulator(method="density_matrix").run(qc).result().data(0)["density_matrix"])
    r = partial_trace(dm, [w for w in range(qc.num_qubits) if w != wire])
    return abs(r.data[0, 1]) / 0.5


def _ring(n):
    return CouplingMap([[i, (i + 1) % n] for i in range(n)]
                       + [[(i + 1) % n, i] for i in range(n)])


# =========================================================================
# NB1 -- noise primitives
# =========================================================================

class TestNB1:

    def test_coherence_law_exact(self):
        for t in [20_000.0, 100_000.0]:
            assert np.isclose(plus_coherence(dephasing_channel(80_000.0, t)),
                              np.exp(-t / 80_000.0), atol=1e-6)

    def test_zero_idle_is_none(self):
        assert dephasing_channel(80_000.0, 0.0) is None
        with pytest.raises(ValueError):
            dephasing_channel(80_000.0, -1.0)

    def test_t2_infinite_preserves_coherence(self):
        assert np.isclose(plus_coherence(dephasing_channel(1e18, 1e5)), 1.0, atol=1e-9)

    def test_avg_gate_fidelity_convention(self):
        for F, n in [(0.9999, 1), (0.9990, 2), (0.9900, 1), (0.9970, 2)]:
            agf = average_gate_fidelity(gate_infidelity_channel(F, n).to_quantumchannel())
            assert np.isclose(agf, F), f"avg_gate_fid != F for F={F}, n={n}"

    def test_f_one_is_identity(self):
        assert np.isclose(
            average_gate_fidelity(gate_infidelity_channel(1.0, 1).to_quantumchannel()), 1.0)

    def test_pure_state_fidelity_equals_F(self):
        for F, n in [(0.99, 1), (0.997, 2)]:
            sop = gate_infidelity_channel(F, n).to_quantumchannel()
            psi = random_statevector(2 ** n, seed=3)
            sf = state_fidelity(DensityMatrix(psi).evolve(sop), DensityMatrix(psi))
            assert np.isclose(sf, F, atol=1e-9)

    def test_composition_is_near_multiplicative(self):
        """Two F=0.99 gates -> 0.9802 vs F^2=0.9801. The 1e-4 excess is depolarizing's
        residual overlap on the failure branch (second order in infidelity), which is
        why the channel convention is NOT an EFCL<->Aer divergence source."""
        sop = gate_infidelity_channel(0.99, 1).to_quantumchannel()
        psi = random_statevector(2, seed=1)
        rho = DensityMatrix(psi)
        assert np.isclose(state_fidelity(rho.evolve(sop).evolve(sop), rho), 0.99 ** 2, atol=1e-3)

    def test_smoke_number_on_real_backend(self):
        """SC |+> idling through one TI 2Q gate: exp(-1.25) = 0.2865."""
        assert np.isclose(idle_coherence_on_aer(80_000.0, 100_000.0), np.exp(-1.25), atol=1e-6)


# =========================================================================
# NB2 -- hardware + communication model
# =========================================================================

class TestNB2:

    def test_hardcoded_specs(self):
        assert TECHS["sc"].f2q == 0.999 and TECHS["sc"].T2 == 80_000.0
        assert TECHS["na"].t2q == 2_000.0 and TECHS["ti"].t2q == 100_000.0
        assert TECHS["ti"].T2 == 2_000_000.0

    def test_sc_topology_by_capacity(self):
        ring = {tuple(sorted(e)) for e in HW.coupling_map("sc", 4).get_edges()}
        assert ring == {(0, 1), (1, 2), (2, 3), (0, 3)}
        assert np.isclose(HW.avg_degree("sc", 4), 2.0)
        assert np.isclose(HW.avg_degree("sc", 2), 1.0)
        assert HW.coupling_map("na", 4) is None and HW.coupling_map("ti", 2) is None

    def test_swap_cost(self):
        assert np.isclose(HW.swap_fidelity("sc"), 0.999 ** 3)
        assert np.isclose(HW.swap_duration("sc"), 600.0)

    def test_machine_builders(self):
        M = homogeneous_machine("sc", 2, 4)
        assert M.n_qubits == 8
        assert [m.qubits for m in M.modules] == [(0, 1, 2, 3), (4, 5, 6, 7)]
        assert [m.tech for m in M.modules] == ["sc", "sc"]
        Hm = heterogeneous_machine([("sc", 4), ("na", 4)])
        assert [m.tech for m in Hm.modules] == ["sc", "na"]

    def test_comm_model_shape(self):
        assert "t_remote" not in COMM, "t_remote must not exist: it conflated two quantities"
        assert set(COMM) == {"f_comm", "f_move", "t_move_derived", "t_move_visible"}
        # scheduled boundary => pre-purified Bell pair => movement is the cleaner primitive
        assert COMM["f_move"] > COMM["f_comm"]

    def test_transfer_latency_is_exposed_by_default(self):
        """v5 default: movement costs critical-path time, derived per pair."""
        assert COMM["t_move_derived"] is True
        assert COMM["t_move_visible"] == 0.0, "the legacy scalar is the foil, and stays 0"

    def test_t_move_is_source_side_and_asymmetric(self):
        """Only the SOURCE performs a 2Q operation in state teleportation (the Bell-basis
        measurement); the destination applies 1Q Pauli corrections. So the latency is the
        origin's gate time, and the direction matters."""
        for a in TECHS:
            for b in TECHS:
                assert t_move(a, b) == TECHS[a].t2q
        assert t_move("sc", "na") == 200.0
        assert t_move("na", "sc") == 2_000.0
        assert t_move("sc", "na") != t_move("na", "sc"), "the rule must NOT be symmetric"

    def test_move_asymmetry_separates_the_two_pairs(self):
        """The paper's Act I / Act II table. Parking costs the same in both pairs;
        RETRIEVAL differs by 100x, and that is what makes SC+TI degenerate while SC+NA
        stays workable. A max-rule t_move would hide this."""
        T2sc = TECHS["sc"].T2
        assert t_move("sc", "ti") == t_move("sc", "na")            # parking: identical
        assert t_move("ti", "sc") / T2sc > 1.0                     # retrieval: lethal
        assert t_move("na", "sc") / T2sc < 0.05                    # retrieval: survivable
        # Retrieval cost is exactly the origin's 2Q gate time, so the ratio between the
        # pairs is the ratio of TI's to NA's gate speed. Derived, not hardcoded, so this
        # keeps holding when the tech table is refrozen.
        assert (t_move("ti", "sc") / t_move("na", "sc")
                == TECHS["ti"].t2q / TECHS["na"].t2q >= 50.0)

    def test_t_move_and_t_comm_are_different_rules(self):
        """A teleported GATE fires a local CNOT at BOTH endpoints in parallel, so it waits
        for the slower one (max). A state TRANSFER measures only at the source. Different
        primitives, different rules -- these were one scalar (`t_remote`) until v4."""
        assert t_move is not t_comm
        assert t_comm("sc", "ti") == 100_000.0 and t_move("sc", "ti") == 200.0
        for a in TECHS:
            for b in TECHS:
                assert t_comm(a, b) == max(TECHS[a].t2q, TECHS[b].t2q)
                assert t_move(a, b) <= t_comm(a, b)

    def test_movement_mode_restores(self):
        before = dict(COMM)
        with movement_mode(derived=False, visible=1234.0):
            assert COMM["t_move_derived"] is False and COMM["t_move_visible"] == 1234.0
        assert COMM == before

    def test_t_comm_derived_per_pair(self):
        assert t_comm("sc", "sc") == 200.0        # homogeneous SC gets a CHEAP remote gate
        assert t_comm("na", "na") == 2_000.0
        assert t_comm("sc", "na") == 2_000.0      # remote gate is time-neutral vs slow endpoint
        assert t_comm("ti", "ti") == 100_000.0
        assert t_comm("sc", "ti") == 100_000.0
        assert t_comm("sc", "ti") == t_comm("ti", "sc")
        for a in TECHS:
            for b in TECHS:
                assert t_comm(a, b) >= max(TECHS[a].t2q, TECHS[b].t2q)

    def test_the_asymmetry_that_carries_the_paper(self):
        """SC+TI is the only pair where a remote gate outlives the spectator's coherence."""
        assert t_comm("sc", "ti") / TECHS["sc"].T2 > 1.0
        assert t_comm("sc", "sc") / TECHS["sc"].T2 < 0.01

    def test_noiseless_techs_restores(self):
        before = dict(TECHS)
        with noiseless_techs():
            assert TECHS["sc"].f2q == 1.0 and TECHS["sc"].T2 == 1e18
            assert TECHS["sc"].t2q == before["sc"].t2q, "gate TIMES must survive"
        assert TECHS == before

    def test_config_drift(self):
        """Skips cleanly when the EFCL configs are not on this machine."""
        drift_check(verbose=False)


# =========================================================================
# NB3 -- router
# =========================================================================

class TestNB3:

    @staticmethod
    def _dense():
        qc = QuantumCircuit(4)
        for a in range(4):
            qc.h(a)
        for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
            qc.cx(a, b)
        return qc

    @staticmethod
    def _asym():
        qc = QuantumCircuit(4)
        for i, ang in enumerate([0.3, 0.7, 1.1, 1.9]):
            qc.rx(ang, i)
        for a, b in [(0, 2), (1, 3), (0, 1), (2, 3)]:   # 0-2, 1-3 non-adjacent -> swaps
            qc.cx(a, b)
        return qc

    def test_all_to_all_passthrough(self):
        d = self._dense()
        r = route(d, None)
        assert r.swap_count == 0 and r.routed_circuit is d and r.final_layout == [0, 1, 2, 3]

    def test_dense_block_forces_swaps_and_preserves_logical_gates(self):
        d = self._dense()
        r = route(d, _ring(4))
        assert r.swap_count > 0
        nonswap = lambda c: {k: v for k, v in c.count_ops().items() if k != "swap"}
        assert nonswap(d) == nonswap(r.routed_circuit)

    def test_un_permute_restores_and_is_non_vacuous(self):
        a = self._asym()
        r = route(a, _ring(4))
        raw = state_fidelity(Statevector(a), Statevector(r.routed_circuit))
        fixed = state_fidelity(Statevector(a),
                               un_permute(Statevector(r.routed_circuit), r.final_layout))
        assert raw < 0.99, "test is vacuous if routing did not permute"
        assert np.isclose(fixed, 1.0, atol=1e-9)

    def test_un_permute_direction_is_correct(self):
        """Controlled 3-cycle: non-involution, so the direction is unambiguous."""
        U = QuantumCircuit(4)
        for i, ang in enumerate([0.3, 0.7, 1.1, 1.9]):
            U.rx(ang, i)
        U.cx(0, 1)
        U.cx(2, 3)
        routed = U.copy()
        routed.swap(0, 1)
        routed.swap(1, 2)
        st = [0, 1, 2, 3]
        for a, b in [(0, 1), (1, 2)]:
            st[a], st[b] = st[b], st[a]
        final_perm = [st.index(v) for v in range(4)]
        inverse = [final_perm.index(i) for i in range(4)]
        assert final_perm != inverse
        sv_o, sv_r = Statevector(U), Statevector(routed)
        assert np.isclose(state_fidelity(sv_o, un_permute(sv_r, final_perm)), 1.0, atol=1e-9)
        assert state_fidelity(sv_o, un_permute(sv_r, inverse)) < 0.99

    def test_cap2_edge_never_swaps(self):
        edge = QuantumCircuit(2)
        edge.h(0)
        edge.cx(0, 1)
        assert route(edge, CouplingMap([[0, 1], [1, 0]])).swap_count == 0

    def test_phase2_stub_is_guarded(self):
        with pytest.raises(NotImplementedError):
            route(self._dense(), _ring(4), initial_layout=[0, 1, 2, 3])

    def test_measurements_rejected(self):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.measure(0, 0)
        with pytest.raises(ValueError):
            route(qc, _ring(2))


# =========================================================================
# Block segmentation
# =========================================================================

class TestSegmentation:

    def test_two_blocks(self):
        sch = [{0: 0}] * 10 + [{0: 1}] * 11
        assert segment_blocks(sch) == [(0, 9), (10, 20)]

    def test_per_layer_switch_is_layer_sync_limit(self):
        assert segment_blocks([{0: 0}, {0: 1}, {0: 0}]) == [(0, 0), (1, 1), (2, 2)]

    def test_no_switch_is_pure_asap_limit(self):
        assert segment_blocks([{0: 0}] * 4) == [(0, 3)]

    def test_ST13_assignment_constant_inside_every_block(self):
        r = random.Random(3)
        for _ in range(200):
            L = r.randint(1, 8)
            s = [{q: r.randint(0, 1) for q in range(3)} for _ in range(L)]
            for (a, b) in segment_blocks(s):
                for li in range(a, b + 1):
                    assert s[li] == s[a]


# =========================================================================
# ST1-ST8 -- v3 regression. MUST be bit-identical.
# =========================================================================

class TestST1toST8:

    def test_ST1_noiseless(self):
        with noiseless_techs():
            f, _ = aer_fidelity([[('2q', 0, 1, CX)]], [{0: 0, 1: 0}], [mod(0, 'ti', [0, 1])])
        assert np.isclose(f, 1.0, atol=1e-9)

    def test_ST2_ti_2q(self):
        f, _ = aer_fidelity([[('2q', 0, 1, CX)]], [{0: 0, 1: 0}], [mod(0, 'ti', [0, 1])])
        assert np.isclose(f, 0.9997, atol=ATOL)

    def test_ST3_sc_idle(self):
        qc, l2w, _ = lower([[('1q', 0, H), ('2q', 2, 3, CX)]], [{0: 0, 1: 0, 2: 1, 3: 1}],
                           [mod(0, 'sc', [0, 1]), mod(1, 'ti', [2, 3])])
        assert np.isclose(coh_of(qc, l2w[0]), np.exp(-99980 / 80000), atol=ATOL)

    def test_ST4_f_comm(self):
        f, _ = aer_fidelity([[('2q', 0, 1, CX)]], [{0: 0, 1: 1}],
                            [mod(0, 'sc', [0]), mod(1, 'na', [1])])
        assert np.isclose(f, 0.95, atol=ATOL)

    def test_ST4_fidelity_invariant_to_t_comm(self):
        """Regression isolation: t_comm moves TIME, never the fidelity of this circuit."""
        f, _ = aer_fidelity([[('2q', 0, 1, CX)]], [{0: 0, 1: 1}],
                            [mod(0, 'sc', [0]), mod(1, 'na', [1])],
                            t_comm_fn=lambda a, b: 0.0)
        assert abs(f - 0.95) < 1e-9

    def test_ST5_routing(self):
        f, d = aer_fidelity(
            [[('2q', 0, 2, CX)], [('2q', 1, 3, CX)], [('2q', 0, 1, CX)], [('2q', 2, 3, CX)]],
            [{0: 0, 1: 0, 2: 0, 3: 0}] * 4, [mod(0, 'sc', [0, 1, 2, 3])])
        assert d.swap_count > 0 and f > 0.9

    def test_ST6_parked_idle_at_ti_T2(self):
        qc, l2w, _ = lower([[('1q', 0, H)], [('2q', 2, 3, CX)]],
                           [{0: 0, 2: 1, 3: 1}, {0: 1, 2: 1, 3: 1}],
                           [mod(0, 'sc', [0, 1]), mod(1, 'ti', [2, 3, 4])])
        exp = (2 * 0.9999 - 1) * (2 * 0.99 - 1) * np.exp(-100000 / 2_000_000)
        assert np.isclose(coh_of(qc, l2w[0]), exp, atol=ATOL)

    def test_ST7_sync_idle_charged_at_pre_move_T2(self):
        """The mover pays for its own slack at its WORSE coherence time before it leaves.
        Pure ASAP would charge 0. This is why t_move_visible = 0 is not a subsidy."""
        qc, l2w, d = lower([[('1q', 0, H), ('2q', 2, 3, CX)], [('2q', 2, 3, CX)]],
                           [{0: 0, 2: 1, 3: 1}, {0: 1, 2: 1, 3: 1}],
                           [mod(0, 'sc', [0, 1]), mod(1, 'ti', [2, 3, 4])])
        exp = ((2 * 0.9999 - 1) * np.exp(-99980 / 80000)
               * (2 * 0.99 - 1) * np.exp(-100000 / 2_000_000))
        assert np.isclose(coh_of(qc, l2w[0]), exp, atol=ATOL)
        assert abs(d.sync_idle.get(0, 0) - 99980) < 1

    def test_ST8_module_scope_leaves_untouched_module_running(self):
        layers = [[('1q', 0, H), ('2q', 2, 3, CX), ('2q', 5, 6, CX)], [('2q', 5, 6, CX)]]
        sch = [{0: 0, 2: 1, 3: 1, 5: 2, 6: 2}, {0: 1, 2: 1, 3: 1, 5: 2, 6: 2}]
        mods = [mod(0, 'sc', [0, 1]), mod(1, 'ti', [2, 3, 4]), mod(2, 'na', [5, 6])]
        _, _, dm_ = lower(layers, sch, mods, sync_scope="module")
        _, _, dg = lower(layers, sch, mods, sync_scope="global")
        assert dm_.sync_idle.get(5, 0) == 0
        assert dg.sync_idle.get(5, 0) > 0

    def test_feasibility_gate(self):
        _, _, d = lower([[('1q', 0, H)]], [{0: 0, 1: 0, 2: 0}], [mod(0, 'ti', [0, 1])])
        assert d.feasible is False
        assert score([[('1q', 0, H)]], [{0: 0, 1: 0, 2: 0}], [mod(0, 'ti', [0, 1])]) is None


# =========================================================================
# ST9-ST15 -- the v4 certificate
# =========================================================================

class TestST9toST15:

    def test_ST9_remote_gate_advances_both_participants(self):
        f, d = aer_fidelity([[('2q', 0, 1, CX)]], [{0: 0, 1: 1}],
                            [mod(0, 'sc', [0]), mod(1, 'ti', [1])])
        assert np.isclose(f, 0.95, atol=ATOL), "aggregate f_comm; no separate f2q"
        assert d.makespan == 100_000.0
        assert d.comm_time == 100_000.0
        assert d.comm_count == 1

    def test_ST10_spectator_dephases_through_a_remote_gate(self):
        """The paper's number. Under t_remote = 0 this was 0 ns and coherence 1.0 --
        EFCL was blind to 97% of a remote gate's cost."""
        qc, l2w, d = lower([[('1q', 2, H), ('2q', 0, 1, CX)]], [{0: 0, 1: 1, 2: 0}],
                           [mod(0, 'sc', [0, 1]), mod(1, 'ti', [2])])
        exp = (2 * 0.9999 - 1) * np.exp(-99980 / 80000)
        assert np.isclose(coh_of(qc, l2w[2]), exp, atol=ATOL)
        assert abs(d.idle_time.get(2, 0) - 99980) < 1

    def test_ST11_remote_branch_keys_on_module_not_technology(self):
        """2xSC baseline honesty: a cross-module SC-SC gate is remote and pays f_comm.
        Otherwise the homogeneous baseline is silently monolithic and wins trivially."""
        fr, dr = aer_fidelity([[('2q', 0, 1, CX)]], [{0: 0, 1: 1}],
                              [mod(0, 'sc', [0]), mod(1, 'sc', [1])])
        fl, dl = aer_fidelity([[('2q', 0, 1, CX)]], [{0: 0, 1: 0}], [mod(0, 'sc', [0, 1])])
        assert np.isclose(fr, 0.95, atol=ATOL)
        assert np.isclose(fl, 0.999, atol=ATOL)
        assert dr.comm_count == 1 and dr.comm_time == 200.0 and dl.comm_count == 0
        assert dr.makespan == dl.makespan == 200.0, \
            "homogeneous SC pays for distribution in fidelity, not in time"

    def test_ST12a_exactly_one_f_move_per_mover(self):
        with noiseless_techs():
            qc, l2w, _ = lower([[('1q', 0, H), ('2q', 2, 3, CX)], [('2q', 2, 3, CX)]],
                               [{0: 0, 2: 1, 3: 1}, {0: 1, 2: 1, 3: 1}],
                               [mod(0, 'sc', [0, 1]), mod(1, 'ti', [2, 3, 4])])
            # 0.9604 would mean f_move applied twice
            assert np.isclose(coh_of(qc, l2w[0]), 2 * 0.99 - 1, atol=ATOL)

    # --- the ST12b pair: same circuit, one schedule that moves and one that pins ---
    _MOVE_LAYERS = [[('1q', 0, H), ('2q', 2, 3, CX)], [('2q', 2, 3, CX)]]
    _MOVE_MODS = [mod(0, 'sc', [0, 1]), mod(1, 'ti', [2, 3, 4])]
    _MOVE_SCHED = [{0: 0, 2: 1, 3: 1}, {0: 1, 2: 1, 3: 1}]
    _PIN_SCHED = [{0: 0, 2: 1, 3: 1}] * 2

    def _moved_vs_pinned(self):
        _, _, moved = lower(self._MOVE_LAYERS, self._MOVE_SCHED, self._MOVE_MODS)
        _, _, pinned = lower(self._MOVE_LAYERS, self._PIN_SCHED, self._MOVE_MODS)
        return moved, pinned

    def test_ST12b_movement_costs_exactly_the_derived_transfer_latency(self):
        """v5. Was `moved.makespan == pinned.makespan` under the overlapped model."""
        moved, pinned = self._moved_vs_pinned()
        assert moved.move_count == 1 and pinned.move_count == 0
        assert moved.makespan == pinned.makespan + t_move('sc', 'ti')
        assert moved.move_time == t_move('sc', 'ti')
        assert pinned.move_time == 0.0

    def test_ST12b_legacy_overlapped_model_is_recovered_exactly(self):
        """The v4 number must still be reachable, or we have deleted the foil rather than
        demoted it. This is the sensitivity arm for the assumption ledger."""
        with movement_mode(derived=False, visible=0.0):
            moved, pinned = self._moved_vs_pinned()
        assert moved.makespan == pinned.makespan
        assert moved.move_count == 1 and moved.move_time == 0.0

    def test_ST16_boundary_clears_on_the_slowest_concurrent_transfer(self):
        """Two movers with different (from, to) pairs at one boundary: the boundary
        advances by the max, not the sum and not the first."""
        layers = [[('1q', 0, H), ('1q', 4, H)], [('1q', 0, H), ('1q', 4, H)]]
        mods = [mod(0, 'sc', [0, 1]), mod(1, 'na', [2, 3]), mod(2, 'ti', [4, 5])]
        sched = [{0: 0, 4: 2}, {0: 1, 4: 1}]        # q0: sc->na (2 us), q4: ti->na (100 us)
        _, _, d = lower(layers, sched, mods, sync_scope="global")
        assert d.move_count == 2
        assert d.move_time == max(t_move('sc', 'na'), t_move('ti', 'na')) == 100_000.0

    def test_ST17_non_mover_in_an_affected_module_pays_the_wait_at_its_own_T2(self):
        """A spectator that never moves still loses coherence to somebody else's transfer.
        This is the cost the overlapped model was hiding."""
        layers = [[('1q', 0, H), ('1q', 1, H)], [('1q', 1, H)]]
        mods = [mod(0, 'sc', [0, 1]), mod(1, 'ti', [2, 3])]
        sched = [{0: 0, 1: 0}, {0: 1, 1: 0}]        # q0 moves sc->ti, q1 stays on SC
        _, _, exposed = lower(layers, sched, mods)
        with movement_mode(derived=False, visible=0.0):
            _, _, overlapped = lower(layers, sched, mods)
        extra = exposed.idle_time[1] - overlapped.idle_time.get(1, 0.0)
        assert np.isclose(extra, t_move('sc', 'ti'))
        assert np.isclose(exposed.sync_idle[1] - overlapped.sync_idle.get(1, 0.0),
                          t_move('sc', 'ti')), "the wait belongs in the sync bucket"

    def test_ST18_mover_is_not_dephased_during_its_own_transfer(self):
        """f_move is the aggregate fidelity of the WHOLE transfer primitive, so charging
        T2 dephasing over t_move on top of it would double-count. The mover's idle budget
        must be identical under both models; only its busy budget grows."""
        layers = [[('1q', 0, H), ('1q', 1, H)], [('1q', 1, H)]]
        mods = [mod(0, 'sc', [0, 1]), mod(1, 'ti', [2, 3])]
        sched = [{0: 0, 1: 0}, {0: 1, 1: 0}]
        _, _, exposed = lower(layers, sched, mods)
        with movement_mode(derived=False, visible=0.0):
            _, _, overlapped = lower(layers, sched, mods)
        assert np.isclose(exposed.idle_time[0], overlapped.idle_time[0])
        assert np.isclose(exposed.busy_time[0] - overlapped.busy_time[0],
                          t_move('sc', 'ti'))

    def test_ST19_uniform_scalar_mode_still_works(self):
        """The legacy scalar knob is a real mode, not dead config."""
        with movement_mode(derived=False, visible=5_000.0):
            moved, pinned = self._moved_vs_pinned()
        assert moved.makespan == pinned.makespan + 5_000.0
        assert moved.move_time == 5_000.0

    def test_ST14_busy_plus_idle_equals_makespan_per_qubit(self):
        """Catches any silent clock advance. The pre-fix code advanced tav by t_remote at
        the block boundary with no dephasing call; this test fails on it."""
        def consistent(d):
            return all(abs(d.busy_time.get(q, 0.) + d.idle_time.get(q, 0.) - d.makespan) < 1e-6
                       for q in set(d.busy_time) | set(d.idle_time))

        _, _, dA = lower([[('1q', 0, H), ('2q', 2, 3, CX)], [('2q', 2, 3, CX)]],
                         [{0: 0, 2: 1, 3: 1}, {0: 1, 2: 1, 3: 1}],
                         [mod(0, 'sc', [0, 1]), mod(1, 'ti', [2, 3, 4])])
        _, _, dB = lower([[('1q', 0, H), ('2q', 2, 3, CX), ('2q', 5, 6, CX)], [('2q', 5, 6, CX)]],
                         [{0: 0, 2: 1, 3: 1, 5: 2, 6: 2}, {0: 1, 2: 1, 3: 1, 5: 2, 6: 2}],
                         [mod(0, 'sc', [0, 1]), mod(1, 'ti', [2, 3, 4]), mod(2, 'na', [5, 6])])
        _, _, dC = lower([[('2q', 0, 2, CX)], [('2q', 1, 3, CX)],
                          [('2q', 0, 1, CX)], [('2q', 2, 3, CX)]],
                         [{0: 0, 1: 0, 2: 0, 3: 0}] * 4, [mod(0, 'sc', [0, 1, 2, 3])])
        _, _, dD = lower([[('1q', 2, H), ('2q', 0, 1, CX)]], [{0: 0, 1: 1, 2: 0}],
                         [mod(0, 'sc', [0, 1]), mod(1, 'ti', [2])])
        for name, d in [("block boundary", dA), ("3-module", dB),
                        ("SC routing", dC), ("remote gate", dD)]:
            assert consistent(d), f"clock/decoherence inconsistency: {name}"

    def test_ST15_next_block_uses_destination_technology_gate_time(self):
        """20 ns of SC 1Q + the sc->ti transfer (200 ns, SOURCE-side) + one TI 2Q gate.
        The GATE must be charged at the destination's rate even though the TRANSFER was
        charged at the source's -- two different lookups, and swapping either one is a
        silent 500x error."""
        args = ([[('1q', 0, H)], [('2q', 0, 2, CX)]],
                [{0: 0, 2: 1}, {0: 1, 2: 1}],
                [mod(0, 'sc', [0, 1]), mod(1, 'ti', [2, 3])])
        _, _, d = lower(*args)
        assert d.makespan == 20.0 + t_move('sc', 'ti') + TECHS['ti'].t2q == 100_220.0, \
            "would be 220 if SC t2q leaked into the gate, 200020 if TI leaked into the move"
        with movement_mode(derived=False, visible=0.0):
            _, _, d0 = lower(*args)
        assert d0.makespan == 100_020.0, "v4 regression value must remain reachable"


# =========================================================================
# NB6b -- routing guards
# =========================================================================

class TestRoutingGuards:

    @staticmethod
    def _sweep(cap, trials, seed):
        rnd = random.Random(seed)
        fired = tried = 0
        for _ in range(trials):
            layers = []
            for _ in range(rnd.randint(2, 6)):
                qs = list(range(cap))
                rnd.shuffle(qs)
                lay = []
                while len(qs) >= 2 and rnd.random() < 0.8:
                    a, b = qs.pop(), qs.pop()
                    lay.append(('2q', a, b, CX))
                if lay:
                    layers.append(lay)
            if not layers:
                continue
            tried += 1
            try:
                lower(layers, [{i: 0 for i in range(cap)}] * len(layers),
                      [Module(0, 'sc', tuple(range(cap)))])
            except AssertionError:
                fired += 1
        return fired, tried

    def test_guard_silent_at_cap4(self):
        """Order-independence is GUARANTEED on a 4-cycle, so this is structural, not luck."""
        fired, tried = self._sweep(4, 300, 11)
        assert tried > 0 and fired == 0

    def test_guard_loud_at_cap6(self):
        fired, _ = self._sweep(6, 200, 7)
        assert fired > 0, "the cap>=5 gate-order bug must remain loud"

    def test_1A_shape_is_silent(self):
        layers = [[('2q', 0, 2, CX), ('2q', 4, 6, CX)],
                  [('2q', 1, 3, CX)],
                  [('2q', 0, 1, CX), ('2q', 3, 7, CX)]]
        sched = [{0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}] * 3
        f, d = aer_fidelity(layers, sched,
                            [Module(0, 'sc', (0, 1, 2, 3)), Module(1, 'na', (4, 5, 6, 7))])
        assert f is not None and 0.0 < f <= 1.0


# =========================================================================
# Carried placement + slot inheritance (v5 -- replaces the Gap-1 guard)
# =========================================================================

class TestCarriedPlacement:
    """An incoming teleported state is materialised at a physical site vacated by an
    outgoing state. Non-migrants keep their placement; migration induces no intra-module
    SWAPs. Each block is then routed from that carried placement."""

    # SC ring (wires 0-3) + NA (wires 4-7). Layer 0 forces SABRE swaps on the ring
    # (0-2 and 1-3 are ring diagonals), then q0 and q4 exchange modules, then more
    # SC-local work on the new residents.
    _LAYERS = [[('2q', 0, 2, CX), ('2q', 1, 3, CX)],
               [('2q', 0, 1, CX)],
               [('2q', 4, 1, CX), ('2q', 2, 3, CX)],
               [('2q', 4, 2, CX)]]
    _MODS = [Module(0, 'sc', (0, 1, 2, 3)), Module(1, 'na', (4, 5, 6, 7))]
    _BASE = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}
    _SWAPPED = {**_BASE, 0: 1, 4: 0}          # q0 -> NA, q4 -> SC (capacity-legal)
    _SCHED = [_BASE, _BASE, _SWAPPED, _SWAPPED]

    def test_migration_no_longer_raises(self):
        f, d = aer_fidelity(self._LAYERS, self._SCHED, self._MODS)
        assert f is not None and d.move_count == 2

    def test_noiseless_migration_round_trip_is_exactly_one(self):
        """THE test. Every channel is the identity, so any fidelity below 1 is pure
        bookkeeping: a gate on the wrong wire, or a migration that relabelled a slot
        without moving the quantum state with it."""
        with noiseless_techs(), noiseless_comm():
            f, d = aer_fidelity(self._LAYERS, self._SCHED, self._MODS)
        assert d.swap_count > 0, "vacuous unless the SC ring actually routed"
        assert d.migration_relabels > 0, "vacuous unless a slot was actually inherited"
        assert np.isclose(f, 1.0, atol=1e-9), f"bookkeeping error: fidelity {f}"

    # Prefix that ends exactly at the migration boundary, plus a routing-free tail layer,
    # so wires observed after the boundary cannot have been disturbed by later SABRE swaps.
    _TAIL = [[('1q', 1, H)]]

    def _wires_at_boundary(self):
        _, l2w, _ = lower(self._LAYERS[:2], [self._BASE] * 2, self._MODS)
        return l2w

    def _wires_after_migration(self):
        _, l2w, _ = lower(self._LAYERS[:2] + self._TAIL,
                          [self._BASE, self._BASE, self._SWAPPED], self._MODS)
        return l2w

    def test_slot_inheritance(self):
        """The arriving qubit lands on exactly the wire the departing qubit vacated."""
        before, after = self._wires_at_boundary(), self._wires_after_migration()
        assert after[4] == before[0], "arrival did not inherit the vacated site"

    def test_non_migrants_keep_their_placement(self):
        """q1, q2, q3 never change module, so the boundary must not disturb their wires --
        this is the whole point of slot inheritance over a canonical reset."""
        before, after = self._wires_at_boundary(), self._wires_after_migration()
        for q in (1, 2, 3):
            assert after[q] == before[q], f"q{q} was displaced by someone else's migration"

    def test_one_exchange_costs_one_relabel(self):
        _, _, d = lower(self._LAYERS[:2] + self._TAIL,
                        [self._BASE, self._BASE, self._SWAPPED], self._MODS)
        assert d.migration_relabels == 1, \
            "an exchange should cost one relabelling swap, not a permutation reset"

    def test_migration_adds_no_routing_swaps(self):
        """Slot inheritance means the boundary itself is SWAP-free. A migration with no
        routing pressure must leave swap_count at zero."""
        layers = [[('1q', 0, H)], [('1q', 4, H)]]
        sched = [self._BASE, self._SWAPPED]
        _, _, d = lower(layers, sched, self._MODS)
        assert d.move_count == 2 and d.migration_relabels == 1
        assert d.swap_count == 0, "no intra-module SWAPs may be charged for migration"

    def test_2xSC_migration_routes_both_modules(self):
        """The hardest case: both modules are routed, so slot inheritance happens on both
        sides of the boundary at once."""
        mods = [Module(0, 'sc', (0, 1, 2, 3)), Module(1, 'sc', (4, 5, 6, 7))]
        with noiseless_techs(), noiseless_comm():
            f, d = aer_fidelity(self._LAYERS, self._SCHED, mods)
        assert d.swap_count > 0 and d.migration_relabels > 0
        assert np.isclose(f, 1.0, atol=1e-9)

    def test_multi_qubit_exchange(self):
        """Two out, two in at one boundary. Pairing is deterministic (sorted logical
        order), so the scorer stays a pure function of the schedule."""
        sched_a = {**self._BASE, 0: 1, 2: 1, 4: 0, 5: 0}
        sched = [self._BASE, self._BASE, sched_a, sched_a]
        with noiseless_techs(), noiseless_comm():
            f, d = aer_fidelity(self._LAYERS[:2] + [[('2q', 1, 3, CX)], [('2q', 4, 5, CX)]],
                                sched, self._MODS)
        assert d.move_count == 4
        assert np.isclose(f, 1.0, atol=1e-9)

    def test_scorer_is_deterministic(self):
        a = score(self._LAYERS, self._SCHED, self._MODS)
        b = score(self._LAYERS, self._SCHED, self._MODS)
        assert a == b

    def test_8q_dynamic_sc_na_now_scores(self):
        """The M2/G1 unblock: an 8q SC+NA schedule with a mid-circuit residency change."""
        s = score(self._LAYERS, self._SCHED, self._MODS)
        assert s is not None and 0.0 < s["fidelity"] <= 1.0
        assert s["n_blocks"] == 2


class TestCircuitAdapter:
    """Qiskit / QASM -> layer format. The round-trip tests are the ones that matter:
    if the adapter reorders or drops a gate, every downstream number is wrong and
    nothing else in the harness would notice."""

    @staticmethod
    def _demo():
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.ccx(0, 1, 2)          # 3Q: must be flattened via its definition
        qc.cx(2, 3)
        qc.rzz(0.4, 1, 3)
        return qc

    def test_round_trip_is_exact(self):
        """layers_to_circuit(circuit_to_layers(c)) must be the same unitary as c."""
        qc = self._demo()
        lc = circuit_to_layers(qc)
        assert np.isclose(state_fidelity(Statevector(qc), Statevector(layers_to_circuit(lc))),
                          1.0, atol=1e-9)

    def test_ideal_state_agrees_with_the_source_circuit(self):
        """The scorer's noiseless reference must match the circuit the user handed in."""
        qc = self._demo()
        lc = circuit_to_layers(qc)
        assert np.isclose(state_fidelity(Statevector(qc), ideal_state(lc, lc.active_qubits())),
                          1.0, atol=1e-9)

    def test_terminal_measurements_and_barriers_are_stripped(self):
        qc = self._demo()
        qc.barrier()
        qc.measure_all()
        lc = circuit_to_layers(qc)
        assert lc.dropped.get("measure") == 4 and lc.dropped.get("barrier", 0) >= 1
        assert all(g[-1].name not in ("measure", "barrier") for lay in lc for g in lay)

    def test_mid_circuit_measurement_is_rejected(self):
        """Dropping it would change what the circuit computes, so this must be loud."""
        qc = QuantumCircuit(2, 1)
        qc.h(0)
        qc.measure(0, 0)
        qc.cx(0, 1)
        with pytest.raises(ValueError, match="mid-circuit"):
            circuit_to_layers(qc)

    def test_three_qubit_gates_are_flattened(self):
        lc = circuit_to_layers(self._demo())
        assert all(g[0] in ('1q', '2q') for lay in lc for g in lay)
        assert lc.n_2q > 1, "ccx should have produced several 2Q gates"

    def test_non_cx_two_qubit_gates_survive(self):
        """The routing proxy rewrites 2Q gates as cx for connectivity only; the state must
        still evolve under the real gate."""
        qc = QuantumCircuit(3)
        qc.cz(0, 1)
        qc.rzz(0.3, 1, 2)
        lc = circuit_to_layers(qc)
        assert sorted({g[-1].name for lay in lc for g in lay if g[0] == '2q'}) == ['cz', 'rzz']

    def test_asap_layering_respects_dependencies(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 1)      # layer 0
        qc.cx(1, 2)      # layer 1 (shares q1)
        qc.h(0)          # layer 1 (q0 free)
        lc = circuit_to_layers(qc)
        assert lc.depth == 2
        assert len(lc[0]) == 1 and len(lc[1]) == 2

    def test_layered_circuit_is_usable_as_layers(self):
        """LayeredCircuit implements the list protocol, so it goes straight into score()."""
        qc = QuantumCircuit(8)
        for a, b in [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (4, 6)]:
            qc.cx(a, b)
        lc = circuit_to_layers(qc)
        sched = [{q: (0 if q < 4 else 1) for q in range(8)}] * lc.depth
        s = score(lc, sched, heterogeneous_machine([("sc", 4), ("na", 4)]))
        assert s is not None and 0.0 < s["fidelity"] <= 1.0

    def test_qasm2_and_qasm3(self):
        from qiskit import qasm2, qasm3
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cz(1, 2)
        for dumps in (qasm2.dumps, qasm3.dumps):
            lc = from_qasm(dumps(qc))
            assert lc.n_qubits == 3 and lc.n_2q == 2

    def test_descriptors(self):
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(0, 1)
        qc.cx(2, 3)
        lc = circuit_to_layers(qc)
        assert lc.interaction_pairs() == {(0, 1): 2, (2, 3): 1}
        assert lc.active_qubits() == [0, 1, 2, 3]
        assert 0.0 <= lc.idle_fraction() < 1.0

    def test_barrier_preserves_the_intended_layer_grid(self):
        """THE structural test. Two gates on disjoint qubits are independent, so ASAP
        would pack them into one layer and silently delete the intended idleness. A
        barrier must prevent that -- otherwise an 'idle-heavy' generated family gets
        compressed into a dense one and M1 measures the wrong circuits."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.barrier()
        qc.h(1)
        assert circuit_to_layers(qc).depth == 2

        loose = QuantumCircuit(2)          # same gates, no barrier
        loose.h(0)
        loose.h(1)
        assert circuit_to_layers(loose).depth == 1, "control: ASAP does pack them"

    def test_partial_barrier_aligns_only_its_qubits(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.barrier(0, 1)
        qc.h(1)
        qc.h(2)                            # untouched by the barrier -> layer 0
        lc = circuit_to_layers(qc)
        assert lc.depth == 2
        assert [g[1] for g in lc[0]] == [0, 2]

    def test_delay_consumes_a_layer_slot(self):
        """A delay emits no gate but must not let the following gate slide earlier,
        or an intended idle slot disappears."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(100, 1)
        qc.h(1)
        lc = circuit_to_layers(qc)
        assert lc.depth == 2 and lc.dropped.get("delay") == 1

    def test_logical_swap_is_refused_by_default(self):
        """A source-level SWAP would be charged one f2q while the model prices a SWAP
        at f2q**3 -- a 3x under-price, and two physically identical SWAPs costing
        different amounts depending on where they came from."""
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        with pytest.raises(ValueError, match="SWAP"):
            circuit_to_layers(qc)

    def test_logical_swap_expands_to_three_cx(self):
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        lc = circuit_to_layers(qc, swap_policy="expand")
        names = [g[-1].name for lay in lc for g in lay]
        assert names == ["cx", "cx", "cx"]
        assert np.isclose(state_fidelity(Statevector(qc),
                                         Statevector(layers_to_circuit(lc))), 1.0, atol=1e-9)

    def test_to_cx_basis_removes_swaps_and_preserves_the_unitary(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.swap(0, 1)
        qc.ccx(0, 1, 2)
        lc = circuit_to_layers(to_cx_basis(qc))
        assert all(g[-1].name != "swap" for lay in lc for g in lay)
        assert np.isclose(state_fidelity(Statevector(qc),
                                         Statevector(layers_to_circuit(lc))), 1.0, atol=1e-9)

    def test_make_layers_does_not_relayer(self):
        """The grid you write is the grid that gets scored."""
        lc = make_layers([[(0, 1)], [], [], [(2, 3)]], n_qubits=4)
        assert lc.depth == 4 and lc[1] == [] and lc.idle_fraction() == 0.75

    def test_make_layers_accepts_mixed_entry_forms(self):
        lc = make_layers([[(0, 1), 2], [('2q', 2, 3), ('1q', 0)]], n_qubits=4)
        assert lc.n_2q == 2 and lc.n_1q == 2

    def test_make_layers_rejects_a_double_booked_qubit(self):
        """Two gates on the same qubit in one layer is not a layer -- `lower` would
        double-book its clock."""
        with pytest.raises(ValueError, match="two gates"):
            make_layers([[(0, 1), (1, 2)]], n_qubits=3)

    def test_validate_layers_rejects_a_logical_swap(self):
        from qiskit.circuit.library import SwapGate
        with pytest.raises(ValueError, match="SWAP"):
            validate_layers([[('2q', 0, 1, SwapGate())]], n_qubits=2)

    def test_activity_exposes_asymmetry(self):
        """M1 needs hot/cold asymmetry; this is the check the generator gates on."""
        lc = make_layers([[(0, 1)], [(0, 1)], [(0, 1)]], n_qubits=4)
        act = lc.activity()
        assert act[0] == 3 and act[2] == 0

    def test_strict_mode(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.barrier()
        with pytest.raises(ValueError, match="strict"):
            circuit_to_layers(qc, strict=True)


# =========================================================================
# Scorer interface
# =========================================================================

class TestScorer:

    def test_score_keys_and_machine_acceptance(self):
        """`lower` accepts a Machine as well as a list of Modules."""
        machine = heterogeneous_machine([("sc", 4), ("na", 4)])
        layers = [[('2q', 0, 1, CX)], [('1q', 2, H)]]
        sched = [{q: (0 if q < 4 else 1) for q in range(8)}] * 2
        s = score(layers, sched, machine)
        assert set(s) >= {"fidelity", "makespan", "tts", "comm_count", "comm_time",
                          "move_count", "swap_count", "n_blocks", "block_makespans"}
        assert 0.0 < s["fidelity"] <= 1.0 and s["makespan"] > 0

    def test_pareto_front(self):
        pts = [{"makespan": 10, "fidelity": 0.9},   # 0 non-dominated
               {"makespan": 20, "fidelity": 0.95},  # 1 non-dominated
               {"makespan": 20, "fidelity": 0.8},   # 2 dominated by 1
               {"makespan": 15, "fidelity": 0.85}]  # 3 dominated by 1
        assert pareto_front(pts) == [0, 1]
