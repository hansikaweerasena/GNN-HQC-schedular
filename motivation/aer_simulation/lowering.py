"""Lowering: block-segmented ASAP, nonzero t_comm, overlapped movement
(extracted verbatim from NB4, v4).

    lower(layers, schedule, modules, sync_scope="module") -> (circuit, log2wire, diagnostics)

The timing model
----------------
A **block** = a maximal contiguous run of layers over which the complete
qubit->module assignment is unchanged. Pure ASAP was internally inconsistent:
it checked capacity per *layer index* while clocks drifted in *wall clock*, so
a schedule could pass the feasibility gate and still enact a physically
impossible occupancy.

  Inside a block   occupancy is constant, so clocks may drift. Pure ASAP, no
                   inter-layer sync.
  At a boundary    (1) dephase each affected qubit from its t_avail up to the
                   block makespan AT ITS PRE-MOVE T2; (2) movement phase
                   (f_move, tech relabel); (3) the next block begins at t_sync
                   exactly -- t_move_visible = 0.

Both limits are recovered: switching every layer -> layer-synchronous clock;
never switching -> pure ASAP. Capacity is sound, because occupancy is constant
within a block and changes only at a synchronised instant, so per-block capacity
checking IS wall-clock capacity checking.

``sync_scope="module"`` (default, physical): only modules whose occupancy
changes synchronise; an untouched module keeps running. ``"global"`` is kept as
a conservative comparison foil.

Movement is not free even at zero visible latency. Step (1) charges a moving
qubit dephasing at its PRE-MOVE T2 from its own t_avail up to the block
makespan, before it may leave. A mover on the critical path pays nothing extra;
a mover that finished early pays for its own slack at the worse coherence time.
This is what ST7 certifies, and it is the honest answer to "isn't zero-latency
movement a subsidy?"

`pos` is a wire, not a module
-----------------------------
``pos[q]`` is a physical wire index; ``schedule`` holds module membership. They
are different things and must not be merged. On migration BOTH are updated --
``tech[q]`` is relabelled and ``pos[q]`` is moved to the inherited slot, with a
matching SWAP emitted so the quantum state follows the label.

Carried placement across residency windows (v5 -- Gap 1 CLOSED)
--------------------------------------------------------------
Each block is routed from the placement inherited from the previous block, by
emitting the routing sub-circuit in SLOT coordinates (`pos[q] - base`). SABRE's
identity layout is then the carried placement by construction, so no
`initial_layout` plumbing is required.

At a migration boundary the hardware abstraction is SLOT INHERITANCE: an
incoming teleported logical state may be materialised at a physical site
vacated by an outgoing state. Non-migrating residents keep their placement, and
migration therefore induces NO additional intra-module SWAPs beyond the modelled
teleportation cost. Departures are paired to arrivals in sorted logical order --
a fixed rule, deliberately not a search, so the scorer stays a pure function of
the schedule and no part of the greedy-to-optimal gap is produced inside the
judge.

This assumption belongs with "unlimited communication qubits" in the ledger: it
presumes every physical site can serve as a teleportation endpoint. On a real
module with a single transducer port an arrival would materialise at that port
and need SWAPs to reach the vacated site. Bias: favours dynamic/heterogeneous
schedules. Declare it.

The relabelling is enacted with noiseless, zero-duration SWAPs in the simulator
(counted in `migration_relabels`, never in `swap_count`). `pos` is a WIRE and
Aer keeps the state on that wire, so updating `pos` without moving the state
would silently apply every later gate to the wrong qubit.

Gap 2: SABRE gate reordering vs the routed event stream. The event stream
records ('swap', pair) and a generic ('gate',) marker with NO gate ID, while
SABRE does reorder independent gates. The quantum state is never corrupted --
`lower()` applies each logical gate at its tracked pos[qa], pos[qb]. What
corrupts is the physics ACCOUNTING (the SC routing penalty). Measured over
random circuits: cap 4 -> 0/295 violations; cap 5 -> 83/300; cap 6 -> 216/300.
Not luck: SABRE's front layer holds only pairwise-disjoint gates, and on a
4-cycle any two disjoint pairs are either both edges or both diagonals -- if
both adjacent no SWAP intervenes, if both diagonals a single SWAP makes both
adjacent at once. Order-independence is guaranteed at cap 4. Guarded by an
adjacency assertion on every intra-module 2Q gate of a routed module.

`cap <= 4` is therefore not merely tidy, it is load-bearing: a misattributed
swap used to cost 200 ns of accounting error, but adjacent to a remote gate it
can now cost 100 us.
"""

from dataclasses import dataclass, field

from qiskit import QuantumCircuit

from .hardware import TECHS, COMM, HW, Module, SABRE_SEEDS, t_comm, t_move
from .noise import dephasing_channel, gate_infidelity_channel
from .routing import route

__all__ = ["Diagnostics", "segment_blocks", "lower", "Module"]


# ---------------------------------------------------------------------------
# Block segmentation
# ---------------------------------------------------------------------------

def segment_blocks(schedule):
    """Maximal contiguous runs of layers with an identical assignment map.

    Note this splits on changes to the COMPLETE assignment vector, so block
    boundaries are global, not per-qubit. Combined with the boundary
    synchronisation in `lower`, this is what keeps per-block capacity checking
    equivalent to wall-clock capacity checking.
    """
    blocks, start = [], 0
    for i in range(1, len(schedule) + 1):
        if i == len(schedule) or schedule[i] != schedule[start]:
            blocks.append((start, i - 1))
            start = i
    return blocks


# ---------------------------------------------------------------------------
# Containers + channel helpers
# ---------------------------------------------------------------------------

@dataclass
class Diagnostics:
    idle_time: dict = field(default_factory=dict)
    sync_idle: dict = field(default_factory=dict)   # idle charged at block boundaries
    busy_time: dict = field(default_factory=dict)   # gate occupancy per logical qubit
    swap_count: int = 0                             # intra-module SC routing SWAPs
    migration_relabels: int = 0                     # noiseless wire relabels at migrations
    comm_count: int = 0
    move_count: int = 0
    comm_time: float = 0.0                          # total remote-gate duration
    move_time: float = 0.0                          # total exposed state-transfer latency
    makespan: float = 0.0
    feasible: bool = True
    n_blocks: int = 0
    blocks: list = field(default_factory=list)
    block_makespans: list = field(default_factory=list)


def _ge(qc, F, w):
    qc.append(gate_infidelity_channel(F, len(w)).to_instruction(), w)


def _dp(qc, T2, t, w, d, lq, bucket=None):
    ch = dephasing_channel(T2, t)
    if ch is not None:
        qc.append(ch.to_instruction(), [w])
    if t > 0:
        d.idle_time[lq] = d.idle_time.get(lq, 0.0) + t
        if bucket == 'sync':
            d.sync_idle[lq] = d.sync_idle.get(lq, 0.0) + t


def _busy(d, lq, t):
    # Every ns of makespan is either busy or idle for every qubit. ST14 enforces it.
    if t > 0:
        d.busy_time[lq] = d.busy_time.get(lq, 0.0) + t


def _as_modules(modules):
    """Accept either a list of Modules or a Machine."""
    return list(modules.modules) if hasattr(modules, "modules") else list(modules)


# ---------------------------------------------------------------------------
# lower()
# ---------------------------------------------------------------------------

def lower(layers, schedule, modules, seeds=SABRE_SEEDS, sync_scope="module",
          t_comm_fn=None, t_move_fn=None):
    """Block-segmented ASAP lowering.

    Parameters
    ----------
    layers : list of layers; each layer is a list of gate tuples
             ('1q', q, gate) or ('2q', qa, qb, gate)
    schedule : list of {logical_qubit: module_id}, one per layer
    modules : list[Module] or Machine
    sync_scope : 'module' (only affected modules synchronise) | 'global' (conservative foil)
    t_comm_fn : override for the REMOTE-GATE duration. Exists ONLY for the ST4
                invariance regression and for sensitivity sweeps. It CANNOT
                affect movement.
    t_move_fn : override for the STATE-TRANSFER latency. Separate knob, separate
                quantity. Default resolution: t_move(a, b) when
                COMM['t_move_derived'], else the COMM['t_move_visible'] scalar.
                Do not reintroduce a single scalar for both.

    Returns
    -------
    (circuit, {logical_qubit: wire}, Diagnostics), or (None, None, Diagnostics)
    when the schedule violates capacity (diagnostics.feasible is False).
    """
    _tc = t_comm if t_comm_fn is None else t_comm_fn
    if t_move_fn is not None:
        _tm = t_move_fn
    elif COMM["t_move_derived"]:
        _tm = t_move
    else:
        _tm = lambda a, b: COMM["t_move_visible"]
    modules = _as_modules(modules)
    assert len(schedule) == len(layers)
    mbi = {m.mid: m for m in modules}
    N = sum(m.cap for m in modules)
    allq = sorted({q for lay in layers for g in lay for q in (g[1:3] if g[0] == '2q' else [g[1]])}
                  | {q for s in schedule for q in s})
    diag = Diagnostics()

    # ---- capacity feasibility gate ----
    for s in schedule:
        load = {}
        for q, mid in s.items():
            load[mid] = load.get(mid, 0) + 1
        if any(load[mid] > mbi[mid].cap for mid in load):
            diag.feasible = False
            return None, None, diag

    blocks = segment_blocks(schedule)
    diag.n_blocks, diag.blocks = len(blocks), blocks

    pos, nxt = {}, {m.mid: list(m.qubits) for m in modules}
    for q in allq:
        pos[q] = nxt[schedule[0][q]].pop(0)
    tech = {q: mbi[schedule[0][q]].tech for q in allq}

    qc = QuantumCircuit(N)
    tav = {q: 0.0 for q in allq}
    sc_ev, ptrs = {}, {}

    for bi, (l0, l1) in enumerate(blocks):

        # ---- block boundary (before every block except the first) ----
        if bi > 0:
            prev, cur = schedule[blocks[bi - 1][0]], schedule[l0]
            movers = [q for q in allq if prev.get(q) != cur.get(q)]
            if movers:
                affected_mods = set()
                for q in movers:
                    affected_mods.add(prev[q])
                    affected_mods.add(cur[q])
                if sync_scope == "global":
                    sync_q = list(allq)
                else:  # per-module: qubits resident in an affected module, before or after
                    sync_q = [q for q in allq
                              if prev.get(q) in affected_mods or cur.get(q) in affected_mods]
                t_sync = max(tav[q] for q in sync_q)
                # 1. finish previous block: dephase to block makespan at PRE-move tech T2.
                #    A mover pays for its own slack at its WORSE coherence time before it
                #    leaves. This is why t_move_visible = 0 is not the subsidy it appears.
                for q in sync_q:
                    _dp(qc, TECHS[tech[q]].T2, t_sync - tav[q], pos[q], diag, q, bucket='sync')
                    tav[q] = t_sync
                # 2. transfer latency, computed from PRE-move technologies (step 3 relabels).
                #    The boundary clears when the SLOWEST concurrent transfer completes.
                t_mv = max(_tm(tech[q], mbi[cur[q]].tech) for q in movers)
                assert t_mv >= 0, f"negative transfer latency {t_mv}"

                # 3. movement phase: fidelity only. Exactly one f_move channel per mover.
                #    Applied at the SOURCE wire, before the state is relocated in step 4.
                for q in movers:
                    _ge(qc, COMM["f_move"], [pos[q]])
                    tech[q] = mbi[cur[q]].tech
                    diag.move_count += 1
                diag.move_time += t_mv

                # 4. SLOT INHERITANCE. An incoming teleported state is materialised at a
                #    physical site vacated by an outgoing state, so non-migrating residents
                #    keep their placement and migration induces no intra-module SWAPs.
                #    Departures are paired to arrivals in sorted logical order -- a fixed
                #    rule, not a search: letting the scorer optimise the pairing would move
                #    part of the greedy-to-optimal gap inside the judge.
                target = {}
                for m in modules:
                    D = [q for q in movers if prev.get(q) == m.mid]
                    A = [q for q in movers if cur.get(q) == m.mid]
                    if not A:
                        continue
                    occupied = {pos[q] for q in allq if prev.get(q) == m.mid}
                    pool = [pos[q] for q in D] + sorted(w for w in m.qubits
                                                        if w not in occupied)
                    assert len(A) <= len(pool), (
                        f"module {m.mid}: {len(A)} arrivals but only {len(pool)} free sites "
                        "-- the capacity gate should have caught this")
                    for q, w in zip(A, pool):
                        target[q] = w

                # Enact the relabelling in the simulator. `pos` is a WIRE, and Aer keeps the
                # state on that wire -- updating pos without moving the state would silently
                # apply every later gate to the wrong qubit. These SWAPs are noiseless and
                # zero-duration: the physical cost of migration is already charged through
                # f_move and t_move. They are NOT SC routing SWAPs and are counted separately.
                owner = {pos[q]: q for q in allq}
                for q in sorted(target, key=lambda x: (pos[x], x)):
                    src, dst = pos[q], target[q]
                    if src == dst:
                        continue
                    r = owner.get(dst)          # None => the destination site is empty
                    qc.swap(src, dst)
                    diag.migration_relabels += 1
                    pos[q], owner[dst] = dst, q
                    if r is not None:
                        pos[r], owner[src] = src, r
                    else:
                        owner.pop(src, None)

                # 5. advance the boundary instant by the transfer latency, charging every
                #    synchronised qubit for the wait. At t_mv = 0 this is a no-op and the
                #    v4 overlapped model is recovered exactly.
                #
                #    Pre-v4 code advanced the clock here WITHOUT dephasing -- a silent,
                #    decoherence-free time advance. ST14 (busy + idle == makespan, per
                #    qubit) is what makes that class of bug impossible to reintroduce, so
                #    every qubit must be booked into exactly one bucket below.
                if t_mv > 0:
                    _movers = set(movers)
                    for q in sync_q:
                        if q in _movers:
                            # In flight. Its transfer infidelity is already aggregated into
                            # f_move; charging T2 dephasing on top would double-count. The
                            # interval is occupancy, not idleness -- book it as busy.
                            _busy(diag, q, t_mv)
                        else:
                            # Sitting in its module waiting for the boundary to clear: real
                            # idle at its own (unchanged) T2.
                            _dp(qc, TECHS[tech[q]].T2, t_mv, pos[q], diag, q, bucket='sync')
                        tav[q] = t_sync + t_mv

        # ---- route this block from the CARRIED placement ----
        # The sub-circuit is emitted in SLOT coordinates (pos[q] - base), so the placement
        # inherited from the previous block IS SABRE's identity layout and no initial_layout
        # plumbing is needed. Re-routing per block also shrinks Gap 2: SABRE cannot reorder
        # gates across a block boundary.
        cur_s = schedule[l0]
        sc_ev = {}
        for m in modules:
            if TECHS[m.tech].all_to_all or m.cap < 3:
                continue
            base = m.qubits[0]
            sub = QuantumCircuit(m.cap)
            n2q = 0
            for li in range(l0, l1 + 1):
                for g in layers[li]:
                    if g[0] == '2q' and cur_s.get(g[1]) == m.mid and cur_s.get(g[2]) == m.mid:
                        sub.cx(pos[g[1]] - base, pos[g[2]] - base)
                        n2q += 1
            if n2q == 0:
                sc_ev[m.mid] = []       # still a routed module: keeps the adjacency guard live
                continue
            rr = route(sub, HW.coupling_map(m.tech, m.cap), seeds)
            diag.swap_count += rr.swap_count
            ev = []
            for inst in rr.routed_circuit.data:
                qs = [rr.routed_circuit.find_bit(b).index for b in inst.qubits]
                if inst.operation.name == "swap":
                    ev.append(('swap', (base + qs[0], base + qs[1])))
                elif inst.operation.name == "cx":
                    ev.append(('gate',))
            sc_ev[m.mid] = ev
        ptrs = {mid: 0 for mid in sc_ev}

        # ---- inside the block: pure ASAP, no global sync between layers ----
        for li in range(l0, l1 + 1):
            for g in layers[li]:
                if g[0] == '1q':
                    q = g[1]
                    sp = TECHS[tech[q]]
                    qc.append(g[2], [pos[q]])
                    _ge(qc, sp.f1q, [pos[q]])
                    tav[q] += sp.t1q
                    _busy(diag, q, sp.t1q)
                else:
                    qa, qb = g[1], g[2]
                    ma, mb = schedule[li][qa], schedule[li][qb]

                    # drain any SWAPs SABRE scheduled before this gate
                    if ma == mb and ma in sc_ev:
                        ev, ptr = sc_ev[ma], ptrs[ma]
                        while ptr < len(ev) and ev[ptr][0] == 'swap':
                            w1, w2 = ev[ptr][1]
                            L1 = next(p for p in allq if pos[p] == w1)
                            L2 = next(p for p in allq if pos[p] == w2)
                            st = max(tav[L1], tav[L2])
                            for lq in (L1, L2):
                                _dp(qc, TECHS[tech[lq]].T2, st - tav[lq], pos[lq], diag, lq)
                            _ge(qc, TECHS[tech[L1]].f2q ** 3, [w1, w2])
                            qc.swap(w1, w2)
                            pos[L1], pos[L2] = w2, w1
                            _sd = 3 * TECHS[tech[L1]].t2q
                            tav[L1] = tav[L2] = st + _sd
                            _busy(diag, L1, _sd)
                            _busy(diag, L2, _sd)
                            ptr += 1
                        if ptr < len(ev):
                            ptr += 1
                        ptrs[ma] = ptr

                    st = max(tav[qa], tav[qb])
                    for q in (qa, qb):
                        _dp(qc, TECHS[tech[q]].T2, st - tav[q], pos[q], diag, q)

                    # GUARD (Gap 2): an intra-module 2Q gate on a routed module must execute
                    # on physically ADJACENT wires. Safe at cap 4; can fire at cap>=5.
                    if ma == mb and ma in sc_ev:
                        _m = mbi[ma]
                        _b = _m.qubits[0]
                        _c = _m.cap
                        _la, _lb = pos[qa] - _b, pos[qb] - _b
                        _adj = (abs(_la - _lb) == 1) or (abs(_la - _lb) == _c - 1)
                        assert _adj, (
                            f"non-adjacent intra-SC 2Q gate: logical ({qa},{qb}) on wires "
                            f"({pos[qa]},{pos[qb]}) of module {ma} (cap {_c}). SABRE gate-order "
                            f"mismatch -- routed event stream lacks gate IDs. Phase-2 fix required.")

                    # REMOTE branch keys on MODULE (ma != mb), never on technology. A
                    # cross-module gate inside a homogeneous 2xSC machine is remote and must
                    # pay f_comm/t_comm, otherwise the homogeneous baseline is silently
                    # monolithic and wins trivially.
                    if ma != mb:
                        _ge(qc, COMM["f_comm"], [pos[qa], pos[qb]])
                        qc.append(g[3], [pos[qa], pos[qb]])
                        dur = _tc(tech[qa], tech[qb])   # residency tech, post-move correct
                        diag.comm_count += 1
                        diag.comm_time += dur
                    else:
                        sp = TECHS[tech[qa]]
                        qc.append(g[3], [pos[qa], pos[qb]])
                        _ge(qc, sp.f2q, [pos[qa], pos[qb]])
                        dur = sp.t2q

                    tav[qa] = tav[qb] = st + dur
                    _busy(diag, qa, dur)
                    _busy(diag, qb, dur)

        diag.block_makespans.append(max(tav.values()))

    # ---- tail: dephase every qubit up to the global makespan ----
    tmax = max(tav.values())
    for q in allq:
        _dp(qc, TECHS[tech[q]].T2, tmax - tav[q], pos[q], diag, q)
    diag.makespan = tmax
    qc.save_density_matrix()
    return qc, {q: pos[q] for q in allq}, diag
