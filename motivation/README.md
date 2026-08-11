# mosaic_aer — Phase-0 Aer harness

One trusted function `assignment -> Aer fidelity`, physics-grounded and independent of EFCL.
Extracted from notebooks NB1–NB6 (v4) so that M1, M2/G1, and later E0/E1 all import the
*same* physics instead of each carrying a copy-pasted variant.

**This package is the judge. EFCL is the training surrogate.** Nothing here should ever be
simplified to make an optimizer's life easier — simplify the surrogate instead.

## Install

```bash
cd mosaic
pip install -e ".[test]"
```

Verified against `qiskit 2.5.1` / `qiskit-aer 0.17.2`.

## Run the certificate

```bash
MOSAIC_CONFIG_DIR=/path/to/cost_configs pytest -q
```

91 tests: the NB1/NB2/NB3 checkpoints, ST1–ST19, carried placement, the circuit adapter
(barrier grid preservation, SWAP pricing, layer validation), the routing guards, and the
config drift check. **Run this before every experiment batch.** If anything here fails, every
number the harness produces is suspect.

## Use

```python
from mosaic_aer import score, heterogeneous_machine, homogeneous_machine, H, CX

machine = heterogeneous_machine([("sc", 4), ("na", 4)])   # 8q, module 0 = SC, module 1 = NA
layers  = [[('2q', 0, 1, CX)], [('1q', 2, H)]]            # one list of gates per layer
sched   = [{q: (0 if q < 4 else 1) for q in range(8)}] * 2 # {logical qubit: module id} per layer

s = score(layers, sched, machine)
# {'fidelity': ..., 'makespan': ..., 'tts': ..., 'comm_count': ..., 'comm_time': ...,
#  'move_count': ..., 'swap_count': ..., 'n_blocks': ..., 'block_makespans': [...]}
```

Gate tuples are `('1q', q, gate)` or `('2q', qa, qb, gate)` with a Qiskit `Gate` instance.
Strip terminal measurements — the scorer compares pre-measurement states, so readout never
enters. `score` returns `None` for a capacity-infeasible schedule.

The homogeneous references are built the same way and are **distributed, not monolithic**:

```python
homogeneous_machine("sc", 2, 4)   # 2xSC — cross-module gates pay f_comm AND t_comm
```

## Layout

| module | from | contents |
|---|---|---|
| `noise` | NB1 | `dephasing_channel`, `gate_infidelity_channel` |
| `hardware` | NB2 | `TECHS`, `COMM`, `t_comm`, `HW`, `Module`, `Machine`, builders |
| `routing` | NB3 | `route`, `un_permute` |
| `lowering` | NB4 | `segment_blocks`, `lower` — block-segmented ASAP |
| `scoring` | NB5 | `aer_fidelity`, `score`, `pareto_front` |
| `circuits` | new | `make_layers`, `circuit_to_layers`, `from_qasm`, `to_cx_basis` |
| `configs` | NB2 §5 | `drift_check` against the EFCL cost configs |

## What changed in extraction

The physics is byte-identical — every ST test reproduces its hand-computed constant. Five
mechanical changes:

1. **`Module` unified.** NB2 and NB4 defined two different `Module` dataclasses
   (`module_id`+stored `coupling_map` vs `mid`+`cap`). Now one frozen dataclass with `mid`
   canonical, `module_id` as an alias, and `coupling_map` **derived** from `tech`/`cap` so it
   cannot desync.
2. **One router.** NB4 carried a stripped-down copy of NB3's `route`. The NB3 version (with
   the measurement guard and the `initial_layout` Phase-2 stub) is now the only one, and
   `ring_cmap` is replaced by `HW.coupling_map(tech, cap)` — identical output, one source.
3. **`noiseless_techs()` context manager** replaces the `TECHS.clear()` / positional-
   `TechSpec` rebuild that ST1 and ST12a used. That pattern broke the moment `TechSpec`
   gained a field; `dataclasses.replace` does not.
4. **`dephasing_channel` returns `None` at `t == 0`** (NB4's behaviour, which `lower`
   depends on) and raises only on genuinely negative `t`, which always means a clock bug.
5. **`lower` accepts a `Machine` or a `list[Module]`.**

Nothing else. `lower()` is verbatim.

## Load-bearing facts you should not "clean up"

- **Slot inheritance at migration boundaries (v5).** An incoming teleported state is
  materialised at a physical site vacated by an outgoing state, so non-migrating residents
  keep their placement and migration induces no intra-module SWAPs. Departure→arrival
  pairing is deterministic (sorted logical order) — deliberately *not* a search, so the
  scorer stays a pure function of the schedule and none of the greedy-to-optimal gap is
  produced inside the judge. Each block is then routed from the carried placement by
  emitting the routing sub-circuit in **slot coordinates**, which makes SABRE's identity
  layout the carried placement by construction (no `initial_layout` plumbing needed).
  The relabelling is enacted with noiseless zero-duration SWAPs counted in
  `migration_relabels`, never in `swap_count` — `pos` is a wire and Aer keeps the state
  on that wire, so relabelling without moving the state would silently misapply every
  later gate.
- **`cap <= 4` on SC modules.** Gap 2 (SABRE reorders gates; the event stream has no gate
  IDs) is order-independent on a 4-cycle and therefore harmless — measured 0/295 violations
  at cap 4, 83/300 at cap 5, 216/300 at cap 6. Going past 8q as 2×4 requires fixing that
  bug first. The adjacency assertion in `lower` is what keeps it loud.
- **The remote branch keys on MODULE, never technology.** A cross-module SC–SC gate is
  remote and pays `f_comm` + `t_comm("sc","sc")`. Remove this and the homogeneous baseline
  becomes a free monolith and wins trivially.
- **Movement latency is exposed by default (v5).** `COMM["t_move_derived"] = True`: the
  boundary advances by `max` over movers of `t_move(from, to)`, movers are booked as *busy*
  (their transfer infidelity is already aggregated into `f_move` — charging T2 on top would
  double-count), and non-movers in the synchronised set pay real idle at their own T2.
  `movement_mode(derived=False, visible=0.0)` recovers the v4 overlapped numbers exactly and
  is the sensitivity foil — keep it, don't delete it.
- **`t_comm` and `t_move_visible` are separate quantities.** They were one scalar
  (`t_remote`) until v4, harmless only because it was 0. Never merge them again.
- **`t_comm = max(t2q_a, t2q_b)` is an optimistic lower bound** — it drops the endpoint
  measurements, the classical round trip, and the Pauli corrections. Say so in the paper.

## Block boundaries and DP state sufficiency

Relevant to the exact-optimizer question, and verified against the code rather than assumed:

- `segment_blocks` splits on changes to the **complete assignment vector**, so block
  boundaries are **global, not per-qubit**.
- At a boundary, every qubit in `sync_q` is dephased up to `t_sync` and has `tav[q] = t_sync`.
  Under the default `sync_scope="module"`, `sync_q` is every qubit resident in an affected
  module before or after the move.

**For a two-module machine any move affects both modules, so `sync_q` is all qubits and the
ready-time vector is uniform at every block boundary.** Assignment state is therefore a
sufficient DP state at block boundaries for the 8q / K=2 configuration. This does *not*
generalise to K≥3 under module scope, where an untouched module keeps running by design
(ST8) and ready times stay heterogeneous across the boundary.

## Assumption ledger — put this table in the evaluation section

| assumption | direction of bias |
|---|---|
| uniform `f_comm`, `f_move` across module pairs | favours heterogeneous |
| pre-shared Bell pairs, unlimited supply | favours heterogeneous |
| unlimited communication qubits (concurrent remote gates never serialise) | favours heterogeneous |
| `t_move = max(t2q_from, t2q_to)`, transfer exposed on the critical path | favours homogeneous / static |
| classical round trip and Pauli correction dropped from `t_move` and `t_comm` | favours heterogeneous |
| `t_comm = max(t2q_a, t2q_b)`, derived per pair | favours homogeneous SC |
| per-gate telegate, no cat-entanglement fan-out amortisation | favours homogeneous |


## Circuits: two ways in

**Synthetic circuits — build the grid directly.**

```python
from mosaic_aer import make_layers
lc = make_layers([[(0, 1)], [], [], [(2, 3)]], n_qubits=8)   # [] = everyone idles
lc.depth, lc.n_2q, lc.idle_fraction(), lc.activity(), lc.interaction_pairs()
```

Nothing is re-layered, so the active/idle pattern you write is the one that gets scored.
Validates as it builds: a qubit appearing twice in one layer, or a logical SWAP, raises.

**Existing circuits (MQT Bench, QASM) — convert.**

```python
from mosaic_aer import circuit_to_layers, from_qasm, to_cx_basis
lc = circuit_to_layers(to_cx_basis(qc))      # normalise the basis first
lc = from_qasm(path_or_string)               # QASM 2 or 3
score(lc, schedule, machine)                 # LayeredCircuit works directly as `layers`
```

QASM 3 needs `pip install qiskit-qasm3-import`; QASM 2 works out of the box.

### Why `make_layers` for M1/M2

`circuit_to_layers` re-derives layers from the dependency graph, and two gates on disjoint
qubits are independent — so ASAP packs them together regardless of what the source circuit
looked like:

```
intended                     after ASAP re-layering
L0:  q0 gate,  q1 idle       L0:  q0 gate, q1 gate
L1:  q0 idle,  q1 gate       (intended idleness gone)
```

Measured on a staggered 12-layer circuit: depth 12 → 6, `idle_fraction` 0.75 → 0.50.
The static-schedule fidelity is unchanged (timing comes from the per-qubit block clock, not
the layer index), but two things break: the descriptors a generator filters on are wrong,
and **`schedule` has one entry per layer, so halving the depth halves the number of points
at which M2 can place a block boundary.** Barriers now prevent this; building directly
avoids the round trip.

### Conversion rules

| operation | treatment |
|---|---|
| `barrier` | alignment point — every spanned qubit advances to the latest frontier among them, so a full-width barrier preserves an intended grid exactly |
| `delay`, `id` | emit no gate but consume a layer slot, preserving an intended idle |
| terminal `measure`/`reset` | stripped (the scorer compares pre-measurement states) |
| mid-circuit `measure`/`reset` | **raises** — dropping one changes what the circuit computes |
| classically-conditioned ops | **raises** — no classical control flow in the model |
| >2-qubit gates | flattened via `.definition` |
| logical `swap` | **raises** by default — see below |
| other 2Q gates (`cz`, `rzz`, ...) | carried through unchanged; the routing proxy rewrites them as `cx` for connectivity only |

### Logical SWAPs are refused by default

The hardware model prices a SWAP as three native 2Q gates (`f2q**3`, `3*t2q`) and charges
routing-inserted SWAPs that way. A SWAP already in the *logical source circuit* would arrive
as an ordinary 2Q gate and be charged a single `f2q`/`t2q` — a 3× under-price, and two
physically identical SWAPs costing different amounts depending on provenance.

So `circuit_to_layers` raises on a source-level `swap`. Fix it with `to_cx_basis(circuit)`
(recommended — it normalises every composite gate at once) or `swap_policy="expand"`.
`swap_policy="as_gate"` keeps the old under-priced behaviour for sensitivity analysis only.

Note this means the existing `TWO_QUBIT_GATES = ("cx", "cz", "swap")` generator cannot be
used for M1/M2 as-is: drop `swap` from the logical alphabet, or normalise the basis.
