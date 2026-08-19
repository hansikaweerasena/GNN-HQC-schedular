# Round-2 fixes — Steps 1–2

All four confirmed and fixed. 24/24 unit + 11/11 integration checks pass.

## 1. `_c_total` not restored on load — real bug, fixed

Confirmed and reproduced. Forcing the stale state (`caps = [20,20]`,
`_c_total = 30`) gives a **column residual of 5.00** with no error raised —
exactly the magnitude reported.

Fixed with a `_load_from_state_dict` hook on `CapacitySinkhorn`, mirroring what
was done for `_T`. After the buffers load it recomputes `_c_total`, refreshes
`log_caps`, and **re-validates the loaded capacities** — a checkpoint is an
untrusted source of caps for the same reason a config file is, and without this
a corrupted or hand-edited checkpoint would bypass every guard added last round.

The note in `STEP1_NOTES.md` claiming "caps already travel in the checkpoint"
was wrong as written: the *buffers* travel, the *derived integer* did not.

**One addition beyond what was asked** (easy to veto): `n_iters` is now a
registered buffer too. It was the last remaining forward-path value that lived
only in Python and was absent from `state_dict()` — i.e. the same failure mode,
one variable over. With it buffered, an eval script that reconstructs the module
with the wrong iteration count now gets the checkpoint's value rather than its
own default. Note this makes new checkpoints incompatible with `strict=True`
loading into an older module definition; harmless today since no Sinkhorn
checkpoints exist yet.

Regression test: build with `[15,15]`/`n_iters=7`, load a `[20,20]`/`n_iters=30`
state dict, assert `c_total == 40`, `n_iters == 30`, residuals `< 1e-5`, and
that the loaded caps are the ones enforced. Plus: invalid caps in a state dict
must raise on load.

## 2. Sync test weaker than the notes claimed — correct, fixed

The claim was true of code that was run, but not of the shipped test — the
committed version only exercised `CapacitySinkhorn.forward()` in isolation. That
is the wrong scope: the sync bugs that cost wall-clock lived in the *caller*
(`float(self.temperature)` in the head), which the unit-level test could never
have caught.

New `tests/test_head_sinkhorn_integration.py` covers the full
`EvolvingGNN.batch_forward -> ClusteringHead -> CapacitySinkhorn` path and
intercepts a wider set: `.item()`, `torch.allclose`, `Tensor.__float__`,
`Tensor.__bool__`, `Tensor.tolist`.

**The test was verified to be able to fail.** Reintroducing
`T=float(self.temperature)` in the head makes it report 8 `float` syncs (one per
layer); reverting returns it to zero. A regression test that cannot detect the
bug it guards is worthless, so this was checked rather than assumed.

## 3. `sinkhorn_iters` default — fixed

`ClusteringHead` and `SegmentClustering` both now default to 30, matching
`CapacitySinkhorn` and the frozen Step-2 result. With `n_iters` also buffered,
there are now two independent protections against an eval script silently
running the wrong count.

## 4. Test imports — fixed

`tests/test_sinkhorn.py` now imports `from src.sinkhorn import ...`, matching the
head's import path.

---

## On the conclusion not to freeze

Agreed, and my own data supports the objection rather than my phrasing of it: at
a shared-preference bias of `b = 1.0`, burden is still 2.1 qubits at `T = 0.1`.
So lowering `T` compresses burden but does not eliminate it, and a genuinely
homogeneous learned policy produces the same symptom as an under-annealed one.
The pilot decides which; nothing is pre-diagnosed.

Worth noting only because it costs nothing and is post-hoc analysis of the pilot
output rather than a new experiment: the two cases are separable after the fact
by re-scoring the *trained* logits at several `T` values offline. Sharp drop =>
temperature; flat => policy. No retraining, no extra branch before the pilot.

---

## Remaining before the pilot

Unchanged from `STEP1_NOTES.md`, plus:

- `cluster_module.reset_diagnostics()` at the start of each epoch, or the
  residual maximum accumulates across the whole run.
- Confirm `sc.max_qubits` in the reconciled config (currently 15 in
  `cost_config_v3.json` against `na = 20`), and the strict `C_total > N` assert.
