# Step 3 — Fixed-N=30 S-vs-R pilot: design and run instructions

**Setting:** SC+NA, caps `[20, 20]`, `C_total = 40`, `N = 30`, so
`rho = 0.75` and every layer carries **10 dummy rows**. Slack is exercised, as
Step 3 requires.

**Arms** (identical in every respect but the capacity mechanism):

| Arm | `capacity_mode` | Loss | Capacity enforced by |
|---|---|---|---|
| **S** | `sinkhorn` | `L = EFCL` | structure (transport polytope) |
| **R** | `softmax` | `L = EFCL + R_cap` | penalty gradient |

---

## What had to change, and why

Four things in the existing script would have invalidated the comparison.

### 1. Nothing was seeded — the largest problem

`torch.manual_seed` was never called, so S and R would have started from
**different random initialisations**, with different data order and different
dropout streams. Any EFCL gap at the gate would then contain an unknown amount
of init noise, and criterion 6 would be measuring the wrong thing.

Added `--seed`, which seeds `random`, `numpy`, `torch`, and CUDA before any
model is constructed, and now also seeds `SortedBatchSampler` (was hardcoded
`seed=0`). Arms sharing a seed share init, data order, and dropout, so the
per-seed S−R difference isolates the mechanism. **This is a paired comparison.**

### 2. One run per arm cannot answer criterion 6

"Not systematically worse beyond normal run-to-run variance" requires knowing
the variance. Three seeds per arm is the minimum that provides it. Six jobs,
all parallel — which is also what you asked for.

The verdict uses the *paired* spread (sd of the per-seed deltas), not the
unpaired spread across runs; the latter confounds mechanism with init.

### 3. The loss is not comparable across arms

R optimises `EFCL + R_cap`; S optimises `EFCL`. The script logged only the
total, and selected `checkpoint_best.pt` on it — so each arm's best checkpoint
would have been chosen under a different objective.

`batch_train_step` now returns **EFCL separately** as a third value. EFCL drives
the console line, `metrics.csv`, the plots, and checkpoint selection. `R_cap` is
still logged for arm R, as a diagnostic only.

### 4. Early stopping would compare the arms at different schedule positions

Patience-3 on a 10-epoch eval cadence can halt an arm ~30 epochs early. Since
`T` anneals to its floor around epoch 115 of 140, two arms stopping at different
epochs are compared at different temperatures, and the gap would partly reflect
schedule position. `--no_early_stop` is passed for both arms in the pilot; both
run the full 140 epochs.

---

## New gate instrumentation

`src/pilot_metrics.py` computes, per evaluation pass:

| Metric | Gate question |
|---|---|
| `sinkhorn_row_res`, `sinkhorn_col_res` | residuals negligible, including late at `T_min` |
| `hardener_burden` | qubits the real hardener moves per layer |
| `argmax_overflow` | capacity excess under plain argmax, before repair |
| `soft_overflow` | should be ~0 for S by construction; sanity check |
| `transition_frac`, `mean_moved` | schedules dynamic over layers |
| `occ_std` | schedules **circuit-dependent** |
| `row_entropy`, `frac_confident` | sharpness, within-arm trend only |

Two notes on reading these.

**Hardener burden is computed for both arms and is the headline ablation
number.** Sinkhorn makes the *soft* `P` feasible; it does not make `argmax(P)`
feasible. R has no structural guarantee at all. The ratio between the two
burdens is the quantitative form of "capacity moved from penalised to
structural."

**`occ_std` catches a collapse the temporal metrics cannot.** A policy that
varies across layers but emits the same partition for every circuit is still
collapsed. Near-zero `occ_std` with a healthy `transition_frac` means the model
learned one circuit-independent rule.

**Entropy is not comparable to pre-Sinkhorn runs.** Where the column constraint
binds, Sinkhorn rows are structurally less sharp than softmax rows at the same
`T`, so Run-6 reference values would read a healthy model as collapsed.

---

## Files

| File | Status |
|---|---|
| `src/pilot_metrics.py` | **new** |
| `utils/train_utils.py` | patched — returns EFCL separately, optional `return_P` |
| `scripts/train_hipergator.py` | patched — 50 hunks, +236/−36 |
| `run_pilot.sbatch` | **new** — 6-task SLURM array |
| `scripts/analyze_pilot.py` | **new** — aggregation + gate verdict |

`batch_train_step` now returns a **3-tuple**. Update any other caller
(`train_test_eval_debug.py`) or it will fail on unpacking.

---

## Pre-launch checklist

1. **Reconcile `cost_config_v3.json`.** It currently reads
   `sc.max_qubits = 15`. With `na = 20` that gives `C_total = 35`, `rho = 0.857`,
   5 dummy rows — a valid but different operating point from the `[20,20]` this
   pilot is specified against. Set both to 20.
   The script now hard-exits if `C_total <= N`.
2. **Add to `CLUSTER_CFG`:** `"capacity_mode": "sinkhorn"`, `"sinkhorn_iters": 30`.
3. **Confirm** `n_samples_train=1200`, `batch_size=32`, `n_epochs=140`.
4. **Dry run both arms first** (~2 min each), which exercises every new code
   path including the CSV writer and diagnostics:
   ```bash
   python scripts/train_hipergator.py --dry_run --capacity_mode sinkhorn --seed 0 --run_tag dry
   python scripts/train_hipergator.py --dry_run --capacity_mode softmax  --seed 0 --run_tag dry
   ```
   Check: `pilot_meta.json` records `rho = 0.75` and `dummy_rows = 10`; arm S
   logs `Cap penalty : DISABLED`; `metrics.csv` has finite `hardener_burden`
   and a `sinkhorn_col_res` around 1e-6.

## Launch

```bash
mkdir -p logs
sbatch run_pilot.sbatch          # array 0-5: S/R x seeds 0,1,2
```

Concurrency-limited allocation: change to `--array=0-5%3`. Tight on GPUs: drop
to seeds 0,1 (`--array=0-3`, edit the arrays). **Do not drop to one seed** —
criterion 6 becomes unevaluable.

## Analyse

```bash
python scripts/analyze_pilot.py results/ --pattern pilot30
```

Prints the per-run table, the paired S−R deltas, and a PASS/FAIL on all six gate
criteria; writes `pilot_summary.csv` and `pilot_comparison.png`. Exit code 0
only if every criterion passes. (The gate logic was smoke-tested on synthetic
runs in both directions — it passes a healthy pilot and fails a degraded one.)

---

## If the gate fails on EFCL

One temperature-schedule tuning attempt is permitted. **Before spending it**,
run the offline diagnostic: load `checkpoint_best.pt`, re-score the *trained*
logits at `T in {0.5, 0.35, 0.25, 0.15, 0.1}`, and recompute hardener burden.
No retraining, no new experiment — it is post-hoc analysis of pilot output.

- Burden drops sharply as `T` falls → temperature problem; lower `T_min` and
  raise `sinkhorn_iters` accordingly (0.25 → 30, 0.15 → 30, 0.1 → 50–75, all
  measured in Step 2).
- Burden stays flat → the learned policy is genuinely homogeneous and lowering
  `T` will not help. That is a hypothesis-space result, not a tuning problem.

If it is still materially worse after the one attempt: stop Sinkhorn
development, ship arm R (already trained, on the matched ASAP cost model), and
invoke the fallback decision.

## Runs in parallel with the pilot

Steps 4A (generator drift at `N ∈ {30,35,39}`) and 5 (sampler waste) need no
Sinkhorn training and no GPU contention with the pilot.
