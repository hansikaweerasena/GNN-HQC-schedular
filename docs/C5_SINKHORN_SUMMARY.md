# C5 — Sinkhorn Assignment Head: Work Summary

Replacing the capacity **regularizer** with a capacity **structure**, from design
through the fixed-N=30 pilot and into standalone evaluation.

---

## 1. Why the change

The capacity regularizer added `R_cap` to the loss, so the physics gradient
(EFCL) and the feasibility gradient competed during training. A reviewer
objected that this *hinders* learning rather than helping it.

The replacement embeds capacity as a structural property of the assignment head,
so it never enters the loss at all:

```
L = EFCL          (arm S)      vs      L = EFCL + R_cap      (arm R)
```

---

## 2. Formulation (frozen)

**Balanced entropic optimal transport with dummy rows.**

For a circuit with `N` logical qubits, `K` technologies of capacity `C_1..C_K`,
and `C_total = sum_k C_k`, the head's logits `Z ∈ R^{N×K}` are augmented with
`C_total − N` **zero-logit dummy rows** and solved as a balanced transport
problem:

- row marginals = 1 for every row (real and dummy)
- column marginals = `C_k`

Discarding the dummy rows gives `P` with `sum_k P[u,k] = 1` for every real qubit
and `sum_u P[u,k] ≤ C_k` for every technology. The unused capacity is absorbed by
dummy mass.

**Key design points:**

- **Dummy rows, not renormalised column marginals.** Scaling the columns down to
  total mass `N` would force a fixed proportional split (15/15 at caps [20,20],
  N=30) and destroy the scheduling decision. The dummy-row form does not:
  measured occupancy under strong preference reaches [19.66, 10.34].
- **A slack ROW, not a slack column.** The surplus is on the capacity side, so
  the dummy source belongs on the row side. A slack column would model
  unplaceable qubits — the infeasible `C_total < N` case, which raises instead.
- **`T` is the entropic regularisation parameter.** The log-kernel is `Z/T`, so
  the head's existing temperature schedule plays the role of `ε`. Only one
  schedule is ever active; running both would apply it twice.
- **Shape convention `[..., N, K]`.** The operator works on the last two
  dimensions, so a single circuit is the no-leading-dims case and a fixed-N
  mini-batch is `[B, N, K]`. The operator carries no batching assumption of its
  own and would not change if ragged batching is adopted later.
- **Rejected:** Gumbel-Sinkhorn, unbalanced/inequality-constrained OT,
  per-capacity-slot expansion, convergence-dependent Python stopping.

**Iteration count: 30**, chosen from a measured sweep rather than guessed.
Residuals reach the float32 floor (~4e-6) by 20 iterations at `T_min = 0.5`;
30 supplies headroom. The float64 control confirms the plateau is precision,
not stalled convergence.

> The budget rests on the cosine head's `Z ∈ [-1,1]` bound. Adding a learnable
> similarity scale would invalidate it and require re-running the sweep.

---

## 3. Scope decision on capacity generalisation

The paper claims **one model per hardware setting**, not one model generalising
across arbitrary capacity vectors. Capacities stay fixed within a training run;
the reviewer-requested imbalanced-capacity result is produced as a *separate
hardware setting*, not as a zero-shot transfer test.

The asymmetry (zero-shot across `N`, retrained across capacity) is principled
and belongs in §8: `N` enters through the graph and a GNN encoder is
size-agnostic by construction, whereas capacity enters only through the
projection, which cannot re-shape the representation it projects.

---

## 4. Files changed

| File | Status | What it does |
|---|---|---|
| `src/sinkhorn.py` | **new** | `capacity_sinkhorn()` functional core + `CapacitySinkhorn` module. Log-domain balanced Sinkhorn with dummy rows; capacity validation and derived quantities hoisted to construction. |
| `src/clustering_head.py` | modified | Split into `compute_logits()` (stages 1–3, unchanged) and `normalize()` (stage 4, `"sinkhorn"` \| `"softmax"`). Head gains `capacity_mode`, `caps`, `sinkhorn_iters`. |
| `src/evolving_gnn.py` | modified | One argument (`n_qubits`) forwarded to the head so per-circuit capacity is not pooled across a batch. Batching itself untouched. |
| `utils/train_utils.py` | modified | `batch_train_step` returns EFCL separately from the loss; optional `return_P` for evaluation diagnostics. |
| `src/pilot_metrics.py` | **new** | Gate diagnostics: hardener burden, argmax overflow, transition statistics, cross-circuit occupancy spread. |
| `scripts/train_hipergator.py` | modified | `--capacity_mode`, `--seed`, `--no_early_stop`; global seeding; `C_total > N` assert; EFCL-based logging and checkpoint selection; per-epoch `metrics.csv`; `pilot_meta.json`; capacity metadata written into `model_arch_params.json`. |
| `run_pilot.sbatch` | **new** | 6-task SLURM array (2 arms × 3 seeds), all parallel. |
| `scripts/analyze_pilot.py` | **new** | Aggregates runs, reports paired S-vs-R deltas, evaluates the gate. |
| `tests/test_sinkhorn.py` | **new** | Marginals, gradients, infeasible capacity, dummy/slack behaviour, iteration sweep. |
| `tests/test_head_sinkhorn_integration.py` | **new** | Full GNN → head → Sinkhorn path, checkpoint round-trip. |
| `eval_scheduler_v1.py` | modified | Reconstructs the Sinkhorn head from saved artifacts (see §7). |
| `fetch_pilot.sh`, `run_eval_sinkhorn.sh` | **new** | Artifact download and evaluation launch. |

**Configuration:** `CLUSTER_CFG` gains `capacity_mode: "sinkhorn"` and
`sinkhorn_iters: 30`. `cost_config_v3.json` capacities set to `[20, 20]`, giving
`C_total = 40`, `N = 30`, `rho = 0.75`, and 10 dummy rows per layer.

`sf`, `beta`, and `lambda_cap` are dead hyperparameters on the Sinkhorn path.

---

## 5. Pilot design (fixed N=30, SC+NA, caps [20,20])

Two arms trained under identical conditions except the capacity mechanism:

| Arm | mode | Loss | Capacity enforced by |
|---|---|---|---|
| **S** | `sinkhorn` | `EFCL` | structure |
| **R** | `softmax` | `EFCL + R_cap` | penalty gradient |

Design requirements, each addressing a specific way the comparison could have
been invalidated:

- **Shared seed per arm pair.** Arms sharing a seed share initialisation, data
  order, and dropout stream, so the per-seed difference isolates the mechanism.
  This is a *paired* comparison.
- **Three seeds per arm.** The gate asks whether S is worse "beyond normal
  run-to-run variance"; one run per arm leaves that variance unknown.
- **EFCL logged separately from the loss.** The arms optimise different
  objectives, so comparing losses would compare different quantities. Checkpoint
  selection also switched to test EFCL.
- **Early stopping disabled.** Arms halting at different epochs would be
  compared at different points on the temperature schedule.
- **`C_total > N` strictly.** At equality there are no dummy rows and the slack
  behaviour the pilot exists to validate is never exercised.

---

## 6. Pilot results and what they show

Arm S, seed 0, 140 epochs (~8.3 h on a B200):

| Quantity | Start | End |
|---|---|---|
| Train EFCL | 10.28 | 6.26 |
| Test EFCL | 10.38 | **6.44** |
| Hardener burden | 1.10 | **0.51** qubits/layer (1.7% of 30) |
| Sinkhorn column residual | 7.6e-06 | 1.3e-04 |
| Gradient norm | 0.72 | 0.87 |
| Transition fraction | 0.50 | 0.52 |
| Cross-circuit occupancy spread | 0.061 | 0.058 |

**Learning is clean.** EFCL falls 39%, monotonically, with no instability. The
train/test gap is 2.9% — no overfitting. Schedules stay both dynamic
(`transition_frac ≈ 0.5`) and circuit-dependent (`occ_std ≈ 0.058`), so the model
did not collapse to a static or circuit-independent partition.

**The block-based ASAP timing model supports learning.** `tau` stays constant
because `set_epoch` correctly early-returns on the ASAP path — confirmation the
C1 timing model is live rather than a frozen buffer. Both arms descend smoothly
with an informative gradient. The earlier C1 smoke-run concern that "EFCL was
essentially flat" is resolved: that was a schedule artifact of the 30-epoch run,
not a defect in the timing model.

**The competing-gradient claim is now quantified.** Arm S's gradient norm stays
flat at 0.72 → 0.87 across the whole run. Arm R's grows roughly fivefold,
1.68 → 8.76, with spikes (7.83 at epoch 53, 8.76 at epoch 99) — the instability
that previously forced gradient-clipping tuning. Meanwhile R's `cap_tr` climbs
monotonically 0.005 → 0.30, confirming the regularizer is genuinely active and
genuinely fighting the physics gradient. This is the clearest evidence yet for
the paper's central claim about the mechanism.

**Two observations that shape what comes next.** The Sinkhorn residual drifts
upward from epoch ~105, tracking the point where the model sharpens into a
bimodal structure sitting exactly at the capacity boundary — the very structure
successful training produces. And S's final 25 epochs bought a 0.3% improvement:
training was ended by the temperature floor and LR decay, not by convergence.
S's gradient norm (~0.9) is an order of magnitude below R's, and the learning
rate was tuned against a gradient scale that no longer exists.

**Together these mean the pilot's 6.44 is a floor on what this architecture can
do, not its ceiling.**

---

## 7. Evaluation path

`eval_scheduler_v1.py` reconstructs both models from saved artifacts. The
capacity mechanism must be supplied at **construction** time — the Sinkhorn head
validates capacities in `__init__`, before `load_state_dict` can supply them from
the checkpoint buffers — so `capacity_mode`, `sinkhorn_iters`, and `caps` are now
written into `model_arch_params.json` by the trainer and read back by the eval.
Runs predating this default to `softmax`.

Three consistency checks were added, each guarding a failure that would produce
plausible but wrong numbers rather than an error:

- **Capacity agreement** between `model_arch_params.json` and
  `cost_config_snapshot.json` — a mismatch means the checkpoint and config
  describe different hardware.
- **Annealed temperature restored** — evaluating at `T_init` instead of `T_min`
  would give much softer assignments and an artificially poor scheduler.
- **Feasibility (`C_total ≥ N`) checked up front**, since with a Sinkhorn head
  this fires inside the forward pass rather than at the hardener.

Reconstruction was verified to reproduce the trained module bit-identically.

**Artifacts required for evaluation** (a few MB total; the model is ~71k
parameters): `model_arch_params.json`, `cost_config_snapshot.json`,
`scheduler_config_snapshot.py`, `checkpoint_best.pt`.

**Evaluation scope:** fixed N=30 only, matching the training setting. No range
mode — the pilot never covered other sizes, and beyond N=40 no capacity-feasible
assignment exists at all.

**Note on baseline naming:** the code exposes `MOSAIC, B1, B3, B4, B5` — there is
no `B2`. Confirm the code-to-paper baseline mapping before writing up the
comparison table.

---

## 8. Direction from here: Sinkhorn only

**The regularizer arm is closed.** Arm R has served its purpose:

1. It supplied the matched reference on the current ASAP cost model, since every
   earlier regularizer result predates that change and was not comparable.
2. It quantified the competing-gradient pathology — a fivefold growth in
   gradient norm, with spikes, against Sinkhorn's flat profile.
3. It confirmed the regularizer is active rather than inert at `rho = 0.75`,
   resolving an open question about whether the ablation was meaningful.

Those are the insights the ablation was designed to produce, and they are
sufficient for the paper's §4 and E4. **No further regularizer experiments will
be run.**

All remaining effort goes to the Sinkhorn path:

- **Evaluate the trained Sinkhorn scheduler standalone** against the synthetic
  baselines at N=30 (`eval_scheduler_v1.py`), then MQT Bench zero-shot
  (`eval_scheduler_v2.py`).
- **Then revisit training improvements** — the learning rate is the leading
  candidate given the gradient-scale change, followed by the temperature
  schedule, which currently wastes the final 25 epochs.
- **Then variable-N training** (stratified 30–39 dataset, generator drift check,
  batching decision) per the frozen plan.

The regularizer code stays in the repository, reachable by
`capacity_mode="softmax"`, purely so the E4 ablation remains reproducible.
