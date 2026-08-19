# C5 — Sinkhorn Assignment Head: Frozen Execution Plan

**Status:** frozen after two review rounds. Anchor deadline ASPLOS Fall, 9 Sept.
Day numbers are relative to kickoff (D0), not calendar dates.

**Frozen decisions (do not relitigate without new evidence):**

| Decision | Choice | Rejected alternative |
|---|---|---|
| OT formulation | Balanced log-domain Sinkhorn, $C-N$ neutral unit dummy rows | Single fat slack row (equivalent, no benefit at $C{=}40$); unbalanced/inequality OT; per-slot expansion |
| Sharpening | Reuse existing $T$ schedule as $\varepsilon$; $Z/T$ is the log-kernel | Separate $\varepsilon$ schedule; Gumbel-Sinkhorn |
| Capacity claim | **A**: one model per hardware setting | **B**: one model generalising over capacity vectors |
| Capacity during training | Fixed $C_k$ per run | Per-batch capacity-split sampling |
| Iteration count | Set empirically from marginal residuals at $T_{\min}$ | Guessed 20–30 |
| Batching | **Undecided** — resolved by measurement in Phase 4B | — |
| Loss | $\mathcal{L} = C_{\text{total}}(P_{1:L})$, no $R_{\text{cap}}$ | Any capacity term in the loss |

---

## Phase 0 — Preconditions (D0, half day)

Nothing below starts until these are true.

1. **Config reconciliation.** `techs.json` canonical → `cost_config_v3.json`
   reconciled. Confirm and freeze:
   - `sc.max_qubits`, `na.max_qubits` (plan assumes 20/20 → $C_{\text{total}}=40$)
   - `n_samples_train=1200`, `batch_size=32`, `n_epochs=140`
   - `num_qubits=30` for the pilot
   Record $\rho = N/C_{\text{total}}$ explicitly in the run log. At 20/20 this is
   0.75 (10 dummy rows). **If $C_{\text{total}} = N$ there are zero dummy rows and the
   constraint becomes a forced equality split — the pilot would be meaningless.**
   Assert `C_total > N` at startup, not just `>=`.
2. **Assert** $N \le \sum_k C_k$ before every Sinkhorn call.
3. **Fallback scripted.** Adaptive-Lagrangian arm written and switchable by
   config flag *now*, while there is no time pressure. It must never need to be
   written after a gate fires.

---

## Phase 1 — Implementation (D1–D2)

### 1.1 Head refactor — `clustering_head.py`

Split the head at the normalisation boundary. Everything upstream is unchanged:

```
compute_logits(h_t, edge_index) -> Z            # projection → prototypes → neighbor blend
normalize(Z, caps, T, mode) -> P                # softmax | sinkhorn
```

This separation is what lets the batched GNN compute all logits together while
Sinkhorn normalises each circuit independently. `mode` is `capacity_mode` in
`CLUSTER_CFG`, values `"sinkhorn" | "softmax"`; the softmax path is retained
verbatim as the ablation arm.

**Preserve the cosine logit bound.** Prototypes stay L2-normalised, the neighbor
blend stays convex. $Z \in [-1,1]$ is the assumption the entire iteration budget
rests on.

### 1.2 `CapacitySinkhorn` module

Inputs: `Z [B,N,K]`, `caps [K]`, `T`, `n_iters`.
Internally: append $C_{\text{total}} - N$ dummy rows with **zero logits** →
$[B, C_{\text{total}}, K]$; log-domain Sinkhorn with row marginals $\mathbf{1}$,
column marginals $C_k$; drop dummy rows; return `[B,N,K]`.

- Fixed iteration count, no Python convergence loop.
- Use a large finite negative constant, never `-inf`, for masked entries —
  $(-\infty) - (-\infty)$ produces NaN.
- Terminate on a **row** update so row sums are exact. A row that doesn't sum to
  1 is physically meaningless; a $10^{-4}$ column overshoot is absorbed by the
  hardener.
- Return residuals `(max_row_err, max_col_err)` alongside `P`.

Output format is unchanged `[N,K]`, so **no EFCL / cost-model code changes.**

### 1.3 Training loop

- `capacity_penalty = None` on the Sinkhorn path — set to `None`, not
  zero-weighted. If both are live the competing gradient is still there and the
  ablation means nothing.
- Delete `R_cap` from the loss; keep `CapacityPenalty` importable for the
  ablation arm only.
- Replace the `cap_penalty` logging channel with Sinkhorn residuals.
- `sf`, `beta`, `lambda_cap` become dead on this path — mark them in the config.

### 1.4 Deliverable

Unit tests: row/column marginals, zero-slack edge case, $C \gg N$ reduces to
softmax, permutation equivariance, gradient finiteness at $T_{\min}$,
infeasible-capacity raises.

---

## Phase 2 — Convergence stress test (D2, no training)

Purely numerical, minutes to run. Sets `sinkhorn_iters` from evidence.

**Grid:** $T \in \{3.0, 1.0, 0.5\}$ × logit regimes {random $\mathcal{U}[-1,1]$,
adversarial all-qubits-prefer-one-tech, near-degenerate} × marginals
{$[20,20]$ balanced; **and every imbalanced setting Phase 6 will use**, e.g.
$[36,4]$, $[40,4]$}.

**Criterion:** smallest `n_iters` keeping both
$\max_u |\sum_k Q_{uk} - 1|$ and $\max_k |\sum_u Q_{uk} - C_k|$ below tolerance
across the whole grid, then ×1.5 headroom. The $40\times K$ transport is
negligible next to GATv2+GRU, so buy headroom — 50 iterations is cheap.

> **Written into the config as a comment:** *any change to logit scale —
> including adding a learnable similarity scale to the cosine head — invalidates
> this budget and requires rerunning Phase 2.* This is the most likely silent
> breakage, because a learnable scale is exactly the fix one reaches for if
> Phase 3 shows diffuse assignments.

Each imbalanced hardware setting in Phase 6 inherits its own iteration budget
from this grid.

---

## Phase 3 — Fixed-$N{=}30$ pilot (D3–D6) — **HARD GATE**

Same dataset, same batching, same cost model as the last known-good run. The
only change in the system is the head. Nothing about variable $N$, generator
parameters, or batch logic is touched.

### Launch two runs together

| Arm | `capacity_mode` | Purpose |
|---|---|---|
| S | `sinkhorn` | the pilot |
| R | `softmax` + `CapacityPenalty` | **matched regularizer reference** |

Arm R is mandatory and must launch with S, not after. Every existing regularizer
result predates the ASAP timing change and is therefore on a different cost
model — it cannot serve as the reference. Arm R is also the E4 ablation arm, so
this costs nothing beyond GPU time.

Identical schedule for both arms ($T: 3.0 \to 0.5$, $\gamma{=}0.9845$, floor at
≈epoch 115 of 140). Any Sinkhorn-specific tuning that happens later requires
rerunning R under the matched schedule.

### Mandatory per-epoch logging

| Metric | Watch for |
|---|---|
| max row residual | any drift from ~0 |
| max column residual (incl. dummy) | **rise in late epochs at $T_{\min}$** — under-converged, raise iters, never raise $T_{\min}$ |
| hardener burden (qubits moved) | should approach ~0 late |
| assignment entropy, frac. top-1 > 0.8 / 0.9 | diffuseness — but see caveat below |
| per-tech occupancy vs. layer | static split |
| frac. layers whose hard assignment changes; mean qubits moved per transition | policy collapse |
| **cross-circuit assignment diversity** | the sharpest collapse signal |
| train/test EFCL, gradient norm | — |

> **Entropy is not comparable to Run 6.** Where the column constraint binds,
> Sinkhorn rows are structurally less sharp than softmax at the same $T$. Old
> reference values would read a healthy model as collapsed. Establish a new
> baseline from arm S itself, and use the behavioural metrics for collapse.
> Note also that per-tech occupancy is partly *forced* by the marginals, so at
> low $\rho$ it weakens as a signal; cross-circuit diversity does not.

### Gate criteria (all must hold)

1. No NaN/Inf, no gradient instability.
2. Row and column residuals below tolerance **throughout**, including late epochs.
3. Hardener burden very small by late training.
4. Assignments circuit-dependent, not one static partition.
5. Train/test EFCL converges.
6. Sinkhorn EFCL not systematically worse than arm R beyond run-to-run variance.

### Pre-committed outcomes

| Outcome | Reading | Action |
|---|---|---|
| Better EFCL + ~0 hardener burden | Strongest result | Proceed |
| Comparable EFCL + ~0 hardener burden | **Feasibility becomes structural at no cost to quality** — still a strong result, and a direct answer to the reviewer's objection that the regularizer hinders learning | Proceed; frame the paper claim this way, not as a margin win |
| Materially worse EFCL | Danger case | See below |

**Danger case:** first distinguish training problem from hypothesis-space
problem. Still-descending curves or persistently diffuse assignments → **one**
diagnostic/tuning cycle (annealing rate or $T_{\text{init}}$ or iteration count),
then re-gate. Converged and still materially worse → the structural constraint
is genuinely harming schedule quality; switch to the **adaptive-Lagrangian
fallback** (config flag, already scripted in Phase 0). Do not silently revert to
the fixed penalty — the reviewer objected to it by name.

### **Abort deadline: end of D8.**
One tuning cycle only. If the gate is not passed by end of D8, flip to the
Lagrangian arm and proceed to Phase 5 with it. A working Lagrangian with a clean
ablation beats a half-debugged OT head.

---

## Phase 4 — Parallel analyses (D3–D6, alongside the pilot)

Neither depends on Sinkhorn training. Run on a separate allocation.

### 4A — Generator drift across $N$

Generate candidate sets at $N = 30,\dots,39$ with **current** parameters and
compare *normalised* descriptors across $N$:

- idle fraction, 1Q/2Q density
- normalised interaction / cut rate (cut is the established axis)
- realised ROI block-height fraction (`max_block_h / N`)
- effective post-layering depth $T$

**Why:** block bounds are absolute qubit counts explicitly tuned per register
width (`# Rectangle bounds - 30`, with a commented `- 20` variant). At $N{=}30$
`max_block_h/N = 0.33`; at $N{=}39$ it is 0.26. Left unfixed, "training across
circuit size" silently also means "training across changing ROI geometry," and
the per-$N$ breakdown in Phase 5 becomes uninterpretable.

**Decision:** descriptors stable → keep the generator. Drifting → convert
spatial block-height parameters to fractions of $N$ and regenerate. Prefer
validation over modification; only change if the data demands it.

### 4B — Batching study

Compare **(i)** global depth-sorted, mixed-$N$ batches vs. **(ii)** $N$-bucketed
then depth-sorted.

Report, per candidate batching:

$$W_{\text{node-layer}} = 1 - \frac{\sum_i N_i T_i}{T_{\max}\sum_i N_i}, \qquad \text{mean batch occupancy}, \qquad \text{measured throughput}$$

Throughput protocol: same pre-cached circuits, same GPU, same batch count,
several timed windows after warm-up, compare **median step time and
circuits/sec**. Excludes dataset construction, CUDA warm-up, checkpointing,
eval cadence.

Wall-clock is required because $N$-bucketing yields more, smaller batches, which
are kernel-launch-bound rather than FLOP-bound — measured timing can invert the
ranking that $W_{\text{node-layer}}$ predicts.

**Decision rule:** meaningful saving for mixed-$N$ → implement ragged batching
(`torch.cat` for hidden states, `torch.split` / `PyGBatch.ptr` for the inverse;
GAT already handles disjoint graphs of differing size). Difference small →
$N$-bucketed, because it is simpler and makes cross-circuit mass leakage
structurally impossible.

**If ragged is chosen**, one additional mandatory test: perturb one circuit in a
mixed batch and assert every other circuit's $P$ is bit-identical. Without
per-circuit sizes the head cannot tell where one circuit ends, capacity mass
leaks across circuits, and training silently optimises a coupled objective —
a plausible loss curve and a wrong paper.

---

## Phase 5 — Variable-$N$ training (D7–D11)

Only after the Phase 3 gate passes and 4A/4B have reported.

1. **Stratified dataset.** 1200 circuits over $N = 30..39$ → **120 per size**.
   For each $N$ *separately*: oversample, measure effective depth, retain the
   120 closest to $T^*$. Then combine. Running the depth filter globally would
   let sizes that happen to land near $T^*$ dominate the size distribution.
2. **Sampler** per the 4B decision, batch order shuffled each epoch.
3. **Sinkhorn is size-agnostic here by construction** — the transport is
   $40 \times K$ for every $N$; only the real/dummy row split changes. $\rho$
   sweeps 0.75 → 0.975 across the size range, so slack magnitude is covered
   without any capacity augmentation.
4. **Report all Phase 3 diagnostics broken out per $N = 30..39$**, especially
   EFCL and hardener burden. A trend across $N$ means the model learned one
   slack regime — and if 4A found drift, disentangle slack-regime effect from
   circuit-family effect before interpreting.

---

## Phase 6 — Imbalanced capacity (D12–D13)

Reviewer requirement: results at $N = 200, 500, 1000$ with imbalanced
capacities, 10:1 given as the example.

- **Separate hardware setting, separate trained model** (claim A). Not a
  generalisation test of the Phase 5 model.
- Each setting inherits its own iteration budget from the Phase 2 grid.
- **Define settings by (ratio, $\rho$), not absolute numbers.** "The 10:1
  setting" is undefined across an $N$ ladder — $[40,4]$ means nothing at
  $N{=}200$ — and $\rho$ must be held fixed or imbalance is confounded with load.
- $[36,4]$ is 9:1, not 10:1. Exact 10:1 needs $C_{\text{total}}$ divisible by 11,
  e.g. $[40,4]$ at $C{=}44$. Either use exact ratios or say "strongly imbalanced."

**Anticipated reviewer question**, worth pre-empting in §8 rather than in
rebuttal: *why zero-shot across $N$ but retrained across capacity?* Because $N$
enters through the graph and a GNN encoder is size-agnostic by construction,
whereas capacity enters only through the projection, which cannot re-shape the
representation it projects. If a later stage wants one model across capacity
vectors, that requires explicit capacity conditioning of the encoder — flag as
future work, not as a limitation of this design.

---

## Critical path and parallelism

```
D0  config freeze + fallback scripted
D1–D2  implementation ─────────────┐
D2  convergence stress test        │
D3–D6  fixed-N pilot (S) + regularizer reference (R)   ║ 4A generator drift
        └─ GATE (abort D8) ────────┘                   ║ 4B batching study
D7–D11 variable-N dataset → batching impl → full run
D12–D13 imbalanced-capacity settings
D14  writing buffer
```

Only Phase 3 blocks Phase 5. Phases 4A/4B are genuinely parallel — the
ten-step list reads more sequentially than it is, which matters with three weeks
on the clock.

---

## Paper deliverables produced by this plan

| Artifact | Source |
|---|---|
| §4 method: Sinkhorn head replaces the capacity-regularizer subsection | Phase 1 |
| §5 implementation: dummy-row formulation, $\varepsilon{=}T$ equivalence, iteration budget | Phases 1–2 |
| **E4 ablation: Sinkhorn vs. regularizer on matched ASAP cost** | Phase 3 arms S/R |
| **Hardener burden, Sinkhorn vs. regularizer** — the cleanest quantitative statement that capacity moved from penalised to structural | Phase 3 |
| Size-sensitivity, per-$N$ breakdown | Phase 5 |
| Imbalanced-capacity results (reviewer requirement) | Phase 6 |
| §8 disclosures | capacity/claim-A scope; any Phase-3 tuning, on a dev split of the anchor pair only; SC $t_{2q}$ 100→200 ns and NA $f_{2q}$ 0.997→0.995 |
