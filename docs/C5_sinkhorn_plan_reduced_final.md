# C5 — Sinkhorn Head: Implementation Plan (reduced)

**Goal:** replace the capacity regularizer with a structurally feasible head,
verify it does not hurt schedule quality, restore variable-$N$ training.
Nothing else.

**Explicitly out of scope:** imbalanced-capacity evaluation ($N{=}200/500/1000$,
10:1 ratios) → separate evaluation plan. Ragged mixed-$N$ batching → conditional
optimization, Step 5 only. Adaptive-Lagrangian fallback → not written unless
invoked; arm R already provides the shippable fallback.

**Critical path:**
`implement → numerical test → 30q pilot → variable-N data → N-bucket training → optimize batching only if needed`

---

## Step 1 — Minimal implementation

**1a. Head refactor.** Split at the normalization boundary; everything upstream
(projection, prototypes, neighbor blend) unchanged.

```
compute_logits(h_t, edge_index) -> Z        # unchanged, Z in [-1,1]
normalize(Z, caps, T, mode) -> P            # "softmax" | "sinkhorn"
```

Prototypes stay L2-normalised and the neighbor blend stays convex — the
$Z \in [-1,1]$ bound is what the iteration budget rests on. **Note in the config:
adding a learnable similarity scale invalidates the budget and requires
rerunning Step 2.**

**1b. `CapacitySinkhorn`.** Balanced log-domain Sinkhorn. Append
$C_{\text{total}} - N$ zero-logit dummy rows → $[R{=}C_{\text{total}}, K]$; row
marginals $\mathbf{1}$, column marginals $C_k$; drop dummy rows; return $[N,K]$.

- **Signature `sinkhorn(Z, caps)` with `Z: [..., N, K]`** — all reductions on the
  last two dims. The module appends the $C_{\text{total}}-N$ dummy rows
  internally to form `[..., R, K]`, and returns only the real rows
  `[..., N, K]`. Single-circuit is the no-leading-dims case, so the operator
  encodes no batching assumption and is unchanged if ragged batching ever
  arrives; a fixed-$N$ batch passes `[B,N,K]` and runs vectorised.
- Terminate on a row update. Rows feed EFCL as probability distributions, so a
  row summing to 0.9999 systematically under-counts cost; a $10^{-7}$ column
  residual is inert. This is ordering, not a tolerance trade — Step 2 sizes
  iterations so **both** residuals are negligible.
- Return `(P, max_row_residual, max_col_residual)`.

Output format is unchanged $[N,K]$ → **no EFCL / cost-model changes.**

**1c. Loss.** $\mathcal{L} = C_{\text{EFCL}}$. Set `capacity_penalty = None` on the
Sinkhorn path — `None`, not zero-weighted; if both are live the competing
gradient remains and the ablation is meaningless. Retain `CapacityPenalty`
importable for arm R. `sf`/`beta`/`lambda_cap` are dead on this path.

Do not touch EFCL, GNN, dataset, or batching.

---

## Step 2 — Numerical smoke test (no training, minutes)

At $T_{\min} = 0.5$, plus $T{=}3.0$ as a sanity check:

| Logits | Capacities |
|---|---|
| random $\mathcal{U}[-1,1]$ | $[20,20]$ |
| all-qubits-prefer-one-tech | one strongly imbalanced setting |

Pick **one** conservative iteration count making both row and column residuals
negligible across all of the above. Don't round up to 50 reflexively — at 80
layers × 140 epochs the count is a linear multiplier on dispatch overhead.

**Tests kept — these four only:** row/column marginals; finite gradients at
$T_{\min}$; infeasible capacity raises; dummy-row/slack behaviour for $N < C$.

The reviewer's "if every technology can fit the whole circuit, would the
scheduler put everything on one technology?" is **not** a unit test — Sinkhorn
still chooses column potentials to meet the prescribed marginals and the dummy
rows interact with them, so no exact softmax equivalence should be asserted.
Answer it empirically in the evaluation: give every technology capacity $\ge N$,
run the trained scheduler, and observe whether the hardened schedule concentrates
on one technology when EFCL favours it.

Then stop working on Sinkhorn internals.

---

## Step 3 — Fixed-$N{=}30$ pilot — **gate**

Existing pipeline unchanged. Confirm before launch: `C_total > N` **strictly**,
and record $\rho = N/C$.

Why strict: at $C_{\text{total}} = N$ Sinkhorn is still meaningful — the split
size is forced, but *which* qubits go where remains the full scheduling problem
(cut structure, activity, idle behaviour, temporal assignment are all still
learned). What is not exercised is any dummy slack. Since the pilot must validate
both capacity enforcement **and** unused-capacity behaviour, require
$C_{\text{total}} > N$ here.

Two runs, launched together, identical schedule ($T: 3.0 \to 0.5$,
$\gamma{=}0.9845$):

| Arm | Config |
|---|---|
| **S** | `sinkhorn`, no capacity penalty |
| **R** | `softmax` + `CapacityPenalty`, current EFCL/ASAP timing |

R is mandatory: every existing regularizer result predates the ASAP change, so
it is not a valid reference. R is also the E4 ablation arm and the shippable
fallback if S fails.

Short smoke run first to catch NaNs.

**Logged:** train/test EFCL · max Sinkhorn marginal residual · hardener burden ·
mean movement / fraction of layer transitions that change assignment · gradient
norm. *(Optional, one scalar: std across circuits of mean per-tech occupancy —
catches a policy that outputs the same partition for every circuit while still
varying across layers.)*

**Gate — five questions:**
1. Numerically stable?
2. Hardener burden becomes very small and substantially lower than arm R by late training?
3. EFCL converges?
4. Matches or beats R within normal run-to-run variance?
5. Schedules circuit-dependent and changing, not static?

**Pass** → freeze Sinkhorn, no further tuning.
**EFCL materially worse** → one tuning attempt, temperature schedule only. Still
worse → stop Sinkhorn development, ship arm R, invoke the fallback decision.

---

## Step 4 — Variable-$N$ dataset

Only now change the data.

- $N = 30..39$, **stratified equally** in both train and test sets: 120 training
  circuits per size and, for a 300-circuit test set, 30 test circuits per size.
- **Depth selection within each $N$ group, not globally, for both train and test.** A global filter lets
  sizes that happen to land near $T^*$ dominate the size distribution.
- **Generator: validate first, rescale only if needed.** Block heights are
  absolute qubit counts tuned at $N{=}30$, so `max_block_h/N` drifts 0.33 → 0.26
  across the range — but 30→39 is not a large span and an unnecessary generator
  change is its own risk. Generate at $N \in \{30, 35, 39\}$ with the **existing**
  generator and compare idle fraction, 2Q density, and normalised ROI/block-height
  statistics. Only on material drift, convert spatial heights to fractions of $N$.

---

## Step 5 — Batching: simple first

Use `N-bucket → T-sort`. Run a variable-$N$ smoke test and record circuits/sec
against the current fixed-$N$ pipeline.

Slowdown acceptable → stop. Only if clearly substantial (~>15–20%) implement
ragged mixed-$N$ batching (`torch.cat` for hidden states, `torch.split` /
`PyGBatch.ptr` for the inverse). If that happens, one extra test is mandatory:
perturb one circuit in a mixed batch, assert every other circuit's $P$ is
bit-identical — otherwise capacity mass leaks across circuits and training
silently optimises a coupled objective.

---

## Step 6 — Full variable-$N$ run

Train the 30–39 model. Report EFCL and hardener burden **by $N$**, primarily to
confirm no deterioration toward $N{=}39$ where slack is smallest (1 dummy row at
$C{=}40$).

Then hand off to the separate evaluation plan (imbalanced capacities, MQT
zero-shot, baselines).
