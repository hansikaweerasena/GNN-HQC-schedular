# Step 1 — implementation notes

## Files

| File | Change |
|---|---|
| `src/sinkhorn.py` | **new** — `capacity_sinkhorn()` + `CapacitySinkhorn` module |
| `src/clustering_head.py` | refactored: `compute_logits()` / `normalize()`; `capacity_mode`, `caps`, `sinkhorn_iters` |
| `src/evolving_gnn.py` | **one argument added** (`n_qubits=N` on the head call). Batching untouched. |
| `utils/train_utils.py` | **unchanged** — `capacity_penalty` is already optional |
| `tests/test_sinkhorn.py` | **new** — the four required checks + iteration sweep |

## Step 2 result — `sinkhorn_iters = 30`

Worst-case residual `max(row, col)` over {T=0.5, 3.0} x {balanced [20,20],
imbalanced [36,4]} x {random, adversarial} logits, N=30:

| n_iters | 5 | 10 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| worst residual | 1.5e-01 | 3.0e-04 | 3.8e-06 | 3.8e-06 | 3.8e-06 | 3.8e-06 |

20 iterations reaches the plateau; **30 is the recommendation** (1.5x headroom).

The 3.8e-06 plateau is **float32 precision, not stalled convergence** — the same
case in float64 gives 1.3e-09 at 20 iterations and 1.4e-14 at 50. Tolerance is
therefore set at 1e-5, which is four orders below the hardener's granularity of
one qubit.

50 was the earlier suggestion; 30 is chosen instead because n_iters is a linear
multiplier on dispatch overhead across 80 layers x 140 epochs, and the sweep
shows nothing is bought above 20.

## Config additions (`CLUSTER_CFG`)

```python
"capacity_mode":  "sinkhorn",   # "sinkhorn" (arm S) | "softmax" (arm R)
"sinkhorn_iters": 30,           # from the Step 2 sweep; see note below
```

> Changing the logit scale — including adding a learnable similarity scale to
> the cosine head — invalidates `sinkhorn_iters` and requires rerunning the
> Step 2 sweep. The budget is derived from the Z in [-1, 1] bound.

## Training-script edits (Step 3)

```python
base_caps = torch.tensor(
    [float(t["capacity"]["max_qubits"]) for t in config["techs"]], dtype=torch.float32)

assert float(base_caps.sum()) > CIRCUIT_SOURCE_CFG["kwargs"]["num_qubits"], (
    "pilot requires C_total > N strictly, so that dummy slack is exercised")

use_sinkhorn = CLUSTER_CFG["capacity_mode"] == "sinkhorn"

cluster_module = SegmentClustering(
    ...,                                   # unchanged
    capacity_mode  = CLUSTER_CFG["capacity_mode"],
    caps           = base_caps if use_sinkhorn else None,
    sinkhorn_iters = CLUSTER_CFG["sinkhorn_iters"],
).to(device)

# Arm S: capacity leaves the loss entirely. None, not zero-weighted --
# if both are live the competing gradient remains and the ablation is void.
cap_penalty_module = None if use_sinkhorn else CapacityPenalty(total_cost_module, config).to(device)
```

Replace the `cap_penalty` logging channel with `cluster_module.diagnostics`
(`{T, row_residual, col_residual}`). Guard the `lambda_cap` log line behind
`if cap_penalty_module is not None`.

## Verified

- Row/column marginals: balanced + imbalanced, random + adversarial logits.
- Batched `[B,N,K]` result identical to per-circuit `[N,K]` (max dev 0.00e+00).
- Finite non-zero gradients at T_min for both capacity settings.
- `sum(caps) < N` and non-integer caps both raise.
- Slack: occupancy reaches [19.66, 10.34] under adversarial logits, i.e. **not**
  pinned to the [15,15] proportional split — the property that distinguishes
  dummy rows from renormalised column marginals. Occupancy approaches but does
  not reach the cap: with Z in [-1,1] the maximum ratio at T=0.5 is exp(4) ~ 55:1,
  so the entropy term legitimately retains a little mass on the disfavoured
  technology.
- `C_total == N` edge case finite and exact ([15,15]); no dummy rows exercised,
  which is why the pilot requires `C_total > N`.
- Both modes run inside `batch_forward`; single-circuit path matches batched
  exactly; gradients flow through GNN + head.
