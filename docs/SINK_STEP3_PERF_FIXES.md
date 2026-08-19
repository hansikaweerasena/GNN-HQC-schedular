# Performance fixes — pre-launch

All three confirmed and fixed. 5/5 regression checks pass.

## 1. Per-circuit `.item()` in `batch_train_step` — mine, fixed

`total_efcl += float(loss.item())` was introduced when EFCL was split out of
the loss. At `batch_size=32`: **32 syncs/batch for arm S, 64 for arm R** (the
penalty `.item()` was pre-existing).

Now accumulated as detached device scalars, read once after the circuit loop:
**exactly 2 syncs per batch** (1 without a penalty), independent of batch size.

One correction to the proposal: this is *not* numerically identical as stated.
The old code accumulated in Python `float64`, so accumulating in `float32` on
device would have changed the last bits. Accumulating in **`float64` on device**
instead makes it exact — verified to 1e-12 — and a scalar `.double()` costs
nothing.

## 2. Grad norm — pre-existing, fixed

`_p.grad.data.norm(2).item()` per parameter per batch: ~40 syncs x 38 batches x
140 epochs ≈ **210k syncs per run**. The "cheap" comment was indeed misleading —
reading already-computed grads is cheap, draining the CUDA queue to do it is not.

Implemented the stronger version: the squared norm is summed on device, the
per-batch norm accumulates as a device scalar, and `.item()` is called **once
per epoch** — 38 syncs/epoch → 1.

Also not bit-identical, and again the new value is the *better* one: the old
version took each per-parameter norm in fp32 before squaring in Python, so it
carried fp32 rounding. Verified to agree to fp32 epsilon (3.4e-8 relative).
Grad norm is diagnostic and never enters the update.

## 3. Hardener transfers in `circuit_diagnostics` — fixed, largest win

`enforce_capacity_sequence` moves each layer to CPU and its result back, so
calling it on device tensors costs `2L` transfers per circuit.

Measured impact at 300 test circuits, `eval_every=5`, 140 epochs:

| | transfers per run |
|---|---|
| before | **1,344,000** |
| after | **8,400** |

Fixed by moving the whole `[L,N,K]` stack to CPU once at the top of
`circuit_diagnostics` and running the entire diagnostic there. This also makes
the ~11 trailing `float()` reads free — they were one sync each on device
tensors, another ~3,300 syncs per eval that the proposal did not mention but
that the same fix removes. Verified: device-targeted `.to()` calls no longer
scale with `L` (0 calls at `L=40`).

The hardener itself is unchanged, so the reported burden is still the real
inference-path number.

**Residual cost worth knowing.** The transfers are gone, but the hardener's own
Python loop is not: it is `O(L*N*K)` per circuit with a Python inner loop, and
measures ~85 ms per 80-layer circuit — roughly **10-12 min per training run** at
300 test circuits and `eval_every=5`. That is acceptable, and vectorising it
would change the greedy repair semantics and therefore the reported burden, so
it is left alone. If evaluation time becomes a problem, the right lever is
`eval_every`, or computing the hardener diagnostic on a fixed subsample of the
test set rather than all of it.
