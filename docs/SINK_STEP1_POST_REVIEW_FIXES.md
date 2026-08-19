# Post-review fixes — Step 1

All four issues confirmed and fixed. 24/24 checks pass; each fix has a
regression test.

## A. Device synchronisation on the hot path — fixed

Confirmed: `capacity_sinkhorn` was calling `torch.allclose`, `caps.sum().item()`
per layer; the head passed `float(self.temperature)` (a device buffer); and
`CapacitySinkhorn.forward` called `.item()` twice on the residuals. Five syncs
per layer x ~80 layers x ~3k forwards/epoch x 140 epochs.

- `validate_caps()` runs once in `__init__` / `set_caps`; `c_total` stored as a
  Python int, `log_caps` as a precomputed buffer.
- `capacity_sinkhorn` gains `validate=False` + `c_total` / `log_caps` on the
  training path; `validate=True` remains the default for standalone/test use.
- The head keeps a **Python-float mirror** `self._T` of the temperature buffer,
  written by `set_epoch` and refreshed on checkpoint load.
- Residuals stay detached device tensors; `.item()` only in `residuals()`.

**Interaction he did not flag:** his fix B (validate `T > 0`) would reintroduce
a sync if `T` arrives as a device tensor. Resolved by the float mirror —
temperature is validated once at construction and once per `set_epoch`, never
per layer.

**Subtlety in the mirror:** the refresh hook must be `_load_from_state_dict`,
not `load_state_dict`. When the head loads as a child of `SegmentClustering`,
PyTorch recurses through the former and an override on the latter is never
called — an eval-only load would then run at `T_init = 3.0` while the buffer
held `T_min = 0.5`, silently evaluating the checkpoint at the wrong temperature.
(My first attempt had exactly this bug; there is now a test for it.)

Regression test: monkeypatches `Tensor.item` and `torch.allclose` and asserts
**zero** calls during a full `batch_forward`.

## B. Residuals overwritten per layer — fixed

Confirmed. `diagnostics` reported only the last layer of the last circuit.
Replaced with a running **maximum** accumulated on device across all forwards
since `reset_diagnostics()`. Call `reset_diagnostics()` at the start of each
epoch and read `diagnostics` when logging.

## C. `T` and `n_iters` guards — added

`T <= 0` and `n_iters <= 0` now raise. Temperatures are also validated at head
construction, so a bad config fails before training rather than producing NaNs.

## D. Capacity validation — fixed

Confirmed `caps = [-1, 41]` was accepted (sum = 40 >= N, integer-valued) and
produced a column residual near 1. Now: strictly positive required, integer
check tightened to `atol=1e-6, rtol=0`, and — the important part — the
**rounded** values are used thereafter. Previously `20.0001` passed
`allclose`, then `log_c` used 20.0001 while the dummy count used `round(sum)`,
leaving row and column total mass unequal and the problem silently unbalanced.

---

# On the conceptual point: argmax feasibility

He is right, and it reproduces exactly. At `C = [20,20]`, `N = 30`, adversarial
logits: soft occupancy `[19.66, 10.34]` (feasible), every row `[0.655, 0.345]`,
so all 30 argmax to technology 0 and the hardener must move 10. Now documented
in the module docstring, and `argmax_violation()` is provided as the
hardener-burden metric.

His proposed paper wording is the right one, and the claim of near-zero burden
must wait for the pilot.

## What the burden actually depends on — measured

Burden is **not** driven by temperature alone. With well-spread logits it is
zero at every temperature. It is driven by how *homogeneous* the per-layer
preferences are. Mean qubits moved (of 30), 200 seeds, `Z = +/-b + U[-0.3,0.3]`
clamped to `[-1,1]`, where `b` is a shared preference for one technology:

| b | T=3.0 | T=1.0 | T=0.5 | T=0.25 | T=0.1 |
|---|---|---|---|---|---|
| 0.0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.2 | 0.61 | 0.61 | 0.64 | 0.69 | 0.66 |
| 0.4 | 3.62 | 3.56 | 3.40 | 2.73 | 1.03 |
| 0.7 | 7.51 | 7.14 | 6.13 | 3.72 | 1.05 |
| 1.0 | 10.00 | 9.97 | 9.35 | 7.06 | 2.10 |

Three consequences for the pilot:

1. **Hardener burden is a proxy for preference homogeneity.** A layer where
   every qubit prefers the same technology is one where capacity, not physics,
   forces the split — and there the *hardener*, not the model, is choosing which
   qubits move. That is a scheduling decision being delegated to a greedy
   rounding rule, which matters for the paper's claim, not just for hygiene.
2. **`T_min = 0.5` does not drive burden to zero** under homogeneous
   preferences. Because cosine logits are bounded to `[-1,1]`, the sharpest
   available ratio at `T_min` is `exp(4) ~ 55:1` — nowhere near the `T -> 0`
   limit where the entropic solution approaches an integral vertex of the
   transport polytope (which is where argmax feasibility would be guaranteed).
3. **The single permitted tuning attempt has a pre-computed budget.** If the
   pilot shows non-trivial burden, the diagnosis is almost certainly `T_min` too
   high, and the iteration counts are already measured: `T_min = 0.25` needs 30
   iterations, `T_min = 0.15` needs 30, `T_min = 0.1` needs 50-75. No second
   sweep required.

---

# Integration items he could not verify

- **`caps` already travel in the checkpoint** — `head.sinkhorn.caps` is a
  registered buffer and appears in `state_dict()`. Verified.
- **Eval scripts will break**, as he says: `SegmentClustering` now defaults to
  `capacity_mode="sinkhorn"` and raises without `caps`. Options: (a) write
  `capacity_mode` / `sinkhorn_iters` / `caps` into the saved run-config snapshot,
  or (b) preferably add one shared `build_cluster_module(cfg, cost_config)`
  helper used by training and eval alike, so construction logic stops being
  duplicated across scripts. (b) is the real fix; (a) is enough to unblock.
  Either way, after the pilot launches, not before.
- `train_hipergator.py` edits remain as specified in `STEP1_NOTES.md`, plus one
  addition: call `cluster_module.reset_diagnostics()` at the start of each epoch.
