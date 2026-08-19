# C1 — Movement-aware ASAP timing surrogate

**Scope:** `cost_function.py` + the two `cost_config_*.json` files. Nothing else changes.

**Naming (use verbatim in §4):** EFCL implements a *movement-aware ASAP timing surrogate*;
`lowering.py` implements *block-synchronized ASAP lowering*. The surrogate deliberately omits
boundary synchronization. Do not claim equivalence.

**Explicitly not doing:** soft block detection, module-scope synchronization, ALAP, measurement
timing, `pos`/SABRE/wire machinery, per-gate transport solves, τ / LSE / hybrid-λ annealing.

---

## Step 0 — Reconcile configs to `techs.json` (do first, ~10 min)

`techs.json` (`_id: techs_v3`, frozen 2026-08-11) is the single source of truth. Verified diff:

### `cost_config_v3.json` (SC+NA)

| tech | field | current | → | reason |
|---|---|---|---|---|
| sc | `coherence.T2` | 80000 | **100000** | |
| na | `gate_fidelity.f2q` | 0.997 | **0.995** | techs_v3 revision (demonstrated Rydberg CZ) |
| na | `gate_time.t1q` | 200 | **500** | |
| na | `gate_time.t2q` | 2000 | **1000** | **10× effect on `B` matrix** |
| na | `coherence.T2` | 200000 | **2000000** | **10× — the parking argument lives here** |
| — | `comm.f_comm` | 0.95 | **0.97** | |

### `cost_config_tp2n_99.json` (SC+TI)

| tech | field | current | → |
|---|---|---|---|
| sc | `coherence.T2` | 80000 | **100000** |
| — | `comm.f_comm` | 0.95 | **0.97** |

All TI values already agree. `routing.kappa` / `all_to_all` already agree in both files — no change.

### New / retired keys (both files)

```jsonc
"timing_model": { "mode": "asap" }        // was "smooth_max"; tau*/lambda* now dead
"comm":  { "t_remote": ... }              // dead on asap path — keep for barrier ablation
"timing":{ "delta": 500.0 }               // dead on asap path — keep for barrier ablation
"decoherence_model": { "mode": ... }      // ignored on asap path
```

> **Flag:** `capacity_penalty.safety_factor` is `20.0` on disk, but the working recipe is `sf=2`.
> Resolve before the smoke run so you know which you're testing.

---

## Step 1 — New module: `ASAPTimingCost`

Replaces `SegmentTimeV3` + `IdleCostV3(idle_only)` on the ASAP path. Both old classes stay
**untouched** behind `mode="barrier"` (needed for E4 and the block-vs-barrier figure).

### Buffers (built once in `__init__`)

```python
# B[i,j] = expected 2Q duration, technology i × technology j
#   diagonal    B[i,i] = t2q_i            (local; routing inflation added per-gate)
#   off-diag    B[i,j] = max(t2q_i, t2q_j)   (t_comm, symmetric — teleported gate)
B = torch.maximum(t2q[:, None], t2q[None, :])   # off-diagonal is already max
B.fill_diagonal_(...)                           # diagonal = t2q_i (same value; explicit for clarity)

invT2 = 1.0 / T2                                 # [K]
```

**Movement time uses `t2q[None, :]` directly — no `[K,K]` move matrix.**

```python
# t_move(i, j) = t2q_i  (SOURCE-SIDE, asymmetric: Bell measurement at source only).
# The relu/moved-mass form below is the exact min-coupling transport cost ONLY because
# this rule depends on the SOURCE index alone. If t_move ever becomes symmetric
# (e.g. max(t2q_i, t2q_j)), relu is no longer the transport cost and this breaks silently.
```

### Forward

```python
ready = torch.zeros(N)                       # [N], out-of-place updates only
Cidle_gate = 0.0

for l, layer in enumerate(layers):
    P = P_seq[l]                             # [N,K]
    invT = P @ invT2                         # [N]  post-move tech (see note)

    # --- boundary: movement busy time (moved mass) ---
    if l > 0:
        leaving  = torch.relu(P_seq[l-1] - P)          # [N,K]
        d_move   = (leaving * t2q).sum(dim=1)          # [N]
        ready    = ready + d_move                      # busy, NOT charged to T2

    # --- 1Q ---
    if oneq_u.numel():
        d1 = P[oneq_u] @ t1q                           # [G1]
        ready = ready.index_add(0, oneq_u, d1)         # out-of-place

    # --- 2Q (measurement: none; circuits have readout stripped) ---
    if tu.numel():
        ru, rv = ready[tu], ready[tv]
        s = torch.maximum(ru, rv)
        Cidle_gate = Cidle_gate + ((s - ru) * invT[tu] + (s - rv) * invT[tv]).sum()

        Pu, Pv = P[tu], P[tv]
        d = torch.einsum('gi,gj,ij->g', Pu, Pv, B) \
            + ((Pu * Pv) * Gamma * t2q).sum(-1)        # Gamma = layer_d["twoq_gamma"]  [G,K]
        new = s + d
        ready = ready.scatter(0, torch.cat([tu, tv]), torch.cat([new, new]))

# --- tail ---
Tmax   = ready.max()
Ctail  = ((Tmax - ready) * (P_seq[-1] @ invT2)).sum()
```

Notes that are load-bearing:

- **Layer index order *is* ASAP.** `ready[u]` advances only when `u` is touched, and the layering
  is a valid topological sort, so walking layers in index order reproduces per-qubit ASAP. Put this
  in a code comment — it is the correctness argument and it is not obvious.
- **1Q and 2Q within a layer touch disjoint qubits**, so their order inside the loop is irrelevant.
  Add a test-only assertion that no qubit appears twice in a layer; `scatter` would silently pick a
  winner otherwise.
- **Idle is charged at `P_l` (post-move).** The migration already happened at the boundary, so the
  qubit waits at its destination `T2`. There is no pre-move wait left to mischarge — that term is
  exactly what omitting boundary sync drops.
- **Movers are booked busy, not idle.** `f_move` already aggregates transfer failure; charging
  `d_move / T2` on top double-counts. Matches `lowering.py`.
- **Diagonal-vs-`Gamma`:** `B` diagonal carries bare `t2q_i`; the second term adds
  `Σ_k P_u(k)P_v(k) Γ_k t2q_k`, giving `(1+Γ)t2q` locally without materializing a `[G,K,K]` tensor.

### Returns

`{"Cidle_gate", "Ctail", "makespan"}` plus detached diagnostics (see Step 4).

Idle is attributed per-segment (gate idle → the layer it occurred in; tail → last segment) so
`per_segment_total` keeps its current shape and nothing downstream breaks.

---

## Step 2 — `CommMoveCostV3`: movement fidelity to moved mass

Same relaxation bug, same fix. Keeps the two definitions of "moved" consistent.

```python
# was: change_prob = 1 - (W[:-1] * W[1:]).sum(-1)     # assumes independence across layers
#      → charges 0.5 movement for a constant [0.5,0.5] assignment
p_moved = torch.relu(W[:-1] - W[1:]).sum(-1)          # [S-1, N]  = ½‖P_{s-1}-P_s‖₁
per_segment_move[:-1] = cmove * p_moved.sum(dim=1)
```

The **communication** (`ccomm × cut_prob`) term is unchanged. Cut probability is a genuine
within-layer independence assumption across two *different* qubits; movement was the same qubit
at two times, which is why only the latter was wrong.

---

## Step 3 — `TotalCost` wiring

- Dispatch on `timing_model.mode ∈ {"asap", "barrier"}`.
- On `"asap"`: assert `comm.t_remote` and `timing.delta` are unread; ignore `decoherence_model.mode`.
  Fail loudly rather than let someone tune a dead knob.
- `C_total = C_exec + (Cidle_gate + Ctail) + C_comm_fidelity + C_move_fidelity`.
- Delete nothing from `ExecCostV3` — `Cm = -log(fm)` stays (`nm` is all zeros anyway; dropping
  measurement *timing* is not dropping measurement *fidelity*).

---

## Step 4 — Correctness tests (all before training)

| # | Test | Assertion |
|---|---|---|
| T0 | Moved mass | `P_{t-1}==P_t` → 0. One-hot SC→NA → `t2q_sc=200`. NA→SC → `t2q_na=1000`. |
| T1 | **Zero-migration vs `lowering.py`** | One-hot `P_seq`, no migration, `Γ=0` profile, readout stripped on both sides → per-qubit idle **and** makespan match to float tolerance. This is the exactness limit: one block means no boundaries, so the surrogate and the lowerer must agree exactly. |
| T2 | Unequal-duration wait | q0 ready at 200, q1 at 2000, then 2Q → q0 charged exactly `1800 × invT2[tech(q0)]`. |
| T3 | Directional movement | i→j and j→i give different `d_move`. |
| T4 | Remote duration | One-hot cross-tech 2Q → `d == max(t2q_i, t2q_j)` exactly. |
| T5 | **Autograd** | One small circuit containing movement + 1Q + a 2Q wait + a remote gate. Soft `P_seq`, `loss.backward()`. Assert every `P.grad` is not None and finite; total grad norm > 0; and the specific layers carrying the wait and the movement each have nonzero norm. Empty/symmetric layers are allowed to be zero. |

T5 is the only test that exercises the soft path. A stray `.detach()`, an in-place op, or a
dropped `scatter` branch passes T0–T4 unchanged.

---

## Step 5 — Diagnostics before committing compute

Four numbers. Each isolates one thing and each has a named failure mode.

1. **idle : comm cost ratio**, barrier vs ASAP, on 2–3 M1 circuits, hand-checked.
   *Expected:* idle does not collapse — it relocates from diffuse depth penalty to cross-technology
   2Q stalls. *If it collapses toward zero for SC+NA,* the learnable signal has moved and you need to
   know now, not on day 9.
2. **Movement-busy fraction:** `T(P, movement enabled) / T(P, movement disabled)` at fixed `P`, at
   init. Isolates the movement term from technology choice. Should be ≈1 after the moved-mass fix.
3. **`lambda_cap` before/after.** `ccomm` moves `-log(0.95)=0.0513 → -log(0.97)=0.0305`, and
   `Δ_2q` is the dominant term, so `lambda_cap` drops ~30%. `Δ_idle` is now derived from the dead
   `timing.delta`. Log `lambda_cap` and per-layer excess. *If capacity violations blow up in the
   smoke run, this is the first suspect, not the timing model.*
4. **Sharpness:** mean `max_k P[u,k]` per epoch. Moved mass removes a sharpening gradient from two
   places at once (timing and fidelity). The bilinear 2Q locality term should carry it — local is
   ~30× cheaper than remote at `f_comm=0.97` — but confirm. *If `P` stays diffuse, this is the
   mechanism, and the fix is Sinkhorn-side, not timing-side.*

---

## Step 6 — Smoke run, then Sinkhorn

One short run: **ASAP + the existing capacity regularizer**, small subset, few epochs. Confirms the
new timing objective trains at all. Only then swap in the Sinkhorn head.

Rationale: ASAP deletes the `tau` annealing curriculum. Sinkhorn's entropy schedule becomes the only
remaining sharpening schedule, and it has a hard abort gate. Landing both at once turns a
two-variable failure into a guessing game. This is one short run, not another experimental branch.

---

## For the paper (§4 / assumption ledger)

- Moved mass is the **optimal-transport relaxation** of the hard movement cost under a source-side
  rule, not a heuristic: minimizing `Σ_{i≠j} π_ij t2q_i` over couplings with the two layers'
  marginals gives `π_ii = min(P_{t-1}(i), P_t(i))`, leaving exactly `relu(P_{t-1}-P_t)`. It also
  agrees with what argmax rounding does at inference. (Pleasant symmetry: movement and the
  assignment head are now both OT.)
- Add to the ledger, next to the existing "favours dynamic" entries: *omitting boundary
  synchronization produces an optimistic timing bias; we expect approximation error to become more
  pronounced for migration-heavy schedules.* **Not** "EFCL makespan is a lower bound" (the composite
  model also differs through Γ vs. SABRE, which is unsigned) and **not** "the gap grows with
  migration count" (the sync penalty depends on slack distribution, not migration count).
- E0 framing: matching Aer's timing improves correlation, so state plainly that E0 now tests the
  **noise channels and routing** — the parts EFCL never sees — not timing fidelity.
- Migration-count sweep: worth doing, as a small **late** validation, not a blocker.
- SC+TI: the residual drift movement scales with `t2q_ti = 100 µs`, so a few-percent layer-to-layer
  drift still costs µs of phantom busy time. Defer unless you actually train SC+TI; diagnostic 2
  above is the number that would explain a TI training failure, and it is free once logged.
