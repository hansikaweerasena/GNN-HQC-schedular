"""Circuit family generator for the Phase-0 motivation experiments (P0.4).

Three randomized families. No hand-crafted instances, no acceptance filter that
could remove circuits unfavourable to HQC.

    hotcore    M1 treatment. A persistent active region plus less-active
               storage qubits -- the structure real algorithms show, where most
               qubit-time is idle but the idleness is unevenly distributed.
    uniform    M1 control. Same MEAN activity as its hotcore partner, but
               activity spread evenly across qubits.
    phased     M2. The active region changes 2-3 times during execution, so the
               preferred placement changes with it.

The one thing the generator is NOT allowed to do
------------------------------------------------
Reject a circuit because HQC would lose on it. Validation here is purely
structural -- "did we build the class we asked for?" -- never outcome-based.
There is deliberately no check that any qubit crosses the duty threshold, no
cut-cost ceiling, and no minimum activity imbalance beyond what distinguishes
hotcore from uniform. If raising cold-qubit activity eventually removes the HQC
advantage, that is a result about when heterogeneity helps, not an instance to
discard.

The one check that IS enforced: fragility
-----------------------------------------
Pure dephasing does nothing to a computational basis state, so a circuit whose
idle qubits sit in |0> measures NOTHING -- 2xSC, 2xNA and SC+NA all return the
same number. That is a broken instrument, symmetric across every arm, and it
can only ever remove instances that carry no information either way.

Fragility is measured from the IDEAL state, not from circuit structure: a qubit
is fragile iff its reduced density matrix is not a pure computational basis
state, i.e. max(|rho01|, 1 - purity) > eps. Structure is unreliable here --
H followed by CX can return a qubit to a basis state, and a Bell-pair half has
|rho01| = 0 yet still decoheres (the coherence lives in the global |00>/|11>
superposition, which purity catches).

Pairing hotcore with uniform
----------------------------
`uniform` is generated to match its hotcore partner's REALISED mean activity,
not its parameters. Same seed index, same mean, different variance -- so the
pair differs in one variable and the comparison can be reported paired.

High mean idleness motivates the regime but does not predict an HQC win: a
circuit where all eight qubits idle 80% uniformly is exactly the control. What
heterogeneity needs is VARIANCE in idleness across qubits. That is the whole
point of the pairing, and it is what makes the control load-bearing rather than
decorative.
"""

from dataclasses import dataclass, field
import random

import numpy as np
from qiskit.circuit.library import HGate, XGate, ZGate, RZGate, CXGate

from .circuits import LayeredCircuit, make_layers
from .lowering import lower
from .hardware import Module, TECHS
from .scoring import ideal_state

__all__ = [
    "FamilySpec", "CircuitProfile", "profile", "fragility",
    "generate", "generate_pair", "generate_family", "FAMILIES",
    "validate_structure", "duty_threshold",
]

H, CX = HGate(), CXGate()

N_QUBITS = 8
DEPTH = 20

# Storage-region activity. The upper end is where the SC+NA advantage disappears at
# depth 20 (measured: +46% at 0.01, +14% at 0.15, +0.3% at 0.20, -4.9% at 0.30), so the
# family SPANS the crossover rather than stopping short of it. Chosen after measuring
# that boundary -- say so, and report the sweep beside the family result so the range is
# a described property of the workload class and not a hidden filter.
P_COLD = (0.01, 0.20)


# ---------------------------------------------------------------------------
# Fragility -- measured from the ideal state
# ---------------------------------------------------------------------------

def fragility(lc, eps=1e-6):
    """{qubit: f} where f = max(|rho01|, 1 - purity) of the qubit's reduced state
    in the NOISELESS circuit. 0 for a computational basis state, 0.5 for |+>,
    0.5 for half of a Bell pair.

    A qubit with f <= eps cannot be touched by pure dephasing, so its idle time
    is invisible to every machine.
    """
    from qiskit.quantum_info import partial_trace, DensityMatrix

    qs = list(range(lc.n_qubits))
    psi = ideal_state(lc, qs)
    rho_full = DensityMatrix(psi)
    out = {}
    for q in qs:
        r = partial_trace(rho_full, [w for w in qs if w != q]).data
        coh = abs(r[0, 1])
        mixed = 1.0 - float(np.real(np.trace(r @ r)))
        out[q] = float(max(coh, mixed))
    return out


# ---------------------------------------------------------------------------
# Profiling -- description and validation only, NEVER selection
# ---------------------------------------------------------------------------

def duty_threshold(tech_fast="sc", tech_slow="na"):
    """Duty cycle above which `tech_fast` is the better home, under the frozen table.

    Compare per qubit doing g 2Q gates and idling I ns:
        cost_fast = (1-f2q_fast)*g + I/T2_fast
        cost_slow = (1-f2q_slow)*g + I/T2_slow
    Break-even at I = thr*g; busy time is t2q_fast*g, so duty = t2q/(t2q+thr).

    Reported alongside results to interpret them. Never used to filter.
    """
    a, b = TECHS[tech_fast], TECHS[tech_slow]
    thr = ((1 - b.f2q) - (1 - a.f2q)) / (1 / a.T2 - 1 / b.T2)
    return a.t2q / (a.t2q + thr)


@dataclass
class CircuitProfile:
    """Descriptive statistics. Recorded for every instance, used for reporting and
    for structural validation -- never to accept or reject."""
    duty: dict                    # {qubit: busy_time / makespan}, all-SC schedule
    fragile: dict                 # {qubit: fragility score}
    activity: dict                # {qubit: layers in which it holds a gate}
    mean_duty: float
    duty_spread: float            # stdev of duty across qubits -- the variable that matters
    idle_fraction: float          # mean fraction of time idle (the "80% idle" statistic)
    n_above_threshold: int        # qubits preferring SC under the frozen table
    contested: bool               # more want SC than SC has capacity for
    n_2q: int
    n_cross_best: int             # cross-module 2Q gates under the best static 4/4 cut

    def summary(self):
        return (f"duty mean={self.mean_duty:.2f} spread={self.duty_spread:.2f} "
                f"idle={self.idle_fraction:.2f} n>thr={self.n_above_threshold}"
                f"{' CONTESTED' if self.contested else ''} "
                f"2q={self.n_2q} cut={self.n_cross_best}")


def profile(lc, cap=4, tech="sc"):
    """Measure a circuit's workload structure.

    Duty is computed ANALYTICALLY from the layer grid using `tech`'s gate times:

        busy[q]   = sum of gate durations involving q
        makespan  = sum over layers of the longest gate in that layer
        duty[q]   = busy[q] / makespan

    Deliberately routing-free. Measuring duty through `lower()` instead
    contaminates it with SWAP overhead, which depends on the module topology and
    on the assignment being profiled -- so there is no single "the duty" that way.
    Measured on a uniform-activity circuit whose gate counts were balanced to
    (5,5,6,6,5,6,6,5), routing pushed the duty spread to 0.17 and made two qubits
    with identical gate counts differ 0.92 vs 0.43. A workload descriptor must not
    move when you change the coupling map.

    This is the global-barrier clock, which is the right convention for a
    descriptor even though `lower()` correctly uses per-block ASAP for physics.
    """
    import itertools

    n = lc.n_qubits
    sp = TECHS[tech]
    busy = {q: 0.0 for q in range(n)}
    makespan = 0.0
    for lay in lc.layers:
        dur = 0.0
        for g in lay:
            d = sp.t2q if g[0] == '2q' else sp.t1q
            dur = max(dur, d)
            for q in (g[1:3] if g[0] == '2q' else [g[1]]):
                busy[q] += d
        makespan += dur
    makespan = makespan or 1.0

    duty = {q: busy[q] / makespan for q in range(n)}
    thr = duty_threshold()
    n_above = sum(1 for v in duty.values() if v > thr)

    pairs = lc.interaction_pairs()
    best = None
    for grp in itertools.combinations(range(n), cap):
        g = set(grp)
        cut = sum(c for (a, b), c in pairs.items() if (a in g) != (b in g))
        if best is None or cut < best:
            best = cut

    return CircuitProfile(
        duty=duty, fragile=fragility(lc), activity=lc.activity(),
        mean_duty=float(np.mean(list(duty.values()))),
        duty_spread=float(np.std(list(duty.values()))),
        idle_fraction=1.0 - float(np.mean(list(duty.values()))),
        n_above_threshold=n_above, contested=n_above > cap,
        n_2q=lc.n_2q, n_cross_best=best or 0)


# ---------------------------------------------------------------------------
# Structural validation -- "did we build the class we asked for?"
# ---------------------------------------------------------------------------

def validate_structure(lc, family, prof=None, meta=None, eps=1e-6, depth=DEPTH):
    """Returns (ok, reasons). Checks the requested CLASS was produced and that the
    instrument can register decoherence at all. Deliberately contains NO check
    that would favour any machine."""
    prof = prof or profile(lc)
    meta = meta or {}
    bad = []

    if lc.n_qubits != N_QUBITS:
        bad.append(f"n_qubits={lc.n_qubits}, expected {N_QUBITS}")
    if lc.depth != depth:
        bad.append(f"depth={lc.depth}, expected {depth}")
    for lay in lc.layers:
        for g in lay:
            if g[-1].name in ("swap", "reset", "measure"):
                bad.append(f"forbidden op '{g[-1].name}'")

    # Fragility: any qubit whose idle time is meant to matter must not sit in a
    # basis state. Applied to EVERY qubit -- an all-basis-state circuit is a
    # broken instrument for every machine equally.
    # Preparation covers 6-8 of 8 qubits, so up to 2 may legitimately stay in a
    # basis state. H is its own inverse and X/Z/RZ preserve basis states, so a
    # prepared qubit can also return to one. Recorded, not fatal: see
    # `n_dead_qubits` in the meta and the generator report.
    dead = [q for q, f in prof.fragile.items() if f <= eps]
    if len(dead) > 3:
        bad.append(f"{len(dead)}/{lc.n_qubits} qubits are in computational basis "
                   f"states {dead} -- dephasing cannot touch them, so their idle "
                   "time is invisible to every machine")

    # Class-defining checks only.
    if family == "hotcore":
        if prof.duty_spread < 0.03:
            bad.append(f"hotcore has no measurable activity imbalance "
                       f"(spread={prof.duty_spread:.3f})")
    elif family == "uniform":
        if prof.duty_spread > 0.15:
            bad.append(f"uniform is not balanced (spread={prof.duty_spread:.3f})")
    elif family == "phased":
        if meta.get("n_boundaries", 0) < 1:
            bad.append("phased instance has no role boundaries at all")
    if family == "hotcore" and meta.get("n_boundaries", 0) != 0:
        bad.append("hotcore must be the zero-boundary case")

    return (not bad), bad


# ---------------------------------------------------------------------------
# Generation -- one workload model, two families
# ---------------------------------------------------------------------------
# `hotcore` (M1) is the ZERO-BOUNDARY special case of `phased` (M2): identical
# gate model, identical activity probabilities, identical preparation. The only
# difference is whether a qubit's role changes during the circuit. That means
# both experiments interrogate one randomized workload class, rather than two
# circuit models whose differences a reviewer would have to be talked through.

@dataclass
class FamilySpec:
    name: str
    description: str
    hypothesis: str
    meta: dict = field(default_factory=dict)


FAMILIES = {
    "hotcore": FamilySpec(
        "hotcore",
        "Zero-boundary case: each qubit holds one role for the whole circuit. "
        "3-5 qubits are active at 70-95%, the rest idle at 1-20%.",
        "Static heterogeneous assignment can exploit complementary speed and "
        "coherence when workload roles are stable but unequal."),
    "phased": FamilySpec(
        "phased",
        "Each qubit independently draws 0-3 role boundaries (10/40/40/10) at "
        "STAGGERED positions, alternating hot/cold, minimum segment 4 layers.",
        "Roles change at different times for different qubits, so no static "
        "assignment is right throughout, and migrations can be batched or not."),
    "uniform": FamilySpec(
        "uniform",
        "Control: same mean activity as its hotcore partner, spread evenly.",
        "Less opportunity for heterogeneity; establishes workload dependence."),
}

# Boundary-count distribution. Most qubits change role once or twice; a few
# never do and a few do three times.
N_BOUNDARY_WEIGHTS = {0: 10, 1: 40, 2: 40, 3: 10}
MIN_SEGMENT = 4          # layers -- see below
P_1Q_RANGE = (0.30, 0.60)
HOT_COUNT_RANGE = (3, 5)  # per layer, vs SC capacity 4
N_PREP_RANGE = (6, 8)     # qubits receiving H at layer 0


def _rand_1q(rng):
    """One of H, X, Z, RZ(theta), uniformly.

    NOTE only H creates superposition. X maps |0> to |1>, Z is a no-op on |0>,
    and RZ adds a phase -- all still computational basis states, which pure
    dephasing cannot touch. H is also its own inverse, so a qubit prepared at
    layer 0 that later draws another H returns to a basis state and idles
    invisibly from then on. `profile().fragile` records the final-state
    fragility so this is measurable; see the generator report for the count.
    """
    return rng.choice([HGate(), XGate(), ZGate(), RZGate(rng.uniform(0.1, 2 * np.pi))])


def _segment_lengths(rng, depth, k, min_seg=MIN_SEGMENT):
    """k+1 segment lengths summing to `depth`, each at least `min_seg`.

    Below `min_seg` a migrated qubit cannot execute enough gates to repay the
    block-boundary cost (f_move per mover, plus t_move dephasing every
    synchronised qubit -- break-even is roughly two gates in the new
    residency), so shorter segments only add boundaries no scheduler should
    ever use. `k` is reduced if `depth` cannot accommodate it.
    """
    while k > 0 and (k + 1) * min_seg > depth:
        k -= 1
    slack = depth - (k + 1) * min_seg
    if k == 0:
        return [depth]
    # random composition of `slack` into k+1 non-negative parts (bars method)
    picks = sorted(rng.sample(range(slack + k), k))
    parts, prev = [], -1
    for pk in picks:
        parts.append(pk - prev - 1)
        prev = pk
    parts.append(slack + k - 1 - prev)
    return [min_seg + x for x in parts]


def _role_timeline(rng, n, depth, allow_boundaries=True, n_hot=None):
    """{qubit: [is_hot per layer]} plus the boundary positions.

    Boundaries are drawn INDEPENDENTLY per qubit, so role changes are
    staggered rather than aligned. That is the point: because a block boundary
    charges `max` over its movers rather than the sum, a scheduler can batch
    two nearby transitions into one boundary or pay for two. Globally aligned
    phases would hide that decision entirely.

    Roles ALTERNATE at every boundary. Drawing each phase's role independently
    would make about half of all boundaries no-ops (hot -> hot), wasting the
    role changes the family exists to create.
    """
    if not allow_boundaries:
        # Zero-boundary case: draw the hot SET once, sized 3-5 (spanning SC
        # capacity 4), and hold it for every layer. Drawing each qubit's role
        # independently at 1/2 and then repairing per layer would break the
        # defining property of this family -- roles would no longer be constant.
        n_hot = n_hot if n_hot is not None else rng.choice([3, 4, 5])
        hot_set = set(rng.sample(range(n), n_hot))
        return ({q: [q in hot_set] * depth for q in range(n)},
                {q: [] for q in range(n)})

    roles, bounds = {}, {}
    for q in range(n):
        k = rng.choices(list(N_BOUNDARY_WEIGHTS), weights=list(N_BOUNDARY_WEIGHTS.values()))[0]
        segs = _segment_lengths(rng, depth, k)
        hot = rng.random() < 0.5
        tl, pos, at = [], [], 0
        for i, L in enumerate(segs):
            tl += [hot] * L
            at += L
            if i < len(segs) - 1:
                pos.append(at)
            hot = not hot
        roles[q] = tl[:depth]
        bounds[q] = pos
    return roles, bounds


def _repair_hot_count(rng, roles, depth, n, lo=HOT_COUNT_RANGE[0], hi=HOT_COUNT_RANGE[1]):
    """Keep the number of hot qubits per layer inside [lo, hi].

    Independent per-qubit sampling gives Binomial(n, 1/2) hot counts, so ~29%
    of layers land outside [3, 5] at n=8. Those layers are degenerate rather
    than difficult: 1 hot qubit leaves the fast module nearly empty, 7 is
    hopeless for any assignment. Repair flips ONE qubit's role at ONE layer,
    which perturbs that layer's gate probability without creating a scheduling
    boundary -- the role timeline sets activity, the scheduler picks its own
    boundaries. The count is returned so the perturbation stays visible.
    """
    n_fix = 0
    for li in range(depth):
        while True:
            hot = [q for q in range(n) if roles[q][li]]
            if lo <= len(hot) <= hi:
                break
            if len(hot) < lo:
                cand = [q for q in range(n) if not roles[q][li]]
                roles[rng.choice(cand)][li] = True
            else:
                roles[rng.choice(hot)][li] = False
            n_fix += 1
    return n_fix


def _prep_layer(rng, n):
    """Layer 0: H on `N_PREP_RANGE` qubits, chosen uniformly.

    Only qubits in a non-basis state can register dephasing at all, so this
    layer is what makes the experiment able to measure anything. Preparing
    6-8 of 8 leaves at most two qubits able to idle invisibly.
    """
    n_h = rng.randint(*N_PREP_RANGE)
    qs = rng.sample(range(n), n_h)
    return [('1q', q, HGate()) for q in sorted(qs)], n_h


def _n_pairs_for(rng, n_active, f, tot1=0, tot2=0):
    """How many CX pairs to form among `n_active` qubits to hit a 1Q GATE
    fraction of `f`.

    Sampling each qubit independently at `f` does not work: a 2Q gate consumes
    two qubits but counts as one gate, and any unpaired leftover falls back to
    1Q. Both effects push the realised fraction up -- measured 0.79 and then
    0.66 against targets of 0.43 and 0.44.

    Solve it exactly instead. With n2 pairs, n1 = A - 2*n2 single-qubit gates:

        n1 / (n1 + n2) = f   =>   n2 = A(1 - f) / (2 - f)

    Small active sets can only realise a coarse set of fractions (A=4 admits
    only 1.0, 0.67, 0.0), so the fractional part is resolved by randomized
    rounding, making the fraction correct IN EXPECTATION across layers.
    """
    if n_active < 2:
        return 0
    # Steer the RUNNING fraction, not this layer's in isolation. Layers with a
    # single active qubit cannot form a pair and are forced to 1Q, which biases
    # the circuit-level fraction upward (measured 0.53 realised against 0.46).
    # Choosing k against the cumulative totals compensates in later layers.
    best, best_err = 0, None
    for k in range(0, n_active // 2 + 1):
        n1, n2 = tot1 + n_active - 2 * k, tot2 + k
        frac = n1 / (n1 + n2) if (n1 + n2) else f
        err = abs(frac - f)
        if best_err is None or err < best_err:
            best, best_err = k, err
    # break ties randomly so the choice is not systematically biased
    ties = [k for k in range(0, n_active // 2 + 1)
            if abs((lambda a, b: a / (a + b) if (a + b) else f)(
                tot1 + n_active - 2 * k, tot2 + k) - f) <= best_err + 1e-12]
    return rng.choice(ties) if ties else best


def _emit_layer(rng, hot_at_layer, p_hot, p_cold, p_1q, n, running=None):
    """One layer: draw the active set from each qubit's role probability, form
    the number of CX pairs that steers the running 1Q gate fraction toward the
    target, and give the remaining active qubits a single-qubit gate."""
    running = running if running is not None else [0, 0]
    active = [q for q in range(n)
              if rng.random() < (p_hot if hot_at_layer[q] else p_cold)]
    rng.shuffle(active)
    n2 = _n_pairs_for(rng, len(active), p_1q, running[0], running[1])
    running[0] += len(active) - 2 * n2
    running[1] += n2
    lay = []
    for i in range(n2):
        a, b = active.pop(), active.pop()
        lay.append(('2q', a, b, CXGate()))
    for q in active:
        lay.append(('1q', q, _rand_1q(rng)))
    return lay


def _build(seed, depth, n, allow_boundaries, family, salt=0, n_hot=None):
    rng = random.Random(seed + salt)
    p_hot = rng.uniform(0.70, 0.95)
    p_cold = rng.uniform(*P_COLD)
    p_1q = rng.uniform(*P_1Q_RANGE)

    body = depth - 1                       # layer 0 is preparation
    roles, bounds = _role_timeline(rng, n, body, allow_boundaries, n_hot)
    n_fix = _repair_hot_count(rng, roles, body, n) if allow_boundaries else 0

    prep, n_h = _prep_layer(rng, n)
    spec, running = [prep], [0, 0]        # prep H gates excluded from the ratio
    for li in range(body):
        spec.append(_emit_layer(rng, {q: roles[q][li] for q in range(n)},
                                p_hot, p_cold, p_1q, n, running))
    lc = make_layers(spec, n_qubits=n, name=f"{family}_s{seed}")

    hot_counts = [sum(roles[q][li] for q in range(n)) for li in range(body)]
    all_bounds = sorted(b for q in bounds for b in bounds[q])
    gaps = [b - a for a, b in zip(all_bounds, all_bounds[1:])]
    meta = dict(
        family=family, seed=seed, depth=depth,
        p_hot=round(p_hot, 3), p_cold=round(p_cold, 3), p_1q=round(p_1q, 3),
        realised_p_1q=round((lc.n_1q - n_h) / max(1, lc.n_1q - n_h + lc.n_2q), 3),
        n_prep_h=n_h, n_hot_repairs=n_fix,
        n_hot=int(round(float(np.mean(hot_counts)))),
        n_hot_min=min(hot_counts), n_hot_max=max(hot_counts),
        n_boundaries=len(all_bounds),
        boundaries_per_qubit={q: bounds[q] for q in range(n) if bounds[q]},
        boundary_layers=all_bounds,
        max_stagger=max(gaps) if gaps else 0,
        n_movers_max=max([hot_counts[i] for i in range(len(hot_counts))]) if hot_counts else 0,
    )
    return lc, meta


def _generate_hotcore(seed, n=N_QUBITS, depth=DEPTH, n_hot=None):
    """Zero-boundary case: every qubit keeps one role throughout.

    `n_hot=None` draws uniformly from {3, 4, 5}. Pass an explicit value (or
    `n_hot='stratified'`) to balance the three capacity regimes exactly, which
    is what the M1 report groups on -- random draws left 19/12/9 over 40 seeds.
    """
    if n_hot == "stratified":
        n_hot = [3, 4, 5][seed % 3]
    return _build(seed, depth, n, allow_boundaries=False, family="hotcore",
                  n_hot=n_hot)


def _generate_phased(seed, n=N_QUBITS, depth=DEPTH):
    """Staggered per-qubit role changes."""
    return _build(seed, depth, n, allow_boundaries=True, family="phased", salt=200_000)


def _build_uniform(seed, k_float, n, depth):
    """Uniform control: activate the least-loaded qubits each layer, paired.

    `k` is forced EVEN so every active qubit is paired for a 2Q gate. Balancing
    gate COUNTS is not enough -- a 2Q gate occupies 200 ns and a 1Q gate 20 ns,
    so an odd-one-out singleton accrues far less busy time. Measured with odd k
    allowed, two qubits with identical gate counts differed 0.92 vs 0.43 in duty.
    """
    rng = random.Random(seed + 100_000)
    prep, _ = _prep_layer(rng, n)
    spec, load, acc = [prep], {q: 0.0 for q in range(n)}, 0.0
    for _ in range(depth - 1):
        acc += k_float
        k = int(round(acc))
        acc -= k
        k = max(0, min(k - (k % 2), n))
        order = sorted(range(n), key=lambda q: (load[q], rng.random()))
        act = order[:k]
        rng.shuffle(act)
        lay = []
        while len(act) >= 2:
            a, b = act.pop(), act.pop()
            lay.append(('2q', a, b, CXGate()))
            load[a] += 1.0
            load[b] += 1.0
        spec.append(lay)
    return make_layers(spec, n_qubits=n, name=f"uniform_s{seed}")


def _generate_uniform(seed, target_mean_activity=None, n=N_QUBITS, depth=DEPTH):
    """Matched to its hotcore partner's REALISED mean duty by bisection -- duty is
    not linear in "qubits active per layer" (gate durations differ and the
    makespan moves), so the target is hit by measurement, not calculation."""
    if target_mean_activity is None:
        rng = random.Random(seed + 100_000)
        return _build_uniform(seed, rng.uniform(3.0, 5.0), n, depth), \
            dict(family="uniform", seed=seed, depth=depth, matched_to=None)

    lo, hi, best = 0.5, float(n), None
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        lc = _build_uniform(seed, mid, n, depth)
        got = profile(lc).mean_duty
        if best is None or abs(got - target_mean_activity) < abs(best[1] - target_mean_activity):
            best = (lc, got, mid)
        if abs(got - target_mean_activity) < 0.005:
            break
        lo, hi = (mid, hi) if got < target_mean_activity else (lo, mid)
    lc, got, k = best
    return lc, dict(family="uniform", seed=seed, depth=depth, k_per_layer=round(k, 2),
                    matched_to=round(target_mean_activity, 4),
                    realised_duty=round(got, 4))


def generate(family, seed, validate=True, **kw):
    """One instance -> (LayeredCircuit, meta, CircuitProfile).

    `validate` runs STRUCTURAL checks only -- "did we build the class we asked
    for?" It never rejects a circuit for being unfavourable to HQC.
    """
    if family == "hotcore":
        lc, meta = _generate_hotcore(seed, **kw)
    elif family == "phased":
        lc, meta = _generate_phased(seed, **kw)
    elif family == "uniform":
        lc, meta = _generate_uniform(seed, **kw)
    else:
        raise ValueError(f"unknown family {family!r}; choose from {sorted(FAMILIES)}")

    prof = profile(lc)
    if validate:
        ok, reasons = validate_structure(lc, family, prof, meta,
                                         depth=kw.get('depth', DEPTH))
        if not ok:
            raise ValueError(f"{family} seed {seed} failed structural validation: "
                             + "; ".join(reasons))
    meta["profile"] = prof
    return lc, meta, prof


def generate_pair(seed, depth=DEPTH):
    """A hotcore instance and its matched uniform control -- same realised mean
    duty, different activity VARIANCE, so the pair can be reported paired."""
    hot_lc, hot_meta, hot_prof = generate("hotcore", seed, depth=depth)
    uni_lc, uni_meta, uni_prof = generate(
        "uniform", seed, target_mean_activity=hot_prof.mean_duty, depth=depth)
    return (hot_lc, hot_meta, hot_prof), (uni_lc, uni_meta, uni_prof)


def generate_family(family, n=30, seed0=0, depth=DEPTH):
    """Deterministic batch. Seeds are seed0 .. seed0+n-1."""
    return [generate(family, seed0 + i, depth=depth) for i in range(n)]
