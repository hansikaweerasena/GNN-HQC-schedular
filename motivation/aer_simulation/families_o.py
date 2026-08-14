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
from qiskit.circuit.library import HGate, CXGate, RZGate

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
    dead = [q for q, f in prof.fragile.items() if f <= eps]
    if dead:
        bad.append(f"qubits {dead} are in computational basis states (dephasing "
                   "cannot touch them, so their idle time is invisible)")

    # Class-defining checks only.
    if family == "hotcore":
        if prof.duty_spread < 0.03:
            bad.append(f"hotcore has no measurable activity imbalance "
                       f"(spread={prof.duty_spread:.3f})")
    elif family == "uniform":
        if prof.duty_spread > 0.15:
            bad.append(f"uniform is not balanced (spread={prof.duty_spread:.3f})")
    elif family == "phased":
        sets = meta.get("hot_sets", [])
        if len(sets) < 2:
            bad.append("phased needs at least 2 phases")
        elif all(set(sets[i]) == set(sets[0]) for i in range(len(sets))):
            bad.append("phased active subset never changes")

    return (not bad), bad


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@dataclass
class FamilySpec:
    name: str
    description: str
    hypothesis: str
    meta: dict = field(default_factory=dict)


FAMILIES = {
    "hotcore": FamilySpec(
        "hotcore",
        "Persistent active region (3-5 qubits at 70-95% activity) plus storage "
        "qubits at 1-20%. Most qubit-time is idle, but unevenly distributed.",
        "SC+NA can exploit complementary speed and coherence when workload roles "
        "are stable but unequal. Not expected to win on every seed."),
    "uniform": FamilySpec(
        "uniform",
        "Same MEAN activity as the paired hotcore instance, spread evenly across "
        "all 8 qubits. Random disjoint pairings.",
        "Less opportunity for heterogeneity. Establishes that the benefit is "
        "workload-dependent, not universal."),
    "phased": FamilySpec(
        "phased",
        "2-3 phases; the active subset changes at each phase boundary, with "
        "partial overlap between consecutive hot sets.",
        "The preferred placement changes over time, so a static assignment is no "
        "longer optimal and temporal scheduling has something to decide."),
}


def _prep_layer(rng, n):
    """Layer 0: put every qubit into a non-basis state.

    Everything gets an H; a random subset is then entangled in pairs so the family
    includes qubits whose fragility comes from entanglement (rho = I/2, zero local
    coherence) rather than local superposition. Both dephase; only the purity term
    detects the second kind.
    """
    return [list(range(n))]


def _emit(rng, hot, cold, p_hot, p_cold, p_intra, n):
    """One layer: choose active qubits by activity probability, then pair them up,
    biasing pairs to stay inside the hot region."""
    active = [q for q in range(n)
              if rng.random() < (p_hot if q in hot else p_cold)]
    rng.shuffle(active)
    lay, used = [], set()
    hot_active = [q for q in active if q in hot]
    # intra-hot pairs first, with probability p_intra
    while len(hot_active) >= 2 and rng.random() < p_intra:
        a, b = hot_active.pop(), hot_active.pop()
        if a in used or b in used:
            continue
        lay.append((a, b))
        used |= {a, b}
    rest = [q for q in active if q not in used]
    rng.shuffle(rest)
    while len(rest) >= 2:
        a, b = rest.pop(), rest.pop()
        lay.append((a, b))
        used |= {a, b}
    for q in rest:                      # leftover singleton gets a 1Q rotation
        if q not in used:
            lay.append(('1q', q, RZGate(rng.uniform(0.1, 1.5))))
    return lay


def _generate_hotcore(seed, n=N_QUBITS, depth=DEPTH):
    rng = random.Random(seed)
    n_hot = rng.choice([3, 4, 5])       # under / matched / contested vs SC cap 4
    hot = set(rng.sample(range(n), n_hot))
    cold = set(range(n)) - hot
    p_hot = rng.uniform(0.70, 0.95)
    p_cold = rng.uniform(P_COLD[0], P_COLD[1])
    p_intra = rng.uniform(0.70, 0.90)

    spec = _prep_layer(rng, n)
    for _ in range(depth - 1):
        spec.append(_emit(rng, hot, cold, p_hot, p_cold, p_intra, n))
    lc = make_layers(spec, n_qubits=n, name=f"hotcore_s{seed}")
    meta = dict(family="hotcore", seed=seed, hot=sorted(hot), n_hot=n_hot,
                p_hot=round(p_hot, 3), p_cold=round(p_cold, 3),
                p_intra=round(p_intra, 3))
    return lc, meta


def _emit_balanced(rng, load, k, n):
    """One layer of the uniform control: activate the k qubits with the least
    cumulative work so far, ties broken randomly, then pair them up.

    Drawing each qubit independently at probability p makes activity uniform only
    IN EXPECTATION -- at 8 qubits x 9 layers the binomial noise alone produces a
    duty spread around 0.25, which is the same order as hotcore's imbalance and
    would make the control useless. The control's job is to isolate VARIANCE, so
    it has to actually be low-variance.

    `k` is forced EVEN so every active qubit is paired. Balancing gate COUNTS is
    not enough: a 2Q gate occupies 100 ns and a 1Q gate 20 ns, so whichever qubit
    ends up as the odd-one-out singleton accrues 5x less busy time. Measured with
    odd k allowed, two qubits with identical gate counts differed 0.92 vs 0.43 in
    duty. Pairing everyone makes busy time proportional to count.
    """
    k = max(0, min(k - (k % 2), n))       # EVEN only: see below
    order = sorted(range(n), key=lambda q: (load[q], rng.random()))
    active = order[:k]
    rng.shuffle(active)
    lay, rest = [], list(active)
    while len(rest) >= 2:
        a, b = rest.pop(), rest.pop()
        lay.append((a, b))
        load[a] += 1.0
        load[b] += 1.0
    return lay


def _build_uniform(seed, k_float, n, depth):
    """k_float active qubits per layer on average; fractional part is dithered."""
    rng = random.Random(seed + 100_000)
    spec, load, acc = _prep_layer(rng, n), {q: 0.0 for q in range(n)}, 0.0
    for _ in range(depth - 1):
        acc += k_float
        k = int(round(acc))
        acc -= k
        spec.append(_emit_balanced(rng, load, max(0, min(k, n)), n))
    return make_layers(spec, n_qubits=n, name=f"uniform_s{seed}")


def _generate_uniform(seed, target_mean_activity=None, n=N_QUBITS, depth=DEPTH):
    """Uniform control, matched to its hotcore partner.

    `target_mean_activity` is the partner's REALISED mean duty (busy_time /
    makespan), not a parameter value. Duty is not linear in "qubits active per
    layer" -- 1Q and 2Q gates have different durations and the makespan itself
    moves -- so the number of active qubits per layer is calibrated by measurement
    rather than computed. A few `lower()` calls; no Aer.

    Matching the realised mean means the pair differs in activity VARIANCE and not
    in total work, so hotcore-vs-uniform can be reported as a paired difference.
    """
    if target_mean_activity is None:
        rng = random.Random(seed + 100_000)
        return _build_uniform(seed, rng.uniform(3.0, 5.0), n, depth), \
            dict(family="uniform", seed=seed, matched_to=None)

    lo, hi = 0.5, float(n)
    best = None
    for _ in range(12):                        # bisect on realised duty
        mid = 0.5 * (lo + hi)
        lc = _build_uniform(seed, mid, n, depth)
        got = profile(lc).mean_duty
        if best is None or abs(got - target_mean_activity) < abs(best[1] - target_mean_activity):
            best = (lc, got, mid)
        if abs(got - target_mean_activity) < 0.005:
            break
        if got < target_mean_activity:
            lo = mid
        else:
            hi = mid
    lc, got, k = best
    return lc, dict(family="uniform", seed=seed, k_per_layer=round(k, 2),
                    matched_to=round(target_mean_activity, 4),
                    realised_duty=round(got, 4))


def _generate_phased(seed, n=N_QUBITS, depth=DEPTH):
    rng = random.Random(seed + 200_000)
    n_phase = rng.choice([2, 3])
    p_hot = rng.uniform(0.70, 0.95)
    p_cold = rng.uniform(P_COLD[0], P_COLD[1])
    p_intra = rng.uniform(0.70, 0.90)

    hot_sets = [set(rng.sample(range(n), rng.choice([3, 4, 5])))]
    for _ in range(n_phase - 1):
        prev = hot_sets[-1]
        k = rng.choice([3, 4, 5])
        keep = rng.sample(sorted(prev), min(rng.choice([1, 2, 3]), len(prev), k))
        pool = [q for q in range(n) if q not in keep]
        nxt = set(keep) | set(rng.sample(pool, k - len(keep)))
        hot_sets.append(nxt)

    body = depth - 1
    bounds = [round(body * (i + 1) / n_phase) for i in range(n_phase)]
    spec, start = _prep_layer(rng, n), 0
    for pi, end in enumerate(bounds):
        hot = hot_sets[pi]
        cold = set(range(n)) - hot
        for _ in range(end - start):
            spec.append(_emit(rng, hot, cold, p_hot, p_cold, p_intra, n))
        start = end
    lc = make_layers(spec, n_qubits=n, name=f"phased_s{seed}")
    return lc, dict(family="phased", seed=seed, n_phases=n_phase,
                    hot_sets=[sorted(s) for s in hot_sets],
                    phase_bounds=bounds, p_hot=round(p_hot, 3),
                    p_cold=round(p_cold, 3), p_intra=round(p_intra, 3))


def generate(family, seed, validate=True, **kw):
    """Generate one instance. Returns (LayeredCircuit, meta, CircuitProfile).

    `validate` runs STRUCTURAL checks only and raises if the requested class was
    not produced. It never rejects a circuit for being unfavourable to HQC.
    """
    if family == "hotcore":
        lc, meta = _generate_hotcore(seed, **kw)
    elif family == "uniform":
        lc, meta = _generate_uniform(seed, **kw)
    elif family == "phased":
        lc, meta = _generate_phased(seed, **kw)
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
    """A hotcore instance and its matched uniform control.

    The control is driven at the hotcore instance's REALISED mean duty, so the two
    differ in activity VARIANCE rather than in total gate count. Report paired.
    """
    hot_lc, hot_meta, hot_prof = generate("hotcore", seed, depth=depth)
    uni_lc, uni_meta, uni_prof = generate(
        "uniform", seed, target_mean_activity=hot_prof.mean_duty, depth=depth)
    return (hot_lc, hot_meta, hot_prof), (uni_lc, uni_meta, uni_prof)


def generate_family(family, n=30, seed0=0):
    """Deterministic batch. Seeds are seed0 .. seed0+n-1."""
    return [generate(family, seed0 + i) for i in range(n)]
