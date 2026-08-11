"""Atomic noise channels  (extracted verbatim from NB1).

Two channels, one source of truth. No other module in this package defines a
noise channel.

Conventions locked in NB1 and preserved here
--------------------------------------------
Idle / decoherence
    ``thermal_relaxation_error(t1=inf, t2=T2, time=t)`` -- pure dephasing, no
    amplitude damping (the configs carry no T1 field). Off-diagonal coherence
    decays by exp(-t/T2), which matches EFCL's idle survival p_idle =
    exp(-dt/T) exactly, with no free parameter.

Gate fidelity -> channel
    Qiskit's ``depolarizing_error(lam, n)`` is E(rho) = (1-lam) rho + lam I/d
    with d = 2**n, giving avg_gate_fid = 1 - lam*(d-1)/d. Hardware reports
    f1q/f2q as *average gate fidelities*, so we set

        lam = (1 - F) * d / (d - 1)

    which makes both the average gate fidelity and the pure-state fidelity of
    every gate equal the config F exactly, for 1Q and 2Q alike. Gates compose
    nearly multiplicatively (two F=0.99 gates give 0.9802 vs F^2 = 0.9801), so
    the exec term tracks EFCL's product-of-F to first order and the channel
    convention is NOT an EFCL<->Aer divergence source.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import thermal_relaxation_error, depolarizing_error
from qiskit.quantum_info import DensityMatrix

__all__ = [
    "dephasing_channel",
    "gate_infidelity_channel",
    "plus_coherence",
    "idle_coherence_on_aer",
]


def dephasing_channel(T2, t):
    """Pure dephasing (T1 -> inf) for an idle qubit of duration `t` ns at coherence `T2` ns.

    Returns a 1-qubit QuantumError, or ``None`` when ``t <= 0`` so call sites can
    skip appending a no-op instruction. Append via ``.to_instruction()``.

    NOTE the ``t <= 0`` -> None behaviour is NB4's variant and is the one the
    lowering pass depends on. NB1's standalone version raised on t < 0; the
    guard is kept below as an explicit error for genuinely negative durations,
    which always indicate a clock bug rather than a zero-length idle.
    """
    if t < 0:
        raise ValueError(f"idle duration must be >= 0, got {t}")
    if t == 0:
        return None
    return thermal_relaxation_error(t1=np.inf, t2=float(T2), time=float(t))


def gate_infidelity_channel(F, num_qubits):
    """Depolarizing channel with average gate fidelity == F (hardware convention).

    lam = (1 - F) * d / (d - 1),  d = 2**num_qubits.

    Used for 1Q/2Q gate errors and, in `lowering`, for the aggregate inter-QPU
    comm (`f_comm`) and move (`f_move`) primitives.
    """
    if not (0.0 <= F <= 1.0):
        raise ValueError(f"F must be in [0, 1], got {F}")
    d = 2 ** num_qubits
    lam = (1.0 - F) * d / (d - 1.0)
    return depolarizing_error(lam, num_qubits)


# --------------------------------------------------------------------------
# Introspection helpers (used by the certificate; not part of the physics)
# --------------------------------------------------------------------------

def plus_coherence(channel):
    """Apply a 1-qubit channel to |+> and return |rho01(t)| / |rho01(0)|."""
    rho = DensityMatrix.from_label("+").evolve(channel.to_quantumchannel())
    return abs(rho.data[0, 1]) / 0.5


def idle_coherence_on_aer(T2, t_idle):
    """Same quantity, but measured through the real AerSimulator density-matrix
    backend rather than `quantum_info` evolution -- i.e. the exact path the
    lowering pass and scorer use."""
    qc = QuantumCircuit(1)
    qc.h(0)
    ch = dephasing_channel(T2, t_idle)
    if ch is not None:
        qc.append(ch.to_instruction(), [0])
    qc.save_density_matrix()
    res = AerSimulator(method="density_matrix").run(qc).result()
    rho = DensityMatrix(res.data(0)["density_matrix"])
    return abs(rho.data[0, 1]) / 0.5
