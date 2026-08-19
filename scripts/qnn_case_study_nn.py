#!/usr/bin/env python3
"""
qnn_case_study.py — Load QNN 10-qubit from MQT Bench, draw circuit diagram,
and overlay a custom SC/TI schedule heatmap for 15 layers.
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit


# ── Circuit preprocessing (from inspect_mqt_circuits_nnnn.py) ─────────────

def _strip_classical_registers(qc):
    new_qc = QuantumCircuit(*qc.qregs)
    for instruction in qc.data:
        if len(instruction.clbits) == 0:
            new_qc.append(instruction)
    return new_qc

def _has_multiqubit_gates(qc):
    for instruction in qc.data:
        if len(instruction.qubits) >= 3:
            return True
    return False

def _decompose_to_1q2q(qc, max_rounds=15):
    qc_out = qc
    for _ in range(max_rounds):
        if not _has_multiqubit_gates(qc_out):
            break
        dag = circuit_to_dag(qc_out)
        did_something = False
        for node in dag.op_nodes():
            if len(node.qargs) < 3:
                continue
            defn = node.op.definition
            if defn is None:
                continue
            sub_dag = circuit_to_dag(defn)
            dag.substitute_node_with_dag(node, sub_dag)
            did_something = True
        qc_out = dag_to_circuit(dag)
        if not did_something:
            break
    return qc_out


# ── Layer extraction ──────────────────────────────────────────────────────

def extract_layers(qc):
    dag = circuit_to_dag(qc)
    qubit_index = {qb: i for i, qb in enumerate(qc.qubits)}
    layers = []
    for dag_layer in dag.layers():
        subdag = dag_layer["graph"]
        layer = {"gates_1q": [], "gates_2q": [], "active_qubits": set(), "twoq_qubits": set()}
        for node in subdag.op_nodes():
            name = node.op.name
            if name in ("barrier", "measure", "reset"):
                continue
            qargs = tuple(qubit_index[qb] for qb in node.qargs)
            layer["active_qubits"].update(qargs)
            if len(qargs) == 1:
                layer["gates_1q"].append((name, qargs))
            elif len(qargs) == 2:
                layer["gates_2q"].append((name, qargs))
                layer["twoq_qubits"].update(qargs)
        if layer["gates_1q"] or layer["gates_2q"]:
            layers.append(layer)
    return layers


# ── Technology colors ─────────────────────────────────────────────────────

TECH_COLORS = [
    "#66B2FF",  # SC — blue
    "#F0CBAF",  # TI — peach
]
TECH_NAMES = ["SC", "TI"]
K = 2


# ── Custom schedule definition ────────────────────────────────────────────
# For each layer, specify which qubits belong to SC (tech 0). Rest → TI (tech 1).

N = 10
T_SCHED = 15

# SC qubit ranges per layer (inclusive)
SC_RANGES = [
    (3, 9),   # layer 0
    (3, 9),   # layer 1
    (3, 9),   # layer 2
    (3, 9),   # layer 3
    (3, 9),   # layer 4
    (8, 9),   # layer 5
    (7, 9),   # layer 6
    (6, 8),   # layer 7
    (5, 7),   # layer 8
    (4, 6),   # layer 9
    (3, 5),   # layer 10
    (2, 4),   # layer 11
    (1, 3),   # layer 12
    (0, 2),   # layer 13
    (0, 1),   # layer 14
]

def build_schedule():
    """Returns [T_SCHED, N] int array: 0=SC, 1=TI."""
    mat = np.ones((T_SCHED, N), dtype=int)  # default TI
    for t, (lo, hi) in enumerate(SC_RANGES):
        for q in range(lo, hi + 1):
            mat[t, q] = 0  # SC
    return mat


# ── Schedule heatmap ──────────────────────────────────────────────────────

def plot_schedule_heatmap(layers, schedule_mat):
    nq = N
    T_show = min(len(layers), T_SCHED)

    # Precompute 2Q edges for overlay
    twoq_edges_per_layer = []
    for t in range(T_show):
        edges = []
        for _, (q0, q1) in layers[t]["gates_2q"]:
            edges.append((q0, q1))
        twoq_edges_per_layer.append(edges)

    fig_w = max(7.0, T_show * 0.50 + 2.0)
    fig_h = nq * 0.32 + 1.4

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    fig.suptitle(
        f"QNN  |  N={nq}  |  Technology Assignment (SC + TI)  "
        f"(first {T_show} layers)",
        fontsize=10, fontweight="bold")

    cmap = ListedColormap(TECH_COLORS[:K])
    bounds = np.arange(-0.5, K + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    ax.imshow(schedule_mat[:T_show].T, aspect="auto", origin="upper",
              cmap=cmap, norm=norm, interpolation="nearest")

    # Overlay 2Q edges
    for t, edges in enumerate(twoq_edges_per_layer):
        for q0, q1 in edges:
            ax.plot([t, t], [q0, q1], color="black", linewidth=0.8, alpha=0.4)

    ax.set_xlabel("Layer", fontsize=9)
    ax.set_ylabel("Qubit", fontsize=9)
    ax.set_yticks(range(nq))
    ax.set_yticklabels([f"Q{q}" for q in range(nq)], fontsize=6)
    ax.set_xticks(range(T_show))
    ax.set_xticklabels([str(t) for t in range(T_show)], fontsize=6)

    legend_patches = [
        mpatches.Patch(color=TECH_COLORS[0], label="SC"),
        mpatches.Patch(color=TECH_COLORS[1], label="TI"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=7, framealpha=0.8)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    from mqt.bench import BenchmarkLevel, get_benchmark

    print("Loading QNN N=10 from MQT Bench...")
    qc_raw = get_benchmark(benchmark="qnn", level=BenchmarkLevel.ALG, circuit_size=10)
    qc_decomposed = _decompose_to_1q2q(qc_raw)
    qc_decomposed.remove_final_measurements(inplace=True)
    qc_clean = _strip_classical_registers(qc_decomposed)

    print(f"  Raw gates: {qc_raw.size()}, Clean gates: {qc_clean.size()}, "
          f"Qubits: {qc_clean.num_qubits}")

    # 1) Draw Qiskit circuit (no barriers, display only)
    fig_circ = qc_clean.draw(
        output="mpl", fold=-1, idle_wires=True,
        style={"backgroundcolor": "#FFFFFF"})
    fig_circ.suptitle(
        f"QNN  |  N={qc_clean.num_qubits}  gates={qc_clean.size()}",
        fontsize=10, fontweight="bold", y=1.02)
    fig_circ.tight_layout()
    plt.show()
    plt.close(fig_circ)

    # 2) Draw schedule heatmap
    layers = extract_layers(qc_clean)
    print(f"  Layers: {len(layers)}")

    schedule_mat = build_schedule()
    print(f"  Schedule: {T_SCHED} layers, {N} qubits, K={K} (SC+TI)")

    plot_schedule_heatmap(layers, schedule_mat)


if __name__ == "__main__":
    main()
