#!/usr/bin/env python3
"""
inspect_mqt_circuits.py — Browse MQT Bench circuit structure for case study selection.

Loads MQT Bench circuits with the same preprocessing pipeline as eval_scheduler_v2.py
(transpile → remove measurements → strip classical regs → validate), then displays
a 2Q activity heatmap for each circuit.

Heatmap: rows = layers (time), cols = qubits.
  - Dark cell  = qubit participates in a 2Q gate in that layer
  - Light cell = qubit is idle (or 1Q-only)

Usage:
    python inspect_mqt_circuits.py --qubit_min 8 --qubit_max 12
    python inspect_mqt_circuits.py --qubit_min 8 --qubit_max 12 --algorithms bv,wstate
    python inspect_mqt_circuits.py --qubit_min 28 --qubit_max 32 --max_layers 30
    python inspect_mqt_circuits.py --qubit_min 8 --qubit_max 12 --save_dir figs/
"""

import argparse
import sys
from datetime import datetime
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag


# ── MQT algorithm registry (same as eval_scheduler_v2.py) ──────────────────

MQT_ALGORITHMS = [
    "qaoa",
    "qft",
    "qftentangled",
    "vqe_real_amp",
    "vqe_su2",
    "vqe_two_local",
    "wstate",
    "qgan",
    "qnn",
    "portfolioqaoa",
    "portfoliovqe",
    "bv",
    "randomcircuit",
]

MQT_BASIS_GATES = ["cx", "rz", "h", "x", "sx"]


# ── Logging ─────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Circuit loading (identical pipeline to eval_scheduler_v2.py) ────────────

def _strip_classical_registers(qc: QuantumCircuit) -> QuantumCircuit:
    new_qc = QuantumCircuit(*qc.qregs)
    for instruction in qc.data:
        if len(instruction.clbits) == 0:
            new_qc.append(instruction)
    return new_qc


def _has_multiqubit_gates(qc: QuantumCircuit) -> bool:
    for instruction in qc.data:
        if len(instruction.qubits) >= 3:
            return True
    return False


def load_mqt_circuits(
    algorithms: List[str],
    qubit_min: int,
    qubit_max: int,
) -> List[Tuple[str, int, QuantumCircuit, QuantumCircuit]]:
    """
    Returns list of (algo_name, num_qubits, qc_clean, qc_raw) tuples.
    qc_raw  = original ALG-level circuit (with measurements, native gates)
    qc_clean = preprocessed circuit (transpiled, measurements removed)
    """
    try:
        from mqt.bench import BenchmarkLevel, get_benchmark
    except ImportError:
        raise ImportError("mqt.bench not installed. Install: pip install mqt.bench")

    circuits = []
    skipped = []

    for algo in algorithms:
        for nq in range(qubit_min, qubit_max + 1):
            try:
                qc_raw = get_benchmark(
                    benchmark=algo,
                    level=BenchmarkLevel.ALG,
                    circuit_size=nq,
                )
                qc_basis = transpile(
                    qc_raw,
                    basis_gates=MQT_BASIS_GATES,
                    optimization_level=1,
                )
                qc_basis.remove_final_measurements(inplace=True)
                qc_clean = _strip_classical_registers(qc_basis)

                if qc_clean.num_clbits != 0:
                    skipped.append((algo, nq, "classical bits remain"))
                    continue
                if qc_clean.num_qubits < qubit_min or qc_clean.num_qubits > qubit_max:
                    skipped.append((algo, nq, f"qubit count -> {qc_clean.num_qubits}"))
                    continue
                if _has_multiqubit_gates(qc_clean):
                    skipped.append((algo, nq, "3+-qubit gates after transpile"))
                    continue
                if qc_clean.size() == 0:
                    skipped.append((algo, nq, "empty circuit"))
                    continue

                circuits.append((algo, qc_clean.num_qubits, qc_clean, qc_raw))
                log(f"  [OK] {algo:20s} N={qc_clean.num_qubits:2d}  "
                    f"gates(raw)={qc_raw.size():5d}  gates(clean)={qc_clean.size():5d}")

            except Exception as e:
                skipped.append((algo, nq, str(e)[:80]))

    if skipped:
        log(f"  Skipped {len(skipped)} (algo, nq) combinations:")
        for algo, nq, reason in skipped[:15]:
            log(f"    {algo:20s} N={nq:2d}  reason: {reason}")
        if len(skipped) > 15:
            log(f"    ... and {len(skipped) - 15} more")

    return circuits


# ── Layer extraction (mirrors CircuitRepresentation._extract_layers) ────────

def extract_layers(qc: QuantumCircuit):
    """
    Extract layers from a QuantumCircuit (same logic as CircuitRepresentation).

    Returns list of dicts, each with:
        'gates_1q': list of (gate_name, (qubit,))
        'gates_2q': list of (gate_name, (q0, q1))
        'active_qubits': set of qubit indices
        'twoq_qubits':   set of qubit indices involved in 2Q gates
    """
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


# ── Gate abbreviations for circuit diagram ──────────────────────────────────

GATE_ABBREV = {
    "cx": "CX", "cz": "CZ", "swap": "SW",
    "h": "H", "x": "X", "y": "Y", "z": "Z",
    "s": "S", "t": "T", "sx": "√X",
    "rx": "Rx", "ry": "Ry", "rz": "Rz",
    "sdg": "S†", "tdg": "T†",
}


def _abbrev(gate_name: str) -> str:
    return GATE_ABBREV.get(gate_name, gate_name[:3])


# ── Raw circuit display (pre-preprocessing) ────────────────────────────────

# Algorithms known to contain composite sub-circuits (e.g. QFT blocks)
# that need decomposition to see individual gates.
NEEDS_DECOMPOSE = {"qft", "qftentangled"}

# Gate names that are "primitive" — stop decomposing when all gates are these
PRIMITIVE_GATES = {
    "cx", "cz", "cy", "swap", "ccx", "cswap",
    "h", "x", "y", "z", "s", "t", "sdg", "tdg",
    "sx", "sxdg", "rx", "ry", "rz", "u", "u1", "u2", "u3",
    "p", "cp", "crx", "cry", "crz",
    "measure", "barrier", "reset", "id",
}


def _needs_further_decompose(qc: QuantumCircuit) -> bool:
    """Check if any gate in the circuit is not primitive (i.e. is a composite block)."""
    for instruction in qc.data:
        if instruction.operation.name not in PRIMITIVE_GATES:
            return True
    return False


def _decompose_fully(qc: QuantumCircuit, max_rounds: int = 10) -> QuantumCircuit:
    """
    Repeatedly decompose until all gates are primitive.
    Needed for QFT/QFTentangled which contain QFT sub-circuit blocks.
    """
    qc_out = qc
    for _ in range(max_rounds):
        if not _needs_further_decompose(qc_out):
            break
        qc_out = qc_out.decompose()
    return qc_out


def _rebuild_with_barriers(qc: QuantumCircuit, max_layers: int = None):
    """
    Rebuild circuit inserting barriers between each DAG layer for visual
    separation.  Optionally truncate to first `max_layers` layers.

    Returns (new_qc, total_layers, was_truncated).
    """
    dag = circuit_to_dag(qc)
    all_layers = list(dag.layers())
    total = len(all_layers)

    cap = max_layers if max_layers is not None else total
    T_show = min(total, cap)
    truncated = total > cap

    new_qc = QuantumCircuit(*qc.qregs, *qc.cregs)
    for i, layer in enumerate(all_layers[:T_show]):
        subdag = layer["graph"]
        for node in subdag.op_nodes():
            new_qc.append(node.op, node.qargs, node.cargs)
        # Insert barrier after each layer (except the last)
        if i < T_show - 1:
            new_qc.barrier()

    return new_qc, total, truncated


def plot_raw_circuit(
    algo: str,
    nq: int,
    qc_raw: QuantumCircuit,
    max_layers: int = None,
    save_dir: str = None,
):
    """
    Display the original ALG-level circuit using Qiskit's built-in drawer.

    - Composite gates (QFT blocks etc.) are fully decomposed so individual
      gates are visible.
    - Barriers are inserted between DAG layers so layer boundaries appear
      as dashed vertical lines in the Qiskit drawing.
    - If max_layers is set, only the first max_layers layers are drawn.
    """
    total_depth = qc_raw.depth()
    total_gates = qc_raw.size()

    # Step 1: decompose composite blocks if needed
    if algo.lower() in NEEDS_DECOMPOSE:
        log(f"    decomposing composite gates for {algo}...")
        qc_expanded = _decompose_fully(qc_raw)
        decompose_note = f"  [decomposed: {qc_raw.size()} -> {qc_expanded.size()} gates]"
    else:
        qc_expanded = qc_raw
        decompose_note = ""

    # Step 2: rebuild with barriers between layers + optional truncation
    qc_draw, total_layers, truncated = _rebuild_with_barriers(qc_expanded, max_layers)

    trunc_note = f"  (first {max_layers} of {total_layers} layers)" if truncated else ""

    try:
        fig = qc_draw.draw(
            output="mpl",
            fold=-1,            # no folding — one long horizontal strip
            idle_wires=True,
            style={"backgroundcolor": "#FFFFFF"},
        )
        fig.suptitle(
            f"RAW circuit (pre-preprocessing):  {algo}  |  "
            f"N={qc_raw.num_qubits}  depth={total_depth}  "
            f"gates={total_gates}{decompose_note}{trunc_note}",
            fontsize=10, fontweight="bold", y=1.02,
        )
        fig.tight_layout()

        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            fname = f"{algo}_N{nq}_raw.png"
            fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")
            log(f"    saved: {fname}")

        plt.show()
        plt.close(fig)

    except Exception as e:
        # Fallback: text drawing if mpl backend fails
        log(f"    mpl draw failed ({e}), falling back to text:")
        print(qc_draw.draw(output="text", fold=80))
        print()


# ── Visualization ───────────────────────────────────────────────────────────

def plot_circuit_and_heatmap(
    algo: str,
    nq: int,
    layers: list,
    max_layers: int,
    save_dir: str = None,
):
    """
    Two-panel figure:
      Top:    circuit diagram (qubit wires + gates)
      Bottom: 2Q activity heatmap

    Both panels share the same axes: x = layer (time), y = qubit.
    """
    T_full = len(layers)
    T_show = min(T_full, max_layers)
    truncated = T_full > max_layers

    # ── Build activity matrix [T_show x N] ──
    twoq_activity = np.zeros((T_show, nq), dtype=float)
    oneq_activity = np.zeros((T_show, nq), dtype=float)
    twoq_edges_per_layer = []

    for t in range(T_show):
        layer = layers[t]
        edges = []
        for _, (q0, q1) in layer["gates_2q"]:
            twoq_activity[t, q0] = 1.0
            twoq_activity[t, q1] = 1.0
            edges.append((q0, q1))
        for _, (q,) in layer["gates_1q"]:
            if twoq_activity[t, q] == 0:
                oneq_activity[t, q] = 1.0
        twoq_edges_per_layer.append(edges)

    combined = twoq_activity + 0.4 * oneq_activity  # 0=idle, 0.4=1Q, 1.0=2Q

    # ── Stats ──
    total_2q = sum(len(l["gates_2q"]) for l in layers)
    total_1q = sum(len(l["gates_1q"]) for l in layers)
    density_2q = twoq_activity.sum() / max(T_show * nq, 1) * 100

    # ── Figure layout ──
    fig_w = max(7.0, T_show * 0.45 + 2.0)
    fig_h = max(5.0, nq * 0.55 + 3.0)
    fig, (ax_circ, ax_heat) = plt.subplots(
        2, 1, figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": [1.2, 1]},
        sharex=True,
    )

    trunc_note = f"  (first {T_show} of {T_full} layers)" if truncated else ""
    fig.suptitle(
        f"{algo}  |  N={nq}  T={T_full}{trunc_note}  |  "
        f"2Q={total_2q}  1Q={total_1q}  2Q-density={density_2q:.1f}%",
        fontsize=10, fontweight="bold",
    )

    # =====================================================================
    # TOP PANEL: Circuit diagram
    # =====================================================================
    ax = ax_circ

    # Draw qubit wires (horizontal lines)
    for q in range(nq):
        ax.hlines(q, -0.5, T_show - 0.5, color="lightgray", linewidth=0.6, zorder=0)

    # Draw gates layer by layer
    for t in range(T_show):
        layer = layers[t]

        # 1Q gates: small colored squares
        for gate_name, (q,) in layer["gates_1q"]:
            ax.add_patch(plt.Rectangle(
                (t - 0.25, q - 0.25), 0.5, 0.5,
                facecolor="#a8d8ea", edgecolor="#3a7ca5",
                linewidth=0.6, zorder=2,
            ))
            ax.text(t, q, _abbrev(gate_name), ha="center", va="center",
                    fontsize=4.5, color="#1a4a6e", zorder=3, fontweight="bold")

        # 2Q gates: vertical line connecting qubits + dots
        for gate_name, (q0, q1) in layer["gates_2q"]:
            qlo, qhi = min(q0, q1), max(q0, q1)
            ax.vlines(t, qlo, qhi, color="#c0392b", linewidth=1.5, zorder=2)
            if gate_name == "cx":
                # Control dot (smaller) on q0, target ⊕ on q1
                ax.plot(t, q0, "o", color="#c0392b", markersize=4, zorder=3)
                ax.plot(t, q1, "o", color="#c0392b", markersize=6, zorder=3,
                        markerfacecolor="white", markeredgewidth=1.2)
                ax.plot([t - 0.08, t + 0.08], [q1, q1],
                        color="#c0392b", linewidth=0.8, zorder=4)
                ax.plot([t, t], [q1 - 0.12, q1 + 0.12],
                        color="#c0392b", linewidth=0.8, zorder=4)
            else:
                # Generic 2Q gate: two dots + label
                ax.plot(t, q0, "o", color="#c0392b", markersize=5, zorder=3)
                ax.plot(t, q1, "o", color="#c0392b", markersize=5, zorder=3)
                mid = (q0 + q1) / 2
                ax.text(t + 0.3, mid, _abbrev(gate_name), ha="left", va="center",
                        fontsize=4, color="#c0392b", zorder=3)

    ax.set_ylabel("Qubit", fontsize=9)
    ax.set_yticks(range(nq))
    ax.set_yticklabels([f"q{q}" for q in range(nq)], fontsize=6)
    ax.set_ylim(nq - 0.5, -0.5)  # q0 on top
    ax.set_xlim(-0.5, T_show - 0.5)
    ax.set_title("Circuit Diagram", fontsize=9, pad=4)
    ax.tick_params(axis="x", labelbottom=False)

    # =====================================================================
    # BOTTOM PANEL: Activity heatmap  (transposed: x=layer, y=qubit)
    # =====================================================================
    ax = ax_heat

    # Transpose combined so shape is [N, T_show] → imshow with x=layer, y=qubit
    heatmap_data = combined.T  # [N, T_show]

    im = ax.imshow(
        heatmap_data,
        aspect="auto",
        origin="upper",       # q0 on top, matching circuit
        cmap="Blues",
        interpolation="nearest",
        vmin=0, vmax=1.0,
    )

    # Draw 2Q interaction lines within each layer (now x=layer, y=qubit)
    for t, edges in enumerate(twoq_edges_per_layer):
        for q0, q1 in edges:
            ax.plot(
                [t, t], [q0, q1],
                color="darkred", linewidth=0.8, alpha=0.5,
            )

    ax.set_xlabel("Layer", fontsize=9)
    ax.set_ylabel("Qubit", fontsize=9)
    ax.set_yticks(range(nq))
    ax.set_yticklabels([f"q{q}" for q in range(nq)], fontsize=6)
    ax.set_title("2Q Activity Heatmap", fontsize=9, pad=4)

    if T_show <= 40:
        tick_step = 1 if T_show <= 20 else 2 if T_show <= 30 else 5
        ax.set_xticks(range(0, T_show, tick_step))
        ax.set_xticklabels([str(t) for t in range(0, T_show, tick_step)], fontsize=6)

    legend_patches = [
        mpatches.Patch(color=plt.cm.Blues(1.0), label="2Q gate"),
        mpatches.Patch(color=plt.cm.Blues(0.4), label="1Q only"),
        mpatches.Patch(facecolor="white", edgecolor="gray", label="Idle"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=7, framealpha=0.8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        fname = f"{algo}_N{nq}_circuit_activity.png"
        fig.savefig(os.path.join(save_dir, fname), dpi=180, bbox_inches="tight")
        log(f"    saved: {fname}")

    plt.show()
    plt.close(fig)


# ── Summary table ───────────────────────────────────────────────────────────

def print_summary_table(circuits_info):
    """Print a compact table of all loaded circuits."""
    print("\n" + "=" * 72)
    print(f"  {'Algorithm':<20s} {'N':>4s} {'T(layers)':>10s} {'2Q gates':>9s} "
          f"{'1Q gates':>9s} {'2Q dens%':>9s}")
    print("-" * 72)
    for info in circuits_info:
        print(f"  {info['algo']:<20s} {info['N']:>4d} {info['T']:>10d} "
              f"{info['total_2q']:>9d} {info['total_1q']:>9d} "
              f"{info['density_2q']:>8.1f}%")
    print("=" * 72)


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Inspect MQT Bench circuit structure for case study selection")
    p.add_argument("--qubit_min", type=int, default=8)
    p.add_argument("--qubit_max", type=int, default=12)
    p.add_argument("--max_layers", type=int, default=30,
                   help="Cap displayed layers (default 30)")
    p.add_argument("--algorithms", type=str, default=None,
                   help="Comma-separated algorithm filter (default: all)")
    p.add_argument("--save_dir", type=str, default=None,
                   help="Save figures to this directory (else show only)")
    p.add_argument("--no_plot", action="store_true",
                   help="Print summary table only, skip plots")
    return p.parse_args()


def main():
    args = parse_args()

    if args.algorithms:
        algorithms = [a.strip() for a in args.algorithms.split(",")]
    else:
        algorithms = MQT_ALGORITHMS

    log(f"Loading MQT circuits: N=[{args.qubit_min}, {args.qubit_max}], "
        f"algorithms={algorithms}")

    mqt_circuits = load_mqt_circuits(algorithms, args.qubit_min, args.qubit_max)

    if not mqt_circuits:
        log("No circuits loaded. Check mqt.bench installation.")
        sys.exit(1)

    log(f"Loaded {len(mqt_circuits)} circuits total.")

    # ── Extract layers and stats ──
    circuits_info = []
    for algo, nq, qc, qc_raw in mqt_circuits:
        layers = extract_layers(qc)
        T = len(layers)
        total_2q = sum(len(l["gates_2q"]) for l in layers)
        total_1q = sum(len(l["gates_1q"]) for l in layers)
        T_show = min(T, args.max_layers)
        twoq_cells = sum(
            len(layers[t]["twoq_qubits"]) for t in range(T_show)
        )
        density = twoq_cells / max(T_show * nq, 1) * 100

        circuits_info.append({
            "algo": algo, "N": nq, "T": T,
            "total_2q": total_2q, "total_1q": total_1q,
            "density_2q": density,
            "layers": layers, "qc": qc, "qc_raw": qc_raw,
        })

    # ── Summary table ──
    print_summary_table(circuits_info)

    # ── Plots ──
    if not args.no_plot:
        for info in circuits_info:
            # 1) Raw circuit (pre-preprocessing, with measurements)
            plot_raw_circuit(
                algo=info["algo"],
                nq=info["N"],
                qc_raw=info["qc_raw"],
                max_layers=args.max_layers,
                save_dir=args.save_dir,
            )
            # 2) Preprocessed circuit diagram + activity heatmap
            plot_circuit_and_heatmap(
                algo=info["algo"],
                nq=info["N"],
                layers=info["layers"],
                max_layers=args.max_layers,
                save_dir=args.save_dir,
            )


if __name__ == "__main__":
    main()
