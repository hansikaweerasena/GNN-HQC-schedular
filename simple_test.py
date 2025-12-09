"""Step-by-step: visualize circuit, layers, and Jaccard segmentation."""
import matplotlib.pyplot as plt
import numpy as np

from circuit_generation import generate_qft_circuit
from qiskit.converters import circuit_to_dag


def jaccard_similarity(a: set, b: set) -> float:
    """Jaccard(A,B) = |A∩B| / |A∪B|."""
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def main():
    # 1) Generate circuit
    qft = generate_qft_circuit(5)
    print("✓ Circuit generated")
    print(f"  Depth: {qft.depth()}")
    print(f"  Gates: {qft.size()}")

    # 1a) Visualize circuit
    print("\n✓ Drawing circuit...")
    qft.draw(output='mpl', fold=-1)
    plt.show()

    # 2) Convert to DAG and extract layers
    dag = circuit_to_dag(qft)
    print("\n✓ Converted to DAG")

    layers = list(dag.layers())
    num_layers = len(layers)
    num_qubits = qft.num_qubits
    print(f"✓ Extracted {num_layers} layers")

    # Map Qiskit qubit objects -> indices
    qubit_index = {qb: i for i, qb in enumerate(qft.qubits)}

    # 3) Extract active qubits per layer
    active_per_layer = []  # list of sets
    gates_per_layer = []

    for layer_idx, layer in enumerate(layers):
        subdag = layer["graph"]
        active = set()
        gates_here = []

        for node in subdag.op_nodes():
            gate_name = node.op.name
            qargs = [qubit_index[qb] for qb in node.qargs]
            gates_here.append((gate_name, qargs))
            for q in qargs:
                active.add(q)

        active_per_layer.append(active)
        gates_per_layer.append(gates_here)

    print("\nFirst 3 layers (gates + active qubits):")
    for i in range(min(3, num_layers)):
        print(f"  Layer {i}: gates={gates_per_layer[i]}, active={sorted(active_per_layer[i])}")

    # 4) Visualize layer activity as a heatmap
    activity = np.zeros((num_qubits, num_layers), dtype=int)
    for l in range(num_layers):
        for q in active_per_layer[l]:
            activity[q, l] = 1

    plt.figure(figsize=(8, 4))
    plt.imshow(activity, aspect='auto', cmap='Blues', interpolation='nearest')
    plt.colorbar(label='Active (1) / Inactive (0)')
    plt.xlabel('Layer')
    plt.ylabel('Qubit')
    plt.title('Layer Activity (Qubit vs Layer)')
    plt.yticks(range(num_qubits))
    plt.xticks(range(num_layers))
    plt.tight_layout()
    plt.show()

    # 5) Jaccard-based temporal segmentation
    threshold = 0.3
    segment_ids = []  # segment index per layer
    segments = []     # list of (start_layer, end_layer)
    current_seg = 0
    current_start = 0
    prev_active = None

    for l in range(num_layers):
        act = active_per_layer[l]
        if prev_active is not None:
            j = jaccard_similarity(prev_active, act)
            # If activity pattern changes a lot, start new segment
            if j < threshold:
                segments.append((current_start, l - 1))
                current_seg += 1
                current_start = l
        segment_ids.append(current_seg)
        prev_active = act

    # close last segment
    segments.append((current_start, num_layers - 1))

    print(f"\n✓ Segmentation with threshold={threshold}")
    print(f"  Number of segments: {len(segments)}")
    for idx, (s, e) in enumerate(segments):
        # collect all active qubits in this segment
        seg_active = set()
        for l in range(s, e + 1):
            seg_active |= active_per_layer[l]
        print(f"  Segment {idx}: layers [{s}..{e}], active qubits={sorted(seg_active)}")

    # 6) Visualize segmentation boundaries on top of activity
    plt.figure(figsize=(8, 4))
    plt.imshow(activity, aspect='auto', cmap='Blues', interpolation='nearest')
    plt.colorbar(label='Active (1) / Inactive (0)')
    plt.xlabel('Layer')
    plt.ylabel('Qubit')
    plt.title(f'Layer Activity with Segments (threshold={threshold})')
    plt.yticks(range(num_qubits))
    plt.xticks(range(num_layers))

    # Draw red vertical lines at segment boundaries
    for _, end_layer in segments[:-1]:  # skip last segment (no boundary after it)
        plt.axvline(x=end_layer + 0.5, color='red', linestyle='--', linewidth=2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
