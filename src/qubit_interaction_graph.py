"""
qubit_interaction_graph.py

Purpose:
    Build a qubit interaction multigraph from a segmented quantum circuit.
    This graph is the input to the GNN spatial encoder.

Key concepts:
    - Nodes = qubits (0..num_qubits-1)
        Node features:
          * first_layer: int, earliest layer where this qubit participates in any gate
          * gate_count: int, total number of gates (1-qubit or 2-qubit) involving this qubit

    - Edges = 2-qubit gate interactions (one edge per gate instance)
        Edge attributes:
          * gate_type: str, e.g. "cx", "cz"
          * layer: int, layer index where this gate occurs
          * segment: int, segment id this gate belongs to

    - MultiGraph: same qubit pair (q1, q2) can have multiple edges
      if they interact in different layers/segments

    - Aggregated stats per pair (stored on first edge for convenience):
          * interaction_layers: List[int], all layers where this pair interacts
          * interaction_count: int, total number of interactions

Main function:
    build_qubit_interaction_multigraph(circuit, segment_ids)
      Input:
        * circuit: CircuitRepresentation object with num_qubits and layers
        * segment_ids: List[int], segment id for each layer
      Output:
        * G: NetworkX MultiGraph with all node/edge features
        * x: numpy array of node features [num_nodes, 2]
        * edge_index: numpy array of edge connectivity [2, num_edges]
        * edge_attr: numpy array of edge features [num_edges, feature_dim]

Helper functions:
    extract_qubit_interactions(circuit, segment_ids)
      Extract all 2-qubit gate interactions from the circuit

    build_qubit_graph(circuit, interactions)
      Build NetworkX MultiGraph structure

    add_node_features(G, circuit)
      Compute and attach node features

    add_aggregated_edge_features_v2(G, interactions)
      Add aggregated per-pair statistics to edges

    multigraph_to_arrays(G, known_gate_types)
      Convert NetworkX graph to numpy arrays (torch-compatible)

    get_segment_subgraph(G, segment_id)
      Extract subgraph for a specific segment

    print_segment_info(G, segment_ids)
      Print per-segment analysis

Usage:
    from src.circuit_representation import CircuitRepresentation
    from src.circuit_segmentation import segment_circuit
    from src.qubit_interaction_graph import build_qubit_interaction_multigraph

    # Load circuit and segment it
    circuit_rep = CircuitRepresentation(qiskit_circuit)
    segments, seg_ids = segment_circuit(circuit_rep.layers, threshold=0.3)

    # Build the qubit interaction graph
    G, x, edge_index, edge_attr = build_qubit_interaction_multigraph(circuit_rep, seg_ids)

    # Now ready for GNN:
    # x → node features for GNN input
    # edge_index → graph connectivity for GNN
    # edge_attr → edge features for GNN
"""

from typing import List, Dict, Tuple
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional



# ============================================================================
# STEP 1: Extract qubit interactions
# ============================================================================

def extract_qubit_interactions(
    circuit,  # CircuitRepresentation
    segment_ids: List[int]
) -> List[Dict]:
    """
    Extract all 2-qubit gate interactions from the circuit.

    Returns:
        List of dicts with keys: q1, q2, gate_type, layer, segment
    """
    interactions = []

    for layer_idx, layer in enumerate(circuit.layers):
        seg_id = segment_ids[layer_idx]

        for gate_name, qubits in layer.gates:
            if len(qubits) == 2:  # Only 2-qubit gates
                q1, q2 = qubits
                interactions.append({
                    'q1': q1,
                    'q2': q2,
                    'gate_type': gate_name,
                    'layer': layer_idx,
                    'segment': seg_id
                })

    return interactions


# ============================================================================
# STEP 2: Build NetworkX MultiGraph
# ============================================================================

def build_qubit_graph(
    circuit,  # CircuitRepresentation
    interactions: List[Dict]
) -> nx.MultiGraph:
    """
    Build a NetworkX MultiGraph with qubits as nodes and 2-qubit interactions as edges.
    """
    G = nx.MultiGraph()

    # Add all qubit nodes
    for q in range(circuit.num_qubits):
        G.add_node(q)

    # Add edges for each 2-qubit interaction
    for inter in interactions:
        q1, q2 = inter['q1'], inter['q2']
        G.add_edge(
            q1, q2,
            gate_type=inter['gate_type'],
            layer=inter['layer'],
            segment=inter['segment']
        )

    return G


# ============================================================================
# STEP 3: Add node features
# ============================================================================

def add_node_features(G: nx.MultiGraph, circuit):
    """
    Compute and add node features:
      - first_layer: earliest layer where this qubit participates in any gate
      - gate_count: total number of gates (1-qubit or 2-qubit) involving this qubit
    """
    num_qubits = circuit.num_qubits

    # Initialize
    first_layer = [None] * num_qubits
    gate_count = [0] * num_qubits

    # Traverse all layers and gates
    for layer_idx, layer in enumerate(circuit.layers):
        for gate_name, qubits in layer.gates:
            for q in qubits:
                gate_count[q] += 1
                if first_layer[q] is None or layer_idx < first_layer[q]:
                    first_layer[q] = layer_idx

    # Attach to graph nodes
    for q in range(num_qubits):
        G.nodes[q]['first_layer'] = (
            first_layer[q] if first_layer[q] is not None else -1
        )
        G.nodes[q]['gate_count'] = gate_count[q]


# ============================================================================
# STEP 4: Add aggregated edge features
# ============================================================================

def add_aggregated_edge_features_v2(G: nx.MultiGraph, interactions: List[Dict]):
    """
    For each qubit pair, aggregate:
      - interaction_layers: list of all layer indices where they interact
      - interaction_count: total count of interactions
    """
    # Aggregate by pair
    pair_stats = {}
    for inter in interactions:
        a, b = sorted((inter['q1'], inter['q2']))
        if (a, b) not in pair_stats:
            pair_stats[(a, b)] = {
                'interaction_layers': [],
                'interaction_count': 0
            }
        pair_stats[(a, b)]['interaction_layers'].append(inter['layer'])
        pair_stats[(a, b)]['interaction_count'] += 1

    # Store on each edge
    for u, v, k, attr in G.edges(keys=True, data=True):
        pair = tuple(sorted((u, v)))
        if pair in pair_stats:
            stats = pair_stats[pair]
            G[u][v][k]['interaction_layers'] = stats['interaction_layers']
            G[u][v][k]['interaction_count'] = stats['interaction_count']


# ============================================================================
# STEP 5: Convert to NumPy arrays (torch-compatible)
# ============================================================================

def multigraph_to_arrays(
    G: nx.MultiGraph,
    known_gate_types: List[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert NetworkX MultiGraph to numpy arrays for GNN input.

    Returns:
        x: Node features [num_nodes, 2] with (first_layer, gate_count)
        edge_index: Edge connectivity [2, num_edges]
        edge_attr: Edge features [num_edges, feature_dim]
               where feature_dim = len(known_gate_types) + 1 (one-hot) + 2 (layer, segment)
    """
    if known_gate_types is None:
        known_gate_types = ['cx', 'cz']

    num_nodes = G.number_of_nodes()

    # Node features
    first_layers = []
    gate_counts = []
    for n in range(num_nodes):
        data = G.nodes[n]
        first_layers.append(float(data.get('first_layer', -1)))
        gate_counts.append(float(data.get('gate_count', 0)))

    x = np.stack([first_layers, gate_counts], axis=-1).astype(np.float32)

    # Helper: one-hot encode gate type
    def one_hot_gate(gt: str, known_types: List[str]) -> np.ndarray:
        vec = np.zeros(len(known_types) + 1, dtype=np.float32)  # +1 for "other"
        try:
            idx = known_types.index(gt.lower())
        except ValueError:
            idx = len(known_types)  # "other" slot
        vec[idx] = 1.0
        return vec

    # Edge features
    edge_u = []
    edge_v = []
    edge_features = []

    for u, v, k, attr in G.edges(keys=True, data=True):
        edge_u.append(u)
        edge_v.append(v)

        gt_vec = one_hot_gate(attr.get('gate_type', 'other'), known_gate_types)
        layer_val = float(attr.get('layer', -1))
        seg_val = float(attr.get('segment', -1))

        feat_vec = np.concatenate([gt_vec, [layer_val, seg_val]])
        edge_features.append(feat_vec)

    edge_index = np.array([edge_u, edge_v], dtype=np.int64)
    edge_attr = (
        np.stack(edge_features, axis=0).astype(np.float32)
        if edge_features
        else np.zeros((0, len(one_hot_gate('cx', known_gate_types)) + 2), dtype=np.float32)
    )

    return x, edge_index, edge_attr


# ============================================================================
# MAIN FUNCTION: Complete pipeline
# ============================================================================

def build_qubit_interaction_multigraph(
    circuit,  # CircuitRepresentation
    segment_ids: List[int],
    known_gate_types: List[str] = None
) -> Tuple[nx.MultiGraph, np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete pipeline: build qubit interaction graph from circuit.

    Args:
        circuit: CircuitRepresentation object
        segment_ids: List[int], segment id for each layer
        known_gate_types: List[str], known gate types (default: ['cx', 'cz'])

    Returns:
        G: NetworkX MultiGraph with node/edge features
        x: Node feature array [num_nodes, 2]
        edge_index: Edge connectivity [2, num_edges]
        edge_attr: Edge feature array [num_edges, feature_dim]
    """
    if known_gate_types is None:
        known_gate_types = ['cx', 'cz']

    # Step 1: Extract interactions
    interactions = extract_qubit_interactions(circuit, segment_ids)

    # Step 2: Build graph
    G = build_qubit_graph(circuit, interactions)

    # Step 3: Add node features
    add_node_features(G, circuit)

    # Step 4: Add aggregated edge features
    add_aggregated_edge_features_v2(G, interactions)

    # Step 5: Convert to arrays
    x, edge_index, edge_attr = multigraph_to_arrays(G, known_gate_types)

    return G, x, edge_index, edge_attr


# ============================================================================
# HELPER: Per-segment analysis
# ============================================================================

def get_segment_subgraph(G: nx.MultiGraph, segment_id: int) -> nx.MultiGraph:
    """
    Extract a subgraph containing only edges from a specific segment.
    """
    H = nx.MultiGraph()

    for u, v, k, attr in G.edges(keys=True, data=True):
        if attr.get('segment') == segment_id:
            if u not in H:
                H.add_node(u, **G.nodes[u])
            if v not in H:
                H.add_node(v, **G.nodes[v])
            H.add_edge(u, v, key=k, **attr)

    return H


def print_segment_info(G: nx.MultiGraph, segment_ids: List[int]):
    """
    Print per-segment analysis.
    """
    num_segments = max(segment_ids) + 1

    print(f"\n=== Per-Segment Qubit Interaction Analysis ===\n")
    print(f"Total segments: {num_segments}\n")

    for seg_id in range(num_segments):
        H = get_segment_subgraph(G, seg_id)

        print(f"Segment {seg_id}:")
        print(f"  Active qubits: {sorted(H.nodes())}")
        print(f"  Interactions: {list(H.edges())}")

        gate_types_in_seg = [
            attr['gate_type']
            for u, v, k, attr in H.edges(keys=True, data=True)
        ]
        print(f"  Gate types: {gate_types_in_seg if gate_types_in_seg else '[]'}")
        print()


def print_graph_stats(G: nx.MultiGraph):
    """Print overall graph statistics."""
    print(f"\n=== Qubit Interaction Graph Statistics ===\n")
    print(f"Number of qubits: {G.number_of_nodes()}")
    print(f"Number of 2-qubit interactions: {G.number_of_edges()}")

    gate_counts = [G.nodes[n].get('gate_count', 0) for n in G.nodes()]
    print(f"Average gates per qubit: {np.mean(gate_counts):.2f}")
    print(f"Max gates on one qubit: {np.max(gate_counts)}")

    # Count gate types
    gate_types = {}
    for u, v, k, attr in G.edges(keys=True, data=True):
        gt = attr.get('gate_type', 'unknown')
        gate_types[gt] = gate_types.get(gt, 0) + 1

    print(f"Gate type distribution: {gate_types}\n")


def visualize_segment_graph(
    G: nx.MultiGraph,
    segment_id: int,
    with_labels: bool = True,
    title: Optional[str] = None,
):
    """
    Visualize the interaction subgraph for a given segment_id.

    - Nodes = qubits
    - Edges = 2-qubit gates that occur in this segment
    - Edge labels show gate_type and layer
    """
    H = get_segment_subgraph(G, segment_id)
    if H.number_of_nodes() == 0:
        print(f"No edges for segment {segment_id}")
        return

    pos = nx.spring_layout(H, seed=42)  # deterministic layout

    plt.figure(figsize=(12, 8))
    nx.draw(
        H,
        pos,
        with_labels=with_labels,
        node_color="lightblue",
        node_size=800,
        edge_color="gray",
    )

    # Edge labels: gate_type@layer
    edge_labels = {}
    for u, v, k, attr in H.edges(keys=True, data=True):
        gt = attr.get("gate_type", "?")
        layer = attr.get("layer", "?")
        edge_labels[(u, v, k)] = f"{gt}@L{layer}"

    # MultiGraph edge labeling: need keys=True
    nx.draw_networkx_edge_labels(
        H,
        pos,
        edge_labels={(u, v): lab for (u, v, k), lab in edge_labels.items()},
        font_size=8,
    )

    if title is None:
        title = f"Segment {segment_id} interaction graph"
    plt.title(title)
    plt.tight_layout()
    plt.show()


