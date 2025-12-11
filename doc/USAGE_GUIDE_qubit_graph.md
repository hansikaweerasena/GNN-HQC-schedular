# Qubit Interaction Graph - Interactive Development Complete ✅

## What You Built

A complete **qubit interaction graph generator** that converts your segmented quantum circuit into a machine learning-ready graph representation.

## 6 Development Steps (All Tested)

### Step 1: Extract 2-Qubit Interactions
```python
interactions = extract_qubit_interactions(circuit, segment_ids)
```
Output: List of dicts with `{q1, q2, gate_type, layer, segment}`

### Step 2: Build NetworkX MultiGraph
```python
G = build_qubit_graph(circuit, interactions)
```
Output: `nx.MultiGraph` with nodes=qubits, edges=interactions

### Step 3: Add Node Features
```python
add_node_features(G, circuit)
```
Adds to each node: `first_layer`, `gate_count`

### Step 4: Add Aggregated Edge Features
```python
add_aggregated_edge_features_v2(G, interactions)
```
Adds to edges: `interaction_layers`, `interaction_count`

### Step 5: Convert to Arrays
```python
x, edge_index, edge_attr = multigraph_to_arrays(G, known_gate_types=['cx', 'cz'])
```
Outputs:
- `x`: Node features `[num_nodes, 2]`
- `edge_index`: Edge connectivity `[2, num_edges]`
- `edge_attr`: Edge features `[num_edges, feature_dim]`

### Step 6: Analyze Per Segment
```python
H = get_segment_subgraph(G, segment_id=0)
print_segment_info(G, segment_ids)
```

## One-Line Pipeline

```python
from src.qubit_interaction_graph import build_qubit_interaction_graph_full

G, x, edge_index, edge_attr = build_qubit_interaction_graph_full(
    circuit_rep,
    segment_ids,
    known_gate_types=['cx', 'cz']
)
```

## Data Shapes

For a circuit with **N qubits**, **E 2-qubit interactions**, and **K gate types**:

| Variable | Shape | Content |
|----------|-------|---------|
| `x` | `(N, 2)` | Node features: `[first_layer, gate_count]` |
| `edge_index` | `(2, E)` | Edge connectivity: `[source_nodes, target_nodes]` |
| `edge_attr` | `(E, K+3)` | Edge features: `[one-hot_gate_type, layer, segment]` |

## Next: Phase 2 - GNN Spatial Encoder

The outputs (`x`, `edge_index`, `edge_attr`) are ready for your GNN:

```python
# These go into a GATv2Conv layer:
x_embedded = gnn(x, edge_index, edge_attr)  # Output: [N, embedding_dim]
```

Your GATv2Conv will learn:
- Which qubits are "important" (high degree, central)
- Which interactions are "critical" (many gates on that edge)
- A spatial embedding for each qubit

## File Location

Save the `qubit_interaction_graph.py` file in:
```
project_root/
├── src/
│   ├── circuit_representation.py    (your existing)
│   ├── circuit_segmentation.py      (your existing)
│   ├── qubit_interaction_graph.py   (NEW - use this!)
```

## Test It

```python
from src.circuit_representation import CircuitRepresentation
from src.circuit_segmentation import segment_circuit
from src.qubit_interaction_graph import build_qubit_interaction_graph_full

# Load your real circuit
circuit_rep = CircuitRepresentation(your_qiskit_circuit)

# Segment it
segments, seg_ids = segment_circuit(circuit_rep.layers, threshold=0.3)

# Build the graph
G, x, edge_index, edge_attr = build_qubit_interaction_graph_full(
    circuit_rep, seg_ids
)

# Check it
print(f"Graph: {G.number_of_nodes()} qubits, {G.number_of_edges()} interactions")
print(f"Node features shape: {x.shape}")
print(f"Edge features shape: {edge_attr.shape}")
```

## Key Functions Reference

| Function | Input | Output | Purpose |
|----------|-------|--------|---------|
| `extract_qubit_interactions` | circuit, segment_ids | List[dict] | Extract 2-qubit gates |
| `build_qubit_graph` | circuit, interactions | nx.MultiGraph | Create graph structure |
| `add_node_features` | G, circuit | None (modifies G) | Compute node stats |
| `add_aggregated_edge_features_v2` | G, interactions | None (modifies G) | Aggregate per-pair stats |
| `multigraph_to_arrays` | G, gate_types | (x, edge_index, edge_attr) | Convert to tensors |
| `build_qubit_interaction_graph_full` | circuit, segment_ids | (G, x, edge_index, edge_attr) | **One-line pipeline** |
| `get_segment_subgraph` | G, segment_id | nx.MultiGraph | Extract segment subgraph |
| `print_graph_stats` | G | None | Print graph statistics |
| `print_segment_info` | G, segment_ids | None | Print per-segment analysis |

## Summary

✅ **Step 1:** Extract interactions from circuit  
✅ **Step 2:** Build NetworkX MultiGraph  
✅ **Step 3:** Add node features (topology-aware)  
✅ **Step 4:** Add aggregated edge features  
✅ **Step 5:** Convert to ML-ready tensors  
✅ **Step 6:** Visualize per segment for validation  

**All 6 steps tested and working!**

Your graph is now ready for the GNN spatial encoder. 🚀
