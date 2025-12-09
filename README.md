# GNN-HQC-schedular

Learning-based **spacetime qubit scheduler** for heterogeneous quantum computing (HQC) using **Graph Neural Networks (GNNs)** and simple temporal segmentation.

---

## 1. Problem: What Are We Solving?

Future heterogeneous quantum systems will have **multiple QPU types** (e.g., ion traps, superconducting, neutral atoms) with different:

- Gate speeds
- Error rates (decoherence)
- Connectivity/topologies
- Qubit capacities

Given a **quantum circuit** (known ahead of time), we want to decide:

> **Where each logical qubit should live over time (in which QPU / region, per segment)**  
> to minimize a cost such as:
> - total noise (dwell + movement),
> - communication overhead,
> - or a weighted combination.

This is a **static, compile-time, multi-constraint optimization** problem, not a runtime RL control problem.

---

## 2. Approach Overview

We build an end-to-end pipeline (starting with a simplified Phase 1 prototype):

1. **Circuit Representation**
   - Use Qiskit to generate / load circuits (e.g., QFT).
   - Convert circuit to a DAG and extract **layers** (parallel gates).
   - For each layer, identify **active qubits** and gate structure.

2. **Temporal Segmentation (Time Axis)**
   - Treat per-layer active qubit sets as a sequence.
   - Use **Jaccard similarity** between consecutive layers:
     - start a new segment when the active set changes “enough”.
   - Result: a small number of **time segments**, each grouping several layers.

3. **Qubit Interaction Graph (Space Axis) – Phase 2**
   - Build a graph where:
     - Nodes = qubits
     - Edges = “these qubits interact via 2-qubit gates”
   - Add temporal features from segments/layers.

4. **GNN-Based Embedding – Phase 2**
   - Apply a GNN (and possibly a temporal model like GRU) to learn **qubit embeddings**:
     - embeddings capture which qubits should be near each other,
       and how they behave over time.

5. **Soft Clustering / Assignment – Phase 2**
   - For each segment, use embeddings to assign qubits to QPU types
     (e.g., compute vs storage) via **soft clustering**.
   - Represent assignment as probabilities σ[q, s, p] over QPUs.

6. **Cost Function & Training – Phase 2**
   - Define a differentiable cost:
     - Execution cost (dwell noise on each QPU)
     - Movement cost (penalize changes in σ between segments)
   - Train GNN end-to-end so that embeddings → clusterings → low cost.

**Phase 1** (what we are building now in 24h) focuses on:
- Steps 1–2 + visualization,
- simple end-to-end prototype with a dummy embedding and cost,
- and a minimal training loop.

---

## 3. Dependencies & Environment Setup

### 3.1. Clone Repository

git clone https://github.com/hansikaweerasena/GNN-HQC-schedular.git
cd GNN-HQC-schedular

### 3.2. Create Virtual Environment

Using venv:

python3 -m venv venv
source venv/bin/activate      # On macOS/Linux
# venv\Scripts\activate       # On Windows

Or using conda:

conda create -n quantum-gnn python=3.10
conda activate quantum-gnn

### 3.3. Install Dependencies

pip install -r requirements.txt

Current minimal requirements.txt (Phase 1):

qiskit==1.0.2
numpy==1.24.3
matplotlib==3.8.0
jupyter==1.0.0
networkx==3.2

(We will add torch and torch-geometric later when we implement the GNN.)

---

## 4. Repository Structure

Current and planned structure:

GNN-HQC-schedular/
├── README.md                  # This file
├── requirements.txt
├── simple_test.py             # Quick end-to-end sanity test (Phase 1)
├── test_thresholds.py         # Compare segmentation across thresholds
├── test_thresholds_visual.py  # Visualize segmentation vs threshold
│
├── src/
│   ├── __init__.py
│   ├── circuit_generation.py      # Generate benchmark circuits (QFT, later QAOA/random)
│   ├── circuit_representation.py  # DAG + layers + active qubits per layer
│   ├── circuit_segmentation.py    # Jaccard-based temporal segmentation into segments
│   ├── circuit_visualization.py   # Circuit diagrams, layer activity heatmaps, segmentation plots
│   ├── qubit_graph.py             # (Phase 2) Build qubit interaction graph + features
│   ├── gnn_model.py               # (Phase 2) GNN/RNN for qubit embeddings
│   ├── clustering.py              # (Phase 2) Soft clustering / assignment per segment
│   ├── cost_model.py              # (Phase 2) Execution + movement cost functions
│   ├── training.py                # (Phase 2) Training loop for GNN
│   └── utils.py                   # Shared utilities
│
├── data/
│   ├── circuits/                  # Saved QASM circuits
│   └── results/                   # Logs, metrics, plots
│
├── notebooks/
│   ├── 01_circuit_exploration.ipynb   # Generate circuit, layers, segmentation (Phase 1)
│   ├── 02_qubit_graph.ipynb           # (Phase 2)
│   ├── 03_gnn_training.ipynb          # (Phase 2)
│   └── 04_evaluation.ipynb            # (Phase 2)
│
└── tests/
    ├── test_circuit_representation.py # (Phase 1) unit tests
    ├── test_segmentation.py           # (Phase 1) unit tests
    └── ...

---

## 5. Phase 1: Working Prototype (End-to-End Skeleton)

Goal in 24 hours:
A simple but complete pipeline that:

1. Generates a circuit (QFT, 5–10 qubits).
2. Converts to DAG and extracts layers.
3. Computes per-layer active qubits.
4. Runs Jaccard-based temporal segmentation with configurable threshold.
5. Visualizes:
   - The circuit
   - Layer activity heatmap (qubit vs layer)
   - Segment boundaries on the heatmap
6. (Optional) Adds a dummy cost function and simple “embedding” to prepare for GNN integration.

Quick Start: Run the basic test

python simple_test.py

This will:

- Print basic circuit stats.
- Show the circuit diagram.
- Show layer activity.
- Show segmentation.

Threshold exploration

python test_thresholds_visual.py

This will show how different Jaccard thresholds (e.g., 0.1, 0.3, 0.5, 0.7, 0.9) change the segmentation, all on the same circuit.

---

## 6. Roadmap (High-Level)

- Phase 1 (now):
  - Circuit generation, DAG, layers
  - Temporal segmentation
  - Visual debugging + small tests
- Phase 2:
  - Qubit interaction graph + temporal features
  - GNN model for qubit embeddings
  - Soft clustering and cost model
  - End-to-end training loop
- Phase 3:
  - More realistic QPU models
  - Evaluation vs baselines (hand-crafted features, greedy)
  - Ablation studies
- Phase 4:
  - Scaling to more qubits
  - Multi-circuit + possible RL layer for runtime adaptation

---

## 7. Notes

- The current implementation is deliberately simplified:
  - Single circuit type (QFT) for debugging
  - No capacity or topology constraints yet
  - No GNN yet (Phase 2)
- The focus of Phase 1 is to ensure:
  - Circuit representation is correct
  - Temporal segmentation behaves as expected
  - Visualization is clear enough to reason about next steps

We will iteratively refine this README as the prototype becomes a full GNN-based scheduler.
