#!/usr/bin/env python3
"""
MOSAIC environment/import sanity test.

Run from the root of the GNN-HQC-schedular repository:

    python test_mosaic_imports.py

This script does not start training. It only verifies that the main Python
dependencies and MOSAIC modules used by the training pipeline can be imported.
"""

import sys

print("=" * 68)
print("MOSAIC IMPORT / ENVIRONMENT TEST")
print("=" * 68)

try:
    import torch
    import torch_geometric
    import qiskit
    from torch_geometric.nn import GATv2Conv
    from torch_geometric.data import Data

    print(f"Python           : {sys.version.split()[0]}")
    print(f"PyTorch          : {torch.__version__}")
    print(f"PyTorch CUDA     : {torch.version.cuda}")
    print(f"PyG              : {torch_geometric.__version__}")
    print(f"Qiskit           : {qiskit.__version__}")
    print(f"CUDA available   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU              : {torch.cuda.get_device_name(0)}")
    print("GATv2Conv import : OK")
    print("PyG Data import  : OK")
except Exception as exc:
    print("\n[FAIL] Core environment import failed:")
    print(repr(exc))
    raise

print("\nTesting MOSAIC imports...")

tests = [
    ("circuit_sources", lambda: __import__("utils.circuit_sources", fromlist=["build_provider"])),
    ("circuit_representation", lambda: __import__("src.circuit_representation", fromlist=["CircuitRepresentation"])),
    ("circuit_segmentation", lambda: __import__("src.circuit_segmentation", fromlist=["segment_circuit"])),
    ("qubit_interaction_graph", lambda: __import__(
        "src.qubit_interaction_graph",
        fromlist=["build_layer_graph_arrays", "compute_window_sizes_from_config"],
    )),
    ("EvolvingGNN", lambda: __import__("src.evolving_gnn", fromlist=["EvolvingGNN"])),
    ("SegmentClustering", lambda: __import__("src.clustering_head", fromlist=["SegmentClustering"])),
    ("cost_function", lambda: __import__(
        "src.cost_function",
        fromlist=["TotalCost", "CapacityPenalty", "SegmentStatsExtractor"],
    )),
    ("train_utils", lambda: __import__("utils.train_utils", fromlist=["train_step", "batch_train_step"])),
]

failed = []
for name, importer in tests:
    try:
        importer()
        print(f"{name:26s}: OK")
    except Exception as exc:
        print(f"{name:26s}: FAIL")
        print(f"  -> {type(exc).__name__}: {exc}")
        failed.append((name, exc))

print("\n" + "=" * 68)

if failed:
    print(f"IMPORT TEST FAILED: {len(failed)} module(s) failed.")
    print("=" * 68)
    sys.exit(1)

print("ALL MOSAIC IMPORTS PASSED")
print("=" * 68)
