# utils/circuit_sources.py
from typing import Callable, Dict, Any, Optional
import numpy as np

from src.circuit_generation import (
    generate_random_circuit_custom,
    generate_roi_composed_circuit,
)

# ----------------------------
# Registry wrappers
# Normalize everything to: fn(seed=..., **kwargs) -> QuantumCircuit
# and prefer using num_qubits in kwargs everywhere.
# ----------------------------

def _random_custom(*, seed: int, num_qubits: int, depth: int, gate_density: float,
                   two_qubit_ratio: float = 0.5, **_ignored):
    return generate_random_circuit_custom(
        n_qubits=num_qubits,
        depth=depth,
        gate_density=gate_density,
        seed=seed,
        two_qubit_ratio=two_qubit_ratio,
    )

def _roi_composed(*, seed: int, num_qubits: int,
                  num_segments: int = 5, segment_depth: int = 10, **_ignored):
    return generate_roi_composed_circuit(
        num_qubits=num_qubits,
        num_segments=num_segments,
        segment_depth=segment_depth,
        seed=seed,
    )

CIRCUIT_SOURCE_REGISTRY: Dict[str, Callable[..., Any]] = {
    "random_custom": _random_custom,
    "roi_composed": _roi_composed,
}

def get_circuit_source(name: str) -> Callable[..., Any]:
    if name not in CIRCUIT_SOURCE_REGISTRY:
        raise ValueError(
            f"Unknown circuit source '{name}'. "
            f"Available: {list(CIRCUIT_SOURCE_REGISTRY.keys())}"
        )
    return CIRCUIT_SOURCE_REGISTRY[name]


# ----------------------------
# Provider (future-proof)
# ----------------------------
class GeneratedCircuitProvider:
    """
    Provides circuits on-demand using a generator registered in CIRCUIT_SOURCE_REGISTRY.
    Handles per-sample randomness in a centralized place (e.g., two_qubit_ratio sampling).
    """
    def __init__(
        self,
        source_name: str,
        source_kwargs: Dict[str, Any],
        seed_base: int,
        two_qubit_bounds: Optional[tuple] = None,
    ):
        self.source_name = source_name
        self.source_kwargs = dict(source_kwargs)
        self.seed_base = int(seed_base)
        self.two_qubit_bounds = two_qubit_bounds
        self.fn = get_circuit_source(source_name)

    def get(self, idx: int):
        seed = self.seed_base + int(idx)
        kwargs = dict(self.source_kwargs)

        # Generator-specific per-sample behavior (kept here, not in Dataset)
        if self.source_name == "random_custom":
            if self.two_qubit_bounds is not None:
                low, high = self.two_qubit_bounds
                rng = np.random.RandomState(seed + 10000)
                kwargs["two_qubit_ratio"] = float(rng.uniform(low, high))
            else:
                kwargs.setdefault("two_qubit_ratio", 0.5)

        return self.fn(seed=seed, **kwargs)


def build_provider(circuit_source_cfg: Dict[str, Any], seed_base: int) -> GeneratedCircuitProvider:
    return GeneratedCircuitProvider(
        source_name=circuit_source_cfg["name"],
        source_kwargs=circuit_source_cfg.get("kwargs", {}),
        seed_base=seed_base,
        two_qubit_bounds=circuit_source_cfg.get("two_qubit_bounds", None),
    )