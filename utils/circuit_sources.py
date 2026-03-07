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

def _random_custom(
    *,
    seed: int,
    num_qubits: int,
    depth: int,
    gate_density: float,
    two_qubit_ratio: float = 0.5,
    use_barriers: bool = True,
    **_ignored,
):
    return generate_random_circuit_custom(
        num_qubits=num_qubits,
        depth=depth,
        gate_density=gate_density,
        seed=seed,
        two_qubit_ratio=two_qubit_ratio,
        use_barriers=use_barriers,
    )


def _roi_composed(*, seed: int, **kwargs):
    """
    Pass-through wrapper for the new ROI generator interface.

    All ROI-specific arguments should come from CIRCUIT_SOURCE_CFG["kwargs"].
    Example kwargs include:
      - num_qubits
      - num_layers
      - option
      - n_rois
      - twoq_to_oneq_ratio
      - idle_density
      - p_bridge_boundary
      - p_bridge_interior
      - noise_1q_prob / noise_2q_prob
      - measure_frac
      - block geometry / long-tall parameters
      - use_barriers
    """
    return generate_roi_composed_circuit(seed=seed, **kwargs)


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


def _sample_weighted_choice(weight_map: Dict[str, float], rng: np.random.RandomState) -> str:
    """
    Sample one key from a {name: weight} mapping.
    Weights do not need to sum to 1.0, but must contain at least one positive value.
    """
    if not isinstance(weight_map, dict) or len(weight_map) == 0:
        raise ValueError("option_mix must be a non-empty dict like {'op1': 0.1, 'op2a': 0.4, ...}")

    names = []
    weights = []
    for name, w in weight_map.items():
        w = float(w)
        if w > 0.0:
            names.append(str(name))
            weights.append(w)

    if not weights:
        raise ValueError("option_mix must contain at least one positive weight.")

    probs = np.asarray(weights, dtype=float)
    probs /= probs.sum()

    # numpy choice behaves more predictably with an index than object arrays
    idx = int(rng.choice(len(names), p=probs))
    return names[idx]


# ----------------------------
# Provider (future-proof)
# ----------------------------
class GeneratedCircuitProvider:
    """
    Provides circuits on-demand using a generator registered in CIRCUIT_SOURCE_REGISTRY.

    Handles per-sample randomness in a centralized place:
      - random_custom: samples two_qubit_ratio from two_qubit_bounds if requested
      - roi_composed: samples option from sampled_kwargs["option_mix"] if requested

    Notes:
      - The ROI mix is approximate over a dataset split because sampling is per-sample.
      - It is deterministic/reproducible for a given seed_base and idx.
    """
    def __init__(
        self,
        source_name: str,
        source_kwargs: Dict[str, Any],
        seed_base: int,
        two_qubit_bounds: Optional[tuple] = None,
        sampled_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.source_name = source_name
        self.source_kwargs = dict(source_kwargs)
        self.seed_base = int(seed_base)
        self.two_qubit_bounds = two_qubit_bounds
        self.sampled_kwargs = dict(sampled_kwargs or {})
        self.fn = get_circuit_source(source_name)

    def get(self, idx: int):
        seed = self.seed_base + int(idx)
        kwargs = dict(self.source_kwargs)

        # Use separate RNG streams for provider-side sampling so behavior stays reproducible
        # even if generator internals change later.
        provider_rng = np.random.RandomState(seed + 10000)

        # Generator-specific per-sample behavior
        if self.source_name == "random_custom":
            if self.two_qubit_bounds is not None:
                low, high = self.two_qubit_bounds
                kwargs["two_qubit_ratio"] = float(provider_rng.uniform(low, high))
            else:
                kwargs.setdefault("two_qubit_ratio", 0.5)

        elif self.source_name == "roi_composed":
            option_mix = self.sampled_kwargs.get("option_mix", None)
            if option_mix is not None:
                kwargs["option"] = _sample_weighted_choice(option_mix, provider_rng)

        return self.fn(seed=seed, **kwargs)


def build_provider(circuit_source_cfg: Dict[str, Any], seed_base: int) -> GeneratedCircuitProvider:
    return GeneratedCircuitProvider(
        source_name=circuit_source_cfg["name"],
        source_kwargs=circuit_source_cfg.get("kwargs", {}),
        seed_base=seed_base,
        two_qubit_bounds=circuit_source_cfg.get("two_qubit_bounds", None),
        sampled_kwargs=circuit_source_cfg.get("sampled_kwargs", {}),
    )