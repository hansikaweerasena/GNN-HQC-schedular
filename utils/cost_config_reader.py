import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import importlib
import os

def load_scheduler_cfg(module_path: str):
    m = importlib.import_module(module_path)
    return m.MODEL_CFG, m.CLUSTER_CFG, m.TRAIN_CFG, m.DATASET_CFG, m.CIRCUIT_SOURCE_CFG


def _default_data_dir() -> Path:
    # src/ -> project_root/ ; project_root/data/
    return Path(__file__).resolve().parent.parent / "data"


def _apply_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(cfg)  # shallow copy

    cfg.setdefault("comm", {})
    cfg["comm"].setdefault("f_comm", 1.0)
    cfg["comm"].setdefault("f_move", 1.0)
    cfg["comm"].setdefault("t_remote", 0.0)

    cfg.setdefault("timing", {})
    cfg["timing"].setdefault("delta", 1.0)

    cfg.setdefault("gate_names", {})
    cfg["gate_names"].setdefault("measure", ["measure", "meas", "m"])

    cfg.setdefault("connectivity_proxy", {})
    cfg["connectivity_proxy"].setdefault("mode", "none")

    return cfg


def _validate(cfg: Dict[str, Any]) -> None:
    if "techs" not in cfg or not isinstance(cfg["techs"], list) or len(cfg["techs"]) == 0:
        raise ValueError("Config must contain non-empty list: cfg['techs'].")

    for i, t in enumerate(cfg["techs"]):
        if not isinstance(t, dict):
            raise ValueError(f"techs[{i}] must be a dict.")

        if "name" not in t:
            raise ValueError(f"techs[{i}] missing required key 'name'.")

        # --- Nested schema validation ---
        gf = t.get("gate_fidelity", {})
        coh = t.get("coherence", {})
        routing = t.get("routing", {})
        gt = t.get("gate_time", {})  # optional

        # Required: gate fidelities (success probabilities)
        for key in ["f1q", "f2q", "fm"]:
            if key not in gf:
                raise ValueError(f"techs[{i}] missing required key gate_fidelity.{key}.")
            val = float(gf[key])
            if not (0.0 < val <= 1.0):
                raise ValueError(
                    f"techs[{i}].gate_fidelity.{key} must be in (0, 1]. Got {val}."
                )

        # Required: coherence.T2
        if "T2" not in coh:
            raise ValueError(f"techs[{i}] missing required key coherence.T2.")
        T2 = float(coh["T2"])
        if T2 <= 0.0:
            raise ValueError(f"techs[{i}].coherence.T2 must be > 0. Got {T2}.")

        # Required: routing.rho
        if "rho" not in routing:
            raise ValueError(f"techs[{i}] missing required key routing.rho.")
        rho = float(routing["rho"])
        if rho < 0.0:
            raise ValueError(f"techs[{i}].routing.rho must be >= 0. Got {rho}.")

        # Optional: gate_time.* if present must be nonnegative
        for tk in ["t1q", "t2q", "tm"]:
            if tk in gt and gt[tk] is not None:
                if float(gt[tk]) < 0.0:
                    raise ValueError(f"techs[{i}].gate_time.{tk} must be >= 0. Got {gt[tk]}.")

    # comm defaults already applied, but validate range if present
    cfg.setdefault("comm", {})
    for key in ["f_comm", "f_move"]:
        val = float(cfg["comm"].get(key, 1.0))
        if not (0.0 < val <= 1.0):
            raise ValueError(f"comm.{key} must be in (0, 1]. Got {val}.")

    # timing.delta
    cfg.setdefault("timing", {})
    delta = float(cfg["timing"].get("delta", 1.0))
    if delta < 0.0:
        raise ValueError(f"timing.delta must be >= 0. Got {delta}.")

    # gate_names.measure
    meas = cfg.get("gate_names", {}).get("measure", [])
    if not isinstance(meas, list) or len(meas) == 0:
        raise ValueError("gate_names.measure must be a non-empty list of strings.")


def load_cost_config(
    filename: str,
    *,
    data_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Load cost model config JSON.

    - If `filename` is absolute: read it directly.
    - If relative: resolve under `data_dir` (defaults to ../data relative to src/).
    - If filename has no suffix, '.json' is appended.
    """
    filename = os.path.join(os.path.dirname(__file__), "..", "configs", filename)
    p = Path(filename)

    if p.suffix == "":
        p = p.with_suffix(".json")

    if not p.is_absolute():
        base = Path(data_dir) if data_dir is not None else _default_data_dir()
        p = (base / p).resolve()

    if not p.exists():
        raise FileNotFoundError(f"Cost config not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg = _apply_defaults(cfg)
    _validate(cfg)
    return cfg