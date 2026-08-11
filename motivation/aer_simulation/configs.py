"""Config drift check (extracted from NB2 section 5).

The tech table is hardcoded in `hardware.py` for speed. This module re-reads the
EFCL cost configs *if present* and asserts the hardcoded values still match, so
the harness cannot silently desync from the training-side source of truth.
Skips cleanly when the configs are not on this machine.

Set MOSAIC_CONFIG_DIR to point at the directory holding cost_config_*.json.
"""

import json
import os

import numpy as np

from .hardware import TECHS, COMM

__all__ = ["find_config", "drift_check", "CONFIG_SEARCH_PATH"]

CONFIG_SEARCH_PATH = [
    os.environ.get("MOSAIC_CONFIG_DIR", ""),
    ".", "..", "./configs", "../configs", "/mnt/project",
]

_PAIRS = [("cost_config_v3.json", ["sc", "na"]),
          ("cost_config_tp2n_99.json", ["sc", "ti"])]


def find_config(filename):
    for d in CONFIG_SEARCH_PATH:
        if not d:
            continue
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return None


def drift_check(verbose=True):
    """Assert the hardcoded TECHS/COMM still match the EFCL configs.

    Returns the number of tech-configs verified (0 = configs not found, skipped).
    """
    checked = 0
    for fn, names in _PAIRS:
        p = find_config(fn)
        if p is None:
            if verbose:
                print(f"  {fn}: not found, skipping")
            continue
        cfg = json.load(open(p))
        by = {t["name"].lower(): t for t in cfg["techs"]}
        for nm in names:
            c, t = by[nm], TECHS[nm]
            assert np.isclose(c["gate_fidelity"]["f1q"], t.f1q), f"{nm} f1q drift"
            assert np.isclose(c["gate_fidelity"]["f2q"], t.f2q), f"{nm} f2q drift"
            assert np.isclose(c["coherence"]["T2"], t.T2), f"{nm} T2 drift"
            assert np.isclose(c["gate_time"]["t2q"], t.t2q), f"{nm} t2q drift"
            checked += 1
        assert np.isclose(cfg["comm"]["f_comm"], COMM["f_comm"]), "f_comm drift"
        assert np.isclose(cfg["comm"]["f_move"], COMM["f_move"]), "f_move drift"

        # DELIBERATE DIVERGENCE, not drift. EFCL still runs t_remote = 0 while this
        # harness uses a nonzero per-pair t_comm. Adding t_comm to EFCL requires
        # block-ASAP -- under a global per-layer clock it would charge every idle
        # qubit the full remote-gate duration -- and therefore a full retrain. That
        # decision is gated on the Phase-2 divergence measurement. Assert the config
        # is still 0 so the divergence stays intentional and visible.
        assert np.isclose(cfg["comm"].get("t_remote", 0.0), 0.0), (
            "cost_config t_remote is no longer 0 -- EFCL and the Aer harness have both "
            "changed. Reconcile deliberately: t_comm in EFCL is unsafe without block-ASAP.")
        if verbose:
            print(f"  {fn}: OK ({', '.join(names)})  "
                  f"[t_remote=0 in EFCL: intentional divergence]")

    if verbose:
        print(f"drift-check passed ({checked} tech-configs verified)" if checked
              else "drift-check skipped (no configs on this machine)")
    return checked
