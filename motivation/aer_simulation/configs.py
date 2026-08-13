"""Config drift check.

`techs_v1.json`, shipped next to this module, is the SINGLE SOURCE OF TRUTH for
the technology table and the communication model. `hardware.py` hardcodes the
same values for speed. `drift_check()` asserts the two still agree, so the
harness cannot silently desync from the frozen table -- which would mean every
figure in the paper was produced against a machine that no longer exists on
paper.

Second, separate job: REPORT (never assert) how far the EFCL training configs
have drifted from the frozen table. They are deliberately behind -- adopting
`t_comm` in EFCL requires block-ASAP first, and any tech-table change forces a
retrain -- so the divergence must stay visible without failing the build.

    drift_check()                 # assert frozen table, report EFCL deltas
    efcl_deltas()                 # just the deltas, as data

Set MOSAIC_CONFIG_DIR to point at the directory holding `cost_config_*.json`.
"""

import json
import os

import numpy as np

from .hardware import TECHS, COMM

__all__ = ["FROZEN_PATH", "load_frozen", "drift_check", "efcl_deltas",
           "find_config", "CONFIG_SEARCH_PATH"]

FROZEN_PATH = os.path.join(os.path.dirname(__file__), "techs.json")

CONFIG_SEARCH_PATH = [
    os.environ.get("MOSAIC_CONFIG_DIR", ""),
    ".", "..", "./configs", "../configs", "/mnt/project",
]

# EFCL configs and which technologies each one defines.
_EFCL = [("cost_config_v3.json", ["sc", "na"]),
         ("cost_config_tp2n_99.json", ["sc", "ti"])]

_FIELDS = [("gate_fidelity", "f1q", "f1q"),
           ("gate_fidelity", "f2q", "f2q"),
           ("gate_time", "t1q", "t1q"),
           ("gate_time", "t2q", "t2q"),
           ("coherence", "T2", "T2")]


def load_frozen(path=FROZEN_PATH):
    with open(path) as fh:
        return json.load(fh)


def find_config(filename):
    for d in CONFIG_SEARCH_PATH:
        if not d:
            continue
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------------------
# 1. The assertion: hardware.py must match the frozen table
# ---------------------------------------------------------------------------

def _check_frozen(verbose=True):
    frozen = load_frozen()
    ft = frozen["techs"]

    assert set(ft) == set(TECHS), (
        f"technology set differs: frozen has {sorted(ft)}, hardware.py has {sorted(TECHS)}")

    for name, spec in ft.items():
        live = TECHS[name]
        for key in ("f1q", "f2q", "t1q", "t2q", "T2", "kappa", "max_qubits"):
            assert np.isclose(spec[key], getattr(live, key)), (
                f"DRIFT: {name}.{key} is {getattr(live, key)} in hardware.py but "
                f"{spec[key]} in {os.path.basename(FROZEN_PATH)} ({frozen['_id']}). "
                "Update BOTH and bump the frozen id.")
        assert spec["all_to_all"] == live.all_to_all, f"DRIFT: {name}.all_to_all"

    for key, val in frozen["comm"].items():
        if key.startswith("_"):
            continue
        assert key in COMM, f"frozen comm key '{key}' missing from hardware.COMM"
        if isinstance(val, bool):
            assert COMM[key] is val, f"DRIFT: COMM['{key}']"
        else:
            assert np.isclose(COMM[key], val), (
                f"DRIFT: COMM['{key}'] is {COMM[key]} but frozen says {val}")

    if verbose:
        print(f"frozen table OK: {frozen['_id']} (frozen {frozen['_frozen']}), "
              f"{len(ft)} technologies + comm model")
    return len(ft)


# ---------------------------------------------------------------------------
# 2. The report: how far the EFCL configs have drifted
# ---------------------------------------------------------------------------

def efcl_deltas():
    """[(config, tech, field, efcl_value, frozen_value), ...] for every mismatch.

    Returns [] when the configs are not on this machine. NEVER asserts: the EFCL
    side is deliberately behind and reconciling it forces a retrain.
    """
    out = []
    for fn, names in _EFCL:
        p = find_config(fn)
        if p is None:
            continue
        cfg = json.load(open(p))
        by = {t["name"].lower(): t for t in cfg.get("techs", [])}
        for nm in names:
            if nm not in by:
                continue
            for section, key, attr in _FIELDS:
                got = by[nm].get(section, {}).get(key)
                want = getattr(TECHS[nm], attr)
                if got is not None and not np.isclose(got, want):
                    out.append((fn, nm, key, got, want))
        for key in ("f_comm", "f_move"):
            got = cfg.get("comm", {}).get(key)
            if got is not None and not np.isclose(got, COMM[key]):
                out.append((fn, "-", key, got, COMM[key]))
        t_rem = cfg.get("comm", {}).get("t_remote")
        if t_rem is not None and not np.isclose(t_rem, 0.0):
            out.append((fn, "-", "t_remote", t_rem, 0.0))
    return out


def drift_check(verbose=True):
    """Assert hardware.py matches the frozen table; report EFCL divergence.

    Returns the number of technologies verified against the frozen table.
    """
    n = _check_frozen(verbose)

    deltas = efcl_deltas()
    if verbose:
        found = [fn for fn, _ in _EFCL if find_config(fn)]
        if not found:
            print("  EFCL configs not on this machine -- divergence report skipped")
        elif not deltas:
            print(f"  EFCL configs ({', '.join(found)}) agree with the frozen table")
        else:
            print(f"  EFCL divergence -- {len(deltas)} field(s), EXPECTED, "
                  "reconcile at Phase 2 (retrain):")
            for fn, tech, key, got, want in deltas:
                print(f"    {fn:<28} {tech:>3}.{key:<9} "
                      f"EFCL={got:<12g} frozen={want:g}")
    return n
