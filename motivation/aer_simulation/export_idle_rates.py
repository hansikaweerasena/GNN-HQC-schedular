"""Append slot-based liveness and idle rate to an M1 result CSV.

The circuit generator is deterministic, so this utility rebuilds each circuit
from the saved ``family``, ``seed`` and ``depth`` fields.  It does not run Aer,
enumerate partitions, or rescore any machine.

Example
-------
python -m aer_simulation.export_idle_rates \
    --input m1_new_100.csv \
    --out m1_new_100_idle_rates.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import statistics
import tempfile

from .families import generate


DERIVED_COLUMNS = ["liveness", "idle_rate"]


def slot_liveness(lc):
    """Return liveness and idle rate over the qubit-by-layer slot grid."""
    active_slots = 0

    for layer_index, layer in enumerate(lc.layers):
        active_qubits = set()
        for gate in layer:
            if gate[0] == "1q":
                active_qubits.add(gate[1])
            elif gate[0] == "2q":
                active_qubits.add(gate[1])
                active_qubits.add(gate[2])
            else:
                raise ValueError(
                    f"unsupported gate kind {gate[0]!r} in layer {layer_index}"
                )
        active_slots += len(active_qubits)

    total_slots = lc.n_qubits * lc.depth
    if total_slots <= 0:
        raise ValueError(
            f"cannot calculate liveness for {lc.n_qubits} qubits at depth {lc.depth}"
        )

    liveness = active_slots / total_slots
    idle_rate = 1.0 - liveness

    # Independent checks against the LayeredCircuit helpers.  The activity sum
    # equals the explicit set-based count because a valid layer cannot use the
    # same qubit twice.
    activity_slots = sum(lc.activity().values())
    if active_slots != activity_slots:
        raise AssertionError(
            f"active-slot mismatch: explicit={active_slots}, activity={activity_slots}"
        )
    if not math.isclose(idle_rate, lc.idle_fraction(), abs_tol=1e-12):
        raise AssertionError(
            f"idle-rate mismatch: explicit={idle_rate}, helper={lc.idle_fraction()}"
        )
    if not math.isclose(liveness + idle_rate, 1.0, abs_tol=1e-12):
        raise AssertionError("liveness and idle rate do not sum to one")
    if not (0.0 <= liveness <= 1.0 and 0.0 <= idle_rate <= 1.0):
        raise AssertionError(
            f"rate outside [0, 1]: liveness={liveness}, idle_rate={idle_rate}"
        )

    return liveness, idle_rate


def _require_equal(label, saved, regenerated, key):
    if saved != regenerated:
        raise ValueError(
            f"{label}: regenerated {key}={regenerated!r}, saved value is {saved!r}"
        )


def _require_close(label, saved, regenerated, key, tolerance):
    if not math.isclose(saved, regenerated, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(
            f"{label}: regenerated {key}={regenerated!r}, saved value is {saved!r}"
        )


def validate_regeneration(row, lc, meta, profile):
    """Check that a regenerated circuit matches its saved experiment row."""
    label = f"family={row['family']} seed={row['seed']} depth={row['depth']}"
    _require_equal(label, int(row["depth"]), lc.depth, "depth")
    _require_equal(label, int(row["n_2q"]), lc.n_2q, "n_2q")

    if row.get("n_hot", "") != "":
        _require_equal(label, int(row["n_hot"]), meta.get("n_hot"), "n_hot")
    if row.get("p_cold", "") != "":
        _require_close(
            label, float(row["p_cold"]), float(meta.get("p_cold")), "p_cold", 5e-4
        )
    if row.get("mean_duty", "") != "":
        _require_close(
            label,
            float(row["mean_duty"]),
            profile.mean_duty,
            "mean_duty",
            5e-5,
        )
    if row.get("duty_spread", "") != "":
        _require_close(
            label,
            float(row["duty_spread"]),
            profile.duty_spread,
            "duty_spread",
            5e-5,
        )
    if row.get("contested", "") != "":
        _require_equal(
            label, bool(int(row["contested"])), bool(profile.contested), "contested"
        )
    if row.get("cut", "") != "":
        _require_equal(label, int(row["cut"]), profile.n_cross_best, "cut")


def append_idle_rates(input_path, output_path, force=False):
    """Regenerate circuits and append liveness/idle-rate columns to a CSV."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"input CSV not found: {input_path}")
    if output_path.exists() and not force:
        raise FileExistsError(
            f"output already exists: {output_path}; pass --force to replace it"
        )
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must be different")

    with input_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"input CSV has no header: {input_path}")
        required = {"seed", "family", "depth", "n_2q"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"input CSV is missing required columns: {sorted(missing)}")
        if any(column in reader.fieldnames for column in DERIVED_COLUMNS):
            raise ValueError(
                f"input CSV already contains one of {DERIVED_COLUMNS}; use the original M1 CSV"
            )
        fieldnames = list(reader.fieldnames) + DERIVED_COLUMNS
        rows = list(reader)

    if not rows:
        raise ValueError(f"input CSV has no data rows: {input_path}")

    seen = set()
    output_rows = []
    liveness_values = []
    idle_values = []
    by_n_hot = {}

    for row in rows:
        seed = int(row["seed"])
        family = row["family"]
        depth = int(row["depth"])
        identity = (family, seed, depth)
        if identity in seen:
            raise ValueError(f"duplicate circuit row: {identity}")
        seen.add(identity)

        lc, meta, profile = generate(family, seed, depth=depth)
        validate_regeneration(row, lc, meta, profile)
        liveness, idle_rate = slot_liveness(lc)

        enriched = dict(row)
        enriched["liveness"] = f"{liveness:.6f}"
        enriched["idle_rate"] = f"{idle_rate:.6f}"
        output_rows.append(enriched)
        liveness_values.append(liveness)
        idle_values.append(idle_rate)
        if row.get("n_hot", "") != "":
            by_n_hot.setdefault(int(row["n_hot"]), []).append(idle_rate)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as target:
            temp_name = target.name
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
        os.replace(temp_name, output_path)
        temp_name = None
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)

    print(f"wrote {output_path} ({len(output_rows)} rows)")
    print(
        "liveness: "
        f"mean={statistics.fmean(liveness_values):.6f} "
        f"median={statistics.median(liveness_values):.6f} "
        f"min={min(liveness_values):.6f} max={max(liveness_values):.6f}"
    )
    print(
        "idle_rate: "
        f"mean={statistics.fmean(idle_values):.6f} "
        f"median={statistics.median(idle_values):.6f} "
        f"min={min(idle_values):.6f} max={max(idle_values):.6f}"
    )
    for n_hot in sorted(by_n_hot):
        values = by_n_hot[n_hot]
        print(
            f"n_hot={n_hot}: n={len(values)} "
            f"mean_idle={statistics.fmean(values):.6f} "
            f"median_idle={statistics.median(values):.6f}"
        )

    return output_rows


def main():
    parser = argparse.ArgumentParser(
        description="Append deterministic slot liveness and idle rate to an M1 CSV."
    )
    parser.add_argument("--input", required=True, help="existing M1 result CSV")
    parser.add_argument("--out", required=True, help="new enriched CSV path")
    parser.add_argument(
        "--force", action="store_true", help="replace --out if it already exists"
    )
    args = parser.parse_args()
    append_idle_rates(args.input, args.out, force=args.force)


if __name__ == "__main__":
    main()
