#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {
    "cut",
    "f_2xSC", "f_2xNA", "f_SCNA",
    "mk_2xSC", "mk_2xNA", "mk_SCNA",
}

# Earthy, darker point colors.
GROUP_STYLE = {
    "small":  {"color": "#3F5F45", "marker": "^"},  # earthy dark green, triangle
    "medium": {"color": "#355C7D", "marker": "o"},  # earthy dark blue, circle
    "high":   {"color": "#8B4A3A", "marker": "x"},  # reddish brown, x
}

GUIDE_GRAY = "#B8B8B8"      # light gray dashed quadrant lines
HQC_REGION_YELLOW = "#FFF4CC"  # light yellow fidelity-win region


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            "CSV is missing required columns: " + ", ".join(sorted(missing))
        )

    # Relative infidelity reduction. Positive => SC+NA has higher fidelity.
    df["fid_gain_vs_sc"] = (
        (df["f_SCNA"] - df["f_2xSC"]) / (1.0 - df["f_2xSC"]) * 100.0
    )
    df["fid_gain_vs_na"] = (
        (df["f_SCNA"] - df["f_2xNA"]) / (1.0 - df["f_2xNA"]) * 100.0
    )

    # Relative makespan reduction. Positive => SC+NA is faster.
    df["mk_gain_vs_sc"] = (
        (df["mk_2xSC"] - df["mk_SCNA"]) / df["mk_2xSC"] * 100.0
    )
    df["mk_gain_vs_na"] = (
        (df["mk_2xNA"] - df["mk_SCNA"]) / df["mk_2xNA"] * 100.0
    )
    return df


def add_cut_groups(df, small_max=3, medium_max=9):
    df = df.copy()

    def classify(cut):
        if cut <= small_max:
            return "small"
        if cut <= medium_max:
            return "medium"
        return "high"

    df["cut_group"] = df["cut"].apply(classify)
    return df


def scatter_by_cut(
    df,
    x_col,
    y_col,
    x_label,
    y_label,
    output_path,
    small_max=3,
    medium_max=9,
    shade_upper_right=False,
):
    # Previous height was 9; reduced by 20% => 7.2.
    fig, ax = plt.subplots(figsize=(16, 7.2))

    xrange = df[x_col].max() - df[x_col].min()
    yrange = df[y_col].max() - df[y_col].min()
    xpad = max(2.0, 0.05 * xrange)
    ypad = max(2.0, 0.05 * yrange)

    xmin = min(df[x_col].min() - xpad, -xpad)
    xmax = max(df[x_col].max() + xpad, xpad)
    ymin = min(df[y_col].min() - ypad, -ypad)
    ymax = max(df[y_col].max() + ypad, ypad)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if shade_upper_right:
        ax.fill_between(
            [0, xmax],
            0,
            ymax,
            color=HQC_REGION_YELLOW,
            alpha=0.60,
            zorder=0,
        )

    groups = [
        ("small", f"Small (0-{small_max})"),
        ("medium", f"Medium ({small_max + 1}-{medium_max})"),
        ("high", f"High ({medium_max + 1}+)"),
    ]

    # 3x the previous marker area (20 -> 60).
    for key, label in groups:
        sub = df[df["cut_group"] == key]
        if not sub.empty:
            style = GROUP_STYLE[key]
            ax.scatter(
                sub[x_col],
                sub[y_col],
                s=60,
                marker=style["marker"],
                color=style["color"],
                alpha=0.90,
                linewidths=1.2 if style["marker"] == "x" else 0.8,
                edgecolors=style["color"] if style["marker"] != "x" else None,
                label=f"{label}, n={len(sub)}",
                zorder=3,
            )

    # Light-gray dashed zero-reference lines.
    ax.axvline(0.0, linestyle="--", linewidth=1.5, color=GUIDE_GRAY, zorder=2)
    ax.axhline(0.0, linestyle="--", linewidth=1.5, color=GUIDE_GRAY, zorder=2)

    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.12, zorder=1)

    ax.legend(
        title="Inter-QPU gate counts",
        fontsize=12,
        title_fontsize=12,
        frameon=True,
        loc="best",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--outdir", type=Path, default=Path("."))
    parser.add_argument("--cut-small-max", type=int, default=3)
    parser.add_argument("--cut-medium-max", type=int, default=9)
    args = parser.parse_args()

    if args.cut_medium_max <= args.cut_small_max:
        raise ValueError("--cut-medium-max must exceed --cut-small-max")

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = add_cut_groups(
        load_results(args.csv),
        small_max=args.cut_small_max,
        medium_max=args.cut_medium_max,
    )

    scatter_by_cut(
        df,
        "fid_gain_vs_sc",
        "fid_gain_vs_na",
        "Relative infidelity reduction vs 2×SC (%)",
        "Relative infidelity reduction vs 2×NA (%)",
        args.outdir / "m1_fidelity_gain.png",
        args.cut_small_max,
        args.cut_medium_max,
        shade_upper_right=True,
    )

    scatter_by_cut(
        df,
        "mk_gain_vs_sc",
        "mk_gain_vs_na",
        "Makespan reduction vs 2×SC (%)",
        "Makespan reduction vs 2×NA (%)",
        args.outdir / "m1_makespan_gain.png",
        args.cut_small_max,
        args.cut_medium_max,
        shade_upper_right=False,
    )

    print(f"Loaded {len(df)} circuits")
    print(f"Wrote {args.outdir / 'm1_fidelity_gain.png'}")
    print(f"Wrote {args.outdir / 'm1_makespan_gain.png'}")


if __name__ == "__main__":
    main()
