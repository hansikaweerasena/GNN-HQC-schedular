#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

csv_path = Path("/mnt/data/m1_new_100_idle_rates.csv")
outdir = Path("/mnt/data/m1_fixed_idle_plots_redtri_v2")
outdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)
idle_col = "idle_rate"

df["fid_gain_vs_sc"] = ((df["f_SCNA"] - df["f_2xSC"]) / (1.0 - df["f_2xSC"])) * 100.0
df["fid_gain_vs_na"] = ((df["f_SCNA"] - df["f_2xNA"]) / (1.0 - df["f_2xNA"])) * 100.0
df["mk_gain_vs_sc"] = ((df["mk_2xSC"] - df["mk_SCNA"]) / df["mk_2xSC"]) * 100.0
df["mk_gain_vs_na"] = ((df["mk_2xNA"] - df["mk_SCNA"]) / df["mk_2xNA"]) * 100.0

if df[idle_col].max() > 1.5:
    df["idle_rate_norm"] = df[idle_col] / 100.0
else:
    df["idle_rate_norm"] = df[idle_col]

def idle_group(v):
    if v < 0.45:
        return "low"
    elif v <= 0.60:
        return "medium"
    else:
        return "high"

df["idle_group"] = df["idle_rate_norm"].apply(idle_group)

GROUP_STYLE = {
    "low":    {"color": "#8B4A3A", "marker": "^", "label": "Low idle (<45%)"},
    "medium": {"color": "#7FA7C9", "marker": "o", "label": "Medium idle (45%–60%)"},
    "high":   {"color": "#8B4A3A", "marker": "x", "label": "High idle (>60%)"},
}
GUIDE_GRAY = "#B8B8B8"
HQC_REGION_YELLOW = "#FFF4CC"

def make_plot(x_col, y_col, x_label, y_label, output_path, shade_upper_right=False):
    fig = plt.figure(figsize=(16, 7.2))
    ax = fig.add_axes([0.08, 0.14, 0.88, 0.80])

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
        ax.fill_between([0, xmax], 0, ymax, color=HQC_REGION_YELLOW, alpha=0.60, zorder=0)

    for key in ["low", "medium", "high"]:
        sub = df[df["idle_group"] == key]
        style = GROUP_STYLE[key]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub[x_col], sub[y_col], s=60, marker=style["marker"],
            color=style["color"], alpha=0.90,
            linewidths=1.2 if style["marker"] == "x" else 0.8,
            edgecolors=style["color"] if style["marker"] != "x" else None,
            label=f"{style['label']}, n={len(sub)}", zorder=3
        )

    ax.axvline(0.0, linestyle="--", linewidth=1.5, color=GUIDE_GRAY, zorder=2)
    ax.axhline(0.0, linestyle="--", linewidth=1.5, color=GUIDE_GRAY, zorder=2)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.12, zorder=1)
    ax.legend(title="Overall idle rate (1−L)", fontsize=12, title_fontsize=12, frameon=True, loc="best")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

make_plot("fid_gain_vs_sc", "fid_gain_vs_na",
          "Relative infidelity reduction vs 2×SC (%)",
          "Relative infidelity reduction vs 2×NA (%)",
          outdir / "m1_fidelity_gain_idle_fixed.png", True)

make_plot("mk_gain_vs_sc", "mk_gain_vs_na",
          "Makespan reduction vs 2×SC (%)",
          "Makespan reduction vs 2×NA (%)",
          outdir / "m1_makespan_gain_idle_fixed.png", False)
