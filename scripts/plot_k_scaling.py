#!/usr/bin/env python3
"""
Generate the K-scaling EFCL figure for proposed paper.
4 panels: (a) SC+NA+TI, (b) SC+TI+ES, (c) NA+TI+ES, (d) SC+NA+TI+ES
Each panel: 2 groups (Synthetic, MQT) × 3 bars (proposed, Best Greedy, Best DQC)
Log-scale y-axis.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Publication style ──
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "lines.linewidth": 0.8,
    "patch.linewidth": 0.5,
})

# ── Data ──
# Baseline mapping: B1=static greedy, B2=sticky greedy (old B3),
#                   B3=Wu beam (old B4), B4=Burt FM (old B5)

profiles = [
    {
        "title": "(a) SC+NA+TI",
        "k": 3,
        "synth": {
            "proposed": 0.182,
            "best_greedy": 5.653, "greedy_label": "B1",
            "best_dqc": 5.329, "dqc_label": "B4",
        },
        "mqt": {
            "proposed": 0.168,
            "best_greedy": 1.442, "greedy_label": "B1",
            "best_dqc": 2.431, "dqc_label": "B4",
        },
    },
    {
        "title": "(b) SC+TI+ES",
        "k": 3,
        "synth": {
            "proposed": 0.179,
            "best_greedy": 2.644, "greedy_label": "B1",
            "best_dqc": 6.565, "dqc_label": "B4",
        },
        "mqt": {
            "proposed": 0.172,
            "best_greedy": 2.619, "greedy_label": "B2",
            "best_dqc": 4.235, "dqc_label": "B3",
        },
    },
    {
        "title": "(c) NA+TI+ES",
        "k": 3,
        "synth": {
            "proposed": 0.170,
            "best_greedy": 1.109, "greedy_label": "B1",
            "best_dqc": 0.822, "dqc_label": "B4",
        },
        "mqt": {
            "proposed": 0.158,
            "best_greedy": 0.468, "greedy_label": "B1",
            "best_dqc": 0.481, "dqc_label": "B4",
        },
    },
    {
        "title": "(d) SC+NA+TI+ES",
        "k": 4,
        "synth": {
            "proposed": 0.224,
            "best_greedy": 5.206, "greedy_label": "B2",
            "best_dqc": 4.514, "dqc_label": "B3",
        },
        "mqt": {
            "proposed": 0.183,
            "best_greedy": 2.423, "greedy_label": "B1",
            "best_dqc": 2.797, "dqc_label": "B3",
        },
    },
]

# ── Colors ──
# c_proposed = "#2563EB"   # blue
# c_greedy = "#D97706"   # amber
# c_dqc    = "#7C3AED"   # purple

c_proposed = "#556B2F"   # deep olive
c_greedy   = "#C96A3D"   # terracotta
c_dqc      = "#D8C3A5"   # warm sand

c_proposed = "#4F5D2F"   # forest brown-green
c_greedy   = "#B85C38"   # clay rust
c_dqc      = "#CBB89D"   # muted beige

# ── Figure ──
fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9), sharey=True)

bar_width = 0.22
group_positions = np.array([0, 1.0])  # Synthetic, MQT
group_labels = ["Synthetic", "MQT"]

for ax, prof in zip(axes, profiles):
    synth = prof["synth"]
    mqt = prof["mqt"]

    vals_proposed = [synth["proposed"], mqt["proposed"]]
    vals_greedy = [synth["best_greedy"], mqt["best_greedy"]]
    vals_dqc    = [synth["best_dqc"], mqt["best_dqc"]]

    greedy_labels = [synth["greedy_label"], mqt["greedy_label"]]
    dqc_labels    = [synth["dqc_label"], mqt["dqc_label"]]

    x = group_positions

    bars_m = ax.bar(x - bar_width, vals_proposed, bar_width,
                    color=c_proposed, edgecolor="black", linewidth=0.4,
                    label="Proposed", zorder=3)
    bars_g = ax.bar(x, vals_greedy, bar_width,
                    color=c_greedy, edgecolor="black", linewidth=0.4,
                    label="Best of B1, B2", zorder=3)
    bars_d = ax.bar(x + bar_width, vals_dqc, bar_width,
                    color=c_dqc, edgecolor="black", linewidth=0.4,
                    label="Best of B3, B4", zorder=3)

    # Label which baseline won above each bar
    for i, (bg, bd) in enumerate(zip(bars_g, bars_d)):
        ax.text(bg.get_x() + bg.get_width() / 2, bg.get_height() * 1.15,
                greedy_labels[i], ha="center", va="bottom", fontsize=6,
                color=c_greedy, fontweight="bold")
        ax.text(bd.get_x() + bd.get_width() / 2, bd.get_height() * 1.15,
                dqc_labels[i], ha="center", va="bottom", fontsize=6,
                color=c_dqc, fontweight="bold")

    # Margin annotation for synthetic
    best_base_synth = min(synth["best_greedy"], synth["best_dqc"])
    margin_synth = best_base_synth / synth["proposed"]
    ax.text(x[0] - bar_width, synth["proposed"] * 0.55,
            f"{margin_synth:.0f}×", ha="center", va="top", fontsize=6.5,
            color=c_proposed, fontweight="bold")

    # Margin annotation for MQT
    best_base_mqt = min(mqt["best_greedy"], mqt["best_dqc"])
    margin_mqt = best_base_mqt / mqt["proposed"]
    ax.text(x[1] - bar_width, mqt["proposed"] * 0.55,
            f"{margin_mqt:.0f}×", ha="center", va="top", fontsize=6.5,
            color=c_proposed, fontweight="bold")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_title(prof["title"], fontsize=8.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    # y-axis limits
    ax.set_ylim(0.05, 15)

# y-label on leftmost panel only
axes[0].set_ylabel("EFCL (log scale)")

# Shared legend at top
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3,
           frameon=True, fancybox=False, edgecolor="gray",
           bbox_to_anchor=(0.5, 1.02), fontsize=7.5)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("k_scaling_efcl.pdf", bbox_inches="tight")
plt.savefig("k_scaling_efcl.png", bbox_inches="tight", dpi=300)
print("Saved k_scaling_efcl.pdf and k_scaling_efcl.png")
