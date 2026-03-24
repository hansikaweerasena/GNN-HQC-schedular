#!/usr/bin/env python3
"""
Analyze effective layer counts produced by the same circuit-generation path used in
train_hipergator.py / train_hipergator_true_batched.py.

It builds circuits from the configured provider, runs CircuitRepresentation, and
reports how many layers remain after Qiskit/CircuitRepresentation layering
(empty layers dropped, merges, etc.). It can also save a histogram.

Example:
    python analyze_layer_counts.py \
        --sched_cfg configs.scheduler_config \
        --split train \
        --max_samples 800 \
        --save_hist results/layer_hist_train.png
"""

import argparse
import os
import sys
from collections import Counter
from statistics import mean, median
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Try a few common project-root candidates so the script can be dropped into
# scripts/ or run from the project root without edits.
_here = os.path.abspath(os.path.dirname(__file__))
_candidates = [
    _here,
    os.path.abspath(os.path.join(_here, "..")),
    os.path.abspath(os.path.join(_here, "..", "..")),
    os.getcwd(),
    os.path.abspath(os.path.join(os.getcwd(), "..")),
]
for _p in _candidates:
    if _p not in sys.path:
        sys.path.append(_p)

from utils.circuit_sources import build_provider
from src.circuit_representation import CircuitRepresentation
from utils.cost_config_reader import load_scheduler_cfg


def parse_args():
    p = argparse.ArgumentParser(description="Analyze effective layer counts for generated circuits")
    p.add_argument("--sched_cfg", type=str, default="configs.scheduler_config",
                   help="Module path for scheduler config (dotted import)")
    p.add_argument("--split", type=str, default="train", choices=["train", "test", "both"],
                   help="Which dataset split seed/count to analyze")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Optional cap on number of samples per split")
    p.add_argument("--save_hist", type=str, default=None,
                   help="Optional path to save histogram PNG")
    p.add_argument("--print_counts", action="store_true",
                   help="Print exact histogram counts in text")
    return p.parse_args()


def get_requested_num_layers(circuit_source_cfg) -> Optional[int]:
    """Best-effort read of requested num_layers from provider config."""
    if not isinstance(circuit_source_cfg, dict):
        return None

    # Common direct layout
    provider_kwargs = circuit_source_cfg.get("provider_kwargs", {})
    if isinstance(provider_kwargs, dict) and "num_layers" in provider_kwargs:
        return int(provider_kwargs["num_layers"])

    # Common nested layout used for generator-specific kwargs
    generator_kwargs = circuit_source_cfg.get("generator_kwargs", {})
    if isinstance(generator_kwargs, dict) and "num_layers" in generator_kwargs:
        return int(generator_kwargs["num_layers"])

    # Mixed-source layout: look inside each source entry and use the first consistent value
    source_mix = circuit_source_cfg.get("source_mix", None)
    if isinstance(source_mix, list):
        vals = []
        for item in source_mix:
            if not isinstance(item, dict):
                continue
            for key in ("provider_kwargs", "generator_kwargs", "kwargs"):
                kw = item.get(key, {})
                if isinstance(kw, dict) and "num_layers" in kw:
                    vals.append(int(kw["num_layers"]))
        if vals and len(set(vals)) == 1:
            return vals[0]
    return None


def summarize_lengths(name, lengths, requested_num_layers: Optional[int], print_counts: bool = False):
    print("\n" + "=" * 72)
    print(f"{name.upper()} SPLIT")
    print("=" * 72)

    n = len(lengths)
    mn = min(lengths)
    mx = max(lengths)
    avg = mean(lengths)
    med = median(lengths)
    print(f"samples              : {n}")
    print(f"effective layer mean : {avg:.3f}")
    print(f"effective layer med. : {med:.3f}")
    print(f"effective layer min  : {mn}")
    print(f"effective layer max  : {mx}")

    if requested_num_layers is not None:
        dropped = [requested_num_layers - x for x in lengths]
        print(f"requested num_layers : {requested_num_layers}")
        print(f"mean dropped layers  : {mean(dropped):.3f}")
        print(f"max dropped layers   : {max(dropped)}")
        print(f"pct exact requested  : {100.0 * sum(x == requested_num_layers for x in lengths) / n:.2f}%")
        print(f"pct lose >=1 layer   : {100.0 * sum(x < requested_num_layers for x in lengths) / n:.2f}%")
        print(f"pct lose >=2 layers  : {100.0 * sum(x <= requested_num_layers - 2 for x in lengths) / n:.2f}%")
        print(f"pct lose >=3 layers  : {100.0 * sum(x <= requested_num_layers - 3 for x in lengths) / n:.2f}%")

    if print_counts:
        ctr = Counter(lengths)
        print("\nHistogram counts:")
        for k in sorted(ctr):
            print(f"  {k:>4d} : {ctr[k]}")


def plot_hist(train_lengths=None, test_lengths=None, requested_num_layers=None, save_path=None):
    plt.figure(figsize=(8, 5))

    all_lengths = []
    if train_lengths:
        all_lengths.extend(train_lengths)
    if test_lengths:
        all_lengths.extend(test_lengths)
    if not all_lengths:
        return

    bins = list(range(min(all_lengths), max(all_lengths) + 2))

    if train_lengths:
        plt.hist(train_lengths, bins=bins, alpha=0.65, label="train", edgecolor="black")
    if test_lengths:
        plt.hist(test_lengths, bins=bins, alpha=0.65, label="test", edgecolor="black")

    if requested_num_layers is not None:
        plt.axvline(requested_num_layers, linestyle="--", linewidth=1.5, label=f"requested={requested_num_layers}")

    plt.xlabel("Effective layer count after representation/layering")
    plt.ylabel("Number of circuits")
    plt.title("Effective layer-count distribution")
    if train_lengths and test_lengths:
        plt.legend()
    elif requested_num_layers is not None:
        plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=160)
        print(f"\nSaved histogram to: {save_path}")


def collect_lengths(provider, n_samples: int):
    lengths = []
    for idx in range(n_samples):
        qc = provider.get(idx)
        rep = CircuitRepresentation(qc)
        lengths.append(len(rep.layers))
    return lengths


def main():
    args = parse_args()

    MODEL_CFG, CLUSTER_CFG, TRAIN_CFG, DATASET_CFG, CIRCUIT_SOURCE_CFG = load_scheduler_cfg(args.sched_cfg)
    requested_num_layers = get_requested_num_layers(CIRCUIT_SOURCE_CFG)

    split_names = [args.split] if args.split in {"train", "test"} else ["train", "test"]

    train_lengths = None
    test_lengths = None

    if "train" in split_names:
        train_provider = build_provider(CIRCUIT_SOURCE_CFG, seed_base=TRAIN_CFG["seed_base_train"])
        n_train = 2000
        if args.max_samples is not None:
            n_train = min(n_train, int(args.max_samples))
        train_lengths = collect_lengths(train_provider, n_train)
        summarize_lengths("train", train_lengths, requested_num_layers, print_counts=args.print_counts)

    if "test" in split_names:
        test_provider = build_provider(CIRCUIT_SOURCE_CFG, seed_base=TRAIN_CFG["seed_base_test"])
        n_test = int(TRAIN_CFG["n_samples_test"])
        if args.max_samples is not None:
            n_test = min(n_test, int(args.max_samples))
        test_lengths = collect_lengths(test_provider, n_test)
        summarize_lengths("test", test_lengths, requested_num_layers, print_counts=args.print_counts)

    if args.save_hist:
        plot_hist(train_lengths=train_lengths, test_lengths=test_lengths,
                  requested_num_layers=requested_num_layers, save_path=args.save_hist)


if __name__ == "__main__":
    main()
