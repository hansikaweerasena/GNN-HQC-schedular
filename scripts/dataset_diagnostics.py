import os, sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.circuit_sources import build_provider
from utils.cost_config_reader import load_cost_config, load_scheduler_cfg
from utils.print_utils import print_run_config

from src.circuit_representation import CircuitRepresentation
from src.circuit_segmentation import segment_circuit
from src.cost_function import TotalCost


# ----------------------------
# Dataset (same “first part” behavior as train_test_eval_debug.py)
# ----------------------------
class CircuitDataset(torch.utils.data.Dataset):
    def __init__(self, provider, n_samples: int, segment_mode: str, segment_threshold: float):
        self.provider = provider
        self.n_samples = int(n_samples)
        self.segment_threshold = float(segment_threshold)
        self.segment_mode = segment_mode

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        qc = self.provider.get(idx)
        rep = CircuitRepresentation(qc)
        segments, seg_ids = segment_circuit(
            rep.layers,
            mode=self.segment_mode,
            threshold=self.segment_threshold
        )
        return segments, rep


# ----------------------------
# Cost helpers
# ----------------------------
def build_allk_P_seq(N: int, K: int, T: int, tech_index: int, device):
    """All qubits in all segments use a single tech."""
    P_seq = []
    for _ in range(T):
        P_t = torch.zeros((N, K), device=device, dtype=torch.float32)
        P_t[:, tech_index] = 1.0
        P_seq.append(P_t)
    return P_seq


@torch.no_grad()
def compute_cost_alltech_and_base_matrix(total_cost_module, segments, rep, device):
    """
    Returns:
      costs_all: [K] total cost when all segments use tech k
      per_seg_base: [T,K] base per-segment cost excluding movement
        (exec + idle + comm) under all-tech-k schedule
      cmove: float scalar (already -log f_move)
    """
    K = total_cost_module.K
    T = len(segments)
    N = rep.num_qubits

    costs_all = np.zeros((K,), dtype=np.float64)
    per_seg_base = np.zeros((T, K), dtype=np.float64)

    # movement scalar (negative log f_move)
    cmove = float(total_cost_module.cmove.detach().cpu().item())

    for k in range(K):
        P_seq = build_allk_P_seq(N, K, T, k, device=device)
        out = total_cost_module(P_seq, segments, rep, debug=False)

        costs_all[k] = float(out["total_cost"].detach().cpu().item())

        # base = exec + idle + comm (movement is 0 in all-tech schedule)
        base = (
            out["per_segment_exec"]
            + out["per_segment_idle"]
            + out["per_segment_comm"]
        )
        per_seg_base[:, k] = base.detach().cpu().numpy()

    return costs_all, per_seg_base, cmove


def oracle_segment_dp(per_seg_base: np.ndarray, cmove: float, N: int):
    """
    Oracle schedule where each segment chooses ONE tech for ALL qubits,
    but tech can vary by segment. Accounts for movement penalty between segments.

    per_seg_base: [T,K] exec+idle+comm per segment under tech k
    transition cost: cmove * N if tech changes else 0
    """
    T, K = per_seg_base.shape
    dp = np.zeros((T, K), dtype=np.float64)

    dp[0, :] = per_seg_base[0, :]

    change_pen = cmove * float(N)

    for t in range(1, T):
        for k in range(K):
            # min over previous tech j
            best_prev = np.inf
            for j in range(K):
                trans = 0.0 if (j == k) else change_pen
                cand = dp[t - 1, j] + trans
                if cand < best_prev:
                    best_prev = cand
            dp[t, k] = per_seg_base[t, k] + best_prev

    return float(dp[T - 1, :].min())


# ----------------------------
# Dataset diagnostics core
# ----------------------------
def analyze_dataset(name, dataset, total_cost_module, device):
    """
    Produces:
      - winner tech histogram
      - best-vs-2nd gap distribution
      - oracle improvement vs best-single-tech distribution
    """
    K = total_cost_module.K
    tech_names = getattr(total_cost_module, "tech_names", [f"tech{k}" for k in range(K)])

    winners = np.zeros((K,), dtype=np.int64)
    gaps = []                 # (second-best - best)/best
    oracle_improvements = []  # (best_single - oracle)/best_single

    for idx in tqdm(range(len(dataset)), desc=f"Analyzing {name}", leave=False):
        segments, rep = dataset[idx]
        if len(segments) == 0:
            continue

        costs_all, per_seg_base, cmove = compute_cost_alltech_and_base_matrix(
            total_cost_module, segments, rep, device=device
        )

        order = np.argsort(costs_all)
        best_k = int(order[0])
        winners[best_k] += 1

        best = float(costs_all[order[0]])
        second = float(costs_all[order[1]]) if K > 1 else float("nan")

        if K > 1 and abs(best) > 1e-12:
            gaps.append((second - best) / abs(best))

        # Oracle segment schedule (single tech per segment, but can change segment-to-segment)
        oracle = oracle_segment_dp(per_seg_base, cmove=cmove, N=rep.num_qubits)
        if abs(best) > 1e-12:
            oracle_improvements.append((best - oracle) / abs(best))

    gaps = np.array(gaps, dtype=np.float64) if len(gaps) else np.zeros((0,), dtype=np.float64)
    oracle_improvements = np.array(oracle_improvements, dtype=np.float64) if len(oracle_improvements) else np.zeros((0,), dtype=np.float64)

    # --------- Print summary ---------
    print(f"\n=== DATASET DIAGNOSTICS: {name} ===")
    print("Winner counts:")
    for k in range(K):
        print(f"  {tech_names[k]}: {winners[k]}")

    if gaps.size > 0:
        print(f"Gap (2nd-best - best)/best: mean={gaps.mean():.4f}, median={np.median(gaps):.4f}, p90={np.quantile(gaps, 0.9):.4f}")
    else:
        print("Gap stats: N/A (K<2 or no samples)")

    if oracle_improvements.size > 0:
        pos = (oracle_improvements > 0).mean() * 100.0
        print(f"Oracle improvement (best_single - oracle)/best_single: mean={oracle_improvements.mean():.4f}, median={np.median(oracle_improvements):.4f}, %positive={pos:.1f}%")
    else:
        print("Oracle improvement stats: N/A")

    # --------- Plots ---------
    plot_winner_hist(name, winners, tech_names)
    plot_gap_hist(name, gaps)
    plot_oracle_improvement_hist(name, oracle_improvements)

    return {
        "winners": winners,
        "gaps": gaps,
        "oracle_improvements": oracle_improvements,
    }


def plot_winner_hist(split_name, winners, tech_names):
    plt.figure(figsize=(7, 4))
    x = np.arange(len(winners))
    plt.bar(x, winners)
    plt.xticks(x, tech_names, rotation=20)
    plt.ylabel("#circuits (winner)")
    plt.title(f"Winner tech across circuits ({split_name})")
    plt.tight_layout()
    plt.show()

# For each circuit, you take the best single-tech cost and the second-best single-tech cost, and compute a relative gap:
# “How much worse is the runner-up compared to the winner?”
# Small gaps = techs are interchangeable; large gaps = winner is clearly better.
def plot_gap_hist(split_name, gaps):
    if gaps.size == 0:
        return
    plt.figure(figsize=(7, 4))
    plt.hist(gaps, bins=30)
    plt.xlabel("(2nd-best - best) / best")
    plt.ylabel("#circuits")
    plt.title(f"Best vs 2nd gap ({split_name})")
    plt.tight_layout()
    plt.show()

# “How much can I improve if I allow the tech choice to change per segment (still one tech per segment), instead of forcing one tech for the whole circuit?”
# “Does mixing technologies across segments actually buy you anything?”
# Near zero = no point mixing; positive = mixing helps.
def plot_oracle_improvement_hist(split_name, oracle_improvements):
    if oracle_improvements.size == 0:
        return
    plt.figure(figsize=(7, 4))
    plt.hist(oracle_improvements * 100.0, bins=30)
    plt.xlabel("% improvement vs best-single-tech")
    plt.ylabel("#circuits")
    plt.title(f"Oracle segment schedule improvement ({split_name})")
    plt.tight_layout()
    plt.show()


# ----------------------------
# Main (structure similar to train_test_eval_debug.py first part)
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sched_cfg", type=str, default="configs.scheduler_config")
    parser.add_argument("--cost_cfg", type=str, default="cost_config_v3.json")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "both"])
    parser.add_argument("--n_train", type=int, default=None)
    parser.add_argument("--n_test", type=int, default=None)
    args = parser.parse_args()

    # 1) read args and get configs (like line ~78-83)
    MODEL_CFG, CLUSTER_CFG, TRAIN_CFG, DATASET_CFG, CIRCUIT_SOURCE_CFG = load_scheduler_cfg(args.sched_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) load cost config and print everything
    config = load_cost_config(args.cost_cfg)
    total_cost_module = TotalCost(config).to(device)

    K = len(config["techs"])
    derived = {"device": str(device), "K_num_clusters": K}
    print_run_config(
        MODEL_CFG=MODEL_CFG,
        CLUSTER_CFG=CLUSTER_CFG,
        TRAIN_CFG=TRAIN_CFG,
        DATASET_CFG=DATASET_CFG,
        CIRCUIT_SOURCE_CFG=CIRCUIT_SOURCE_CFG,
        derived=derived,
    )

    # 3) create train/test datasets (like line ~117-122)
    n_train = int(args.n_train) if args.n_train is not None else int(TRAIN_CFG["n_samples_train"])
    n_test  = int(args.n_test)  if args.n_test  is not None else int(TRAIN_CFG["n_samples_test"])

    segment_threshold = float(DATASET_CFG["segment_threshold"])
    segment_mode = str(DATASET_CFG["segmentation_mode"])

    train_provider = build_provider(CIRCUIT_SOURCE_CFG, seed_base=int(TRAIN_CFG["seed_base_train"]))
    test_provider  = build_provider(CIRCUIT_SOURCE_CFG, seed_base=int(TRAIN_CFG["seed_base_test"]))

    train_dataset = CircuitDataset(train_provider, n_samples=n_train, segment_mode=segment_mode, segment_threshold=segment_threshold)
    test_dataset  = CircuitDataset(test_provider,  n_samples=n_test,  segment_mode=segment_mode, segment_threshold=segment_threshold)

    print(f"\nDatasets ready: train={len(train_dataset)}  test={len(test_dataset)}")
    print(f"Segmentation: mode={segment_mode} threshold={segment_threshold}")
    print(f"Circuit source: {CIRCUIT_SOURCE_CFG.get('name')} kwargs={CIRCUIT_SOURCE_CFG.get('kwargs', {})}")

    # 4-7) run requested diagnostics
    if args.split in ("train", "both"):
        analyze_dataset("train", train_dataset, total_cost_module, device=device)

    if args.split in ("test", "both"):
        analyze_dataset("test", test_dataset, total_cost_module, device=device)


if __name__ == "__main__":
    main()