import os, sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.circuit_sources import build_provider
from utils.cost_config_reader import load_cost_config, load_scheduler_cfg
from utils.print_utils import print_run_config

from src.circuit_representation import CircuitRepresentation
from src.circuit_segmentation import segment_circuit
from src.cost_function import TotalCost
from src.qubit_interaction_graph import build_segment_graph_arrays


def softmax_np(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

def entropy_np(p, axis=-1, normalize=True):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    H = -np.sum(p * np.log(p), axis=axis)
    if normalize:
        K = p.shape[axis]
        H = H / (math.log(K) + 1e-12)   # 0..1
    return H

def simple_kmeans(X, k=4, seed=0, iters=30):
    """
    Lightweight kmeans (no sklearn dependency).
    X: [M, D]
    returns labels [M], centers [k, D]
    """
    rng = np.random.RandomState(seed)
    X = np.asarray(X, dtype=np.float64)
    M = X.shape[0]
    if M == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((k, X.shape[1]), dtype=np.float64)

    # init centers from random points
    idx = rng.choice(M, size=min(k, M), replace=False)
    centers = X[idx].copy()
    if centers.shape[0] < k:
        # pad if fewer points than k
        pad = np.repeat(centers[:1], k - centers.shape[0], axis=0)
        centers = np.concatenate([centers, pad], axis=0)

    labels = np.zeros((M,), dtype=np.int64)
    for _ in range(iters):
        # assign
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # [M,k]
        new_labels = np.argmin(d2, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        # update
        for j in range(k):
            pts = X[labels == j]
            if len(pts) > 0:
                centers[j] = pts.mean(axis=0)
    return labels, centers


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

def sample_random_P_seq(N, K, T, seed, mode="iid", p_stay=0.8, device="cpu"):
    rng = np.random.RandomState(seed)
    P_seq = []
    prev_assign = rng.randint(0, K, size=N)

    for t in range(T):
        if mode == "iid":
            assign = rng.randint(0, K, size=N)
        else:  # markov
            assign = prev_assign.copy()
            flip = rng.rand(N) > p_stay
            assign[flip] = rng.randint(0, K, size=np.sum(flip))
            prev_assign = assign

        P_t = torch.zeros((N, K), dtype=torch.float32, device=device)
        P_t[torch.arange(N, device=device), torch.tensor(assign, device=device)] = 1.0
        P_seq.append(P_t)

    return P_seq

def build_allk_P_seq(N: int, K: int, T: int, tech_index: int, device):
    """All qubits in all segments use a single tech."""
    P_seq = []
    for _ in range(T):
        P_t = torch.zeros((N, K), device=device, dtype=torch.float32)
        P_t[:, tech_index] = 1.0
        P_seq.append(P_t)
    return P_seq

def build_uniform_P_seq(N: int, K: int, T: int, device):
    P_seq = []
    val = 1.0 / float(K)
    for _ in range(T):
        P_t = torch.full((N, K), val, device=device, dtype=torch.float32)
        P_seq.append(P_t)
    return P_seq

def oracle_segment_dp_with_path(per_seg_base: np.ndarray, cmove: float, N: int):
    """
    DP over tech-per-segment (all qubits in a segment share the same tech),
    includes movement penalty when tech changes between segments.

    Returns: (best_total, tech_seq_list_len_T)
    """
    T, K = per_seg_base.shape
    dp = np.zeros((T, K), dtype=np.float64)
    parent = np.full((T, K), -1, dtype=np.int64)

    dp[0, :] = per_seg_base[0, :]
    change_pen = cmove * float(N)

    for t in range(1, T):
        for k in range(K):
            best_prev = np.inf
            best_j = 0
            for j in range(K):
                trans = 0.0 if (j == k) else change_pen
                cand = dp[t - 1, j] + trans
                if cand < best_prev:
                    best_prev = cand
                    best_j = j
            dp[t, k] = per_seg_base[t, k] + best_prev
            parent[t, k] = best_j

    last_k = int(np.argmin(dp[T - 1, :]))
    best_total = float(dp[T - 1, last_k])

    # backtrack
    tech_seq = [0] * T
    tech_seq[T - 1] = last_k
    for t in range(T - 1, 0, -1):
        tech_seq[t - 1] = int(parent[t, tech_seq[t]])

    return best_total, tech_seq

@torch.no_grad()
def expected_cut_prob(rep, segments, P_seq):
    """
    Returns:
      cut_avg: average cut probability over all segments (weighted by 2Q weights)
      cut_per_seg: list length T
    """
    per_segment_graphs = build_segment_graph_arrays(rep, segments)

    cut_per_seg = []
    total_w_all = 0.0
    total_cut_w_all = 0.0

    for seg_id, x_s, edge_index_s, edge_attr_s in per_segment_graphs:
        # edge_index_s: shape [2, E], edge_attr_s: [E, ...] weights assumed in col 0
        if edge_index_s.size == 0 or edge_index_s.shape[1] == 0:
            cut_per_seg.append(0.0)
            continue

        ei = torch.tensor(edge_index_s, dtype=torch.long)
        w = torch.tensor(edge_attr_s[:, 0], dtype=torch.float32)  # assumes weight in first column

        P_t = P_seq[seg_id]  # [N,K]

        u = ei[0]  # [E]
        v = ei[1]  # [E]

        # same-tech probability: sum_k P[u,k]*P[v,k]
        same = (P_t[u] * P_t[v]).sum(dim=1)  # [E]
        cut = 1.0 - same                     # [E]

        total_w = float(w.sum().item())
        total_cut_w = float((w * cut).sum().item())

        cut_prob = (total_cut_w / (total_w + 1e-12))
        cut_per_seg.append(float(cut_prob))

        total_w_all += total_w
        total_cut_w_all += total_cut_w

    cut_avg = total_cut_w_all / (total_w_all + 1e-12)
    return float(cut_avg), cut_per_seg

@torch.no_grad()
def compute_alltech_mats(total_cost_module, segments, rep, device):
    """
    Compute per-tech totals and per-segment component matrices under ALL-TECH schedules.
    Returns:
      totals[K]
      exec[T,K], idle[T,K], comm[T,K], move[T,K], seg_total[T,K]
      base[T,K] = exec+idle+comm (no move)
      cmove scalar
    """
    K = total_cost_module.K
    T = len(segments)
    N = rep.num_qubits

    totals = np.zeros((K,), dtype=np.float64)
    exec_m = np.zeros((T, K), dtype=np.float64)
    idle_m = np.zeros((T, K), dtype=np.float64)
    comm_m = np.zeros((T, K), dtype=np.float64)
    move_m = np.zeros((T, K), dtype=np.float64)
    segtot_m = np.zeros((T, K), dtype=np.float64)

    cmove = float(total_cost_module.cmove.detach().cpu().item())

    for k in range(K):
        P_seq = build_allk_P_seq(N, K, T, k, device=device)
        out = total_cost_module(P_seq, segments, rep, debug=False)

        totals[k] = float(out["total_cost"].detach().cpu().item())
        exec_m[:, k] = out["per_segment_exec"].detach().cpu().numpy()
        idle_m[:, k] = out["per_segment_idle"].detach().cpu().numpy()
        comm_m[:, k] = out["per_segment_comm"].detach().cpu().numpy()
        move_m[:, k] = out["per_segment_move"].detach().cpu().numpy()
        segtot_m[:, k] = out["per_segment_total"].detach().cpu().numpy()

    base = exec_m + idle_m + comm_m
    return totals, exec_m, idle_m, comm_m, move_m, segtot_m, base, cmove


@torch.no_grad()
def compute_debug_segment_features(total_cost_module, segments, rep, device):
    """
    Assignment-independent circuit/segment stats via debug=True.
    Uses uniform P just to trigger debug collection.
    Returns dict of arrays length T:
      n1q_sum, nm_sum, num_edges, twoq_ops

    Note: gamma is now edge-wise (gamma_e [E_s, K]) and lives in stats["edges"],
    not as a per-segment scalar, so it is not included here.
    """
    K = total_cost_module.K
    T = len(segments)
    N = rep.num_qubits

    P_seq = build_uniform_P_seq(N, K, T, device=device)
    out = total_cost_module(P_seq, segments, rep, debug=True)

    dbg = out.get("debug_stats", None)
    if dbg is None:
        return None

    # Each should be tensor length T
    return {
        "n1q_sum": dbg["n1q_sum"].detach().cpu().numpy(),
        "nm_sum": dbg["nm_sum"].detach().cpu().numpy(),
        "num_edges": dbg["num_edges"].detach().cpu().numpy(),
        "twoq_ops": dbg["twoq_ops"].detach().cpu().numpy(),
    }

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
def analyze_dataset( name, dataset, total_cost_module, device, alpha_entropy=1.0, mix_samples=8, mix_mode="markov", p_stay=0.90, k_segtypes=4):
    """
    Adds:
      b.1  unique best techs per circuit (ignoring move + oracle schedule)
      b.2  segment-type composition (debug feature clustering)
      b.3  component dominance shares under oracle schedule
      b.4  sensitivity: scaling idle/comm changes winner
      b.5  entropy of per-segment soft tech preference
      c.1-c.4 comm analyses
    """
    MIX_SAMPLES = 8          # start small
    MIX_MODE = "markov"      # "iid" or "markov"
    P_STAY = 0.9             # higher => less move

    K = total_cost_module.K
    tech_names = getattr(total_cost_module, "tech_names", [f"tech_{k}" for k in range(K)])

    winners = np.zeros((K,), dtype=np.int64)
    gaps = []
    oracle_improvements = []

    # b.1
    uniq_best_ign_move = []   # unique best tech per segment if ignoring movement
    uniq_oracle = []          # unique techs in oracle tech-per-seg schedule (includes move)

    # b.2 (segment features across dataset)
    seg_feature_rows = []
    seg_besttech_rows = []    # best tech per segment (ignoring move)
    seg_oracletech_rows = []  # oracle chosen tech for segment

    # b.3 (component shares under oracle schedule)
    shares = {"exec": [], "idle": [], "comm": [], "move": []}

    # b.4 sensitivity flip rates
    # scale sets you can tweak
    idle_scales = [0.5, 1.0, 2.0]
    comm_scales = [0.5, 1.0, 2.0]
    flip_counts = np.zeros((len(idle_scales) + len(comm_scales),), dtype=np.int64)
    flip_labels = [f"idle×{s}" for s in idle_scales] + [f"comm×{s}" for s in comm_scales]
    n_for_sens = 0

    # b.5 entropy over segments
    seg_entropy = []

    # c.1 comm presence/share under oracle schedule
    comm_share_oracle = []
    comm_nonzero_frac_oracle = []

    # comm without assignments: uniform soft schedule stats
    comm_share_uniform = []
    comm_nonzero_frac_uniform = []

    impr_best_perseg_nomove = []    # (best_single - best_perseg_nomove)/best_single
    gap_nomove_to_oracle = []       # (best_perseg_nomove - oracle_total)/best_single  (move penalty effect)

    comm_total_oracle = []
    move_total_oracle = []
    commmove_total_oracle = []

    comm_total_uniform = []
    move_total_uniform = []
    commmove_total_uniform = []

    mix_comm = []
    mix_move = []
    mix_commmove = []
    mix_cutavg = []
    mix_best_commmove = []   # best sample per circuit
    mix_best_cutavg = []

    for idx in range(len(dataset)):
        segments, rep = dataset[idx]
        if len(segments) == 0:
            continue

        T = len(segments)
        N = rep.num_qubits

        totals, exec_m, idle_m, comm_m, move_m, segtot_m, base_m, cmove = compute_alltech_mats(
            total_cost_module, segments, rep, device=device
        )

        # ---- 1) winner tech across circuits (single-tech) ----
        order = np.argsort(totals)
        best_k = int(order[0])
        winners[best_k] += 1

        if K > 1 and abs(totals[order[0]]) > 1e-12:
            gaps.append((float(totals[order[1]]) - float(totals[order[0]])) / abs(float(totals[order[0]])))

        # ---- b.1 unique best techs per segment (ignoring movement) ----
        seg_best = np.argmin(base_m, axis=1)  # [T]
        uniq_best_ign_move.append(len(set(map(int, seg_best.tolist()))))

        # ---- b.5 entropy of soft preference per segment ----
        # p(k|t) ∝ exp(-alpha * base_cost(t,k))
        P_seg = softmax_np(-alpha_entropy * base_m, axis=1)  # [T,K]
        H_seg = entropy_np(P_seg, axis=1, normalize=True)    # [T]
        seg_entropy.extend(H_seg.tolist())

        # ---- 2) oracle segment schedule improvement (vs best single-tech) ----
        oracle_total, tech_seq = oracle_segment_dp_with_path(base_m, cmove=cmove, N=N)
        best_single = float(totals[best_k])
        if abs(best_single) > 1e-12:
            oracle_improvements.append((best_single - oracle_total) / abs(best_single))

        uniq_oracle.append(len(set(map(int, tech_seq))))

        # new addition: how good is the oracle schedule if we ignore movement penalty? (i.e. sum_t min_k base[t,k])
        best_perseg_nomove = float(np.min(base_m, axis=1).sum())  # sum_t min_k base[t,k]

        if abs(best_single) > 1e-12:
            impr_best_perseg_nomove.append((best_single - best_perseg_nomove) / abs(best_single))
            gap_nomove_to_oracle.append((best_perseg_nomove - oracle_total) / abs(best_single))

        # Build oracle P_seq and get component breakdown (b.3 + c.1)
        P_seq_oracle = []
        for t in range(T):
            P_t = torch.zeros((N, K), dtype=torch.float32, device=device)
            P_t[:, int(tech_seq[t])] = 1.0
            P_seq_oracle.append(P_t)

        out_oracle = total_cost_module(P_seq_oracle, segments, rep, debug=False)

        tot = float(out_oracle["total_cost"].detach().cpu().item())
        exec_sum = float(out_oracle["per_segment_exec"].sum().detach().cpu().item())
        idle_sum = float(out_oracle["per_segment_idle"].sum().detach().cpu().item())
        comm_sum = float(out_oracle["per_segment_comm"].sum().detach().cpu().item())
        move_sum = float(out_oracle["per_segment_move"].sum().detach().cpu().item())

        comm_o = float(out_oracle["per_segment_comm"].sum().detach().cpu().item())
        move_o = float(out_oracle["per_segment_move"].sum().detach().cpu().item())

        comm_total_oracle.append(comm_o)
        move_total_oracle.append(move_o)
        commmove_total_oracle.append(comm_o + move_o)

        if abs(tot) > 1e-12:
            shares["exec"].append(exec_sum / tot)
            shares["idle"].append(idle_sum / tot)
            shares["comm"].append(comm_sum / tot)
            shares["move"].append(move_sum / tot)

        # c.1 comm presence / share under oracle
        per_comm = out_oracle["per_segment_comm"].detach().cpu().numpy()
        comm_share_oracle.append(comm_sum / (tot + 1e-12))
        comm_nonzero_frac_oracle.append(float(np.mean(per_comm > 1e-12)))

        # ---- comm “without assignments”: uniform soft P ----
        P_seq_u = build_uniform_P_seq(N, K, T, device=device)
        out_u = total_cost_module(P_seq_u, segments, rep, debug=False)
        tot_u = float(out_u["total_cost"].detach().cpu().item())
        comm_u = float(out_u["per_segment_comm"].sum().detach().cpu().item())
        per_comm_u = out_u["per_segment_comm"].detach().cpu().numpy()

        move_u = float(out_u["per_segment_move"].sum().detach().cpu().item())

        comm_total_uniform.append(comm_u)
        move_total_uniform.append(move_u)
        commmove_total_uniform.append(comm_u + move_u)

        comm_share_uniform.append(comm_u / (tot_u + 1e-12))
        comm_nonzero_frac_uniform.append(float(np.mean(per_comm_u > 1e-12)))

        # ---- c.2 variability across tech (comm only) ----
        # We'll summarize later using comm_m totals

        # ---- c.3 per-segment comm preference (argmin over comm matrix) ----
        # We'll aggregate later across dataset

        # ---- b.2 segment-type composition via debug stats ----
        dbg = compute_debug_segment_features(total_cost_module, segments, rep, device=device)
        if dbg is not None:
            # segment length (layers per segment)
            seg_len = np.array([(s.layer_range[1] - s.layer_range[0] + 1) for s in segments], dtype=np.float64)

            # build [T, D] feature rows
            X = np.stack([
                dbg["n1q_sum"].astype(np.float64),
                dbg["nm_sum"].astype(np.float64),
                dbg["twoq_ops"].astype(np.float64),
                dbg["num_edges"].astype(np.float64),
                seg_len,
            ], axis=1)  # D=5

            seg_feature_rows.append(X)
            seg_besttech_rows.append(seg_best.astype(np.int64))
            seg_oracletech_rows.append(np.array(tech_seq, dtype=np.int64))

        # ---- b.4 sensitivity: scale idle / comm and see winner flips (single-tech) ----
        # Compute per-tech component sums under all-tech schedules
        exec_sum_k = exec_m.sum(axis=0)  # [K]
        idle_sum_k = idle_m.sum(axis=0)
        comm_sum_k = comm_m.sum(axis=0)
        move_sum_k = move_m.sum(axis=0)

        baseline_winner = int(np.argmin(exec_sum_k + idle_sum_k + comm_sum_k + move_sum_k))
        n_for_sens += 1

        # idle scales
        for i, s in enumerate(idle_scales):
            scaled = exec_sum_k + (s * idle_sum_k) + comm_sum_k + move_sum_k
            if int(np.argmin(scaled)) != baseline_winner:
                flip_counts[i] += 1

        # comm scales
        off = len(idle_scales)
        for j, s in enumerate(comm_scales):
            scaled = exec_sum_k + idle_sum_k + (s * comm_sum_k) + move_sum_k
            if int(np.argmin(scaled)) != baseline_winner:
                flip_counts[off + j] += 1


        best_this = None
        best_cut = None

        for s in range(MIX_SAMPLES):
            P_mix = sample_random_P_seq(
                N=rep.num_qubits,
                K=K,
                T=len(segments),
                seed=12345 + 1000*idx + s,
                mode=mix_mode,
                p_stay=p_stay,
                device=device,
            )

            out_m = total_cost_module(P_mix, segments, rep, debug=False)
            comm_mx = float(out_m["per_segment_comm"].sum().detach().cpu().item())
            move_mx = float(out_m["per_segment_move"].sum().detach().cpu().item())
            cm = comm_mx + move_mx

            cut_avg, _ = expected_cut_prob(rep, segments, P_mix)

            mix_comm.append(comm_mx)
            mix_move.append(move_mx)
            mix_commmove.append(cm)
            mix_cutavg.append(cut_avg)

            if best_this is None or cm < best_this:
                best_this = cm
                best_cut = cut_avg

        if best_this is not None:
            mix_best_commmove.append(best_this)
            mix_best_cutavg.append(best_cut)

    # ---- Print + Plots (original + new) ----
    print(f"\n=== DATASET DIAGNOSTICS: {name} ===")
    print("Winner counts (best single-tech):")
    for k in range(K):
        print(f"  {tech_names[k]}: {winners[k]}")

    gaps = np.asarray(gaps, dtype=np.float64)
    oracle_improvements = np.asarray(oracle_improvements, dtype=np.float64)

    if gaps.size > 0:
        print(f"Gap (2nd-best - best)/best: mean={gaps.mean():.4f}, median={np.median(gaps):.4f}, p90={np.quantile(gaps, 0.9):.4f}")
    if oracle_improvements.size > 0:
        pos = (oracle_improvements > 0).mean() * 100.0
        print(f"Oracle improvement: mean={oracle_improvements.mean():.4f}, median={np.median(oracle_improvements):.4f}, %positive={pos:.1f}%")

    # original plots
    plot_winner_hist(name, winners, tech_names)
    plot_gap_hist(name, gaps)
    plot_oracle_improvement_hist(name, oracle_improvements)

    # b.1 plots
    plot_hist(f"#unique best techs per circuit (ignoring move) ({name})", uniq_best_ign_move, bins=20, xlabel="#unique techs")
    plot_hist(f"#unique techs in oracle schedule ({name})", uniq_oracle, bins=20, xlabel="#unique techs")

    plot_hist(f"[1] Improvement of per-segment best (NO move) vs best-single (%) ({name})",
          np.asarray(impr_best_perseg_nomove) * 100.0, bins=30, xlabel="%")

    plot_hist(f"[1] Penalty: (per-seg best NO-move) → oracle (includes move) (%) ({name})",
            np.asarray(gap_nomove_to_oracle) * 100.0, bins=30, xlabel="%")

    # b.3 cost shares
    plot_component_share_hists(name, shares)

    # b.4 sensitivity flip rates
    if n_for_sens > 0:
        flip_rates = flip_counts / float(n_for_sens)
        plot_sensitivity_flip_rates(name, flip_labels, flip_rates)

    # b.5 entropy
    plot_hist(f"Segment soft-preference entropy (0..1) ({name})", seg_entropy, bins=30, xlabel="entropy")

    # c.1 comm presence/share (oracle + uniform)
    plot_hist(f"Comm share under ORACLE schedule (%) ({name})", np.asarray(comm_share_oracle) * 100.0, bins=30, xlabel="comm share (%)")
    plot_hist(f"Comm share under UNIFORM schedule (%) ({name})", np.asarray(comm_share_uniform) * 100.0, bins=30, xlabel="comm share (%)")

    plot_hist(f"Fraction of segments with comm>0 (ORACLE) ({name})", comm_nonzero_frac_oracle, bins=20, xlabel="fraction")
    plot_hist(f"Fraction of segments with comm>0 (UNIFORM) ({name})", comm_nonzero_frac_uniform, bins=20, xlabel="fraction")

    # c.4 correlation scatter
    plot_comm_share_scatter(name, comm_share_uniform, oracle_improvements)

    # b.2 segment-type composition (cluster segment feature rows)
    if len(seg_feature_rows) > 0:
        Xall = np.concatenate(seg_feature_rows, axis=0)  # [M,5]
        # standardize for clustering
        mu = Xall.mean(axis=0, keepdims=True)
        sd = Xall.std(axis=0, keepdims=True) + 1e-9
        Xz = (Xall - mu) / sd

        labels, centers = simple_kmeans(Xz, k=k_segtypes, seed=0, iters=40)
        print(f"\n[b.2] Segment-type clustering: k={k_segtypes}, total_segments={Xall.shape[0]}")
        for c in range(k_segtypes):
            cnt = int(np.sum(labels == c))
            if cnt == 0:
                continue
            # show unnormalized center (approx) by mapping back
            cen = centers[c] * sd.flatten() + mu.flatten()
            print(f"  cluster{c}: count={cnt}  mean(n1q,nm,twoq_ops,num_edges,seg_len)={cen.round(2).tolist()}")

        # also show “best tech” distribution per cluster (ignoring move)
        best_all = np.concatenate(seg_besttech_rows, axis=0)
        print("\n[b.2] Best-tech distribution per segment-cluster (ignoring move):")
        for c in range(k_segtypes):
            idxs = np.where(labels == c)[0]
            if len(idxs) == 0:
                continue
            counts = np.bincount(best_all[idxs], minlength=K)
            s = ", ".join([f"{tech_names[k]}:{int(counts[k])}" for k in range(K)])
            print(f"  cluster{c}: {s}")

    # plot_hist(f"Comm variability across tech per circuit (max-min) ({name})",
    #         comm_var_per_circuit, bins=30, xlabel="max(comm_k)-min(comm_k)")

    plot_hist(f"[2] Oracle comm total ({name})", comm_total_oracle, bins=30, xlabel="comm")
    plot_hist(f"[2] Oracle move total ({name})", move_total_oracle, bins=30, xlabel="move")
    plot_hist(f"[2] Oracle (comm+move) total ({name})", commmove_total_oracle, bins=30, xlabel="comm+move")

    plot_hist(f"[2] Uniform comm total ({name})", comm_total_uniform, bins=30, xlabel="comm")
    plot_hist(f"[2] Uniform move total ({name})", move_total_uniform, bins=30, xlabel="move")
    plot_hist(f"[2] Uniform (comm+move) total ({name})", commmove_total_uniform, bins=30, xlabel="comm+move")

    # c.2 + c.3 (comm variability + comm preference) aggregated via an extra pass
    # (we need comm matrices; easiest is to compute segment comm-best from base all-tech matrices per circuit)
    # To keep runtime low, we’ll just report based on comm_share_uniform and printed plots above.
    # plt.figure(figsize=(7, 4))
    # x = np.arange(K)
    # plt.bar(x, comm_segbest_counts)
    # plt.xticks(x, tech_names, rotation=20)
    # plt.ylabel("#segments")
    # plt.title(f"Comm-preferred tech per segment ({name})")
    # plt.tight_layout()
    # plt.show()


    plot_hist(f"[4] Mixed schedules: comm total ({name})", mix_comm, bins=30, xlabel="comm")
    plot_hist(f"[4] Mixed schedules: move total ({name})", mix_move, bins=30, xlabel="move")
    plot_hist(f"[4] Mixed schedules: (comm+move) total ({name})", mix_commmove, bins=30, xlabel="comm+move")
    plot_hist(f"[4] Mixed schedules: avg cut probability ({name})", mix_cutavg, bins=30, xlabel="cut prob")

    plot_hist(f"[4] Per-circuit BEST sampled (comm+move) ({name})", mix_best_commmove, bins=30, xlabel="best comm+move")
    plot_hist(f"[4] Per-circuit BEST sampled avg cut prob ({name})", mix_best_cutavg, bins=30, xlabel="cut prob")

    return {
        "winners": winners,
        "gaps": gaps,
        "oracle_improvements": oracle_improvements,
        "uniq_best_ign_move": np.asarray(uniq_best_ign_move),
        "uniq_oracle": np.asarray(uniq_oracle),
        "seg_entropy": np.asarray(seg_entropy),
        "comm_share_oracle": np.asarray(comm_share_oracle),
        "comm_share_uniform": np.asarray(comm_share_uniform),
    }


def print_means_table(split_name, tech_names, stats: dict):
    """
    stats contains arrays/lists. We'll print mean/median for key metrics.
    """
    def fmt(x):
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return "N/A"
        return f"mean={x.mean():.4g}, med={np.median(x):.4g}"

    print("\n" + "="*80)
    print(f"MEANS SUMMARY: {split_name}")
    print("="*80)

    keys = [
        ("best_vs_2nd_gap", stats.get("gaps", [])),
        ("oracle_improvement", stats.get("oracle_improvements", [])),
        ("perseg_best_nomove_improvement", stats.get("impr_best_perseg_nomove", [])),
        ("nomove_to_oracle_penalty", stats.get("gap_nomove_to_oracle", [])),
        ("seg_entropy", stats.get("seg_entropy", [])),
        ("oracle_comm", stats.get("comm_total_oracle", [])),
        ("oracle_move", stats.get("move_total_oracle", [])),
        ("oracle_comm_plus_move", stats.get("commmove_total_oracle", [])),
        ("uniform_comm", stats.get("comm_total_uniform", [])),
        ("uniform_move", stats.get("move_total_uniform", [])),
        ("uniform_comm_plus_move", stats.get("commmove_total_uniform", [])),
        ("mixed_comm", stats.get("mix_comm", [])),
        ("mixed_move", stats.get("mix_move", [])),
        ("mixed_comm_plus_move", stats.get("mix_commmove", [])),
        ("mixed_cut_avg", stats.get("mix_cutavg", [])),
    ]

    for name, arr in keys:
        print(f"{name:32s} : {fmt(arr)}")

    # Winner distribution
    winners = stats.get("winners", None)
    if winners is not None:
        total = winners.sum()
        print("\nWinner tech distribution (best single-tech):")
        for k, c in enumerate(winners):
            pct = (c / total * 100.0) if total > 0 else 0.0
            print(f"  {tech_names[k]:15s}  {int(c):5d}  ({pct:5.1f}%)")

    print("="*80 + "\n")

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


def plot_hist(title, data, bins=30, xlabel="", ylabel="#items"):
    if data is None or len(data) == 0:
        return
    plt.figure(figsize=(7, 4))
    plt.hist(np.asarray(data), bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_component_share_hists(split_name, shares_dict):
    """
    shares_dict: {"exec": [...], "idle": [...], "comm": [...], "move": [...]}
    each entry is list of shares in [0,1]
    """
    for key, arr in shares_dict.items():
        plot_hist(
            title=f"{key} share of total ({split_name})",
            data=np.asarray(arr) * 100.0,
            bins=30,
            xlabel=f"{key} share (%)",
        )

def plot_sensitivity_flip_rates(split_name, labels, flip_rates):
    plt.figure(figsize=(8, 4))
    x = np.arange(len(labels))
    plt.bar(x, np.asarray(flip_rates) * 100.0)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("% circuits whose winner changes")
    plt.title(f"Sensitivity: winner flip rate ({split_name})")
    plt.tight_layout()
    plt.show()

def plot_comm_share_scatter(split_name, comm_share, oracle_impr):
    if len(comm_share) == 0 or len(oracle_impr) == 0:
        return
    comm_share = np.asarray(comm_share)
    oracle_impr = np.asarray(oracle_impr)
    m = min(len(comm_share), len(oracle_impr))
    comm_share = comm_share[:m]
    oracle_impr = oracle_impr[:m]

    plt.figure(figsize=(6, 4))
    plt.scatter(comm_share * 100.0, oracle_impr * 100.0, s=12)
    plt.xlabel("Comm share (%)")
    plt.ylabel("Oracle improvement (%)")
    plt.title(f"Comm share vs oracle improvement ({split_name})")
    plt.tight_layout()
    plt.show()

    # correlation
    if m >= 5:
        corr = np.corrcoef(comm_share, oracle_impr)[0, 1]
        print(f"[Corr] comm_share vs oracle_improvement: {corr:+.3f}")

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
    parser.add_argument("--mix_samples", type=int, default=8,
                    help="Random per-qubit schedules sampled per circuit.")
    parser.add_argument("--mix_mode", type=str, default="markov", choices=["iid", "markov"],
                        help="iid: independent per segment. markov: per-qubit persistence across segments.")
    parser.add_argument("--p_stay", type=float, default=0.90,
                        help="(markov only) probability a qubit stays on same tech between segments.")
    parser.add_argument("--k_segtypes", type=int, default=4,
                        help="Number of segment-type clusters for composition analysis.")
    parser.add_argument("--option_mix", type=str, default=None,
                        help="Comma-separated op:weight pairs for ROI option mix, e.g. "
                             "'op1:0.25,op2a:0.25,op2b:0.25,op3:0.25'. "
                             "Overrides the scheduler config option field. "
                             "Default: equal weights across all four options.")
    args = parser.parse_args()

    # 1) read args and get configs (like line ~78-83)
    MODEL_CFG, CLUSTER_CFG, TRAIN_CFG, DATASET_CFG, CIRCUIT_SOURCE_CFG = load_scheduler_cfg(args.sched_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) load cost config and print everything
    config = load_cost_config(args.cost_cfg)
    total_cost_module = TotalCost(config).to(device)

    K = len(config["techs"])
    tech_names = [t.get("name", f"tech_{k}") for k, t in enumerate(config["techs"])]
    total_cost_module.tech_names = tech_names   # attach for downstream functions

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

    # --- Option mix for ROI circuits ---
    # For diagnostics we always want to see all tiling options, not just the one
    # hardcoded in the scheduler config.  Build an option_mix dict and inject it
    # into the circuit source config's sampled_kwargs so the provider will sample
    # per-circuit.  The CLI --option_mix flag lets you override this.
    _DEFAULT_OPTION_MIX = {"op1": 0.25, "op2a": 0.25, "op2b": 0.25, "op3": 0.25}

    if CIRCUIT_SOURCE_CFG.get("name") == "roi_composed":
        if args.option_mix is not None:
            # parse "op1:0.25,op2a:0.25,..." from CLI
            try:
                parsed_mix = {}
                for token in args.option_mix.split(","):
                    k, v = token.strip().split(":")
                    parsed_mix[k.strip()] = float(v.strip())
                inject_mix = parsed_mix
            except Exception as e:
                print(f"[WARNING] Could not parse --option_mix ({e}), using default equal mix.")
                inject_mix = _DEFAULT_OPTION_MIX
        else:
            inject_mix = _DEFAULT_OPTION_MIX

        # Inject into a copy of the config so we don't mutate the original
        CIRCUIT_SOURCE_CFG = dict(CIRCUIT_SOURCE_CFG)
        sampled_kwargs = dict(CIRCUIT_SOURCE_CFG.get("sampled_kwargs", {}))
        sampled_kwargs["option_mix"] = inject_mix
        CIRCUIT_SOURCE_CFG["sampled_kwargs"] = sampled_kwargs
        # Remove the hardcoded 'option' key from kwargs if present so the
        # per-sample option_mix takes effect instead
        if "kwargs" in CIRCUIT_SOURCE_CFG and "option" in CIRCUIT_SOURCE_CFG["kwargs"]:
            CIRCUIT_SOURCE_CFG["kwargs"] = dict(CIRCUIT_SOURCE_CFG["kwargs"])
            del CIRCUIT_SOURCE_CFG["kwargs"]["option"]
        print(f"[diagnostics] option_mix injected: {inject_mix}")

    train_provider = build_provider(CIRCUIT_SOURCE_CFG, seed_base=int(TRAIN_CFG["seed_base_train"]))
    test_provider  = build_provider(CIRCUIT_SOURCE_CFG, seed_base=int(TRAIN_CFG["seed_base_test"]))
    train_dataset = CircuitDataset(train_provider, n_samples=n_train, segment_mode=segment_mode, segment_threshold=segment_threshold)
    test_dataset  = CircuitDataset(test_provider,  n_samples=n_test,  segment_mode=segment_mode, segment_threshold=segment_threshold)

    print(f"\nDatasets ready: train={len(train_dataset)}  test={len(test_dataset)}")
    print(f"Segmentation: mode={segment_mode} threshold={segment_threshold}")
    print(f"Circuit source: {CIRCUIT_SOURCE_CFG.get('name')} kwargs={CIRCUIT_SOURCE_CFG.get('kwargs', {})}")

    # 4-7) run requested diagnostics
    if args.split in ("train", "both"):
        stats = analyze_dataset(
            "train",
            train_dataset,
            total_cost_module,
            device=device,
            mix_samples=args.mix_samples,
            mix_mode=args.mix_mode,
            p_stay=args.p_stay,
            k_segtypes=args.k_segtypes,
        )
        print_means_table("train/both", tech_names, stats)

    if args.split in ("test", "both"):
        stats = analyze_dataset(
            "test",
            train_dataset,
            total_cost_module,
            device=device,
            mix_samples=args.mix_samples,
            mix_mode=args.mix_mode,
            p_stay=args.p_stay,
            k_segtypes=args.k_segtypes,
        )
        print_means_table("test/both", tech_names, stats)

if __name__ == "__main__":
    main()