import numpy as np
import matplotlib.pyplot as plt

def compute_drivers(total_cost_module, P_seq, segments, rep, device):
    """
    Computes interpretable drivers:
      - L, dt = L * delta
      - twoq_ops, avg_cut_prob
      - avg_move_change
    Uses the same stats_extractor as TotalCost.
    """
    import torch  # if not already imported at top

    dtype = P_seq[0].dtype
    N = P_seq[0].shape[0]
    S = len(P_seq)

    stats = total_cost_module.stats_extractor(segments, rep, N=N, device=device, dtype=dtype)

    L = stats["L"]                               # [S]
    dt = L * total_cost_module.delta.to(dtype)   # [S]

    W = torch.stack(P_seq, dim=0).to(dtype)      # [S,N,K]

    # --- comm drivers ---
    twoq_ops = torch.zeros((S,), device=device, dtype=dtype)
    avg_cut_prob = torch.zeros((S,), device=device, dtype=dtype)

    for s in range(S):
        e = stats["edges"][s]
        u_idx, v_idx = e["u"], e["v"]
        omega = e["w"].to(dtype)

        if u_idx.numel() == 0:
            continue

        Wu = W[s, u_idx, :]                 # [E,K]
        Wv = W[s, v_idx, :]                 # [E,K]
        local_prob = (Wu * Wv).sum(dim=1)   # [E]
        cut_prob = 1.0 - local_prob         # [E]

        twoq_ops[s] = omega.sum()
        denom = torch.clamp(omega.sum(), min=1e-12)
        avg_cut_prob[s] = (omega * cut_prob).sum() / denom

    # --- move driver ---
    avg_move_change = torch.zeros((S,), device=device, dtype=dtype)
    if S >= 2:
        stay_prob = (W[:-1] * W[1:]).sum(dim=2)     # [S-1,N]
        change_prob = 1.0 - stay_prob               # [S-1,N]
        avg_move_change[:-1] = change_prob.mean(dim=1)
        avg_move_change[-1] = 0.0

    def to_np(x): return x.detach().cpu().numpy()
    return {
        "L": to_np(L),
        "dt": to_np(dt),
        "twoq_ops": to_np(twoq_ops),
        "avg_cut_prob": to_np(avg_cut_prob),
        "avg_move_change": to_np(avg_move_change),
    }


def plot_cost_dashboard(cost_out, drivers, *, title_prefix="Cost", show=True, save_path_prefix=None):
    """
    Draws:
      (1) stacked breakdown
      (2) heatmap of components x segments (log1p)
      (3) overlays with drivers
    """
    total = cost_out["per_segment_total"].detach().cpu().numpy()
    exec_c = cost_out["per_segment_exec"].detach().cpu().numpy()
    idle_c = cost_out["per_segment_idle"].detach().cpu().numpy()
    comm_c = cost_out["per_segment_comm"].detach().cpu().numpy()
    move_c = cost_out["per_segment_move"].detach().cpu().numpy()
    S = len(total)
    x = np.arange(S)

    # --------------------------
    # (1) Stacked breakdown
    # --------------------------
    plt.figure(figsize=(12, 4))
    plt.stackplot(x, exec_c, idle_c, comm_c, move_c, labels=["exec", "idle", "comm", "move"])
    plt.plot(x, total, linewidth=2, label="total")
    plt.xlabel("Segment index")
    plt.ylabel("Cost")
    plt.title(f"{title_prefix}: breakdown (stacked)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    if save_path_prefix:
        plt.savefig(f"{save_path_prefix}_stacked.png", dpi=200)

    # --------------------------
    # (2) Heatmap (log1p)
    # --------------------------
    rows = [
        ("exec", exec_c),
        ("idle", idle_c),
        ("comm", comm_c),
        ("move", move_c),
        ("total", total),
    ]
    # optional exec subcomponents if present
    if "per_segment_C1q" in cost_out:
        rows.insert(1, ("C1q", cost_out["per_segment_C1q"].detach().cpu().numpy()))
    if "per_segment_C2q_local" in cost_out:
        rows.insert(2, ("C2q_local", cost_out["per_segment_C2q_local"].detach().cpu().numpy()))
    if "per_segment_Cm" in cost_out:
        rows.insert(3, ("Cm", cost_out["per_segment_Cm"].detach().cpu().numpy()))

    labels = [r[0] for r in rows]
    mat = np.vstack([r[1] for r in rows])
    mat_show = np.log1p(mat)

    plt.figure(figsize=(12, 3.5))
    plt.imshow(mat_show, aspect="auto")
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel("Segment index")
    plt.title(f"{title_prefix}: heatmap (log1p)")
    plt.colorbar(label="log(1 + cost)")
    plt.tight_layout()
    if save_path_prefix:
        plt.savefig(f"{save_path_prefix}_heatmap.png", dpi=200)

    # --------------------------
    # (3) Overlays with drivers
    # --------------------------
    # (3a) Comm vs avg cut prob
    plt.figure(figsize=(12, 4))
    ax1 = plt.gca()
    ax1.plot(x, comm_c, linewidth=2)
    ax1.set_xlabel("Segment index")
    ax1.set_ylabel("Comm cost")
    ax1.set_title(f"{title_prefix}: comm vs avg cut probability")
    ax2 = ax1.twinx()
    ax2.plot(x, drivers["avg_cut_prob"], linestyle="--", linewidth=2)
    ax2.set_ylabel("Avg cut probability")
    plt.tight_layout()
    if save_path_prefix:
        plt.savefig(f"{save_path_prefix}_comm_cut.png", dpi=200)

    # (3b) Comm vs #2Q ops
    plt.figure(figsize=(12, 4))
    ax1 = plt.gca()
    ax1.plot(x, comm_c, linewidth=2)
    ax1.set_xlabel("Segment index")
    ax1.set_ylabel("Comm cost")
    ax1.set_title(f"{title_prefix}: comm vs total 2Q ops")
    ax2 = ax1.twinx()
    ax2.plot(x, drivers["twoq_ops"], linestyle="--", linewidth=2)
    ax2.set_ylabel("Total 2Q ops (sum ω)")
    plt.tight_layout()
    if save_path_prefix:
        plt.savefig(f"{save_path_prefix}_comm_twoqops.png", dpi=200)

    # (3c) Idle vs dt = L*delta
    plt.figure(figsize=(12, 4))
    ax1 = plt.gca()
    ax1.plot(x, idle_c, linewidth=2)
    ax1.set_xlabel("Segment index")
    ax1.set_ylabel("Idle cost")
    ax1.set_title(f"{title_prefix}: idle vs dt (= L·delta)")
    ax2 = ax1.twinx()
    ax2.plot(x, drivers["dt"], linestyle="--", linewidth=2)
    ax2.set_ylabel("dt = L * delta")
    plt.tight_layout()
    if save_path_prefix:
        plt.savefig(f"{save_path_prefix}_idle_dt.png", dpi=200)

    # (3d) Move vs avg change prob
    plt.figure(figsize=(12, 4))
    ax1 = plt.gca()
    ax1.plot(x, move_c, linewidth=2)
    ax1.set_xlabel("Segment index")
    ax1.set_ylabel("Move cost")
    ax1.set_title(f"{title_prefix}: move vs avg change probability")
    ax2 = ax1.twinx()
    ax2.plot(x, drivers["avg_move_change"], linestyle="--", linewidth=2)
    ax2.set_ylabel("Avg change probability")
    plt.tight_layout()
    if save_path_prefix:
        plt.savefig(f"{save_path_prefix}_move_change.png", dpi=200)

    if show:
        plt.show()