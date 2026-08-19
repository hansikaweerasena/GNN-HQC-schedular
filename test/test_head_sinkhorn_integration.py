# tests/test_head_sinkhorn_integration.py
"""
Integration checks for the Sinkhorn head inside EvolvingGNN.batch_forward.

Separate from tests/test_sinkhorn.py so that the unit tests stay free of a
PyTorch Geometric dependency.

The synchronisation regression here is the one that matters. The unit-level
version only exercises CapacitySinkhorn.forward() in isolation; the sync bugs
that actually cost wall-clock lived in the *caller* -- ClusteringHead passing
float(self.temperature), a device buffer, once per layer. Only the full
GNN -> head -> Sinkhorn path catches that class of regression.
"""

import torch
from torch_geometric.data import Data

from src.clustering_head import SegmentClustering
from src.evolving_gnn import EvolvingGNN

torch.manual_seed(0)

NF, EF, H, K, N, L, B = 16, 5, 32, 2, 30, 8, 4
CAPS = torch.tensor([20.0, 20.0])

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def circuit(seed):
    g = torch.Generator().manual_seed(seed)
    return [
        Data(
            x=torch.randn(N, NF, generator=g),
            edge_index=torch.randint(0, N, (2, 12), generator=g),
            edge_attr=torch.randn(12, EF, generator=g),
        )
        for _ in range(L)
    ]


def build(mode="sinkhorn", **kw):
    gnn = EvolvingGNN(
        node_feat_dim=NF, edge_feat_dim=EF, mlp_hidden_dim=32, mlp_out_dim=H,
        gnn_out_dim=H, gru_hidden_dim=H, heads=2, dropout=0.0,
    )
    extra = dict(caps=CAPS, sinkhorn_iters=30) if mode == "sinkhorn" else {}
    head = SegmentClustering(
        hidden_dim=H, num_clusters=K, capacity_mode=mode,
        temperature_init=3.0, temperature_min=0.5, temperature_gamma=0.9845,
        **extra, **kw,
    )
    return gnn, head


# --------------------------------------------------------------- feasibility
def t_batched_feasibility():
    for mode in ("sinkhorn", "softmax"):
        gnn, head = build(mode)
        gnn.eval(); head.eval()
        with torch.no_grad():
            res = gnn.batch_forward([circuit(i) for i in range(B)], [L] * B,
                                    cluster_head=head.head)
        P = torch.stack([p for b in range(B) for p in res[b]["P_seq"]])
        rows = torch.allclose(P.sum(-1), torch.ones(B * L, N), atol=1e-5)
        cols = bool((P.sum(-2) <= CAPS + 1e-4).all())
        check(f"I1 [{mode}] rows sum to 1", rows)
        if mode == "sinkhorn":
            check(f"I1 [{mode}] columns respect capacity", cols,
                  f"max occupancy {float(P.sum(-2).max()):.4f} vs cap 20")


# ------------------------------------------------------- FULL-PATH sync test
def t_no_sync_full_path():
    """
    Zero CPU<->GPU synchronisation anywhere in GNN -> head -> Sinkhorn.

    Runs on CPU tensors, but every offending call (.item(), float() on a
    tensor, torch.allclose, bool() on a tensor) is intercepted regardless of
    device, so the check is device-independent.
    """
    gnn, head = build("sinkhorn")
    gnn.eval(); head.eval()
    head.set_epoch(140)                      # annealed T, as in late training

    calls = []
    orig = {
        "item": torch.Tensor.item,
        "allclose": torch.allclose,
        "float": torch.Tensor.__float__,
        "bool": torch.Tensor.__bool__,
        "tolist": torch.Tensor.tolist,
    }
    torch.Tensor.item = lambda self: (calls.append("item"), orig["item"](self))[1]
    torch.allclose = lambda *a, **k: (calls.append("allclose"), orig["allclose"](*a, **k))[1]
    torch.Tensor.__float__ = lambda self: (calls.append("float"), orig["float"](self))[1]
    torch.Tensor.__bool__ = lambda self: (calls.append("bool"), orig["bool"](self))[1]
    torch.Tensor.tolist = lambda self: (calls.append("tolist"), orig["tolist"](self))[1]
    try:
        with torch.no_grad():
            gnn.batch_forward([circuit(i) for i in range(B)], [L] * B,
                              cluster_head=head.head)
    finally:
        torch.Tensor.item = orig["item"]
        torch.allclose = orig["allclose"]
        torch.Tensor.__float__ = orig["float"]
        torch.Tensor.__bool__ = orig["bool"]
        torch.Tensor.tolist = orig["tolist"]

    check(f"I2 zero device syncs across {B} circuits x {L} layers",
          len(calls) == 0, f"offending calls: {calls[:8]}")

    # Reading diagnostics is where the sync is *supposed* to happen.
    d = head.diagnostics
    check("I2 diagnostics readable after the forward",
          set(d) == {"T", "row_residual", "col_residual"} and d["col_residual"] < 1e-5,
          f"{d}")


# --------------------------------------------------------- checkpoint config
def t_checkpoint_carries_config():
    """
    An eval script that rebuilds the module with wrong caps / n_iters and loads
    the checkpoint must end up with the checkpoint's values, not its own.
    """
    _, saved = build("sinkhorn")
    saved.set_epoch(140)

    wrong = SegmentClustering(
        hidden_dim=H, num_clusters=K, capacity_mode="sinkhorn",
        caps=torch.tensor([15.0, 15.0]), sinkhorn_iters=7,
        temperature_init=3.0, temperature_min=0.5, temperature_gamma=0.9845,
    )
    wrong.load_state_dict(saved.state_dict())

    sk = wrong.head.sinkhorn
    check("I3 caps from checkpoint", torch.allclose(sk.caps, CAPS), f"{sk.caps.tolist()}")
    check("I3 c_total recomputed", sk.c_total == 40, f"c_total={sk.c_total}")
    check("I3 n_iters from checkpoint", sk.n_iters == 30, f"n_iters={sk.n_iters}")
    check("I3 temperature mirror refreshed", abs(wrong.head._T - 0.5) < 1e-6,
          f"_T={wrong.head._T:.4f}")


# ------------------------------------------------------------------ gradients
def t_gradients():
    gnn, head = build("sinkhorn")
    gnn.train(); head.train()
    res = gnn.batch_forward([circuit(i) for i in range(B)], [L] * B,
                            cluster_head=head.head)
    loss = sum((p * torch.randn_like(p)).sum() for b in range(B) for p in res[b]["P_seq"])
    loss.backward()
    gnn_ok = [p.grad for p in gnn.parameters() if p.grad is not None]
    head_ok = [p.grad for p in head.parameters() if p.grad is not None]
    check("I4 finite gradients through the encoder",
          len(gnn_ok) > 0 and all(torch.isfinite(g).all() for g in gnn_ok))
    check("I4 finite non-zero gradients into the head",
          len(head_ok) > 0 and any(float(g.abs().max()) > 0 for g in head_ok))


if __name__ == "__main__":
    t_batched_feasibility()
    t_no_sync_full_path()
    t_checkpoint_carries_config()
    t_gradients()
    nf = sum(1 for _, ok in RESULTS if not ok)
    print(f"\n{len(RESULTS) - nf}/{len(RESULTS)} checks passed")
    raise SystemExit(1 if nf else 0)
