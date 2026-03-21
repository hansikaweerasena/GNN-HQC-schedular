"""
train_test_eval_debug.py

Main training + evaluation entry point for the MOSAIC scheduler.

Pipeline per circuit:
  1. CircuitRepresentation  — extract layers from Qiskit circuit
  2. segment_circuit        — layer mode: one "segment" per layer (needed by TotalCost)
  3. build_layer_graph_arrays — build backbone graphs + windowed features (new input)
  4. EvolvingGNN            — MLP -> GATv2 -> GRU  =>  h_seq
  5. SegmentClustering      — prototype clustering  =>  P_seq
  6. TotalCost              — differentiable cost   =>  scalar loss
  7. BPTT + Adam update

Window sizes (W_short, W_long) are derived from the cost config's kappa values,
not from hyperparameter tuning.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from tqdm import tqdm
import os, sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.circuit_sources import build_provider
from src.circuit_representation import CircuitRepresentation
from src.circuit_segmentation import segment_circuit
from src.qubit_interaction_graph import (
    build_layer_graph_arrays,
    compute_window_sizes_from_config,
    NODE_FEAT_DIM,
    EDGE_FEAT_DIM,
)
from src.evolving_gnn import EvolvingGNN
from src.clustering_head import SegmentClustering
from src.cost_function import TotalCost, CapacityPenalty
from utils.train_utils import train_step
from utils.cost_config_reader import load_cost_config
from utils.print_utils import print_run_config
from utils.cost_config_reader import load_scheduler_cfg


# =============================================================================
# Input construction helpers
# =============================================================================


def build_layer_data_list(circuit, w_short: int, w_long: int, device: torch.device):
    """
    Build a list of PyG Data objects (one per circuit layer) from the new
    windowed backbone graph input pipeline.

    Tensors are moved to `device` here — once per circuit — so the training
    loop never needs to touch device placement.

    Args:
        circuit:  CircuitRepresentation
        w_short:  short window radius = ceil(max_kappa)
        w_long:   long  window radius = 2 * w_short
        device:   torch.device to place all tensors on

    Returns:
        layer_data_list: List[Data], length = T (number of circuit layers)
    """
    arrays = build_layer_graph_arrays(circuit, w_short, w_long)
    return [
        Data(
            x          = torch.tensor(x_np,  dtype=torch.float32).to(device),
            edge_index = torch.tensor(ei_np, dtype=torch.long).to(device),
            edge_attr  = torch.tensor(ea_np, dtype=torch.float32).to(device),
        )
        for x_np, ei_np, ea_np in arrays
    ]


# =============================================================================
# Dataset
# =============================================================================


class CircuitDataset(Dataset):
    def __init__(
        self,
        provider,
        n_samples: int,
        segment_mode: str,
        segment_threshold: float,
        w_short: int,
        w_long: int,
        device: torch.device = None,
    ):
        self.provider           = provider
        self.n_samples          = int(n_samples)
        self.segment_threshold  = float(segment_threshold)
        self.segment_mode       = segment_mode
        self.w_short            = w_short
        self.w_long             = w_long
        self.device             = device or torch.device("cpu")
        self._cache             = {}  # idx -> (layer_data_list, segments, rep)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]

        qc  = self.provider.get(idx)
        rep = CircuitRepresentation(qc)

        # segments still needed by TotalCost for gate counting
        segments, seg_ids = segment_circuit(
            rep.layers,
            mode=self.segment_mode,
            threshold=self.segment_threshold,
        )

        # one graph per layer, tensors placed on device here — once, not per step
        layer_data_list = build_layer_data_list(rep, self.w_short, self.w_long, self.device)

        result = (layer_data_list, segments, rep)
        self._cache[idx] = result
        return result


def collate_fn(batch):
    return batch  # variable-length sequences; return as-is


# =============================================================================
# Evaluation helper
# =============================================================================


def evaluate_model(
    model,
    cluster_module,
    cost_module,
    test_loader,
    device,
    capacity_penalty=None,
):
    model.eval()
    cluster_module.eval()
    total_loss, total_cap, all_per_seg = 0.0, 0.0, []

    with torch.no_grad():
        for batch in test_loader:
            for layer_data_list, segments, rep in batch:
                loss, per_seg, cap_val = train_step(
                    model, cluster_module, cost_module,
                    layer_data_list, segments, rep,
                    optimizer=None,
                    capacity_penalty=capacity_penalty,
                )
                total_loss += loss
                total_cap  += cap_val
                all_per_seg.append(per_seg.cpu().numpy())

    n = len(test_loader.dataset)
    return total_loss / n, total_cap / n, np.concatenate(all_per_seg)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sched_cfg", type=str, default="configs.scheduler_config")
    parser.add_argument("--cost_cfg",  type=str, default="cost_config_v3.json")
    args = parser.parse_args()

    MODEL_CFG, CLUSTER_CFG, TRAIN_CFG, DATASET_CFG, CIRCUIT_SOURCE_CFG = \
        load_scheduler_cfg(args.sched_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = load_cost_config(args.cost_cfg)

    # Derive window sizes from cost config kappa values (not hyperparameters)
    w_short, w_long = compute_window_sizes_from_config(config)
    print(f"Window sizes: W_short={w_short}, W_long={w_long}")

    K = len(config["techs"])

    derived = {
        "device": str(device),
        "K_num_clusters": K,
        "w_short": w_short,
        "w_long":  w_long,
        "node_feat_dim": NODE_FEAT_DIM,
        "edge_feat_dim": EDGE_FEAT_DIM,
    }
    print_run_config(
        MODEL_CFG=MODEL_CFG,
        CLUSTER_CFG=CLUSTER_CFG,
        TRAIN_CFG=TRAIN_CFG,
        DATASET_CFG=DATASET_CFG,
        CIRCUIT_SOURCE_CFG=CIRCUIT_SOURCE_CFG,
        derived=derived,
    )

    # Providers (different seed bases => no overlap)
    train_provider = build_provider(CIRCUIT_SOURCE_CFG, seed_base=TRAIN_CFG["seed_base_train"])
    test_provider  = build_provider(CIRCUIT_SOURCE_CFG, seed_base=TRAIN_CFG["seed_base_test"])

    # Datasets
    train_dataset = CircuitDataset(
        train_provider,
        n_samples=TRAIN_CFG["n_samples_train"],
        segment_mode=DATASET_CFG["segmentation_mode"],
        segment_threshold=DATASET_CFG["segment_threshold"],
        w_short=w_short,
        w_long=w_long,
        device=device,
    )
    test_dataset = CircuitDataset(
        test_provider,
        n_samples=TRAIN_CFG["n_samples_test"],
        segment_mode=DATASET_CFG["segmentation_mode"],
        segment_threshold=DATASET_CFG["segment_threshold"],
        w_short=w_short,
        w_long=w_long,
        device=device,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=TRAIN_CFG["batch_size"],
        shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=TRAIN_CFG["batch_size"],
        shuffle=False, collate_fn=collate_fn, num_workers=0,
    )
    print(f"Train: {len(train_dataset)} circuits,  Test: {len(test_dataset)} circuits")

    # ---- Build models ----
    evol_model = EvolvingGNN(
        node_feat_dim  = NODE_FEAT_DIM,
        edge_feat_dim  = EDGE_FEAT_DIM,
        mlp_hidden_dim = MODEL_CFG["mlp_hidden_dim"],
        mlp_out_dim    = MODEL_CFG["mlp_out_dim"],
        gnn_out_dim    = MODEL_CFG["gnn_out_dim"],
        gru_hidden_dim = MODEL_CFG["gru_hidden_dim"],
        heads          = MODEL_CFG["heads"],
        dropout        = MODEL_CFG["dropout"],
        bptt_steps     = MODEL_CFG["bptt_steps"],
        activation     = MODEL_CFG["activation"],
    ).to(device)

    cluster_module = SegmentClustering(
        hidden_dim   = evol_model.rnn_hidden_dim,
        num_clusters = K,
        temperature  = CLUSTER_CFG["temperature"],
    ).to(device)

    total_cost_module = TotalCost(config).to(device)
    cap_penalty_module = CapacityPenalty(total_cost_module, config).to(device)

    print(f"Capacity penalty: lambda_cap={cap_penalty_module.lambda_cap.item():.6f}, "
          f"caps={cap_penalty_module.cap.tolist()}, beta={cap_penalty_module.beta}")

    optimizer = torch.optim.Adam(
        list(evol_model.parameters()) + list(cluster_module.parameters()),
        lr=TRAIN_CFG["lr"],
    )

    # ---- Fixed sample for per-epoch debugging ----
    # Tensors are already on device (dataset handles placement)
    fixed_layer_data_list, fixed_segments, fixed_rep = train_dataset[0]

    # ---- Init debug ----
    with torch.no_grad():
        evol_model.eval()
        cluster_module.eval()
        h_seq, _ = evol_model(fixed_layer_data_list)
        P_seq    = cluster_module(h_seq)
        print("INIT P_start(q0, layer0) =", P_seq[0][0].cpu().numpy())

    print("Pre-train prototypes mean:", cluster_module.head.cluster_prototypes.mean().item())
    print("Pre-train prototypes std:",  cluster_module.head.cluster_prototypes.std().item())

    # ---- Training Loop ----
    train_losses, test_losses = [], []

    for epoch in tqdm(range(TRAIN_CFG["n_epochs"]), desc="Epochs"):
        total_cost_module.set_epoch(epoch)
        evol_model.train()
        cluster_module.train()

        epoch_train_loss  = 0.0
        epoch_cap_penalty = 0.0

        for batch in train_loader:
            batch_loss = 0.0
            batch_cap  = 0.0
            batch_count = 0

            for layer_data_list, segments, rep in batch:
                loss, per_seg, cap_val = train_step(
                    evol_model, cluster_module, total_cost_module,
                    layer_data_list, segments, rep, optimizer,
                    capacity_penalty=cap_penalty_module,
                )
                batch_loss  += loss
                batch_cap   += cap_val
                batch_count += 1

            epoch_train_loss  += batch_loss  / batch_count
            epoch_cap_penalty += batch_cap   / batch_count

        avg_train_loss  = epoch_train_loss  / len(train_loader)
        avg_cap_penalty = epoch_cap_penalty / len(train_loader)
        train_losses.append(avg_train_loss)

        # ---- Per-epoch fixed-circuit debug ----
        with torch.no_grad():
            evol_model.eval()
            cluster_module.eval()

            h_seq, z_seq = evol_model(fixed_layer_data_list)
            h0 = h_seq[0]
            print(f"h0 mean: {h0.mean().item():.4f}  std: {h0.std().item():.4f}")

            P_seq    = cluster_module(h_seq)
            cost_out = total_cost_module(P_seq, fixed_segments, fixed_rep)
            fixed_loss = cost_out["total_cost"].item()

            cap_out    = cap_penalty_module(P_seq)
            fixed_cap  = cap_out["penalty"].item()
            fixed_excess = cap_out["per_layer_excess"]

            T = len(P_seq)
            P_start = P_seq[0][0]
            P_mid   = P_seq[T // 2][0]
            P_end   = P_seq[-1][0]

        print(f"Epoch {epoch}: C_total={fixed_loss:.4f}, R_cap={fixed_cap:.6f}, "
              f"avg_cap_penalty={avg_cap_penalty:.6f}")
        print(f"  layers_with_excess={int((fixed_excess > 0).sum())}/{len(fixed_excess)}")
        print("  P_start(q0, layer0) =", P_start.detach().cpu().numpy())
        print("  P_mid  (q0, layerM) =", P_mid.detach().cpu().numpy())
        print("  P_end  (q0, layerT) =", P_end.detach().cpu().numpy())

        # Test every 10 epochs
        if epoch % 10 == 0:
            test_loss, test_cap, test_per_seg = evaluate_model(
                evol_model, cluster_module, total_cost_module,
                test_loader, device, capacity_penalty=cap_penalty_module,
            )
            test_losses.append(test_loss)
            print(f"Epoch {epoch:3d}: train={avg_train_loss:.4f}, test={test_loss:.4f}, "
                  f"test_R_cap={test_cap:.6f}")
            print(f"  Test per_layer mean: {test_per_seg.mean():.4f}")

    torch.save(evol_model.state_dict(),    "evol_model_final.pt")
    torch.save(cluster_module.state_dict(), "cluster_head_final.pt")

    print("Post-train prototypes mean:", cluster_module.head.cluster_prototypes.mean().item())
    print("Post-train prototypes std:",  cluster_module.head.cluster_prototypes.std().item())

    # ---- Plot ----
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train", alpha=0.8)
    if test_losses:
        plt.plot(np.arange(0, len(test_losses) * 10, 10), test_losses, label="Test", alpha=0.8)
    plt.xlabel("Epoch"); plt.ylabel("Average Total Cost")
    plt.title("Training Curve"); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.boxplot(
        [train_losses[-10:], test_losses[-1:] if test_losses else []],
        labels=["Train (last 10)", "Test (last)"],
    )
    plt.ylabel("Total Cost"); plt.title("Loss Distribution")

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nFinal Results:")
    print(f"Train loss: {train_losses[-1]:.4f}")
    print(f"Test loss:  {test_losses[-1]:.4f}" if test_losses else "Test loss: N/A")

    return evol_model, cluster_module, total_cost_module


if __name__ == "__main__":
    model, cluster, cost = main()
