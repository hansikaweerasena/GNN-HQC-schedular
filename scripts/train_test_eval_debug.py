import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from tqdm import tqdm
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.circuit_generation import generate_random_circuit_custom
from src.circuit_representation import CircuitRepresentation
from src.circuit_segmentation import segment_circuit
from src.qubit_interaction_graph import build_segment_graph_arrays
from src.evolving_gnn import EvolvingGNN
from src.clustering_head import SegmentClustering
from src.cost_function import TotalCost
from src.train_utils import train_step


# ADD THIS FUNCTION (from your single test script)
def build_segment_data_list(rep, segments):
    per_segment_graphs = build_segment_graph_arrays(rep, segments)
    segment_data_list = []
    for seg_id, x_s, edge_index_s, edge_attr_s in per_segment_graphs:
        x_t = torch.tensor(x_s, dtype=torch.float32)
        ei_t = torch.tensor(edge_index_s, dtype=torch.long)
        ea_t = torch.tensor(edge_attr_s, dtype=torch.float32)
        segment_data_list.append(Data(x=x_t, edge_index=ei_t, edge_attr=ea_t))
    return segment_data_list


class CircuitDataset(Dataset):
    def __init__(self, n_samples=1000, n_qubits=10, depth=20, gate_density=0.3, seed_base=42, two_qubit_bounds=None):
        self.n_samples = n_samples
        self.n_qubits = n_qubits
        self.depth = depth
        self.gate_density = gate_density
        self.seed_base = seed_base
        self.two_qubit_bounds = two_qubit_bounds

        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        seed = self.seed_base + idx  # Sequential seeds = reproducible!
        rng = np.random.RandomState(seed + 10000)

        if self.two_qubit_bounds is not None:
            low, high = self.two_qubit_bounds
            two_qubit_ratio = np.random.uniform(low, high)
        else:
            two_qubit_ratio = 0.5 # default
        qc = generate_random_circuit_custom(
            n_qubits=self.n_qubits,
            depth=self.depth,
            gate_density=self.gate_density,
            seed=seed,
            two_qubit_ratio=two_qubit_ratio
        )
        rep = CircuitRepresentation(qc)
        segments, seg_ids = segment_circuit(rep.layers, threshold=0.3)
        segment_data_list = build_segment_data_list(rep, segments)
        return segment_data_list, segments, rep

def collate_fn(batch):
    # Variable length segments, return as-is
    return batch

def evaluate_model(model, cluster_module, cost_module, test_loader, device):
    model.eval()
    cluster_module.eval()
    total_loss = 0
    all_per_seg = []
    
    with torch.no_grad():
        for batch in test_loader:
            for segment_data_list, segments, rep in batch:
                loss, per_seg = train_step(
                    model, cluster_module, cost_module, 
                    segment_data_list, segments, rep, 
                    optimizer=None  # No optimization in eval
                )
                total_loss += loss
                all_per_seg.append(per_seg.cpu().numpy())
    
    return total_loss / len(test_loader.dataset), np.concatenate(all_per_seg)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    N_SAMPLES_TRAIN = 800
    N_SAMPLES_TEST = 200
    BATCH_SIZE = 4  # Small batches due to variable segment lengths
    N_EPOCHS = 50
    
    # Datasets (different seed bases = no overlap)
    train_dataset = CircuitDataset(n_samples=N_SAMPLES_TRAIN, seed_base=42, two_qubit_bounds=(0.1, 0.9))
    test_dataset = CircuitDataset(n_samples=N_SAMPLES_TEST, seed_base=1000, two_qubit_bounds=(0.1, 0.9))

    fixed_segment_data_list, fixed_segments, fixed_rep = train_dataset[0]
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    
    print(f"Train: {len(train_dataset)} circuits, Test: {len(test_dataset)} circuits")
    
    # ===== Build models (using first train sample for dims) =====
    sample_data = train_dataset[0]
    sample_segment_data_list = sample_data[0]
    in_dim_node = sample_segment_data_list[0].x.size(1)
    in_dim_edge = (sample_segment_data_list[0].edge_attr.size(1) 
                  if sample_segment_data_list[0].edge_attr.numel() > 0 else 0)
    
    evol_model = EvolvingGNN(
        in_dim_node=in_dim_node,
        in_dim_edge=in_dim_edge,
        gnn_hidden_dim=32,
        gnn_out_dim=16,
        rnn_hidden_dim=32,
        heads=4,
    ).to(device)
    
    K = 2
    cluster_module = SegmentClustering(
        hidden_dim=evol_model.rnn_hidden_dim,
        num_clusters=K,
        temperature=5.0,
    ).to(device)
    
    # Cost module
    exec_costs_1q = [0.05, 0.15]
    exec_costs_2q = [0.20, 0.05]
    idle_costs = [0.10, 0.12]
    move_costs = [0.03, 0.03]
    
    total_cost_module = TotalCost(exec_costs_1q, exec_costs_2q, idle_costs, move_costs).to(device)
    total_cost_module = TotalCost(config).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        list(evol_model.parameters()) + list(cluster_module.parameters()),
        lr=1e-4,
    )
    
    # Training history
    train_losses, test_losses = [], []

    # ---- Init debug BEFORE training loop ----
    with torch.no_grad():
        evol_model.eval()
        cluster_module.eval()
        h_seq, _ = evol_model(fixed_segment_data_list)
        P_seq = cluster_module(h_seq)
        print("INIT P_start(q0, seg0) =", P_seq[0][0].cpu().numpy())
    
    # Pre-train prototypes
    print("Pre-train prototypes mean:", cluster_module.head.cluster_prototypes.mean().item())
    print("Pre-train prototypes std:", cluster_module.head.cluster_prototypes.std().item())

    # ===== Training Loop =====
    for epoch in tqdm(range(N_EPOCHS), desc="Epochs"):
        evol_model.train()
        cluster_module.train()
        
        epoch_train_loss = 0
        num_circuits = 0
        
        for batch in train_loader:
            batch_loss = 0
            batch_count = 0
            
            for segment_data_list, segments, rep in batch:
                loss, per_seg = train_step(
                    evol_model, cluster_module, total_cost_module,
                    segment_data_list, segments, rep, optimizer
                )
                batch_loss += loss
                batch_count += 1
            
            epoch_train_loss += batch_loss / batch_count
            num_circuits += batch_count
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

         # ---- Fixed-circuit debug ----
        with torch.no_grad():
            evol_model.eval()
            cluster_module.eval()

            h_seq, z_seq = evol_model(fixed_segment_data_list)
            h0 = h_seq[0]  # [N, H]
            print("h0 mean:", h0.mean().item(), "std:", h0.std().item())
            print("h0[0][:5]:", h0[0][:5].detach().cpu().numpy())
            P_seq = cluster_module(h_seq)          # list length T, each [N, K]
            cost_out = total_cost_module(P_seq, fixed_segments, fixed_rep)
            fixed_loss = cost_out["total_cost"].item()

            # pick qubit 0, first/middle/last segment
            P_start = P_seq[0][0]                  # [K]
            P_mid   = P_seq[len(P_seq)//2][0]      # [K]
            P_end   = P_seq[-1][0]                 # [K]

        print(f"Epoch {epoch}: fixed_train_circuit_loss={fixed_loss:.4f}")
        print("  P_start(q0, seg0) =", P_start.detach().cpu().numpy())
        print("  P_mid  (q0, segM) =", P_mid.detach().cpu().numpy())
        print("  P_end  (q0, segT) =", P_end.detach().cpu().numpy())

        
        # Test every 10 epochs
        if epoch % 10 == 0:
            test_loss, test_per_seg = evaluate_model(
                evol_model, cluster_module, total_cost_module, test_loader, device
            )
            test_losses.append(test_loss)
            
            print(f"Epoch {epoch:3d}: train={avg_train_loss:.4f}, test={test_loss:.4f}")
            print(f"Test per_segment mean: {test_per_seg.mean():.4f}")
    
    torch.save(evol_model.state_dict(), "evol_model_final.pt")
    torch.save(cluster_module.state_dict(), "cluster_head_final.pt")

    # Post-train prototypes  
    print("Post-train prototypes mean:", cluster_module.head.cluster_prototypes.mean().item())
    print("Post-train prototypes std:", cluster_module.head.cluster_prototypes.std().item())

    # ===== Plot Results =====
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train", alpha=0.8)
    if len(test_losses) > 0:
        plt.plot(np.arange(0, len(test_losses)*10, 10), test_losses, label="Test", alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Average Total Cost")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([train_losses[-10:], test_losses[-1:] if test_losses else []], 
                labels=["Train (last 10)", "Test (last)"])
    plt.ylabel("Total Cost")
    plt.title("Loss Distribution")
    
    plt.tight_layout()
    plt.savefig("training_results.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nFinal Results:")
    print(f"Train loss: {train_losses[-1]:.4f}")
    print(f"Test loss:  {test_losses[-1]:.4f}" if test_losses else "Test loss: N/A")
    
    return evol_model, cluster_module, total_cost_module

if __name__ == "__main__":
    model, cluster, cost = main()
