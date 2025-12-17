# src/cost_function.py

from typing import List, Dict
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class TechCosts:
    execution_cost_1q: float        # cost contribution per qubit for 1-qubit gate
    execution_cost_2q: float        # cost contribution per qubit for 2-qubit gate


class ExecCost(nn.Module):
    def __init__(self, tech_costs: List[TechCosts]):
        super().__init__()
        self.K = len(tech_costs)
        self.exec_cost_1q = nn.Parameter(
            torch.tensor(
                [tc.execution_cost_1q for tc in tech_costs],
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.exec_cost_2q = nn.Parameter(
            torch.tensor(
                [tc.execution_cost_2q for tc in tech_costs],
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    def forward(
        self,
        P_seq: List[torch.Tensor],  # list of [num_qubits, K]
        segments,                   # List[Segment]
        circuit,                    # CircuitRepresentation
        debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        device = P_seq[0].device
        dtype = P_seq[0].dtype

        total_exec = torch.tensor(0.0, device=device, dtype=dtype)
        per_segment_exec = []  # list of tensors [scalar per segment]

        for t, (P_t, seg) in enumerate(zip(P_seq, segments)):
            # P_t: [num_qubits, K]
            N, K = P_t.shape
            assert K == self.K

            exec_cost_t = torch.tensor(0.0, device=device, dtype=dtype)

            # Go through all gates in this segment
            for layer_idx in seg.layers:
                layer = circuit.layers[layer_idx]
                for gate_name, qubits in layer.gates:
                    if len(qubits) == 1:
                        cost_vec = self.exec_cost_1q.to(device)  # [K]
                    elif len(qubits) == 2:
                        cost_vec = self.exec_cost_2q.to(device)  # [K]
                    else:
                        continue  # ignore exotic gates for now

                    for q in qubits:
                        # P_t[q]: [K]
                        expected_cost_q = (P_t[q] * cost_vec).sum()
                        exec_cost_t = exec_cost_t + expected_cost_q

            total_exec = total_exec + exec_cost_t
            per_segment_exec.append(exec_cost_t)

            if debug:
                # Safe debug print; does not affect gradients
                print(
                    f"[ExecCost] seg {seg.segment_idx} "
                    f"exec_cost_t = {exec_cost_t.detach().cpu().item():.4f}"
                )

        # [T] tensor of per-segment exec costs
        per_segment_exec_t = torch.stack(per_segment_exec)  # shape [T]

        return {
            "execution_cost": total_exec,           # scalar tensor
            "per_segment_costs": per_segment_exec_t,  # [T] tensor
        }


class IdleCost(nn.Module):
    """
    Idle cost using per-qubit soft tech assignments.


    For each segment t and qubit q:
    idle_layers[q] = # of layers in this segment where q has no gate
    E[idle(q,t)] = idle_layers[q] * sum_k P_{q,t,k} * idle_cost[k]
    """


    def __init__(self, idle_costs: List[float]):
        super().__init__()
        self.K = len(idle_costs)
        self.idle_cost = nn.Parameter(
        torch.tensor(idle_costs, dtype=torch.float32),
        requires_grad=False,
        )                   


    def forward(
        self,
        P_seq: List[torch.Tensor],
        segments,
        circuit,
        debug: bool = False,
    ):
        device = P_seq[0].device
        dtype = P_seq[0].dtype

        total_idle = torch.tensor(0.0, device=device, dtype=dtype)
        per_segment_idle = []  # list of tensors

        for t, (P_t, seg) in enumerate(zip(P_seq, segments)):
            N, K = P_t.shape
            assert K == self.K

            # Compute idle_layers[q] for this segment [Python list → tensor]
            idle_layers = [0] * N
            for layer_idx in seg.layers:
                layer = circuit.layers[layer_idx]
                active_qubits = set()
                for gate_name, qubits in layer.gates:
                    active_qubits.update(qubits)  # faster than nested loop
                for q in range(N):
                    if q not in active_qubits:
                        idle_layers[q] += 1
            
            idle_layers_t = torch.tensor(idle_layers, device=device, dtype=dtype)  # [N]

            # Expected idle cost per qubit: sum_k P_{q,k} * idle_cost[k]
            exp_idle_per_qubit = (P_t * self.idle_cost.to(device)).sum(dim=1)  # [N]

            # Segment total: sum_q (idle_layers[q] * exp_idle_per_qubit[q])
            idle_cost_t = (idle_layers_t * exp_idle_per_qubit).sum()

            total_idle += idle_cost_t
            per_segment_idle.append(idle_cost_t)

            if debug:
                print(f"[IdleCost] seg {seg.segment_idx} "
                    f"idle_layers_sum={idle_layers_t.sum().item():.1f} "
                    f"idle_cost_t={idle_cost_t.detach().cpu().item():.4f}")

        per_segment_idle_t = torch.stack(per_segment_idle)  # [T]

        return {
            "idle_cost": total_idle,
            "per_segment_costs": per_segment_idle_t,  # tensor [T]
        }

class MovementCost(nn.Module):
    """
    Movement cost using soft change in tech assignment between segments.


    For each qubit q, segments t-1 -> t:
    diff_{q,k} = |P_t[q,k] - P_{t-1}[q,k]|
    E[move(q,t)] = sum_k diff_{q,k} * move_cost[k]
    """


    def __init__(self, move_costs: List[float]):
        super().__init__()
        self.K = len(move_costs)
        self.move_cost = nn.Parameter(
        torch.tensor(move_costs, dtype=torch.float32),
        requires_grad=False,
        )

    def forward(
        self,
        P_seq: List[torch.Tensor],  # list of [num_qubits, K]
        debug: bool = False,
    ):
        device = P_seq[0].device
        dtype = P_seq[0].dtype

        total_move = torch.tensor(0.0, device=device, dtype=dtype)
        per_segment_move = []  # list of tensors

        P_prev = None
        for t, P_t in enumerate(P_seq):
            N, K = P_t.shape
            assert K == self.K

            if P_prev is None:
                move_cost_t = torch.tensor(0.0, device=device, dtype=dtype)
            else:
                # [N, K] L1 difference
                diff = torch.abs(P_t - P_prev)
                # Expected move per qubit: sum_k diff[q,k] * move_cost[k]
                move_per_qubit = (diff * self.move_cost.to(device)).sum(dim=1)  # [N]
                move_cost_t = move_per_qubit.sum()  # scalar

            total_move += move_cost_t
            per_segment_move.append(move_cost_t)

            if debug:
                print(f"[MoveCost] seg {t} "
                    f"move_cost_t={move_cost_t.detach().cpu().item():.4f} "
                    f"L1_change_avg={(torch.norm(P_t - P_prev, p=1, dim=1).mean().item() if P_prev is not None else 0):.4f}")

            P_prev = P_t

        per_segment_move_t = torch.stack(per_segment_move)  # [T]

        return {
            "movement_cost": total_move,           # scalar tensor
            "per_segment_costs": per_segment_move_t,  # [T] tensor
        }

class TotalCost(nn.Module):
    """
    Combine execution, idle, and movement costs into a single scalar.
    """
    def __init__(
        self,
        exec_costs_1q: List[float],
        exec_costs_2q: List[float],
        idle_costs: List[float],
        move_costs: List[float],
    ):
        super().__init__()
        assert len(exec_costs_1q) == len(exec_costs_2q) == len(idle_costs) == len(move_costs)
        self.K = len(exec_costs_1q)
        self.exec_cost_module = ExecCost(
            [TechCosts(execution_cost_1q=c1, execution_cost_2q=c2)
             for c1, c2 in zip(exec_costs_1q, exec_costs_2q)]
        )
        self.idle_cost_module = IdleCost(idle_costs)
        self.move_cost_module = MovementCost(move_costs)

    def forward(
        self,
        P_seq: List[torch.Tensor],  # [T] of [num_qubits, K]
        segments,
        circuit,
        debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # Run sub-modules
        exec_res = self.exec_cost_module(P_seq, segments, circuit, debug=debug)
        idle_res = self.idle_cost_module(P_seq, segments, circuit, debug=debug)
        move_res = self.move_cost_module(P_seq, debug=debug)

        # Totals (all tensors, gradients preserved)
        total_exec = exec_res["execution_cost"]
        total_idle = idle_res["idle_cost"]
        total_move = move_res["movement_cost"]
        total_cost = total_exec + total_idle + total_move

        # Per-segment totals (tensor addition, fully differentiable)
        per_segment_total = (
            exec_res["per_segment_costs"] + 
            idle_res["per_segment_costs"] + 
            move_res["per_segment_costs"]
        )

        if debug:
            print(f"[TotalCost] total={total_cost.detach().cpu().item():.4f}")
            print(f"[TotalCost] per_segment_total sum={per_segment_total.sum().detach().cpu().item():.4f}")
            print(f"[TotalCost] exec={total_exec.detach().cpu().item():.4f}")
            print(f"[TotalCost] idle={total_idle.detach().cpu().item():.4f}")
            print(f"[TotalCost] move={total_move.detach().cpu().item():.4f}")

        return {
            "total_cost": total_cost,
            "per_segment_total": per_segment_total,
        }