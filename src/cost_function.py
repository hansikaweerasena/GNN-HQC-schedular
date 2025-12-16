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
    """
    Execution cost using per-qubit soft tech assignments.

    P_seq[t]: [num_qubits, K] soft tech probs for segment t.
    For each gate g in segment t, touching qubits Q_g:
        E[cost(g)] = sum_{q in Q_g} sum_k P_{q,t,k} * exec_cost[k]
    """

    def __init__(self, tech_costs: List[TechCosts]):
        super().__init__()
        self.K = len(tech_costs)
        self.exec_cost_1q = nn.Parameter(
            torch.tensor([tc.execution_cost_1q for tc in tech_costs], dtype=torch.float32),
            requires_grad=False,
        )
        self.exec_cost_2q = nn.Parameter(
            torch.tensor([tc.execution_cost_2q for tc in tech_costs], dtype=torch.float32),
            requires_grad=False,
        )


    def forward(
        self,
        P_seq: List[torch.Tensor],  # list of [num_qubits, K]
        segments,                   # List[Segment]
        circuit,                    # CircuitRepresentation
    ) -> Dict[str, torch.Tensor]:
        device = P_seq[0].device
        dtype = P_seq[0].dtype

        total_exec = torch.tensor(0.0, device=device, dtype=dtype)
        per_segment = []

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
                        exec_cost_t += expected_cost_q

            total_exec += exec_cost_t

            per_segment.append(
                {
                    "segment_idx": seg.segment_idx,
                    "execution_cost": float(exec_cost_t.detach().cpu()),
                }
            )

        return {
            "execution_cost": total_exec,
            "per_segment_costs": per_segment,
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
        P_seq: List[torch.Tensor],  # list of [num_qubits, K]
        segments,                   # List[Segment]
        circuit,                    # CircuitRepresentation
    ):
        device = P_seq[0].device
        dtype = P_seq[0].dtype

        total_idle = torch.tensor(0.0, device=device, dtype=dtype)
        per_segment = []

        for t, (P_t, seg) in enumerate(zip(P_seq, segments)):
            # P_t: [N, K]
            N, K = P_t.shape
            assert K == self.K

            # compute idle_layers for this segment
            idle_layers = [0] * N  # idle layer count per qubit

            for layer_idx in seg.layers:
                layer = circuit.layers[layer_idx]
                active_qubits = set()
                for gate_name, qubits in layer.gates:
                    for q in qubits:
                        active_qubits.add(q)
                # for qubits not in active_qubits -> idle in this layer
                for q in range(N):
                    if q not in active_qubits:
                        idle_layers[q] += 1

            idle_layers_t = torch.tensor(idle_layers, device=device, dtype=dtype)  # [N]

            # expected idle cost per qubit: sum_k P_{q,k} * idle_cost[k]
            exp_idle_per_qubit = (P_t * self.idle_cost.to(device)).sum(dim=1)  # [N]

            # multiply by idle_layers[q], then sum over qubits
            idle_cost_t = (idle_layers_t * exp_idle_per_qubit).sum()

            total_idle += idle_cost_t

            per_segment.append(
                {
                    "segment_idx": seg.segment_idx,
                    "idle_cost": float(idle_cost_t.detach().cpu()),
                }
            )

        return {
            "idle_cost": total_idle,
            "per_segment_costs": per_segment,
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
    ):
        device = P_seq[0].device
        dtype = P_seq[0].dtype

        total_move = torch.tensor(0.0, device=device, dtype=dtype)
        per_segment = []

        P_prev = None

        for t, P_t in enumerate(P_seq):
            N, K = P_t.shape
            assert K == self.K

            if P_prev is None:
                move_cost_t = torch.tensor(0.0, device=device, dtype=dtype)
            else:
                # [N, K]
                diff = torch.abs(P_t - P_prev)
                # expected move per qubit: sum_k diff_{q,k} * move_cost[k]
                move_per_qubit = (diff * self.move_cost.to(device)).sum(dim=1)  # [N]
                move_cost_t = move_per_qubit.sum()

            total_move += move_cost_t

            per_segment.append(
                {
                    "segment_idx": t,
                    "movement_cost": float(move_cost_t.detach().cpu()),
                }
            )

            P_prev = P_t

        return {
            "movement_cost": total_move,
            "per_segment_costs": per_segment,
        }


class TotalCost(nn.Module):
    """
    Combine execution, idle, and movement costs into a single scalar.

    Uses:
    - ExecCost (per-qubit, per-gate)
    - IdleCost (per-qubit, per-idle-layer)
    - MovementCost (per-qubit, per-change-in-tech)
    """

    def __init__(
        self,
        exec_costs: List[float],
        idle_costs: List[float],
        move_costs: List[float],
    ):
        super().__init__()
        assert len(exec_costs) == len(idle_costs) == len(move_costs)
        self.K = len(exec_costs)
        self.exec_cost_module = ExecCost(
            [TechCosts(execution_cost_per_gate=c) for c in exec_costs]
        )
        self.idle_cost_module = IdleCost(idle_costs)
        self.move_cost_module = MovementCost(move_costs)

    def forward(
        self,
        P_seq: List[torch.Tensor],  # [T] of [num_qubits, K]
        segments,
        circuit,
    ) -> Dict[str, torch.Tensor]:
        # Execution
        exec_res = self.exec_cost_module(P_seq, segments, circuit)
        # Idle
        idle_res = self.idle_cost_module(P_seq, segments, circuit)
        # Movement (only needs P_seq)
        move_res = self.move_cost_module(P_seq)

        total_exec = exec_res["execution_cost"]
        total_idle = idle_res["idle_cost"]
        total_move = move_res["movement_cost"]

        total_cost = total_exec + total_idle + total_move

        return {
            "total_cost": total_cost,
            "execution_cost": total_exec,
            "idle_cost": total_idle,
            "movement_cost": total_move,
            "exec_per_segment": exec_res["per_segment_costs"],
            "idle_per_segment": idle_res["per_segment_costs"],
            "move_per_segment": move_res["per_segment_costs"],
        }
