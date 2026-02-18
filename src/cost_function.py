# src/cost_function.py


# This module will define a differntiable, paramterized cost function which is used to BPTT for the the learned schedular.
# BPTT will collect sequences of soft tech assignments P_seq = [P_1, P_2, ..., P_T] where each P_t is a [num_qubits, K] tensor of probabilities.
# The cost function will compute a single scalar by combining:
# 1. Execution cost: based on expected gate costs given P_t and the circuit structure
# 2. Idle cost: based on expected idle time for qubits not involved in gates
# 3. Movement cost: based on expected changes in tech assignment between segments

from typing import List, Dict
from dataclasses import dataclass
import torch
import torch.nn as nn

class ExecCost(nn.Module):
    def __init__(self, exec_costs_1q: List[float], exec_costs_2q: List[float], dtype=torch.float32):
        super().__init__()
        assert len(exec_costs_1q) == len(exec_costs_2q)
        self.K = len(exec_costs_1q)

        self.register_buffer("exec_cost_1q", torch.tensor(exec_costs_1q, dtype=dtype))  # [K]
        self.register_buffer("exec_cost_2q", torch.tensor(exec_costs_2q, dtype=dtype))  # [K]

    def forward(
        self,
        P_seq: List[torch.Tensor],  # list of [num_qubits, K]
        segments,                   # List[Segment]
        circuit,                    # CircuitRepresentation
        debug: bool = False,
    ) -> Dict[str, torch.Tensor]:

        total_exec = torch.zeros((), device=P_seq[0].device, dtype=P_seq[0].dtype)
        per_segment_exec = [] # list of tensors [scalar per segment]

        for P_t, seg in zip(P_seq, segments):
            N, K = P_t.shape
            assert K == self.K

            exec_cost_t = torch.zeros((), device=P_t.device, dtype=P_t.dtype)

            # Go through all gates in this segment
            for layer_idx in seg.layers:
                layer = circuit.layers[layer_idx]
                for gate_name, qubits in layer.gates:
                    if len(qubits) == 1:
                        cost_vec = self.exec_cost_1q  # already on correct device
                    elif len(qubits) == 2:
                        cost_vec = self.exec_cost_2q
                    else:
                        continue  # ignore exotic gates for now

                    for q in qubits:
                        exec_cost_t = exec_cost_t + (P_t[q] * cost_vec).sum()

            total_exec = total_exec + exec_cost_t
            per_segment_exec.append(exec_cost_t)

            if debug:
                print(f"[ExecCost] seg {seg.segment_idx} exec_cost_t={exec_cost_t.detach().cpu().item():.4f}")

        return {
            "execution_cost": total_exec,
            "per_segment_costs": torch.stack(per_segment_exec),  # [T]
        }

class IdleCost(nn.Module):
    def __init__(self, idle_costs: List[float], dtype=torch.float32):
        super().__init__()
        self.K = len(idle_costs)
        self.register_buffer("idle_cost", torch.tensor(idle_costs, dtype=dtype))  # [K]

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
            exp_idle_per_qubit = (P_t * self.idle_cost).sum(dim=1)

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

    def __init__(self, move_costs: List[float], dtype=torch.float32):
        super().__init__()
        self.K = len(move_costs)
        self.register_buffer("move_cost", torch.tensor(move_costs, dtype=dtype))  # [K]


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
                move_per_qubit = (diff * self.move_cost).sum(dim=1)
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


from typing import Any, Optional

def _require(d: Dict[str, Any], path: str):
    """Fetch nested key path like 'gate_fidelity.f1q' and throw a clear error if missing."""
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f"Missing config key: {path}")
        cur = cur[k]
    return cur

def _parse_tech_buffers(config: Dict[str, Any], dtype=torch.float32) -> Dict[str, torch.Tensor]:
    techs = config["techs"]
    if not isinstance(techs, list) or len(techs) == 0:
        raise ValueError("config['techs'] must be a non-empty list")

    names = [t.get("name", f"tech{i}") for i, t in enumerate(techs)]

    F1q = torch.tensor([_require(t, "gate_fidelity.f1q") for t in techs], dtype=dtype)
    F2q = torch.tensor([_require(t, "gate_fidelity.f2q") for t in techs], dtype=dtype)
    Fm  = torch.tensor([_require(t, "gate_fidelity.fm")  for t in techs], dtype=dtype)

    T2  = torch.tensor([_require(t, "coherence.T2") for t in techs], dtype=dtype)

    t1q = torch.tensor([_require(t, "gate_time.t1q") for t in techs], dtype=dtype)
    t2q = torch.tensor([_require(t, "gate_time.t2q") for t in techs], dtype=dtype)
    tm  = torch.tensor([_require(t, "gate_time.tm")  for t in techs], dtype=dtype)

    rho = torch.tensor([_require(t, "routing.rho") for t in techs], dtype=dtype)

    return {
        "names": names,  # python list (not tensor)
        "F1q": F1q, "F2q": F2q, "Fm": Fm,
        "T2": T2,
        "t1q": t1q, "t2q": t2q, "tm": tm,
        "rho": rho,
    }

def _parse_comm_buffers(config: Dict[str, Any], dtype=torch.float32) -> Dict[str, torch.Tensor]:
    comm = config.get("comm", {})
    f_comm = torch.tensor(comm.get("f_comm", 1.0), dtype=dtype)     # default: no penalty
    t_remote = torch.tensor(comm.get("t_remote", 0.0), dtype=dtype) # default: no latency
    return {"f_comm": f_comm, "t_remote": t_remote}



class TotalCost(nn.Module):
    """
    Phase-1: parameterized, device-safe init.
    Forward interface remains: (P_seq, segments, circuit) -> dict with total_cost, per_segment_total.

    For now we keep legacy exec/idle/move costs from config['legacy_costs'] to keep pipeline running.
    Next phases will replace these modules with probabilistic (LaTeX) versions using the stored buffers.
    """

    def __init__(self, config: Dict[str, Any], dtype=torch.float32):
        super().__init__()

        # --- Parse tech + comm profiles (buffers for later probabilistic model) ---
        tech_bufs = _parse_tech_buffers(config, dtype=dtype)
        comm_bufs = _parse_comm_buffers(config, dtype=dtype)

        self.tech_names = tech_bufs["names"]
        self.K = len(self.tech_names)

        # Register tech buffers (device-safe constants)
        self.register_buffer("F1q", tech_bufs["F1q"])
        self.register_buffer("F2q", tech_bufs["F2q"])
        self.register_buffer("Fm",  tech_bufs["Fm"])
        self.register_buffer("T2",  tech_bufs["T2"])
        self.register_buffer("t1q", tech_bufs["t1q"])
        self.register_buffer("t2q", tech_bufs["t2q"])
        self.register_buffer("tm",  tech_bufs["tm"])
        self.register_buffer("rho", tech_bufs["rho"])

        # Register comm buffers
        self.register_buffer("f_comm", comm_bufs["f_comm"])
        self.register_buffer("t_remote", comm_bufs["t_remote"])

        # --- TEMP: legacy costs to keep training loop working until prob terms are implemented ---
        legacy = config.get("legacy_costs", None)
        if legacy is None:
            raise ValueError("Phase-1 requires config['legacy_costs'] temporarily to keep pipeline runnable.")

        exec_costs_1q = legacy["exec_costs_1q"]
        exec_costs_2q = legacy["exec_costs_2q"]
        idle_costs    = legacy["idle_costs"]
        move_costs    = legacy["move_costs"]

        assert len(exec_costs_1q) == self.K
        assert len(exec_costs_2q) == self.K
        assert len(idle_costs)    == self.K
        assert len(move_costs)    == self.K

        self.exec_cost_module = ExecCost(exec_costs_1q, exec_costs_2q, dtype=dtype)
        self.idle_cost_module = IdleCost(idle_costs, dtype=dtype)
        self.move_cost_module = MovementCost(move_costs, dtype=dtype)

    def forward(
        self,
        P_seq: List[torch.Tensor],
        segments,
        circuit,
        debug: bool = False,
    ) -> Dict[str, torch.Tensor]:

        exec_res = self.exec_cost_module(P_seq, segments, circuit, debug=debug)
        idle_res = self.idle_cost_module(P_seq, segments, circuit, debug=debug)
        move_res = self.move_cost_module(P_seq, debug=debug)

        total_exec = exec_res["execution_cost"]
        total_idle = idle_res["idle_cost"]
        total_move = move_res["movement_cost"]
        total_cost = total_exec + total_idle + total_move

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