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
    f_comm  = torch.tensor(comm.get("f_comm", 1.0), dtype=dtype)   # remote entanglement primitive success
    f_move  = torch.tensor(comm.get("f_move", 1.0), dtype=dtype)   # inter-segment movement primitive success
    t_remote = torch.tensor(comm.get("t_remote", 0.0), dtype=dtype) # optional; may be unused in v3
    return {"f_comm": f_comm, "f_move": f_move, "t_remote": t_remote}


def _parse_timing_buffers(config: Dict[str, Any], dtype=torch.float32) -> Dict[str, torch.Tensor]:
    timing = config.get("timing", {})
    delta = torch.tensor(timing.get("delta", 1.0), dtype=dtype)  # per-layer time proxy (LaTeX δ)
    return {"delta": delta}


def _neglog_clamped(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Stable -log(x) for probabilities.
    Clamps to [eps, 1.0] so we never take log(0) or log(>1).
    """
    return -torch.log(torch.clamp(x, min=eps, max=1.0))

class TotalCost(nn.Module):
    """
    Phase-1: parameterized, device-safe init.
    Forward interface remains: (P_seq, segments, circuit) -> dict with total_cost, per_segment_total.

    For now we keep legacy exec/idle/move costs from config['legacy_costs'] to keep pipeline running.
    Next phases will replace these modules with probabilistic (LaTeX) versions using the stored buffers.
    """

    def __init__(self, config: Dict[str, Any], dtype=torch.float32):
        super().__init__()

        # --- Parse profiles (parameterized inputs) ---
        tech_bufs = _parse_tech_buffers(config, dtype=dtype)
        comm_bufs = _parse_comm_buffers(config, dtype=dtype)
        timing_bufs = _parse_timing_buffers(config, dtype=dtype)

        self.tech_names = tech_bufs["names"]
        self.K = len(self.tech_names)

        # --- Register tech buffers (interpreted as success probs / timescales) ---
        self.register_buffer("F1q", tech_bufs["F1q"])   # treat as p_{1q}^k in LaTeX
        self.register_buffer("F2q", tech_bufs["F2q"])   # treat as p_{2q}^k
        self.register_buffer("Fm",  tech_bufs["Fm"])    # treat as p_m^k
        self.register_buffer("T2",  tech_bufs["T2"])    # treat as T^k (effective coherence timescale)
        self.register_buffer("t1q", tech_bufs["t1q"])
        self.register_buffer("t2q", tech_bufs["t2q"])
        self.register_buffer("tm",  tech_bufs["tm"])
        self.register_buffer("rho", tech_bufs["rho"])

        # --- Register comm/timing buffers ---
        self.register_buffer("f_comm", comm_bufs["f_comm"])
        self.register_buffer("f_move", comm_bufs["f_move"])
        self.register_buffer("t_remote", comm_bufs["t_remote"])  # optional
        self.register_buffer("delta", timing_bufs["delta"])

        # --- NEW: Precompute LaTeX additive failure costs (negative log success) ---
        # c_{1q}^k = -log(p_{1q}^k), etc.
        self.register_buffer("c1q", _neglog_clamped(self.F1q))       # [K]
        self.register_buffer("c2q", _neglog_clamped(self.F2q))       # [K]
        self.register_buffer("cm",  _neglog_clamped(self.Fm))        # [K]
        self.register_buffer("ccomm", _neglog_clamped(self.f_comm))  # scalar
        self.register_buffer("cmove", _neglog_clamped(self.f_move))  # scalar

    def forward(
        self,
        P_seq: List[torch.Tensor],
        segments,
        circuit,
        debug: bool = False,
    ) -> Dict[str, torch.Tensor]:

        device = P_seq[0].device
        dtype = P_seq[0].dtype
        S = len(P_seq)

        # PLACEHOLDER for Phase A: keep interface stable.
        # Next phases will compute these from circuit segment stats per LaTeX.
        per_segment_exec = torch.zeros(S, device=device, dtype=dtype)
        per_segment_idle = torch.zeros(S, device=device, dtype=dtype)
        per_segment_comm = torch.zeros(S, device=device, dtype=dtype)
        per_segment_move = torch.zeros(S, device=device, dtype=dtype)

        per_segment_total = per_segment_exec + per_segment_idle + per_segment_comm + per_segment_move
        total_cost = per_segment_total.sum()

        if debug:
            print("[TotalCost v3] Phase A placeholder forward. Costs are zero until exec/idle/comm/move are implemented.")

        return {
            "total_cost": total_cost,
            "per_segment_total": per_segment_total,

            # interpretability (safe additions)
            "per_segment_exec": per_segment_exec,
            "per_segment_idle": per_segment_idle,
            "per_segment_comm": per_segment_comm,
            "per_segment_move": per_segment_move,
        }