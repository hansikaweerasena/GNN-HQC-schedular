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
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
from typing import Any, Optional


from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Optional

# need not to explicitly be a nn.Module, but doing so allows us to easily register buffers for tech profiles and other config-derived tensors
class SegmentStatsExtractor(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        gate_names_cfg = config.get("gate_names", {})
        measure_list = gate_names_cfg.get("measure", ["measure", "meas", "m"])
        self.measure_gate_names = set(str(x).strip().lower() for x in measure_list)

        gamma_cfg = config.get("connectivity_proxy", {})
        self.gamma_mode = gamma_cfg.get("mode", "none")

        self._cached_key = None
        self._cached_val = None

    def _is_measure_gate(self, gate_name: Any) -> bool:
        return str(gate_name).strip().lower() in self.measure_gate_names

    def _compute_gamma_value(
        self,
        *,
        N: int,
        L_s: int,
        edge_counts: Dict[Tuple[int, int], int],
    ) -> float:
        mode = (self.gamma_mode or "none").lower()
        if mode == "none":
            return 0.0
        if mode == "edge_density":
            denom = max(1.0, (N * (N - 1)) / 2.0)
            return float(len(edge_counts)) / denom
        if mode == "twoq_per_layer":
            total_2q = float(sum(edge_counts.values()))
            return total_2q / max(1.0, float(L_s))
        return 0.0

    def forward(
        self,
        segments,
        circuit,
        N: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, Any]:
        cache_key = (id(segments), id(circuit), int(N), str(device), str(dtype))
        if self._cached_key == cache_key and self._cached_val is not None:
            return self._cached_val

        S = len(segments)
        n1q = torch.zeros((S, N), device=device, dtype=dtype)
        nm  = torch.zeros((S, N), device=device, dtype=dtype)
        L   = torch.zeros((S,), device=device, dtype=dtype)
        gamma = torch.zeros((S,), device=device, dtype=dtype)
        edges: List[Dict[str, torch.Tensor]] = []

        for s, seg in enumerate(segments):
            layers = getattr(seg, "layers", seg)
            L_s = len(layers)
            L[s] = float(L_s)

            edge_counts: Dict[Tuple[int, int], int] = defaultdict(int)

            for layer in layers:
                gates = getattr(layer, "gates", [])
                for gate_name, qubits in gates:
                    if qubits is None:
                        continue
                    qlist = list(qubits)

                    if len(qlist) == 1:
                        u = int(qlist[0])
                        if 0 <= u < N:
                            if self._is_measure_gate(gate_name):
                                nm[s, u] += 1.0
                            else:
                                n1q[s, u] += 1.0

                    elif len(qlist) == 2:
                        u = int(qlist[0]); v = int(qlist[1])
                        if u == v:
                            continue
                        if not (0 <= u < N and 0 <= v < N):
                            continue
                        a, b = (u, v) if u < v else (v, u)
                        edge_counts[(a, b)] += 1

                    else:
                        # Multi-qubit gates ignored for now to avoid silent assumption breaks
                        continue

            if len(edge_counts) == 0:
                e_u = torch.empty((0,), device=device, dtype=torch.long)
                e_v = torch.empty((0,), device=device, dtype=torch.long)
                e_w = torch.empty((0,), device=device, dtype=dtype)
            else:
                pairs = list(edge_counts.keys())
                weights = [edge_counts[p] for p in pairs]
                e_u = torch.tensor([p[0] for p in pairs], device=device, dtype=torch.long)
                e_v = torch.tensor([p[1] for p in pairs], device=device, dtype=torch.long)
                e_w = torch.tensor(weights, device=device, dtype=dtype)

            edges.append({"u": e_u, "v": e_v, "w": e_w})

            gamma[s] = float(self._compute_gamma_value(N=N, L_s=L_s, edge_counts=edge_counts))

        stats = {"L": L, "n1q": n1q, "nm": nm, "edges": edges, "gamma": gamma}
        self._cached_key = cache_key
        self._cached_val = stats
        return stats

class ExecCostV3(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        P_seq: List[torch.Tensor],         # list of [N,K]
        stats: Dict[str, Any],             # from SegmentStatsExtractor
        *,
        c1q: torch.Tensor,                 # [K]
        c2q: torch.Tensor,                 # [K]
        cm: torch.Tensor,                  # [K]
        rho: torch.Tensor,                 # [K]
    ) -> Dict[str, torch.Tensor]:

        device = P_seq[0].device
        dtype = P_seq[0].dtype
        S = len(P_seq)

        W = torch.stack(P_seq, dim=0)  # [S,N,K]

        # 1Q: Σ_u n1q[s,u] * <w, c1q>
        E_c1q = torch.einsum("suk,k->su", W, c1q.to(dtype))
        C1q = (stats["n1q"] * E_c1q).sum(dim=1)  # [S]

        # Meas: Σ_u nm[s,u] * <w, cm>
        E_cm = torch.einsum("suk,k->su", W, cm.to(dtype))
        Cm = (stats["nm"] * E_cm).sum(dim=1)    # [S]

        # Local 2Q with inflation: Σ_(u,v) ω_uv * Σ_k w_u,k w_v,k (1 + rho_k Γ[s]) c2q_k
        Gamma = stats["gamma"].to(dtype)  # [S]
        infl_base = rho.to(dtype)         # [K]
        c2q = c2q.to(dtype)

        C2q_local = torch.zeros((S,), device=device, dtype=dtype)

        for s in range(S):
            e = stats["edges"][s]
            u_idx = e["u"]
            v_idx = e["v"]
            omega = e["w"].to(dtype)

            if u_idx.numel() == 0:
                continue

            Wu = W[s, u_idx, :]  # [E,K]
            Wv = W[s, v_idx, :]  # [E,K]
            joint = Wu * Wv      # [E,K]

            infl = (1.0 + infl_base * Gamma[s])     # [K]
            per_edge_cost = torch.einsum("ek,k->e", joint, infl * c2q)  # [E]

            C2q_local[s] = (omega * per_edge_cost).sum()

        Cexec = C1q + Cm + C2q_local

        return {
            "per_segment_C1q": C1q,
            "per_segment_Cm": Cm,
            "per_segment_C2q_local": C2q_local,
            "per_segment_exec": Cexec,
        }
    

class IdleCostV3(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        P_seq: List[torch.Tensor],
        stats: Dict[str, Any],
        *,
        delta: torch.Tensor,     # scalar
        T2: torch.Tensor,        # [K]
    ) -> Dict[str, torch.Tensor]:
        S = len(P_seq)
        device = P_seq[0].device
        dtype = P_seq[0].dtype
        # placeholder until next phase
        return {"per_segment_idle": torch.zeros((S,), device=device, dtype=dtype)}


class CommMoveCostV3(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        P_seq: List[torch.Tensor],
        stats: Dict[str, Any],
        *,
        ccomm: torch.Tensor,     # scalar
        cmove: torch.Tensor,     # scalar
    ) -> Dict[str, torch.Tensor]:
        S = len(P_seq)
        device = P_seq[0].device
        dtype = P_seq[0].dtype
        # placeholder until later phases
        return {
            "per_segment_comm": torch.zeros((S,), device=device, dtype=dtype),
            "per_segment_move": torch.zeros((S,), device=device, dtype=dtype),
        }


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


def _normalize_gate_name(name: Any) -> str:
    return str(name).strip().lower()


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

        self.stats_extractor = SegmentStatsExtractor(config)
        self.exec_cost = ExecCostV3()
        self.idle_cost = IdleCostV3()
        self.comm_move_cost = CommMoveCostV3()

        # --- Segment parsing / stats configuration (python attributes, not buffers) ---
        gate_names_cfg = config.get("gate_names", {})
        measure_list = gate_names_cfg.get("measure", ["measure", "meas", "m"])
        self.measure_gate_names = set(_normalize_gate_name(x) for x in measure_list)

        gamma_cfg = config.get("connectivity_proxy", {})
        self.gamma_mode = gamma_cfg.get("mode", "none")  # default: no inflation unless you enable it


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
        N = P_seq[0].shape[0]

        stats = self.stats_extractor(segments, circuit, N=N, device=device, dtype=dtype)

        exec_out = self.exec_cost(P_seq, stats, c1q=self.c1q, c2q=self.c2q, cm=self.cm, rho=self.rho)
        idle_out = self.idle_cost(P_seq, stats, delta=self.delta, T2=self.T2)
        comm_out = self.comm_move_cost(P_seq, stats, ccomm=self.ccomm, cmove=self.cmove)

        per_segment_exec = exec_out["per_segment_exec"]
        per_segment_idle = idle_out["per_segment_idle"]
        per_segment_comm = comm_out["per_segment_comm"]
        per_segment_move = comm_out["per_segment_move"]

        per_segment_total = per_segment_exec + per_segment_idle + per_segment_comm + per_segment_move
        total_cost = per_segment_total.sum()

        out = {
            "total_cost": total_cost,
            "per_segment_total": per_segment_total,
            "per_segment_exec": per_segment_exec,
            "per_segment_idle": per_segment_idle,
            "per_segment_comm": per_segment_comm,
            "per_segment_move": per_segment_move,

            # exec interpretability
            "per_segment_C1q": exec_out["per_segment_C1q"],
            "per_segment_Cm": exec_out["per_segment_Cm"],
            "per_segment_C2q_local": exec_out["per_segment_C2q_local"],
        }

        if debug:
            out["debug_stats"] = {
                "L": stats["L"].detach(),
                "gamma": stats["gamma"].detach(),
                "n1q_sum": stats["n1q"].sum(dim=1).detach(),
                "nm_sum": stats["nm"].sum(dim=1).detach(),
                "num_edges": torch.tensor([e["w"].numel() for e in stats["edges"]], device=device, dtype=dtype),
                "twoq_ops": torch.tensor([float(e["w"].sum()) for e in stats["edges"]], device=device, dtype=dtype),
            }

        return out