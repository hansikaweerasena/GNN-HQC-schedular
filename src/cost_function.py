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

        # --- Segment parsing / stats configuration (python attributes, not buffers) ---
        gate_names_cfg = config.get("gate_names", {})
        measure_list = gate_names_cfg.get("measure", ["measure", "meas", "m"])
        self.measure_gate_names = set(_normalize_gate_name(x) for x in measure_list)

        gamma_cfg = config.get("connectivity_proxy", {})
        self.gamma_mode = gamma_cfg.get("mode", "none")  # default: no inflation unless you enable it

        # Optional: tiny cache for repeated calls on the same circuit/segments object
        self._cached_stats_key = None
        self._cached_stats_val = None


    def _extract_segment_stats(
        self,
        segments,
        circuit,
        N: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, Any]:
        """
        Extract LaTeX-v3 primitives per segment:
        L[s], n1q[s,u], nm[s,u], edges[s] = {u,v,w}, gamma[s]
        Output is ready for downstream cost computation on `device`.
        """

        # Simple cache keyed by object identity (stats depend only on circuit/segments, not on P_seq)
        cache_key = (id(segments), id(circuit), int(N))
        if self._cached_stats_key == cache_key and self._cached_stats_val is not None:
            return self._cached_stats_val

        S = len(segments)

        # Counts as float tensors (we will multiply by weights later)
        n1q = torch.zeros((S, N), device=device, dtype=dtype)
        nm  = torch.zeros((S, N), device=device, dtype=dtype)
        L   = torch.zeros((S,), device=device, dtype=dtype)
        gamma = torch.zeros((S,), device=device, dtype=dtype)

        edges: List[Dict[str, torch.Tensor]] = []

        for s, seg in enumerate(segments):
            layers = getattr(seg, "layers", seg)  # allow seg itself to be iterable as layers
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
                            if _is_measure_gate(gate_name, self.measure_gate_names):
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
                        # If multi-qubit gates appear, you can decide later how to decompose.
                        # For now, ignore to avoid silently breaking assumptions.
                        continue

            # Convert edge multiset to tensors
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

            # Γ(s) proxy (default 0)
            gamma_val = _compute_gamma_value(
                mode=self.gamma_mode,
                N=N,
                L_s=L_s,
                edge_counts=edge_counts,
            )
            gamma[s] = float(gamma_val)

        stats = {
            "L": L,               # [S]
            "n1q": n1q,           # [S, N]
            "nm": nm,             # [S, N]
            "edges": edges,       # list length S, each has u,v,w
            "gamma": gamma,       # [S]
        }

        self._cached_stats_key = cache_key
        self._cached_stats_val = stats
        return stats

    def forward(
        self,
        P_seq: List[torch.Tensor],
        segments,
        circuit,
        debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        
        N = P_seq[0].shape[0]
        stats = self._extract_segment_stats(
            segments=segments,
            circuit=circuit,
            N=N,
            device=device,
            dtype=dtype,
        )

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

        out = {
            "total_cost": total_cost,
            "per_segment_total": per_segment_total,
            "per_segment_exec": per_segment_exec,
            "per_segment_idle": per_segment_idle,
            "per_segment_comm": per_segment_comm,
            "per_segment_move": per_segment_move,
        }
        if debug:
            out["debug_stats"] = {
                "L": stats["L"].detach(),
                "gamma": stats["gamma"].detach(),
                "n1q_sum": stats["n1q"].sum(dim=1).detach(),
                "nm_sum": stats["nm"].sum(dim=1).detach(),
                "num_edges": torch.tensor([e["w"].numel() for e in stats["edges"]], device=device, dtype=dtype),
                "twoq_ops": torch.tensor([e["w"].sum().item() for e in stats["edges"]], device=device, dtype=dtype),
            }
        return out