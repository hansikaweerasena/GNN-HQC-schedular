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
import networkx as nx
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict
from configs.scheduler_config import DATASET_CFG  # type: ignore



# need not to explicitly be a nn.Module, but doing so allows us to easily register buffers for tech profiles and other config-derived tensors
class SegmentStatsExtractor(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        gate_names_cfg = config.get("gate_names", {})
        measure_list = gate_names_cfg.get("measure", ["measure", "meas", "m"])
        self.measure_gate_names = set(str(x).strip().lower() for x in measure_list)

        gamma_cfg = config.get("connectivity_proxy", {})
        self.gamma_mode = gamma_cfg.get("mode", "none")
        self.gamma_eps = float(gamma_cfg.get("eps", 1e-12))
        hyb = gamma_cfg.get("hyb_weights", {}) or {}
        self.gamma_hyb = [float(hyb.get(f"a{i}", 0.0)) for i in range(4)]

        # --- NEW: history-based effective multigraph (used only when segments are layer-wise) ---
        hist_cfg = gamma_cfg.get("history", {}) or {}
        self.gamma_hist_enabled = bool(hist_cfg.get("enabled", False))
        self.gamma_hist_alpha = float(hist_cfg.get("alpha", 0.85))
        # prune edges whose EWMA weight falls below this threshold (0 disables pruning)
        self.gamma_hist_cutoff = float(hist_cfg.get("cutoff", 0.0))

        self._cached_key = None
        self._cached_val = None

    def _is_measure_gate(self, gate_name: Any) -> bool:
        return str(gate_name).strip().lower() in self.measure_gate_names

    def _segment_nodes_and_deg(
        self,
        edge_counts: Dict[Tuple[int, int], float],
    ) -> Tuple[Dict[int, float], List[int], float]:
        """Return (deg_s, V_s, bar_d_s) for the segment interaction graph."""
        deg_s: Dict[int, float] = defaultdict(float)
        nodes = set()
        for (a, b), n in edge_counts.items():
            nn = float(n)
            deg_s[a] += nn
            deg_s[b] += nn
            nodes.add(a)
            nodes.add(b)
        V_s = list(nodes)
        if len(V_s) == 0:
            bar_d_s = 0.0
        else:
            bar_d_s = float(sum(deg_s[u] for u in V_s)) / float(len(V_s))
        return deg_s, V_s, bar_d_s


    def _weighted_mean_over_edges(
        self,
        edge_counts: Dict[Tuple[int, int], float],
        val_map: Dict[Tuple[int, int], float],
    ) -> float:
        """Weighted mean of val_map over edges using multiplicities as weights."""
        total = float(sum(edge_counts.values()))
        if total <= 0.0:
            return 0.0
        acc = 0.0
        for e, w in edge_counts.items():
            acc += float(w) * float(val_map.get(e, 0.0))
        return acc / total


    def _gamma_pair_degree_pressure(
        self,
        *,
        edge_counts: Dict[Tuple[int, int], float],
    ) -> Dict[Tuple[int, int], float]:
        deg_s, _, bar_d_s = self._segment_nodes_and_deg(edge_counts)
        denom = 2.0 * bar_d_s + self.gamma_eps
        out: Dict[Tuple[int, int], float] = {}
        for (a, b), _n_uv in edge_counts.items():
            out[(a, b)] = (deg_s.get(a, 0.0) + deg_s.get(b, 0.0)) / denom
        return out


    def _gamma_pair_congestion(
        self,
        *,
        edge_counts: Dict[Tuple[int, int], float],
    ) -> Dict[Tuple[int, int], float]:
        deg_s, _, bar_d_s = self._segment_nodes_and_deg(edge_counts)
        denom = 2.0 * bar_d_s + self.gamma_eps
        out: Dict[Tuple[int, int], float] = {}
        for (a, b), n_uv in edge_counts.items():
            nn = float(n_uv)
            out[(a, b)] = ((deg_s.get(a, 0.0) - nn) + (deg_s.get(b, 0.0) - nn)) / denom
        return out


    def _gamma_pair_betweeness(
        self,
        *,
        edge_counts: Dict[Tuple[int, int], float],
    ) -> Dict[Tuple[int, int], float]:
        """
        Pair proxy based on edge betweenness centrality, weighted by multiplicity.

        We treat multiplicity n_uv as a *stronger/shorter* connection by setting a distance-like
        attribute: length(u,v) = 1 / (n_uv + eps). Then compute weighted betweenness on this length.
        Finally normalize by max betweenness within the segment.
        """
        if len(edge_counts) == 0:
            return {}

        G = nx.Graph()
        for (a, b), n_uv in edge_counts.items():
            length = 1.0 / (float(n_uv) + self.gamma_eps)
            G.add_edge(int(a), int(b), length=length)

        btw = nx.edge_betweenness_centrality(G, weight="length", normalized=True)

        raw: Dict[Tuple[int, int], float] = {}
        for (u, v), val in btw.items():
            uu, vv = int(u), int(v)
            if uu > vv:
                uu, vv = vv, uu
            raw[(uu, vv)] = float(val)

        max_b = max(raw.values()) if len(raw) > 0 else 0.0
        denom = max_b + self.gamma_eps

        out: Dict[Tuple[int, int], float] = {}
        for e in edge_counts.keys():
            out[e] = float(raw.get(e, 0.0)) / denom
        return out


    def _gamma_pair_hybrid(
        self,
        *,
        edge_counts: Dict[Tuple[int, int], float],
    ) -> Dict[Tuple[int, int], float]:
        g_deg = self._gamma_pair_degree_pressure(edge_counts=edge_counts)
        g_con = self._gamma_pair_congestion(edge_counts=edge_counts)
        g_btw = self._gamma_pair_betweeness(edge_counts=edge_counts)

        out: Dict[Tuple[int, int], float] = {}
        for e in edge_counts.keys():
            out[e] = (
                self.gamma_hyb[0]
                + self.gamma_hyb[1] * float(g_deg.get(e, 0.0))
                + self.gamma_hyb[2] * float(g_con.get(e, 0.0))
                + self.gamma_hyb[3] * float(g_btw.get(e, 0.0))
            )
        return out


    def _compute_gamma_value(
        self,
        *,
        N: int,
        L_s: int,
        edge_counts: Dict[Tuple[int, int], float],
    ) -> Tuple[float, Optional[Dict[Tuple[int, int], float]]]:
        """
        Returns:
        gamma_s: scalar proxy for segment s (for dashboards/backward-compat).
        gamma_map: optional per-edge proxy {(u,v)->Gamma(u,v,s)} for pair_* modes.
        """
        mode = (self.gamma_mode or "none").lower()

        # --- legacy scalar modes (no per-edge map) ---
        if mode == "none":
            return 0.0, None
        if mode == "edge_density":
            denom = max(1.0, (N * (N - 1)) / 2.0)
            return float(len(edge_counts)) / denom, None
        if mode == "twoq_per_layer":
            total_2q = float(sum(edge_counts.values()))
            return total_2q / max(1.0, float(L_s)), None

        # --- pair modes (return per-edge map + scalar summary) ---
        # accept a couple safe aliases (optional)
        if mode == "pair_congestion":
            mode = "pair_congestion_"
        if mode in {"pair_betweenness", "pair_btw"}:
            mode = "pair_betweeness"

        if mode == "pair_degree_pressure":
            gmap = self._gamma_pair_degree_pressure(edge_counts=edge_counts)
            return self._weighted_mean_over_edges(edge_counts, gmap), gmap
        if mode == "pair_congestion_":
            gmap = self._gamma_pair_congestion(edge_counts=edge_counts)
            return self._weighted_mean_over_edges(edge_counts, gmap), gmap
        if mode == "pair_betweeness":
            gmap = self._gamma_pair_betweeness(edge_counts=edge_counts)
            return self._weighted_mean_over_edges(edge_counts, gmap), gmap
        if mode == "pair_hybrid":
            gmap = self._gamma_pair_hybrid(edge_counts=edge_counts)
            return self._weighted_mean_over_edges(edge_counts, gmap), gmap

        return 0.0, None

    def _ewma_update_and_prune(
        self,
        hist: Dict[Tuple[int, int], float],
        cur: Dict[Tuple[int, int], int],
    ) -> None:
        """Update EWMA multigraph in-place and prune small (old) edges.

        Update rule:
          hist[e] <- alpha * hist[e] + cur[e]
        for edges in current layer; for all other edges:
          hist[e] <- alpha * hist[e]

        Pruning rule:
          - Always keep edges that appear in the current layer.
          - Otherwise drop edges with hist[e] < cutoff.
        """
        if not hist and not cur:
            return

        a = float(self.gamma_hist_alpha)
        # clamp alpha to a safe range to avoid surprises
        if a < 0.0:
            a = 0.0
        if a > 1.0:
            a = 1.0

        # decay existing history
        if len(hist) > 0:
            for e in list(hist.keys()):
                hist[e] = float(hist[e]) * a

        # add current layer counts
        for e, n in cur.items():
            hist[e] = float(hist.get(e, 0.0)) + float(n)

        # prune small edges (but never prune current-layer edges)
        cutoff = float(self.gamma_hist_cutoff)
        if cutoff > 0.0 and len(hist) > 0:
            cur_keys = set(cur.keys())
            for e in list(hist.keys()):
                if e in cur_keys:
                    continue
                if float(hist[e]) < cutoff:
                    del hist[e]

    def _resolve_gates(self, layer_ref, circuit):
        """
        layer_ref can be:
        - int: layer id into circuit.layers
        - CircuitLayer: already resolved
        Returns List[(gate_name, qargs_tuple)].
        """
        if hasattr(layer_ref, "gates"):
            return layer_ref.gates
        if isinstance(layer_ref, int):
            return circuit.layers[layer_ref].gates
        return []

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
        layer_ops: List[List[Dict[str, torch.Tensor]]] = []

        # --- NEW: when segments are layer-wise, compute Γ from a history-based EWMA multigraph ---
        seg_mode = str(DATASET_CFG.get("segmentation_mode", "")).strip().lower()
        use_gamma_history = bool(self.gamma_hist_enabled) and seg_mode == "layer"
        ewma_edge_counts: Dict[Tuple[int, int], float] = {} if use_gamma_history else {}

        for s, seg in enumerate(segments):
            layers = getattr(seg, "layers", seg)
            L_s = len(layers)
            L[s] = float(L_s)

            edge_counts: Dict[Tuple[int, int], int] = defaultdict(int)

            # Collect per-layer operation lists for differentiable timing + idle-only decoherence
            layer_ops_s_raw: List[Dict[str, Any]] = []

            for layer_id in layers:
                ops_1q: List[int] = []
                ops_m: List[int] = []
                ops2_u: List[int] = []
                ops2_v: List[int] = []
                ops2_pairs: List[Tuple[int, int]] = []  # sorted (a,b) for gamma lookup

                gates = self._resolve_gates(layer_id, circuit)
                for gate_name, qubits in gates:
                    if qubits is None:
                        continue
                    qlist = list(qubits)

                    if len(qlist) == 1:
                        u = int(qlist[0])
                        if 0 <= u < N:
                            if self._is_measure_gate(gate_name):
                                nm[s, u] += 1.0
                                ops_m.append(u)
                            else:
                                n1q[s, u] += 1.0
                                ops_1q.append(u)

                    elif len(qlist) == 2:
                        u = int(qlist[0]); v = int(qlist[1])
                        if u == v:
                            continue
                        if not (0 <= u < N and 0 <= v < N):
                            continue
                        a, b = (u, v) if u < v else (v, u)
                        edge_counts[(a, b)] += 1
                        ops2_u.append(u)
                        ops2_v.append(v)
                        ops2_pairs.append((a, b))

                    else:
                        # Multi-qubit gates ignored for now to avoid silent assumption breaks
                        continue

                # Store raw per-layer ops (convert to tensors after gamma_map is available)
                layer_ops_s_raw.append({
                    "ops_1q": ops_1q,
                    "ops_m": ops_m,
                    "ops2_u": ops2_u,
                    "ops2_v": ops2_v,
                    "ops2_pairs": ops2_pairs,
                })

            # Compute Γ from either the segment multigraph (default) or an EWMA history multigraph (layer-wise mode)
            if use_gamma_history:
                # Update the rolling multigraph with the current layer/segment
                self._ewma_update_and_prune(ewma_edge_counts, edge_counts)
                gamma_s_raw, gamma_map = self._compute_gamma_value(N=N, L_s=L_s, edge_counts=ewma_edge_counts)

                # For pair-wise modes (gamma_map != None), summarize Γ(s) as the weighted mean
                # over *current* edges so dashboards reflect the current layer in its temporal context.
                if gamma_map is not None and len(edge_counts) > 0:
                    gamma_s = self._weighted_mean_over_edges(edge_counts, gamma_map)
                else:
                    gamma_s = float(gamma_s_raw)
            else:
                gamma_s, gamma_map = self._compute_gamma_value(N=N, L_s=L_s, edge_counts=edge_counts)

            gamma[s] = float(gamma_s)

            # Convert per-layer op lists to tensors and attach per-gate gamma when available
            layer_ops_s: List[Dict[str, torch.Tensor]] = []
            for lr in layer_ops_s_raw:
                oneq_u = torch.tensor(lr["ops_1q"], device=device, dtype=torch.long) if len(lr["ops_1q"]) > 0 else torch.empty((0,), device=device, dtype=torch.long)
                meas_u = torch.tensor(lr["ops_m"],  device=device, dtype=torch.long) if len(lr["ops_m"])  > 0 else torch.empty((0,), device=device, dtype=torch.long)
                twoq_u = torch.tensor(lr["ops2_u"], device=device, dtype=torch.long) if len(lr["ops2_u"]) > 0 else torch.empty((0,), device=device, dtype=torch.long)
                twoq_v = torch.tensor(lr["ops2_v"], device=device, dtype=torch.long) if len(lr["ops2_v"]) > 0 else torch.empty((0,), device=device, dtype=torch.long)

                d = {"oneq_u": oneq_u, "meas_u": meas_u, "twoq_u": twoq_u, "twoq_v": twoq_v}

                if gamma_map is not None and len(lr["ops2_pairs"]) > 0:
                    Ge = torch.tensor([float(gamma_map.get(p, 0.0)) for p in lr["ops2_pairs"]], device=device, dtype=dtype)
                    d["twoq_gamma"] = Ge

                layer_ops_s.append(d)

            layer_ops.append(layer_ops_s)

            if len(edge_counts) == 0:
                e_u = torch.empty((0,), device=device, dtype=torch.long)
                e_v = torch.empty((0,), device=device, dtype=torch.long)
                e_w = torch.empty((0,), device=device, dtype=dtype)
                e_dict = {"u": e_u, "v": e_v, "w": e_w}
            else:
                pairs = list(edge_counts.keys())
                weights = [edge_counts[p] for p in pairs]
                e_u = torch.tensor([p[0] for p in pairs], device=device, dtype=torch.long)
                e_v = torch.tensor([p[1] for p in pairs], device=device, dtype=torch.long)
                e_w = torch.tensor(weights, device=device, dtype=dtype)

                e_dict = {"u": e_u, "v": e_v, "w": e_w}
                if gamma_map is not None:
                    e_dict["gamma_e"] = torch.tensor(
                        [float(gamma_map.get(p, 0.0)) for p in pairs],
                        device=device,
                        dtype=dtype,
                    )

            edges.append(e_dict)

        stats = {"L": L, "n1q": n1q, "nm": nm, "edges": edges, "gamma": gamma, "layer_ops": layer_ops}
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
        debug: bool = False,
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

        # Local 2Q with inflation
        Gamma = stats["gamma"].to(dtype)  # [S]
        infl_base = rho.to(dtype)         # [K]
        c2q = c2q.to(dtype)

        C2q_local = torch.zeros((S,), device=device, dtype=dtype)

        # Debug accumulators
        if debug:
            num_edges = torch.zeros((S,), device=device, dtype=dtype)
            twoq_ops  = torch.zeros((S,), device=device, dtype=dtype)
            avg_local_prob = torch.zeros((S,), device=device, dtype=dtype)  # avg Σ_k w_u,k w_v,k over edges (weighted)

        for s in range(S):
            e = stats["edges"][s]
            u_idx = e["u"]
            v_idx = e["v"]
            omega = e["w"].to(dtype)

            E = int(u_idx.numel())
            if E == 0:
                continue

            Wu = W[s, u_idx, :]  # [E,K]
            Wv = W[s, v_idx, :]  # [E,K]
            joint = Wu * Wv      # [E,K]

            gamma_e = e.get("gamma_e", None)

            if gamma_e is None:
                infl = (1.0 + infl_base * Gamma[s])             # [K]
                per_edge_cost = torch.einsum("ek,k->e", joint, infl * c2q)  # [E]
            else:
                Ge = gamma_e.to(dtype)                          # [E]
                infl_e = 1.0 + Ge[:, None] * infl_base[None, :] # [E,K]
                per_edge_cost = (joint * infl_e * c2q[None, :]).sum(dim=1)  # [E]

            C2q_local[s] = (omega * per_edge_cost).sum()

            if debug:
                num_edges[s] = float(E)
                twoq_ops[s] = omega.sum()

                # local_prob(e) = Σ_k joint[e,k]
                local_prob = joint.sum(dim=1)  # [E]
                denom = torch.clamp(omega.sum(), min=1e-12)
                avg_local_prob[s] = (omega * local_prob).sum() / denom

        Cexec = C1q + Cm + C2q_local

        out = {
            "per_segment_C1q": C1q,
            "per_segment_Cm": Cm,
            "per_segment_C2q_local": C2q_local,
            "per_segment_exec": Cexec,
        }

        if debug:
            out["exec_num_edges"] = num_edges.detach()
            out["exec_twoq_ops"] = twoq_ops.detach()
            out["exec_gamma"] = Gamma.detach()
            out["exec_avg_local_prob"] = avg_local_prob.detach()
            # Helpful “rates” for interpretability
            out["exec_1q_ops"] = stats["n1q"].sum(dim=1).detach()
            out["exec_meas_ops"] = stats["nm"].sum(dim=1).detach()

        return out
    

class IdleCostV3(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        P_seq: List[torch.Tensor],
        stats: Dict[str, Any],
        *,
        delta: torch.Tensor,     # legacy scalar per-layer proxy (used only when timing model disabled)
        T2: torch.Tensor,        # [K]
        decoherence_mode: str = "all_qubits",   # "all_qubits" | "idle_only"
        dt_override: Optional[torch.Tensor] = None,          # [S] if provided
        per_layer_dt: Optional[List[torch.Tensor]] = None,   # list of [num_layers_s]
        debug: bool = False,
    ) -> Dict[str, torch.Tensor]:

        device = P_seq[0].device
        dtype = P_seq[0].dtype
        S = len(P_seq)

        # Stack assignments: W[s,u,k]
        W = torch.stack(P_seq, dim=0).to(dtype)   # [S,N,K]
        N = W.size(1)

        # 1/T2^k (clamp to avoid divide-by-zero)
        invT = 1.0 / torch.clamp(T2.to(dtype), min=1e-12)  # [K]

        mode = (decoherence_mode or "all_qubits").strip().lower()

        if mode == "idle_only":
            if per_layer_dt is None:
                raise ValueError("IdleCostV3: per_layer_dt must be provided when decoherence_mode='idle_only'")

            # Idle-only decoherence:
            #   C_idle(s) = Σ_{layers ℓ in s} δ_ℓ * Σ_{u ∈ idle(ℓ)} Σ_k w_{u,k}(s) * (1/T2_k)
            Cidle = torch.zeros((S,), device=device, dtype=dtype)

            # Precompute invT dot-product helper:
            for s in range(S):
                layers_s = stats["layer_ops"][s]
                dt_layers = per_layer_dt[s]
                if int(dt_layers.numel()) != len(layers_s):
                    raise ValueError(
                        f"IdleCostV3: per_layer_dt[{s}] has length {int(dt_layers.numel())} "
                        f"but stats['layer_ops'][{s}] has {len(layers_s)} layers."
                    )

                for ell, layer_d in enumerate(layers_s):
                    dL = dt_layers[ell]

                    # active qubits are those touched by any op in the layer
                    active_parts = []
                    if layer_d["oneq_u"].numel() > 0:
                        active_parts.append(layer_d["oneq_u"])
                    if layer_d["meas_u"].numel() > 0:
                        active_parts.append(layer_d["meas_u"])
                    if layer_d["twoq_u"].numel() > 0:
                        active_parts.append(layer_d["twoq_u"])
                        active_parts.append(layer_d["twoq_v"])

                    if len(active_parts) == 0:
                        # No ops → nobody is active; idle set is all qubits. But dL is likely 0 anyway.
                        idle_idx = torch.arange(N, device=device, dtype=torch.long)
                    else:
                        active = torch.unique(torch.cat(active_parts, dim=0))
                        mask = torch.ones((N,), device=device, dtype=torch.bool)
                        mask[active] = False
                        idle_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
                        if idle_idx.numel() == 0:
                            continue

                    # sum over idle qubits: Σ_u Σ_k w_{u,k} * invT[k]
                    sum_idle = torch.einsum("uk,k->", W[s, idle_idx, :], invT)
                    Cidle[s] = Cidle[s] + dL * sum_idle

            out = {"per_segment_idle": Cidle}
            if debug:
                out["idle_mode"] = torch.tensor(1.0, device=device, dtype=dtype)  # marker
            return out

        # -------------------------
        # Legacy / all-qubits mode
        # -------------------------
        if dt_override is None:
            # Δt(s) = L(s) * δ (legacy)
            L = stats["L"].to(dtype)                  # [S]
            dt = L * delta.to(dtype)                  # [S]
        else:
            dt = dt_override.to(dtype)

        # 1/T^k (clamp to avoid divide-by-zero)
        invT = 1.0 / torch.clamp(T2.to(dtype), min=1e-12)  # [K]

        # Cidle[s] = dt[s] * Σ_u Σ_k W[s,u,k] * invT[k]
        sum_w_invT = torch.einsum("suk,k->s", W, invT)      # [S]
        Cidle = dt * sum_w_invT                              # [S]

        out = {"per_segment_idle": Cidle}
        if debug:
            out["idle_dt"] = dt.detach()
            out["idle_sum_w_invT"] = sum_w_invT.detach()
        return out

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
        debug: bool = False,
    ) -> Dict[str, torch.Tensor]:

        device = P_seq[0].device
        dtype = P_seq[0].dtype
        S = len(P_seq)

        W = torch.stack(P_seq, dim=0).to(dtype)  # [S,N,K]

        # -------------------------
        # Communication: per segment
        # -------------------------
        per_segment_comm = torch.zeros((S,), device=device, dtype=dtype)

        if debug:
            comm_num_edges = torch.zeros((S,), device=device, dtype=dtype)
            comm_twoq_ops = torch.zeros((S,), device=device, dtype=dtype)
            comm_avg_cut_prob = torch.zeros((S,), device=device, dtype=dtype)

        ccomm_s = ccomm.to(dtype)  # scalar

        for s in range(S):
            e = stats["edges"][s]
            u_idx = e["u"]
            v_idx = e["v"]
            omega = e["w"].to(dtype)

            E = int(u_idx.numel())
            if E == 0:
                continue

            Wu = W[s, u_idx, :]          # [E,K]
            Wv = W[s, v_idx, :]          # [E,K]
            local_prob = (Wu * Wv).sum(dim=1)     # [E]  = Σ_k w_u,k w_v,k
            cut_prob = 1.0 - local_prob           # [E]

            # C_comm(s) = ccomm * Σ_e ω_e * cut_prob_e
            weighted_cut = (omega * cut_prob).sum()         # scalar
            per_segment_comm[s] = ccomm_s * weighted_cut

            if debug:
                comm_num_edges[s] = float(E)
                comm_twoq_ops[s] = omega.sum()
                denom = torch.clamp(omega.sum(), min=1e-12)
                comm_avg_cut_prob[s] = (omega * cut_prob).sum() / denom

        # -------------------------
        # Movement: between segments
        # -------------------------
        per_segment_move = torch.zeros((S,), device=device, dtype=dtype)
        cmove_s = cmove.to(dtype)  # scalar

        if S >= 2:
            # stay_prob[s,u] = Σ_k w[s,u,k] * w[s+1,u,k]
            stay_prob = (W[:-1, :, :] * W[1:, :, :]).sum(dim=2)  # [S-1, N]
            change_prob = 1.0 - stay_prob                        # [S-1, N]

            # per_segment_move[s] = cmove * Σ_u change_prob[s,u]
            per_segment_move[:-1] = cmove_s * change_prob.sum(dim=1)

            if debug:
                move_total_change = change_prob.sum(dim=1)        # [S-1]
                move_avg_change = change_prob.mean(dim=1)         # [S-1]
                # pad to length S for easy plotting
                move_total_change_padded = torch.zeros((S,), device=device, dtype=dtype)
                move_avg_change_padded = torch.zeros((S,), device=device, dtype=dtype)
                move_total_change_padded[:-1] = move_total_change
                move_avg_change_padded[:-1] = move_avg_change

        out = {
            "per_segment_comm": per_segment_comm,
            "per_segment_move": per_segment_move,
        }

        if debug:
            out["comm_num_edges"] = comm_num_edges.detach()
            out["comm_twoq_ops"] = comm_twoq_ops.detach()
            out["comm_avg_cut_prob"] = comm_avg_cut_prob.detach()

            if S >= 2:
                out["move_total_change"] = move_total_change_padded.detach()
                out["move_avg_change"] = move_avg_change_padded.detach()

        return out

class SegmentTimeV3(nn.Module):
    """Differentiable segment-time model (v3).

    Computes per-gate expected durations from soft assignments w_{u,k}(s),
    aggregates per-layer time with a smooth max (LogSumExp), and sums over
    layers to obtain per-segment time.

    Supports:
      - mode = 'smooth_max' or 'hybrid' (smooth-max mixed with average)
      - annealed temperature tau (passed in as a buffer)
      - per-edge (gate-level) routing inflation when twoq_gamma is available
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _smooth_max_lse(x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        tau = torch.clamp(tau, min=1e-9)
        return tau * torch.logsumexp(x / tau, dim=0)

    def forward(
        self,
        P_seq: List[torch.Tensor],     # list of [N,K]
        stats: Dict[str, Any],
        *,
        t1q: torch.Tensor,             # [K]
        t2q: torch.Tensor,             # [K]
        tm: torch.Tensor,              # [K]
        t_remote: torch.Tensor,        # scalar
        rho: torch.Tensor,             # [K]
        tau: torch.Tensor,             # scalar (temperature)
        mode: str = "smooth_max",      # "smooth_max" | "hybrid"
        hybrid_lambda: Optional[torch.Tensor] = None,  # scalar in [0,1]
        use_routing_inflation_time: bool = True,
        debug: bool = False,
    ) -> Dict[str, Any]:

        device = P_seq[0].device
        dtype = P_seq[0].dtype
        S = len(P_seq)

        W = torch.stack(P_seq, dim=0).to(dtype)  # [S,N,K]

        t1q = t1q.to(device=device, dtype=dtype)
        t2q = t2q.to(device=device, dtype=dtype)
        tm  = tm.to(device=device, dtype=dtype)
        rho = rho.to(device=device, dtype=dtype)
        t_remote = t_remote.to(device=device, dtype=dtype)
        tau = tau.to(device=device, dtype=dtype)

        if hybrid_lambda is None:
            hybrid_lambda = torch.tensor(1.0, device=device, dtype=dtype)
        else:
            hybrid_lambda = hybrid_lambda.to(device=device, dtype=dtype)

        mode = (mode or "smooth_max").strip().lower()
        Gamma_s = stats.get("gamma", None)
        if Gamma_s is not None:
            Gamma_s = Gamma_s.to(dtype)

        per_segment_dt = torch.zeros((S,), device=device, dtype=dtype)
        per_layer_dt: List[torch.Tensor] = []

        for s in range(S):
            layers_s = stats["layer_ops"][s]
            dt_layers = torch.zeros((len(layers_s),), device=device, dtype=dtype)

            for ell, layer_d in enumerate(layers_s):
                times: List[torch.Tensor] = []

                # 1Q gate times
                oneq_u = layer_d["oneq_u"]
                if oneq_u.numel() > 0:
                    Wu = W[s, oneq_u, :]  # [G,K]
                    times.append(torch.einsum("gk,k->g", Wu, t1q))

                # measurement times
                meas_u = layer_d["meas_u"]
                if meas_u.numel() > 0:
                    Wm = W[s, meas_u, :]
                    times.append(torch.einsum("gk,k->g", Wm, tm))

                # 2Q gate times (local mixture + remote fallback)
                twoq_u = layer_d["twoq_u"]
                if twoq_u.numel() > 0:
                    twoq_v = layer_d["twoq_v"]
                    Wu = W[s, twoq_u, :]  # [G,K]
                    Wv = W[s, twoq_v, :]  # [G,K]
                    joint = Wu * Wv       # [G,K]
                    local_prob = joint.sum(dim=1)  # [G]
                    remote_prob = 1.0 - local_prob

                    # per-edge routing inflation on *local* 2Q time
                    if use_routing_inflation_time:
                        Ge = layer_d.get("twoq_gamma", None)
                        if Ge is None:
                            # fallback to scalar Gamma(s) if available
                            if Gamma_s is None:
                                infl = 1.0
                                local_time = torch.einsum("gk,k->g", joint, t2q)
                            else:
                                infl = (1.0 + rho * Gamma_s[s])  # [K]
                                local_time = torch.einsum("gk,k->g", joint, infl * t2q)
                        else:
                            Ge = Ge.to(dtype)  # [G]
                            infl_e = 1.0 + Ge[:, None] * rho[None, :]  # [G,K]
                            local_time = (joint * infl_e * t2q[None, :]).sum(dim=1)  # [G]
                    else:
                        local_time = torch.einsum("gk,k->g", joint, t2q)

                    times.append(local_time + remote_prob * t_remote)

                if len(times) == 0:
                    delta_L = torch.tensor(0.0, device=device, dtype=dtype)
                    dt_layers[ell] = delta_L
                    continue

                x = torch.cat(times, dim=0)  # [num_ops_in_layer]
                smax = self._smooth_max_lse(x, tau)

                if mode == "hybrid":
                    lam = torch.clamp(hybrid_lambda, 0.0, 1.0)
                    avg = x.mean()
                    delta_L = lam * smax + (1.0 - lam) * avg
                else:
                    delta_L = smax

                dt_layers[ell] = delta_L

            per_layer_dt.append(dt_layers)
            per_segment_dt[s] = dt_layers.sum()

        out: Dict[str, Any] = {"per_segment_dt": per_segment_dt, "per_layer_dt": per_layer_dt}
        if debug:
            out["timing_tau"] = tau.detach()
            out["timing_mode"] = mode
        return out


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

def _parse_timing_model_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    """Parse differentiable timing model configuration (optional)."""
    tm = config.get("timing_model", {}) or {}
    mode = str(tm.get("mode", "none")).strip().lower()

    return {
        "mode": mode,  # "none" | "smooth_max" | "hybrid"
        "tau0": float(tm.get("tau0", 1.0)),
        "tau_min": float(tm.get("tau_min", 1e-6)),
        "tau_gamma": float(tm.get("tau_gamma", 1.0)),
        "lambda0": float(tm.get("lambda0", 1.0)),
        "lambda_max": float(tm.get("lambda_max", 1.0)),
        "lambda_gamma": float(tm.get("lambda_gamma", 1.0)),
        "use_routing_inflation_time": bool(tm.get("use_routing_inflation_time", True)),
    }


def _parse_decoherence_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    """Parse decoherence aggregation mode (optional)."""
    dm = config.get("decoherence_model", {}) or {}
    mode = str(dm.get("mode", "all_qubits")).strip().lower()
    return {"mode": mode}  # "all_qubits" | "idle_only"




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

        # --- Optional: differentiable timing model + decoherence mode ---
        timing_model_cfg = _parse_timing_model_cfg(config)
        decoh_cfg = _parse_decoherence_cfg(config)

        self.timing_mode = timing_model_cfg["mode"]          # "none" | "smooth_max" | "hybrid"
        self.decoh_mode = decoh_cfg["mode"]                  # "all_qubits" | "idle_only"
        self.use_routing_inflation_time = bool(timing_model_cfg["use_routing_inflation_time"])

        # Annealing schedules (epoch-driven)
        self._tau0 = float(timing_model_cfg["tau0"])
        self._tau_min = float(timing_model_cfg["tau_min"])
        self._tau_gamma = float(timing_model_cfg["tau_gamma"])

        self._lam0 = float(timing_model_cfg["lambda0"])
        self._lam_max = float(timing_model_cfg["lambda_max"])
        self._lam_gamma = float(timing_model_cfg["lambda_gamma"])

        # Register buffers so tau/lambda move with the module across devices
        self.register_buffer("tau", torch.tensor(self._tau0, dtype=dtype))
        self.register_buffer("hybrid_lambda", torch.tensor(self._lam0, dtype=dtype))

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
        self.register_buffer("f_comm", comm_bufs["f_comm"]) # f_comm is remote gate execution fidelity
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

        self.segment_time = SegmentTimeV3()


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

        exec_out = self.exec_cost(P_seq, stats, c1q=self.c1q, c2q=self.c2q, cm=self.cm, rho=self.rho, debug=debug)
        # Optional: compute differentiable per-layer / per-segment durations
        dt_override = None
        per_layer_dt = None
        if self.timing_mode in {"smooth_max", "hybrid"}:
            t_out = self.segment_time(
                P_seq,
                stats,
                t1q=self.t1q,
                t2q=self.t2q,
                tm=self.tm,
                t_remote=self.t_remote,
                rho=self.rho,
                tau=self.tau,
                mode=self.timing_mode,
                hybrid_lambda=self.hybrid_lambda,
                use_routing_inflation_time=self.use_routing_inflation_time,
                debug=debug,
            )
            dt_override = t_out["per_segment_dt"]
            per_layer_dt = t_out["per_layer_dt"]


        if self.decoh_mode == "idle_only" and per_layer_dt is None:
            raise ValueError("TotalCost: decoherence_mode='idle_only' requires timing_model.mode in {'smooth_max','hybrid'}")

        idle_out = self.idle_cost(
            P_seq,
            stats,
            delta=self.delta,
            T2=self.T2,
            decoherence_mode=self.decoh_mode,
            dt_override=dt_override,
            per_layer_dt=per_layer_dt,
            debug=debug,
        )
        comm_out = self.comm_move_cost(P_seq, stats, ccomm=self.ccomm, cmove=self.cmove, debug=debug)

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


    def set_epoch(self, epoch: int) -> None:
        """Update annealed temperature (and hybrid weight) once per epoch.

        Exponential decay:
          tau(e) = max(tau_min, tau0 * tau_gamma^e)

        For hybrid mode (optional):
          lambda(e) = min(lambda_max, lambda0 * lambda_gamma^e)
        """
        e = int(epoch)
        with torch.no_grad():
            tau_new = max(self._tau_min, self._tau0 * (self._tau_gamma ** e))
            self.tau.fill_(float(tau_new))

            if self.timing_mode == "hybrid":
                lam_new = min(self._lam_max, self._lam0 * (self._lam_gamma ** e))
                self.hybrid_lambda.fill_(float(lam_new))

