"""Compiled PLX program: generated simulator + runtime data (parameters, states, env).

A Program is the executable digital twin. It is *data-driven*: parameters,
initial states and environment inputs are tensors that the data-integration
layer can overwrite at any time without recompiling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch

from . import ast as A
from . import rt
from .behavior import chan_key
from .elaborate import FlatModel

TERM_KINDS = ("revenue", "cost", "emission", "metric")


@dataclass
class TermRef:
    gid: str
    key: str
    kind: str
    name: str
    final: bool
    unit: Optional[str] = None


@dataclass
class ConRef:
    gid: str
    key: str
    name: str
    kind: str
    soft: bool
    is_bound: bool
    info: Any


class Program:
    def __init__(self, comp, src: str, bsrc: str):
        self.comp = comp
        self.flat: FlatModel = comp.flat
        self.groups = comp.groups
        self.gmap = comp.gmap
        self.source = src
        self.bounds_source = bsrc
        self.physical = comp.physical
        ns: dict = {}
        exec(compile(src, "<plx-sim>", "exec"), ns)
        self._sim = ns["simulate"]
        ns2: dict = {}
        exec(compile(bsrc, "<plx-bounds>", "exec"), ns2)
        self._bounds = ns2["bounds"]
        self.K = comp.K
        self.routed = comp.routed
        self.n_routes = comp.n_routes
        self.edge_ridx = comp.edge_ridx
        self.inst_group = comp.inst_group
        self.dtype = torch.float32
        self.P0 = self._default_params()
        self.S0 = self._default_states()
        self.E0 = dict(self.flat.env_params)
        self.env_inputs = set(self.flat.env_inputs)
        self.terms: list[TermRef] = []
        self.cons: list[ConRef] = []
        for g in self.groups:
            for k, ti in g.ir.terms.items():
                self.terms.append(TermRef(g.gid, k, ti.kind, ti.name, ti.final, ti.unit))
            for ck, ci in g.ir.cons.items():
                self.cons.append(ConRef(g.gid, ck, ci.name, ci.kind, ci.soft is not None, ci.is_bound, ci))
        sim = self.flat.sim
        self.dt = sim.dt
        self.horizon = sim.horizon
        self.T = max(1, int(round(sim.horizon / sim.dt)))

    def to_dtype(self, dtype) -> "Program":
        self.dtype = dtype
        self.P0 = {g: {k: v.to(dtype) for k, v in d.items()} for g, d in self.P0.items()}
        self.S0 = {g: {k: v.to(dtype) for k, v in d.items()} for g, d in self.S0.items()}
        self.invalidate()
        return self

    # ------------------------------------------------------------------ data
    def _default_params(self) -> dict:
        P: dict = {}
        for g in self.groups:
            if g.is_env:
                continue
            P[g.gid] = {}
            for p in g.ir.params:
                vals = []
                ok = True
                for iid in g.inst_ids:
                    v = self.flat.instances[iid].params.get(p)
                    if v is None or isinstance(v, str):
                        ok = False
                        break
                    vals.append(float(v))
                if ok:
                    P[g.gid][p] = torch.tensor(vals, dtype=self.dtype).view(1, -1, 1)
        return P

    def _default_states(self) -> dict:
        S: dict = {}
        for g in self.groups:
            S[g.gid] = {}
            for s in g.ir.states:
                if g.is_env:
                    S[g.gid][s] = torch.zeros(1, 1, dtype=self.dtype)
                    continue
                vals = [float(self.flat.instances[iid].state0.get(s, 0.0)) for iid in g.inst_ids]
                S[g.gid][s] = torch.tensor(vals, dtype=self.dtype).view(1, -1)
        return S

    def env_tensors(self, E: Optional[dict], T: int) -> dict:
        out = {}
        E = E or {}
        for name, v0 in self.E0.items():
            v = E.get(name, v0)
            if isinstance(v, torch.Tensor):
                t = v.to(self.dtype)
                if t.dim() == 0:
                    t = t.view(1, 1, 1)
                elif t.dim() == 1:
                    t = t.view(1, 1, -1)
                elif t.dim() == 2:
                    t = t.view(t.shape[0], 1, -1)
                if t.shape[-1] not in (1, T):
                    t = _fit_series(t, T)
            elif isinstance(v, (list, tuple)):
                t = _fit_series(torch.tensor(v, dtype=self.dtype).view(1, 1, -1), T)
            else:
                t = torch.tensor(float(v), dtype=self.dtype).view(1, 1, 1)
                if name in self.env_inputs:
                    t = t.expand(1, 1, T)
            out[name] = t
        return out

    def default_routing(self, B: int = 1, T: int = 1) -> torch.Tensor:
        R = torch.zeros(B, self.n_routes, T, dtype=self.dtype)
        for rp in self.routed:
            R[:, rp.ridx[0], :] = 1.0
        return R

    def bounds(self, P: Optional[dict] = None, E: Optional[dict] = None, dt: Optional[float] = None) -> dict:
        P = P if P is not None else self.P0
        Et = self.env_tensors(E, 1)
        return self._bounds(P, Et, dt if dt is not None else self.dt)

    def default_decisions(self, B: int = 1, T: Optional[int] = None, P: Optional[dict] = None) -> dict:
        T = T or self.T
        bnd = self.bounds(P)
        D: dict = {}
        for g in self.groups:
            D[g.gid] = {}
            for d, info in g.ir.decisions.items():
                lo, hi, df = bnd[g.gid][f"dec:{d}"]
                lo, hi = _t(lo), _t(hi)
                if df is None:
                    df = hi if info.domain in ("bool", "int") else (lo + hi) / 2
                v = _t(df)
                v = torch.minimum(torch.maximum(v, lo), hi)
                v = v.to(self.dtype)
                D[g.gid][d] = v.expand(B, g.n, T).clone() if v.dim() == 3 else \
                    v.reshape(1, -1, 1).expand(B, g.n, T).clone()
        return D

    # ------------------------------------------------------------------ simulation
    def simulate(self, D: dict, R: Optional[torch.Tensor] = None, P: Optional[dict] = None,
                 S0: Optional[dict] = None, E: Optional[dict] = None, T: Optional[int] = None,
                 dt: Optional[float] = None, t0: float = 0.0, want_mag: bool = False) -> dict:
        T = T or self.T
        dt = float(dt if dt is not None else self.dt)
        B = 1
        for gd in D.values():
            for v in gd.values():
                B = max(B, v.shape[0])
                break
        if R is None:
            R = self.default_routing(1, 1)
        P = P if P is not None else self.P0
        S0 = S0 if S0 is not None else self.S0
        Et = self.env_tensors(E, T)
        rt.set_dtype(self.dtype)
        return self._sim(P, D, S0, Et, R, self.K, B, T, dt, float(t0), want_mag)

    def node_var(self, gid: str, key: str) -> str:
        return self.comp.nodes[f"{gid}|{key}"].var

    def final_states(self, raw: dict) -> dict:
        """State values after the simulated horizon: gid -> name -> [B, n]."""
        out = {}
        for g in self.groups:
            out[g.gid] = {}
            for s in g.ir.states:
                v = raw[self.node_var(g.gid, f"state:{s}") + "_f"]
                out[g.gid][s] = v if v.dim() == 2 else v.reshape(v.shape[0], -1)
        return out

    def signal(self, raw: dict, gid: str, key: str):
        """Signal by telemetry key ('node:local:x', 'state:x', 'in:p:r', 'param:x', 'dec:x')."""
        kind, _, name = key.partition(":")
        if kind == "node":
            return raw[self.node_var(gid, name)]
        if kind == "state":
            return raw[self.node_var(gid, f"state:{name}")]
        if kind == "in":
            return raw[self.node_var(gid, f"in:{name}")]
        raise KeyError(key)

    def get(self, raw: dict, gid: str, key: str, B: int = 1, T: Optional[int] = None) -> torch.Tensor:
        """Fetch a signal tensor [B, n, T] by group-local key (e.g. 'local:liquid', 'out:prod:oil')."""
        g = self.gmap[gid]
        if key.startswith("dec:"):
            raise KeyError("decisions are inputs")
        var = self.node_var(gid, key)
        x = raw[var]
        if T is not None and x.dim() == 3:
            x = rt.full(x, max(B, x.shape[0]), g.n, T)
        return x

    def evaluate(self, raw: dict, dt: Optional[float] = None, E: Optional[dict] = None,
                 B: Optional[int] = None, include_final: bool = True) -> dict:
        """Objective totals, soft penalties and hard-constraint violations."""
        dt = float(dt if dt is not None else self.dt)
        bnd = self._bounds_cache()
        totals = {k: 0.0 for k in TERM_KINDS}
        named: dict = {}
        for tr in self.terms:
            if tr.final and not include_final:
                continue
            if tr.final:
                x = raw[f"F_{tr.gid}_{_ident(tr.key)}"]
                v = x.sum(dim=tuple(range(1, x.dim())))
            else:
                x = raw[self.node_var(tr.gid, tr.key)]
                x = _bnt(x, self.gmap[tr.gid].n)
                v = x.sum(dim=(1, 2)) * dt
                if x.shape[-1] == 1:
                    v = v * _T_of(raw)
            totals[tr.kind] = totals[tr.kind] + v
            named[tr.name] = named.get(tr.name, 0.0) + v
        soft = 0.0
        hard: dict = {}
        resid: dict = {}
        for cr in self.cons:
            h = raw[self.node_var(cr.gid, cr.key)]
            h = _bnt(h, self.gmap[cr.gid].n)
            x = torch.abs(h) if cr.kind == "eq" else torch.relu(h)
            if not cr.soft:
                resid[(cr.gid, cr.key)] = h
            if cr.soft:
                w = bnd[cr.gid].get(f"soft:{cr.key}", 0.0)
                w = _t(w)
                pen = (x * w).sum(dim=(1, 2)) * dt
                if x.shape[-1] == 1:
                    pen = pen * _T_of(raw)
                soft = soft + pen
            else:
                hard[(cr.gid, cr.key)] = x
        out = dict(totals)
        out["named"] = named
        out["soft_penalty"] = soft
        out["hard"] = hard
        out["resid"] = resid
        out["objective"] = self._objective(totals, named, soft, E)
        return out

    def _bounds_cache(self):
        key = id(self.P0)
        if getattr(self, "_bc_key", None) != key:
            self._bc = self.bounds()
            self._bc_key = key
        return self._bc

    def invalidate(self) -> None:
        self._bc_key = None

    def _objective(self, totals: dict, named: dict, soft, E: Optional[dict]):
        obj = self.flat.sim.objective
        env = dict(self.E0)
        if E:
            for k, v in E.items():
                if not isinstance(v, (torch.Tensor, list, tuple)):
                    env[k] = float(v)
        if obj is None:
            val = _as_t(totals["revenue"]) - _as_t(totals["cost"])
            return val - _as_t(soft)
        sense, expr = obj

        def ev(e):
            if isinstance(e, A.Num):
                return e.value
            if isinstance(e, A.Name):
                if e.id in env:
                    return env[e.id]
                if e.id in totals:
                    return totals[e.id]
                if e.id in named:
                    return named[e.id]
                raise KeyError(f"objective: unknown name '{e.id}'")
            if isinstance(e, A.Member) and isinstance(e.obj, A.Name) and e.obj.id == "env":
                return env[e.attr]
            if isinstance(e, A.Call) and e.func in ("sum", "total"):
                a = e.args[0]
                if isinstance(a, A.Name):
                    k = {"revenues": "revenue", "costs": "cost", "emissions": "emission", "metrics": "metric"}.get(a.id, a.id)
                    if k in totals:
                        return totals[k]
                    if k in named:
                        return named[k]
                    return 0.0
                return ev(a)
            if isinstance(e, A.Binary):
                l, r = ev(e.left), ev(e.right)
                return {"+": lambda: l + r, "-": lambda: l - r, "*": lambda: l * r,
                        "/": lambda: l / r}[e.op]()
            if isinstance(e, A.Unary) and e.op == "-":
                return -ev(e.operand)
            raise ValueError(f"unsupported objective expression {type(e).__name__}")

        val = _as_t(ev(expr))
        if sense == "min":
            val = -val
        return val - _as_t(soft)

    # ------------------------------------------------------------------ introspection
    def signal_index(self) -> list:
        out = []
        for g in self.groups:
            for k in g.ir.nodes:
                out.append((g.gid, k))
        return out

    def summary(self) -> dict:
        n_dec = sum(g.n * len(g.ir.decisions) for g in self.groups)
        return {
            "groups": len(self.groups),
            "instances": len([i for i in self.flat.instances if not i.is_junction]),
            "junctions": len([i for i in self.flat.instances if i.is_junction]),
            "edges": len(self.flat.edges),
            "decision_signals": n_dec,
            "routed_edges": self.n_routes,
            "constraints": sum(g.n * len(g.ir.cons) for g in self.groups),
            "blocks": {k: sum(1 for b in self.comp.blocks if b.kind == k) for k in ("vec", "acc", "scan")},
            "lines_of_generated_code": self.source.count("\n"),
        }


def _ident(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_]", "_", s)


def _t(x):
    return x if isinstance(x, torch.Tensor) else torch.tensor(float(x))


def _as_t(x):
    return x if isinstance(x, torch.Tensor) else torch.tensor(float(x))


def _bnt(x, n):
    x = _t(x)
    if x.dim() == 0:
        return x.view(1, 1, 1)
    if x.dim() == 2:
        return x.unsqueeze(-1)
    return x


def _T_of(raw: dict) -> int:
    for v in raw.values():
        if isinstance(v, torch.Tensor) and v.dim() == 3 and v.shape[-1] > 1:
            return v.shape[-1]
    return 1


def _fit_series(t: torch.Tensor, T: int) -> torch.Tensor:
    L = t.shape[-1]
    if L == T:
        return t
    if L > T:
        return t[..., :T]
    reps = math.ceil(T / L)
    return t.repeat(*([1] * (t.dim() - 1)), reps)[..., :T]
