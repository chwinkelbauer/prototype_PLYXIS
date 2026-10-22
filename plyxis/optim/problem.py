"""Optimisation problem = compiled twin + decision space + operating context.

The context (current state estimate, calibrated parameters, price forecasts,
start time) comes from the data-integration layer; the problem itself is
model-agnostic and works for any PLX twin.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from .space import DecisionSpace


@dataclass
class Evaluation:
    objective: torch.Tensor            # [B] (to maximise, currency)
    viol: dict                         # (gid, con key) -> scaled violation [B, n, T]
    totals: dict                       # revenue / cost / emission / named terms
    raw: dict
    x: torch.Tensor
    resid: dict = None                 # (gid, con key) -> scaled signed residual (<=0 feasible)
    kinds: dict = None

    def max_violation(self) -> torch.Tensor:
        if not self.viol:
            return torch.zeros_like(self.objective)
        return torch.stack([v.reshape(v.shape[0], -1).amax(1) if v.shape[0] == self.objective.shape[0]
                            else v.reshape(v.shape[0], -1).amax(1).expand_as(self.objective)
                            for v in self.viol.values()], 1).amax(1)

    def sum_violation(self) -> torch.Tensor:
        if not self.viol:
            return torch.zeros_like(self.objective)
        s = 0.0
        for v in self.viol.values():
            vv = v.reshape(v.shape[0], -1).sum(1)
            s = s + (vv if vv.shape[0] == self.objective.shape[0] else vv.expand_as(self.objective))
        return s


class Problem:
    def __init__(self, program, space: Optional[DecisionSpace] = None, P: Optional[dict] = None,
                 S0: Optional[dict] = None, E: Optional[dict] = None, t0: float = 0.0,
                 T: Optional[int] = None, dt: Optional[float] = None):
        self.prog = program
        self.space = space or DecisionSpace.build(program, T=T, dt=dt)
        self.T, self.dt = self.space.T, self.space.dt
        self.P = P if P is not None else program.P0
        self.S0 = S0 if S0 is not None else program.S0
        self.E = E or {}
        self.t0 = t0
        self.n_evals = 0
        self.con_kind = {(c.gid, c.key): c.kind for c in program.cons}
        self._inst_maps()
        self.calibrate_scales()

    # ------------------------------------------------------------------ setup
    def _inst_maps(self) -> None:
        """global instance index per (group, local) for per-asset attribution."""
        self.inst_ids = [i.id for i in self.prog.flat.instances]
        self.n_inst = len(self.inst_ids) + 1          # + environment pseudo-instance
        self.group_index = {}
        for g in self.prog.groups:
            if g.is_env:
                self.group_index[g.gid] = torch.tensor([self.n_inst - 1], dtype=torch.long)
            else:
                self.group_index[g.gid] = torch.tensor(g.inst_ids, dtype=torch.long)

    def calibrate_scales(self, x: Optional[torch.Tensor] = None) -> None:
        x = x if x is not None else self.space.default_u(1)
        with torch.no_grad():
            D, R = self.space.decode(x, self.P, relax=True)
            raw = self.prog.simulate(D, R, self.P, self.S0, self.E, self.T, self.dt, self.t0, want_mag=True)
            ev = self.prog.evaluate(raw, self.dt, self.E)
        scale = float(torch.as_tensor(ev["revenue"]).abs().mean() + torch.as_tensor(ev["cost"]).abs().mean())
        self.J_scale = max(scale, 1.0)
        self.con_scale = {}
        for cr in self.prog.cons:
            if cr.soft:
                continue
            m = raw.get(f"mag|{cr.gid}|{cr.key}")
            n = self.prog.gmap[cr.gid].n
            if m is None:
                self.con_scale[(cr.gid, cr.key)] = torch.ones(1, n, 1)
                continue
            m = torch.as_tensor(m, dtype=torch.float32)
            if m.dim() == 0:
                m = m.view(1, 1, 1)
            if m.dim() == 2:
                m = m.unsqueeze(-1)
            per = m.mean(dim=(0, 2), keepdim=True).expand(1, n, 1)
            glob = float(m.mean())
            floor = max(glob * 0.05, 1e-6)
            per = torch.where(per > floor, per, torch.full_like(per, max(glob, 1e-3)))
            self.con_scale[(cr.gid, cr.key)] = per.clone()

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, x: torch.Tensor, relax: bool = True, P: Optional[dict] = None,
                 S0: Optional[dict] = None, E: Optional[dict] = None) -> Evaluation:
        P = P if P is not None else self.P
        S0 = S0 if S0 is not None else self.S0
        E = E if E is not None else self.E
        D, R = self.space.decode(x, P, relax=relax)
        raw = self.prog.simulate(D, R, P, S0, E, self.T, self.dt, self.t0)
        ev = self.prog.evaluate(raw, self.dt, E)
        self.n_evals += x.shape[0]
        viol, resid = {}, {}
        for key, v in ev["hard"].items():
            s = self.con_scale.get(key)
            viol[key] = v / s if s is not None else v
            h = ev["resid"][key]
            resid[key] = h / s if s is not None else h
        obj = torch.as_tensor(ev["objective"])
        if obj.dim() == 0:
            obj = obj.expand(x.shape[0])
        return Evaluation(obj, viol, ev, raw, x, resid, self.con_kind)

    def fitness(self, x: torch.Tensor, mu: float = 10.0, relax: bool = False) -> torch.Tensor:
        """Scalar penalised fitness for population methods (higher is better)."""
        with torch.no_grad():
            e = self.evaluate(x, relax=relax)
            return e.objective / self.J_scale - mu / self.T * e.sum_violation()

    # ------------------------------------------------------------------ attribution
    def per_instance(self, e: Evaluation, mu: float = 10.0) -> torch.Tensor:
        """Penalised objective attributed to each instance: [B, n_inst] (env pseudo-instance last)."""
        B = e.objective.shape[0]
        acc = torch.zeros(B, self.n_inst)
        dt = self.dt
        sign = {"revenue": 1.0, "cost": -1.0, "emission": 0.0, "metric": 0.0}
        cprice = 0.0
        obj = self.prog.flat.sim.objective
        if obj is not None and "carbon_price" in self.prog.E0:
            cprice = float(self.E.get("carbon_price", self.prog.E0["carbon_price"])) \
                if not isinstance(self.E.get("carbon_price"), torch.Tensor) else 0.0
            sign["emission"] = -cprice
        for tr in self.prog.terms:
            s = sign.get(tr.kind, 0.0)
            if s == 0.0:
                continue
            if tr.final:
                x = e.raw[f"F_{tr.gid}_{_ident(tr.key)}"]
                v = x.reshape(x.shape[0], -1) if x.dim() > 1 else x.view(1, -1)
            else:
                x = e.raw[self.prog.node_var(tr.gid, tr.key)]
                x = _bnt(x)
                v = x.sum(-1) * dt * (self.T if x.shape[-1] == 1 else 1)
            v = v.expand(B, -1) if v.shape[0] == 1 else v
            acc = acc.index_add(1, self.group_index[tr.gid], s * v)
        acc = acc / self.J_scale
        for (gid, ck), g in e.viol.items():
            g = _bnt(g)
            v = g.sum(-1)
            v = v.expand(B, -1) if v.shape[0] == 1 else v
            acc = acc.index_add(1, self.group_index[gid], -mu / self.T * v)
        bnd = self.prog._bounds_cache()
        for cr in self.prog.cons:
            if not cr.soft:
                continue
            x = _bnt(e.raw[self.prog.node_var(cr.gid, cr.key)])
            w = bnd[cr.gid].get(f"soft:{cr.key}", 0.0)
            w = w if isinstance(w, torch.Tensor) else torch.tensor(float(w))
            v = (x * w).sum(-1) * dt
            v = v.expand(B, -1) if v.shape[0] == 1 else v
            acc = acc.index_add(1, self.group_index[cr.gid], -v / self.J_scale)
        return acc

    def with_context(self, P=None, S0=None, E=None, t0=None) -> "Problem":
        """Cheap copy with a new operating context (keeps scales)."""
        q = Problem.__new__(Problem)
        q.__dict__.update(self.__dict__)
        if P is not None:
            q.P = P
        if S0 is not None:
            q.S0 = S0
        if E is not None:
            q.E = E
        if t0 is not None:
            q.t0 = t0
        q.n_evals = 0
        return q


def _ident(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_]", "_", s)


def _bnt(x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x)
    if x.dim() == 0:
        return x.view(1, 1, 1)
    if x.dim() == 2:
        return x.unsqueeze(-1)
    return x
