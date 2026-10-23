"""Neural sub-solvers: a typed spatio-temporal graph network over the twin's netlist.

Each instance (well, pad, battery, compressor, lane ...) is a node; flow edges
are graph edges; every node carries a time axis (one slot per planning step).
The network maps

    (local parameters, current state, exogenous forecasts, constraint prices lambda)
        -> local decisions of every asset (+ routing of every routed edge)

so it acts as a *learned best-response* of every subsystem to the prices
published by the coordinator. It is trained self-supervised through the
differentiable twin (no labelled data, no external AI model):

    loss = -J(x) / J_scale + sum_c lambda_c * h_c(x) + mu * sum relu(h_c(x))^2

with randomised operating contexts (parameter drift, failures, tank levels,
price scenarios) and randomised prices. At solve time it is used

  * as an instant warm start (milliseconds) after any event, and
  * inside a dual-decomposition loop: lambda <- relu(lambda + eta * h(x(lambda))),
    where every iteration costs one network forward + one simulation.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn

from .gradient import integrality_penalty
from .problem import Problem


# ----------------------------------------------------------------------------- graph view
class TwinGraph:
    def __init__(self, problem: Problem):
        prog, sp = problem.prog, problem.space
        self.prog, self.space, self.T = prog, sp, sp.T
        flat = prog.flat
        self.N = len(flat.instances)
        self.type_keys = sorted({g.key for g in prog.groups if not g.is_env})
        self.type_index = {k: i for i, k in enumerate(self.type_keys)}
        self.groups = [g for g in prog.groups if not g.is_env]
        # node feature layout per group: params + states (numeric)
        self.feat_names = {}
        self.feat_scale = {}
        for g in self.groups:
            pnames = [p for p in g.ir.params if p in prog.P0[g.gid]]
            snames = list(g.ir.states)
            self.feat_names[g.gid] = (pnames, snames)
            sc = []
            for p in pnames:
                v = prog.P0[g.gid][p].abs().mean()
                sc.append(float(v) if float(v) > 1e-9 else 1.0)
            for s in snames:
                v = prog.S0[g.gid][s].abs().mean()
                info = g.ir.states[s]
                sc.append(float(v) if float(v) > 1e-9 else 1.0)
            self.feat_scale[g.gid] = torch.tensor(sc if sc else [1.0])
        self.max_feat = max(len(a) + len(b) for a, b in self.feat_names.values()) or 1
        # constraint price layout per group
        self.cons = {g.gid: [c for c in prog.cons if c.gid == g.gid and not c.soft] for g in self.groups}
        self.max_cons = max([len(v) for v in self.cons.values()] + [1])
        # edges (instance level, deduplicated)
        pairs = set()
        for e in flat.edges:
            a, b = flat.ports[e.src].inst, flat.ports[e.dst].inst
            pairs.add((a, b))
        pairs = sorted(pairs)
        self.src = torch.tensor([a for a, _ in pairs], dtype=torch.long)
        self.dst = torch.tensor([b for _, b in pairs], dtype=torch.long)
        self.deg_in = torch.zeros(self.N).index_add(0, self.dst, torch.ones(len(pairs))).clamp(min=1)
        self.deg_out = torch.zeros(self.N).index_add(0, self.src, torch.ones(len(pairs))).clamp(min=1)
        # routed edges -> (src inst, dst inst)
        self.route_src = torch.zeros(prog.n_routes, dtype=torch.long)
        self.route_dst = torch.zeros(prog.n_routes, dtype=torch.long)
        for rp in prog.routed:
            for r, eid in zip(rp.ridx, rp.edges):
                e = flat.edges[eid]
                self.route_src[r] = flat.ports[e.src].inst
                self.route_dst[r] = flat.ports[e.dst].inst
        # env inputs used as time features
        self.inputs = sorted(prog.env_inputs)
        self.input_scale = {k: max(abs(float(prog.E0[k])), 1e-6) for k in self.inputs}

    # ------------------------------------------------------------------ feature tensors
    def node_features(self, P: dict, S0: dict, B: int) -> torch.Tensor:
        X = torch.zeros(B, self.N, self.max_feat)
        for g in self.groups:
            pnames, snames = self.feat_names[g.gid]
            cols = []
            for p in pnames:
                cols.append(_bn(P[g.gid][p], B, g.n))
            for s in snames:
                cols.append(_bn(S0[g.gid][s], B, g.n))
            if not cols:
                continue
            F = torch.stack(cols, -1) / self.feat_scale[g.gid].view(1, 1, -1)
            F = torch.sign(F) * torch.log1p(F.abs())           # robust to scale changes
            idx = torch.tensor(g.inst_ids, dtype=torch.long)
            X[:, idx, :F.shape[-1]] = F
        return X

    def node_type(self) -> torch.Tensor:
        t = torch.zeros(self.N, dtype=torch.long)
        for g in self.groups:
            t[torch.tensor(g.inst_ids)] = self.type_index[g.key]
        return t

    def time_features(self, E: dict, t0: float, B: int) -> torch.Tensor:
        T, dt = self.T, self.space.dt
        tt = t0 + torch.arange(T, dtype=torch.float32) * dt
        hour = tt % 24.0
        cols = [torch.sin(2 * math.pi * hour / 24).expand(B, T), torch.cos(2 * math.pi * hour / 24).expand(B, T),
                (torch.arange(T, dtype=torch.float32) / T).expand(B, T)]
        for k in self.inputs:
            v = E.get(k, self.prog.E0[k]) if E else self.prog.E0[k]
            v = torch.as_tensor(v, dtype=torch.float32)
            if v.dim() == 0:
                v = v.expand(T)
            v = v.reshape(-1, v.shape[-1]) if v.dim() > 1 else v.view(1, -1)
            if v.shape[-1] != T:
                v = v[..., :T] if v.shape[-1] > T else torch.cat([v, v[..., -1:].expand(-1, T - v.shape[-1])], -1)
            cols.append((v / self.input_scale[k]).expand(B, T))
        return torch.stack(cols, -1)                               # [B, T, F]

    def price_features(self, lam: Optional[dict], B: int) -> torch.Tensor:
        L = torch.zeros(B, self.N, self.T, self.max_cons)
        if not lam:
            return L
        for g in self.groups:
            idx = torch.tensor(g.inst_ids, dtype=torch.long)
            for j, c in enumerate(self.cons[g.gid]):
                v = lam.get((c.gid, c.key))
                if v is None:
                    continue
                v = v if v.dim() == 3 else v.unsqueeze(-1)
                v = v.expand(B, g.n, self.T) if v.shape[0] == 1 else v
                L[:, idx, :, j] = torch.log1p(100.0 * v.clamp(min=0)) / 4.0
        return L


def _bn(x: torch.Tensor, B: int, n: int) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.dim() == 3:
        x = x[..., 0]
    if x.dim() == 1:
        x = x.view(1, -1)
    return x.expand(B, n)


# ----------------------------------------------------------------------------- network
class SubSolverNet(nn.Module):
    def __init__(self, graph: TwinGraph, d: int = 32, rounds: int = 3, slot: int = 4,
                 base_x: Optional[torch.Tensor] = None):
        super().__init__()
        self.g = graph
        self.d, self.rounds, self.slot = d, rounds, slot
        self.S = math.ceil(graph.T / slot)
        bx = base_x if base_x is not None else graph.space.default_u(1)
        self.register_buffer("base_logit", torch.logit(bx.clamp(0.02, 0.98)).detach().clone())
        nt = len(graph.type_keys)
        self.type_emb = nn.Embedding(nt, d)
        self.feat_enc = nn.ModuleList([nn.Linear(graph.max_feat, d) for _ in range(nt)])
        self.price_enc = nn.ModuleList([nn.Linear(graph.max_cons, d) for _ in range(nt)])
        n_tf = 3 + len(graph.inputs)
        self.time_enc = nn.Sequential(nn.Linear(n_tf, d), nn.GELU(), nn.Linear(d, d))
        self.w_in = nn.ModuleList([nn.Linear(d, d) for _ in range(rounds)])
        self.w_out = nn.ModuleList([nn.Linear(d, d) for _ in range(rounds)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Linear(3 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
                                  for _ in range(rounds)])
        self.norm = nn.ModuleList([nn.LayerNorm(3 * d) for _ in range(rounds)])
        self.tconv = nn.ModuleList([nn.Conv1d(d, d, 5, padding=2, groups=d) for _ in range(rounds)])
        # decision heads: one per (type key, decision)
        self.heads = nn.ModuleDict()
        for b in graph.space.blocks:
            key = self._head_key(b.gid, b.name)
            if key not in self.heads:
                self.heads[key] = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.route_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        for head in list(self.heads.values()) + [self.route_head]:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.node_type = graph.node_type()

    def set_base(self, x: torch.Tensor) -> None:
        """Re-centre the residual policy on a new reference plan (e.g. the latest optimum)."""
        self.base_logit.copy_(torch.logit(x.detach().clamp(0.02, 0.98)))

    def _head_key(self, gid: str, name: str) -> str:
        key = self.g.prog.gmap[gid].key
        return f"{self.g.type_index[key]}__{name}"

    def forward(self, P: dict, S0: dict, E: dict, t0: float, lam: Optional[dict], B: int) -> torch.Tensor:
        g, d, T, N = self.g, self.d, self.g.T, self.g.N
        X = g.node_features(P, S0, B)                                    # [B, N, F]
        L = _slots(g.price_features(lam, B), self.slot, 2)              # [B, N, S, C]
        tf = self.time_enc(_slots(g.time_features(E, t0, B), self.slot, 1))   # [B, S, d]
        T = self.S
        h_static = self.type_emb(self.node_type).unsqueeze(0).expand(B, N, d).clone()
        h_price = torch.zeros(B, N, T, d)
        for ti in range(len(g.type_keys)):
            m = self.node_type == ti
            if not bool(m.any()):
                continue
            h_static[:, m] = h_static[:, m] + self.feat_enc[ti](X[:, m])
            h_price[:, m] = self.price_enc[ti](L[:, m])
        h = h_static.unsqueeze(2) + tf.unsqueeze(1) + h_price              # [B, N, T, d]
        for k in range(self.rounds):
            hi = self.w_in[k](h)
            ho = self.w_out[k](h)
            m_in = torch.zeros_like(h).index_add(1, g.dst, hi.index_select(1, g.src)) / g.deg_in.view(1, -1, 1, 1)
            m_out = torch.zeros_like(h).index_add(1, g.src, ho.index_select(1, g.dst)) / g.deg_out.view(1, -1, 1, 1)
            h = h + self.upd[k](self.norm[k](torch.cat([h, m_in, m_out], -1)))
            z = h.reshape(B * N, T, d).transpose(1, 2)
            h = h + self.tconv[k](z).transpose(1, 2).reshape(B, N, T, d)
        # decode decisions (residual logits on top of the reference plan)
        sp = g.space
        base = sp.split(self.base_logit.expand(B, -1))
        parts = {}
        for b in sp.blocks:
            grp = g.prog.gmap[b.gid]
            idx = torch.tensor(grp.inst_ids, dtype=torch.long)
            y = self.heads[self._head_key(b.gid, b.name)](h.index_select(1, idx)).squeeze(-1)   # [B, n, S]
            y = _pool(_unslot(y, self.slot, g.T), b.hold, b.H)
            parts[(b.gid, b.name)] = torch.sigmoid(base[(b.gid, b.name)] + y)
        if sp.route is not None:
            hs = h.index_select(1, g.route_src)
            hd = h.index_select(1, g.route_dst)
            y = self.route_head(torch.cat([hs, hd], -1)).squeeze(-1)
            y = _pool(_unslot(y, self.slot, g.T), sp.route.hold, sp.route.H)
            parts["route"] = torch.sigmoid(base["route"] + 4.0 * y)
        return sp.join(parts)


def _slots(x: torch.Tensor, slot: int, dim: int) -> torch.Tensor:
    """average consecutive time steps into slots along `dim`."""
    if slot <= 1:
        return x
    T = x.shape[dim]
    S = math.ceil(T / slot)
    pad = S * slot - T
    if pad:
        tail = x.narrow(dim, T - 1, 1).expand(*[(pad if i == dim else -1) for i in range(x.dim())])
        x = torch.cat([x, tail], dim)
    shape = list(x.shape)
    shape[dim:dim + 1] = [S, slot]
    return x.reshape(shape).mean(dim + 1)


def _unslot(y: torch.Tensor, slot: int, T: int) -> torch.Tensor:
    return y.repeat_interleave(slot, dim=-1)[..., :T] if slot > 1 else y


def _pool(y: torch.Tensor, hold: int, H: int) -> torch.Tensor:
    B, n, T = y.shape
    pad = H * hold - T
    if pad > 0:
        y = torch.cat([y, y[..., -1:].expand(B, n, pad)], -1)
    return y[..., :H * hold].reshape(B, n, H, hold).mean(-1)


# ----------------------------------------------------------------------------- context sampling
@dataclass
class Context:
    P: dict
    S0: dict
    E: dict
    lam: Optional[dict]
    t0: float
    B: int


class ContextSampler:
    """Domain randomisation of operating contexts (drift, failures, inventories, prices)."""

    def __init__(self, problem: Problem, seed: int = 0):
        self.pb = problem
        self.prog = problem.prog
        self.gen = torch.Generator().manual_seed(seed)
        bound_params = set()
        from ..lang.behavior import walk_refs
        for g in self.prog.groups:
            for info in g.ir.decisions.values():
                for e in (info.lo, info.hi, info.default):
                    if e is not None:
                        for kind, name in walk_refs(e):
                            if kind == "param":
                                bound_params.add((g.gid, name))
        self.perturb = {}
        for g in self.prog.groups:
            if g.is_env:
                continue
            for pname in g.ir.params:
                if pname not in self.prog.P0[g.gid] or (g.gid, pname) in bound_params:
                    continue
                decl = g.ir.type.params.get(pname)
                if decl is not None and (decl.calibrate is not None or "cap" in pname):
                    self.perturb[(g.gid, pname)] = 0.18 if decl.calibrate is not None else 0.12

    def sample_events(self, B: int, base: Optional[Context] = None) -> Context:
        """Realistic re-planning situations: mild drift around the current state plus 0-2
        operational events (unit trip, well failures, injectivity loss, price shocks)."""
        gen, prog, T = self.gen, self.prog, self.pb.T
        base_P = base.P if base else self.pb.P
        base_S = base.S0 if base else self.pb.S0
        P = {}
        for gid, d in base_P.items():
            P[gid] = {}
            for name, v in d.items():
                s = self.perturb.get((gid, name))
                if s is None:
                    P[gid][name] = v
                    continue
                nv = v * torch.exp(torch.randn(B, v.shape[1], 1, generator=gen) * s * 0.3)
                if "cap" in name and (torch.rand(1, generator=gen).item() < 0.25):
                    k = int(torch.randint(0, v.shape[1], (1,), generator=gen))
                    nv[:, k] = nv[:, k] * (0.45 + 0.4 * torch.rand(B, 1, generator=gen))
                if name == "wc":
                    nv = nv.clamp(0.0, 0.98)
                P[gid][name] = nv
        S0 = {}
        for g in prog.groups:
            S0[g.gid] = {}
            for sname, info in g.ir.states.items():
                v = base_S[g.gid][sname]
                v = v.expand(B, v.shape[1]).clone() if v.shape[0] == 1 else v.clone()
                if info.lo is not None:
                    bnd = prog._bounds_cache()[g.gid].get(f"bound:{sname}")
                    if bnd is not None:
                        lo = torch.as_tensor(bnd[0], dtype=torch.float32).reshape(1, -1).expand(B, v.shape[1])
                        hi = torch.as_tensor(bnd[1], dtype=torch.float32).reshape(1, -1).expand(B, v.shape[1])
                        v = (v + (hi - lo) * 0.2 * torch.randn(B, v.shape[1], generator=gen)).clamp(
                            lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo))
                elif not info.has_update and float(v.max()) >= 1.0:
                    # failures / trips: a few instances lose one unit
                    p = 0.03 if v.shape[1] > 20 else 0.15
                    drop = (torch.rand(B, v.shape[1], generator=gen) < p).float()
                    v = (v - drop).clamp(min=0.0)
                elif sname == "health":
                    v = (v - 0.15 * torch.rand(B, v.shape[1], generator=gen)).clamp(0.05, 1.0)
                S0[g.gid][sname] = v
        E = {}
        hours = torch.arange(T, dtype=torch.float32)
        for k in prog.env_inputs:
            v0 = float(prog.E0[k])
            lvl = v0 * torch.exp(torch.randn(B, 1, generator=gen) * 0.1)
            ser = lvl.expand(B, T).clone()
            if "power" in k:
                ph = hours.view(1, -1) % 24
                ser = lvl * (0.75 + 1.6 * ((ph >= 15) & (ph < 20)).float() + 0.2 * ((ph >= 7) & (ph < 15)).float())
                ser = ser + (torch.rand(B, T, generator=gen) < 0.02).float() * 5 * v0
            if "gas" in k and torch.rand(1, generator=gen).item() < 0.25:
                start = int(torch.randint(0, T // 2, (1,), generator=gen))
                ser[:, start:start + 24] = -abs(v0) * (0.5 + torch.rand(B, 1, generator=gen))
            E[k] = ser
        return Context(P, S0, E, None, float(torch.randint(0, 24, (1,), generator=gen).item()), B)

    def sample(self, B: int, with_prices: float = 0.5, base: Optional[Context] = None) -> Context:
        gen, prog, T = self.gen, self.prog, self.pb.T
        base_P = base.P if base else self.pb.P
        base_S = base.S0 if base else self.pb.S0
        P = {}
        for gid, d in base_P.items():
            P[gid] = {}
            for name, v in d.items():
                s = self.perturb.get((gid, name))
                if s is None:
                    P[gid][name] = v
                else:
                    noise = torch.exp(torch.randn(B, v.shape[1], 1, generator=gen) * s)
                    nv = v * noise
                    decl = prog.gmap[gid].ir.type.params[name]
                    if name in ("wc",):
                        nv = nv.clamp(0.0, 0.98)
                    P[gid][name] = nv
        S0 = {}
        for g in prog.groups:
            S0[g.gid] = {}
            for sname, info in g.ir.states.items():
                v = base_S[g.gid][sname]
                v = v.expand(B, v.shape[1]).clone() if v.shape[0] == 1 else v.clone()
                if info.lo is not None:
                    bnd = prog._bounds_cache()[g.gid].get(f"bound:{sname}")
                    if bnd is not None:
                        lo = torch.as_tensor(bnd[0], dtype=torch.float32).reshape(1, -1)
                        hi = torch.as_tensor(bnd[1], dtype=torch.float32).reshape(1, -1)
                        lo, hi = lo.expand(B, v.shape[1]), hi.expand(B, v.shape[1])
                        u = 0.08 + 0.84 * torch.rand(B, v.shape[1], generator=gen)
                        v = lo + (hi - lo) * u
                elif not info.has_update and float(v.max()) >= 1.0:
                    drop = (torch.rand(B, v.shape[1], generator=gen) < 0.06).float()
                    v = (v - drop).clamp(min=0.0)
                elif sname == "health":
                    v = (v * (0.55 + 0.45 * torch.rand(B, v.shape[1], generator=gen))).clamp(0.05, 1.0)
                S0[g.gid][sname] = v
        E = {}
        hours = torch.arange(T, dtype=torch.float32)
        for k in prog.env_inputs:
            v0 = float(prog.E0[k])
            lvl = v0 * torch.exp(torch.randn(B, 1, generator=gen) * 0.25)
            amp = 0.45 * torch.rand(B, 1, generator=gen)
            ph = 24 * torch.rand(B, 1, generator=gen)
            ser = lvl * (1 + amp * torch.sin(2 * math.pi * (hours.view(1, -1) + ph) / 24))
            if "gas" in k:
                shock = (torch.rand(B, 1, generator=gen) < 0.2).float()
                ser = ser - shock * abs(v0) * (1 + torch.rand(B, 1, generator=gen))
            if "power" in k:
                spike = (torch.rand(B, T, generator=gen) < 0.03).float() * 4 * v0
                ser = ser + spike
            E[k] = ser
        lam = None
        if with_prices > 0 and torch.rand(1, generator=gen).item() < with_prices:
            lam = {}
            for c in prog.cons:
                if c.soft:
                    continue
                n = prog.gmap[c.gid].n
                active = (torch.rand(B, n, 1, generator=gen) < 0.25).float()
                lam[(c.gid, c.key)] = active * torch.rand(B, n, T, generator=gen) * 0.05
        t0 = float(torch.randint(0, 24, (1,), generator=gen).item())
        return Context(P, S0, E, lam, t0, B)


# ----------------------------------------------------------------------------- training
@dataclass
class TrainLog:
    step: int
    loss: float
    objective: float
    violation: float
    t: float


def policy_loss(problem: Problem, x: torch.Tensor, ctx: Context, mu: float = 2.0, anneal: float = 0.0):
    pb = problem.with_context(P=ctx.P, S0=ctx.S0, E=ctx.E, t0=ctx.t0)
    e = pb.evaluate(x, relax=True)
    Jn = e.objective / problem.J_scale
    pen = torch.zeros(x.shape[0])
    for key, h in e.resid.items():
        h = h if h.dim() == 3 else h.unsqueeze(-1)
        h = h.expand(x.shape[0], *h.shape[1:]) if h.shape[0] == 1 else h
        hp = torch.abs(h) if e.kinds.get(key) == "eq" else torch.relu(h)
        pen = pen + mu * (hp * hp).sum(dim=(1, 2)) / problem.T
        if ctx.lam is not None and key in ctx.lam:
            pen = pen + (ctx.lam[key] * h).sum(dim=(1, 2))
    if anneal > 0:
        pen = pen + anneal * integrality_penalty(problem.space, x)
    return (-Jn + pen).mean(), e


def train(problem: Problem, net: SubSolverNet, steps: int = 400, B: int = 8, lr: float = 3e-3, seed: int = 0,
          callback: Optional[Callable] = None, time_budget: Optional[float] = None, mu: float = 20.0,
          experts: Optional[list] = None, imitation: float = 0.0) -> list:
    """Self-supervised training through the differentiable twin.
    `experts` = list of (Context, x*) pairs from previous cascade solves (expert iteration)."""
    sampler = ContextSampler(problem, seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 2), pct_start=0.15)
    logs: list = []
    t0 = time.perf_counter()
    for step in range(steps):
        ctx = sampler.sample(B)
        x = net(ctx.P, ctx.S0, ctx.E, ctx.t0, ctx.lam, B)
        frac = step / max(1, steps - 1)
        loss, e = policy_loss(problem, x, ctx, mu=mu * (0.05 + 0.95 * frac), anneal=0.02 * frac)
        if experts and imitation > 0:
            ectx, xs = experts[step % len(experts)]
            xe = net(ectx.P, ectx.S0, ectx.E, ectx.t0, ectx.lam, ectx.B)
            loss = loss + imitation * ((xe - xs) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 10 == 0 or step == steps - 1:
            rec = TrainLog(step, float(loss), float(e.objective.mean()), float(e.max_violation().mean()),
                           time.perf_counter() - t0)
            logs.append(rec)
            if callback:
                callback(rec)
        if time_budget is not None and time.perf_counter() - t0 > time_budget:
            break
    return logs


def stack_contexts(ctxs: list) -> Context:
    """Concatenate single-scenario contexts into one batch."""
    B = len(ctxs)

    def cat(vals, n_dims):
        vs = [v if v.shape[0] == 1 else v[:1] for v in vals]
        return torch.cat(vs, 0)
    P = {gid: {k: cat([c.P[gid][k] for c in ctxs], 3) for k in ctxs[0].P[gid]} for gid in ctxs[0].P}
    S0 = {gid: {k: cat([c.S0[gid][k] for c in ctxs], 2) for k in ctxs[0].S0[gid]} for gid in ctxs[0].S0}
    E = {}
    for k in ctxs[0].E:
        vs = [torch.as_tensor(c.E[k], dtype=torch.float32) for c in ctxs]
        vs = [v.view(1, -1) if v.dim() <= 1 else v[:1].reshape(1, -1) for v in vs]
        E[k] = torch.cat(vs, 0)
    return Context(P, S0, E, None, ctxs[0].t0, B)


def generate_experts(problem: Problem, n: int, solver_factory: Callable, seed: int = 7,
                     callback: Optional[Callable] = None, base: Optional[Context] = None) -> list:
    """Solve n realistic situations with the full cascade -> (context, optimal x) pairs."""
    sampler = ContextSampler(problem, seed)
    out = []
    for i in range(n):
        ctx = sampler.sample_events(1, base=base)
        pb = problem.with_context(P=ctx.P, S0=ctx.S0, E=ctx.E, t0=ctx.t0)
        res = solver_factory(pb).solve()
        out.append((ctx, res.x.detach().clone(), res.objective))
        if callback:
            callback({"stage": "experts", "i": i, "objective": res.objective, "feasible": res.feasible,
                      "seconds": res.seconds})
    return out


def train_imitation(problem: Problem, net: "SubSolverNet", experts: list, steps: int = 300, B: int = 8,
                    lr: float = 2e-3, ss_weight: float = 0.05, seed: int = 0,
                    callback: Optional[Callable] = None) -> list:
    """Expert iteration: imitate cascade solutions (+ a small self-supervised term)."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 2), pct_start=0.1)
    logs = []
    t0 = time.perf_counter()
    for step in range(steps):
        idx = torch.randint(0, len(experts), (min(B, len(experts)),), generator=g).tolist()
        ctx = stack_contexts([experts[i][0] for i in idx])
        target = torch.cat([experts[i][1] for i in idx], 0)
        x = net(ctx.P, ctx.S0, ctx.E, ctx.t0, None, ctx.B)
        loss = ((x - target) ** 2).mean() * 100.0
        if ss_weight > 0:
            ss, e = policy_loss(problem, x, ctx, mu=5.0)
            loss = loss + ss_weight * ss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 10 == 0 or step == steps - 1:
            rec = {"stage": "imitation", "step": step, "loss": float(loss.detach()), "t": time.perf_counter() - t0}
            logs.append(rec)
            if callback:
                callback(rec)
    return logs


# ----------------------------------------------------------------------------- inference
@dataclass
class NeuralResult:
    x: torch.Tensor
    objective: float
    max_violation: float
    lam: dict
    history: list
    seconds: float


@torch.no_grad()
def dual_coordination(problem: Problem, net: SubSolverNet, iters: int = 15, eta: float = 0.05,
                      lam0: Optional[dict] = None, callback: Optional[Callable] = None) -> NeuralResult:
    """Price coordination with neural best responses: every iteration is one network forward
    for all subsystems at once plus one twin simulation."""
    t0 = time.perf_counter()
    lam = {k: v.clone() for k, v in lam0.items()} if lam0 else {}
    best = None
    hist = []
    for k in range(iters):
        x = net(problem.P, problem.S0, problem.E, problem.t0, lam or None, 1)
        e = problem.evaluate(x, relax=True)
        mv = float(e.max_violation()[0])
        obj = float(e.objective[0])
        score = obj / problem.J_scale - 10.0 / problem.T * float(e.sum_violation()[0])
        if best is None or score > best[0]:
            best = (score, x.clone(), obj, mv)
        for key, h in e.resid.items():
            h = h if h.dim() == 3 else h.unsqueeze(-1)
            lam[key] = torch.relu(lam.get(key, torch.zeros_like(h)) + eta * h)
        rec = {"stage": "neural", "iter": k, "objective": obj, "max_violation": mv, "t": time.perf_counter() - t0}
        hist.append(rec)
        if callback:
            callback(rec)
    return NeuralResult(best[1], best[2], best[3], lam, hist, time.perf_counter() - t0)


def save(net: SubSolverNet, path: str, meta: dict) -> None:
    torch.save({"state": net.state_dict(), "meta": meta}, path)


def load(net: SubSolverNet, path: str) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = net.load_state_dict(blob["state"], strict=False)
    return {"meta": blob.get("meta", {}), "missing": missing, "unexpected": unexpected}
