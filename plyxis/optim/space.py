"""Decision space of a compiled twin.

Every PLX decision becomes a block of normalised variables u in [0, 1]
(one per instance and hold-block); every routed out-port edge becomes a
routing variable. The same representation is shared by gradient solvers,
metaheuristics and neural sub-solvers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

from ..lang import rt

ROUTE_EPS = 1e-3


@dataclass
class DecBlock:
    gid: str
    name: str
    domain: str
    n: int
    hold: int          # steps per block
    H: int             # number of blocks
    offset: int        # offset in the flat vector
    unit: Optional[str] = None

    @property
    def size(self) -> int:
        return self.n * self.H


@dataclass
class RouteBlock:
    hold: int
    H: int
    E: int
    offset: int

    @property
    def size(self) -> int:
        return self.E * self.H


@dataclass
class DecisionSpace:
    program: object
    T: int
    dt: float
    route_hold: int = 6
    blocks: list = field(default_factory=list)
    route: Optional[RouteBlock] = None
    N: int = 0

    @classmethod
    def build(cls, program, T: Optional[int] = None, dt: Optional[float] = None,
              route_hold: Optional[int] = None) -> "DecisionSpace":
        T = T or program.T
        dt = dt or program.dt
        rh = route_hold or int(program.flat.sim.solver.get("route_hold", 6))
        sp = cls(program, T, dt, rh)
        off = 0
        for g in program.groups:
            for name, info in g.ir.decisions.items():
                hold = int(info.hold) if info.hold_hours is None else max(1, int(round(info.hold_hours / dt)))
                H = math.ceil(T / hold)
                b = DecBlock(g.gid, name, info.domain, g.n, hold, H, off, info.unit)
                sp.blocks.append(b)
                off += b.size
        if program.n_routes:
            H = math.ceil(T / rh)
            sp.route = RouteBlock(rh, H, program.n_routes, off)
            off += sp.route.size
        sp.N = off
        sp._index_cache = {}
        sp._discrete_mask = None
        sp._bounds = None
        return sp

    # ------------------------------------------------------------------ layout helpers
    def block(self, gid: str, name: str) -> DecBlock:
        for b in self.blocks:
            if b.gid == gid and b.name == name:
                return b
        raise KeyError((gid, name))

    def expand_index(self, hold: int, H: int) -> torch.Tensor:
        key = (hold, H)
        if key not in self._index_cache:
            self._index_cache[key] = torch.clamp(torch.arange(self.T) // hold, max=H - 1)
        return self._index_cache[key]

    def split(self, x: torch.Tensor) -> dict:
        """flat [B, N] -> dict of [B, n, H] (+ 'route': [B, E, H])."""
        B = x.shape[0]
        out = {}
        for b in self.blocks:
            out[(b.gid, b.name)] = x[:, b.offset:b.offset + b.size].view(B, b.n, b.H)
        if self.route is not None:
            r = self.route
            out["route"] = x[:, r.offset:r.offset + r.size].view(B, r.E, r.H)
        return out

    def join(self, parts: dict) -> torch.Tensor:
        chunks = []
        for b in self.blocks:
            v = parts[(b.gid, b.name)]
            chunks.append(v.reshape(v.shape[0], -1))
        if self.route is not None:
            v = parts["route"]
            chunks.append(v.reshape(v.shape[0], -1))
        return torch.cat(chunks, 1)

    def discrete_mask(self) -> torch.Tensor:
        if self._discrete_mask is None:
            m = torch.zeros(self.N, dtype=torch.bool)
            for b in self.blocks:
                if b.domain in ("int", "bool"):
                    m[b.offset:b.offset + b.size] = True
            if self.route is not None:
                excl = torch.zeros(self.route.E, dtype=torch.bool)
                for rp in self.program.routed:
                    if rp.exclusive:
                        excl[rp.ridx] = True
                m[self.route.offset:self.route.offset + self.route.size] = \
                    excl.view(-1, 1).expand(-1, self.route.H).reshape(-1)
            self._discrete_mask = m
        return self._discrete_mask

    # ------------------------------------------------------------------ bounds
    def bounds(self, P: Optional[dict] = None) -> dict:
        if P is None and self._bounds is not None:
            return self._bounds
        bnd = self.program.bounds(P, dt=self.dt)
        out = {}
        for b in self.blocks:
            lo, hi, df = bnd[b.gid][f"dec:{b.name}"]
            lo = _shape(lo, b.n)
            hi = _shape(hi, b.n)
            hi = torch.maximum(hi, lo)
            if df is None:
                df = hi if b.domain in ("int", "bool") else (lo + hi) / 2
            df = torch.minimum(torch.maximum(_shape(df, b.n), lo), hi)
            out[(b.gid, b.name)] = (lo, hi, df)
        if P is None:
            self._bounds = out
        return out

    def invalidate(self) -> None:
        self._bounds = None
        self.program.invalidate()

    # ------------------------------------------------------------------ canonical vectors
    def default_u(self, B: int = 1, P: Optional[dict] = None) -> torch.Tensor:
        bnd = self.bounds(P)
        parts = {}
        for b in self.blocks:
            lo, hi, df = bnd[(b.gid, b.name)]
            u = (df - lo) / torch.clamp(hi - lo, min=1e-12)
            u = torch.where(hi - lo > 1e-12, u, torch.zeros_like(u))
            parts[(b.gid, b.name)] = u.reshape(1, b.n, 1).expand(B, b.n, b.H)
        if self.route is not None:
            r = torch.zeros(B, self.route.E, self.route.H)
            for rp in self.program.routed:
                r[:, rp.ridx[0], :] = 1.0
            parts["route"] = r
        return self.join(parts).clone()

    def random_u(self, B: int, gen: Optional[torch.Generator] = None) -> torch.Tensor:
        return torch.rand(B, self.N, generator=gen)

    # ------------------------------------------------------------------ decoding
    def decode(self, x: torch.Tensor, P: Optional[dict] = None, relax: bool = True) -> tuple:
        """Normalised flat vector -> (D values [B, n, T] per group, routing splits R [B, E, T])."""
        parts = self.split(x)
        bnd = self.bounds(P)
        D: dict = {}
        for b in self.blocks:
            u = parts[(b.gid, b.name)]
            idx = self.expand_index(b.hold, b.H)
            u = u.index_select(2, idx)
            lo, hi, _ = bnd[(b.gid, b.name)]
            v = lo + (hi - lo) * u
            if b.domain in ("int", "bool") and not relax:
                v = rt.round(v)
            D.setdefault(b.gid, {})[b.name] = v
        for g in self.program.groups:
            D.setdefault(g.gid, {})
        R = None
        if self.route is not None:
            r = parts["route"].index_select(2, self.expand_index(self.route.hold, self.route.H))
            R = self.normalise_routes(r, relax)
        return D, R

    def normalise_routes(self, r: torch.Tensor, relax: bool = True) -> torch.Tensor:
        """Per out-port normalisation of routing weights -> split fractions."""
        prog = self.program
        if not hasattr(self, "_port_of_route"):
            port_of = torch.zeros(prog.n_routes, dtype=torch.long)
            excl = torch.zeros(prog.n_routes, dtype=torch.bool)
            for k, rp in enumerate(prog.routed):
                port_of[rp.ridx] = k
                if rp.exclusive:
                    excl[rp.ridx] = True
            self._port_of_route = port_of
            self._excl = excl
            self._n_ports = len(prog.routed)
        w = r.clamp(0.0, 1.0) + ROUTE_EPS
        B, E, T = w.shape
        tot = torch.zeros(B, self._n_ports, T, dtype=w.dtype).index_add(1, self._port_of_route, w)
        split = w / tot.index_select(1, self._port_of_route)
        if not relax and bool(self._excl.any()):
            # exclusive ports: one-hot on the largest weight (straight-through gradient)
            best = torch.zeros(B, self._n_ports, T, dtype=w.dtype).index_reduce(
                1, self._port_of_route, w, "amax", include_self=False)
            onehot = (w >= best.index_select(1, self._port_of_route) - 1e-12).to(w.dtype)
            # tie-break: keep only the first max per port
            onehot = self._first_only(onehot)
            hard = split + (onehot - split).detach()
            split = torch.where(self._excl.view(1, -1, 1), hard, split)
        return split

    def _first_only(self, onehot: torch.Tensor) -> torch.Tensor:
        out = onehot.clone()
        for rp in self.program.routed:
            if not rp.exclusive or len(rp.ridx) < 2:
                continue
            seen = torch.zeros_like(out[:, rp.ridx[0], :])
            for r in rp.ridx:
                keep = out[:, r, :] * (1 - seen)
                out[:, r, :] = keep
                seen = torch.clamp(seen + keep, max=1.0)
        return out

    def round_discrete(self, x: torch.Tensor) -> torch.Tensor:
        """Snap integer/bool decisions to their grid and exclusive routes to one-hot (in u-space)."""
        x = x.clone()
        parts = self.split(x)
        bnd = self.bounds()
        for b in self.blocks:
            if b.domain not in ("int", "bool"):
                continue
            lo, hi, _ = bnd[(b.gid, b.name)]
            u = parts[(b.gid, b.name)]
            span = torch.clamp(hi - lo, min=1e-12)
            v = torch.round(lo + span * u)
            parts[(b.gid, b.name)].copy_(torch.where(hi - lo > 1e-12, (v - lo) / span, torch.zeros_like(u)))
        if self.route is not None:
            r = parts["route"]
            for rp in self.program.routed:
                if rp.exclusive and len(rp.ridx) > 1:
                    sub = r[:, rp.ridx, :]
                    k = sub.argmax(1, keepdim=True)
                    oh = torch.zeros_like(sub).scatter_(1, k, 1.0)
                    r[:, rp.ridx, :] = oh
        return self.join(parts)

    # ------------------------------------------------------------------ rolling horizon
    def shift(self, x: torch.Tensor, hours: float) -> torch.Tensor:
        """Shift a plan forward in time (warm start for the next re-plan)."""
        steps = int(round(hours / self.dt))
        if steps <= 0:
            return x.clone()
        parts = self.split(x)
        out = {}
        for b in self.blocks:
            u = parts[(b.gid, b.name)]
            k = min(b.H - 1, steps // b.hold)
            out[(b.gid, b.name)] = torch.cat([u[:, :, k:], u[:, :, -1:].expand(-1, -1, k)], 2) if k > 0 else u
        if self.route is not None:
            u = parts["route"]
            k = min(self.route.H - 1, steps // self.route.hold)
            out["route"] = torch.cat([u[:, :, k:], u[:, :, -1:].expand(-1, -1, k)], 2) if k > 0 else u
        return self.join(out).clone()

    def describe(self) -> dict:
        return {
            "variables": self.N,
            "discrete": int(self.discrete_mask().sum()),
            "blocks": [{"group": b.gid, "type": self.program.gmap[b.gid].ir.type.short, "decision": b.name,
                        "instances": b.n, "hold_steps": b.hold, "blocks": b.H, "domain": b.domain}
                       for b in self.blocks],
            "routing": None if self.route is None else {"edges": self.route.E, "blocks": self.route.H,
                                                         "hold_steps": self.route.hold},
        }


def _shape(x, n: int) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.tensor(float(x))
    t = t.to(torch.float32)
    if t.dim() == 0:
        return t.view(1, 1, 1).expand(1, n, 1)
    if t.dim() == 3:
        if t.shape[-1] != 1:
            t = t[..., :1]
        return t.expand(t.shape[0], n, 1) if t.shape[1] in (1, n) else t
    return t.reshape(1, -1, 1).expand(1, n, 1)
