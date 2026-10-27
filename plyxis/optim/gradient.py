"""Augmented-Lagrangian method (ALM) with projected Adam on the differentiable twin.

    max_u  J(u)   s.t.  g_c(u) <= 0  (all hard constraints, scaled),   u in [0, 1]^N

The simulator is differentiated end-to-end with torch autograd, so the cost of
a gradient is a small multiple of one simulation regardless of the number of
decision variables. Multipliers lambda_c are the shadow prices of the
constraints (marginal value of one more unit of capacity), used for
explanations.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from .problem import Evaluation, Problem


@dataclass
class ALMResult:
    x: torch.Tensor
    objective: float
    max_violation: float
    multipliers: dict
    history: list = field(default_factory=list)
    evals: int = 0
    seconds: float = 0.0
    feasible: bool = False
    population: Optional[torch.Tensor] = None


class _FlatResiduals:
    """All constraint residuals of an evaluation as one [B, M] tensor (one op instead of one per constraint)."""

    def __init__(self, e: Evaluation, B: int):
        self.keys, self.shapes, self.sizes = [], [], []
        eq = []
        for key, h in e.resid.items():
            h = _full(h, B)
            self.keys.append(key)
            self.shapes.append(tuple(h.shape[1:]))
            n = int(h[0].numel())
            self.sizes.append(n)
            eq.append(torch.full((n,), e.kinds.get(key) == "eq", dtype=torch.bool))
        self.eq = torch.cat(eq) if eq else torch.zeros(0, dtype=torch.bool)
        self.M = int(self.eq.numel())

    def flat(self, e: Evaluation, B: int) -> torch.Tensor:
        if not self.keys:
            return torch.zeros(B, 0)
        return torch.cat([_full(e.resid[k], B).reshape(B, -1) for k in self.keys], 1)

    def from_dict(self, lam: Optional[dict], B: int) -> torch.Tensor:
        parts = []
        for k, shape, n in zip(self.keys, self.shapes, self.sizes):
            v = None if lam is None else lam.get(k)
            if v is None:
                parts.append(torch.zeros(B, n))
                continue
            v = _full(v.detach(), B)
            parts.append(v.reshape(B, -1)[:, :n] if v.reshape(B, -1).shape[1] >= n
                         else v.reshape(B, -1).expand(B, n))
        return torch.cat(parts, 1) if parts else torch.zeros(B, 0)

    def to_dict(self, flat: torch.Tensor) -> dict:
        out, i = {}, 0
        B = flat.shape[0]
        for k, shape, n in zip(self.keys, self.shapes, self.sizes):
            out[k] = flat[:, i:i + n].reshape(B, *shape)
            i += n
        return out


def alm(problem: Problem, x0: torch.Tensor, iters: int = 300, lr: float = 0.03, rho: float = 2.0,
        rho_max: float = 1e4, outer: int = 15, mask: Optional[torch.Tensor] = None, relax: bool = True,
        tol: float = 2e-3, callback: Optional[Callable] = None, time_budget: Optional[float] = None,
        lam0: Optional[dict] = None, lr_final: float = 0.2, log_every: int = 5, stage: str = "polish",
        anneal: float = 0.0, patience: Optional[int] = None, rtol: float = 2e-4) -> ALMResult:
    """anneal > 0 adds an integrality penalty (ramped up over the run) that drives integer,
    boolean and exclusive-routing decisions towards integral values before rounding.

    patience: stop early once the best feasible objective has not improved by more than `rtol`
    (relative) for `patience` iterations - converged stages do not burn their whole budget."""
    t_start = time.perf_counter()
    x = x0.detach().clone().clamp(0, 1).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))
    B = x.shape[0]
    hist: list = []
    best_x, best_key = None, None
    last_mv = math.inf
    evals0 = problem.n_evals
    fr: Optional[_FlatResiduals] = None
    lam = None
    best_obj_seen, stall = -math.inf, 0
    for it in range(iters):
        frac = it / max(1, iters - 1)
        for gparam in opt.param_groups:
            gparam["lr"] = lr * (lr_final + (1 - lr_final) * 0.5 * (1 + math.cos(math.pi * frac)))
        e = problem.evaluate(x, relax=relax)
        if fr is None:
            fr = _FlatResiduals(e, B)
            lam = fr.from_dict(lam0, B)
        h = fr.flat(e, B)
        Jn = e.objective / problem.J_scale
        if fr.M:
            z = torch.relu(lam + rho * h)
            pen_in = (z * z - lam * lam) / (2 * rho)
            pen_eq = lam * h + 0.5 * rho * h * h
            pen = torch.where(fr.eq, pen_eq, pen_in).sum(1)
        else:
            pen = torch.zeros(B)
        if anneal > 0.0:
            pen = pen + anneal * frac ** 2 * integrality_penalty(problem.space, x)
        loss = (-Jn + pen).sum()
        opt.zero_grad()
        loss.backward()
        if mask is not None:
            x.grad[:, ~mask] = 0.0
        opt.step()
        with torch.no_grad():
            x.clamp_(0.0, 1.0)
            hd = h.detach()
            mv = torch.where(fr.eq, hd.abs(), torch.relu(hd)).amax(1) if fr.M else torch.zeros(B)
            obj = e.objective.detach()
            for b in range(B):
                feas = bool(mv[b] <= tol)
                k = (feas, float(obj[b]) if feas else -float(mv[b]))
                if best_key is None or k > best_key:
                    best_key, best_x = k, e.x[b:b + 1].detach().clone()
            if (it + 1) % outer == 0 and fr.M:
                lam = torch.where(fr.eq, lam + rho * hd, torch.relu(lam + rho * hd))
                cur = float(mv.min())
                if cur > 0.5 * last_mv and cur > tol:
                    rho = min(rho * 3.0, rho_max)
                last_mv = cur
            if it % log_every == 0 or it == iters - 1:
                rec = {"stage": stage, "iter": it, "objective": float(obj.max()),
                       "max_violation": float(mv.min()), "rho": rho,
                       "t": time.perf_counter() - t_start, "evals": problem.n_evals - evals0}
                hist.append(rec)
                if callback:
                    callback(rec)
            if patience is not None and best_key is not None and best_key[0]:
                if best_key[1] > best_obj_seen + rtol * max(abs(best_obj_seen), 1.0):
                    best_obj_seen, stall = best_key[1], 0
                else:
                    stall += 1
                if stall >= patience:
                    break
        if time_budget is not None and time.perf_counter() - t_start > time_budget:
            break
    # final evaluation of the best iterate
    with torch.no_grad():
        ef = problem.evaluate(best_x, relax=relax)
    mult = fr.to_dict(lam[:1].detach()) if fr is not None and lam is not None else {}
    return ALMResult(best_x, float(ef.objective[0]), float(ef.max_violation()[0]), mult, hist,
                     problem.n_evals - evals0, time.perf_counter() - t_start,
                     float(ef.max_violation()[0]) <= tol, x.detach().clone())


def integrality_penalty(space, x: torch.Tensor) -> torch.Tensor:
    """sum of frac*(1-frac) over discrete decisions (value space) + s*(1-s) over exclusive routes."""
    parts = space.split(x)
    bnd = space.bounds()
    pen = torch.zeros(x.shape[0])
    for b in space.blocks:
        if b.domain not in ("int", "bool"):
            continue
        lo, hi, _ = bnd[(b.gid, b.name)]
        v = lo + (hi - lo) * parts[(b.gid, b.name)]
        f = v - torch.floor(v).detach()
        pen = pen + (f * (1 - f)).sum(dim=(1, 2))
    if space.route is not None and hasattr(space, "_excl") is False:
        space.normalise_routes(parts["route"][:, :, :1])
    if space.route is not None and bool(space._excl.any()):
        s = space.normalise_routes(parts["route"], relax=True)
        ex = space._excl.view(1, -1, 1).to(s.dtype)
        pen = pen + (s * (1 - s) * ex).sum(dim=(1, 2))
    return pen


def shadow_prices(problem: Problem, lam: dict) -> dict:
    """Convert multipliers to currency per canonical unit (per hour of binding)."""
    out = {}
    for key, l in lam.items():
        s = problem.con_scale.get(key)
        if s is None:
            continue
        out[key] = (l * problem.J_scale / torch.clamp(s, min=1e-9) / problem.dt).detach()
    return out


def _full(g: torch.Tensor, B: int) -> torch.Tensor:
    if g.dim() == 2:
        g = g.unsqueeze(-1)
    if g.shape[0] != B:
        g = g.expand(B, *g.shape[1:])
    return g
