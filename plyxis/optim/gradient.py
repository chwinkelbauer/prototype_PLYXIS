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


def alm(problem: Problem, x0: torch.Tensor, iters: int = 300, lr: float = 0.03, rho: float = 2.0,
        rho_max: float = 1e4, outer: int = 15, mask: Optional[torch.Tensor] = None, relax: bool = True,
        tol: float = 2e-3, callback: Optional[Callable] = None, time_budget: Optional[float] = None,
        lam0: Optional[dict] = None, lr_final: float = 0.2, log_every: int = 5, stage: str = "polish",
        anneal: float = 0.0) -> ALMResult:
    """anneal > 0 adds an integrality penalty (ramped up over the run) that drives integer,
    boolean and exclusive-routing decisions towards integral values before rounding."""
    t_start = time.perf_counter()
    x = x0.detach().clone().clamp(0, 1).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))
    B = x.shape[0]
    lam: dict = {} if lam0 is None else {k: v.detach().clone() for k, v in lam0.items()}
    hist: list = []
    best_x, best_key = None, None
    last_mv = math.inf
    evals0 = problem.n_evals
    for it in range(iters):
        frac = it / max(1, iters - 1)
        for gparam in opt.param_groups:
            gparam["lr"] = lr * (lr_final + (1 - lr_final) * 0.5 * (1 + math.cos(math.pi * frac)))
        e = problem.evaluate(x, relax=relax)
        Jn = e.objective / problem.J_scale
        pen = torch.zeros(B)
        for key, h in e.resid.items():
            h = _full(h, B)
            if key not in lam:
                lam[key] = torch.zeros_like(h)
            if e.kinds.get(key) == "eq":
                pen = pen + (lam[key] * h).sum(dim=(1, 2)) + 0.5 * rho * (h * h).sum(dim=(1, 2))
            else:   # Powell-Hestenes-Rockafellar term for h <= 0
                z = torch.relu(lam[key] + rho * h)
                pen = pen + ((z * z - lam[key] * lam[key]) / (2 * rho)).sum(dim=(1, 2))
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
        with torch.no_grad():
            mv = e.max_violation()
            obj = e.objective
            key_rows = [(bool(mv[b] <= tol), float(obj[b]), -float(mv[b])) for b in range(B)]
            for b in range(B):
                k = (key_rows[b][0], key_rows[b][1] if key_rows[b][0] else key_rows[b][2])
                if best_key is None or k > best_key:
                    best_key, best_x = k, e.x[b:b + 1].detach().clone()
            if (it + 1) % outer == 0:
                for key, h in e.resid.items():
                    h = _full(h, B).detach()
                    if e.kinds.get(key) == "eq":
                        lam[key] = lam[key] + rho * h
                    else:
                        lam[key] = torch.relu(lam[key] + rho * h)
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
        if time_budget is not None and time.perf_counter() - t_start > time_budget:
            break
    # final evaluation of the best iterate
    with torch.no_grad():
        ef = problem.evaluate(best_x, relax=relax)
    return ALMResult(best_x, float(ef.objective[0]), float(ef.max_violation()[0]),
                     {k: v[:1].detach() for k, v in lam.items()}, hist, problem.n_evals - evals0,
                     time.perf_counter() - t_start, float(ef.max_violation()[0]) <= tol, x.detach().clone())


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
