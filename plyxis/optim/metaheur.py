"""Batched population metaheuristics on the box [0, 1]^N.

graviSearch (gravitational search), PSO, GA and DE are implemented fully
vectorised: one generation = one batched simulation of the whole population,
which uses every CPU core through torch.

Each method has a *cooperative* (separable) mode: when the fitness function
returns one score per subsystem ([P, K]) and `var_sid` maps every variable to a
subsystem, selection / attraction happens per subsystem. With a decomposed
model this turns one N-dimensional search into K small searches evaluated in
parallel - the classic remedy (cooperative coevolution) for the collapse of
metaheuristics in high dimension.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch


@dataclass
class MHResult:
    method: str
    best_x: torch.Tensor
    best_f: float
    history: list = field(default_factory=list)
    evals: int = 0
    seconds: float = 0.0


class _Tracker:
    def __init__(self, method, callback, log_every, budget):
        self.method, self.callback, self.log_every, self.budget = method, callback, log_every, budget
        self.t0 = time.perf_counter()
        self.history = []
        self.evals = 0

    def log(self, it, best_f, force=False):
        if it % self.log_every == 0 or force:
            rec = {"stage": self.method, "iter": it, "fitness": float(best_f), "evals": self.evals,
                   "t": time.perf_counter() - self.t0}
            self.history.append(rec)
            if self.callback:
                self.callback(rec)

    def out_of_time(self) -> bool:
        return self.budget is not None and time.perf_counter() - self.t0 > self.budget


def _eval(fit, X, tr: _Tracker, K: Optional[int]):
    f = fit(X)
    tr.evals += X.shape[0]
    if K is None and f.dim() == 2:
        f = f.sum(1)
    return f


def _assemble(X: torch.Tensor, F: torch.Tensor, var_sid: torch.Tensor) -> torch.Tensor:
    """Per subsystem pick the variables of the best individual: F [P, K] -> x [N]."""
    best_per_k = F.argmax(0)                 # [K]
    rows = best_per_k[var_sid]               # [N]
    return X[rows, torch.arange(X.shape[1])]


# ----------------------------------------------------------------------------- PSO
def pso(fit: Callable, N: int, pop: int = 32, iters: int = 100, x0: Optional[torch.Tensor] = None,
        var_sid: Optional[torch.Tensor] = None, K: Optional[int] = None, w=(0.9, 0.4), c1: float = 1.5,
        c2: float = 1.5, vmax: float = 0.2, seed: int = 0, callback=None, time_budget=None,
        log_every: int = 5, init_spread: float = 1.0) -> MHResult:
    g = torch.Generator().manual_seed(seed)
    tr = _Tracker("pso", callback, log_every, time_budget)
    X = torch.rand(pop, N, generator=g)
    if x0 is not None:
        X = (x0.view(1, -1) + init_spread * (X - 0.5) * 0.5).clamp(0, 1) if init_spread < 1.0 else X
        X[: x0.shape[0]] = x0
    V = (torch.rand(pop, N, generator=g) - 0.5) * 0.1
    F = _eval(fit, X, tr, K)
    Pb, Pf = X.clone(), F.clone()
    sep = var_sid is not None and F.dim() == 2
    for it in range(iters):
        ww = w[0] + (w[1] - w[0]) * it / max(1, iters - 1)
        if sep:
            G = _assemble(Pb, Pf, var_sid).view(1, -1)
        else:
            G = Pb[Pf.argmax()].view(1, -1)
        r1, r2 = torch.rand(pop, N, generator=g), torch.rand(pop, N, generator=g)
        V = (ww * V + c1 * r1 * (Pb - X) + c2 * r2 * (G - X)).clamp(-vmax, vmax)
        X = (X + V).clamp(0, 1)
        F = _eval(fit, X, tr, K)
        better = F > Pf
        if sep:
            m = better[:, var_sid]                       # per variable: did its subsystem improve?
            Pb = torch.where(m, X, Pb)
            Pf = torch.where(better, F, Pf)
        else:
            Pb[better] = X[better]
            Pf[better] = F[better]
        best = _assemble(Pb, Pf, var_sid) if sep else Pb[Pf.argmax()]
        tr.log(it, Pf.sum(1).max() if sep else Pf.max())
        if tr.out_of_time():
            break
    if sep:
        best = _assemble(Pb, Pf, var_sid).view(1, -1)
        bf = _eval(fit, best, tr, None)
        bf = float(bf.sum()) if bf.dim() > 1 else float(bf)
        return MHResult("pso-coop", best, bf, tr.history, tr.evals, time.perf_counter() - tr.t0)
    return MHResult("pso", Pb[Pf.argmax()].view(1, -1), float(Pf.max()), tr.history, tr.evals,
                    time.perf_counter() - tr.t0)


# ----------------------------------------------------------------------------- graviSearch (GSA)
def gravisearch(fit: Callable, N: int, pop: int = 32, iters: int = 100, x0: Optional[torch.Tensor] = None,
                G0: float = 0.6, alpha: float = 12.0, seed: int = 0, callback=None, time_budget=None,
                log_every: int = 5, var_sid=None, K=None) -> MHResult:
    """Gravitational search (Rashedi et al. 2009): agents attract each other with forces
    proportional to fitness-derived masses; the attracting set (k-best) shrinks over time."""
    g = torch.Generator().manual_seed(seed)
    tr = _Tracker("gravisearch", callback, log_every, time_budget)
    X = torch.rand(pop, N, generator=g)
    if x0 is not None:
        X[: x0.shape[0]] = x0
    V = torch.zeros(pop, N)
    F = _eval(fit, X, tr, None)
    best_x, best_f = X[F.argmax()].clone(), float(F.max())
    for it in range(iters):
        Gt = G0 * math.exp(-alpha * it / iters)
        worst, best = F.min(), F.max()
        m = (F - worst) / (best - worst + 1e-12)
        M = m / (m.sum() + 1e-12)
        kbest = max(1, int(round(pop * (1 - it / iters))))
        idx = torch.topk(F, kbest).indices
        Xk, Mk = X[idx], M[idx]
        diff = Xk.unsqueeze(0) - X.unsqueeze(1)                          # [P, k, N]
        R = diff.norm(dim=2, keepdim=True) / math.sqrt(N)
        rnd = torch.rand(pop, kbest, 1, generator=g)
        Acc = (rnd * Gt * Mk.view(1, -1, 1) * diff / (R + 1e-3)).sum(1)
        V = torch.rand(pop, N, generator=g) * V + Acc
        X = (X + V.clamp(-0.2, 0.2)).clamp(0, 1)
        F = _eval(fit, X, tr, None)
        if float(F.max()) > best_f:
            best_f, best_x = float(F.max()), X[F.argmax()].clone()
        tr.log(it, best_f)
        if tr.out_of_time():
            break
    return MHResult("gravisearch", best_x.view(1, -1), best_f, tr.history, tr.evals, time.perf_counter() - tr.t0)


# ----------------------------------------------------------------------------- GA
def ga(fit: Callable, N: int, pop: int = 32, iters: int = 100, x0: Optional[torch.Tensor] = None,
       var_sid: Optional[torch.Tensor] = None, K: Optional[int] = None, pc: float = 0.9,
       pm: Optional[float] = None, sigma: float = 0.1, elite: int = 2, seed: int = 0, callback=None,
       time_budget=None, log_every: int = 5, mask: Optional[torch.Tensor] = None,
       discrete: Optional[torch.Tensor] = None, repair: Optional[Callable] = None) -> MHResult:
    """Real-coded GA: tournament selection, uniform crossover, gaussian / bit-flip mutation.
    Cooperative mode selects parents per subsystem. `mask` restricts the search to some variables;
    `repair(X)` (e.g. a short gradient polish) turns it into a memetic algorithm."""
    g = torch.Generator().manual_seed(seed)
    tr = _Tracker("ga" if var_sid is None else "ga-coop", callback, log_every, time_budget)
    pm = pm if pm is not None else min(0.2, 4.0 / N)
    if x0 is not None and x0.shape[0] >= pop:
        X = x0[:pop].clone()
    else:
        X = torch.rand(pop, N, generator=g)
        if x0 is not None:
            X[: x0.shape[0]] = x0
            base = x0[0].view(1, -1)
            if mask is not None:
                X = torch.where(mask.view(1, -1), X, base)
    if repair is not None:
        X = repair(X)
    F = _eval(fit, X, tr, K)
    sep = var_sid is not None and F.dim() == 2
    for it in range(iters):
        Ftot = F.sum(1) if sep else F
        order = torch.argsort(Ftot, descending=True)
        kids = []
        nk = pop - elite
        if sep:
            # per subsystem tournament: each variable inherits from the winner for its subsystem
            a = torch.randint(0, pop, (nk, F.shape[1]), generator=g)
            b = torch.randint(0, pop, (nk, F.shape[1]), generator=g)
            ka = torch.where(F[a, torch.arange(F.shape[1])] >= F[b, torch.arange(F.shape[1])], a, b)
            a2 = torch.randint(0, pop, (nk, F.shape[1]), generator=g)
            b2 = torch.randint(0, pop, (nk, F.shape[1]), generator=g)
            kb = torch.where(F[a2, torch.arange(F.shape[1])] >= F[b2, torch.arange(F.shape[1])], a2, b2)
            pa = X[ka[:, var_sid], torch.arange(N)]
            pb = X[kb[:, var_sid], torch.arange(N)]
        else:
            a = torch.randint(0, pop, (nk, 2), generator=g)
            win1 = torch.where(F[a[:, 0]] >= F[a[:, 1]], a[:, 0], a[:, 1])
            a = torch.randint(0, pop, (nk, 2), generator=g)
            win2 = torch.where(F[a[:, 0]] >= F[a[:, 1]], a[:, 0], a[:, 1])
            pa, pb = X[win1], X[win2]
        cross = torch.rand(nk, N, generator=g) < 0.5
        do = (torch.rand(nk, 1, generator=g) < pc)
        child = torch.where(cross & do, pb, pa)
        mut = torch.rand(nk, N, generator=g) < pm
        noise = torch.randn(nk, N, generator=g) * sigma
        child = torch.where(mut, child + noise, child)
        if discrete is not None:
            flip = mut & discrete.view(1, -1)
            child = torch.where(flip, 1.0 - child.round(), child)
        if mask is not None:
            child = torch.where(mask.view(1, -1), child, X[order[0]].view(1, -1))
        child = child.clamp(0, 1)
        if repair is not None:
            child = repair(child)
        Fc = _eval(fit, child, tr, K)
        X = torch.cat([X[order[:elite]], child], 0)
        F = torch.cat([F[order[:elite]], Fc], 0)
        tr.log(it, (F.sum(1) if sep else F).max())
        if tr.out_of_time():
            break
    Ftot = F.sum(1) if sep else F
    i = int(Ftot.argmax())
    return MHResult(tr.method, X[i:i + 1].clone(), float(Ftot[i]), tr.history, tr.evals,
                    time.perf_counter() - tr.t0)


# ----------------------------------------------------------------------------- DE
def de(fit: Callable, N: int, pop: int = 32, iters: int = 100, x0: Optional[torch.Tensor] = None,
       F_w: float = 0.6, CR: float = 0.9, seed: int = 0, callback=None, time_budget=None,
       log_every: int = 5, var_sid=None, K=None) -> MHResult:
    """Differential evolution DE/rand/1/bin (the 'DEoptim' family)."""
    g = torch.Generator().manual_seed(seed)
    tr = _Tracker("de", callback, log_every, time_budget)
    X = torch.rand(pop, N, generator=g)
    if x0 is not None:
        X[: x0.shape[0]] = x0
    F = _eval(fit, X, tr, None)
    for it in range(iters):
        r = torch.stack([torch.randperm(pop, generator=g)[:3] for _ in range(pop)])
        Vm = X[r[:, 0]] + F_w * (X[r[:, 1]] - X[r[:, 2]])
        cross = torch.rand(pop, N, generator=g) < CR
        jr = torch.randint(0, N, (pop,), generator=g)
        cross[torch.arange(pop), jr] = True
        U = torch.where(cross, Vm, X).clamp(0, 1)
        Fu = _eval(fit, U, tr, None)
        better = Fu > F
        X = torch.where(better.view(-1, 1), U, X)
        F = torch.where(better, Fu, F)
        tr.log(it, F.max())
        if tr.out_of_time():
            break
    i = int(F.argmax())
    return MHResult("de", X[i:i + 1].clone(), float(F[i]), tr.history, tr.evals, time.perf_counter() - tr.t0)


METHODS = {"pso": pso, "gravisearch": gravisearch, "ga": ga, "de": de}


# ----------------------------------------------------------------------------- test functions
def rastrigin(X: torch.Tensor) -> torch.Tensor:
    Z = (X - 0.5) * 10.24
    return -(10 * Z.shape[1] + (Z * Z - 10 * torch.cos(2 * math.pi * Z)).sum(1))


def sphere(X: torch.Tensor) -> torch.Tensor:
    return -((X - 0.3) ** 2).sum(1)
