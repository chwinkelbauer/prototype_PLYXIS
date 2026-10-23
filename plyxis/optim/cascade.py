"""PLYXIS Cascade - the layered solver.

    Stage 0  Compile & decompose     netlist -> decision space, subsystems, scales
    Stage 1  Neural sub-solvers      learned best responses + price coordination      (ms)
    Stage 2  Coordinated relaxation  multi-start augmented Lagrangian on the full twin  (s)
             (integer / boolean / exclusive-routing decisions annealed towards integrality)
    Stage 3  Discrete search         memetic GA: randomised roundings, cooperative       (s)
             recombination per subsystem, batched gradient repair
    Stage 4  Polish & verify         continuous refinement with discrete decisions fixed;
                                     exact simulation; shadow prices

Every stage streams progress records (for the optimizer console in the GUI) and
the result carries the convergence history, stage timings and shadow prices.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from . import metaheur as MH
from .gradient import alm, shadow_prices
from .partition import Partition, build_partition, subsystem_fitness
from .problem import Problem


@dataclass
class CascadeConfig:
    relax_iters: int = 220
    relax_lr: float = 0.04
    relax_rho: float = 0.05
    anneal: float = 0.05
    population: int = 16
    generations: int = 2
    repair_iters: int = 90
    polish_iters: int = 60
    neural_iters: int = 8
    use_neural: bool = True
    multistart: int = 3
    seed: int = 0
    time_budget: Optional[float] = None


@dataclass
class CascadeResult:
    x: torch.Tensor
    objective: float
    max_violation: float
    feasible: bool
    prices: dict
    multipliers: dict
    history: list = field(default_factory=list)
    stages: list = field(default_factory=list)
    seconds: float = 0.0
    evals: int = 0
    baseline_objective: Optional[float] = None


class Cascade:
    def __init__(self, problem: Problem, net=None, partition: Optional[Partition] = None,
                 config: Optional[CascadeConfig] = None):
        self.pb = problem
        self.net = net
        self.cfg = config or CascadeConfig()
        self.part = partition or build_partition(problem.prog, problem.space)

    def _emit(self, cb, rec):
        rec = dict(rec)
        rec["t_total"] = time.perf_counter() - self._t0
        self._hist.append(rec)
        if cb:
            cb(rec)

    def solve(self, warm: Optional[torch.Tensor] = None, lam0: Optional[dict] = None,
              callback: Optional[Callable] = None) -> CascadeResult:
        pb, sp, cfg = self.pb, self.pb.space, self.cfg
        self._t0 = time.perf_counter()
        self._hist = []
        stages = []
        evals0 = pb.n_evals
        gen = torch.Generator().manual_seed(cfg.seed)
        mask = sp.discrete_mask()

        # ------------------------------------------------ stage 0: problem summary
        self._emit(callback, {"stage": "compile", "event": "start", "variables": sp.N,
                              "discrete": int(mask.sum()), "subsystems": self.part.K})
        stages.append({"stage": "compile", "seconds": 0.0, "variables": sp.N, "discrete": int(mask.sum()),
                       "subsystems": self.part.K})

        # ------------------------------------------------ stage 1: neural sub-solvers
        starts = []
        t = time.perf_counter()
        if self.net is not None and cfg.use_neural:
            from .neural import dual_coordination
            nres = dual_coordination(pb, self.net, iters=cfg.neural_iters, lam0=lam0,
                                     callback=lambda r: self._emit(callback, r))
            starts.append(nres.x)
            stages.append({"stage": "neural", "seconds": time.perf_counter() - t, "objective": nres.objective,
                           "max_violation": nres.max_violation})
        if warm is not None:
            starts.append(warm.view(1, -1))
        x_def = sp.default_u(1)
        if len(starts) < cfg.multistart:
            eq = x_def.clone()
            sp.split(eq)["route"][:] = 0.5 if sp.route is not None else 0
            starts.append(eq)
        if len(starts) < cfg.multistart:
            starts.append(x_def)
        while len(starts) < cfg.multistart:
            starts.append((starts[0] + 0.15 * (torch.rand(1, sp.N, generator=gen) - 0.5)).clamp(0, 1))
        X0 = torch.cat(starts[:max(cfg.multistart, 1)], 0)

        # ------------------------------------------------ stage 2: coordinated relaxation
        t = time.perf_counter()
        rel = alm(pb, X0, iters=cfg.relax_iters, lr=cfg.relax_lr, rho=cfg.relax_rho, outer=10,
                  anneal=cfg.anneal, stage="relaxation", callback=lambda r: self._emit(callback, r),
                  lam0=_expand_lam(lam0, X0.shape[0]) if lam0 else None)
        stages.append({"stage": "relaxation", "seconds": time.perf_counter() - t, "objective": rel.objective,
                       "max_violation": rel.max_violation, "starts": X0.shape[0]})

        # ------------------------------------------------ stage 3: discrete search (memetic, cooperative)
        t = time.perf_counter()
        pop = randomized_roundings(sp, rel.x, cfg.population, gen, extra=[warm] if warm is not None else None)
        best_x, best_obj, best_mv = None, -float("inf"), float("inf")
        lam_rep = None
        n_gen = cfg.generations
        if int(mask.sum()) == 0:          # nothing discrete: go straight to the polish
            n_gen = 0
            with torch.no_grad():
                e_rel = pb.evaluate(rel.x, relax=False)
            best_x, best_obj, best_mv = rel.x.clone(), float(e_rel.objective[0]), float(e_rel.max_violation()[0])
            lam_rep = rel.multipliers
        for gen_i in range(n_gen):
            rep = alm(pb, pop, iters=cfg.repair_iters, lr=0.02, rho=0.5, outer=10, mask=~mask, relax=False,
                      stage="discrete", callback=lambda r, g=gen_i: self._emit(callback, dict(r, generation=g)),
                      lam0=None)
            lam_rep = rep.multipliers
            with torch.no_grad():
                ev = pb.evaluate(rep_population(rep, pop), relax=False)
            score = _score(pb, ev)
            i = int(score.argmax())
            mv = float(ev.max_violation()[i])
            obj = float(ev.objective[i])
            if (mv <= 2e-3 and (best_mv > 2e-3 or obj > best_obj)) or (best_mv > 2e-3 and mv < best_mv):
                best_x, best_obj, best_mv = ev.x[i:i + 1].clone(), obj, mv
            self._emit(callback, {"stage": "discrete", "event": "generation", "generation": gen_i,
                                  "objective": obj, "max_violation": mv,
                                  "population_objectives": [float(v) for v in ev.objective]})
            if gen_i + 1 < cfg.generations:
                # cooperative recombination: per subsystem take the discrete choices of the best individual
                F = subsystem_fitness(pb, ev, self.part)
                coop = MH._assemble(ev.x, F, self.part.var_sid).view(1, -1)
                children = randomized_roundings(sp, ev.x[i:i + 1], cfg.population - 2, gen, flip=0.08)
                pop = torch.cat([ev.x[i:i + 1], coop, children], 0)
                pop = sp.round_discrete(pop)
        stages.append({"stage": "discrete", "seconds": time.perf_counter() - t, "objective": best_obj,
                       "max_violation": best_mv, "population": cfg.population, "generations": cfg.generations})

        # ------------------------------------------------ stage 4: polish & verify
        t = time.perf_counter()
        pol = alm(pb, best_x, iters=cfg.polish_iters + (cfg.repair_iters if n_gen == 0 else 0),
                  lr=0.01 if n_gen else 0.02, rho=1.0, outer=10, mask=~mask, relax=False,
                  stage="polish", callback=lambda r: self._emit(callback, r), lam0=lam_rep and
                  {k: v[:1] for k, v in lam_rep.items()})
        x_final = pol.x if (pol.max_violation <= max(best_mv, 2e-3) + 1e-9 and pol.objective >= best_obj - 1) or \
            pol.max_violation < best_mv else best_x
        with torch.no_grad():
            ef = pb.evaluate(x_final, relax=False)
        obj, mv = float(ef.objective[0]), float(ef.max_violation()[0])
        prices = shadow_prices(pb, pol.multipliers)
        stages.append({"stage": "polish", "seconds": time.perf_counter() - t, "objective": obj, "max_violation": mv})
        self._emit(callback, {"stage": "done", "objective": obj, "max_violation": mv, "feasible": mv <= 2e-3})
        return CascadeResult(x_final, obj, mv, mv <= 2e-3, prices, pol.multipliers, self._hist, stages,
                             time.perf_counter() - self._t0, pb.n_evals - evals0)


# ----------------------------------------------------------------------------- helpers
def rep_population(rep, pop):
    """Final repaired population of a batched ALM run."""
    return rep.population if rep.population is not None else pop


def _score(pb: Problem, ev) -> torch.Tensor:
    return ev.objective / pb.J_scale - 50.0 / pb.T * ev.sum_violation()


def _expand_lam(lam: dict, B: int) -> dict:
    out = {}
    for k, v in lam.items():
        v = v if v.dim() == 3 else v.unsqueeze(-1)
        out[k] = v.expand(B, *v.shape[1:]).clone() if v.shape[0] == 1 else v
    return out


def randomized_roundings(sp, x: torch.Tensor, P: int, gen: torch.Generator, extra=None,
                         flip: float = 0.0) -> torch.Tensor:
    """P candidate integer solutions around a relaxed solution (first = nearest rounding)."""
    bnd = sp.bounds()
    out = []
    for k in range(P):
        y = x[:1].clone()
        parts = sp.split(y)
        for b in sp.blocks:
            if b.domain not in ("int", "bool"):
                continue
            lo, hi, _ = bnd[(b.gid, b.name)]
            u = parts[(b.gid, b.name)]
            v = lo + (hi - lo) * u
            fl = torch.floor(v)
            f = v - fl
            if k == 0:
                vv = torch.round(v)
            else:
                vv = fl + (torch.rand(f.shape, generator=gen) < f).float()
                if flip > 0:
                    fl2 = (torch.rand(f.shape, generator=gen) < flip).float()
                    vv = torch.where(fl2 > 0, torch.clamp(vv + torch.where(torch.rand(f.shape, generator=gen) < 0.5,
                                                                           -1.0, 1.0), lo, hi), vv)
            parts[(b.gid, b.name)].copy_(torch.where(hi - lo > 1e-9, (vv - lo) / (hi - lo).clamp(min=1e-9),
                                                     torch.zeros_like(u)))
        if sp.route is not None:
            r = parts["route"]
            s = sp.normalise_routes(r)
            for rp in sp.program.routed:
                if rp.exclusive and len(rp.ridx) > 1:
                    probs = s[0, rp.ridx, :].T
                    if k == 0:
                        ch = probs.argmax(1)
                    else:
                        if flip > 0:
                            probs = (1 - flip) * probs + flip / probs.shape[1]
                        ch = torch.multinomial(probs.clamp(min=1e-6), 1, generator=gen).squeeze(1)
                    r[0, rp.ridx, :] = torch.zeros_like(probs).scatter_(1, ch.view(-1, 1), 1.0).T
        out.append(y)
    if extra:
        for e in extra:
            if e is not None:
                out.append(sp.round_discrete(e.view(1, -1)))
    return torch.cat(out, 0)
