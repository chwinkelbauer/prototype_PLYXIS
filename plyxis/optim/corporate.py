"""Corporate planning with plants at their own time step: column generation between the group and its plants.

`plyxis/lang/hierarchy.py` splits one twin into a master program (the group at its step, every plant an
envelope of column slots) and one program per plant (its units at the plant's step, every connection that leaves
it priced at its boundary). This module alternates them (Dantzig-Wolfe, convexified per block):

1. price: the derivative of the master's Lagrangian - objective minus the multipliers times the constraints, as
   the ALM sees them - with respect to the plant's boundary flows, hour by hour (the probe slot carries them).
   What a tonne delivered to a market, a tonne of scrap bought or a slab sent to a sister plant is worth to the
   group in that hour, with every binding limit in it;
2. plant: solve the plant's program against those prices - its units at the plant's step, all its own costs,
   inventories and limits. One schedule: its boundary flows and its internal value, both per master step;
3. master: the schedule becomes a column; the master re-solves the group with the weights of the plant's columns
   (one per block of hours, adding up to one in every hour) as decisions.

Round 0 seeds every plant with schedules at a few price levels (zero = the plant on its own, and the group's
prices as it stands, scaled), so the first master can always mix a feasible plan. Prices are smoothed between
rounds (they over-react to the last column otherwise). It stops when no plant finds a schedule worth more at the
current prices than the master's mix of its columns (the reduced cost, relative to the plan's value), or after
`rounds`.

A mix of schedules is not a schedule. At the end each plant re-solves holding its flows near the master's mix
(`tracking`), and that one schedule is committed: the master re-plans the group around it. The value reported
is the committed plan's, not the mix's.

The blocks are independent given their prices: with `workers` > 1 they are solved side by side in worker processes,
each of which compiles its own copy of the twin (`Hierarchy.args`) once.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from ..lang.program import TERM_KINDS, _bnt, _ident
from .cascade import Cascade, CascadeConfig
from .problem import Problem


@dataclass
class CorporateConfig:
    rounds: int = 4
    master: CascadeConfig = field(default_factory=CascadeConfig)
    plant: CascadeConfig = field(default_factory=CascadeConfig)
    tol: float = 1e-3                 # stop when every plant's reduced cost is below this share of the plan's value
    smoothing: float = 0.5            # weight of the master's new prices against the last ones (dual smoothing)
    start_scales: tuple = (0.0, 0.7, 1.0)   # round 0: schedules at these multiples of the group's prices as it stands
                                            # (0 = the plant on its own)
    extra_scales: tuple = (1.3, 1.6)  # ... and at these where fewer than two of those keep the block's own limits (a
                                      # works that cannot run at part load: its furnaces have a minimum wind)
    col_tol: float = 0.01             # a schedule breaking its block's own limits by more than this is no point of the
                                      # block's input-output function: kept out of the master while the block has one
                                      # that keeps them
    commit: bool = True               # track the master's mix with one schedule and re-plan the group around it
    commit_iters: int = 3             # re-price and re-solve while the committed plan breaks a limit of the master
    price_viol: float = 0.05          # masters up to this violation still price with their multipliers
    new_weight: float = 0.2           # a new column enters the master's warm start with this weight (at 0 a converged
                                      # master never moves it)
    feas_tol: float = 2e-3            # a committed schedule must keep its block's limits to this (the cascade's own)
    phase1: bool = True               # if the first master is infeasible: an artificial schedule per block (no flows at
                                      # the fence, a large cost) keeps it feasible while the real ones cannot share a
                                      # limit, so its multipliers price that limit (Phase I)
    phase1_cost: float = 2.0          # its value per hour: the worst real one's minus this many times their mean size
    workers: int = 0                  # > 1: solve the blocks side by side in this many processes
    worker_threads: int = 1           # torch threads per worker process


@dataclass
class PlantState:
    columns: list = field(default_factory=list)      # dicts: flows [C, T_m], value_h [T_m], internal, x
    x: Optional[torch.Tensor] = None                 # last plant solution (warm start)
    prices: Optional[torch.Tensor] = None            # [C, T_m] canonical: d(Lagrangian)/d(flow) per master step


@dataclass
class CorporatePlan:
    master_x: torch.Tensor
    objective: float                  # the committed plan: the group re-planned around each plant's one schedule
    feasible: bool
    mix_objective: float              # the master's value with mixed columns (what column generation converged to)
    weights: dict                     # plant -> [K, T_m] weights of its columns in the last mix
    plant_x: dict                     # plant -> x of the schedule it runs
    plant_eval: dict                  # plant -> Evaluation of that schedule
    plant_flows: dict                 # plant -> [C, T_m] boundary flows of that schedule
    mix_flows: dict                   # plant -> [C, T_m] the master's mix of columns
    rounds: list                      # per round: values, reduced costs, seconds
    seconds: float = 0.0
    plant_violation: dict = field(default_factory=dict)   # plant -> max violation of its own limits in the schedule
    master_feasible: bool = False     # the group's program alone (feasible = this and every plant's schedule)


def _upsample(v, ratio: int, T_m: int):
    if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[-1] == T_m and ratio > 1:
        return v.repeat_interleave(ratio, dim=-1)
    return v


def objective_coefficients(prog, E: Optional[dict] = None) -> dict:
    """The simulation's objective is linear in the totals (revenue, cost, emission, metric): its coefficients."""
    zero = {k: torch.tensor(0.0) for k in TERM_KINDS}
    try:
        base = float(prog._objective(dict(zero), {}, 0.0, E))
        out = {}
        for k in TERM_KINDS:
            t = dict(zero)
            t[k] = torch.tensor(1.0)
            out[k] = float(prog._objective(t, {}, 0.0, E)) - base
        return out
    except KeyError:                          # an objective over named terms: revenue - cost
        return {"revenue": 1.0, "cost": -1.0, "emission": 0.0, "metric": 0.0}


def step_values(prog, raw: dict, dt: float, E: Optional[dict], exclude=(), coefs: Optional[dict] = None) -> torch.Tensor:
    """The objective of one evaluated schedule (batch 0) step by step: terms, soft penalties, end values at the last
    step. Adds up to Program.evaluate's objective (minus the terms in `exclude`)."""
    coefs = coefs or objective_coefficients(prog, E)
    T = int(raw.get("__T__", 1))
    v = torch.zeros(T)
    for tr in prog.terms:
        c = coefs.get(tr.kind, 0.0)
        if tr.name in exclude or c == 0.0:
            continue
        if tr.final:
            x = raw[f"F_{tr.gid}_{_ident(tr.key)}"]
            v[-1] += c * float(x.reshape(x.shape[0], -1)[0].sum())
            continue
        x = _bnt(raw[prog.node_var(tr.gid, tr.key)], prog.gmap[tr.gid].n)[0].detach()
        per = (x.sum(0) * dt).reshape(-1)
        v += c * (per.expand(T) if per.numel() == 1 else per)
    bnd = prog._bounds_cache()
    for cr in prog.cons:
        if not cr.soft:
            continue
        h = _bnt(raw[prog.node_var(cr.gid, cr.key)], prog.gmap[cr.gid].n)[0].detach()
        x = h.abs() if cr.kind == "eq" else torch.relu(h)
        w = bnd[cr.gid].get(f"soft:{cr.key}", 0.0)
        w = (w if isinstance(w, torch.Tensor) else torch.tensor(float(w))).reshape(-1)
        w = w.view(-1, 1) if w.numel() == x.shape[0] else w.reshape(())
        per = (x * w).sum(0) * dt
        v -= per.reshape(-1).expand(T) if per.numel() == 1 else per.reshape(-1)
    return v


class CorporatePlanner:
    def __init__(self, hier, config: Optional[CorporateConfig] = None, E: Optional[dict] = None, t0: float = 0.0):
        self.h = hier
        self.cfg = config or CorporateConfig()
        self.E = dict(E or {})
        self.t0 = t0
        self.pbm = Problem(hier.master, E=self.E, t0=t0)
        self.T_m, self.dt_m = self.pbm.T, self.pbm.dt
        self.pbp, self.coefs = {}, {}
        for name, p in hier.plants.items():
            Ep = {k: _upsample(v, p.ratio, self.T_m) for k, v in self.E.items()}
            self.pbp[name] = Problem(p.program, E=Ep, t0=t0)
            self.coefs[name] = objective_coefficients(p.program, Ep)
            assert self.pbp[name].T == self.T_m * p.ratio, (name, self.pbp[name].T, self.T_m, p.ratio)
        self.state = {name: PlantState() for name in hier.plants}
        self.P = {g: {k: v.clone() for k, v in d.items()} for g, d in hier.master.P0.items()}
        self.S0 = {g: {k: v.clone() for k, v in d.items()} for g, d in hier.master.S0.items()}

    # ------------------------------------------------------------------ master
    def _slot_traj(self, gid: str, name: str) -> torch.Tensor:
        """A column slot state as a [1, n, T_m] trajectory (exogenous input)."""
        v = self.S0[gid][name]
        if v.dim() == 2:
            v = v.unsqueeze(-1).expand(1, v.shape[1], self.T_m).clone()
            self.S0[gid][name] = v
        return v

    def _load_column(self, plant, j: int, col: dict, fixed: bool = False) -> None:
        gid, li = plant.slots[j]
        self.P[gid]["w_hi"][0, li, 0] = 1.0
        self.P[gid]["w_lo"][0, li, 0] = 1.0 if fixed else 0.0
        self._slot_traj(gid, "cost_h")[0, li, :] = -col["value_h"] / self.dt_m
        for c in plant.crossings:
            self._slot_traj(gid, f"f{c.k}")[0, li, :] = col["flows"][c.k]

    def _unload(self, plant, j: int) -> None:
        gid, li = plant.slots[j]
        self.P[gid]["w_hi"][0, li, 0] = 0.0
        self.P[gid]["w_lo"][0, li, 0] = 0.0

    def master_problem(self) -> Problem:
        return self.pbm.with_context(P=self.P, S0=self.S0)

    def prices(self, pb: Problem, x: torch.Tensor, lam: Optional[dict]) -> dict:
        """d(objective - J_scale * sum(lambda * residual)) / d(probe flows) for every plant: [C, T_m] each."""
        S0 = {g: dict(d) for g, d in pb.S0.items()}
        deltas = {}
        for name, p in self.h.plants.items():
            gid, li = p.slots[-1]
            for c in p.crossings:
                base = self._slot_traj(gid, f"f{c.k}")
                d = torch.zeros_like(base, requires_grad=True)
                deltas[(name, c.k)] = (d, li)
                S0[gid][f"f{c.k}"] = base + d
        e = pb.evaluate(x[:1].detach(), relax=False, S0=S0)
        L = e.objective.sum()
        if lam:
            for key, l in lam.items():
                h = e.resid.get(key)
                if h is not None:
                    L = L - pb.J_scale * (l.detach().reshape(h.shape) * h).sum()
        grads = torch.autograd.grad(L, [d for d, _ in deltas.values()], allow_unused=True)
        out = {name: torch.zeros(len(p.crossings), self.T_m) for name, p in self.h.plants.items()}
        for ((name, k), (d, li)), g in zip(deltas.items(), grads):
            if g is not None:
                out[name][k] = g[0, li, :].detach()
        return out

    def weights(self, x: torch.Tensor) -> dict:
        """The master's weights of every plant's columns, per master step: [K, T_m]."""
        pb = self.master_problem()
        D, _ = pb.space.decode(x[:1], pb.P, relax=False)
        out = {}
        for name, p in self.h.plants.items():
            K = len(self.state[name].columns)
            out[name] = torch.stack([D[p.slots[j][0]]["w"][0, p.slots[j][1], :].detach() for j in range(K)]) \
                if K else torch.zeros(0, self.T_m)
        return out

    # ------------------------------------------------------------------ plants
    def _plant_problem(self, name: str, prices: torch.Tensor, track: Optional[tuple] = None) -> Problem:
        """The plant's program with its boundary priced (and, to commit, its flows tracked)."""
        p = self.h.plants[name]
        pb0 = self.pbp[name]
        S0 = {g: dict(d) for g, d in pb0.S0.items()}
        P = {g: dict(d) for g, d in pb0.P.items()}
        T_p = pb0.T
        per_unit = prices / self.dt_m                                  # value per unit of flow and hour
        for c, (gid, li) in zip(p.crossings, p.sinks):
            for nm in ("price", "target"):
                cur = S0[gid][nm]
                cur = cur.unsqueeze(-1).expand(1, cur.shape[1], T_p).clone() if cur.dim() == 2 else cur.clone()
                if nm == "price":
                    cur[0, li, :] = per_unit[c.k].repeat_interleave(p.ratio)
                elif track is not None:
                    cur[0, li, :] = track[0][c.k].repeat_interleave(p.ratio)
                S0[gid][nm] = cur
            P[gid]["track"] = P[gid]["track"].clone()
            P[gid]["track"][0, li, 0] = 0.0 if track is None else float(track[1][c.k])
        return pb0.with_context(P=P, S0=S0)

    def solve_plant(self, name: str, prices: torch.Tensor, track: Optional[tuple] = None, callback=None) -> dict:
        """The plant's best schedule against boundary prices [C, T_m] (canonical, d value / d flow per master step)."""
        st = self.state[name]
        pb = self._plant_problem(name, prices, track)
        t = time.perf_counter()
        res = Cascade(pb, config=self.cfg.plant).solve(warm=st.x, callback=callback)
        return self._column(name, pb, res.x[:1].detach().clone(), bool(res.feasible), float(res.max_violation),
                            time.perf_counter() - t)

    def _column(self, name: str, pb: Problem, x: torch.Tensor, feasible: bool, max_violation: float,
                seconds: float) -> dict:
        p, st = self.h.plants[name], self.state[name]
        T_p = pb.T
        xs = x
        with torch.no_grad():
            ev = pb.evaluate(xs, relax=False)
        named = ev.totals["named"]
        one = lambda v: float(torch.as_tensor(v).reshape(-1)[0])  # noqa: E731
        boundary = one(named.get("boundary", 0.0))
        tracking = one(named.get("boundary_tracking", 0.0))
        internal = float(ev.objective[0]) - boundary + tracking
        per_step = step_values(p.program, ev.raw, pb.dt, pb.E, exclude=("boundary", "boundary_tracking"),
                               coefs=self.coefs[name])
        value_h = per_step.reshape(self.T_m, p.ratio).sum(-1)
        flows = torch.zeros(len(p.crossings), self.T_m)
        for c, (gid, li) in zip(p.crossings, p.sinks):
            x = ev.raw[p.program.node_var(gid, "local:flow")]
            x = x.reshape(x.shape[0], -1, x.shape[-1])[0, li if x.shape[1] > 1 else 0]
            if x.shape[-1] == 1:
                x = x.expand(T_p)
            flows[c.k] = x.detach().reshape(self.T_m, p.ratio).mean(-1)
        st.x = xs
        return {"flows": flows, "value_h": value_h, "internal": internal, "objective": float(ev.objective[0]),
                "boundary": boundary, "x": st.x, "feasible": feasible,
                "max_violation": max_violation, "seconds": seconds, "eval": ev, "problem": pb,
                "value_check": float(value_h.sum()) - internal}

    def solve_plants(self, jobs: list) -> list:
        """Several (name, prices, track) solves: side by side in the worker processes if there are any. A plant
        solved twice in one call starts both times from its last schedule."""
        if self.cfg.workers <= 1 or len(jobs) <= 1:
            return [self.solve_plant(n, pr, tr) for n, pr, tr in jobs]
        pool = self._pool()
        warm = {n: self.state[n].x for n, _, _ in jobs}
        futs = [pool.submit(_worker_solve, n, pr, tr, warm[n]) for n, pr, tr in jobs]
        out = []
        for (n, pr, tr), f in zip(jobs, futs):
            r = f.result()
            pb = self._plant_problem(n, pr, tr)
            out.append(self._column(n, pb, r["x"], r["feasible"], r["max_violation"], r["seconds"]))
        return out

    def _pool(self):
        if getattr(self, "_executor", None) is None:
            import concurrent.futures as cf
            import multiprocessing as mp
            spec = {"hier": self.h.args, "config": self.cfg, "E": self.E, "t0": self.t0,
                    "threads": self.cfg.worker_threads}
            self._executor = cf.ProcessPoolExecutor(max_workers=self.cfg.workers, mp_context=mp.get_context("spawn"),
                                                    initializer=_worker_init, initargs=(spec,))
        return self._executor

    def close(self) -> None:
        if getattr(self, "_executor", None) is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    # ------------------------------------------------------------------ loop
    def _seed_weight(self, x: torch.Tensor, name: str, j: int) -> torch.Tensor:
        """The warm start with column j of a plant at `new_weight` in every block, the others scaled down."""
        p, sp = self.h.plants[name], self.pbm.space
        x = x.clone()
        a = self.cfg.new_weight
        for jj in range(len(self.state[name].columns)):
            gid, li = p.slots[jj]
            b = sp.block(gid, "w")
            idx = b.offset + li * b.H + torch.arange(b.H)
            x[0, idx] = a if jj == j else x[0, idx] * (1.0 - a)
        return x

    def _add_column(self, name: str, col: dict, w: Optional[torch.Tensor]) -> int:
        p, st = self.h.plants[name], self.state[name]
        if len(st.columns) >= self.h.k_max:
            # full: replace the column the master uses least (never the one it leans on most, never the artificial)
            use = w.sum(-1).clone()
            for jj, c in enumerate(st.columns):
                if c.get("artificial"):
                    use[jj] = float("inf")
            j = int(use.argmin())
            st.columns[j] = col
        else:
            st.columns.append(col)
            j = len(st.columns) - 1
        self._load_column(p, j, col)
        return j

    def _master(self, x_m, lam, rec: dict):
        pb = self.master_problem()
        t = time.perf_counter()
        res = Cascade(pb, config=self.cfg.master).solve(warm=x_m, lam0=lam)
        x = res.x[:1].detach().clone()
        # a clearly infeasible master's multipliers measure the infeasibility, not what a tonne is worth: price from
        # the objective alone then (a nearly feasible one keeps them - without them a binding market looks open)
        new = self.prices(pb, x, res.multipliers if res.max_violation <= self.cfg.price_viol else None)
        with torch.no_grad():
            ev = pb.evaluate(x, relax=False)
        worst = sorted(((float(v.max()), f"{pb.prog.gmap[g].ir.type.short}.{k}") for (g, k), v in ev.viol.items()),
                       reverse=True)[:4]
        rec["master"] = {"objective": res.objective, "feasible": bool(res.feasible), "max_violation": res.max_violation,
                         "seconds": time.perf_counter() - t,
                         "violations": {k: round(v, 4) for v, k in worst if v > 1e-3},
                         "weights": {n: w.mean(-1).tolist() for n, w in self.weights(x).items()},
                         "artificial": {n: float(w[j].mean()) for n, w in self.weights(x).items()
                                        for j, c in enumerate(self.state[n].columns) if c.get("artificial")}}
        return res, x, res.multipliers, new

    def _record(self, rec: dict, name: str, col: dict, prices: torch.Tensor, rc, t: float) -> None:
        rec["plants"].setdefault(name, []).append(
            {"internal": col["internal"], "boundary_at_prices": float((prices * col["flows"]).sum()), "reduced_cost": rc,
             "feasible": col["feasible"], "max_violation": col["max_violation"],
             "seconds": col.get("seconds", time.perf_counter() - t),
             "flows_per_step": col["flows"].sum(-1).tolist(), "price_mean": (prices.mean(-1) / self.dt_m).tolist()})

    def _value_h(self, name: str, ev) -> torch.Tensor:
        p = self.h.plants[name]
        pb = self.pbp[name]
        per = step_values(p.program, ev.raw, pb.dt, pb.E, exclude=("boundary", "boundary_tracking"), coefs=self.coefs[name])
        return per.reshape(self.T_m, p.ratio).sum(-1)

    @staticmethod
    def _value_at(col: dict, prices: torch.Tensor) -> torch.Tensor:
        """A column's value per master step at these prices: its internal value plus its boundary flows priced."""
        return col["value_h"] + (prices * col["flows"]).sum(0)

    def solve(self, callback: Optional[Callable] = None) -> CorporatePlan:
        t_start = time.perf_counter()
        cfg = self.cfg
        rounds = []
        pb = self.master_problem()
        base = self.prices(pb, pb.space.default_u(1), None)          # the group as it stands, no plant flow yet
        rec = {"round": 0, "plants": {}}
        jobs = [(name, base[name] * s, None) for name in self.h.plants for s in cfg.start_scales]
        t = time.perf_counter()
        start: dict = {}
        for (name, pr, _), col in zip(jobs, self.solve_plants(jobs)):
            self._record(rec, name, col, pr, None, t)
            start.setdefault(name, []).append(col)
        more = [(name, base[name] * s, None) for name, cols in start.items()
                if sum(c["max_violation"] <= cfg.col_tol for c in cols) < 2 for s in cfg.extra_scales]
        for (name, pr, _), col in zip(more, self.solve_plants(more)):
            self._record(rec, name, col, pr, None, t)
            start[name].append(col)
        for name, cols in start.items():
            ok = [c for c in cols if c["max_violation"] <= cfg.col_tol]
            for c in ok or [min(cols, key=lambda c: c["max_violation"])]:
                self._add_column(name, c, None)
            rec.setdefault("kept_out", {})[name] = len(cols) - len(ok or [None])
        for name in self.h.plants:
            self.state[name].prices = base[name]
        res_m, x_m, lam, new = self._master(None, None, rec)
        if cfg.phase1 and res_m.max_violation > cfg.price_viol:
            # no mix of the real schedules keeps the group's limits (every network buys scrap as if it were plentiful):
            # Phase I - an artificial schedule per block, no flow at the fence and worse every hour than any real one
            rec["phase1"] = {"violation": res_m.max_violation, "violations": rec["master"].get("violations")}
            for name in self.h.plants:
                vals = torch.stack([c["value_h"] for c in self.state[name].columns])
                art = vals.min(0).values - cfg.phase1_cost * vals.abs().mean().clamp(min=1.0)
                p = self.h.plants[name]
                j = self._add_column(name, {"flows": torch.zeros(len(p.crossings), self.T_m), "value_h": art,
                                            "internal": float(art.sum()), "artificial": True, "feasible": False,
                                            "max_violation": float("inf"), "x": None, "eval": None},
                                     torch.zeros(len(self.state[name].columns), self.T_m))
                x_m = self._seed_weight(x_m, name, j)
            res_m, x_m, lam, new = self._master(x_m, None, rec)
        prices = new
        rounds.append(rec)
        if callback:
            callback(rec)
        for r in range(1, cfg.rounds):
            rec = {"round": r, "plants": {}}
            added = False
            w_all = self.weights(x_m)
            t = time.perf_counter()
            cols = self.solve_plants([(name, prices[name], None) for name in self.h.plants])
            for name, col in zip(self.h.plants, cols):
                st = self.state[name]
                st.prices = prices[name]
                # reduced cost: the new schedule's value at these prices against the master's current mix
                w = w_all[name]
                val_mix = sum(float((w[j] * self._value_at(c, prices[name])).sum()) for j, c in enumerate(st.columns))
                rc = float(self._value_at(col, prices[name]).sum()) - val_mix
                self._record(rec, name, col, prices[name], rc, t)
                if rc <= cfg.tol * max(abs(res_m.objective), 1.0):
                    continue
                if col["max_violation"] > cfg.col_tol and any(c["max_violation"] <= cfg.col_tol for c in st.columns):
                    rec.setdefault("kept_out", {})[name] = 1           # breaks the block's own limits
                    continue
                j = self._add_column(name, col, w)
                x_m = self._seed_weight(x_m, name, j)
                added = True
            if not added:
                rec["converged"] = True
                rounds.append(rec)
                if callback:
                    callback(rec)
                break
            res_m, x_m, lam, new = self._master(x_m, lam, rec)
            # dual smoothing: the next prices lie between the last ones and the master's new ones
            prices = {n: cfg.smoothing * new[n] + (1.0 - cfg.smoothing) * prices[n] for n in prices}
            rounds.append(rec)
            if callback:
                callback(rec)
        mix_objective = res_m.objective
        weights = self.weights(x_m)
        plant_x, plant_eval, plant_flows, mix, target = {}, {}, {}, {}, {}
        for name, p in self.h.plants.items():
            st, w = self.state[name], weights[name]
            mix[name] = target[name] = sum(w[j] * c["flows"] for j, c in enumerate(st.columns))
            real = [jj for jj, c in enumerate(st.columns) if not c.get("artificial")]
            j = max(real, key=lambda jj: float(w[jj].sum()))
            chosen = st.columns[j]
            plant_x[name], plant_eval[name], plant_flows[name] = chosen["x"], chosen["eval"], chosen["flows"]
            st.x = chosen["x"]
            if cfg.commit and float(w[j].min()) < 0.98:
                plant_flows[name] = None            # a mix: one schedule is solved for it below
        obj, feas = mix_objective, bool(res_m.feasible)
        viol = {name: float(plant_eval[name].max_violation()[0]) for name in self.h.plants}
        if cfg.commit:
            # a mix of schedules is not a schedule: each plant solves one near the master's mix (a deviation of the
            # mix's own size costs about what that flow is worth at the current prices), the group is re-planned
            # around it; while the committed plan breaks a limit of the group, re-price and solve again
            for it in range(max(1, cfg.commit_iters)):
                jobs = []
                for name, p in self.h.plants.items():
                    st = self.state[name]
                    if plant_flows[name] is not None and it == 0:
                        continue
                    m = target[name]
                    scale = m.abs().mean(-1).clamp(min=1e-6)
                    track_w = st.prices.abs().mean(-1) / self.dt_m / scale
                    jobs.append((name, st.prices, (m, track_w)))
                t = time.perf_counter()
                rec = {"round": f"commit {it}", "plants": {}}
                done = dict(zip([j[0] for j in jobs], self.solve_plants(jobs)))
                # a schedule just outside the block's limits: once more from itself (the polish goes on) before the
                # fallback below swaps it for a pool schedule that may be far from the mix
                again = [j for j in jobs if done[j[0]]["max_violation"] > cfg.feas_tol]
                for (name, pr, tr), col in zip(again, self.solve_plants(again)):
                    if col["max_violation"] < done[name]["max_violation"]:
                        done[name] = col
                        rec.setdefault("resolved", []).append(name)
                for name, pr, _ in jobs:
                    chosen = done[name]
                    self._record(rec, name, chosen, pr, None, t)
                    if chosen["max_violation"] > cfg.feas_tol:
                        # the schedule near the mix breaks the block's own limits: commit the schedule of the pool
                        # that keeps them and that the mix leans on most (a real schedule, if not the mix's)
                        st, w = self.state[name], weights[name]
                        ok = [j for j, c in enumerate(st.columns) if c["max_violation"] <= cfg.feas_tol]
                        if ok:
                            j = max(ok, key=lambda j: float(w[j].sum()) if j < w.shape[0] else 0.0)
                            chosen = st.columns[j]
                            rec.setdefault("fallback", {})[name] = j
                    plant_x[name], plant_eval[name], plant_flows[name] = chosen["x"], chosen["eval"], chosen["flows"]
                    viol[name] = float(chosen["max_violation"])
                if jobs:
                    rounds.append(rec)
                for name, p in self.h.plants.items():
                    for jj in range(len(self.state[name].columns)):
                        self._unload(p, jj)
                    self._load_column(p, 0, {"flows": plant_flows[name], "value_h": self._value_h(name, plant_eval[name])},
                                      fixed=True)
                rec = {"round": f"commit {it}", "plants": {}}
                res_c, x_m, lam_c, new = self._master(x_m, lam, rec)
                rounds.append(rec)
                if callback:
                    callback(rec)
                obj, feas = res_c.objective, bool(res_c.feasible)
                if feas:
                    break
                # the group cannot take the committed flows: the new prices say where, the target moves with them
                for name in self.h.plants:
                    st = self.state[name]
                    st.prices = cfg.smoothing * new[name] + (1.0 - cfg.smoothing) * st.prices
                    target[name] = plant_flows[name]
        self.close()
        plants_ok = all(v <= cfg.feas_tol for v in viol.values())
        return CorporatePlan(x_m, obj, feas and plants_ok, mix_objective, weights, plant_x, plant_eval, plant_flows, mix,
                             rounds, time.perf_counter() - t_start, viol, feas)


# ----------------------------------------------------------------------------- worker processes
_W: dict = {}


def _worker_init(spec: dict) -> None:
    torch.set_num_threads(max(1, int(spec["threads"])))
    from ..lang.hierarchy import compile_hierarchy
    h = compile_hierarchy(**spec["hier"])
    cfg = spec["config"]
    _W["cp"] = CorporatePlanner(h, CorporateConfig(**{**cfg.__dict__, "workers": 0}), E=spec["E"], t0=spec["t0"])


def _worker_solve(name: str, prices, track, warm) -> dict:
    cp = _W["cp"]
    cp.state[name].x = warm
    col = cp.solve_plant(name, prices, track)
    return {k: col[k] for k in ("x", "feasible", "max_violation", "seconds")}


__all__ = ["CorporatePlanner", "CorporateConfig", "CorporatePlan", "objective_coefficients", "step_values"]
