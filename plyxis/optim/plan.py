"""Plan: the product of one optimisation run, packaged for operations and the GUI."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from ..lang import units as U
from . import dispatch as DSP
from .explain import bottlenecks
from .problem import Problem


@dataclass
class Plan:
    pid: int
    t0: float                       # sim time (h) the plan starts at
    created: float                  # wall-clock
    trigger: str
    x: torch.Tensor
    objective: float
    feasible: bool
    max_violation: float
    totals: dict
    series: dict
    instructions: list
    lineups: list
    trips: dict
    bottlenecks: list
    multipliers: dict
    solver: dict
    jobs: list = field(default_factory=list)
    raw: Optional[dict] = None

    def summary(self) -> dict:
        return {"id": self.pid, "t0": self.t0, "created": self.created, "trigger": self.trigger,
                "objective": self.objective, "feasible": self.feasible, "max_violation": self.max_violation,
                "totals": self.totals, "series": self.series, "bottlenecks": self.bottlenecks[:8],
                "solver": {k: v for k, v in self.solver.items() if k != "history"},
                "n_instructions": len(self.instructions), "n_trips": len(self.trips.get("trips", [])),
                "jobs": self.jobs}


def per_step(problem: Problem, raw: dict) -> dict:
    """Field-level time series (display units)."""
    prog = problem.prog
    T = problem.T
    out: dict = {"hours": [problem.t0 + t * problem.dt for t in range(T)]}
    kind_tot = {"revenue": torch.zeros(T), "cost": torch.zeros(T), "emission": torch.zeros(T)}
    metrics: dict = {}
    for tr in prog.terms:
        if tr.final:
            continue
        x = raw[prog.node_var(tr.gid, tr.key)]
        x = x if x.dim() == 3 else x.view(1, 1, -1)
        v = x[0].sum(0)
        v = v.expand(T) if v.shape[0] == 1 else v
        if tr.kind in kind_tot:
            kind_tot[tr.kind] = kind_tot[tr.kind] + v
        else:
            m = metrics.setdefault(tr.name, {"unit": tr.unit, "v": torch.zeros(T)})
            m["v"] = m["v"] + v
    out["revenue_usd_h"] = [round(float(v), 1) for v in kind_tot["revenue"]]
    out["cost_usd_h"] = [round(float(v), 1) for v in kind_tot["cost"]]
    out["emission_t_h"] = [round(float(v) / 1000.0, 3) for v in kind_tot["emission"]]
    for name, m in metrics.items():
        unit = m["unit"]
        vals = m["v"]
        if unit:
            try:
                vals = vals / U.parse_unit(unit)[0]
            except Exception:  # noqa: BLE001
                pass
        out[name] = {"unit": unit or "", "values": [round(float(v), 2) for v in vals]}
    return out


def build_plan(problem: Problem, res, pid: int, trigger: str, t0_wall: Optional[float] = None,
               jobs: Optional[list] = None) -> Plan:
    with torch.no_grad():
        e = problem.evaluate(res.x, relax=False)
    tot = e.totals
    hours = problem.T * problem.dt
    totals = {
        "horizon_h": hours,
        "objective_usd": float(res.objective),
        "revenue_usd": float(torch.as_tensor(tot["revenue"]).reshape(-1)[0]),
        "cost_usd": float(torch.as_tensor(tot["cost"]).reshape(-1)[0]),
        "emission_t": float(torch.as_tensor(tot["emission"]).reshape(-1)[0]) / 1000.0,
        "named": {},
    }
    for tr in problem.prog.terms:
        v = tot["named"].get(tr.name)
        if v is None:
            continue
        v = float(torch.as_tensor(v).reshape(-1)[0])
        if tr.kind == "metric" and tr.unit:
            try:
                # metrics are integrated over the horizon -> express as average rate
                v = U.from_canonical(v / hours, tr.unit)
            except Exception:  # noqa: BLE001
                pass
        totals["named"][tr.name] = {"kind": tr.kind, "value": round(v, 2), "unit": tr.unit or ("USD" if tr.kind in
                                    ("revenue", "cost") else ("kgCO2e" if tr.kind == "emission" else ""))}
    series = per_step(problem, e.raw)
    instr = DSP.instructions(problem, res.x, problem.t0)
    lups = DSP.lineups(problem, res.x, problem.t0)
    trips = DSP.truck_trips(problem, e.raw, problem.t0)
    bn = bottlenecks(problem, res.multipliers, res.x)
    solver = {"seconds": res.seconds, "evals": res.evals, "stages": res.stages,
              "variables": problem.space.N, "discrete": int(problem.space.discrete_mask().sum()),
              "history": res.history}
    return Plan(pid, problem.t0, t0_wall or time.time(), trigger, res.x.detach().clone(), float(res.objective),
                bool(res.feasible), float(res.max_violation), totals, series, instr, lups, trips, bn,
                res.multipliers, solver, jobs or [], e.raw)
