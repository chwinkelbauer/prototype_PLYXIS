"""Maintenance dispatch: value every repair with the twin, then schedule rigs and crews.

Value of a job = exact counterfactual: simulate the current plan with the asset
restored and compare. This automatically accounts for bottlenecks - restoring a
gassy well is worth little while the compressor station is the constraint.

Scheduling (parallel machines, travel/setup times, weighted completion) is
NP-hard; it is low-dimensional, so random-key differential evolution solves it
in a fraction of a second.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

from ..lang.consteval import haversine_km
from .problem import Problem


@dataclass
class Job:
    jid: str
    asset_path: str
    label: str
    kind: str                 # repair | restart | preventive
    resource_type: str        # workover_rig | field_crew
    hours: float
    at: Optional[tuple]
    value_per_h: float = 0.0
    reason: str = ""
    restore: list = field(default_factory=list)   # (gid, state, li, value)


def value_restorations(problem: Problem, x: torch.Tensor, jobs: list) -> None:
    """Fill job.value_per_h by counterfactual simulation (one batched run)."""
    if not jobs:
        return
    prog, sp = problem.prog, problem.space
    B = 1 + len(jobs)
    S0 = {}
    for gid, d in problem.S0.items():
        S0[gid] = {}
        for k, v in d.items():
            S0[gid][k] = v.expand(B, v.shape[1]).clone() if v.shape[0] == 1 else v[:1].expand(B, v.shape[1]).clone()
    X = x.detach().expand(B, -1).clone()
    xd = sp.default_u(1)
    for j, job in enumerate(jobs, start=1):
        for gid, state, li, val in job.restore:
            S0[gid][state][j, li] = val
            # give the restored asset its default operating decisions
            for b in sp.blocks:
                if b.gid == gid:
                    sl = slice(b.offset + li * b.H, b.offset + (li + 1) * b.H)
                    X[j, sl] = xd[0, sl]
    with torch.no_grad():
        e = problem.evaluate(X, relax=False, S0=S0)
    base = float(e.objective[0])
    horizon = problem.T * problem.dt
    for j, job in enumerate(jobs, start=1):
        job.value_per_h = max(0.0, (float(e.objective[j]) - base) / horizon)


def schedule(jobs: list, resources: list, now: float = 0.0, iters: int = 120, pop: int = 30,
             seed: int = 0) -> list:
    """resources: dicts {name, label, type, at:(lat,lon), speed_kmh, free_at}.
    Minimise sum(value * completion) + travel. Returns the scheduled job list."""
    if not jobs or not resources:
        return []
    g = torch.Generator().manual_seed(seed)
    J = len(jobs)

    def decode(keys: list) -> tuple:
        order = sorted(range(J), key=lambda i: keys[i])
        state = {r["name"]: (r["free_at"], r["at"]) for r in resources}
        plan, cost = [], 0.0
        for i in order:
            job = jobs[i]
            cands = [r for r in resources if r["type"] == job.resource_type] or resources
            best = None
            for r in cands:
                free, loc = state[r["name"]]
                d = haversine_km(loc[0], loc[1], job.at[0], job.at[1]) * 1.3 if (loc and job.at) else 5.0
                travel = d / max(r["speed_kmh"], 1e-6)
                start = max(now, free) + travel
                end = start + job.hours
                if best is None or end < best[0]:
                    best = (end, start, travel, r)
            end, start, travel, r = best
            state[r["name"]] = (end, job.at or state[r["name"]][1])
            cost += job.value_per_h * (end - now) + 50.0 * travel
            plan.append((job, r, start, end, travel))
        return cost, plan

    keys = torch.rand(pop, J, generator=g)
    cost = torch.tensor([decode(k.tolist())[0] for k in keys])
    # seed one individual with value-density order (classic WSPT heuristic)
    wspt = torch.tensor([-(jobs[i].value_per_h / max(jobs[i].hours, 0.1)) for i in range(J)])
    keys[0] = (wspt - wspt.min()) / (wspt.max() - wspt.min() + 1e-9)
    cost[0] = decode(keys[0].tolist())[0]
    for _ in range(iters):
        r = torch.stack([torch.randperm(pop, generator=g)[:3] for _ in range(pop)])
        mut = (keys[r[:, 0]] + 0.6 * (keys[r[:, 1]] - keys[r[:, 2]])).clamp(0, 1)
        cross = torch.rand(pop, J, generator=g) < 0.8
        trial = torch.where(cross, mut, keys)
        tc = torch.tensor([decode(k.tolist())[0] for k in trial])
        better = tc < cost
        keys = torch.where(better.view(-1, 1), trial, keys)
        cost = torch.where(better, tc, cost)
    best = int(cost.argmin())
    _, plan = decode(keys[best].tolist())
    out = []
    for rank, (job, r, start, end, travel) in enumerate(sorted(plan, key=lambda p: p[2])):
        out.append({"id": job.jid, "asset": job.asset_path, "label": job.label, "kind": job.kind,
                    "resource": r["name"], "resource_label": r["label"], "start": round(start, 2),
                    "end": round(end, 2), "travel_h": round(travel, 2), "hours": job.hours,
                    "value_per_h": round(job.value_per_h, 1), "reason": job.reason,
                    "at": list(job.at) if job.at else None})
    return out


def find_jobs(problem: Problem, x: torch.Tensor, health_threshold: float = 0.2,
              predicted: Optional[dict] = None, horizon_h: float = 24.0) -> list:
    """Derive maintenance jobs from the current twin state (failures, trips, degraded pumps).

    A pump is replaced preventively when its condition trend predicts the alarm level within
    `horizon_h` (and its estimated health is below 60 %), or when its estimated health is very low.
    A pump that is worn but stable keeps running: replacing it early would waste remaining life."""
    predicted = predicted or {}
    prog = problem.prog
    jobs: list = []
    rig_hours = _param_of_type(prog, "job_hours", "workover_rig", 30.0)
    crew_hours = _param_of_type(prog, "job_hours", "field_crew", 3.0)
    for g in prog.groups:
        if g.is_env:
            continue
        S = problem.S0[g.gid]
        tname = g.ir.type.short
        for li, iid in enumerate(g.inst_ids):
            inst = prog.flat.instances[iid]
            lab = inst.label or inst.path
            if "up" in S and float(S["up"].reshape(-1)[li]) < 0.5 and "q_max" in g.ir.params:
                jobs.append(Job(f"J-{inst.path}-repair", inst.path, lab, "repair", "workover_rig", rig_hours,
                                inst.at, reason="ESP failed - replace pump", restore=[(g.gid, "up", li, 1.0)]))
            elif "up" in S and float(S["up"].reshape(-1)[li]) < 0.5:
                jobs.append(Job(f"J-{inst.path}-restart", inst.path, lab, "restart", "field_crew", crew_hours,
                                inst.at, reason=f"{tname.replace('_', ' ')} down - inspect and restart",
                                restore=[(g.gid, "up", li, 1.0)]))
            if "avail" in S and "n_units" in g.ir.params:
                n = float(prog.P0[g.gid]["n_units"].reshape(-1)[li])
                a = float(S["avail"].reshape(-1)[li])
                if a < n - 0.5:
                    jobs.append(Job(f"J-{inst.path}-unit", inst.path, lab, "restart", "field_crew", crew_hours,
                                    inst.at, reason=f"{int(n - a)} compressor unit(s) tripped - restart",
                                    restore=[(g.gid, "avail", li, n)]))
            if "health" in S and "up" in S and float(S["up"].reshape(-1)[li]) >= 0.5:
                h = float(S["health"].reshape(-1)[li])
                eta = predicted.get(inst.path)
                if h < health_threshold or (eta is not None and eta < horizon_h and h < 0.6):
                    why = (f"vibration trend: alarm expected in ~{eta:.0f} h, pump health {h:.0%}" if eta is not None
                           else f"pump health {h:.0%}")
                    jobs.append(Job(f"J-{inst.path}-prev", inst.path, lab, "preventive", "workover_rig", rig_hours,
                                    inst.at, reason=f"{why} - planned replacement before failure",
                                    restore=[(g.gid, "health", li, 1.0)]))
    value_restorations(problem, x, jobs)
    for j in jobs:
        if j.kind == "preventive":
            if j.value_per_h >= 1.0 and "trend" not in j.reason:
                j.reason += f" (avoids ~${j.value_per_h:,.0f}/h expected failure cost)"
        elif j.value_per_h >= 1.0:
            j.reason += f" - restoring is worth ${j.value_per_h:,.0f}/h at current bottlenecks"
        else:
            j.reason += " - restores full capacity (not a bottleneck right now)"
    return jobs


def resources_of(problem: Problem, busy: Optional[dict] = None) -> list:
    prog = problem.prog
    out = []
    busy = busy or {}
    for g in prog.groups:
        if g.is_env or "job_hours" not in g.ir.params:
            continue
        for li, iid in enumerate(g.inst_ids):
            inst = prog.flat.instances[iid]
            sp_kmh = float(prog.P0[g.gid]["move_speed"].reshape(-1)[li]) / 1000.0 \
                if "move_speed" in prog.P0[g.gid] else 30.0
            out.append({"name": inst.path, "label": inst.label or inst.path, "type": g.ir.type.short,
                        "at": inst.at, "speed_kmh": sp_kmh, "free_at": busy.get(inst.path, 0.0)})
    return out


def _param_of_type(prog, pname: str, tname: str, default: float) -> float:
    for g in prog.groups:
        if not g.is_env and g.ir.type.short == tname and pname in prog.P0[g.gid]:
            return float(prog.P0[g.gid][pname].reshape(-1)[0])
    return default
