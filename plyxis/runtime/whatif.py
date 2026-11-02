"""What-if runs: what would an outage, a capacity change or a price shock cost, and what would the plan do about it?

Three values on the live twin, from the current state and forecast, nothing is executed:

    nominal      a new plan with the same budget, conditions as forecast
    no reaction  the current plan under the changed conditions
    re-planned   a new plan under the changed conditions

`re-planned - no reaction` is what reacting is worth; `nominal - re-planned` is what the event costs anyway. Both
new plans get the same budget, so the difference is the event and not a better search.
"""
from __future__ import annotations

import dataclasses
import time

import torch

from ..optim.cascade import Cascade, CascadeConfig

QUICK = CascadeConfig(relax_iters=60, multistart=2, population=4, repair_iters=30, polish_iters=40, neural_iters=1,
                      warm_relax=1.0)


def _apply(pb, E: dict, changes: list) -> list:
    """Changes on a problem context. Returns what was applied, in words."""
    done, sched = [], {}
    for c in changes:
        kind = c.get("kind")
        if kind == "price":
            name, f = c["name"], float(c.get("factor", 1.0))
            if name not in E:
                raise ValueError(f"unknown price {name}")
            start = int(float(c.get("start_h", 0.0)) / pb.dt)
            stop = pb.T if c.get("hours") is None else min(pb.T, start + max(1, int(float(c["hours"]) / pb.dt)))
            v = torch.as_tensor(E[name]).clone().reshape(-1)
            v = v.expand(pb.T).clone() if v.numel() == 1 else v
            v[start:stop] = v[start:stop] * f
            E[name] = v.reshape(torch.as_tensor(E[name]).shape) if torch.as_tensor(E[name]).numel() == v.numel() else v
            done.append(f"{name} x {f:g} for {(stop - start) * pb.dt:g} h")
            continue
        path = c.get("asset")
        iid = pb.prog.flat.by_path.get(path)
        if iid is None:
            raise ValueError(f"unknown asset {path}")
        gid, li = pb.prog.inst_group[iid]
        if kind == "outage":
            hours = float(c.get("hours", pb.T * pb.dt))
            states = pb.S0.get(gid, {})
            if "up" in states:
                back, name = float(states["up"].reshape(-1)[li]), "up"
            elif "avail" in states:
                back, name = float(states["avail"].reshape(-1)[li]), "avail"
            else:
                raise ValueError(f"{path} has no availability state")
            states[name].reshape(-1)[li] = 0.0
            sched[path] = (name, back, hours)
            done.append(f"{path} out of service for {hours:g} h")
        elif kind == "capacity":
            p, f = c["param"], float(c.get("factor", 1.0))
            if p not in pb.P.get(gid, {}):
                raise ValueError(f"{path} has no property {p}")
            pb.P[gid][p].reshape(-1)[li] *= f
            done.append(f"{path}.{p} x {f:g}")
        else:
            raise ValueError(f"unknown change {kind}")
    if sched:
        pb.set_state_schedule(sched)
    return done


def _terms(pb, x) -> tuple:
    with torch.no_grad():
        e = pb.evaluate(x, relax=False)
    named = {}
    for tr in pb.prog.terms:
        v = e.totals["named"].get(tr.name)
        if v is not None and tr.kind in ("revenue", "cost"):
            s = 1.0 if tr.kind == "revenue" else -1.0
            named[tr.name] = named.get(tr.name, 0.0) + s * float(torch.as_tensor(v).reshape(-1)[0])
    return float(e.objective[0]), float(e.max_violation()[0]), named


def run(engine, changes: list, name: str = "what-if", config: CascadeConfig = QUICK) -> dict:
    """One what-if on the engine's live twin. Waits for a running plan; the plant clock keeps its own pace."""
    planner = engine.planner
    t0 = time.perf_counter()
    if planner.plan is None:
        return {"ok": False, "error": "no plan yet"}
    prog = engine.twin.model.program
    with planner.lock:
        t = engine.t
        E0 = engine._forecast(prog.T, prog.dt)
        base = planner.problem(t, E0)
        outages = {p: o for p, o in engine.outages.items() if o[2] > t}
        if outages:
            base.set_state_schedule({p: (s, v, until - t) for p, (s, v, until) in outages.items()})
        plan = planner.plan
        x = plan.x
        if getattr(planner, "_plan_space", None) is not None and planner._plan_space is not base.space:
            from .planner import remap_x
            x = remap_x(planner._plan_space, base.space, x)
        x = base.space.shift(x, t - plan.t0)
        cfg = dataclasses.replace(config, exact_branches=(), discrete="ga")
        nom = Cascade(base, net=planner.net, partition=planner.part, config=cfg).solve(warm=x)
        nominal, _, terms0 = _terms(base, nom.x)
        E1 = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in E0.items()}
        case = planner.problem(t, E1)
        if outages:
            case.set_state_schedule({p: (s, v, until - t) for p, (s, v, until) in outages.items()})
        try:
            applied = _apply(case, E1, changes)
        except (KeyError, ValueError) as ex:
            return {"ok": False, "error": str(ex)}
        case.E = E1
        stay, stay_v, _ = _terms(case, x)
        res = Cascade(case, net=planner.net, partition=planner.part, config=cfg).solve(warm=x)
        replanned, viol, terms1 = _terms(case, res.x)
    moved = _changes(case, nom.x, res.x)
    bridge = sorted(({"term": k, "delta": round(terms1.get(k, 0.0) - terms0.get(k, 0.0))} for k in set(terms0) | set(terms1)),
                    key=lambda r: -abs(r["delta"]))
    return {"ok": True, "name": name, "t": t, "clock": engine.clock_str(t), "changes": applied,
            "horizon_h": prog.T * prog.dt, "nominal": round(nominal), "no_reaction": round(stay),
            "replanned": round(replanned), "cost_of_event": round(nominal - replanned),
            "value_of_reacting": round(replanned - stay), "no_reaction_violation": round(stay_v, 4),
            "replanned_violation": round(viol, 4), "feasible": res.feasible, "bridge": bridge[:10],
            "decisions": moved, "seconds": round(time.perf_counter() - t0, 1)}


def _changes(pb, x0, x1, top: int = 12) -> list:
    """Assets whose plan moved most (mean over the horizon, share of the decision's range)."""
    a, b = pb.space.split(x0), pb.space.split(x1)
    out = []
    for blk in pb.space.blocks:
        g = pb.prog.gmap[blk.gid]
        d = (b[(blk.gid, blk.name)] - a[(blk.gid, blk.name)])[0].mean(-1)
        for li in torch.nonzero(d.abs() > 0.05).reshape(-1).tolist():
            inst = pb.prog.flat.instances[g.inst_ids[li]]
            out.append({"asset": inst.path, "label": inst.label or inst.path, "decision": blk.name,
                        "change_share": round(float(d[li]), 3)})
    out.sort(key=lambda r: -abs(r["change_share"]))
    return out[:top]
