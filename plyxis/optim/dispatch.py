"""Dispatch: turn the optimised plan into executable work.

* operating instructions: set-point intervals per asset (merged consecutive blocks)
* line-ups: destination of every routed out-port per block
* truck trips: the fluid lane plan becomes individual timed trips assigned to trucks
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from ..lang import units as U
from .problem import Problem


def _inst(prog, iid):
    return prog.flat.instances[iid]


def instructions(problem: Problem, x: torch.Tensor, t0: float = 0.0) -> list:
    """Set-point intervals for every decision of every asset."""
    prog, sp = problem.prog, problem.space
    bnd = sp.bounds()
    parts = sp.split(x.detach())
    out = []
    for b in sp.blocks:
        g = prog.gmap[b.gid]
        if g.is_env:
            continue
        lo, hi, _ = bnd[(b.gid, b.name)]
        u = parts[(b.gid, b.name)][0]                 # [n, H]
        vals = lo[0, :, 0:1] + (hi - lo)[0, :, 0:1] * u
        if b.domain in ("int", "bool"):
            vals = torch.round(vals)
        info = g.ir.decisions[b.name]
        for li, iid in enumerate(g.inst_ids):
            inst = _inst(prog, iid)
            row = vals[li].tolist()
            start = 0
            for h in range(1, b.H + 1):
                if h == b.H or abs(row[h] - row[start]) > 1e-3 * max(1.0, abs(row[start])):
                    v = row[start]
                    out.append({"asset": inst.path, "label": inst.label or inst.path, "type": inst.type.short,
                                "decision": b.name, "start": t0 + start * b.hold * problem.dt,
                                "end": t0 + min(h * b.hold, problem.T) * problem.dt,
                                "value": v, "display": _fmt(v, info.unit, float(lo.reshape(-1)[li if lo.numel() > 1 else 0]),
                                                           float(hi.reshape(-1)[li if hi.numel() > 1 else 0]), b.domain)})
                    start = h
    return out


def lineups(problem: Problem, x: torch.Tensor, t0: float = 0.0) -> list:
    prog, sp = problem.prog, problem.space
    if sp.route is None:
        return []
    r = sp.split(x.detach())["route"]
    s = sp.normalise_routes(r.index_select(2, sp.expand_index(sp.route.hold, sp.route.H)), relax=True)[0]
    out = []
    for rp in prog.routed:
        src = _inst(prog, rp.inst)
        dests = []
        for eid in rp.edges:
            e = prog.flat.edges[eid]
            dests.append(_dest_label(prog, e))
        for h in range(sp.route.H):
            tt = min(h * sp.route.hold, problem.T - 1)
            shares = [float(s[ri, tt]) for ri in rp.ridx]
            k = max(range(len(shares)), key=lambda i: shares[i])
            out.append({"asset": src.path, "label": src.label or src.path, "port": rp.name,
                        "start": t0 + h * sp.route.hold * problem.dt,
                        "end": t0 + min((h + 1) * sp.route.hold, problem.T) * problem.dt,
                        "exclusive": rp.exclusive, "to": dests[k],
                        "shares": {dests[i]: round(shares[i], 3) for i in range(len(shares))}})
    return out


def _dest_label(prog, e) -> str:
    """Follow link instances (flowlines, lanes) to the real destination asset."""
    dport = prog.flat.ports[e.dst]
    inst = _inst(prog, dport.inst)
    if inst.is_link:
        ins = [n for n, p in inst.type.ports.items() if p.direction == "in"]
        if ins and dport.name != ins[0]:
            return inst.label or inst.path          # e.g. truck allocation to a lane's fleet port
    seen = 0
    while (inst.is_link or inst.is_junction) and seen < 5:
        outs = [prog.flat.ports[pid] for pid in inst.ports.values() if prog.flat.ports[pid].direction == "out"]
        if not outs or not outs[0].edges:
            break
        inst = _inst(prog, prog.flat.ports[prog.flat.edges[outs[0].edges[0]].dst].inst)
        seen += 1
    return inst.label or inst.path


def truck_trips(problem: Problem, raw: dict, t0: float = 0.0, max_trips: int = 600) -> dict:
    """Discrete trips from the fluid lane plan; trips are assigned to the earliest free truck."""
    prog = problem.prog
    lanes = []
    for g in prog.groups:
        if g.is_env:
            continue
        ir = g.ir
        if not {"load", "distance"} <= set(ir.params) or "local:cycle" not in ir.nodes:
            continue
        cargo = next((p for p in ir.ports.values() if p.direction == "in" and p.name != "fleet"), None)
        if cargo is None:
            continue
        for li, iid in enumerate(g.inst_ids):
            inst = _inst(prog, iid)
            flow = 0.0
            res_name = None
            for (r, a) in cargo.channels:
                if a is not None:
                    continue
                key = f"in:{cargo.name}:{r.split('.')[-1]}"
                v = raw[prog.node_var(g.gid, key)]
                v = v[0, li] if v.dim() == 3 else v.reshape(-1)[li:li + 1]
                if float(v.abs().sum()) > 0:
                    res_name = r.split(".")[-1]
                flow = flow + v
            P = prog.P0[g.gid]
            par = {k: float(P[k].reshape(-1)[li]) for k in ("load", "distance", "speed", "load_time") if k in P}
            lanes.append({"inst": inst, "flow": flow if isinstance(flow, torch.Tensor) else torch.zeros(problem.T),
                          "par": par, "res": res_name or "cargo"})
    fleets = [g for g in prog.groups if not g.is_env and "n_trucks" in g.ir.params]
    n_trucks = int(sum(float(prog.P0[g.gid]["n_trucks"].sum()) for g in fleets)) or 1
    free = [0.0] * n_trucks
    trips = []
    dt = problem.dt
    events = []
    for lane in lanes:
        par = lane["par"]
        load = par.get("load", 1.0)
        acc = 0.0
        for t in range(problem.T):
            vol = float(lane["flow"][t]) * dt
            acc += vol
            while acc >= load * 0.999 and len(trips) + len(events) < max_trips:
                acc -= load
                frac = 1.0 - (acc / vol if vol > 1e-9 else 0.0)
                ready = (t + max(0.0, min(1.0, frac))) * dt
                events.append((ready, lane))
    events.sort(key=lambda z: z[0])
    for ready, lane in events:
        par = lane["par"]
        k = min(range(n_trucks), key=lambda i: free[i])
        start = max(ready, free[k])
        speed = max(par.get("speed", 1.0), 1e-6)
        travel = par.get("distance", 0.0) / speed
        lt = par.get("load_time", 0.0)
        arrive = start + lt + travel
        back = arrive + travel
        free[k] = back
        inst = lane["inst"]
        trips.append({"truck": f"T{k + 1:02d}", "lane": inst.path, "label": inst.label or inst.path,
                      "cargo": lane["res"], "volume": U.from_canonical(par.get("load", 0.0), "bbl"),
                      "ready": t0 + ready, "depart": t0 + start, "loaded": t0 + start + lt,
                      "arrive": t0 + arrive, "return": t0 + back, "delay": round(start - ready, 2),
                      "path": [list(p) for p in (inst.line or [])]})
    return {"trips": trips, "trucks": n_trucks}


def truck_positions(trips: list, now: float, n_trucks: int, yard: Optional[tuple] = None) -> list:
    """Interpolated truck positions at sim time `now` (hours) for the live map."""
    pos = {}
    for tr in trips:
        path = tr.get("path") or []
        if len(path) < 2:
            continue
        a, b = path[0], path[-1]
        state, p = None, None
        if tr["depart"] <= now < tr["loaded"]:
            state, p = "loading", a
        elif tr["loaded"] <= now < tr["arrive"]:
            f = (now - tr["loaded"]) / max(tr["arrive"] - tr["loaded"], 1e-6)
            state, p = "loaded", _lerp(a, b, f)
        elif tr["arrive"] <= now < tr["return"]:
            f = (now - tr["arrive"]) / max(tr["return"] - tr["arrive"], 1e-6)
            state, p = "returning", _lerp(b, a, f)
        if state:
            pos[tr["truck"]] = {"truck": tr["truck"], "lat": p[0], "lon": p[1], "state": state, "lane": tr["label"],
                                "cargo": tr["cargo"]}
    out = list(pos.values())
    if yard:
        for k in range(n_trucks):
            name = f"T{k + 1:02d}"
            if name not in pos:
                out.append({"truck": name, "lat": yard[0] + 0.0004 * (k % 4), "lon": yard[1] + 0.0006 * (k // 4),
                            "state": "idle", "lane": None, "cargo": None})
    return out


def _lerp(a, b, f):
    f = max(0.0, min(1.0, f))
    return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)


def _fmt(v: float, unit: Optional[str], lo: float, hi: float, domain: str) -> str:
    if domain == "bool":
        return "on" if v >= 0.5 else "off"
    if domain == "int":
        return str(int(round(v)))
    if unit:
        try:
            return f"{U.from_canonical(v, unit):,.0f} {unit}"
        except Exception:  # noqa: BLE001
            pass
    if 0.0 <= lo and hi <= 1.0 + 1e-9:
        return f"{v * 100:.0f} %"
    return f"{v:,.2f}"
