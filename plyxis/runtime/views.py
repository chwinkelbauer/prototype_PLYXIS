"""Read models for the GUI: twin structure (map), schedule (Gantt), asset details, library."""
from __future__ import annotations

from typing import Optional

import torch

from ..lang import ast as A
from ..lang import units as U
from ..lang.consteval import const_eval
from ..optim.partition import build_partition


def _geom(t) -> dict:
    out = {}
    for k, v in t.geom.items():
        try:
            out[k] = const_eval(v, {})
        except Exception:  # noqa: BLE001
            pass
    return out


def _disp(v: float, unit: Optional[str]):
    if unit:
        try:
            return U.from_canonical(v, unit)
        except Exception:  # noqa: BLE001
            return v
    return v


def structure(engine) -> dict:
    model = engine.twin.model
    flat, prog = model.flat, model.program
    live = engine.twin.live
    try:
        part = build_partition(prog)
        sub = part.of_instance
        parts = part.describe(flat)
    except Exception:  # noqa: BLE001
        sub, parts = {}, []
    assets = []
    for inst in flat.instances:
        if inst.is_junction:
            continue
        g = _geom(inst.type)
        gid, li = prog.inst_group[inst.id]
        params = {}
        for p, decl in inst.type.params.items():
            if p in inst.params:
                base = inst.params[p]
                cur = float(live.P[gid][p].reshape(-1)[li]) if p in live.P.get(gid, {}) else base
                params[p] = {"value": round(_disp(cur, decl.unit), 4), "engineering": round(_disp(base, decl.unit), 4),
                             "unit": decl.unit or "", "calibrated": decl.calibrate is not None, "doc": decl.doc}
        assets.append({
            "path": inst.path, "label": inst.label or inst.path, "type": inst.type.short,
            "kind": "line" if inst.line else "point", "layer": inst.meta.get("layer") or g.get("layer", "other"),
            "icon": inst.meta.get("icon") or g.get("icon", ""), "at": list(inst.at) if inst.at else None,
            "line": [list(p) for p in inst.line] if inst.line else None, "tag": inst.tag, "parent": inst.parent,
            "is_link": inst.is_link, "params": params, "subsystem": sub.get(inst.id),
            "resources": sorted({r.split(".")[-1] for pid in inst.ports.values()
                                 for r in flat.ports[pid].resources if flat.ports[pid].direction == "in"}),
            "doc": inst.type.doc,
        })
    edges = []
    for e in flat.edges:
        a = flat.instances[flat.ports[e.src].inst]
        b = flat.instances[flat.ports[e.dst].inst]
        if a.is_link or b.is_link or a.is_junction or b.is_junction:
            continue
        if a.at and b.at:
            edges.append({"src": a.path, "dst": b.path, "resources": [r.split(".")[-1] for r in e.resources],
                          "line": [list(a.at), list(b.at)]})
    layers = sorted({a["layer"] for a in assets})
    return {"assets": assets, "edges": edges, "layers": layers, "origin": list(flat.origin) if flat.origin else None,
            "env": flat.env_type.name, "version": engine.twin.version, "subsystems": parts,
            "summary": prog.summary(), "library": library(model)}


def library(model) -> list:
    out = []
    seen = set()
    for inst in model.flat.instances:
        t = inst.type
        if t.name in seen or inst.is_junction:
            continue
        seen.add(t.name)
        out.append({
            "name": t.short, "qualified": t.name, "doc": t.doc, "base": t.base, "package": t.package,
            "params": [{"name": p.name, "unit": p.unit, "calibrate": p.calibrate is not None, "doc": p.doc}
                       for p in t.params.values()],
            "states": [{"name": s.name, "unit": s.unit} for s in t.states.values()],
            "decisions": [{"name": d.name, "domain": d.domain, "unit": d.unit} for d in t.decisions.values()],
            "ports": [{"name": p.name, "direction": p.direction, "resources": p.resources, "exclusive": p.exclusive}
                      for p in t.ports.values()],
            "telemetry": [{"signal": x.signal, "tag": x.tag, "unit": x.unit} for x in t.telemetry],
            "count": sum(1 for i in model.flat.instances if i.type.name == t.name),
        })
    out.sort(key=lambda r: -r["count"])
    return out


def gantt(engine) -> dict:
    plan = engine.planner.plan
    if plan is None:
        return {"rows": [], "t0": engine.t, "t1": engine.t + 72}
    pb = plan.problem
    rows = []
    t0, t1 = plan.t0, plan.t0 + pb.T * pb.dt
    by_asset: dict = {}
    for ins in plan.instructions:
        by_asset.setdefault((ins["type"], ins["asset"], ins["label"]), []).append(ins)
    order = {"compressor_station": 0, "battery": 1, "pad": 2, "esp_well": 3, "truck_lane": 4}

    def seg_list(items, decision):
        return [{"start": s["start"], "end": s["end"], "value": s["value"], "label": s["display"]}
                for s in items if s["decision"] == decision]
    for (typ, path, label), items in sorted(by_asset.items(), key=lambda kv: (order.get(kv[0][0], 9), kv[0][1])):
        decisions = sorted({s["decision"] for s in items})
        for d in decisions:
            rows.append({"group": typ, "asset": path, "label": label, "decision": d, "kind": "setpoint",
                         "segments": seg_list(items, d)})
    for lu in _merge_lineups(plan.lineups):
        rows.append(lu)
    trucks: dict = {}
    for tr in plan.trips.get("trips", []):
        trucks.setdefault(tr["truck"], []).append({"start": tr["depart"], "end": tr["return"], "label": tr["label"],
                                                   "value": tr["volume"], "cargo": tr["cargo"]})
    for k in sorted(trucks):
        rows.append({"group": "truck", "asset": k, "label": f"Truck {k}", "decision": "trips", "kind": "trips",
                     "segments": trucks[k]})
    jobs: dict = {}
    for j in [{**v["job"], "status": v["status"]} for v in engine.jobs.values()]:
        jobs.setdefault(j["resource_label"], []).append({"start": j["start"], "end": j["end"], "label": j["label"],
                                                         "value": j["value_per_h"], "status": j["status"],
                                                         "reason": j["reason"], "kind": j["kind"]})
    for k in sorted(jobs):
        rows.append({"group": "crew", "asset": k, "label": k, "decision": "jobs", "kind": "jobs", "segments": jobs[k]})
    return {"rows": rows, "t0": t0, "t1": t1, "now": engine.t, "plan_id": plan.pid}


def _merge_lineups(lineups: list) -> list:
    by: dict = {}
    for lu in lineups:
        by.setdefault((lu["asset"], lu["port"], lu["label"]), []).append(lu)
    rows = []
    for (asset, port, label), items in sorted(by.items()):
        segs = []
        for it in sorted(items, key=lambda r: r["start"]):
            if segs and segs[-1]["label"] == it["to"] and abs(segs[-1]["end"] - it["start"]) < 1e-6:
                segs[-1]["end"] = it["end"]
            else:
                segs.append({"start": it["start"], "end": it["end"], "label": it["to"], "value": 1.0,
                             "shares": it["shares"]})
        rows.append({"group": "routing", "asset": asset, "label": f"{label} {port.replace('_', ' ')}",
                     "decision": port, "kind": "lineup", "segments": segs})
    return rows


def asset_detail(engine, path: str) -> dict:
    model = engine.twin.model
    flat, prog = model.flat, model.program
    iid = flat.by_path.get(path)
    if iid is None:
        return {"error": f"unknown asset {path}"}
    inst = flat.instances[iid]
    gid, li = prog.inst_group[iid]
    live = engine.twin.live
    params = {}
    for p, decl in inst.type.params.items():
        if p in live.P.get(gid, {}):
            hist = live.param_history.get((path, p), [])
            params[p] = {"value": _disp(float(live.P[gid][p].reshape(-1)[li]), decl.unit),
                         "engineering": _disp(inst.params.get(p, 0.0), decl.unit), "unit": decl.unit or "",
                         "calibrated": decl.calibrate is not None,
                         "history": [[t, _disp(v, decl.unit)] for t, v in hist[-200:]]}
    states = {}
    for s, decl in inst.type.states.items():
        if s in live.S.get(gid, {}):
            states[s] = {"value": _disp(float(live.S[gid][s].reshape(-1)[li]), decl.unit), "unit": decl.unit or ""}
    tags = {}
    for tag in engine.twin.tags.by_path.get(path, []):
        h = engine.tag_history.get(tag)
        info = engine.twin.tags.get(tag)
        tags[tag] = {"unit": info.unit or "", "signal": info.key.split(":")[-1],
                     "history": [list(p) for p in list(h)[-288:]] if h else []}
    plan_series = {}
    plan = engine.planner.plan
    if plan is not None:
        pb = plan.problem
        g = pb.prog.gmap[gid]
        raw = plan.raw or {}
        for k in list(g.ir.nodes):
            if not (k.startswith("local:") or k.startswith("out:") or k.startswith("term:")):
                continue
            try:
                x = raw[pb.prog.node_var(gid, k)]
            except KeyError:
                continue
            x = x if x.dim() == 3 else x.view(1, 1, -1)
            v = x[0, li if x.shape[1] > 1 else 0]
            if v.numel() == 1:
                continue
            unit = g.ir.units.get(k)
            if k.startswith("out:"):
                port, res = k.split(":")[1], k.split(":")[2]
                rt = next((r for name, r in pb.prog.flat.lib.resources.items() if name.split(".")[-1] == res), None)
                unit = rt.display_rate_unit if rt else None
            vals = [round(_disp(float(z), unit), 3) for z in v.tolist()]
            plan_series[k] = {"unit": unit or "", "values": vals}
        for s in g.ir.states:
            try:
                x = raw[pb.prog.node_var(gid, f"state:{s}")]
                x = x if x.dim() == 3 else x.view(1, 1, -1)
                v = x[0, li if x.shape[1] > 1 else 0]
                decl = inst.type.states[s]
                if v.numel() > 1:
                    plan_series[f"state:{s}"] = {"unit": decl.unit or "",
                                                 "values": [round(_disp(float(z), decl.unit), 3) for z in v.tolist()]}
            except KeyError:
                pass
    instr = [i for i in (plan.instructions if plan else []) if i["asset"] == path]
    alerts = [a for a in engine.alerts if a.get("asset") == path][-20:]
    return {"path": path, "label": inst.label or path, "type": inst.type.short, "doc": inst.type.doc,
            "at": list(inst.at) if inst.at else None, "tag": inst.tag, "params": params, "states": states,
            "tags": tags, "plan": plan_series, "plan_t0": plan.t0 if plan else None,
            "plan_dt": plan.problem.dt if plan else None, "instructions": instr, "alerts": alerts}
