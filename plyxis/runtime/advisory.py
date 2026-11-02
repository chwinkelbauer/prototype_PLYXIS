"""Operating modes and the recommendation workflow (pilot phases 2-3, docs/PILOT_PLAN_OIL_GAS.md).

    autonomous   the plan reaches the plant (the showcase)
    advisory     the plan is a list of recommendations; the plant only changes what an operator accepts
                 or modifies; a rejected or modified decision is pinned for the planner, which re-optimises
                 everything else around it
    shadow       the plant runs on manual operations; plans are computed and compared, nothing is executed

Every decision is written to the audit log with who took it, when and why. Plans carry a validity time: a plan
that is too old is no recommendation any more, and the plant holds its last set points.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from typing import Optional

import torch

from ..optim import dispatch as DSP
from .planner import _route_keys

MODES = ("autonomous", "advisory", "shadow")
ACTIONS = ("accept", "modify", "reject")


class AuditLog:
    """Append-only record of plans, decisions, mode changes and twin commits (JSON lines when given a file)."""

    def __init__(self, path: Optional[str] = None, maxlen: int = 5000):
        self.entries: deque = deque(maxlen=maxlen)
        self.path = path
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def add(self, kind: str, t: float, actor: str = "plyxis", **data) -> dict:
        rec = {"wall": round(time.time(), 3), "t": round(float(t), 4), "kind": kind, "actor": actor, **data}
        self.entries.append(rec)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        return rec

    def tail(self, n: int = 300, kind: Optional[str] = None) -> list:
        rows = [e for e in self.entries if kind is None or e["kind"] == kind]
        return rows[-n:]


class Advisory:
    def __init__(self, engine, mode: str = "autonomous", shift_h: float = 12.0, valid_h: float = 12.0,
                 min_change: float = 0.05, audit: Optional[AuditLog] = None):
        self.e = engine
        self.mode = mode if mode in MODES else "autonomous"
        self.shift_h = shift_h          # recommendations cover the next shift
        self.valid_h = valid_h          # a plan older than this is no recommendation any more
        self.min_change = min_change    # below this share of a decision's range a change is not worth a recommendation
        self.audit = audit or AuditLog()
        self.recs: dict = {}            # id -> recommendation, in order of creation
        self.held: dict = {}            # (path, decision) -> set point the plant runs (advisory)
        self.held_routes: dict = {}     # route key -> split the plant runs (advisory)
        self.approved: dict = {}        # (path, decision) -> (start, value): accepted, due at that time
        self.approved_routes: dict = {}  # route key -> split
        self.overrides: dict = {}       # (path, decision) -> (value, until): what the planner must respect
        self.shadow_log: list = []
        self.expired_alert = False

    # ------------------------------------------------------------------ mode
    def set_mode(self, mode: str, actor: str = "operator") -> dict:
        if mode not in MODES:
            return {"ok": False, "error": f"mode must be one of {', '.join(MODES)}"}
        if mode != self.mode:
            self.audit.add("mode", self.e.t, actor, previous=self.mode, mode=mode)
            self.mode = mode
            self.approved, self.approved_routes = {}, {}
            if mode != "advisory":              # outside advisory mode the plan rules again
                self.overrides = {}
            else:                               # from now on the plant runs what it runs now, until someone decides
                self.held, self.held_routes = self.e.plants["plyxis"].readback()
                if self.e.planner.plan is not None:
                    self.on_plan(self.e.planner.plan)
            for r in self.recs.values():
                if r["status"] == "pending" and mode != "advisory":
                    r["status"] = "superseded"
        return {"ok": True, "mode": self.mode}

    # ------------------------------------------------------------------ what the plant runs
    def plan_valid(self, t: float) -> bool:
        """A plan expires only when no newer one is on its way: at 600x time compression a 60 s solve is 10 plant hours."""
        e, plan = self.e, self.e.planner.plan
        if plan is None:
            return False
        solving = e.planner.running or (e._future is not None and not e._future.done())
        return solving or t - plan.t0 <= self.valid_h + 1e-9

    def setpoints(self, t: float) -> tuple:
        """Set points for the PLYXIS plant at time t in autonomous and advisory mode."""
        if self.e.planner.plan is not None and not self.plan_valid(t):
            if not self.expired_alert:          # no stale recommendation: the plant holds what it runs
                self.expired_alert = True
                self.e._alert("critical", "plan", f"The last plan is older than {self.valid_h:.0f} h: no recommendation, "
                                                  "the plant holds its last set points")
                self.audit.add("plan_expired", t, plan=self.e.planner.plan.pid)
            return {}, {}
        self.expired_alert = False
        if self.mode == "autonomous":
            return self.e.planner.setpoints_at(t)
        for key, (start, v) in list(self.approved.items()):
            if start <= t + 1e-9:
                self.held[key] = v
                del self.approved[key]
        self.held_routes.update(self.approved_routes)
        self.approved_routes = {}
        return dict(self.held), dict(self.held_routes)

    def pins(self, t: float) -> dict:
        """Decisions the planner must respect: {(path, decision): (value, hours from t)}."""
        for key in [k for k, (_, until) in list(self.overrides.items()) if until <= t + 1e-9]:
            del self.overrides[key]
        return {k: (v, until - t) for k, (v, until) in list(self.overrides.items())}

    # ------------------------------------------------------------------ recommendations
    def on_plan(self, plan) -> None:
        """Turn a new plan into the changes it asks for in the next shift, relative to what the plant runs."""
        if getattr(plan, "interim", False):
            return
        pb = plan.problem
        self.audit.add("plan", self.e.t, plan=plan.pid, trigger=plan.trigger, objective=round(plan.objective),
                       feasible=plan.feasible, seconds=round(plan.solver.get("seconds", 0.0), 2))
        current, routes_now = self.e.plants["plyxis"].readback()
        if self.mode == "advisory":
            current, routes_now = {**current, **self.held}, {**routes_now, **self.held_routes}
        for r in list(self.recs.values()):
            if r["status"] == "pending":
                r["status"] = "superseded"
        bnd = pb.space.bounds(pb.P)
        prog = pb.prog
        status = "pending" if self.mode == "advisory" else "shadow" if self.mode == "shadow" else "automatic"
        n, end = 0, plan.t0 + self.shift_h
        seen = set()
        for ins in plan.instructions:
            key = (ins["asset"], ins["decision"])
            if key in seen or ins["start"] >= end or key in self.overrides or key not in current:
                continue
            iid = prog.flat.by_path.get(ins["asset"])
            if iid is None:
                continue
            gid, li = prog.inst_group[iid]
            blk = pb.space.block(gid, ins["decision"])
            lo, hi, _ = bnd[(gid, ins["decision"])]
            lo_v, hi_v = float(lo.reshape(-1)[li if lo.numel() > 1 else 0]), float(hi.reshape(-1)[li if hi.numel() > 1 else 0])
            cur, to = float(current[key]), float(ins["value"])
            discrete = blk.domain in ("int", "bool")
            changed = abs(to - cur) >= 0.5 if discrete else abs(to - cur) > self.min_change * max(hi_v - lo_v, 1e-9)
            if not changed:
                continue
            seen.add(key)
            n += 1
            unit = prog.gmap[gid].ir.decisions[ins["decision"]].unit
            rid = f"{plan.pid}-{n}"
            self.recs[rid] = {"id": rid, "plan": plan.pid, "kind": "setpoint", "asset": ins["asset"], "label": ins["label"],
                              "type": ins["type"], "decision": ins["decision"], "from": cur, "to": to,
                              "lo": lo_v, "hi": hi_v, "unit": unit, "domain": blk.domain,
                              "from_display": DSP._fmt(cur, unit, lo_v, hi_v, blk.domain), "to_display": ins["display"],
                              "start": max(ins["start"], plan.t0), "end": ins["end"], "status": status,
                              "share": abs(to - cur) / max(hi_v - lo_v, 1e-9), "created": self.e.t}
        self._route_recs(plan, routes_now, status, n)
        self._trim()
        if self.mode == "shadow":
            self._shadow_value(plan, current, routes_now)

    def _route_recs(self, plan, routes_now: dict, status: str, n: int) -> None:
        _, routes = self.e.planner.setpoints_at(plan.t0)
        by: dict = {}
        for k, v in routes.items():
            by.setdefault(k[:2], []).append((k, v))
        for (src, port), rows in by.items():
            change = sum(abs(v - routes_now.get(k, 0.0)) for k, v in rows)
            if change <= 0.2:
                continue
            n += 1
            rid = f"{plan.pid}-{n}"
            names = lambda vals: ", ".join(f"{k[2].replace('__', ' to ')} {v * 100:.0f} %" for k, v in vals if v > 0.005) or "none"  # noqa: E731
            self.recs[rid] = {"id": rid, "plan": plan.pid, "kind": "lineup", "asset": src, "label": src, "type": "line-up",
                              "decision": port, "routes": {"|".join(k): v for k, v in rows},
                              "from_display": names([(k, routes_now.get(k, 0.0)) for k, _ in rows]),
                              "to_display": names(rows), "start": plan.t0, "end": plan.t0 + self.shift_h,
                              "status": status, "share": change / 2.0, "created": self.e.t}

    def _trim(self, keep: int = 3000) -> None:
        while len(self.recs) > keep:
            self.recs.pop(next(iter(self.recs)))

    # ------------------------------------------------------------------ decisions
    def decide(self, ids, action: str, actor: str = "operator", reason: str = "", value: Optional[float] = None) -> dict:
        if action not in ACTIONS:
            return {"ok": False, "error": f"action must be one of {', '.join(ACTIONS)}"}
        if self.mode != "advisory":
            return {"ok": False, "error": "recommendations are only decided in advisory mode"}
        if action in ("reject", "modify") and not reason.strip():
            return {"ok": False, "error": "a rejection or modification needs a reason: it is how the twin learns what it misses"}
        targets = [r for r in self.recs.values() if r["status"] == "pending"] if ids == "all" else \
            [self.recs[i] for i in ids if i in self.recs and self.recs[i]["status"] == "pending"]
        if action == "modify" and (value is None or len(targets) != 1 or targets[0]["kind"] != "setpoint"):
            return {"ok": False, "error": "modify takes one set point and a value"}
        t, need_plan = self.e.t, False
        for r in targets:
            r.update(decided_by=actor, decided_at=t, reason=reason)
            if r["kind"] == "lineup":
                keys = {tuple(k.split("|")): v for k, v in r["routes"].items()}
                if action == "accept":
                    self.approved_routes.update(keys)
                    r["status"] = "accepted"
                else:                            # the line-up stays as it runs; the planner has to live with it
                    for key in keys:
                        self.overrides[("route", key)] = (float(self.held_routes.get(key, 0.0)), t + self.shift_h)
                    r["status"] = "rejected"
                    need_plan = True
            else:
                key = (r["asset"], r["decision"])
                if action == "accept":
                    self.approved[key] = (r["start"], r["to"])
                    r["status"] = "accepted"
                else:
                    held = float(value) if action == "modify" else float(self.held.get(key, r["from"]))
                    held = min(max(held, r["lo"]), r["hi"])
                    self.held[key] = held
                    self.overrides[key] = (held, t + self.shift_h)
                    r["status"] = "modified" if action == "modify" else "rejected"
                    r["value"] = held
                    need_plan = True
            self.audit.add("decision", t, actor, action=action, rec=r["id"], asset=r["asset"], decision=r["decision"],
                           to=r.get("to_display"), value=r.get("value"), reason=reason)
        if need_plan:                            # the rest of the plan re-optimises around what was decided
            self.e.request_replan("operator decision")
        return {"ok": True, "decided": len(targets), "replan": need_plan}

    # ------------------------------------------------------------------ shadow mode
    def _shadow_value(self, plan, current: dict, routes_now: dict) -> None:
        """Pilot phase 2: the plan's value against holding what the operators run now, on the calibrated twin."""
        pb = plan.problem
        x = pb.space.default_u(1)
        parts = pb.space.split(x)
        bnd = pb.space.bounds(pb.P)
        prog = pb.prog
        for b in pb.space.blocks:
            g = prog.gmap[b.gid]
            lo, hi, _ = bnd[(b.gid, b.name)]
            for li, iid in enumerate(g.inst_ids):
                v = current.get((prog.flat.instances[iid].path, b.name))
                if v is not None:
                    span = float((hi - lo).reshape(-1)[li if (hi - lo).numel() > 1 else 0])
                    low = float(lo.reshape(-1)[li if lo.numel() > 1 else 0])
                    parts[(b.gid, b.name)][0, li, :] = 0.0 if span < 1e-12 else min(max((v - low) / span, 0.0), 1.0)
        if pb.space.route is not None:
            for key, r in _route_keys(prog).items():
                if key in routes_now:
                    parts["route"][0, r, :] = float(routes_now[key])
        with torch.no_grad():
            e = pb.evaluate(x, relax=False)
        self.shadow_log.append({"t": plan.t0, "plan": plan.pid, "plan_value": round(plan.objective),
                                "hold_value": round(float(e.objective[0])),
                                "gap": round(plan.objective - float(e.objective[0])),
                                "plan_feasible": plan.feasible, "hold_max_violation": round(float(e.max_violation()[0]), 4)})
        self.shadow_log = self.shadow_log[-200:]

    # ------------------------------------------------------------------ views
    def _override_view(self) -> list:
        out, lineups = [], {}
        for k, (v, u) in list(self.overrides.items()):
            if k[0] == "route":                  # one line per source and port
                lineups[k[1][:2]] = u
            else:
                out.append({"asset": k[0], "decision": k[1], "value": v, "until": u})
        out += [{"asset": src, "decision": f"{port} line-up", "value": None, "until": u} for (src, port), u in lineups.items()]
        return out

    def view(self, limit: int = 400) -> dict:
        rows = list(self.recs.values())
        latest = self.e.planner.plan.pid if self.e.planner.plan is not None else None
        current = [r for r in rows if r["plan"] == latest or r["status"] == "pending"]
        current.sort(key=lambda r: -r["share"])
        counts: dict = {}
        for r in rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        plan = self.e.planner.plan
        return {"mode": self.mode, "modes": list(MODES), "shift_h": self.shift_h, "plan": latest,
                "plan_valid": self.plan_valid(self.e.t), "valid_until": plan.t0 + self.valid_h if plan else None,
                "recommendations": current[:limit], "total": len(current), "counts": counts,
                "recent_decisions": [r for r in rows if r.get("decided_at") is not None][-40:][::-1],
                "overrides": self._override_view(),
                "shadow": self.shadow_log[-48:]}
