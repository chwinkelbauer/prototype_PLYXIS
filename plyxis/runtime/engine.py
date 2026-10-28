"""PLYXIS runtime engine - the live operating loop.

            +-------------------- PLYXIS world --------------------+
  events -> | plant emulator -> connectors -> estimation / anomaly |
            |      ^                         -> discovery -> twin |
            |      |  set points, line-ups, crews                   |
            |      +---------- planner (cascade) <-----------------+
            +-------------------------------------------------------+
  events -> | plant emulator <- legacy operators (rules)            |  <- baseline world
            +-------------------------------------------------------+

Both worlds start from the same state, see the same events, prices and noise;
the difference in realised KPIs is the value of autonomous operations.
"""
from __future__ import annotations

import dataclasses
import math
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import torch

from ..data.anomaly import AnomalyDetector
from ..data.connectors import EmulatorConnector, HttpPushConnector, MarketConnector
from ..data.emulator import Birth, PlantEmulator, Telemetry
from ..data.forecast import Forecaster
from ..lang import units as U
from ..optim import dispatch as DSP
from ..optim.explain import explain_instance
from .planner import Planner
from .scenario import PriceModel, Scenario
from .twin_manager import TwinManager

MARKET_TAGS = {"oil_price": "MKT.WTI", "gas_price": "MKT.WAHA", "power_price": "MKT.ERCOT.WEST"}


class Engine:
    def __init__(self, scenario_dir: str, seed: int = 0, use_neural: bool = True, auto_discovery: bool = True,
                 speed: float = 600.0, sync_plans: bool = False, online_learning: bool = True,
                 log: Optional[Callable] = None):
        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
        self.sc = Scenario.load(scenario_dir)
        self.seed = seed
        self.use_neural = use_neural
        self.auto_discovery = auto_discovery
        self.sync_plans = sync_plans
        self.online_learning = online_learning
        self.log = log or (lambda *a: None)
        self.speed = speed
        self.running = False
        self._listeners: list = []
        self._lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="planner")
        self.est_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="estimator")
        self._calib_future = None
        self._future = None
        self._urgent = False
        self.hold_on_disruption = True
        self._thread = None
        self._stop = False
        self._build()

    # ------------------------------------------------------------------ construction / reset
    def _build(self) -> None:
        sc = self.sc
        self.t0 = sc.start_hour
        self.t = sc.start_hour
        self.prices = PriceModel(sc.prices, seed=11 + self.seed)
        pf = lambda name, clock: self.prices.canonical(name, clock)  # noqa: E731
        metrics = sc.raw.get("kpi", {}).get("metrics")
        self.plants = {
            "plyxis": PlantEmulator(sc.truth_files, sc.truth_env, dt=sc.dt_emu, seed=self.seed, name="plyxis",
                                    hidden=list(sc.hidden), defines_prices=pf, kpi_metrics=metrics),
            "legacy": PlantEmulator(sc.truth_files, sc.truth_env, dt=sc.dt_emu, seed=self.seed, name="legacy",
                                    hidden=list(sc.hidden), defines_prices=pf, kpi_metrics=metrics),
        }
        for p in self.plants.values():
            p.t = self.t
        Legacy = sc.legacy_class()
        self.legacy_ops = Legacy(self.plants["legacy"], sc.legacy_cfg) if Legacy else None
        self.twin = TwinManager(sc.twin_files, sc.twin_env, sc.twin_sim, dt_data=sc.dt_emu)
        net_path = os.path.join(sc.root, "models", "subsolver.pt")
        cfg = dataclasses.replace(Planner.LIVE, **sc.raw["planner"]) if sc.raw.get("planner") else None
        self.planner = Planner(self.twin, cfg=cfg, net_path=net_path if self.use_neural else None,
                               online_learning=self.online_learning)
        site = sc.raw.get("title", sc.name.split(" - ")[0])
        self.scada = EmulatorConnector(f"SCADA gateway - {site}", "OPC UA / MQTT Sparkplug B gateway (emulated)")
        tags = sc.raw.get("market_tags", MARKET_TAGS)
        self.market = MarketConnector("Market data", lambda n, c: self.prices.display(n, c),
                                      {k: v for k, v in tags.items() if k in self.twin.model.program.env_inputs},
                                      sc.raw.get("market_desc", "WTI, Waha hub, ERCOT West (synthetic)"))
        self.http = HttpPushConnector()
        self.connectors = [self.scada, self.market, self.http]
        prog = self.twin.model.program
        self.forecaster = Forecaster(sorted(prog.env_inputs),
                                     {k: float(prog.E0[k]) for k in prog.env_inputs})
        self.anomaly = AnomalyDetector(self.twin.tags, self.twin.model.flat)
        self.alerts: deque = deque(maxlen=400)
        self.timeline: list = []
        self.fired: set = set()
        self.restores: list = []                # (t_end, event) of temporary capacity changes
        self.outages: dict = {}                 # path -> (state, value when back, estimated t back): crew estimates
        self.jobs: dict = {}                  # id -> {job, status}
        self.latest: dict = {}                # tag -> (value, t)
        self.tag_history: dict = {}
        self.last_calib = self.t
        self.last_plan_t = -1e9
        self.pending_trigger: Optional[str] = None
        self.steps = 0
        self.kpi_history: list = []
        self.calib_log: list = []
        self.twin_events: list = []
        self.plan_events: deque = deque(maxlen=50)
        self.learn_log: list = []

    def reset(self) -> None:
        with self._lock:
            self.running = False
            if self._future is not None:
                self._future.cancel()
            self._calib_future = None            # results of the previous run must never reach the new twin
            self._build()
            self.emit({"type": "reset"})

    # ------------------------------------------------------------------ listeners (GUI push)
    def subscribe(self, fn: Callable) -> None:
        self._listeners.append(fn)

    def unsubscribe(self, fn: Callable) -> None:
        if fn in self._listeners:
            self._listeners.remove(fn)

    def emit(self, msg: dict) -> None:
        for fn in list(self._listeners):
            try:
                fn(msg)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ helpers
    def clock_str(self, t: Optional[float] = None) -> str:
        t = self.t if t is None else t
        day = int(t // 24) + 1
        h = t % 24
        return f"Day {day} {int(h):02d}:{int(round((h % 1) * 60)) % 60:02d}"

    def _alert(self, severity: str, kind: str, message: str, asset: Optional[str] = None, data=None) -> dict:
        a = {"t": self.t, "clock": self.clock_str(), "severity": severity, "kind": kind, "message": message,
             "asset": asset, "data": data or {}}
        self.alerts.append(a)
        self.emit({"type": "alert", "alert": a})
        return a

    def _timeline(self, kind: str, title: str, text: str = "", **kw) -> None:
        rec = {"t": self.t, "clock": self.clock_str(), "kind": kind, "title": title, "text": text, **kw}
        self.timeline.append(rec)
        self.emit({"type": "timeline", "event": rec})

    # ------------------------------------------------------------------ planning
    def request_replan(self, trigger: str) -> None:
        if self.sync_plans:
            self._replan(trigger)
            return
        if self.planner.running or (self._future is not None and not self._future.done()):
            self.pending_trigger = trigger
            return
        self._future = self.executor.submit(self._replan, trigger)

    def _forecast(self, T: int, dt: float) -> dict:
        return self.forecaster.forecast(self.t, T, dt)

    URGENT = ("disruption", "twin rewritten", "market notice", "midstream notice", "returned to service",
              "twin edited", "initial")

    def _replan(self, trigger: str) -> None:
        self._urgent = any(k in trigger for k in self.URGENT)
        try:
            pb_T = self.twin.model.program.T
            E = self._forecast(pb_T, self.twin.model.program.dt)
            busy = {j["job"]["resource"]: j["job"]["end"] for j in self.jobs.values() if j["status"] == "active"}
            self.emit({"type": "optimizer", "event": "start", "trigger": trigger, "t": self.t, "clock": self.clock_str()})
            t_solve = self.t

            def cb(rec):
                if rec.get("iter", 0) % 10 == 0 or rec.get("event") or rec.get("stage") in ("done", "neural"):
                    self.emit({"type": "optimizer", "event": "progress", "rec": _jsonable(rec)})
            worked = {j["job"]["asset"] for j in self.jobs.values() if j["status"] == "active"}
            predicted = {p: max(0.0, eta - (self.t - t0)) for p, (t0, eta) in self.anomaly.predicted.items()
                         if self.t - t0 < 48.0}
            def interim(p):
                # first response is live: release the clock hold, keep refining in the background
                self._urgent = False
                self.emit({"type": "plan", "plan": {"id": p.pid, "t": t_solve, "clock": self.clock_str(t_solve),
                                                    "trigger": p.trigger, "objective": p.objective,
                                                    "feasible": p.feasible, "seconds": p.solver["seconds"],
                                                    "variables": p.solver["variables"], "interim": True}})
                self._timeline("plan", f"First response ({trigger})",
                               f"set points updated in {p.solver['seconds']:.1f} s - full optimisation continues")
            outages = {p: o for p, o in self.outages.items() if o[2] > t_solve}
            plan = self.planner.solve(t_solve, E, trigger, busy=busy, callback=cb, worked_assets=worked,
                                      predicted=predicted, urgent=self._urgent, outages=outages,
                                      quick_first=self._urgent and not self.sync_plans and "initial" not in trigger,
                                      on_interim=interim)
            self.last_plan_t = t_solve
            # merge maintenance jobs: keep active/done, replace scheduled
            with self._lock:
                for jid in [k for k, v in self.jobs.items() if v["status"] == "scheduled"]:
                    del self.jobs[jid]
                for j in plan.jobs:
                    if j["id"] not in self.jobs and j["asset"] not in worked:
                        self.jobs[j["id"]] = {"job": j, "status": "scheduled"}
            rec = {"id": plan.pid, "t": t_solve, "clock": self.clock_str(t_solve), "trigger": trigger,
                   "objective": plan.objective, "feasible": plan.feasible,
                   "seconds": plan.solver["seconds"], "variables": plan.solver["variables"]}
            self.plan_events.append(rec)
            self.emit({"type": "plan", "plan": rec})
            self._timeline("plan", f"Re-plan #{plan.pid} ({trigger})",
                           f"{plan.solver['variables']:,} decisions optimised in {plan.solver['seconds']:.1f} s")
            if self.online_learning and self.planner.net is not None and plan.pid % 3 == 0:
                info = self.planner.learn(steps=20)
                if info and "error" not in info:
                    self.learn_log.append({"t": self.t, **info})
        except Exception as ex:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self._alert("warning", "optimizer", f"re-plan failed: {ex}")
        finally:
            self._urgent = False
            if self.pending_trigger and not self.sync_plans:
                trig, self.pending_trigger = self.pending_trigger, None
                self._future = self.executor.submit(self._replan, trig)

    # ------------------------------------------------------------------ scripted events
    def _events_due(self) -> None:
        for i, ev in enumerate(self.sc.events):
            if i in self.fired:
                continue
            if self.sc.start_hour + ev["t"] <= self.t + 1e-9:
                self.fired.add(i)
                self._apply_event(ev)
        # temporary capacity changes end on their own (e.g. a gas-plant upset of 10 h)
        for item in [r for r in self.restores if r[0] <= self.t + 1e-9]:
            self.restores.remove(item)
            self._apply_event(item[1])

    def _note_outage(self, ev: dict, asset: str) -> None:
        """Crew estimate of an outage with a known duration (`eta_hours`, default `hours`; `eta: false` = no
        estimate): the twin plans the asset back at that time instead of assuming it stays down."""
        if not ev.get("hours") or ev.get("eta", True) is False:
            return
        back = self.t + float(ev.get("eta_hours", ev["hours"]))
        if ev["kind"] == "fail":
            self.outages[asset] = ("up", 1.0, back)
            return
        prog = self.twin.model.program
        iid = prog.flat.by_path.get(asset)
        n = self.twin.live.P[prog.inst_group[iid][0]].get("n_units") if iid is not None else None
        if n is not None:
            self.outages[asset] = ("avail", float(n.reshape(-1)[prog.inst_group[iid][1]]), back)

    def _apply_event(self, ev: dict, manual: bool = False) -> None:
        kind = ev["kind"]
        title = ev.get("title", kind)
        if kind in ("repair", "restore_units"):
            for a in ev.get("assets") or [ev.get("asset")]:
                self.outages.pop(a, None)
        if kind == "announce":
            s, e = self.sc.start_hour + ev["start"], self.sc.start_hour + ev["end"]
            if ev.get("relative", True) is False:
                s, e = ev["start"], ev["end"]
            u = self.prices.unit(ev["name"])
            v = U.to_canonical(ev["value"], u) if u else ev["value"]
            self.forecaster.announce(ev["name"], s, e, v)
            if self.legacy_ops is not None and hasattr(self.legacy_ops, "notice"):
                self.legacy_ops.notice(dict(ev, start=s, end=e), self.t)     # both control rooms get the notice
            self._timeline("market", title, ev.get("narrative", ""))
            self._alert("info", "market", title)
            self.request_replan(f"market notice: {ev['name']}")
            return
        if ev.get("assets"):                    # one event on several assets (e.g. a level closed by seismicity)
            for a in ev["assets"]:
                self._apply_event({k: v for k, v in ev.items() if k not in ("assets", "hours", "title", "narrative")}
                                  | {"asset": a, "quiet": True}, manual)
            if ev.get("hours") and not ev.get("restore"):
                back = {"kind": "repair" if kind == "fail" else "restore_units", "assets": ev["assets"], "restore": True,
                        "replace": False, "title": ev.get("restore_title") or "Back in service", "narrative": ""}
                self.restores.append((self.t + float(ev["hours"]), back))
                if kind in ("fail", "trip"):
                    for a in ev["assets"]:
                        self._note_outage(ev, a)
            self._timeline("event", title, ev.get("narrative", ""), asset=ev["assets"][0], manual=manual)
            if self.legacy_ops is not None and hasattr(self.legacy_ops, "notice"):
                self.legacy_ops.notice(ev, self.t)
            return
        if kind == "capacity" and ev.get("hours") and not ev.get("restore"):
            back = dict(ev, factor=1.0 / float(ev.get("factor", 1.0)), restore=True, hours=None,
                        title=f"{ev.get('restore_title') or 'Capacity restored'}", narrative="")
            self.restores.append((self.t + float(ev["hours"]), back))
        if kind in ("fail", "trip") and ev.get("hours") and not ev.get("restore"):
            # an outage with a known duration (crew estimate): the asset comes back on its own
            back = {"kind": "repair" if kind == "fail" else "restore_units", "asset": ev["asset"], "restore": True,
                    "replace": False, "title": ev.get("restore_title") or f"{ev['asset']} back in service", "narrative": ""}
            self.restores.append((self.t + float(ev["hours"]), back))
            self._note_outage(ev, ev["asset"])
        # `hours` of a capacity event is handled here (auto-restore); for a price it is the override's duration
        skip = ("t", "kind", "asset", "title", "narrative", "notice", "restore", "restore_title", "quiet") + \
            (("hours",) if kind in ("capacity", "fail", "trip") else ())
        for name, plant in self.plants.items():
            kw = {k: v for k, v in ev.items() if k not in skip}
            if kind == "price":
                u = self.prices.unit(ev["name"])
                kw["value"] = U.to_canonical(ev["value"], u) if u else ev["value"]
            plant.apply_event(kind, ev.get("asset"), **kw)
            if kind == "commission" and name == "plyxis":
                birth = plant.births(ev["asset"])
                self.scada.push([birth])
        if not ev.get("quiet"):
            self._timeline("event", title, ev.get("narrative", ""), asset=ev.get("asset"), manual=manual)
        if ev.get("notice") and kind == "capacity":
            # a notice (e.g. midstream curtailment) is information both control rooms receive at once
            prog = self.twin.model.program
            iid = prog.flat.by_path.get(ev["asset"])
            if iid is not None:
                gid, li = prog.inst_group[iid]
                for P in (self.twin.live.P, self.twin.live.anchor_P, self.twin.live.prior_P):
                    if ev["param"] in P.get(gid, {}):
                        P[gid][ev["param"]].reshape(-1)[li] *= float(ev.get("factor", 1.0))
            if self.legacy_ops is not None and hasattr(self.legacy_ops, "notice"):
                self.legacy_ops.notice(ev, self.t)
            self.request_replan("midstream notice")

    def inject(self, ev: dict) -> dict:
        """Manual event from the GUI (e.g. trip a compressor during a live demo)."""
        ev = dict(ev)
        ev.setdefault("title", f"{ev['kind']} {ev.get('asset', '')}".strip())
        with self._lock:                 # never mutate the plants while the plant clock is stepping them
            self._apply_event(ev, manual=True)
        return {"ok": True, "t": self.t}

    # ------------------------------------------------------------------ maintenance execution
    def _maintenance(self) -> None:
        plant = self.plants["plyxis"]
        changed = False
        for jid, rec in list(self.jobs.items()):
            j = rec["job"]
            if rec["status"] == "scheduled" and j["start"] <= self.t:
                rec["status"] = "active"
                if j["kind"] == "preventive":
                    plant.apply_event("fail", j["asset"])
                self._timeline("maintenance", f"{j['resource_label']} started: {j['label']}", j["reason"])
            elif rec["status"] == "active" and j["end"] <= self.t:
                rec["status"] = "done"
                units = "units" in jid or "unit" in j["reason"]
                if units:
                    plant.apply_event("restore_units", j["asset"])
                else:   # workovers install a new pump; a crew restart does not
                    plant.apply_event("repair", j["asset"], replace=j["kind"] in ("repair", "preventive"))
                self._close_work_order(j, units)
                self._timeline("maintenance", f"{j['label']} back in service", f"{j['resource_label']} finished")
                changed = True
        if changed:
            self.request_replan("asset returned to service")

    def _after_calibration(self, rep: list) -> bool:
        """Commit calibration results to the twin repository; True if a disruption was detected."""
        critical = False
        big = [r for r in rep if r[1] in ("inj_cap", "liq_cap") or
               (r[1] == "q_max" and abs(r[3] - r[2]) > 0.15 * abs(r[2]))]
        if rep:
            commit = self.twin.write_calibration(self.t)
            self.calib_log.append({"t": self.t, "changes": len(rep), "commit": commit["id"] if commit else None,
                                   "items": [_calib_item(r) for r in rep[:30]]})
            if commit:
                self.emit({"type": "twin", "commit": commit})
        for r in big:
            if r[1] in ("inj_cap", "liq_cap") and abs(r[3] - r[2]) > 0.1 * abs(r[2]):
                self._alert("warning", "calibration",
                            f"{r[0]}: {r[1]} re-estimated {U.from_canonical(r[2], 'bbl/d'):,.0f} -> "
                            f"{U.from_canonical(r[3], 'bbl/d'):,.0f} bbl/d", r[0])
                critical = True
        return critical

    def _close_work_order(self, j: dict, units: bool) -> None:
        """A closed work order is information for the twin (as from SAP PM / Maximo): the asset is back
        in service and a replaced pump is new - known before the next telemetry or estimation."""
        prog = self.twin.model.program
        iid = prog.flat.by_path.get(j["asset"])
        if iid is None:
            return
        gid, li = prog.inst_group[iid]
        S, P = self.twin.live.S.get(gid, {}), self.twin.live.P.get(gid, {})
        if units:
            if "avail" in S and "n_units" in P:
                self.twin.estimator.reset_states(gid, li, {"avail": float(P["n_units"].reshape(-1)[li])})
            return
        values = {"up": 1.0}
        if j["kind"] in ("repair", "preventive") and "health" in S:
            values["health"] = 1.0
            self.anomaly.predicted.pop(j["asset"], None)        # the new pump has no adverse trend
        self.twin.estimator.reset_states(gid, li, values)

    # ------------------------------------------------------------------ one emulator step
    def step(self) -> None:
        with self._lock:
            sc, dt = self.sc, self.sc.dt_emu
            self._events_due()
            # --- PLYXIS world: apply plan, run plant
            pp = self.plants["plyxis"]
            vals, routes = self.planner.setpoints_at(self.t)
            if vals:
                pp.set_controls(vals, routes)
            self._maintenance()
            out = pp.step()
            self.scada.push(out["telemetry"])
            for ev in out["events"]:
                self._timeline("failure", f"{ev.get('label') or ev['asset']}: {ev['detail']}", "", asset=ev["asset"])
            # --- legacy world
            pl = self.plants["legacy"]
            if self.legacy_ops is not None:
                self.legacy_ops.control(self.t)
            pl.step()
            # --- data integration (PLYXIS world)
            msgs = []
            for c in self.connectors:
                msgs.extend(c.poll(self.t))
            tele = []
            for m in msgs:
                if isinstance(m, Birth):
                    self.twin.discovery.note_birth(m.prefix, m.meta)
                    self._alert("info", "discovery", f"new device announced: {m.prefix} ({m.meta.get('type', '?')})")
                    continue
                if m.tag.startswith("MKT."):
                    name = next((k for k, v in MARKET_TAGS.items() if v == m.tag), None)
                    if name:
                        u = self.prices.unit(name)
                        self.forecaster.observe(name, m.t, U.to_canonical(m.value, u) if u else m.value)
                    self.latest[m.tag] = (m.value, m.t)
                    continue
                self.latest[m.tag] = (m.value, m.t)
                h = self.tag_history.setdefault(m.tag, deque(maxlen=432))
                h.append((round(m.t, 3), m.value))
                if self.twin.tags.get(m.tag) is None and m.quality != "test":
                    prop = self.twin.discovery.observe(m.tag, self.t)
                    if prop is not None:
                        self._on_discovery(prop)
                    continue
                tele.append(m)
            vals_rb, routes_rb = pp.readback()
            D, R = self.twin.live.controls_from_paths(vals_rb, routes_rb)
            E_now = {k: torch.tensor([self.prices.canonical(k, self.t)]) for k in self.twin.model.program.env_inputs}
            changes = self.twin.estimator.ingest(self.t, tele, D, R, E_now)
            self.twin.estimator.propagate()
            preds = self.twin.estimator.predict_last()
            alerts = self.anomaly.update(self.t, tele, preds)
            critical = False
            for a in alerts:
                d = a.to_dict()
                d["clock"] = self.clock_str(a.t)
                self.alerts.append(d)
                self.emit({"type": "alert", "alert": d})
                if a.severity == "critical" and a.kind == "status":
                    critical = True
            if changes["states"]:
                for path, name, old, new in changes["states"]:
                    if name in ("up", "avail") and new < old:
                        critical = True
            self.t = round(self.t + dt, 9)
            self.steps += 1
            # --- calibration (headless/deterministic: inline; live: on a worker so the plant clock never stalls)
            if self._calib_future is not None and self._calib_future.done():
                fut, self._calib_future = self._calib_future, None
                try:
                    res = fut.result()
                except Exception as ex:  # noqa: BLE001
                    res = None
                    self._alert("warning", "calibration", f"estimation failed: {ex}")
                if res is not None and res.get("version") == self.twin.version:
                    critical = self._after_calibration(self.twin.estimator.calibrate_apply(res)) or critical
            if self.t - self.last_calib >= sc.calibrate_every - 1e-9 and self._calib_future is None:
                self.last_calib = self.t
                tags = self.twin.tags
                est = self.twin.estimator
                est.released = {tags.get(tag).path for (tag, kind) in self.anomaly.active
                                if kind == "mismatch" and tags.get(tag) is not None}
                if self.sync_plans:
                    critical = self._after_calibration(est.calibrate(iters=25)) or critical
                else:
                    version = self.twin.version

                    def job(est=est, version=version):
                        r = est.calibrate_compute(iters=25)
                        if r is not None:
                            r["version"] = version
                        return r
                    self._calib_future = self.est_executor.submit(job)
            # --- KPI history (both worlds)
            if self.steps % 3 == 0:
                self.kpi_history.append(self._kpi_point())
                if len(self.kpi_history) > 5000:
                    self.kpi_history = self.kpi_history[-5000:]
            # --- re-planning triggers
            due = self.t - self.last_plan_t >= sc.replan_every - 1e-9
            if critical:
                self.request_replan("disruption detected")
            elif due and not self.planner.running:
                self.request_replan("rolling horizon")

    def _on_discovery(self, prop) -> None:
        self._alert("warning", "discovery", f"unknown asset {', '.join(prop.assets)} on tag {prop.tag} - "
                                            "twin update proposed", prop.assets[0] if prop.assets else None)
        self._timeline("twin", f"Discovery: {', '.join(prop.assets)}", prop.code)
        self.emit({"type": "discovery", "proposal": prop.to_dict()})
        if self.auto_discovery:
            self.apply_proposal(prop.pid, author="auto: discovery")

    def apply_proposal(self, pid: str, author: str = "engineer") -> dict:
        with self._lock:
            prop = self.twin.discovery.proposals.get(pid)
            if prop is None or prop.status != "pending":
                return {"ok": False, "error": "unknown or already handled proposal"}
            res = self.twin.apply_discovery(prop, self.t, author=author)
            if res.get("ok"):
                prev = self.anomaly.predicted
                self.anomaly = AnomalyDetector(self.twin.tags, self.twin.model.flat)
                self.anomaly.predicted = prev            # keep condition trends across twin rewrites
                self._timeline("twin", f"Twin rewritten (v{self.twin.version}): {', '.join(prop.assets)} added",
                               f"commit {res['commit']['id'] if res.get('commit') else ''}")
                self.emit({"type": "twin", "commit": res.get("commit"), "structure_changed": True})
                self.request_replan("twin rewritten (new asset)")
            return res

    def reject_proposal(self, pid: str) -> dict:
        prop = self.twin.discovery.proposals.get(pid)
        if prop is None:
            return {"ok": False}
        prop.status = "rejected"
        return {"ok": True}

    def edit_twin(self, files: dict, message: str, author: str = "engineer") -> dict:
        with self._lock:
            res = self.twin.apply_edit(files, author, message, self.t)
            if res.get("ok"):
                prev = self.anomaly.predicted
                self.anomaly = AnomalyDetector(self.twin.tags, self.twin.model.flat)
                self.anomaly.predicted = prev            # keep condition trends across twin rewrites
                self._timeline("twin", f"Twin edited by {author} (v{self.twin.version})", message)
                self.emit({"type": "twin", "commit": res.get("commit"), "structure_changed": True})
                self.request_replan("twin edited")
            return res

    # ------------------------------------------------------------------ run loop (server)
    def holding(self) -> bool:
        """While a disruption-triggered re-plan is computed the plant clock waits: a solve of
        10-20 s is instantaneous relative to plant dynamics, but not at 3600x time compression."""
        return self.hold_on_disruption and self._urgent and self.planner.running

    def start(self) -> None:
        """Run the plant clock in its own thread (keeps the web server responsive)."""
        if self._thread is not None:
            return
        self._stop = False

        def loop():
            last = time.perf_counter()
            while not self._stop:
                time.sleep(0.1)
                now = time.perf_counter()
                dt, last = now - last, now
                try:
                    self.advance(dt)
                except Exception as ex:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    self._alert("warning", "engine", f"simulation error: {ex}")
                    self.running = False
        self._thread = threading.Thread(target=loop, name="plant-clock", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True

    def advance(self, wall_seconds: float) -> int:
        """Advance the simulation by speed * wall_seconds (sim seconds); returns emulator steps run."""
        if not self.running or self.holding():
            return 0
        sim_h = self.speed * wall_seconds / 3600.0
        self._acc = getattr(self, "_acc", 0.0) + sim_h
        n = 0
        self._acc = min(self._acc, 6 * self.sc.dt_emu)          # never try to catch up more than 1 h
        if self.t >= self.sc.start_hour + self.sc.end_hour - 1e-9:
            self.running = False
            if not getattr(self, "_ended", False):
                self._ended = True
                self._timeline("event", "Scenario complete",
                               "The scripted showcase has ended. Both worlds can be continued with Play.")
            return 0
        while self._acc >= self.sc.dt_emu - 1e-12 and n < 6:
            self._acc -= self.sc.dt_emu
            self.step()
            n += 1
            if self.holding():
                self._acc = 0.0
                break
        return n

    def run_headless(self, hours: float, progress: Optional[Callable] = None) -> None:
        """Run for a number of sim hours synchronously (tests, CLI, benchmarks)."""
        end = self.t + hours
        if self.planner.plan is None:
            self.request_replan("initial plan")
            self._wait_plan()
        while self.t < end - 1e-9:
            self.step()
            if not self.sync_plans:
                self._wait_plan()
            if progress and self.steps % 6 == 0:
                progress(self)

    def _wait_plan(self) -> None:
        while self._future is not None and not self._future.done():
            time.sleep(0.05)

    # ------------------------------------------------------------------ views
    def _kpi_point(self) -> dict:
        a, b = self.plants["plyxis"].kpi, self.plants["legacy"].kpi
        return {"t": self.t,
                "p": {k: round(v, 2) for k, v in (a.smooth or a.rate).items()},
                "l": {k: round(v, 2) for k, v in (b.smooth or b.rate).items()},
                "pc": round(a.cum["profit"], 0), "lc": round(b.cum["profit"], 0)}

    def kpis(self) -> dict:
        a, b = self.plants["plyxis"].kpi.now(), self.plants["legacy"].kpi.now()
        ca, cb = a["cum"], b["cum"]
        hrs = max(ca["hours"], 1e-9)
        cprice = float(self.twin.model.program.E0.get("carbon_price", 0.0)) * 1000.0     # USD/t
        delta = {
            "profit": ca["profit"] - cb["profit"],
            "profit_after_carbon": ca.get("profit_after_carbon", 0.0) - cb.get("profit_after_carbon", 0.0),
            "oil_bbl": ca["oil_bbl"] - cb["oil_bbl"],
            "flared_mscf": ca["flared_mscf"] - cb["flared_mscf"],
            "emission_t": (ca["emission_kg"] - cb["emission_kg"]) / 1000.0,
            **{k: ca[k] - cb.get(k, 0.0) for k in ca if k.startswith("m_")},
        }
        return {"plyxis": a, "legacy": b, "delta": delta, "hours": hrs,
                "value_per_day": delta["profit"] / hrs * 24.0, "carbon_price_usd_t": cprice,
                "bridge": self._bridge(ca, cb)}

    BRIDGE_LABELS = {"crude": "Oil sales", "gas": "Gas sales", "power": "Power", "pumping": "Water injection power",
                     "hauling": "Trucking", "disposal": "Disposal fees", "treating": "Treating",
                     "lease": "Lease operating cost", "restarts": "Well restarts", "starts": "Compressor starts",
                     "standby": "Compressor standby", "unplanned_failures": "Unplanned pump failures",
                     "pump_replacement": "Pump replacements", "inventory_value": "Inventories (tanks, bins, stockpiles)",
                     "equipment_value": "Pump life consumed", "metal": "Copper sales", "mining": "Mining consumables",
                     "loading": "Loader operating cost", "blasting": "Explosives and charging",
                     "ownership": "Fleet ownership", "labour": "Labour", "overhead": "G&A and camp",
                     "energy": "Mill power", "consumables": "Mill consumables", "wear": "Crusher wear",
                     "overflow": "Sump overflow (clean-up)"}

    def _bridge(self, ca: dict, cb: dict) -> list:
        """Operating-profit difference split into realised terms (sums exactly to delta['profit'])."""
        ka, kb = self.plants["plyxis"].kpi, self.plants["legacy"].kpi
        rows = [(n, ka.terms.get(n, 0.0) - kb.terms.get(n, 0.0)) for n in set(ka.terms) | set(kb.terms)
                if n not in ka.EXPECTATION_TERMS]
        rows += [(n, ca[n] - cb[n]) for n in ("inventory_value", "equipment_value")]
        return sorted(({"key": n, "label": self.BRIDGE_LABELS.get(n, n.replace("_", " ")), "delta": round(d)}
                       for n, d in rows if abs(d) >= 1.0), key=lambda r: -abs(r["delta"]))

    def snapshot(self, full: bool = False) -> dict:
        plan = self.planner.plan
        trucks = []
        if plan is not None:
            yard = None
            for inst in self.twin.model.flat.instances:
                if inst.type.short == "truck_fleet":
                    yard = inst.at
            trucks = DSP.truck_positions(plan.trips.get("trips", []), self.t, plan.trips.get("trucks", 0), yard)
        assets = {}
        for tag, (v, t) in self.latest.items():
            info = self.twin.tags.get(tag)
            if info is None:
                continue
            assets.setdefault(info.path, {})[tag.split(".", 2)[-1] if tag.count(".") >= 2 else tag] = round(v, 3)
        jobs = [{**j["job"], "status": j["status"]} for j in self.jobs.values()]
        snap = {"type": "snapshot", "t": self.t, "clock": self.clock_str(), "running": self.running,
                "holding": self.holding(),
                "speed": self.speed, "kpis": self.kpis(), "trucks": trucks, "assets": assets,
                "optimizer": {"running": self.planner.running, "plan_id": plan.pid if plan else None,
                              "last_plan_t": self.last_plan_t, "neural": self.planner.net_info,
                              "progress_tail": [_jsonable(r) for r in self.planner.progress[-3:]]},
                "twin": {"version": self.twin.version, "commits": len(self.twin.repo.commits)},
                "jobs": jobs, "prices": {k: round(self.prices.display(k, self.t), 4) for k in self.sc.prices},
                "alerts_tail": list(self.alerts)[-6:]}
        if full:
            snap["kpi_history"] = self.kpi_history[-1500:]
            snap["timeline"] = self.timeline[-200:]
            snap["alerts"] = list(self.alerts)[-200:]
        return snap

    def explain(self, path: str) -> dict:
        plan = self.planner.plan
        if plan is None:
            return {"error": "no plan yet"}
        pb = plan.problem
        iid = pb.prog.flat.by_path.get(path)
        if iid is None:
            return {"error": f"unknown asset {path}"}
        with self._lock:
            out = explain_instance(pb, plan.x, plan.multipliers, iid)
        out["t0"] = plan.t0
        out["plan_id"] = plan.pid
        return out


def _calib_item(r) -> dict:
    path, name, old, new, how = r
    return {"asset": path, "property": name, "old": old, "new": new, "how": how,
            "change_pct": round(100.0 * (new - old) / old, 1) if abs(old) > 1e-12 else None}


def _jsonable(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [x if isinstance(x, (int, float, str, bool)) else str(x) for x in v][:50]
        else:
            out[k] = str(v)
    return out
