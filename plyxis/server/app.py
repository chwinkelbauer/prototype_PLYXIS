"""PLYXIS server: REST + WebSocket API and the static operations console.

    python -m plyxis serve            (http://localhost:8000)
"""
from __future__ import annotations

import asyncio
import contextlib
import glob
import json
import math
import os
import time
import traceback
from typing import Optional

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from ..runtime import report, views, whatif
from ..runtime.engine import Engine

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEB_DIST = os.path.join(ROOT, "web", "dist")


def create_app(scenario_dir: Optional[str] = None, speed: float = 600.0, autoplay: bool = False,
               use_neural: bool = True) -> FastAPI:
    scenario_dir = scenario_dir or os.path.join(ROOT, "scenarios", "kestrel")
    @contextlib.asynccontextmanager
    async def lifespan(_app):
        state["loop"] = asyncio.get_running_loop()
        asyncio.create_task(boot())
        asyncio.create_task(ticker())
        yield
        if state.get("engine") is not None:
            try:
                state["engine"].stop()
            except Exception:  # noqa: BLE001
                pass

    app = FastAPI(title="PLYXIS", version="0.2", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    state = {"engine": None, "clients": set(), "loop": None, "booting": True, "boot_msg": "starting",
             "boot_error": None, "scenario_dir": scenario_dir}

    def eng() -> Engine:
        e = state["engine"]
        if e is None:
            raise HTTPException(503, f"engine starting: {state['boot_msg']}")
        return e

    # ------------------------------------------------------------------ push channel
    def push(msg: dict) -> None:
        loop = state["loop"]
        if loop is None:
            return
        data = _dumps(msg)
        for q in list(state["clients"]):
            try:
                loop.call_soon_threadsafe(_put, q, data)
            except RuntimeError:
                pass

    async def ticker():
        last_snap = 0.0
        failed = None
        while True:
            await asyncio.sleep(0.15)
            e = state["engine"]
            if e is None or state["booting"]:     # a snapshot would take the console off its starting screen
                continue
            now = time.perf_counter()
            if now - last_snap > 0.5:
                last_snap = now
                if state["clients"]:
                    try:
                        push(e.snapshot())
                        failed = None
                    except Exception as ex:  # noqa: BLE001  one bad tick must not end the push channel for good
                        if repr(ex) != failed:
                            failed = repr(ex)
                            traceback.print_exc()

    def boot_status(message: str) -> None:
        state["boot_msg"] = message
        push({"type": "booting", "message": message})

    async def boot():
        loop = asyncio.get_running_loop()
        sdir = state["scenario_dir"]
        e = None
        try:
            boot_status("compiling twin, emulators and loading neural sub-solvers")
            e = await loop.run_in_executor(None, lambda: Engine(sdir, speed=speed, use_neural=use_neural, audit_path=_audit_path(sdir)))
            e.subscribe(push)
            state["engine"] = e
            boot_status("computing initial plan")
            await loop.run_in_executor(None, lambda: e._replan("initial plan"))
            if e.planner.plan is None:            # _replan logs and survives: a later re-plan may fail, the first may not
                why = next((a["message"] for a in reversed(e.alerts) if a["kind"] == "optimizer"), "no plan")
                raise RuntimeError(f"the first plan failed: {why}")
            e.start()
            if autoplay:
                e.running = True
        except Exception as ex:  # noqa: BLE001  nobody awaits this task: unreported, the console would wait for ever
            traceback.print_exc()
            if e is not None:
                e.unsubscribe(push)
                e.stop()
            state["engine"], state["booting"], state["boot_msg"] = None, False, "failed"
            state["boot_error"] = f"{type(ex).__name__}: {ex}"
            push({"type": "boot_error", "message": state["boot_error"]})
            return
        state["booting"] = False
        state["boot_msg"] = "ready"
        push({"type": "ready"})

    # ------------------------------------------------------------------ meta / structure
    @app.get("/api/health")
    def health():
        return {"ok": state["boot_error"] is None, "booting": state["booting"], "message": state["boot_msg"],
                "error": state["boot_error"], "scenario": os.path.basename(os.path.abspath(state["scenario_dir"]))}

    @app.get("/api/scenarios")
    def scenarios():
        out = []
        for f in sorted(glob.glob(os.path.join(ROOT, "scenarios", "*", "scenario.json"))):
            d = os.path.dirname(f)
            with open(f, encoding="utf-8") as fh:
                raw = json.load(fh)
            out.append({"id": os.path.basename(d), "name": raw.get("name"), "title": raw.get("title", raw.get("name")),
                        "region": raw.get("region", ""), "subtitle": raw.get("subtitle", ""),
                        "active": os.path.abspath(d) == os.path.abspath(state["scenario_dir"])})
        return out

    @app.post("/api/scenario")
    async def select_scenario(body: dict = Body(...)):
        d = os.path.join(ROOT, "scenarios", os.path.basename(str(body.get("id", ""))))
        if not os.path.exists(os.path.join(d, "scenario.json")):
            raise HTTPException(404, "unknown scenario")
        if state["booting"]:
            raise HTTPException(409, "a scenario is still starting")
        old, state["engine"] = state["engine"], None
        state["booting"], state["boot_msg"], state["scenario_dir"] = True, "switching scenario", d
        state["boot_error"] = None
        if old is not None:                       # the old engine must not push into the new session
            old.unsubscribe(push)
            old.running = False
            await asyncio.get_running_loop().run_in_executor(None, old.stop)
            for ex in (old.executor, old.est_executor):
                ex.shutdown(wait=False, cancel_futures=True)
        push({"type": "booting", "message": "switching scenario"})
        asyncio.create_task(boot())
        return {"ok": True, "id": os.path.basename(d)}

    @app.get("/api/meta")
    def meta():
        e = eng()
        sc = e.sc
        return {"id": os.path.basename(sc.root), "title": sc.raw.get("title", sc.name), "region": sc.raw.get("region", ""),
                "subtitle": sc.raw.get("subtitle", ""), "kpi": sc.raw.get("kpi"),
                "name": sc.name, "description": sc.description, "start_hour": sc.start_hour,
                "events": sc.events, "demo": sc.demo, "speed": e.speed, "dt_emulator_h": sc.dt_emu,
                "replan_every_h": sc.replan_every, "prices": {k: {"unit": v.get("unit")} for k, v in sc.prices.items()}}

    @app.get("/api/structure")
    def structure():
        return views.structure(eng())

    @app.get("/api/snapshot")
    def snapshot(full: int = 0):
        return _plain(eng().snapshot(full=bool(full)))

    # ------------------------------------------------------------------ control
    @app.post("/api/control")
    def control(body: dict = Body(...)):
        e = eng()
        action = body.get("action")
        if action == "play":
            if e.t >= e.sc.start_hour + e.sc.end_hour - 1e-9:
                e.sc.end_hour = 1e9              # continue beyond the scripted scenario on request
            e.running = True
        elif action == "pause":
            e.running = False
        elif action == "speed":
            e.speed = float(body.get("value", 600))
        elif action == "reset":
            e.stop()
            e._thread = None
            e.reset()
            e.subscribe(push)
            e._replan("initial plan")
            e.start()
        elif action == "step":
            for _ in range(int(body.get("value", 1))):
                e.step()
        elif action == "seek":
            target = e.sc.start_hour + float(body.get("value", 0))
            was = e.running
            e.running = False
            while e.t < target - 1e-9:
                e.step()
                e._wait_plan()
            e.running = was
        elif action == "hold":
            e.hold_on_disruption = bool(body.get("value", 1))
        return {"running": e.running, "speed": e.speed, "t": e.t, "clock": e.clock_str()}

    @app.post("/api/replan")
    def replan(body: dict = Body(default={})):
        eng().request_replan(body.get("trigger", "manual request"))
        return {"ok": True}

    @app.post("/api/event")
    def event(body: dict = Body(...)):
        return eng().inject(body)

    # ------------------------------------------------------------------ plan / schedule / explain
    @app.get("/api/plan")
    def plan():
        e = eng()
        p = e.planner.plan
        if p is None:
            return {"plan": None}
        return _plain({"plan": p.summary(), "lineups": p.lineups[:400],
                                      "trips": p.trips.get("trips", [])[:400],
                                      "jobs": [{**v["job"], "status": v["status"]} for v in e.jobs.values()],
                                      "history": list(e.plan_events)})

    @app.get("/api/plan/gantt")
    def gantt():
        return _plain(views.gantt(eng()))

    @app.get("/api/plan/instructions")
    def instructions(asset: Optional[str] = None):
        p = eng().planner.plan
        if p is None:
            return []
        rows = p.instructions if asset is None else [i for i in p.instructions if i["asset"] == asset]
        return rows[:3000]

    @app.get("/api/explain/{path:path}")
    def explain(path: str):
        return _plain(eng().explain(path))

    @app.get("/api/asset/{path:path}")
    def asset(path: str):
        return _plain(views.asset_detail(eng(), path))

    # ------------------------------------------------------------------ optimizer
    @app.get("/api/optimizer")
    def optimizer():
        e = eng()
        pl = e.planner
        runs = list(pl.runs)
        sp = pl.base_pb.space if pl.base_pb is not None else None
        secs = sorted(r["seconds"] for r in runs)
        sla = {"plans": len(secs), "p50_s": secs[len(secs) // 2], "p95_s": secs[min(len(secs) - 1, int(0.95 * len(secs)))],
               "max_s": secs[-1], "replan_every_h": e.sc.replan_every} if secs else None
        return _plain({
            "running": pl.running, "progress": pl.progress[-400:], "runs": runs[-12:], "sla": sla,
            "space": sp.describe() if sp else None, "neural": pl.net_info, "learn": e.learn_log[-20:],
            "subsystems": pl.part.describe(e.twin.model.flat) if pl.base_pb is not None else [],
            "bottlenecks": pl.plan.bottlenecks if pl.plan else [],
        })

    # ------------------------------------------------------------------ twin studio
    @app.get("/api/twin/files")
    def twin_files():
        e = eng()
        return {"files": e.twin.sources(), "version": e.twin.version, "env": e.twin.env,
                "stdlib": _stdlib_sources()}

    @app.get("/api/twin/generated")
    def twin_generated():
        p = eng().twin.model.program
        return {"simulate": p.source, "bounds": p.bounds_source, "summary": p.summary()}

    @app.post("/api/twin/check")
    def twin_check(body: dict = Body(...)):
        e = eng()
        files = dict(e.twin.sources())
        files.update(body.get("files", {}))
        return {"diagnostics": e.twin.check(files)}

    @app.get("/api/twin/lint")
    def twin_lint():
        from ..lang.lint import as_dicts, lint
        return as_dicts(lint(eng().twin.model))

    @app.post("/api/twin/apply")
    def twin_apply(body: dict = Body(...)):
        return eng().edit_twin(body.get("files", {}), body.get("message", "edit"), body.get("author", "engineer"))

    @app.get("/api/twin/commits")
    def twin_commits():
        return eng().twin.repo.log(200)

    @app.get("/api/twin/commit/{cid}")
    def twin_commit(cid: str):
        c = eng().twin.repo.get(cid)
        if c is None:
            raise HTTPException(404)
        return c.to_dict(with_diff=True)

    # ------------------------------------------------------------------ operating modes, decisions, handover
    @app.get("/api/advisory")
    def advisory():
        return _plain(eng().advisory.view())

    @app.post("/api/advisory/mode")
    def advisory_mode(body: dict = Body(...)):
        e = eng()
        with e._lock:                    # never while the plant clock is stepping
            return e.advisory.set_mode(str(body.get("mode", "")), str(body.get("actor", "operator")))

    @app.post("/api/advisory/decide")
    def advisory_decide(body: dict = Body(...)):
        e = eng()
        with e._lock:
            return e.advisory.decide(body.get("ids", []), str(body.get("action", "")), str(body.get("actor", "operator")),
                                     str(body.get("reason", "")), body.get("value"))

    @app.get("/api/audit")
    def audit(limit: int = 300, kind: Optional[str] = None):
        return _plain(eng().advisory.audit.tail(limit, kind))

    @app.get("/api/workorders.csv")
    def workorders():
        return PlainTextResponse(report.work_orders_csv(eng()), media_type="text/csv",
                                 headers={"Content-Disposition": "attachment; filename=plyxis-work-orders.csv"})

    @app.get("/api/report/shift")
    def report_shift(hours: float = 12.0, format: str = "json"):
        r = report.shift_report(eng(), hours)
        if format == "md":
            return PlainTextResponse(report.to_markdown(r), media_type="text/markdown")
        return _plain(r)

    @app.post("/api/whatif")
    def what_if(body: dict = Body(...)):
        return _plain(whatif.run(eng(), body.get("changes", []), str(body.get("name", "what-if"))))

    # ------------------------------------------------------------------ data integration
    @app.get("/api/data/connectors")
    def connectors():
        e = eng()
        return [c.info() for c in e.connectors]

    @app.get("/api/data/tags")
    def tags(prefix: str = "", limit: int = 400):
        e = eng()
        out = []
        for tag, (v, t) in sorted(e.latest.items()):
            if prefix and not tag.startswith(prefix):
                continue
            info = e.twin.tags.get(tag)
            out.append({"tag": tag, "value": v, "t": t, "mapped": info is not None or tag.startswith("MKT."),
                        "asset": info.path if info else None, "signal": info.key if info else None,
                        "unit": info.unit if info else None})
            if len(out) >= limit:
                break
        return out

    @app.get("/api/data/tag")
    def tag_history(tag: str):
        h = eng().tag_history.get(tag)
        return {"tag": tag, "history": [list(p) for p in h] if h else []}

    @app.get("/api/data/quality")
    def data_quality():
        return _plain(eng().quality.summary())

    @app.get("/api/data/calibration")
    def calibration():
        e = eng()
        est = e.twin.estimator
        uncertainty = []
        for gid, pname in sorted(est.direct):
            g = e.twin.model.program.gmap[gid]
            if pname in g.ir.type.params and g.ir.type.params[pname].calibrate is not None:
                u = est.uncertainty(gid, pname)
                uncertainty.append({"property": f"{g.ir.type.short}.{pname}", "median_pct": round(float(u.median()) * 100, 1),
                                    "best_pct": round(float(u.min()) * 100, 1), "worst_pct": round(float(u.max()) * 100, 1),
                                    "sampled": sum(1 for k in est.test_var if k[1] == pname), "instances": g.n})
        return _plain({"log": e.calib_log[-50:], "direct": sorted(f"{a}.{b}" for a, b in e.twin.estimator.direct),
                       "uncertainty": uncertainty, "margins": e.margin_log[-1:],
                                      "calibrated": [f"{e.twin.model.program.gmap[g].ir.type.short}.{p}"
                                                     for g, p, _, _ in e.twin.estimator.calib]})

    @app.get("/api/data/discovery")
    def discovery():
        return [p.to_dict() for p in eng().twin.discovery.proposals.values()]

    @app.post("/api/data/discovery/{pid}/{action}")
    def discovery_action(pid: str, action: str):
        e = eng()
        if action == "approve":
            return e.apply_proposal(pid)
        if action == "reject":
            return e.reject_proposal(pid)
        raise HTTPException(400, "action must be approve or reject")

    @app.post("/api/ingest")
    def ingest(body: dict = Body(...)):
        e = eng()
        n = e.http.submit(body.get("records", []), e.t)
        return {"accepted": n}

    @app.get("/api/alerts")
    def alerts():
        return list(eng().alerts)[-300:]

    @app.get("/api/timeline")
    def timeline():
        return eng().timeline[-400:]

    # ------------------------------------------------------------------ benchmarks
    @app.get("/api/bench")
    def bench():
        out = {}
        for f in glob.glob(os.path.join(ROOT, "bench", "results", "*.json")):
            name = os.path.splitext(os.path.basename(f))[0]
            try:
                with open(f) as fh:
                    d = json.load(fh)
                if name.startswith("showcase_run"):
                    d = {k: d[k] for k in ("hours", "kpis", "calibration_error", "runs", "bridge", "skipped_events", "seed")
                         if k in d}
                out[name] = d
            except Exception:  # noqa: BLE001
                pass
        return out

    # ------------------------------------------------------------------ websocket
    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        state["clients"].add(q)
        try:
            if state["boot_error"]:
                first = {"type": "boot_error", "message": state["boot_error"]}
            elif state["booting"] or state["engine"] is None:
                first = {"type": "booting", "message": state["boot_msg"]}
            else:
                first = state["engine"].snapshot(full=True)
            await socket.send_text(_dumps(first))
            while True:
                data = await q.get()
                await socket.send_text(data)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            state["clients"].discard(q)

    # ------------------------------------------------------------------ static web app
    if os.path.isdir(WEB_DIST):
        app.mount("/assets", StaticFiles(directory=os.path.join(WEB_DIST, "assets")), name="assets")

        @app.get("/{full_path:path}")
        def spa(full_path: str):
            f = os.path.join(WEB_DIST, full_path)
            if full_path and os.path.isfile(f):
                return FileResponse(f)
            return FileResponse(os.path.join(WEB_DIST, "index.html"))
    else:
        @app.get("/")
        def no_web():
            return JSONResponse({"message": "PLYXIS API running. Build the web console: cd web && npm install && "
                                            "npm run build"})
    return app


def _audit_path(sdir: str) -> str:
    """One audit file per console session and case study (runs/ is not in version control)."""
    name = f"{os.path.basename(os.path.abspath(sdir))}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    return os.path.join(ROOT, "runs", "audit", name)


def _put(q: asyncio.Queue, data: str) -> None:
    if q.full():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    q.put_nowait(data)


def _default(o):
    try:
        import torch
        if isinstance(o, torch.Tensor):
            return o.tolist()
    except Exception:  # noqa: BLE001
        pass
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _finite(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, list):
        return [_finite(v) for v in o]
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    return o


def _plain(msg):
    """JSON-compatible copy of a payload; NaN and infinities become null."""
    return _finite(json.loads(json.dumps(msg, default=_default)))


def _dumps(msg) -> str:
    """Strict JSON: the browser's JSON.parse rejects a whole message for one NaN token, so a single bad
    value would silently stop every snapshot."""
    try:
        return json.dumps(msg, default=_default, allow_nan=False)
    except ValueError:
        return json.dumps(_plain(msg), allow_nan=False)


def _stdlib_sources() -> dict:
    from ..lang import stdlib_files
    out = {}
    for f in stdlib_files():
        with open(f, encoding="utf-8") as fh:
            out[os.path.basename(f)] = fh.read()
    return out
