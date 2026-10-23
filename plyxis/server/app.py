"""PLYXIS server: REST + WebSocket API and the static operations console.

    python -m plyxis serve            (http://localhost:8000)
"""
from __future__ import annotations

import asyncio
import contextlib
import glob
import json
import os
import time
from typing import Optional

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..runtime import views
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
    state = {"engine": None, "clients": set(), "loop": None, "booting": True, "boot_msg": "starting"}

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
        data = json.dumps(msg, default=_default)
        for q in list(state["clients"]):
            try:
                loop.call_soon_threadsafe(_put, q, data)
            except RuntimeError:
                pass

    async def ticker():
        last_snap = 0.0
        last = time.perf_counter()
        while True:
            await asyncio.sleep(0.15)
            e = state["engine"]
            if e is None:
                continue
            now = time.perf_counter()
            if now - last_snap > 0.5:
                last_snap = now
                if state["clients"]:
                    push(e.snapshot())

    async def boot():
        loop = asyncio.get_running_loop()
        state["boot_msg"] = "compiling twin, emulators and loading neural sub-solvers"
        e = await loop.run_in_executor(None, lambda: Engine(scenario_dir, speed=speed, use_neural=use_neural))
        e.subscribe(push)
        state["engine"] = e
        state["boot_msg"] = "computing initial plan"
        await loop.run_in_executor(None, lambda: e._replan("initial plan"))
        state["booting"] = False
        state["boot_msg"] = "ready"
        e.start()
        if autoplay:
            e.running = True
        push({"type": "ready"})


    # ------------------------------------------------------------------ meta / structure
    @app.get("/api/health")
    def health():
        return {"ok": True, "booting": state["booting"], "message": state["boot_msg"]}

    @app.get("/api/meta")
    def meta():
        e = eng()
        sc = e.sc
        return {"name": sc.name, "description": sc.description, "start_hour": sc.start_hour,
                "events": sc.events, "demo": sc.demo, "speed": e.speed, "dt_emulator_h": sc.dt_emu,
                "replan_every_h": sc.replan_every, "prices": {k: {"unit": v.get("unit")} for k, v in sc.prices.items()}}

    @app.get("/api/structure")
    def structure():
        return views.structure(eng())

    @app.get("/api/snapshot")
    def snapshot(full: int = 0):
        return json.loads(json.dumps(eng().snapshot(full=bool(full)), default=_default))

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
        return json.loads(json.dumps({"plan": p.summary(), "lineups": p.lineups[:400],
                                      "trips": p.trips.get("trips", [])[:400],
                                      "jobs": [{**v["job"], "status": v["status"]} for v in e.jobs.values()],
                                      "history": list(e.plan_events)}, default=_default))

    @app.get("/api/plan/gantt")
    def gantt():
        return json.loads(json.dumps(views.gantt(eng()), default=_default))

    @app.get("/api/plan/instructions")
    def instructions(asset: Optional[str] = None):
        p = eng().planner.plan
        if p is None:
            return []
        rows = p.instructions if asset is None else [i for i in p.instructions if i["asset"] == asset]
        return rows[:3000]

    @app.get("/api/explain/{path:path}")
    def explain(path: str):
        return json.loads(json.dumps(eng().explain(path), default=_default))

    @app.get("/api/asset/{path:path}")
    def asset(path: str):
        return json.loads(json.dumps(views.asset_detail(eng(), path), default=_default))

    # ------------------------------------------------------------------ optimizer
    @app.get("/api/optimizer")
    def optimizer():
        e = eng()
        pl = e.planner
        runs = list(pl.runs)
        sp = pl.base_pb.space if pl.base_pb is not None else None
        return json.loads(json.dumps({
            "running": pl.running, "progress": pl.progress[-400:], "runs": runs[-12:],
            "space": sp.describe() if sp else None, "neural": pl.net_info, "learn": e.learn_log[-20:],
            "subsystems": pl.part.describe(e.twin.model.flat) if pl.base_pb is not None else [],
            "bottlenecks": pl.plan.bottlenecks if pl.plan else [],
        }, default=_default))

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

    @app.get("/api/data/calibration")
    def calibration():
        e = eng()
        return json.loads(json.dumps({"log": e.calib_log[-50:], "direct": sorted(f"{a}.{b}" for a, b in e.twin.estimator.direct),
                                      "calibrated": [f"{e.twin.model.program.gmap[g].ir.type.short}.{p}"
                                                     for g, p, _, _ in e.twin.estimator.calib]}, default=_default))

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
                    d = {k: d[k] for k in ("hours", "kpis", "calibration_error", "runs") if k in d}
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
            if state["engine"] is not None:
                await socket.send_text(json.dumps(state["engine"].snapshot(full=True), default=_default))
            else:
                await socket.send_text(json.dumps({"type": "booting", "message": state["boot_msg"]}))
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


def _stdlib_sources() -> dict:
    from ..lang import stdlib_files
    out = {}
    for f in stdlib_files():
        with open(f, encoding="utf-8") as fh:
            out[os.path.basename(f)] = fh.read()
    return out
