"""Live engine and server API (end to end, short horizon)."""
import pytest

from conftest import KESTREL


@pytest.fixture(scope="module")
def engine():
    from plyxis.optim.cascade import CascadeConfig
    from plyxis.runtime.engine import Engine
    eng = Engine(KESTREL, sync_plans=True, use_neural=False, online_learning=False)
    eng.planner.cfg = CascadeConfig(relax_iters=50, population=4, generations=1, repair_iters=25, polish_iters=10,
                                    use_neural=False, multistart=2)
    eng.run_headless(2.0)
    return eng


def test_engine_runs_both_worlds(engine):
    k = engine.kpis()
    assert k["plyxis"]["cum"]["oil_bbl"] > 0 and k["legacy"]["cum"]["oil_bbl"] > 0
    assert engine.planner.plan is not None and engine.planner.plan.feasible in (True, False)
    assert engine.steps == 12
    snap = engine.snapshot(full=True)
    assert snap["assets"] and "kpi_history" in snap


def test_engine_reacts_to_injected_trip(engine):
    n_plans = len(engine.plan_events)
    engine.inject({"kind": "trip", "asset": "CS1", "units": 1, "title": "test trip"})
    engine.run_headless(0.5)
    assert len(engine.plan_events) > n_plans                       # disruption -> immediate re-plan
    assert any(a.get("kind") == "status" for a in engine.alerts)


def test_explanation_endpoint_logic(engine):
    ex = engine.explain("P01_W01")
    assert "decisions" in ex and ex["decisions"]


def test_server_api(engine, monkeypatch):
    from fastapi.testclient import TestClient
    from plyxis.server import app as appmod
    app = appmod.create_app()
    # inject the already running engine instead of booting a new one
    with TestClient(app) as client:
        import time
        for _ in range(200):
            if client.get("/api/health").json()["booting"] is False:
                break
            time.sleep(0.5)
        assert client.get("/api/meta").status_code == 200
        st = client.get("/api/structure").json()
        assert len(st["assets"]) > 150 and st["library"]
        assert client.get("/api/plan/gantt").json()["rows"]
        files = client.get("/api/twin/files").json()["files"]
        assert "field.plx" in files and "live.plx" in files
        diags = client.post("/api/twin/check", json={"files": {"field.plx": files["field.plx"] + "\nthis is not plx"}}).json()
        assert any(d["severity"] == "error" for d in diags["diagnostics"])
