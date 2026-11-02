"""Console start-up: whatever goes wrong must reach the browser, never leave it on the starting screen."""
import json
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from plyxis.server import app as appmod


def strict(text: str) -> dict:
    """Parse like the browser's JSON.parse: NaN and Infinity tokens are errors."""
    def reject(tok):
        raise ValueError(f"invalid JSON token {tok}")
    return json.loads(text, parse_constant=reject)


class FakeEngine:
    """Stands in for the twin so these tests exercise the server, not a real boot."""
    fail_init = None
    plan = "plan"
    plan_gate = None
    bad_snapshots = 0
    calls = 0

    def __init__(self, sdir, speed=600.0, use_neural=True, **kw):
        if self.fail_init:
            raise self.fail_init
        self.planner = SimpleNamespace(plan=None)
        self.alerts = [{"kind": "optimizer", "message": "re-plan failed: singular"}]
        self.running = False

    def subscribe(self, fn):
        pass

    def unsubscribe(self, fn):
        pass

    def _replan(self, trigger):
        if FakeEngine.plan_gate is not None:
            FakeEngine.plan_gate.wait(20)
        self.planner.plan = FakeEngine.plan

    def start(self):
        pass

    def stop(self):
        pass

    def snapshot(self, full=False):
        FakeEngine.calls += 1
        if FakeEngine.bad_snapshots > 0:
            FakeEngine.bad_snapshots -= 1
            raise RuntimeError("dictionary changed size during iteration")
        return {"type": "snapshot", "t": 6.0, "rate": float("nan"), "values": [1.0, float("inf")]}


@pytest.fixture
def fake(monkeypatch):
    for k, v in {"fail_init": None, "plan": "plan", "plan_gate": None, "bad_snapshots": 0, "calls": 0}.items():
        setattr(FakeEngine, k, v)
    monkeypatch.setattr(appmod, "Engine", FakeEngine)
    return FakeEngine


def settled(client) -> dict:
    for _ in range(100):
        h = client.get("/api/health").json()
        if not h["booting"]:
            return h
        time.sleep(0.05)
    raise AssertionError("server still booting")


def test_engine_failure_reaches_the_console(fake):
    fake.fail_init = ValueError("cannot compile field.plx")
    with TestClient(appmod.create_app()) as client:
        h = settled(client)
        assert h["ok"] is False and "cannot compile field.plx" in h["error"]
        with client.websocket_connect("/ws") as ws:
            msg = strict(ws.receive_text())
        assert msg["type"] == "boot_error" and "cannot compile field.plx" in msg["message"]


def test_failed_first_plan_is_a_failed_start(fake):
    fake.plan = None
    with TestClient(appmod.create_app()) as client:
        h = settled(client)
        assert h["ok"] is False and "the first plan failed: re-plan failed: singular" in h["error"]
        assert client.get("/api/meta").status_code == 503


def test_retry_after_a_failed_start(fake):
    fake.fail_init = ValueError("broken")
    with TestClient(appmod.create_app()) as client:
        assert settled(client)["ok"] is False
        fake.fail_init = None
        assert client.post("/api/scenario", json={"id": "kestrel"}).status_code == 200
        h = settled(client)
        assert h["ok"] is True and h["error"] is None and h["message"] == "ready"


def test_no_snapshot_before_the_first_plan(fake):
    fake.plan_gate = threading.Event()
    with TestClient(appmod.create_app()) as client:
        with client.websocket_connect("/ws") as ws:
            assert strict(ws.receive_text())["type"] == "booting"
            time.sleep(0.8)                   # several ticker periods with the engine up and the plan running
            fake.plan_gate.set()
            seen = []
            while "ready" not in seen:
                seen.append(strict(ws.receive_text())["type"])
            assert "snapshot" not in seen
            assert strict(ws.receive_text())["type"] == "snapshot"


def test_ticker_survives_a_failing_snapshot(fake):
    with TestClient(appmod.create_app()) as client:
        settled(client)
        with client.websocket_connect("/ws") as ws:
            assert strict(ws.receive_text())["type"] == "snapshot"
            start = fake.calls
            fake.bad_snapshots = 2
            deadline = time.time() + 5
            while fake.calls < start + 3 and time.time() < deadline:
                time.sleep(0.05)
            assert fake.calls >= start + 3    # two failures and then another attempt: still ticking


def test_payloads_are_strict_json():
    msg = {"a": float("nan"), "b": [1.5, float("inf")], "c": {"d": -float("inf"), "e": 2}}
    want = {"a": None, "b": [1.5, None], "c": {"d": None, "e": 2}}
    assert strict(appmod._dumps(msg)) == want
    assert appmod._plain(msg) == want
    assert strict(appmod._dumps({"ok": 1.0})) == {"ok": 1.0}
