"""Scenario mechanics the A/B comparison depends on: event semantics, notices to both control rooms,
forecasts around announced transients, failure threshold, value bridge."""
import pytest

from conftest import KESTREL


@pytest.fixture(scope="module")
def eng():
    from plyxis.optim.cascade import CascadeConfig
    from plyxis.runtime.engine import Engine
    e = Engine(KESTREL, sync_plans=True, use_neural=False, online_learning=False)
    e.planner.cfg = CascadeConfig(relax_iters=40, population=4, generations=1, repair_iters=20, polish_iters=10,
                                  use_neural=False, multistart=2)
    e.run_headless(1.0)
    return e


def _event(eng, kind, **match):
    return next(ev for ev in eng.sc.events if ev["kind"] == kind and all(ev.get(k) == v for k, v in match.items()))


def test_bridge_sums_to_profit_difference(eng):
    k = eng.kpis()
    assert k["bridge"], "both worlds diverge within the first hour"
    total = sum(r["delta"] for r in k["bridge"])
    assert abs(total - k["delta"]["profit"]) <= 0.5 * len(k["bridge"]) + 1.0
    assert not {"risk", "flare_risk", "pump_wear"} & {r["key"] for r in k["bridge"]}   # planning-only terms


def test_failure_threshold_hits_one_pump(eng):
    pp = eng.plants["legacy"]
    g, li = pp.path_index["P01_W08"]
    pp.S[g]["health"].reshape(-1)[li] = 0.05
    assert [e["asset"] for e in pp._failures()] == ["P01_W08"]


def test_price_event_lasts_its_hours_and_notices_reach_manual_ops(eng):
    eng._apply_event(_event(eng, "price", name="power_price"))
    for plant in eng.plants.values():
        assert plant.overrides["power_price"][1] == pytest.approx(eng.t + 2.0)
    eng._apply_event(_event(eng, "announce", name="power_price"))
    s, e = eng.legacy_ops.dr_windows[-1]
    assert e - s == pytest.approx(2.0)
    pp = eng.plants["legacy"]
    g, li = pp.path_index["P01_W01"]
    eng.legacy_ops.control(s + 0.5)
    assert float(pp.D[g]["speed"].reshape(-1)[li]) == pytest.approx(eng.sc.legacy_cfg["dr_speed"])
    eng.legacy_ops.control(e)
    assert float(pp.D[g]["speed"].reshape(-1)[li]) == pytest.approx(eng.sc.legacy_cfg["speed"])


def test_capacity_notice_updates_twin_and_restores(eng):
    from plyxis.lang import units as U
    ev = _event(eng, "capacity", asset="GASPLANT")
    prog = eng.twin.model.program
    gid, li = prog.inst_group[prog.flat.by_path["GASPLANT"]]
    before = float(eng.twin.live.P[gid]["cap"].reshape(-1)[li])
    eng._apply_event(ev)
    after = float(eng.twin.live.P[gid]["cap"].reshape(-1)[li])
    assert after == pytest.approx(before * ev["factor"], rel=1e-5)          # the twin is told, not left to guess
    assert U.from_canonical(after, "Mscf/d") == pytest.approx(24000, rel=0.01)
    assert eng.legacy_ops.takeaway_since is not None                       # manual ops react at once too
    t_back, back = eng.restores[-1]
    assert t_back == pytest.approx(eng.t + ev["hours"]) and back["restore"]
    assert back["factor"] * ev["factor"] == pytest.approx(1.0)


def test_forecast_does_not_extrapolate_announced_transients():
    from plyxis.data.forecast import Forecaster
    f = Forecaster(["power_price", "gas_price"], {"power_price": 0.05, "gas_price": 1.2})
    f.announce("power_price", 65.0, 67.0, 5.0)
    f.announce("gas_price", 36.0, 60.0, -2.5)
    for i in range(66 * 6):
        t = i / 6
        f.observe("power_price", t, 5.0 if 65 <= t < 67 else 0.05)
        f.observe("gas_price", t, -2.5 if 36 <= t < 60 else 1.2)
    fc = f.forecast(65.9, 12, 1.0)
    p = [float(v) for v in fc["power_price"]]
    assert p[0] == pytest.approx(5.0) and max(p[2:]) < 0.1                # spike only inside its window
    g = [float(v) for v in f.forecast(50.0, 14, 1.0)["gas_price"]]
    assert g[0] == pytest.approx(-2.5) and g[-1] == pytest.approx(1.2)      # back to normal after the notice
