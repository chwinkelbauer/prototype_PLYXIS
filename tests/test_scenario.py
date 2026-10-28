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


def test_outage_with_crew_estimate_is_planned_back(eng):
    import torch
    eng._apply_event({"kind": "fail", "asset": "P02_W03", "hours": 2.0, "title": "test outage"})
    state, value, back = eng.outages["P02_W03"]
    assert (state, value) == ("up", 1.0) and back == pytest.approx(eng.t + 2.0)
    pb = eng.planner.problem(eng.t, eng._forecast(eng.twin.model.program.T, eng.twin.model.program.dt))
    prog = pb.prog
    gid, li = prog.inst_group[prog.flat.by_path["P02_W03"]]
    pb.S0[gid]["up"].reshape(-1)[li] = 0.0                    # the twin has seen the trip
    assert pb.set_state_schedule({"P02_W03": (state, value, back - eng.t)}) == ["P02_W03"]
    e = pb.evaluate(pb.space.default_u(1), relax=False)
    up = torch.as_tensor(e.raw[prog.node_var(gid, "state:up")]).expand(1, -1, pb.T)[0, li]
    k = int(round(2.0 / pb.dt))
    assert float(up[:k].max()) == 0.0 and float(up[k:].min()) == 1.0
    eng._apply_event(eng.restores[-1][1])                      # back in service: the estimate is cleared
    assert "P02_W03" not in eng.outages


def test_idle_mill_is_not_off_spec_and_value_to_go_is_planning_only():
    import os

    import torch

    from conftest import ROOT
    from plyxis.data.emulator import _Kpi
    from plyxis.lang import compile_model
    from plyxis.optim.problem import Problem
    m = compile_model([os.path.join(ROOT, "scenarios", "corvus", "field.plx")], env="corvus")
    pb = Problem(m.program)
    D, R = pb.space.decode(pb.space.default_u(1), pb.P, relax=False)
    gid = m.program.inst_group[m.program.flat.by_path["ROM"]][0]
    D[gid]["reclaim"] = torch.zeros_like(D[gid]["reclaim"])           # mill stands for the whole horizon
    raw = m.program.simulate(D, R, pb.P, pb.S0, pb.E, pb.T, pb.dt, pb.t0)
    gm = m.program.inst_group[m.program.flat.by_path["MILL"]][0]
    for con in ("con:grade_low", "con:grade_high"):                  # no head grade, nothing off-spec
        assert float(torch.relu(torch.as_tensor(raw[m.program.node_var(gm, con)])).max()) == 0.0
    final = {tr.name for tr in m.program.terms if tr.final}
    assert "value_to_go" in final and "value_to_go" in _Kpi.EXPECTATION_TERMS


def test_sump_overflow_is_charged_in_the_kpi():
    import os

    from conftest import ROOT
    from plyxis.data.emulator import PlantEmulator
    pp = PlantEmulator([os.path.join(ROOT, "scenarios", "corvus", "field.plx")], "corvus", noise=0.0)
    g, li = pp.path_index["PS_S3"]
    pp.S[g]["level"].reshape(-1)[li] = float(pp.P[g]["sump"].reshape(-1)[li])     # full sump ...
    pp.S[g]["up"].reshape(-1)[li] = 0.0                                          # ... and the pump failed
    pp.step()
    inflow = float(pp.P[g]["inflow"].reshape(-1)[li])                            # m3/h (canonical)
    assert pp.kpi.cum["overflow_m3"] == pytest.approx(inflow * pp.dt, rel=1e-3)
    assert pp.kpi.terms["overflow"] == pytest.approx(-60.0 * inflow * pp.dt, rel=1e-3)
