"""Operating modes, operator decisions, handover report, data quality, what-if, working-place assays, margins."""
import dataclasses
import json
import os

import pytest
import torch

from plyxis.lang import units as U
from plyxis.optim.cascade import CascadeConfig
from plyxis.runtime import report, whatif
from plyxis.runtime.engine import Engine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KESTREL = os.path.join(ROOT, "scenarios", "kestrel")
CORVUS = os.path.join(ROOT, "scenarios", "corvus")
CHEAP = CascadeConfig(relax_iters=20, multistart=2, population=2, repair_iters=8, polish_iters=8, neural_iters=1)


def _engine(path=KESTREL, **kw):
    e = Engine(path, sync_plans=True, use_neural=False, online_learning=False, **kw)
    e.planner.cfg = dataclasses.replace(CHEAP)
    return e


def _pending_speed(e):
    """A speed recommendation: the plant is made to run one well far from the plan, so the plan asks for a change."""
    vals, _ = e.planner.setpoints_at(e.t)
    key = next(k for k in sorted(vals) if k[1] == "speed")
    e.advisory.held[key] = 0.35 if vals[key] > 0.7 else 1.0
    e.request_replan("test")
    return next(r for r in e.advisory.view()["recommendations"] if r["status"] == "pending" and r["kind"] == "setpoint"
                and r["decision"] == "speed")


@pytest.fixture(scope="module")
def eng():
    e = _engine()
    e.run_headless(1.0)
    return e


def test_autonomous_is_the_default_and_lists_what_it_does(eng):
    v = eng.advisory.view()
    assert v["mode"] == "autonomous" and v["plan_valid"]
    assert all(r["status"] == "automatic" for r in v["recommendations"])
    assert eng.advisory.decide("all", "accept")["ok"] is False      # nothing to decide outside advisory mode


def test_advisory_holds_the_plant_until_someone_decides():
    e = _engine()
    e.run_headless(1.0)
    pp = e.plants["plyxis"]
    assert e.advisory.set_mode("advisory", "shift lead")["ok"]
    held, _ = pp.readback()
    e.run_headless(1.0)
    after, _ = pp.readback()
    # nothing accepted, nothing moves up: the set points stay; only the plant's own protection may throttle them
    assert all(e.advisory.held[k] == held[k] for k in held)
    assert not [k for k in held if k[1] == "speed" and after[k] > held[k] + 1e-6]
    r = _pending_speed(e)
    r["start"] = e.t                                                 # due now
    assert e.advisory.decide([r["id"]], "accept", "shift lead")["decided"] == 1
    e.step()
    assert e.advisory.held[(r["asset"], r["decision"])] == pytest.approx(r["to"])
    assert pp.control_value(r["asset"], r["decision"]) <= r["to"] + 1e-5
    log = e.advisory.audit.tail(kind="decision")
    assert log[-1]["actor"] == "shift lead" and log[-1]["rec"] == r["id"]


def test_rejection_needs_a_reason_and_pins_the_decision():
    e = _engine()
    e.run_headless(0.5)
    e.advisory.set_mode("advisory")
    r = _pending_speed(e)
    assert e.advisory.decide([r["id"]], "reject")["ok"] is False
    assert e.advisory.decide([r["id"]], "reject", "foreman", "pad is being serviced")["replan"] is True
    plan = e.planner.plan                                           # re-planned around the decision
    pb, key = plan.problem, (r["asset"], "speed")
    gid, li = pb.prog.inst_group[pb.prog.flat.by_path[r["asset"]]]
    D, _ = pb.space.decode(plan.x, pb.P, relax=False)
    assert float(D[gid]["speed"][0, li, 0]) == pytest.approx(e.advisory.held[key], rel=1e-4)
    assert key in e.advisory.overrides
    assert not e.advisory.pins(e.t + e.advisory.shift_h + 1.0)       # the override ends with the shift


def test_modify_uses_the_operators_value():
    e = _engine()
    e.run_headless(0.5)
    e.advisory.set_mode("advisory")
    r = _pending_speed(e)
    v = (r["lo"] + r["hi"]) / 2
    assert e.advisory.decide([r["id"]], "modify", "foreman", "vibration on this pump", v)["ok"]
    e.step()
    assert e.advisory.held[(r["asset"], "speed")] == pytest.approx(v)
    assert e.plants["plyxis"].control_value(r["asset"], "speed") <= v + 1e-4


def test_rejected_lineup_is_held_for_the_planner():
    e = _engine()
    e.run_headless(0.5)
    e.advisory.set_mode("advisory")
    # the plant runs another line-up than the plan's, so the plan asks for a change
    by: dict = {}
    for k in e.advisory.held_routes:
        by.setdefault(k[:2], []).append(k)
    group = next(v for v in by.values() if len(v) > 1 and v[0][1] == "header")
    best = max(group, key=lambda k: e.advisory.held_routes[k])
    other = next(k for k in group if k != best)
    e.advisory.held_routes[best], e.advisory.held_routes[other] = 0.0, 1.0
    e.request_replan("test")
    r = next((r for r in e.advisory.view()["recommendations"] if r["status"] == "pending" and r["kind"] == "lineup"), None)
    assert r is not None, "the plan should want a line-up changed"
    assert e.advisory.decide([r["id"]], "reject", "foreman", "valve is locked out")["replan"] is True
    keys = [tuple(k.split("|")) for k in r["routes"]]
    assert all(("route", k) in e.advisory.overrides for k in keys)
    pb = e.planner.plan.problem
    assert pb.pin_mask is not None and float(pb.pin_mask.sum()) > 0
    assert any(o["decision"].endswith("line-up") for o in e.advisory.view()["overrides"])
    _, routes = e.planner.setpoints_at(e.t)
    held = {k: e.advisory.overrides[("route", k)][0] for k in keys}
    assert all(abs(routes[k] - held[k]) < 0.02 for k in keys)        # the new plan keeps it


def test_a_plan_far_below_the_last_one_gets_a_second_opinion(monkeypatch):
    from plyxis.runtime import planner as PL
    e = _engine()
    e.planner.cfg = dataclasses.replace(e.planner.cfg, second_opinion=0.05)
    e.run_headless(0.2)
    first = e.planner.plan.objective
    real = PL.Cascade.solve
    calls = []

    def lost_once(self, warm=None, lam0=None, callback=None):
        r = real(self, warm=warm, lam0=lam0, callback=callback)
        calls.append(self.cfg.seed)
        if len(calls) == 1:                      # the first search lost its way
            r.objective = 0.5 * first
        return r
    monkeypatch.setattr(PL.Cascade, "solve", lost_once)
    e.request_replan("test")
    run = e.planner.runs[-1]
    assert len(calls) == 2 and calls[0] != calls[1]        # another random stream
    assert run["second_opinion"]["kept"] == "second"
    assert run["objective"] > 0.9 * first
    # a normal plan costs nothing extra
    calls.clear()
    e.planner.plan.objective = -1e12
    e.request_replan("test")
    assert len(calls) == 1 and e.planner.runs[-1]["second_opinion"] is False


def test_a_cold_plan_is_solved_from_more_random_streams(monkeypatch):
    from plyxis.runtime import planner as PL
    e = _engine()
    e.planner.cfg = dataclasses.replace(e.planner.cfg, cold_starts=2)
    real = PL.Cascade.solve
    calls = []

    def first_lost(self, warm=None, lam0=None, callback=None):
        r = real(self, warm=warm, lam0=lam0, callback=callback)
        calls.append((self.cfg.seed, warm is None))
        if len(calls) == 1:                      # the first stream's search lost its way
            r.objective = r.objective - 1e9
        return r
    monkeypatch.setattr(PL.Cascade, "solve", first_lost)
    e.request_replan("initial plan")
    run = e.planner.runs[-1]
    assert len(calls) == 2 and calls[0][0] != calls[1][0] and all(cold for _s, cold in calls)
    assert run["cold_starts"] and run["objective"] > run["cold_starts"][0] - 1e6   # the second stream's plan is kept
    # a plan that starts from the running plan is solved once
    calls.clear()
    e.request_replan("test")
    assert len(calls) == 1 and e.planner.runs[-1]["cold_starts"] == []


def test_stale_plan_is_no_recommendation():
    e = _engine()
    e.run_headless(0.5)
    plan_t = e.planner.plan.t0
    assert e.advisory.setpoints(plan_t + 1.0)[0]
    assert e.advisory.setpoints(plan_t + e.advisory.valid_h + 1.0) == ({}, {})
    assert any(a["kind"] == "plan" for a in e.alerts)


def test_shadow_mode_mirrors_manual_operations_and_values_the_plan():
    e = _engine()
    e.advisory.set_mode("shadow")
    e.run_headless(3.0)
    a, b = e.plants["plyxis"].kpi.now()["cum"], e.plants["legacy"].kpi.now()["cum"]
    assert a["oil_bbl"] == pytest.approx(b["oil_bbl"], rel=1e-3)     # same plant, same operators, same noise
    assert e.advisory.shadow_log and "gap" in e.advisory.shadow_log[-1]


def test_shift_report_is_complete_and_serialisable(eng):
    r = report.shift_report(eng, 12.0)
    json.dumps(r)
    md = report.to_markdown(r)
    for section in ("# Shift report", "## Result", "## What happened", "## The current plan", "## Next shift", "## Data quality"):
        assert section in md


def test_quality_flags_a_frozen_sensor_and_the_model_takes_over():
    e = _engine()
    tag = next(t for t in e.plants["plyxis"].tele if t[3].endswith("TK.OIL.LVL"))[3]
    e.run_headless(0.5)
    e.inject({"kind": "sensor", "tag": tag, "fault": "freeze", "hours": 6})
    e.run_headless(3.0)
    q = e.quality.summary()
    assert any(i["tag"] == tag and i["issue"] == "frozen" for i in q["issues"])
    assert any(a["kind"] == "data quality" for a in e.alerts)
    info = e.twin.tags.get(tag)
    assert (info.gid, "oil_lvl", info.li) in e.twin.estimator.distrusted
    live = float(e.twin.live.S[info.gid]["oil_lvl"].reshape(-1)[info.li])
    pp = e.plants["plyxis"]
    gid, li = pp.path_index[info.path]
    truth = float(pp.S[gid]["oil_lvl"].reshape(-1)[li])
    cap = float(pp.P[gid]["oil_tank"].reshape(-1)[li])
    stuck = U.to_canonical(e.quality.state[tag].last, info.unit)
    assert abs(live - truth) <= abs(stuck - truth) + 0.01 * cap      # the prediction is no worse than the stuck reading
    assert abs(live - truth) < 0.10 * cap


def test_quality_has_no_false_alarms_on_clean_data():
    e = _engine()
    e.run_headless(4.0)
    assert e.quality.summary()["bad_now"] == 0 and e.quality.flagged_total == 0


def test_quality_flags_stale_and_impossible_readings():
    e = _engine()
    e.run_headless(0.5)
    level = next(t for t in e.plants["plyxis"].tele if t[3].endswith("TK.OIL.LVL"))[3]
    other = next(t for t in e.plants["plyxis"].tele if t[3].endswith("TK.WTR.LVL"))[3]
    e.inject({"kind": "sensor", "tag": level, "fault": "dropout", "hours": 6})
    e.inject({"kind": "sensor", "tag": other, "fault": "value", "value": -5000.0, "hours": 6})
    e.run_headless(2.0)
    kinds = {i["tag"]: i["issue"] for i in e.quality.summary()["issues"]}
    assert kinds.get(level) == "stale" and kinds.get(other) == "out of range"


def test_whatif_outage_and_price(eng):
    comp = next(i.path for i in eng.twin.model.flat.instances if i.type.short == "compressor_station")
    r = whatif.run(eng, [{"kind": "outage", "asset": comp, "hours": 6}], "station down", config=CHEAP)
    assert r["ok"] and r["nominal"] > 0 and r["changes"]
    p = whatif.run(eng, [{"kind": "price", "name": "oil_price", "factor": 0.5}], "oil at half", config=CHEAP)
    assert p["ok"] and p["cost_of_event"] > 0
    assert whatif.run(eng, [{"kind": "outage", "asset": "NOPE"}])["ok"] is False


def test_pins_are_exact_and_only_where_asked():
    e = _engine()
    e.run_headless(0.2)
    plan = e.planner.plan
    pb = plan.problem
    well = next(i.path for i in pb.prog.flat.instances if i.type.short == "esp_well")
    gid, li = pb.prog.inst_group[pb.prog.flat.by_path[well]]
    q = pb.with_context()
    assert q.set_pins({(well, "speed"): (0.5, 4.0)}) > 0
    x = q.pinned(plan.x)
    u = q.space.split(x)[(gid, "speed")][0, li]
    assert float(u[0]) == pytest.approx(float(u[1]))
    assert float(u[-1]) == pytest.approx(float(q.space.split(plan.x)[(gid, "speed")][0, li, -1]))
    assert torch.equal(pb.pinned(plan.x), plan.x)                    # the base problem has no pins


def test_assays_sample_working_places_in_both_worlds_with_the_same_rule():
    e = _engine(CORVUS)
    e.planner.cfg = dataclasses.replace(CHEAP, discrete="ga", exact_branches=())
    e.run_headless(3.0)
    pa, pl = e.plants["plyxis"], e.plants["legacy"]
    assert len(pa.last_assay) >= 6 and len(pl.last_assay) >= 6
    assert e.legacy_ops.assay_var                                     # grade control reconciles its block model
    assert all(v < 0.15 ** 2 for v in e.twin.estimator.test_var.values())
    gid = next(g.gid for g in e.twin.model.program.groups if not g.is_env and g.ir.type.short == "stope")
    u = e.twin.estimator.uncertainty(gid, "grade")
    assert float(u.min()) < float(u.max())                            # sampled stopes are known better
    m = e.margin_log[-1]
    assert m["asset"] == "MILL" and 0.0 < m["margin"] < 0.05
