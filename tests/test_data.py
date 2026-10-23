"""Data integration: emulator, connectors, estimation (MHE), anomaly detection, discovery, twin versioning."""
import os

import pytest
import torch

from plyxis.data.anomaly import AnomalyDetector
from plyxis.data.connectors import EmulatorConnector, HttpPushConnector
from plyxis.data.discovery import Discovery
from plyxis.data.emulator import PlantEmulator, Telemetry
from plyxis.data.estimation import Estimator, LiveTwin, TagMap
from plyxis.data.repo import TwinRepo
from plyxis.lang import units as U

from conftest import MODELS

SMALL = os.path.join(MODELS, "small.plx")


def test_emulator_steps_and_publishes_telemetry():
    emu = PlantEmulator([SMALL], "testfield", dt=0.25, seed=1, noise=0.0)
    out = emu.step()
    tags = {m.tag for m in out["telemetry"]}
    assert "T.W1.FLOW.LIQ" in tags and "T.B1.TK.LVL" in tags
    assert emu.kpi.cum["revenue"] > 0
    assert emu.t == pytest.approx(0.25)


def test_emulator_saturates_tanks_physically():
    emu = PlantEmulator([SMALL], "testfield", dt=1.0, seed=1, noise=0.0)
    gid, li = emu.path_index["b1"]
    emu.set_controls({("b1", "draw"): 0.0, ("b2", "draw"): 0.0})
    for _ in range(40):
        emu.step()
    tank = float(emu.P[gid]["tank"].reshape(-1)[li])
    assert float(emu.S[gid]["lvl"].reshape(-1)[li]) <= tank + 1e-6


def test_protection_throttles_overloaded_facility():
    emu = PlantEmulator([SMALL], "testfield", dt=0.5, seed=1, noise=0.0)
    emu.set_controls({("w1", "speed"): 1.0, ("w2", "speed"): 1.0, ("w3", "speed"): 1.0})
    emu.step()
    assert emu.throttle_log, "gas capacity overload must trigger the protection layer"
    # throttled only the producers, never the tank pump-outs (they do not feed the gas overload)
    gid, li = emu.path_index["b1"]
    assert float(emu.D[gid]["draw"][0, li, 0]) == pytest.approx(U.to_canonical(30, "bbl/h"), rel=1e-4)


def test_moving_horizon_estimation_recovers_deliverability(small_model):
    # the 'real' plant has 20 % more deliverability on well 1 than the twin believes
    emu = PlantEmulator([SMALL], "testfield", dt=0.25, seed=2, noise=0.3)
    g, li = emu.path_index["w1"]
    true_q = float(emu.P[g]["q_max"].reshape(-1)[li]) * 1.2
    emu.P[g]["q_max"].reshape(-1)[li] = true_q
    emu.P_nominal[g]["q_max"].reshape(-1)[li] = true_q
    live = LiveTwin(small_model, window=12)
    tags = TagMap(small_model.program, small_model.flat)
    est = Estimator(live, tags, 0.25)
    gt, lt = small_model.program.inst_group[small_model.flat.by_path["w1"]]
    before = float(live.P[gt]["q_max"].reshape(-1)[lt])
    for k in range(40):                                   # 10 h of telemetry, re-calibrated hourly like the engine
        out = emu.step()
        vals, routes = emu.readback()
        D, R = live.controls_from_paths(vals, routes)
        est.ingest(emu.t, out["telemetry"], D, R, {k2: torch.tensor([float(v)]) for k2, v in emu.env_now().items()})
        if k >= 8 and k % 4 == 3:
            est.calibrate(iters=25)
    after = float(live.P[gt]["q_max"].reshape(-1)[lt])
    assert abs(after - true_q) < abs(before - true_q) * 0.5


def _feed(emu, live, est, k_total, on_step=None, calibrate_from=8):
    for k in range(k_total):
        if on_step:
            on_step(k)
        out = emu.step()
        vals, routes = emu.readback()
        D, R = live.controls_from_paths(vals, routes)
        est.ingest(emu.t, out["telemetry"], D, R, {k2: torch.tensor([float(v)]) for k2, v in emu.env_now().items()})
        if k >= calibrate_from and k % 4 == 3:
            est.calibrate(iters=25)


def test_step_change_is_tracked_when_mismatch_releases_priors(small_model):
    """A persistent mismatch (here: well 2 loses half its deliverability) releases the anchor and
    smoothness priors of that asset, so the estimate follows the step instead of crawling."""
    emu = PlantEmulator([SMALL], "testfield", dt=0.25, seed=3, noise=0.3)
    live = LiveTwin(small_model, window=12)
    est = Estimator(live, TagMap(small_model.program, small_model.flat), 0.25)
    g, li = emu.path_index["w2"]

    def on_step(k):
        if k == 16:
            emu.P_nominal[g]["q_max"].reshape(-1)[li] *= 0.5
        est.released = {"w2"} if k > 16 else set()
    _feed(emu, live, est, 44, on_step)
    gt, lt = small_model.program.inst_group[small_model.flat.by_path["w2"]]
    true_q = float(emu.P[g]["q_max"].reshape(-1)[li])
    assert abs(float(live.P[gt]["q_max"].reshape(-1)[lt]) - true_q) / true_q < 0.1


def test_reset_states_after_work_order(small_model):
    emu = PlantEmulator([SMALL], "testfield", dt=0.25, seed=4, noise=0.0)
    live = LiveTwin(small_model, window=12)
    est = Estimator(live, TagMap(small_model.program, small_model.flat), 0.25)
    _feed(emu, live, est, 6, calibrate_from=99)
    gt, lt = small_model.program.inst_group[small_model.flat.by_path["w1"]]
    live.S[gt]["up"].reshape(-1)[lt] = 0.0
    est.reset_states(gt, lt, {"up": 1.0})
    assert float(live.S[gt]["up"].reshape(-1)[lt]) == 1.0
    assert not any(k[0] == gt and k[1] == lt for b in live.buf for k in b["meas"])


def test_anomaly_detector_flags_trips_and_trends(small_model):
    tags = TagMap(small_model.program, small_model.flat)
    det = AnomalyDetector(tags, small_model.flat)
    det.update(0.0, [Telemetry("T.W1.RUN", 1.0, None, 0.0)], {})
    alerts = det.update(0.25, [Telemetry("T.W1.RUN", 0.0, None, 0.25)], {})
    assert any(a.kind == "status" and a.severity == "critical" for a in alerts)


def test_discovery_generates_reviewable_plx(small_model):
    rules = small_model.flat.env_type.rules
    disc = Discovery(rules, {i.path for i in small_model.flat.instances}, {"T.W1", "T.W2", "T.W3", "T.B1", "T.B2"})
    disc.note_birth("T.W4", {"lat": 31.45, "lon": -103.5, "type": "well"})
    assert disc.observe("T.W2.FLOW.LIQ", 1.0) is None                    # known asset
    p = disc.observe("T.W4.FLOW.LIQ", 2.0)
    assert p is not None and p.assets == ["w4"]
    assert "init w4 as well" in p.code and "connect(w4.prod, m.feed)" in p.code and "31.45" in p.code


def test_twin_manager_applies_discovery_and_keeps_state(tmp_path):
    from plyxis.runtime.twin_manager import TwinManager
    import shutil
    for f in ("small.plx",):
        shutil.copy(os.path.join(MODELS, f), tmp_path / f)
    tm = TwinManager([str(tmp_path / "small.plx")], "testfield", "plan", dt_data=0.25)
    gid, li = tm.model.program.inst_group[tm.model.flat.by_path["w2"]]
    tm.live.P[gid]["q_max"].reshape(-1)[li] = 1.2345
    tm.discovery.note_birth("T.W9", {"lat": 31.5, "lon": -103.4})
    prop = tm.discovery.observe("T.W9.FLOW.LIQ", 3.0)
    res = tm.apply_discovery(prop, 3.0)
    assert res["ok"] and tm.version == 2
    assert tm.model.flat.instance("w9") is not None
    gid2, li2 = tm.model.program.inst_group[tm.model.flat.by_path["w2"]]
    assert float(tm.live.P[gid2]["q_max"].reshape(-1)[li2]) == pytest.approx(1.2345)
    assert len(tm.repo.commits) == 2 and "w9" in tm.repo.commits[-1].diff


def test_repo_diffs():
    repo = TwinRepo({"a.plx": "x = 1;\n"})
    c = repo.commit({"a.plx": "x = 2;\n"}, "change", "engineer", 1.0)
    assert "+x = 2;" in c.diff and "-x = 1;" in c.diff
    assert repo.commit({"a.plx": "x = 2;\n"}, "noop", "engineer", 2.0) is None


def test_connectors():
    sc = EmulatorConnector("scada")
    sc.push([Telemetry("A.B", 1.0, None, 0.0)])
    assert len(sc.poll(0.0)) == 1 and sc.stats.messages == 1
    http = HttpPushConnector()
    assert http.submit([{"tag": "X.Y", "value": 3}], 1.0) == 1
    assert http.poll(1.0)[0].tag == "X.Y"
