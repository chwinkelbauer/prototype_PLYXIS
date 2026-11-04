"""Aquila Steel: the steel library's mechanics the corporate A/B depends on, and a competent baseline that runs
without the plant's protection and without drawing a yard below empty."""
import os

import pytest
import torch

from conftest import ROOT

AQUILA = os.path.join(ROOT, "scenarios", "aquila")


@pytest.fixture(scope="module")
def aquila():
    from plyxis.lang import compile_model
    return compile_model([os.path.join(AQUILA, "field.plx")], env="aquila")


def _sim(model, set_dec=None, states=None):
    from plyxis.optim.problem import Problem
    pb = Problem(model.program)
    D, R = pb.space.decode(pb.space.default_u(1), pb.P, relax=False)
    for (path, name), v in (set_dec or {}).items():
        g, li = model.program.inst_group[model.program.flat.by_path[path]]
        D[g][name][:, li] = v
    S0 = pb.S0
    for (path, name), v in (states or {}).items():
        g, li = model.program.inst_group[model.program.flat.by_path[path]]
        S0[g][name].reshape(-1)[li] = v
    raw = model.program.simulate(D, R, pb.P, S0, pb.E, pb.T, pb.dt, pb.t0)
    pb.D = D
    return pb, raw


def _sig(model, raw, path, key):
    g, li = model.program.inst_group[model.program.flat.by_path[path]]
    x = model.program.signal(raw, g, key)
    return x.reshape(x.shape[0], -1, x.shape[-1])[0, li]


def test_group_size(aquila):
    flat = aquila.program.flat
    count = lambda t: sum(1 for i in flat.instances if i.type.short == t)  # noqa: E731
    assert count("grid") == 13 and count("blast_furnace") == 15 and count("eaf") == 10
    assert count("steel_market") == 12 and count("dri_plant") == 4


def test_one_term_name_one_kind():
    from plyxis.lang import compile_model
    src = """
define t as pkg {
  define widget as ent { behavior: cost burn = 1 [USD/h]; emission burn = 1 [kgCO2e/h]; };
};
use t;
define site as env { contains: init w as widget; };
define run as sim { env: site; horizon = 2 [h]; step = 1 [h]; objective: maximize sum(revenue) - sum(cost); };
"""
    with pytest.raises(ValueError, match="burn"):
        compile_model(["clash.plx"], sources={"clash.plx": src}, include_stdlib=False)


def test_converter_blows_all_hot_metal_and_casts_the_rest_to_pig_iron(aquila):
    pb, raw = _sim(aquila)
    hm = sum(_sig(aquila, raw, b, "node:local:made") for b in ("NL_BF1", "NL_BF2"))
    made = _sig(aquila, raw, "NL_BOF", "node:local:made")
    g, li = aquila.program.inst_group[aquila.program.flat.by_path["NL_BOF"]]
    share = pb.D[g]["hm_share"][0, li]
    assert torch.allclose(made, hm / share * 0.91, rtol=1e-4)
    assert float(_sig(aquila, raw, "NL_BOF", "node:local:pig").abs().max()) < 1e-3
    _pb, raw = _sim(aquila, states={("NL_BOF", "up"): 0.0})          # converter shop down: all iron to the pig caster
    pig = _sig(aquila, raw, "NL_BOF", "node:local:pig")
    assert torch.allclose(pig, hm, rtol=1e-4)
    assert float(_sig(aquila, raw, "NL_BOF", "node:local:made").abs().max()) < 1e-3


def test_spot_sales_are_discounted_and_a_backlog_is_delivered_at_contract(aquila):
    pb, raw = _sim(aquila)
    g, li = aquila.program.inst_group[aquila.program.flat.by_path["EU_HRC"]]
    contract = float(pb.P[g]["contract"].reshape(-1)[li])
    delivered = _sig(aquila, raw, "EU_HRC", "node:local:delivered")
    backlog = _sig(aquila, raw, "EU_HRC", "state:backlog")
    spot = _sig(aquila, raw, "EU_HRC", "node:local:spot_sold")
    expect = torch.relu(delivered - contract - torch.relu(backlog) / pb.dt)
    assert torch.allclose(spot, expect, atol=1e-3)
    assert float(backlog.min()) >= 0.0                                # nothing is delivered ahead of the call-offs


def test_inventories_are_worth_their_variable_cost_at_most_nrv():
    import csv
    with open(os.path.join(AQUILA, "product_yards.csv"), encoding="utf-8") as fh:
        yards = list(csv.DictReader(fh))
    with open(os.path.join(AQUILA, "lanes_hrc.csv"), encoding="utf-8") as fh:
        hrc = {r["site"]: float(r["freight"]) for r in csv.DictReader(fh)}
    coil = [y for y in yards if y["kind"] == "hrc" and y["site"] == "NL"][0]
    assert float(coil["unit_cost"]) <= 0.98 * (680 - hrc["NL"]) + 1e-6
    assert all(float(y["reserve"]) == float(y["stock"]) for y in yards)     # both worlds hold the S&OP stock target


def test_baseline_runs_without_protection_or_underflow():
    from plyxis.data.emulator import PlantEmulator
    from plyxis.runtime.scenario import PriceModel, Scenario
    sc = Scenario.load(AQUILA)
    prices = PriceModel(sc.prices, seed=11)
    emu = PlantEmulator(sc.truth_files, sc.truth_env, dt=sc.dt_emu, seed=0, name="legacy",
                        defines_prices=lambda n, c: prices.canonical(n, c), kpi_metrics=sc.raw["kpi"]["metrics"],
                        assays=sc.raw.get("assays"))
    ops = sc.legacy_class()(emu, sc.legacy_cfg)
    while emu.t < 12.0 - 1e-9:
        ops.control(emu.t)
        ops.assays(emu.step()["telemetry"])
    assert emu.throttle_log == [] and emu.underflow == {}
    cum = emu.kpi.cum
    assert cum["profit"] > 0
    contract = sum(float(m["contract"]) for m in ops.markets.values())
    assert 0.9 * contract < cum["m_shipped"] / 12.0 < 1.25 * contract          # t over 12 h
    stock = sum(ops._stock(y) for y in ops.yards)
    assert stock >= 0.98 * sum(v["target"] for v in ops.yards.values())    # it does not sell its yards down


def test_lint_finds_a_tag_shared_by_two_assets():
    from plyxis.lang import compile_model
    from plyxis.lang.lint import lint
    src = """
define t as pkg {
  define widget as ent { telemetry: r <- "R"; behavior: r = 1; cost run = 1 [USD/h]; };
};
use t;
define site as env { contains: for i in 1..2 { init w{i} as widget { tag = "W"; }; } };
define run as sim { env: site; horizon = 2 [h]; step = 1 [h]; objective: maximize sum(revenue) - sum(cost); };
"""
    m = compile_model(["dup.plx"], sources={"dup.plx": src}, include_stdlib=False)
    assert [f for f in lint(m, gradient_points=0) if f.code == "duplicate-tag"]
