"""Plants at their own time step in one twin: finding them, the split, per-unit dynamic programs, column generation."""
import itertools
import os

import pytest
import torch

from plyxis.lang import compile_model
from plyxis.lang.hierarchy import compile_hierarchy, detect_plants
from plyxis.optim.cascade import Cascade, CascadeConfig
from plyxis.optim.corporate import CorporateConfig, CorporatePlanner, step_values
from plyxis.optim.problem import Problem
from plyxis.optim.unitdp import dp_gap, dp_reset, dp_run, find_rules

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TOY = open(os.path.join(ROOT, "examples", "plants_toy.plx"), encoding="utf-8").read()
SRC = {"toy.plx": TOY}
PEAK = {"peak": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32)}
# the same toy with the plant declared by hand (a `plants:` section overrides the detector)
DECL = TOY.replace("    connect(SUP.o, MKT.rec);\n", "    connect(SUP.o, MKT.rec);\n    define P as group { P_*, L_P };\n").replace(
    "  objective: maximize sum(revenue) - sum(cost);\n};", "  objective: maximize sum(revenue) - sum(cost);\n  plants: P = 30 [min];\n};")
WORKS = os.path.join(ROOT, "scenarios", "aquila_works", "field.plx")


def test_plants_section_and_wildcard_groups():
    m = compile_model(["toy.plx"], sources={"toy.plx": DECL})
    flat = m.flat
    assert sorted(flat.instances[i].path for i in flat.inst_groups["P"]) == ["L_P", "P_MAKER", "P_STORE"]
    assert flat.sim.plants["P"] == pytest.approx(flat.sim.dt / 2)
    h = compile_hierarchy(["toy.plx"], sources={"toy.plx": DECL})
    assert list(h.plants) == ["P"] and not h.found


def test_the_plant_is_found_from_the_equations():
    m = compile_model(["toy.plx"], sources=SRC)
    found = detect_plants(m)
    assert len(found) == 1 and found[0].dt == pytest.approx(m.flat.sim.dt / 2)
    # the maker changes its rate every half hour; its store holds three hours of flow (too small to cut at), and the
    # lane draws from the store: the lane's market takes the flow per hour, the fence is there
    assert sorted(m.flat.instances[i].path for i in found[0].members) == ["L_P", "P_MAKER", "P_STORE"]
    assert [(m.flat.instances[v].path, port) for v, port, _ in found[0].cut] == [("MKT", "rec")]
    # a store of 300 hours decouples the maker: the fence moves to the store
    big = TOY.replace("properties: cap = 200 [t]; unit_cost = 40", "properties: cap = 20000 [t]; unit_cost = 40")
    m = compile_model(["toy.plx"], sources={"toy.plx": big})
    found = detect_plants(m)
    assert [m.flat.instances[i].path for i in found[0].members] == ["P_MAKER"]
    assert [(m.flat.instances[v].path, port) for v, port, _ in found[0].cut] == [("P_STORE", "feed")]
    # nothing finer than the hour: no plant
    coarse = TOY.replace("default 0.5 * cap hold 30 [min];", "default 0.5 * cap hold 1 [h];")
    assert detect_plants(compile_model(["toy.plx"], sources={"toy.plx": coarse})) == []


def test_a_delay_given_as_a_duration_is_the_same_time_at_any_step():
    src = """
define lagged as ent {
  properties: lag = 45 [min];
  decision: u in [0, 1] default 0;
  behavior:
    y = delay(time, lag);
    metric late = time - y [h];
};
define e as env { contains: init A as lagged; };
define s15 as sim { env: e; horizon = 4 [h]; step = 15 [min]; objective: maximize sum(metric); };
define s5 as sim { env: e; horizon = 4 [h]; step = 5 [min]; objective: maximize sum(metric); };
"""
    for sim, steps in (("s15", 3), ("s5", 9)):
        prog = compile_model(["l.plx"], sources={"l.plx": src}, sim=sim).program
        pb = Problem(prog)
        with torch.no_grad():
            raw = pb.evaluate(pb.space.default_u(1), relax=False).raw
        g = [g for g in prog.groups if not g.is_env][0]
        y = raw[prog.node_var(g.gid, "local:y")].reshape(-1)
        tm = torch.arange(prog.T, dtype=torch.float32) * prog.dt
        assert torch.allclose(y[steps:], tm[:-steps], atol=1e-5), sim


def test_split_finds_every_crossing_and_keeps_the_rest():
    h = compile_hierarchy(["toy.plx"], sources=SRC, k_max=3)
    assert list(h.plants) == ["P.maker"]
    p = h.plants["P.maker"]
    assert [(c.src, c.src_port, c.dst, c.dst_port, c.short) for c in p.crossings] == [("L_P", "deliver", "MKT", "rec", "widget")]
    assert p.ratio == 2 and p.program.T == 2 * h.master.T
    assert sorted(i.path for i in p.flat.instances) == ["L_P", "PLX_BND_P_maker_0", "P_MAKER", "P_STORE"]
    paths = [i.path for i in h.master_flat.instances]
    assert {"MKT", "SUP", "PLX_SIMPLEX_P_maker"} <= set(paths) and sum(x.startswith("PLX_COL_P_maker_") for x in paths) == 4
    # inside the plant: maker -> store, lane -> store, lane -> sink; the master: supplier and every slot into the market
    assert len(p.flat.edges) == 3
    mkt = h.master_flat.instances[h.master_flat.by_path["MKT"]]
    assert len(h.master_flat.ports[mkt.ports["rec"]].edges) == 1 + 4
    # the plant's environment metric sums over its own makers, the master's over none (an empty sum is 0)
    assert any(t.name == "made" for t in p.program.terms) and any(t.name == "made" for t in h.master.terms)


def test_a_connection_into_a_plant_is_refused():
    src = DECL.replace("define P as group { P_*, L_P };", "define P as group { P_* };")
    with pytest.raises(ValueError, match="connection into plant 'P'"):
        compile_hierarchy(["toy.plx"], sources={"toy.plx": src})


def _brute(T, ok, value):
    best, arg = -1e30, None
    for d in itertools.product([0, 1], repeat=T):
        if ok(d):
            v = value(d)
            if v > best:
                best, arg = v, d
    return best


def test_unit_dynamic_programs_are_exact():
    gen = torch.Generator().manual_seed(3)
    for _ in range(20):
        T = 9
        g = torch.randn(T, generator=gen)
        val = lambda d: float(sum(gi * di for gi, di in zip(g.tolist(), d)))  # noqa: E731
        # gap: at most one 1 in any 3 consecutive steps
        allowed = torch.rand(T, generator=gen) > 0.2
        d = dp_gap(g, 3, allowed)
        ok = lambda d: all(sum(d[k:k + 3]) <= 1 for k in range(T)) and all(a or not x for a, x in zip(allowed.tolist(), d))  # noqa: E731
        assert ok([int(v) for v in d]) and val(d.tolist()) == pytest.approx(_brute(T, ok, val), abs=1e-5)
        # run: at most 2 consecutive ones
        d = dp_run(g, 2)
        ok = lambda d: all(sum(d[k:k + 3]) <= 2 for k in range(T))  # noqa: E731
        assert ok([int(v) for v in d]) and val(d.tolist()) == pytest.approx(_brute(T, ok, val), abs=1e-5)
        # reset: a counter growing by inc, reset by d, never above L
        inc = torch.rand(T, generator=gen)
        L, w0 = 1.3, 0.2

        def ok(d, inc=inc.tolist()):
            w = w0
            for t in range(T):
                if w > L + 1e-9:
                    return False
                w = (w + inc[t]) * (1 - d[t])
            return True
        d = dp_reset(g, inc, w0, L)
        assert ok([int(v) for v in d]) and val(d.tolist()) == pytest.approx(_brute(T, ok, val), abs=1e-5)


def test_a_steel_works_splits_at_its_slab_yard_and_its_rules_are_found():
    h = compile_hierarchy([WORKS])
    # the converters' 45-minute tap to tap needs 15-minute steps; the furnaces, torpedo pool, ladle furnace and
    # casters join them through small buffers; the slab yard (days of stock) cuts the strip mill's side off
    assert sorted(h.plants) == ["DE.converter", "DE.hsm_works"]
    steel, mill = h.plants["DE.converter"], h.plants["DE.hsm_works"]
    paths = lambda p: sorted(p.flat.instances[i].path for i in range(len(p.flat.instances))  # noqa: E731
                             if not p.flat.instances[i].path.startswith("PLX_"))
    assert paths(steel) == sorted(["DE_BF1", "DE_BF2", "DE_BF3", "DE_HM", "DE_PIG", "DE_BOF1", "DE_BOF2", "DE_BOF3",
                                   "DE_LADLE", "DE_CC1", "DE_CC2"])
    assert paths(mill) == ["DE_HSM", "DE_RF1", "DE_RF2"]
    rules = {}
    for name, p in h.plants.items():
        pb = Problem(p.program)
        rules[name] = {(r.kind, pb.prog.gmap[r.gid].ir.type.short, r.dec) for r in find_rules(pb.prog, pb.space)}
        assert p.program.T == 4 * h.master.T
    assert rules["DE.converter"] == {("gap", "converter", "tap"), ("run", "caster_works", "run")}
    assert rules["DE.hsm_works"] == {("reset", "hsm_works", "change")}
    # one crossing per port on the other side: the power of nine units into the grid connection is one flow
    grid = [c for c in steel.crossings if c.dst == "DE_GRID"]
    assert len(grid) == 1 and len(grid[0].srcs) == 9 and len(steel.crossings) == 6 and len(mill.crossings) == 6


def test_step_values_add_up_to_the_objective():
    h = compile_hierarchy(["toy.plx"], sources=SRC)
    p = h.plants["P.maker"]
    pb = Problem(p.program, E={"peak": PEAK["peak"].repeat_interleave(2)})
    with torch.no_grad():
        e = pb.evaluate(pb.space.default_u(1), relax=False)
    v = step_values(p.program, e.raw, pb.dt, pb.E)
    assert float(v.sum()) == pytest.approx(float(e.objective[0]), rel=1e-4, abs=1e-2)


def test_column_generation_matches_the_whole_model():
    cfg = CascadeConfig(relax_iters=300, population=8, generations=1, repair_iters=60, polish_iters=200, multistart=4)
    whole = Problem(compile_model(["toy.plx"], sources=SRC).program, E=PEAK)
    ref = Cascade(whole, config=cfg).solve()
    h = compile_hierarchy(["toy.plx"], sources=SRC, k_max=6)
    plan = CorporatePlanner(h, CorporateConfig(rounds=6, master=cfg, plant=cfg), E=PEAK).solve()
    assert plan.feasible
    # the committed plan (one real schedule of the plant, the group re-planned around it) is as good as solving the
    # whole model at once (hand-worked optimum 37,100)
    assert plan.objective >= 0.95 * ref.objective


def test_regional_networks_are_found_at_the_markets_they_share():
    h = compile_hierarchy([os.path.join(ROOT, "scenarios", "aquila", "field.plx")], split="regions")
    # no unit of the hourly group needs a finer step; the markets several networks sell to (and the scrap markets
    # they buy from) stay with the group, the sites tied by their own transfers form the networks
    assert all(f.kind == "region" and f.dt == h.master.dt for f in h.found)
    assert {"DE+FR+NL", "AR+TX", "BR+IN", "SA", "PL", "MX", "GJ", "OD"} <= set(h.plants)
    master_types = {g.ir.type.short for g in h.master.groups if not g.is_env and g.n}
    assert {"steel_market", "scrap_market"} <= master_types
    assert not master_types & {"blast_furnace", "eaf", "slab_caster", "product_yard", "lane_hrc"}
    eu = h.plants["DE+FR+NL"]
    assert {c.dst for c in eu.crossings} == {"EU_HRC", "SCRAP_EU"}
    # the master keeps no decisions of its own: only the weights of the networks' schedules
    assert all(g.ir.type.short.startswith("plx_col_") for g in h.master.groups if not g.is_env and g.ir.decisions)


def test_a_counter_that_a_decision_resets_is_vectorised_and_exact(monkeypatch):
    # wear <- (wear + rolled * dt) * (1 - change): an affine state, computed by an associative scan instead of a
    # Python loop over the steps; the same values as the loop
    h = compile_hierarchy([WORKS])
    p = h.plants["DE.hsm_works"]
    assert "for t in range(T_len)" not in p.program.source and "(associative scan)" in p.program.source
    pb = Problem(p.program)
    u = torch.cat([pb.space.default_u(1), pb.space.random_u(3, torch.Generator().manual_seed(4))])
    with torch.no_grad():
        fast = pb.evaluate(u, relax=True)
    monkeypatch.setenv("PLX_NO_AFFINE", "1")
    h2 = compile_hierarchy([WORKS])
    p2 = h2.plants["DE.hsm_works"]
    assert "for t in range(T_len)" in p2.program.source
    pb2 = Problem(p2.program)
    with torch.no_grad():
        slow = pb2.evaluate(u, relax=True)
    assert torch.allclose(fast.objective, slow.objective, rtol=1e-6)
    assert torch.allclose(fast.max_violation(), slow.max_violation(), rtol=1e-4, atol=1e-5)


def test_plants_and_regional_networks_side_by_side():
    h = compile_hierarchy([WORKS], split="both")
    kinds = {f.name: f.kind for f in h.found}
    assert kinds["DE.converter"] == kinds["DE.hsm_works"] == "plant"
    assert sum(k == "region" for k in kinds.values()) >= 8
    # the plants' fence units stay with the group: no block trades with another block (the compile would refuse it)
    master_paths = {i.path for i in h.master_flat.instances}
    assert {"DE_SLABS", "DE_GRID", "DE_GASNET", "EU_HRC"} <= master_paths
    assert h.plants["DE.converter"].ratio == 4 and h.plants["NL"].ratio == 1


def test_blocks_solved_side_by_side_in_worker_processes():
    cfg = CascadeConfig(relax_iters=300, population=8, generations=1, repair_iters=60, polish_iters=200, multistart=4)
    whole = Cascade(Problem(compile_model(["toy.plx"], sources=SRC).program, E=PEAK), config=cfg).solve()
    h = compile_hierarchy(["toy.plx"], sources=SRC, k_max=6)
    plan = CorporatePlanner(h, CorporateConfig(rounds=6, master=cfg, plant=cfg, workers=2), E=PEAK).solve()
    # the start columns are solved cold in two processes (not one after another from the last), the result stays
    # as good as the whole model at once
    assert plan.feasible and plan.objective >= 0.95 * whole.objective
    assert plan.plant_violation["P.maker"] <= 2e-3
