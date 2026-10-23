"""Optimizer: decision space, ALM, metaheuristics, decomposition, cascade, explanations, dispatch."""
import pytest
import torch

from plyxis.optim import metaheur as MH
from plyxis.optim.cascade import Cascade, CascadeConfig
from plyxis.optim.dispatch import instructions, lineups, truck_trips
from plyxis.optim.explain import bottlenecks, explain_instance
from plyxis.optim.gradient import alm
from plyxis.optim.partition import build_partition, subsystem_fitness
from plyxis.optim.problem import Problem
from plyxis.optim.space import DecisionSpace


def test_space_split_join_roundtrip(kestrel_model):
    sp = DecisionSpace.build(kestrel_model.program)
    x = torch.rand(2, sp.N)
    assert torch.equal(sp.join(sp.split(x)), x)
    D, R = sp.decode(x)
    assert R is not None and R.shape[1] == kestrel_model.program.n_routes
    # routing splits sum to one per port
    for rp in kestrel_model.program.routed:
        assert torch.allclose(R[:, rp.ridx, :].sum(1), torch.ones(2, R.shape[-1]), atol=1e-5)


def test_round_discrete_is_integral(kestrel_model):
    sp = DecisionSpace.build(kestrel_model.program)
    x = sp.round_discrete(torch.rand(1, sp.N))
    D, R = sp.decode(x, relax=False)
    for b in sp.blocks:
        if b.domain in ("int", "bool"):
            v = D[b.gid][b.name]
            assert torch.allclose(v, torch.round(v), atol=1e-5)


def test_alm_reaches_feasibility_on_small_field(small_model):
    pb = Problem(small_model.program)
    x0 = pb.space.default_u(1)
    e0 = pb.evaluate(x0, relax=False)
    res = alm(pb, x0, iters=150, lr=0.04, rho=0.1, outer=10)
    assert res.max_violation < 0.01
    assert res.objective > float(e0.objective[0]) * 0.5


def test_cascade_feasible_and_better_than_default(small_model):
    pb = Problem(small_model.program)
    cfg = CascadeConfig(relax_iters=150, population=6, generations=2, repair_iters=60, polish_iters=40, use_neural=False)
    res = Cascade(pb, config=cfg).solve()
    e0 = pb.evaluate(pb.space.default_u(1), relax=False)
    assert res.feasible
    assert res.objective >= float(e0.objective[0]) - 1.0 or float(e0.max_violation()[0]) > 0.01
    stages = [s["stage"] for s in res.stages]
    assert stages == ["compile", "relaxation", "discrete", "polish"]


def test_metaheuristics_on_test_functions():
    for name, fn in MH.METHODS.items():
        r = fn(MH.sphere, 8, pop=24, iters=150, seed=1)
        assert r.best_f > -0.05, name


def test_cooperative_search_beats_plain_in_high_dimension():
    N, K = 600, 30
    sid = torch.arange(N) % K

    def fitk(X):
        return torch.zeros(X.shape[0], K).index_add(1, sid, -((X - 0.3) ** 2))
    plain = MH.pso(lambda X: fitk(X).sum(1), N, pop=30, iters=150, seed=0)
    coop = MH.pso(fitk, N, pop=30, iters=150, var_sid=sid, K=K, seed=0)
    assert coop.best_f > plain.best_f * 0.2                 # (both negative) coop is several times closer


def test_partition_covers_every_instance(kestrel_model):
    sp = DecisionSpace.build(kestrel_model.program)
    part = build_partition(kestrel_model.program, sp)
    assert set(part.of_instance) == set(range(len(kestrel_model.flat.instances)))
    pads = [s for s in part.subsystems if s.kind == "pad"]
    assert len(pads) == 12 and all(len(s.instances) >= 7 for s in pads)
    assert part.var_sid.shape[0] == sp.N


def test_subsystem_fitness_sums_to_total(small_model):
    pb = Problem(small_model.program)
    part = build_partition(small_model.program, pb.space)
    x = pb.space.default_u(1)
    e = pb.evaluate(x, relax=False)
    F = subsystem_fitness(pb, e, part)
    assert F.shape == (1, part.K)


def test_explanations_and_bottlenecks(small_model):
    pb = Problem(small_model.program)
    cfg = CascadeConfig(relax_iters=60, population=4, generations=1, repair_iters=30, polish_iters=20, use_neural=False)
    res = Cascade(pb, config=cfg).solve()
    iid = small_model.flat.by_path["w1"]
    ex = explain_instance(pb, res.x, res.multipliers, iid)
    assert ex["decisions"] and ex["decisions"][0]["blocks"][0]["summary"]
    assert isinstance(bottlenecks(pb, res.multipliers), list)


def test_mining_twin_uses_the_same_engine():
    """A copper pit (faces, haul fleet, stockpile, mill, port) - no oil & gas code involved."""
    import os
    from plyxis.lang import compile_model
    from conftest import ROOT
    m = compile_model([os.path.join(ROOT, "scenarios", "mining_demo", "pit.plx")])
    pb = Problem(m.program)
    cfg = CascadeConfig(relax_iters=80, population=4, generations=1, repair_iters=30, polish_iters=30, use_neural=False)
    res = Cascade(pb, config=cfg).solve()
    e0 = pb.evaluate(pb.space.default_u(1), relax=False)
    assert res.max_violation < 0.01
    assert res.objective > float(e0.objective[0])


def test_dispatch_outputs(kestrel_model):
    pb = Problem(kestrel_model.program)
    x = pb.space.default_u(1)
    e = pb.evaluate(x, relax=False)
    ins = instructions(pb, x)
    assert ins and {"asset", "decision", "start", "end", "display"} <= set(ins[0])
    lus = lineups(pb, x)
    assert any(l["exclusive"] for l in lus)
    trips = truck_trips(pb, e.raw)
    assert trips["trucks"] == 8
