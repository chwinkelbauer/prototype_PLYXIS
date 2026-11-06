"""The exact layer: the twin's equations as algebra (algebra.py), HiGHS on them (exact.py), and the reduction engine
(reduce.py) - every result checked against the torch simulator or the whole-model MILP."""
import os

import numpy as np
import pytest
import torch

from plyxis.lang import compile_model
from plyxis.optim.algebra import check, linearize
from plyxis.optim.exact import fitness, model_F, slp, solve
from plyxis.optim.problem import Problem
from plyxis.optim.reduce import Engine, contract

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pb(path, sim=None):
    m = compile_model([os.path.join(ROOT, path)], sim=sim)
    return m, Problem(m.program)


def _state_rel(lm) -> float:
    """Largest state-update residual at the reference, relative to the size of its own trajectory."""
    worst = 0.0
    for nid, (off, n, T, gid) in lm.lifted.items():
        vals = lm.x_ref[off:off + n * T].reshape(n, T)
        ci = [i for i, c in enumerate(lm.cons) if c[0] in ("state", "delay") and
              (f"{c[1]}|{c[2]}" == nid if c[0] == "state" else c[1] == nid)]
        if ci:
            r = lm.r0[lm.row_con == ci[0]].reshape(n, T)
            worst = max(worst, float((np.abs(r) / np.maximum(np.abs(vals).max(1, keepdims=True), 1.0)).max()))
    return worst


@pytest.mark.parametrize("path,sim", [("scenarios/kestrel/field.plx", None), ("scenarios/aquila/field.plx", None),
                                      ("tests/models/kilns.plx", None)])
def test_algebraic_model_is_the_simulators_own(path, sim):
    m, pb = _pb(path, sim)
    x = torch.rand(1, pb.space.N, generator=torch.Generator().manual_seed(3))
    lm = linearize(pb, x)
    c = check(pb, lm, eps=0.02)
    ref = c["at_reference"]
    assert abs(ref["objective_model"] - ref["objective_twin"]) <= 1e-5 * abs(ref["objective_twin"]) + 1.0
    assert ref["rows_max_rel_err"] < 1e-3                       # float32 simulator against float64 algebra
    assert _state_rel(lm) < 1e-5
    assert c["rows_affine"]["max_err_rel_to_max_change"] < 1e-2  # affine rows: exact up to float32 noise


def test_one_lp_is_the_optimum_of_an_affine_twin():
    m, pb = _pb("examples/plants_toy.plx")
    lm = linearize(pb)
    assert lm.describe()["rows_by_level"]["nonlinear"] == 0
    s = solve(lm)
    F, viol, obj = fitness(pb, np.clip(s.x[:lm.N], 0, 1))
    # peak = 0 all day: 600 t made at 40, 50 t from the store, 70 t bought at 70, 720 t sold at 100
    assert obj == pytest.approx(43_100.0, abs=1.0)
    assert viol < 1e-6
    assert F == pytest.approx(s.F, abs=1e-6)
    r = slp(pb, iters=5)                                       # the sequential LP stops at the same plan
    assert r.objective == pytest.approx(43_100.0, abs=1.0)


def test_piecewise_terms_of_discrete_decisions_are_exact():
    """Start counts and minimum runs (pos() of on/off decisions) are encoded with indicators: the whole MILP's
    plan is worth on the twin exactly what the model says, and keeps every rule."""
    m, pb = _pb("tests/models/kilns.plx")
    lm = linearize(pb)
    d = lm.describe()
    assert d["exact_piecewise_terms"] > 0 and d["rows_by_level"]["piecewise_affine"] == 0
    w = solve(lm, integer=True)
    assert w.status == "optimal"
    F, viol, obj = fitness(pb, w.x[:lm.N])
    assert viol < 1e-6
    assert F == pytest.approx(w.F, abs=1e-6)


def test_engine_finds_the_units_and_certifies_its_plan():
    m, pb = _pb("tests/models/kilns.plx")
    lm = linearize(pb)
    st = contract(lm, int_cap=90, flat=m.program.flat)                    # one kiln's integers per block
    assert sorted(b.name for b in st["blocks"]) == ["K1", "K2", "K3"]      # the kilns, found - nothing declared
    assert all(b.pattern == "unit" for b in st["blocks"])
    big = contract(lm, int_cap=200, flat=m.program.flat)                   # room for two: the strongest pair merges
    assert sorted(len(b.atoms) for b in big["blocks"]) == [1, 2]
    w = solve(lm, integer=True)
    eng = Engine(lm, int_cap=90)
    r = eng.solve()
    assert r.bound >= w.F - 1e-6                    # a valid bound: no plan can beat it
    assert r.F <= w.F + 1e-6
    assert r.gap < 0.02
    F, viol, _ = fitness(pb, r.x[:lm.N])
    assert viol < 1e-6 and F == pytest.approx(r.F, abs=1e-6)
    assert model_F(lm, r.x) == pytest.approx(r.F)


def test_alm_returns_the_iterate_it_scored():
    """The ALM keeps its best iterate by (feasible, objective). It used to copy the plan after the optimiser step
    had moved it in place, so a feasible optimum given as the start came back one step away and worse (the steel
    group: a plan worth USD 1,394.7 M returned as 1,387.2 M with violation 0.29)."""
    from plyxis.optim.gradient import alm
    m, pb = _pb("examples/plants_toy.plx")
    lm = linearize(pb)
    s = solve(lm)
    x0 = torch.as_tensor(np.clip(s.x[:lm.N], 0, 1), dtype=torch.float32).view(1, -1)
    e0 = pb.evaluate(x0, relax=True)
    r = alm(pb, x0, iters=5, lr=0.05, rho=0.05, outer=10, tol=0.01)
    assert r.max_violation <= 0.01
    assert r.objective >= float(e0.objective[0]) - 1e-3 * abs(float(e0.objective[0]))


def test_warm_started_lps_reach_the_same_plan():
    """The sequential LP's LPs from the last basis (HiGHS directly, every row kept): the same plan as cold LPs."""
    m, pb = _pb("examples/plants_toy.plx")
    cold = slp(pb, iters=6)
    warm = slp(pb, iters=6, warm_lp=True)
    assert warm.objective == pytest.approx(cold.objective, abs=1.0)
    assert warm.max_violation < 1e-6


def test_polish_keeps_the_discrete_decisions():
    """fix_discrete: on/off decisions stay where they were; only the continuous ones move."""
    m, pb = _pb("tests/models/kilns.plx")
    x0 = pb.space.default_u(1)
    r = slp(pb, x0, iters=5, fix_discrete=True, warm_lp=True)
    dm = pb.space.discrete_mask()
    assert torch.equal(r.x.view(-1)[dm], pb.space.round_discrete(x0).view(-1)[dm])
    F0, _, _ = fitness(pb, x0)
    assert r.F >= F0 - 1e-9
