"""Structure recognition (plyxis/optim/structure.py) and exact LP branches (branchlp.py)."""
import json
import os

import pytest
import torch

from conftest import ROOT


@pytest.fixture(scope="module")
def corvus():
    from plyxis.lang import compile_model
    from plyxis.optim.problem import Problem
    sd = os.path.join(ROOT, "scenarios", "corvus")
    sc = json.load(open(os.path.join(sd, "scenario.json")))
    m = compile_model([os.path.join(sd, f) for f in sc["twin"]["files"]], env=sc["twin"]["env"])
    pb = Problem(m.program)
    pb.calibrate_scales()
    return pb


def test_curvature_rules():
    from plyxis.lang.behavior import C, Op, Ref
    from plyxis.optim.structure import Form, _join
    a, cvx = Form("affine", "+"), Form("convex", "+")
    assert _join(a, a, "add").curv == "affine"
    assert _join(a, cvx, "add").curv == "convex"
    assert _join(cvx, cvx.neg(), "add").curv == "nonlinear"            # convex + concave: no sign
    assert cvx.scaled("-").curv == "concave" and cvx.scaled("?").curv == "nonlinear"
    del C, Op, Ref


def test_mine_classes(corvus):
    from plyxis.optim.structure import classify_types
    cls = {t.type: t.cls for t in classify_types(corvus.prog).values()}
    assert cls["pump_station"] == "lp"                  # sump + pump against a price: a linear programme
    assert cls["vent_district"] == "lp"                 # regulator air allocation
    assert cls["stope"] == "slp"                        # bilinear (blast x broken), piecewise affine once blasts are fixed
    assert cls["orepass"] == "slp_quality"              # blended grade (pooling): affine once the grade is frozen
    assert cls["main_fan"] == "convex"                  # power ~ speed^3, at a price of the current (positive) sign


def test_wells_are_convex():
    """On/off fixed, a well's rate is affine in its speed (the pwl has one piece on the speed range), power and wear are
    convex, health is concave and the failure risk convex in it: the oil field's wells are a convex class, not nonlinear."""
    from plyxis.lang import compile_model
    from plyxis.optim.structure import Analyser, classify_types
    m = compile_model([os.path.join(ROOT, "scenarios", "kestrel", "field.plx")])
    types = {t.type: t for t in classify_types(m.program).values()}
    assert types["esp_well"].cls == "convex", types["esp_well"].reasons
    assert types["truck_fleet"].cls == "lp" and types["compressor_station"].cls == "slp"
    A = Analyser(m.program, frozen_quality=True, fixed_discrete=True)
    g = next(g for g in m.program.comp.groups if not g.is_env and g.ir.type.short == "esp_well")
    assert A.node(g, "out:prod:oil").curv == "affine"
    assert A.node(g, "next:health").curv == "concave"
    assert A.node(g, "term:cost:risk").curv == "convex"
    assert A.node(g, "term:cost:pump_wear").curv == "convex"
    assert A.node(g, "term:cost:power").curv == "convex"
    assert "power_price" in A.assumed                    # the verdict rests on the price's sign, and says so


def test_reverse_mode_agrees_with_finite_differences_at_the_bounds():
    """Every first-order stage trusts these slopes. Half of a plan's well speeds sit exactly at the upper bound."""
    from plyxis.lang import compile_model
    from plyxis.optim import branchlp as BL
    from plyxis.optim.problem import Problem
    m = compile_model([os.path.join(ROOT, "scenarios", "kestrel", "field.plx")])
    pb = Problem(m.program)
    var = BL.class_variables(pb, ("convex",))[::97]
    x = pb.space.round_discrete(pb.space.default_u(1))
    x[0, var] = 1.0
    X = x.clone().requires_grad_(True)
    rev = torch.autograd.grad(pb.evaluate(X, relax=False).objective.sum(), X)[0][0, var]
    with torch.no_grad():
        h = 0.01
        Y = x.repeat(len(var), 1)
        Y[torch.arange(len(var)), torch.as_tensor(var)] -= h
        fd = (pb.evaluate(x, relax=False).objective - pb.evaluate(Y, relax=False).objective) / h
    assert torch.max(torch.abs(rev - fd)) < 0.05 * torch.max(torch.abs(fd))


def test_curvature_of_a_state_is_verified_by_induction():
    """A concave update that is fed by the state's own convex function must not pass as concave."""
    from plyxis.lang.behavior import C, Op, Ref
    from plyxis.optim.structure import Analyser, Form

    class _G:                                            # the smallest asset the analyser reads
        gid, is_env = "g", False

        class ir:
            nodes = {"next:h": Op("sub", (Op("mul", (C(0.5), Op("pow", (Ref("state", "h"), C(2.0))))), Op("pow", (Ref("dec", "u"), C(2.0)))))}

            class S:
                has_update, lo = True, None
            states = {"h": S}
    A = Analyser.__new__(Analyser)
    A.forms, A._busy, A._assume, A.assumed, A.frozen, A.fixed, A.quality_states = {}, set(), {}, set(), False, False, {}
    A.dec_form, A.dec_range = {("g", "u"): Form("affine", "+")}, {}
    f = A._state(_G, "h")
    assert f.curv == "nonlinear"


def test_tangent_gap_of_an_affine_branch_is_its_gain(corvus):
    from plyxis.optim import branchlp as BL
    pb = corvus
    x = pb.space.round_discrete(pb.space.default_u(1))
    var = BL.class_variables(pb, ("lp",))[:48]
    gap = BL.tangent_gap(pb, x, var, BL.jacobian(pb, x, var))
    res = BL.solve_branches(pb, x, var, affine=True)
    assert gap == pytest.approx(res.F - res.F0, abs=1e-3)   # the LP's promise is what the twin confirms
    assert BL.tangent_gap(pb, res.x, var, BL.jacobian(pb, res.x, var)) < 1e-3


def test_branch_lp_is_exact_and_verified(corvus):
    from plyxis.optim import branchlp as BL
    pb = corvus
    x = pb.space.round_discrete(pb.space.default_u(1))
    var = BL.class_variables(pb, ("lp",))
    assert len(var) > 0
    jac = BL.jacobian(pb, x, var[:48])
    # affine branch: the finite-difference Jacobian predicts the twin exactly for a large step
    d = torch.zeros_like(x)
    d[0, var[:48]] = torch.as_tensor(0.3 * (0.5 - x[0, var[:48]].numpy()), dtype=x.dtype)
    import numpy as np
    pred = jac["h0"] + jac["J"] @ d[0, var[:48]].numpy()
    with torch.no_grad():
        e = pb.evaluate(x + d, relax=False)
    o, hard, soft = BL._pieces(pb, e)
    real = np.array([float(BL._flat(hard[m[1:3]] if m[0] == "h" else soft[m[1:3]], 1)[0][m[3]]) for m in jac["meta"]])
    assert np.max(np.abs(pred - real)) < 1e-3 * max(1.0, np.max(np.abs(real)))
    res = BL.solve_branches(pb, x, var, affine=True)
    assert res.F >= res.F0 - 1e-9                        # never worse; verified on the full twin
