"""Extended oil & gas library: the new blueprints compile, simulate and optimise sensibly."""
import os

import pytest
import torch

from plyxis.lang import compile_model
from plyxis.optim.cascade import Cascade, CascadeConfig
from plyxis.optim.problem import Problem

HERE = os.path.dirname(os.path.abspath(__file__))
CHEAP = CascadeConfig(relax_iters=200, multistart=4, population=4, repair_iters=60, polish_iters=120, use_neural=False)


@pytest.fixture(scope="module")
def solved():
    m = compile_model([os.path.join(HERE, "models", "artificial_lift.plx")], env="lift_field", sim="lift_day", physical=True)
    pb = Problem(m.program)
    return m, pb, Cascade(pb, config=CHEAP).solve()


def _decision(pb, x, path, name):
    gid, li = pb.prog.inst_group[pb.prog.flat.by_path[path]]
    D, _ = pb.space.decode(x, pb.P, relax=False)
    return D[gid][name][0, li]


def _node(pb, x, path, key):
    gid, li = pb.prog.inst_group[pb.prog.flat.by_path[path]]
    with torch.no_grad():
        e = pb.evaluate(x, relax=False)
    v = pb.prog.signal(e.raw, gid, key)
    v = v.reshape(v.shape[0], -1, v.shape[-1])
    return v[0, li if v.shape[1] > 1 else 0]


def test_the_plan_is_feasible(solved):
    _, _, res = solved
    assert res.feasible and res.objective > 0


def test_scarce_lift_gas_goes_to_the_better_well(solved):
    _, pb, res = solved
    a = _decision(pb, res.x, "W1", "lift").mean()      # 1,500 bbl/d well
    b = _decision(pb, res.x, "W2", "lift").mean()      # 900 bbl/d well
    assert a > b
    total = _node(pb, res.x, "LC", "node:local:delivered").max()
    assert float(total) <= float(pb.P[pb.prog.inst_group[pb.prog.flat.by_path["LC"]][0]]["cap"].reshape(-1)[0]) * 1.01


def test_lift_gas_never_goes_far_beyond_its_optimum(solved):
    _, pb, res = solved
    opt = float(pb.P[pb.prog.inst_group[pb.prog.flat.by_path["W1"]][0]]["lift_opt"].reshape(-1)[0])
    assert float(_decision(pb, res.x, "W1", "lift").max()) <= 1.05 * opt


def test_pipeline_pressure_grows_with_the_square_of_the_flow(solved):
    m, pb, res = solved
    gid, li = pb.prog.inst_group[pb.prog.flat.by_path["LINE"]]
    P = {g: {k: v.clone() for k, v in d.items()} for g, d in pb.P.items()}
    q = _node(pb, res.x, "LINE", "node:local:flow")
    dp = _node(pb, res.x, "LINE", "node:local:dp")
    k = float(P[gid]["dp_design"].reshape(-1)[0]) / float(P[gid]["q_design"].reshape(-1)[0]) ** 2
    assert torch.allclose(dp, k * q * q, rtol=1e-3, atol=1e-3)
    assert float(dp.max()) <= float(P[gid]["dp_max"].reshape(-1)[0]) * 1.01


def test_vru_runs_when_gas_and_carbon_pay_for_its_power(solved):
    _, pb, res = solved
    assert float(_decision(pb, res.x, "VR", "on").mean()) >= 0.5


def test_plunger_cycles_stay_near_their_best(solved):
    _, pb, res = solved
    c = float(_decision(pb, res.x, "PL", "cycles").mean())
    assert 6.0 <= c <= 18.0


def test_kestrel_net_layout_is_untouched_by_unused_blueprints():
    from plyxis.optim import neural as NN
    root = os.path.dirname(HERE)
    sdir = os.path.join(root, "scenarios", "kestrel")
    m = compile_model([os.path.join(sdir, "field.plx")])
    graph = NN.TwinGraph(Problem(m.program))
    net = NN.SubSolverNet(graph, d=32, rounds=3, slot=4)
    own = net.state_dict()
    saved = torch.load(os.path.join(sdir, "models", "subsolver.pt"), map_location="cpu", weights_only=False)["state"]
    bad = [k for k, v in saved.items() if k not in own or own[k].shape != v.shape]
    assert not [k for k in bad if k != "base_logit"]
