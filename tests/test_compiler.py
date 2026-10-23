"""Code generator correctness: compiled time-vectorised simulator == reference interpreter."""
import os

import pytest
import torch

from plyxis.lang import compile_model
from plyxis.lang.interp import Interpreter

from conftest import MODELS


def _random_inputs(prog, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    D = prog.default_decisions(B=1, T=T)
    for gid in D:
        for k in D[gid]:
            D[gid][k] = D[gid][k] * (0.3 + 1.4 * torch.rand(D[gid][k].shape, generator=g, dtype=D[gid][k].dtype))
    R = torch.rand(1, prog.n_routes, T, generator=g, dtype=prog.dtype)
    for rp in prog.routed:
        s = R[:, rp.ridx, :].sum(1, keepdim=True)
        R[:, rp.ridx, :] = R[:, rp.ridx, :] / s
    return D, R


def _compare(prog, physical, T=12, E=None, tol=1e-9):
    prog.to_dtype(torch.float64)
    D, R = _random_inputs(prog, T)
    raw = prog.simulate(D, R, T=T, dt=1.0, t0=3.0, E=E)
    ref = Interpreter(prog, physical=physical).run(D, R, T, 1.0, t0=3.0, E=E)
    worst = 0.0
    for (gid, k), v in ref.items():
        x = raw[prog.comp.nodes[f"{gid}|{k}"].var]
        if x.dim() == 3 and (x.shape[-1] == 1 or x.shape[1] != v.shape[1]):
            x = x.expand(1, v.shape[1], T)
        worst = max(worst, float((x - v).abs().max()))
    prog.to_dtype(torch.float32)
    return worst


@pytest.mark.parametrize("physical", [False, True])
def test_stress_model_matches_interpreter(physical):
    m = compile_model([os.path.join(MODELS, "stress.plx")], include_stdlib=False, physical=True)
    prog = m.physical if physical else m.program
    kinds = [b.kind for b in prog.comp.blocks]
    assert "scan" in kinds and "acc" in kinds                    # all scheduling paths exercised
    E = {"diesel": torch.linspace(1.0, 1.5, 30, dtype=torch.float64)}
    assert _compare(prog, physical, T=30, E=E) < 1e-9


def test_kestrel_matches_interpreter(kestrel_model):
    assert _compare(kestrel_model.program, False, T=6) < 1e-6


def test_planning_and_physical_share_grouping(kestrel_model):
    a = [(g.gid, g.key, g.level, len(g.inst_ids)) for g in kestrel_model.program.groups]
    b = [(g.gid, g.key, g.level, len(g.inst_ids)) for g in kestrel_model.physical.groups]
    assert a == b


def test_time_vectorisation_has_no_scans_for_kestrel(kestrel_model):
    kinds = [b.kind for b in kestrel_model.program.comp.blocks]
    assert "scan" not in kinds and kinds.count("acc") >= 3


def test_gradients_flow_through_the_twin(small_model):
    from plyxis.optim.problem import Problem
    pb = Problem(small_model.program)
    x = pb.space.default_u(1).requires_grad_(True)
    pb.evaluate(x).objective.sum().backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0


def test_exogenous_state_trajectories(small_model):
    """States without dynamics can be replayed as observed trajectories [B, n, T]."""
    prog = small_model.physical
    T = 4
    D = prog.default_decisions(B=1, T=T)
    S0 = {g: dict(d) for g, d in prog.S0.items()}
    gw = next(g.gid for g in prog.groups if g.ir.type.short == "well")
    up = torch.ones(1, 3, T)
    up[0, 0, 2:] = 0.0                                          # well 1 trips at step 2
    S0[gw]["up"] = up
    raw = prog.simulate(D, None, S0=S0, T=T, dt=1.0)
    liq = raw[prog.node_var(gw, "local:liquid")]
    assert float(liq[0, 0, 1]) > 0 and float(liq[0, 0, 3]) == 0.0


def test_generated_code_is_python(small_model):
    src = small_model.program.source
    compile(src, "<gen>", "exec")
    assert "def simulate" in src and "rt.gather_sum" in src
