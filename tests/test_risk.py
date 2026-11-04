"""Planning under uncertainty (plyxis/optim/risk.py), the estimator's covariance and the price uncertainty."""
import math
import os

import pytest
import torch

from conftest import ROOT


@pytest.fixture(scope="module")
def kestrel():
    from plyxis.lang import compile_model
    from plyxis.optim.problem import Problem
    m = compile_model([os.path.join(ROOT, "scenarios", "kestrel", "field.plx")])
    pb = Problem(m.program)
    return m, pb


def test_limits_match_sampling(kestrel):
    """Sigma points per factor give every limit's spread: within a few per cent of Monte Carlo."""
    from plyxis.optim import risk as RK
    m, pb = kestrel
    x = pb.space.round_discrete(pb.space.default_u(1))
    unc = RK.Uncertainty.prior(m.program, 0.10, {"power_price": 0.02})
    out = RK.propagate(pb, x, unc, samples=0)
    mc = RK.monte_carlo(pb, x, unc, n=1024)
    ratios = []
    for key, r in out.rows.items():
        s_mc = mc["rows"][key]["std"].reshape(-1)
        live = s_mc > 1e-4 * max(float(s_mc.max()), 1e-12)
        if live.any():
            ratios.append(r["std"].reshape(-1).double()[live] / s_mc[live])
    ratios = torch.cat(ratios)
    assert ratios.numel() > 100
    assert 0.9 < float(ratios.median()) < 1.1
    assert out.simulations == 2 * out.K + 1


def test_backoffs_tighten_and_are_monotone(kestrel):
    from plyxis.optim import risk as RK
    m, pb = kestrel
    x = pb.space.round_discrete(pb.space.default_u(1))
    out = RK.propagate(pb, x, RK.Uncertainty.prior(m.program, 0.10), samples=0)
    b05, b20 = RK.backoffs(out, 0.05), RK.backoffs(out, 0.20)
    assert all(float(b05[k].min()) >= 0 for k in b05)
    assert all(bool((b05[k] >= b20[k] - 1e-6).all()) for k in b05)          # a smaller eps never loosens a limit
    with torch.no_grad():
        e0 = pb.evaluate(x, relax=False)
        pb.set_backoff(b05)
        e1 = pb.evaluate(x, relax=False)
        pb.set_backoff(None)
    assert float(e1.sum_violation()[0]) >= float(e0.sum_violation()[0])


def test_value_sigma_is_a_constraint(kestrel):
    from plyxis.optim import risk as RK
    m, pb = kestrel
    x = pb.space.round_discrete(pb.space.default_u(1))
    unc = RK.Uncertainty.prior(m.program, 0.10, {"oil_price": 0.02})
    out = RK.propagate(pb, x, unc, samples=0)
    rp = RK.RiskProblem(pb, out, unc, cap=0.5 * out.value["std"], r=3)
    with torch.no_grad():
        e = rp.evaluate(x.repeat(2, 1), relax=False)
    s = e.totals["value_sigma"]
    assert s.shape == (2,) and float(s[0]) == pytest.approx(float(s[1]))
    assert float(s[0]) == pytest.approx(out.value["std"], rel=0.25)          # r directions + the constant rest
    assert float(e.resid[("risk", "value_sigma")][0, 0, 0]) > 0              # cap below the spread: violated


def test_price_sigma_from_observations():
    from plyxis.data.forecast import Forecaster
    g = torch.Generator().manual_seed(3)
    f = Forecaster(["power_price"], {"power_price": 50.0})
    lv = math.log(50.0)
    for h in range(24 * 6):
        lv += 0.03 * float(torch.randn(1, generator=g))
        season = 0.2 * math.sin(2 * math.pi * (h % 24) / 24)              # a daily profile is not uncertainty
        f.observe("power_price", float(h), math.exp(lv + season))
    assert f.log_sigma("power_price") == pytest.approx(0.03, rel=0.25)
    assert Forecaster(["gas_price"], {"gas_price": 2.0}).log_sigma("gas_price", default=0.02) == 0.02


def test_estimator_covariance_learns():
    """After some hours of data the posterior of the wells' rates is tighter than the engineering prior, positive
    definite, and handed to planning with the directly tested water cuts."""
    from plyxis.runtime.engine import Engine
    eng = Engine(os.path.join(ROOT, "scenarios", "kestrel"), seed=0, sync_plans=True, use_neural=False)
    eng.run_headless(8)
    est = eng.twin.estimator
    assert est.cov_info.get("measurements", 0) > 0
    sd = est.cov.diagonal().sqrt()
    q = [i for i, (g, p, li) in enumerate(est.cov_index) if p == "q_max"]
    assert float(sd[q].median()) < 0.15
    assert float(torch.linalg.eigvalsh(est.cov).min()) > 0
    unc = est.uncertainty_model({"oil_price": 0.01})
    assert len(unc.params) >= len(est.cov_index) and unc.cov.shape[0] == len(unc.params)
