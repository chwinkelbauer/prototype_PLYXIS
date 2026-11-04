"""Distributions of a plan's outcomes: propagation (plyxis/optim/risk.py) against Monte Carlo.

    python -m bench.risk                     # both twins, ~5 min
    python -m bench.risk --ref 8192 kestrel

For a plan of the live cascade (scenario planner budget, no neural start), every calibratable property is
uncertain (log-normal, the engineering prior sigma) and prices follow log random walks. The reference is
Monte Carlo with `--ref` samples. Reported per twin:

    limits      over every hard-constraint residual (asset x hour) with a spread: ratio of the propagated to
                the reference standard deviation, error of the violation probability (sigma points only)
    plan value  mean, standard deviation and 5 % quantile, and their error against the reference over five
                sampling seeds: sigma points alone, sigma points + N Sobol points for the remainder, and plain
                Monte Carlo with the same total number of simulations

Writes bench/results/risk.json.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from plyxis.lang import compile_model  # noqa: E402
from plyxis.optim import risk as RK  # noqa: E402
from plyxis.optim.cascade import Cascade  # noqa: E402
from plyxis.optim.problem import Problem  # noqa: E402
from plyxis.runtime.planner import Planner  # noqa: E402

PRICE_SIGMA = 0.02          # log price per sqrt(hour): about +-14 % after two days
SEEDS = 5


def _q(v: torch.Tensor, q: float) -> float:
    return float(torch.quantile(v.double(), q)) if v.numel() else float("nan")


def _rms(xs: list, ref: float) -> float:
    return math.sqrt(sum((x - ref) ** 2 for x in xs) / len(xs))


def compare(pb: Problem, x: torch.Tensor, unc: RK.Uncertainty, n_ref: int, n_rem: int) -> dict:
    ref = RK.monte_carlo(pb, x, unc, n=n_ref, seed=12345)
    v = ref["value"]
    R = {"mean": float(v.mean()), "std": float(v.std()), "q05": _q(v, 0.05)}
    res = {"reference": {**R, "samples": n_ref, "seconds": ref["seconds"]}}
    # limits: sigma points
    out = RK.propagate(pb, x, unc, samples=0)
    ratios, perr = [], []
    for key, r in out.rows.items():
        m = ref["rows"].get(key)
        if m is None:
            continue
        s_mc, s_pr = m["std"].reshape(-1), r["std"].reshape(-1).double()
        live = s_mc > 1e-6 * max(float(s_mc.max()), 1e-12)
        if live.any():
            ratios.append(s_pr[live] / s_mc[live])
        p_pr, p_mc = out.prob_violation(key).reshape(-1).double(), m["p_violation"].reshape(-1)
        sel = (p_mc > 0.02) & (p_mc < 0.98)
        if sel.any():
            perr.append((p_pr[sel] - p_mc[sel]).abs())
    ratios = torch.cat(ratios) if ratios else torch.zeros(0)
    perr = torch.cat(perr) if perr else torch.zeros(0)
    res["limits"] = {"factors": out.K, "simulations": out.simulations, "seconds": out.seconds,
                     "rows_with_spread": int(ratios.numel()), "std_ratio_median": _q(ratios, 0.5),
                     "std_ratio_p10": _q(ratios, 0.1), "std_ratio_p90": _q(ratios, 0.9),
                     "rows_uncertain": int(perr.numel()), "p_violation_abs_error_mean": float(perr.mean()) if perr.numel() else None,
                     "p_violation_abs_error_p90": _q(perr, 0.9)}
    # plan value: three estimators at equal cost, over sampling seeds
    sig = {"mean": out.value["mean"], "std": out.value["std"], "q05": out.quantile(0.05)}
    hyb, mc = {"mean": [], "std": [], "q05": []}, {"mean": [], "std": [], "q05": []}
    n_hyb = 0
    for s in range(SEEDS):
        o = RK.propagate(pb, x, unc, samples=n_rem, seed=s)
        n_hyb = o.simulations
        for k, val in (("mean", o.value["mean"]), ("std", o.value["std"]), ("q05", o.quantile(0.05))):
            hyb[k].append(val)
        m = RK.monte_carlo(pb, x, unc, n=n_hyb, seed=100 + s)["value"]
        for k, val in (("mean", float(m.mean())), ("std", float(m.std())), ("q05", _q(m, 0.05))):
            mc[k].append(val)
    res["value"] = {"simulations_each": n_hyb, "remainder_samples": n_rem,
                    "sigma_points_only": {k: {"estimate": sig[k], "error": sig[k] - R[k]} for k in R},
                    "sigma_points_and_sobol": {k: {"rms_error": _rms(hyb[k], R[k]), "estimates": hyb[k]} for k in R},
                    "monte_carlo_same_cost": {k: {"rms_error": _rms(mc[k], R[k]), "estimates": mc[k]} for k in R}}
    return res


def run(scen: str, n_ref: int, n_rem: int, seed: int = 0) -> dict:
    sd = os.path.join(ROOT, "scenarios", scen)
    sc = json.load(open(os.path.join(sd, "scenario.json")))
    m = compile_model([os.path.join(sd, f) for f in sc["twin"]["files"]], env=sc["twin"]["env"])
    pb = Problem(m.program)
    cfg = dataclasses.replace(Planner.LIVE, **{**sc.get("planner", {}), "use_neural": False, "seed": seed})
    t = time.perf_counter()
    x = Cascade(pb, config=cfg).solve().x[:1]
    out = {"scenario": scen, "plan_seconds": time.perf_counter() - t}
    prices = {k: PRICE_SIGMA for k in m.program.flat.env_inputs if k.endswith("price")}
    for name, unc in (("properties", RK.Uncertainty.prior(m.program, 0.15)),
                      ("properties_and_prices", RK.Uncertainty.prior(m.program, 0.15, prices))):
        r = out[name] = compare(pb, x, unc, n_ref, n_rem)
        L, V, Rf = r["limits"], r["value"], r["reference"]
        print(f"{scen} {name}: {L['factors']} factors | limits: std ratio median {L['std_ratio_median']:.3f} "
              f"(p10 {L['std_ratio_p10']:.3f}, p90 {L['std_ratio_p90']:.3f}), P(violation) error {L['p_violation_abs_error_mean']:.3f} "
              f"over {L['rows_uncertain']} rows, {L['simulations']} sims {L['seconds']:.1f} s", flush=True)
        print(f"   value reference mean {Rf['mean']:,.0f} std {Rf['std']:,.0f} q05 {Rf['q05']:,.0f} ({Rf['samples']} samples) | "
              f"sigma points only: std {V['sigma_points_only']['std']['estimate']:,.0f} | with {V['remainder_samples']} Sobol "
              f"({V['simulations_each']} sims): rms error mean {V['sigma_points_and_sobol']['mean']['rms_error']:,.0f} "
              f"std {V['sigma_points_and_sobol']['std']['rms_error']:,.0f} q05 {V['sigma_points_and_sobol']['q05']['rms_error']:,.0f} | "
              f"plain MC same cost: mean {V['monte_carlo_same_cost']['mean']['rms_error']:,.0f} std {V['monte_carlo_same_cost']['std']['rms_error']:,.0f} "
              f"q05 {V['monte_carlo_same_cost']['q05']['rms_error']:,.0f}", flush=True)
    return out


def frontier(scen: str, fractions=(1.0, 0.9, 0.8, 0.7, 0.6), seed: int = 0) -> dict:
    """Expected plan value against its standard deviation: a plan of the live cascade, polished under a cap on the
    value's spread (`RiskProblem`) at fractions of the uncapped spread. Inputs: the engineering prior (0.15) on
    every calibratable property and PRICE_SIGMA on prices. The founder picks the cap from this curve."""
    from plyxis.optim.gradient import alm
    sd = os.path.join(ROOT, "scenarios", scen)
    sc = json.load(open(os.path.join(sd, "scenario.json")))
    m = compile_model([os.path.join(sd, f) for f in sc["twin"]["files"]], env=sc["twin"]["env"])
    pb = Problem(m.program)
    cfg = dataclasses.replace(Planner.LIVE, **{**sc.get("planner", {}), "use_neural": False, "seed": seed})
    x0 = Cascade(pb, config=cfg).solve().x[:1]
    unc = RK.Uncertainty.prior(m.program, 0.15, {k: PRICE_SIGMA for k in m.program.flat.env_inputs if k.endswith("price")})
    base = RK.propagate(pb, x0, unc, samples=128)
    mask = pb.space.discrete_mask()
    pts = []
    for f in fractions:
        t = time.perf_counter()
        x = x0
        if f < 1.0:
            rp = RK.RiskProblem(pb, base, unc, cap=f * base.value["std"], r=4)
            x = alm(rp, x0, iters=cfg.polish_iters * 2, lr=0.01, rho=1.0, outer=10, mask=~mask, relax=False, stage="risk").x[:1]
        o = RK.propagate(pb, x, unc, samples=128)
        with torch.no_grad():
            mv = float(pb.evaluate(x, relax=False).max_violation()[0])
        pts.append({"cap_fraction": f, "mean": o.value["mean"], "std": o.value["std"], "q05": o.quantile(0.05),
                    "nominal": o.value["f0"], "max_violation": mv, "seconds": time.perf_counter() - t})
        print(f"{scen} cap {f:.2f} x sigma: expected value {o.value['mean']:,.0f}  std {o.value['std']:,.0f}  "
              f"5 % quantile {o.quantile(0.05):,.0f}  violation {mv:.4f}", flush=True)
    return {"scenario": scen, "points": pts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="*", default=["kestrel", "corvus"])
    ap.add_argument("--ref", type=int, default=8192)
    ap.add_argument("--remainder", type=int, default=128)
    ap.add_argument("--frontier", action="store_true", help="only the expected value / spread trade-off (into risk.json)")
    args = ap.parse_args()
    if args.frontier:
        path = os.path.join(ROOT, "bench", "results", "risk.json")
        old = json.load(open(path)) if os.path.exists(path) else {}
        old["frontier"] = [frontier(s) for s in args.scenarios]
        with open(path, "w") as fh:
            json.dump(old, fh, indent=1, default=float)
        return
    res = {"price_sigma_per_sqrt_h": PRICE_SIGMA, "property_sigma": 0.15, "seeds": SEEDS,
           "runs": [run(s, args.ref, args.remainder) for s in args.scenarios]}
    path = os.path.join(ROOT, "bench", "results", "risk.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(res)
    with open(path, "w") as fh:
        json.dump(old, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
