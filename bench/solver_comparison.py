"""Solver comparison on the Kestrel twin (same problem, wall-clock budgets).

Methods
  pso / gravisearch / ga / de      classic population metaheuristics on the full decision vector
  pso-coop / ga-coop               the same with per-subsystem credit assignment (decomposition)
  alm                              single-start augmented Lagrangian on the differentiable twin
  cascade                          PLYXIS cascade without neural sub-solvers
  cascade+neural                   PLYXIS cascade with neural sub-solvers

Every final solution is evaluated identically: discrete decisions rounded, exact simulation,
objective (72 h profit minus carbon cost, USD) and max scaled constraint violation.

    python -m bench.solver_comparison --budget 60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from plyxis.lang import compile_model  # noqa: E402
from plyxis.optim import metaheur as MH  # noqa: E402
from plyxis.optim import neural as NN  # noqa: E402
from plyxis.optim.cascade import Cascade, CascadeConfig  # noqa: E402
from plyxis.optim.gradient import alm  # noqa: E402
from plyxis.optim.partition import build_partition, subsystem_fitness  # noqa: E402
from plyxis.optim.problem import Problem  # noqa: E402


def final_eval(pb, x):
    with torch.no_grad():
        e = pb.evaluate(pb.space.round_discrete(x.view(1, -1)), relax=False)
    return float(e.objective[0]), float(e.max_violation()[0]), float(e.sum_violation()[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=60.0)
    ap.add_argument("--pop", type=int, default=48)
    args = ap.parse_args()
    torch.manual_seed(0)
    kdir = os.path.join(ROOT, "scenarios", "kestrel")
    m = compile_model([os.path.join(kdir, "field.plx")])
    pb = Problem(m.program)
    sp = pb.space
    part = build_partition(m.program, sp)
    pb.evaluate(sp.default_u(2))                      # warm-up
    N, mu = sp.N, 10.0
    x0 = sp.default_u(1)
    curves, table = {}, []

    def fit_scalar(X):
        return pb.fitness(X, mu=mu, relax=False)

    def fit_sub(X):
        with torch.no_grad():
            e = pb.evaluate(sp.round_discrete(X), relax=False)
            return subsystem_fitness(pb, e, part, mu=mu)

    def record(name, res_x, t, curve):
        obj, mv, sv = final_eval(pb, res_x)
        table.append({"method": name, "objective": obj, "max_violation": mv, "sum_violation": sv, "seconds": t})
        curves[name] = curve
        print(f"{name:18s} objective {obj:14,.0f}  max viol {mv:8.4f}  {t:6.1f}s", flush=True)

    for name, fn, kw in (("pso", MH.pso, {}), ("gravisearch", MH.gravisearch, {}), ("ga", MH.ga, {}),
                         ("de", MH.de, {})):
        t0 = time.perf_counter()
        r = fn(fit_scalar, N, pop=args.pop, iters=10 ** 6, x0=x0, time_budget=args.budget, log_every=2, **kw)
        record(name, r.best_x, time.perf_counter() - t0,
               [[h["t"], h["fitness"] * pb.J_scale] for h in r.history])
    for name, fn in (("pso-coop", MH.pso), ("ga-coop", MH.ga)):
        t0 = time.perf_counter()
        r = fn(fit_sub, N, pop=args.pop, iters=10 ** 6, x0=x0, var_sid=part.var_sid, K=part.K,
               time_budget=args.budget, log_every=2)
        record(name, r.best_x, time.perf_counter() - t0,
               [[h["t"], h["fitness"] * pb.J_scale] for h in r.history])
    # single-start gradient (ALM) then round + short repair
    t0 = time.perf_counter()
    rel = alm(pb, x0, iters=220, lr=0.04, rho=0.05, outer=10, anneal=0.05)
    mask = sp.discrete_mask()
    rep = alm(pb, sp.round_discrete(rel.x), iters=90, lr=0.02, rho=0.5, outer=10, mask=~mask, relax=False)
    record("alm", rep.x, time.perf_counter() - t0, [[h["t"], h["objective"]] for h in rel.history])
    # cascade without / with neural sub-solvers
    for name, use_nn in (("cascade", False), ("cascade+neural", True)):
        net = None
        if use_nn:
            path = os.path.join(kdir, "models", "subsolver.pt")
            if not os.path.exists(path):
                continue
            net = NN.SubSolverNet(NN.TwinGraph(pb), d=32, rounds=3, slot=4)
            own = net.state_dict()
            st = torch.load(path, map_location="cpu", weights_only=False)["state"]
            net.load_state_dict({k: v for k, v in st.items() if k in own and own[k].shape == v.shape}, strict=False)
        t0 = time.perf_counter()
        hist = []
        res = Cascade(pb, net=net, partition=part, config=CascadeConfig(use_neural=use_nn)).solve(
            callback=lambda r: hist.append([r.get("t_total", 0.0), r.get("objective")]) if r.get("objective") else None)
        record(name, res.x, time.perf_counter() - t0, hist)
    out = {"variables": N, "discrete": int(sp.discrete_mask().sum()), "subsystems": part.K, "budget_s": args.budget,
           "population": args.pop, "table": table, "curves": curves}
    os.makedirs(os.path.join(ROOT, "bench", "results"), exist_ok=True)
    with open(os.path.join(ROOT, "bench", "results", "solver_comparison.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
