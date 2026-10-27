"""Solver budget: how much plan quality does the live planner's budget buy?

Re-plans sampled event situations (trips, failures, injectivity loss, price shocks, as in the
neural ablation) warm-started from the nominal plan, with three cascade configurations:

    previous live  - the live configuration before the stage ablation (3 starts, short polish)
    live           - the configuration used by runtime/planner.py
    large          - a much larger budget, as a reference for the headroom left

    python -m bench.solver_budget --cases 5

Writes bench/results/solver_budget.json.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from plyxis.lang import compile_model  # noqa: E402
from plyxis.optim.cascade import Cascade, CascadeConfig  # noqa: E402
from plyxis.optim.neural import ContextSampler  # noqa: E402
from plyxis.optim.problem import Problem  # noqa: E402
from plyxis.runtime.planner import Planner  # noqa: E402

PREVIOUS = dict(relax_iters=160, population=12, generations=1, repair_iters=80, polish_iters=50, multistart=3)
LARGE = dict(relax_iters=480, population=24, generations=3, repair_iters=150, polish_iters=150, multistart=8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=5)
    args = ap.parse_args()
    torch.manual_seed(0)
    m = compile_model([os.path.join(ROOT, "scenarios", "kestrel", "field.plx")])
    prob = Problem(m.program)
    base = Cascade(prob, config=CascadeConfig(use_neural=False)).solve()
    print(f"nominal {base.objective:,.0f} feasible {base.feasible} ({base.seconds:.1f} s)", flush=True)
    live = {k: getattr(Planner.LIVE, k) for k in ("relax_iters", "population", "generations", "repair_iters",
                                                  "polish_iters", "multistart")}
    configs = {"previous live": PREVIOUS, "live": live, "large": LARGE}
    sampler = ContextSampler(prob, seed=123)
    ctxs = [sampler.sample_events(1) for _ in range(args.cases)]
    rows = []
    for i, ctx in enumerate(ctxs):
        pb = prob.with_context(P=ctx.P, S0=ctx.S0, E=ctx.E, t0=ctx.t0)
        for name, kw in configs.items():
            t = time.perf_counter()
            r = Cascade(pb, config=CascadeConfig(use_neural=False, **kw)).solve(warm=base.x)
            sec = time.perf_counter() - t
            rows.append({"case": i, "config": name, "objective": r.objective, "max_violation": r.max_violation,
                         "feasible": r.feasible, "seconds": sec})
            print(f"case {i} {name:14s} {r.objective:12,.0f}  viol {r.max_violation:.4f}  {sec:5.1f} s", flush=True)
    summary = []
    for name, kw in configs.items():
        sel = [r for r in rows if r["config"] == name]
        summary.append({"config": name, "settings": kw, "mean_objective": st.mean(r["objective"] for r in sel),
                        "feasible": sum(r["feasible"] for r in sel), "cases": len(sel),
                        "mean_seconds": st.mean(r["seconds"] for r in sel)})
    ref = summary[0]["mean_objective"]
    for s in summary:
        s["vs_previous_pct"] = (s["mean_objective"] - ref) / abs(ref) * 100
        print(f"{s['config']:14s} mean {s['mean_objective']:12,.0f} ({s['vs_previous_pct']:+.1f} %)  "
              f"feasible {s['feasible']}/{s['cases']}  {s['mean_seconds']:.1f} s")
    out = {"nominal": base.objective, "variables": prob.space.N, "cases": rows, "summary": summary}
    os.makedirs(os.path.join(ROOT, "bench", "results"), exist_ok=True)
    with open(os.path.join(ROOT, "bench", "results", "solver_budget.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
