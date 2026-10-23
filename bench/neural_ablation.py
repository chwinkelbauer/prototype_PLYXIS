"""Ablation: what do the neural sub-solvers contribute to re-planning after events?

1. expert iteration: solve N realistic event situations with the cascade (warm-started from
   the current plan, exactly as in live operation) and train the neural sub-solver on them
2. on held-out situations compare, at equal repair budgets:
     reuse      - keep the previous plan, then gradient repair
     neural     - neural sub-solver plan (ms), then gradient repair
     cascade    - full cascade warm-started from the previous plan
     cascade+NN - full cascade with the neural start added

Writes bench/results/neural_ablation.json and the trained weights to
scenarios/kestrel/models/subsolver.pt.

    python -m bench.neural_ablation --experts 60 --test 8
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from plyxis.lang import compile_model  # noqa: E402
from plyxis.optim import neural as NN  # noqa: E402
from plyxis.optim.cascade import Cascade, CascadeConfig  # noqa: E402
from plyxis.optim.gradient import alm  # noqa: E402
from plyxis.optim.problem import Problem  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=60)
    ap.add_argument("--test", type=int, default=8)
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()
    torch.manual_seed(0)
    kdir = os.path.join(ROOT, "scenarios", "kestrel")
    os.makedirs(os.path.join(kdir, "models"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "bench", "results"), exist_ok=True)
    m = compile_model([os.path.join(kdir, "field.plx")])
    prob = Problem(m.program)
    prob.evaluate(prob.space.default_u(2))
    full = CascadeConfig(use_neural=False)
    t0 = time.perf_counter()
    base = Cascade(prob, config=full).solve()
    print(f"nominal plan {base.objective:,.0f} ({time.perf_counter() - t0:.1f}s)", flush=True)
    xstar = base.x
    fast = CascadeConfig(relax_iters=120, population=10, generations=1, repair_iters=70, polish_iters=40,
                         use_neural=False, multistart=2)
    cache = os.path.join(ROOT, "bench", "results", "experts.pkl")
    t0 = time.perf_counter()
    experts = NN.generate_experts(prob, args.experts + args.test,
                                  lambda pb: _Warm(Cascade(pb, config=fast), xstar),
                                  callback=lambda r: print(r, flush=True))
    with open(cache, "wb") as fh:
        pickle.dump(experts, fh)
    print(f"experts: {time.perf_counter() - t0:.0f}s", flush=True)
    train_set, test_set = experts[:args.experts], experts[args.experts:]
    net = NN.SubSolverNet(NN.TwinGraph(prob), d=32, rounds=3, slot=4, base_x=xstar)
    t0 = time.perf_counter()
    NN.train_imitation(prob, net, [(c, x) for c, x, _ in train_set], steps=args.steps, B=8, lr=3e-3,
                       ss_weight=0.02, callback=lambda r: print(r, flush=True) if r["step"] % 50 == 0 else None)
    train_s = time.perf_counter() - t0
    NN.save(net, os.path.join(kdir, "models", "subsolver.pt"), {"experts": len(train_set), "steps": args.steps})
    mask = prob.space.discrete_mask()
    rows = []
    for k, (ctx, xe, oe) in enumerate(test_set):
        pb = prob.with_context(P=ctx.P, S0=ctx.S0, E=ctx.E, t0=ctx.t0)
        row = {"case": k}
        with torch.no_grad():
            t1 = time.perf_counter()
            xp = net(ctx.P, ctx.S0, ctx.E, ctx.t0, None, 1)
            row["neural_ms"] = (time.perf_counter() - t1) * 1e3
            for tag, x in (("reuse_raw", xstar), ("neural_raw", xp)):
                e = pb.evaluate(prob.space.round_discrete(x), relax=False)
                row[tag] = [float(e.objective[0]), float(e.max_violation()[0])]
        for tag, x in (("reuse_repair", xstar), ("neural_repair", xp)):
            t1 = time.perf_counter()
            r = alm(pb, prob.space.round_discrete(x), iters=80, lr=0.02, rho=0.5, outer=10, mask=~mask, relax=False)
            row[tag] = [r.objective, r.max_violation, time.perf_counter() - t1]
        for tag, nn_ in (("cascade", None), ("cascade_nn", net)):
            cfg = CascadeConfig(relax_iters=120, population=10, generations=1, repair_iters=70, polish_iters=40,
                                use_neural=nn_ is not None, neural_iters=1, multistart=3)
            r = Cascade(pb, net=nn_, config=cfg).solve(warm=xstar)
            row[tag] = [r.objective, r.max_violation, r.seconds]
        print(row, flush=True)
        rows.append(row)
    out = {"nominal": base.objective, "train_seconds": train_s, "experts": len(train_set), "cases": rows}
    with open(os.path.join(ROOT, "bench", "results", "neural_ablation.json"), "w") as fh:
        json.dump(out, fh, indent=1)


class _Warm:
    def __init__(self, cas, warm):
        self.cas, self.warm = cas, warm

    def solve(self):
        return self.cas.solve(warm=self.warm)


if __name__ == "__main__":
    main()
