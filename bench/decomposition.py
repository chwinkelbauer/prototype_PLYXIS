"""Deterministic sections (learned graph filter + exact section enumeration, plyxis/optim/sections.py).

For each case study and seed, the full cascade is solved twice from the same configuration: with the GA
discrete stage alone, and with the exact section search on top of the GA's best plan ("ga+sections").
Also measured: the graph filter (fit to exact coupling probes, forward time vs. probing every candidate
coupling) and the native C++ kernels against their NumPy reference on the real section solve.

    python -m bench.decomposition --seeds 3

Writes bench/results/decomposition.json.
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

from plyxis import native  # noqa: E402
from plyxis.lang import compile_model  # noqa: E402
from plyxis.optim import sections as SX  # noqa: E402
from plyxis.optim.cascade import Cascade, CascadeConfig  # noqa: E402
from plyxis.optim.problem import Problem  # noqa: E402

BASE = dict(use_neural=False, relax_iters=120, multistart=4, population=8, repair_iters=60, polish_iters=80)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--scenarios", nargs="*", default=["kestrel", "corvus"])
    ap.add_argument("--fields", nargs="*", type=int, default=[1, 2, 4, 8])
    args = ap.parse_args()
    out = {"native": native.available(), "scenarios": {}}
    for scen in args.scenarios:
        m = compile_model([os.path.join(ROOT, "scenarios", scen, "field.plx")])
        pb = Problem(m.program)
        rows = []
        for seed in range(args.seeds):
            for method in ("ga", "ga+sections"):
                t = time.perf_counter()
                r = Cascade(pb, config=CascadeConfig(**BASE, seed=seed, discrete=method)).solve()
                d = [s for s in r.stages if s["stage"] == "discrete"][0]
                rows.append({"seed": seed, "method": method, "objective": r.objective, "max_violation": r.max_violation,
                             "feasible": r.feasible, "seconds": time.perf_counter() - t, "discrete_seconds": d["seconds"],
                             "accepted": d.get("sections_accepted")})
                print(f"{scen} seed {seed} {method:12s} {r.objective:14,.0f} feas {r.feasible} {rows[-1]['seconds']:5.1f}s",
                      flush=True)
        info = SX.cached_solver(pb)[1]
        solver, _, net, g = SX._CACHE[next(k for k in SX._CACHE if k[0] == id(pb.prog))]
        # filter vs. probing every candidate coupling exactly
        lay = solver.lay
        x0 = pb.space.round_discrete(pb.space.default_u(1))
        t = time.perf_counter()
        SX.coupling_labels(pb, x0, lay, g, max_pairs=len(g.pairs))
        probe_s = time.perf_counter() - t
        t = time.perf_counter()
        with torch.no_grad():
            for _ in range(20):
                net(g)
        filt_ms = (time.perf_counter() - t) / 20 * 1000
        # native vs NumPy kernels on the real section solve
        kern = {}
        for flag in (False, True):
            native.force_numpy(flag)
            t = time.perf_counter()
            solver.solve(x0, sweeps=1)
            kern["numpy" if flag else "native"] = time.perf_counter() - t
        native.force_numpy(False)
        summ = {}
        for method in ("ga", "ga+sections"):
            sel = [r for r in rows if r["method"] == method]
            summ[method] = {"mean_objective": st.mean(r["objective"] for r in sel), "feasible": sum(r["feasible"] for r in sel),
                            "runs": len(sel), "mean_seconds": st.mean(r["seconds"] for r in sel)}
        ga = summ["ga"]["mean_objective"]
        summ["ga+sections"]["vs_ga_pct"] = (summ["ga+sections"]["mean_objective"] - ga) / abs(ga) * 100
        out["scenarios"][scen] = {"variables": pb.space.N, "discrete": int(pb.space.discrete_mask().sum()), "rows": rows,
                                  "summary": summ, "sections": info, "candidate_couplings": int(len(g.pairs)),
                                  "probe_all_couplings_s": probe_s, "filter_forward_ms": filt_ms,
                                  "section_sweep_s": kern}
        print(json.dumps(out["scenarios"][scen]["summary"], indent=1), info, f"probe {probe_s:.1f}s filter {filt_ms:.1f}ms",
              kern, flush=True)
    # scaling: enterprise twins of N fields (bench/scaling.py) - sections grow with the site, the batch does not
    out["scaling"] = []
    for n in args.fields:
        f = os.path.join(ROOT, "scenarios", "enterprise", f"enterprise_{n}.plx")
        if not os.path.exists(f):
            continue
        m = compile_model([f])
        pb = Problem(m.program)
        t = time.perf_counter()
        solver, info = SX.cached_solver(pb)
        setup = time.perf_counter() - t
        x0 = pb.space.round_discrete(pb.space.default_u(1))
        t = time.perf_counter()
        res = solver.solve(x0, sweeps=1)
        sweep = time.perf_counter() - t
        with torch.no_grad():
            t = time.perf_counter()
            pb.evaluate(pb.space.default_u(1), relax=False)
            sim = time.perf_counter() - t
        row = {"fields": n, "variables": pb.space.N, "discrete_units": info["discrete_units"], "sections": info["sections"],
               "largest_section": info["largest_section"], "colours": info["colours"], "setup_s": setup,
               "sweep_s": sweep, "simulate_s": sim, "evals": res.evals}
        out["scaling"].append(row)
        print("scaling", row, flush=True)
    os.makedirs(os.path.join(ROOT, "bench", "results"), exist_ok=True)
    with open(os.path.join(ROOT, "bench", "results", "decomposition.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
