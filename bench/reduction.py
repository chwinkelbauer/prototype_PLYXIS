"""The reduction engine (plyxis/optim/reduce.py) against the whole-model MILP as a twin grows.

    python -m bench.reduction [--sites 1 2 4 8 16] [--whole-limit 300]

The twin: N kiln sites (tests/models/kilns.plx: three on/off kilns with fixed costs, a day tariff and a three-hour
minimum run, a store, a lane), each with its own costs, all selling into one market that takes 55 t/h per site -
the shared limit binds, so the sites must be coordinated. Both methods solve the same algebraic model of the twin
(algebra.py, exact here: affine, minimum runs and start counts encoded with indicators) at the default plan:

    whole    HiGHS MILP on the whole model (time limit --whole-limit; its own gap when it stops)
    engine   filters -> blocks (kilns found as units) -> Dantzig-Wolfe with exact block MILPs (HiGHS), master LP,
             Lagrangian bound -> commit (agreed integers fixed, the rest one MILP) -> block sweeps

Every plan is then simulated on the torch twin (value and violation as the planner sees them). Also reports what
the filters find in the four case-study twins. Writes bench/results/reduction.json. Timing-sensitive: run alone.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

import numpy as np

from plyxis.lang import compile_model
from plyxis.optim.algebra import linearize
from plyxis.optim.exact import fitness, solve
from plyxis.optim.problem import Problem
from plyxis.optim.reduce import Engine, contract

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KILNS = os.path.join(ROOT, "tests", "models", "kilns.plx")


def source(n: int) -> str:
    """N sites, costs varied by site (deterministic), one market for all of them."""
    rng = np.random.default_rng(7)
    lines = ["use kilns;", "define kiln_group as env {", "  contains:",
             f"    init MKT as market {{ demand = {55 * n} [t/h]; }};"]
    for s in range(n):
        f = rng.uniform(0.8, 1.25, 3)
        u = rng.uniform(0.85, 1.2, 3)
        caps = (30, 25, 20)
        fixed = (900, 600, 700)
        unit = (35, 35, 30)
        for k in range(3):
            lines.append(f"    init S{s}_K{k + 1} as kiln {{ cap = {caps[k]} [t/h]; fixed = {fixed[k] * f[k]:.0f} [USD/h]; "
                         f"unit_cost = {unit[k] * u[k]:.1f} [USD/t]; }};")
            lines.append(f"    connect(S{s}_K{k + 1}.o, S{s}_ST.feed);")
        lines.insert(len(lines) - 6, f"    init S{s}_ST as store;")
        lines.append(f"    init S{s}_L as lane;")
        lines.append(f"    connect(S{s}_L.draw, S{s}_ST.draws);")
        lines.append(f"    connect(S{s}_L.deliver, MKT.rec);")
    lines += ["};", "define kilns_scale as sim {", "  env: kiln_group; horizon = 24 [h]; step = 1 [h];",
              "  objective: maximize sum(revenue) - sum(cost);", "};"]
    return "\n".join(lines) + "\n"


def case(n: int, whole_limit: float, int_cap: int) -> dict:
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "group.plx")
        with open(p, "w") as fh:
            fh.write(source(n))
        m = compile_model([KILNS, p], sim="kilns_scale")
    pb = Problem(m.program)
    t = time.perf_counter()
    lm = linearize(pb)
    t_lin = time.perf_counter() - t
    out = {"sites": n, "kilns": 3 * n, "model": lm.describe(), "linearize_s": t_lin}
    t = time.perf_counter()
    w = solve(lm, integer=True, want_duals=False, time_limit=whole_limit)
    out["whole"] = {"seconds": time.perf_counter() - t, "status": w.status, "F": w.F, "gap": w.mip_gap}
    if w.x is not None:
        F, viol, obj = fitness(pb, w.x[:lm.N])
        out["whole"].update({"twin_F": F, "twin_violation": viol, "twin_objective": obj})
    t = time.perf_counter()
    eng = Engine(lm, int_cap=int_cap, flat=m.program.flat)
    rep = eng.report()
    r = eng.solve(rounds=60)
    sec = time.perf_counter() - t
    F, viol, obj = fitness(pb, r.x[:lm.N])
    out["engine"] = {"seconds": sec, "F": r.F, "bound": r.bound, "gap": r.gap, "rounds": r.rounds,
                     "blocks": r.blocks, "columns": r.columns, "pricing_s": r.pricing_seconds,
                     "master_s": r.master_seconds, "commit_s": r.commit_seconds, "commit": r.commit,
                     "patterns": rep["patterns"], "largest_block_ints": rep["largest_block_ints"],
                     "master_columns": rep["master_columns"], "coupling_rows": rep["coupling_rows"],
                     "twin_F": F, "twin_violation": viol, "twin_objective": obj}
    return out


def structure() -> dict:
    out = {}
    for key, (folder, sim) in {"kestrel": ("kestrel", None), "corvus": ("corvus", None), "aquila": ("aquila", None),
                               "aquila_works": ("aquila_works", "corp96")}.items():
        m = compile_model([os.path.join(ROOT, "scenarios", folder, "field.plx")], sim=sim)
        pb = Problem(m.program)
        t = time.perf_counter()
        lm = linearize(pb)
        t_lin = time.perf_counter() - t
        t = time.perf_counter()
        st = contract(lm, int_cap=200, flat=m.program.flat)
        t_c = time.perf_counter() - t
        out[key] = {"linearize_s": t_lin, "contract_s": t_c, "model": lm.describe(), "levels": st["levels"],
                    "blocks": [{"name": b.name, "units": len(b.atoms), "ints": b.ints, "cols": int(b.cols.size),
                                "pattern": b.pattern} for b in st["blocks"]],
                    "master_columns": int(st["master_cols"].size), "coupling_rows": int(st["master_rows"].size)}
        print(f"== {key}: linearise {t_lin:.2f}s, contract {t_c:.2f}s, {len(st['blocks'])} blocks, master "
              f"{st['master_cols'].size} columns", flush=True)
    return out


def slp_vs_cascade(folder: str, sim=None, iters: int = 25) -> dict:
    """The sequential LP on the algebraic model against the production cascade (live budget, no neural start), both
    from the default plan, judged on the twin (F = objective / J_scale - MU / T * violation)."""
    from plyxis.optim.cascade import Cascade, CascadeConfig
    from plyxis.optim.exact import slp
    m = compile_model([os.path.join(ROOT, "scenarios", folder, "field.plx")], sim=sim)
    pb = Problem(m.program)
    x0 = pb.space.default_u(1)
    F0, v0, o0 = fitness(pb, x0, True)
    r = slp(pb, x0, iters=iters)
    F1, v1, o1 = fitness(pb, r.x, True)
    cfg = CascadeConfig(relax_iters=200, population=12, generations=1, repair_iters=80, polish_iters=120,
                        multistart=8, neural_iters=1, use_neural=False)
    t = time.perf_counter()
    c = Cascade(pb, config=cfg).solve()
    tc = time.perf_counter() - t
    F2, v2, o2 = fitness(pb, c.x, False)
    out = {"twin": folder, "start": {"F": F0, "violation": v0, "objective": o0},
           "slp": {"seconds": r.seconds, "lp_seconds": r.lp_seconds, "linearize_seconds": r.lin_seconds,
                   "iterations": r.iterations, "F": F1, "violation": v1, "objective": o1},
           "cascade": {"seconds": tc, "F": F2, "violation": v2, "objective": o2}}
    print(json.dumps(out), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--whole-limit", type=float, default=300.0)
    ap.add_argument("--int-cap", type=int, default=90)
    ap.add_argument("--parts", nargs="+", default=["scaling", "structure"], choices=["scaling", "structure", "slp"])
    ap.add_argument("--no-structure", action="store_true")
    a = ap.parse_args()
    path = os.path.join(ROOT, "bench", "results", "reduction.json")
    res = {}
    if os.path.exists(path):
        with open(path) as fh:
            res = json.load(fh)
    parts = [p for p in a.parts if not (p == "structure" and a.no_structure)]

    def save():
        with open(path, "w") as fh:
            json.dump(res, fh, indent=1, default=float)

    if "scaling" in parts:
        res.update({"scaling": [], "whole_limit_s": a.whole_limit, "int_cap": a.int_cap})
        for n in a.sites:
            c = case(n, a.whole_limit, a.int_cap)
            res["scaling"].append(c)
            w, e = c["whole"], c["engine"]
            print(f"sites {n:3d}: whole {w['status']:10s} {w['seconds']:7.1f}s F {w['F']:.5f} gap {w['gap']} | engine "
                  f"{e['seconds']:6.1f}s F {e['F']:.5f} bound {e['bound']:.5f} gap {e['gap']:.2%} rounds {e['rounds']} "
                  f"blocks {e['blocks']} | twin viol whole {w.get('twin_violation')} engine {e['twin_violation']:.2g}",
                  flush=True)
            save()
    if "structure" in parts:
        res["structure"] = structure()
        save()
    if "slp" in parts:
        res["slp_vs_cascade"] = [slp_vs_cascade("aquila")]
        save()


if __name__ == "__main__":
    main()
