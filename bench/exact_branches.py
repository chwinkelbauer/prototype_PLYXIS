"""Structure recognition and exact LP branches on the case-study twins.

    python -m bench.exact_branches                  # both twins, ~5 min

For each twin: the structure report (share of decisions per proven class), then from a plan of the live
cascade (scenario planner budget, no neural start), two ways to spend the same extra time:
    exact branches   the branches proven affine are re-solved as one LP (HiGHS), verified on the twin
    more polish      the cascade's polish (augmented Lagrangian) continued for at least the same wall time
Writes bench/results/exact_branches.json.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from plyxis.lang import compile_model  # noqa: E402
from plyxis.optim import branchlp as BL  # noqa: E402
from plyxis.optim import structure as ST  # noqa: E402
from plyxis.optim.cascade import Cascade  # noqa: E402
from plyxis.optim.gradient import alm  # noqa: E402
from plyxis.optim.problem import Problem  # noqa: E402
from plyxis.runtime.planner import Planner  # noqa: E402


def score(pb, x):
    with torch.no_grad():
        e = pb.evaluate(x, relax=False)
    return {"F": float(e.objective[0] / pb.J_scale - BL.MU / pb.T * e.sum_violation()[0]),
            "objective": float(e.objective[0]), "max_violation": float(e.max_violation()[0])}


def run(scen: str, seed: int = 0) -> dict:
    sd = os.path.join(ROOT, "scenarios", scen)
    sc = json.load(open(os.path.join(sd, "scenario.json")))
    m = compile_model([os.path.join(sd, f) for f in sc["twin"]["files"]], env=sc["twin"]["env"])
    pb = Problem(m.program)
    pb.calibrate_scales()
    rep = ST.report(m.program)
    cfg = dataclasses.replace(Planner.LIVE, **sc.get("planner", {}), use_neural=False, seed=seed)
    t = time.perf_counter()
    res = Cascade(pb, config=cfg).solve()
    out = {"scenario": scen, "structure": rep, "cascade_seconds": time.perf_counter() - t}
    x = res.x[:1]
    out["cascade"] = score(pb, x)
    var = BL.class_variables(pb, ("lp",))
    br = BL.solve_branches(pb, x, var, affine=True)
    out["exact_branches"] = {**score(pb, br.x), "seconds": br.seconds, "jacobian_seconds": br.jac_seconds,
                             "lp_seconds": br.lp_seconds, "variables": br.variables, "rows": br.rows,
                             "iterations": br.iterations}
    mask = pb.space.discrete_mask()
    lam = {k: v[:1] for k, v in res.multipliers.items()} if res.multipliers else None   # continue the polish as it was
    iters, spent = 20, 0.0
    while True:                                   # more polish for at least the same wall time
        t = time.perf_counter()
        pol = alm(pb, x, iters=iters, lr=0.01, rho=1.0, outer=10, mask=~mask, relax=False, stage="polish",
                  lam0=lam)
        spent = time.perf_counter() - t
        if spent >= br.seconds or iters >= 4000:
            break
        iters *= 2
    out["more_polish"] = {**score(pb, pol.x[:1]), "seconds": spent, "iterations": iters}
    both = BL.solve_branches(pb, pol.x[:1], var, affine=True)
    out["polish_then_branches"] = {**score(pb, both.x), "seconds": spent + both.seconds}
    # sequential LP over the whole continuous layer the structure analysis classifies as (piecewise) affine
    var_all = BL.class_variables(pb, ("lp", "slp", "slp_quality"))
    sl = BL.solve_branches(pb, x, var_all, affine=False, iters=8)
    out["slp_layer"] = {**score(pb, sl.x), "seconds": sl.seconds, "variables": sl.variables, "rows": sl.rows,
                        "iterations": sl.iterations, "lp_seconds": sl.lp_seconds, "jacobian_seconds": sl.jac_seconds,
                        "history": sl.history}
    iters, spent = 20, 0.0
    while True:                                   # the same time spent on the polish instead
        t = time.perf_counter()
        pol2 = alm(pb, x, iters=iters, lr=0.01, rho=1.0, outer=10, mask=~mask, relax=False, stage="polish",
                   lam0=lam)
        spent = time.perf_counter() - t
        if spent >= sl.seconds or iters >= 8000:
            break
        iters *= 2
    out["slp_layer_polish_equal_time"] = {**score(pb, pol2.x[:1]), "seconds": spent, "iterations": iters}
    # speed: half the cascade's iterations, then the exact branches
    half = dataclasses.replace(cfg, relax_iters=cfg.relax_iters // 2, polish_iters=cfg.polish_iters // 2,
                               repair_iters=cfg.repair_iters // 2)
    t = time.perf_counter()
    rh = Cascade(pb, config=half).solve()
    bh = BL.solve_branches(pb, rh.x[:1], var, affine=True)
    out["half_cascade_plus_branches"] = {**score(pb, bh.x), "seconds": time.perf_counter() - t,
                                         "cascade_only": score(pb, rh.x[:1])}
    # the convex class as well, and how much the tangent planes at the end point still promise
    var_cvx = BL.class_variables(pb, ("lp", "slp", "slp_quality", "convex"))
    cv = BL.solve_branches(pb, x, var_cvx, affine=False, iters=8)
    gap = BL.tangent_gap(pb, cv.x, var_cvx, BL.jacobian(pb, cv.x, var_cvx, step=0.005))
    out["slp_with_convex"] = {**score(pb, cv.x), "seconds": cv.seconds, "variables": cv.variables, "rows": cv.rows,
                              "iterations": cv.iterations, "lp_seconds": cv.lp_seconds, "jacobian_seconds": cv.jac_seconds,
                              "tangent_gap_F": gap, "tangent_gap_usd": None if gap is None else gap * pb.J_scale,
                              "history": cv.history}
    return out


def main() -> None:
    names = sys.argv[1:] or ["corvus", "kestrel"]
    path = os.path.join(ROOT, "bench", "results", "exact_branches.json")
    # one twin at a time keeps the plans of the others; their classes follow the analyser as it is now
    res = {"runs": [r for r in json.load(open(path))["runs"] if r["scenario"] not in names]} if os.path.exists(path) else {"runs": []}
    for r in res["runs"]:
        sc = json.load(open(os.path.join(ROOT, "scenarios", r["scenario"], "scenario.json")))
        m = compile_model([os.path.join(ROOT, "scenarios", r["scenario"], f) for f in sc["twin"]["files"]], env=sc["twin"]["env"])
        r["structure"] = ST.report(m.program)
    for scen in names:
        r = run(scen)
        res["runs"].append(r)
        c, e, p = r["cascade"], r["exact_branches"], r["more_polish"]
        print(f"{scen}: LP share {100 * r['structure']['share_by_class'].get('lp', 0):.1f} % | cascade obj {c['objective']:,.0f} "
              f"| exact branches {e['objective']:,.0f} ({e['objective'] - c['objective']:+,.0f}, {e['seconds']:.1f} s, "
              f"{e['variables']} vars, {e['rows']} rows) | more polish {p['objective']:,.0f} "
              f"({p['objective'] - c['objective']:+,.0f}, {p['seconds']:.1f} s) | half cascade + branches "
              f"{r['half_cascade_plus_branches']['objective']:,.0f} in {r['half_cascade_plus_branches']['seconds']:.1f} s "
              f"vs full {r['cascade_seconds']:.1f} s | F {c['F']:.5f} / {e['F']:.5f} / {p['F']:.5f}", flush=True)
        sl, sp = r["slp_layer"], r["slp_layer_polish_equal_time"]
        print(f"   SLP layer ({sl['variables']} vars, {sl['rows']} rows, {sl['iterations']} it, {sl['seconds']:.1f} s): "
              f"obj {sl['objective']:,.0f} F {sl['F']:.5f} viol {sl['max_violation']:.4f} | polish same time: "
              f"obj {sp['objective']:,.0f} F {sp['F']:.5f} viol {sp['max_violation']:.4f}", flush=True)
        cv = r["slp_with_convex"]
        print(f"   SLP with the convex class ({cv['variables']} vars, {cv['rows']} rows, {cv['iterations']} it, {cv['seconds']:.1f} s): "
              f"obj {cv['objective']:,.0f} F {cv['F']:.5f} viol {cv['max_violation']:.4f}, tangent gap USD {cv['tangent_gap_usd']:,.0f}", flush=True)
    os.makedirs(os.path.join(ROOT, "bench", "results"), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
