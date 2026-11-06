"""How good is one steel-group plan? The cascade's plan against a better one, and what restarts and the exact polish add.

    python -m bench.plan_quality

On the steel group's default planning problem (nominal prices, start states, 96 h):

  stored plans   bench/results/arms/aquila_plan_cascade.pt - the cascade's plan (live budget, no neural start);
                 bench/results/arms/aquila_plan_better.pt  - a plan an earlier code version found by accident (a
                 short relaxation, before the ALM's best-iterate fix). Both are evaluated here, term by term: what the
                 better plan does differently (shipments, spot sales, blast-furnace output by market and site)
  cold plans     the current cascade at 8 restarts (seeds 0, 1) and 16 (seed 0)
  step           one LP step (trust region 0.02) from the cascade's plan: each term's predicted change against the
                 twin's, and the limits it breaks - where the first-order model is wrong
  polish_starts  the polish (up to 40 iterations) from four stored plans: the cascade's, the better one, a cold plan of the
                 current code (seed 1, `aquila_plan_cold_seed1.pt`) and one of the earlier code from a top-of-range start
                 (`aquila_plan_cold_upper_start.pt`) - where it ends depends on where it starts
  polish         the sequential LP on the twin's algebraic model (exact.slp, 15 iterations) from the cascade's plan,
                 from a trust region of 0.005 (floor 1e-3) and of 0.02 (floor 2e-4)

Writes bench/results/plan_quality_aquila.json (masses in kg: the compiled twin works in base units). Plan values are in the planner's own objective (after carbon,
including planning-only terms); a plant runs only the first hours of each plan, so they bound what an A/B can show.
"""
from __future__ import annotations

import json
import os
import time

import torch

from plyxis.lang import compile_model
from plyxis.optim.cascade import Cascade, CascadeConfig
from plyxis.optim.exact import fitness, slp
from plyxis.optim.problem import Problem

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = os.path.join(ROOT, "bench", "results", "arms")


def describe(pb, x) -> dict:
    prog = pb.prog
    with torch.no_grad():
        e = pb.evaluate(x.view(1, -1), relax=False)
    F, mv, obj = fitness(pb, x)
    named = {k: float(torch.as_tensor(v).sum()) for k, v in e.totals["named"].items()}

    def per(tname, key):
        out = {}
        for g in prog.groups:
            if g.is_env or g.ir.type.short != tname or key not in g.ir.nodes:
                continue
            v = prog.get(e.raw, g.gid, key)
            v = v[0] if v.dim() == 3 else v
            for li, iid in enumerate(g.inst_ids):
                out[prog.flat.instances[iid].path] = float(v[li].sum()) * pb.dt
        return out

    return {"objective": obj, "F": F, "max_violation": mv,
            "shipped_kg": sum(per("steel_market", "local:delivered").values()),
            "spot_kg": sum(per("steel_market", "local:spot_sold").values()),
            "hot_metal_kg": sum(per("blast_furnace", "local:made").values()),
            "delivered_by_market_kg": per("steel_market", "local:delivered"),
            "spot_by_market_kg": per("steel_market", "local:spot_sold"),
            "hot_metal_by_furnace_kg": per("blast_furnace", "local:made"),
            "terms": {k: v for k, v in named.items()}}


def step_diagnostic(pb, x, radius: float = 0.02) -> dict:
    """One LP step of the given trust region from plan x: the model's predicted change of every objective term
    against the twin's, and the limits the twin finds broken."""
    import numpy as np
    from collections import defaultdict
    from plyxis.optim.algebra import linearize
    from plyxis.optim.exact import solve
    prog = pb.prog
    lm = linearize(pb, x.view(1, -1))
    lz = pb._linearizer
    s = solve(lm, radius=radius, want_duals=False)
    dx = np.zeros(lz.nvar)
    dx[:lm.nvar] = s.x - lm.x_ref
    pred = defaultdict(float)
    for tr in prog.terms:
        gg = prog.gmap[tr.gid]
        lin = lz._expr(gg.ir.finals[tr.key], gg, True) if tr.final else lz.L[f"{tr.gid}|{tr.key}"]
        if lin.M is not None:
            pred[tr.name] += float((lin.M @ dx).sum()) * (1.0 if tr.final else pb.dt)
    u1 = torch.as_tensor(np.clip(s.x[:lm.N], 0, 1), dtype=torch.float32).view(1, -1)
    with torch.no_grad():
        e0, e1 = pb.evaluate(x.view(1, -1), relax=False), pb.evaluate(u1, relax=False)
    act = {k: float(torch.as_tensor(e1.totals["named"][k]).sum() - torch.as_tensor(e0.totals["named"][k]).sum())
           for k in e0.totals["named"]}
    kinds = {tr.name: tr.kind for tr in prog.terms}
    terms = sorted(({"term": k, "kind": kinds.get(k), "model": pred.get(k, 0.0), "twin": act.get(k, 0.0)}
                    for k in set(pred) | set(act) if kinds.get(k) in ("revenue", "cost")),
                   key=lambda r: -abs(r["model"] - r["twin"]))
    broken = {}
    for (gid, ck), v in e1.viol.items():
        mv = float(v.max())
        if mv > 1e-3:
            key = prog.gmap[gid].ir.type.short + "." + ck
            broken[key] = max(broken.get(key, 0.0), mv)
    return {"radius": radius, "objective_before": float(e0.objective[0]), "objective_after": float(e1.objective[0]),
            "model_F": s.F, "model_F_ref": s.F_ref, "terms": terms[:12], "broken_limits": broken}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["stored", "cold", "polish", "step"])
    parts = ap.parse_args().parts
    m = compile_model([os.path.join(ROOT, "scenarios", "aquila", "field.plx")])
    pb = Problem(m.program)
    path = os.path.join(ROOT, "bench", "results", "plan_quality_aquila.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    out.setdefault("stored", {})
    for name in ("cascade", "better") if "stored" in parts else ():
        p = os.path.join(ARMS, f"aquila_plan_{name}.pt")
        if os.path.exists(p):
            out["stored"][name] = describe(pb, torch.load(p).view(-1))
            d = out["stored"][name]
            print(f"stored {name}: USD {d['objective']:,.0f} F {d['F']:.5f} shipped {d['shipped_kg'] / 1e3:,.0f} t "
                  f"spot {d['spot_kg'] / 1e3:,.0f} t hot metal {d['hot_metal_kg'] / 1e3:,.0f} t", flush=True)
    live = dict(relax_iters=200, population=12, generations=1, repair_iters=80, polish_iters=120, neural_iters=1,
                use_neural=False, relax_tol=0.01)
    if "cold" in parts:
        out["cold"] = []
    for ms, seed in ((8, 0), (8, 1), (16, 0)) if "cold" in parts else ():
        t = time.perf_counter()
        r = Cascade(pb, config=CascadeConfig(multistart=ms, seed=seed, **live)).solve()
        d = describe(pb, r.x.view(-1))
        out["cold"].append({"multistart": ms, "seed": seed, "seconds": time.perf_counter() - t,
                            **{k: d[k] for k in ("objective", "F", "max_violation", "shipped_kg", "spot_kg", "hot_metal_kg")}})
        print(json.dumps(out["cold"][-1]), flush=True)
    p = os.path.join(ARMS, "aquila_plan_cascade.pt")
    if "step" in parts and os.path.exists(p):
        out["step"] = step_diagnostic(pb, torch.load(p).view(-1))
        print(json.dumps({k: v for k, v in out["step"].items() if k != "terms"}), flush=True)
    if "polish_starts" in parts:
        # the polish from several plans: where it ends depends on where it starts (keep the best of several?)
        out["polish_starts"] = []
        for name in ("cascade", "better", "cold_seed1", "cold_upper_start"):
            q = os.path.join(ARMS, f"aquila_plan_{name}.pt")
            if not os.path.exists(q):
                continue
            x0 = torch.load(q).view(-1)
            _, _, o0 = fitness(pb, x0)
            r = slp(pb, x0, iters=40, radius=0.02, min_radius=2e-4)
            d = describe(pb, r.x.view(-1))
            out["polish_starts"].append({"start": name, "start_objective": o0, "iterations": r.iterations,
                                         "seconds": r.seconds,
                                         **{k: d[k] for k in ("objective", "F", "max_violation", "shipped_kg", "spot_kg")}})
            print(json.dumps(out["polish_starts"][-1]), flush=True)
    if "polish" in parts and os.path.exists(p):
        x = torch.load(p).view(-1)
        out["polish"] = []
        for radius, floor in ((0.005, 1e-3), (0.02, 2e-4)):
            r = slp(pb, x, iters=15, radius=radius, min_radius=floor)
            d = describe(pb, r.x.view(-1))
            out["polish"].append({"radius": radius, "min_radius": floor, "iterations": r.iterations, "seconds": r.seconds,
                                  **{k: d[k] for k in ("objective", "F", "max_violation", "shipped_kg", "spot_kg", "hot_metal_kg")}})
            print(json.dumps(out["polish"][-1]), flush=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
