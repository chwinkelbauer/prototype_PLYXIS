"""How fast can a plan be, and what does a smaller cascade budget cost once the exact polish finishes the plan?

    python -m bench.planner_speed [--twins kestrel corvus aquila] [--repeats 2]

For each case study's default planning problem (nominal context, start states) and each configuration: a cold plan
(no start) and a warm plan (from the cold plan shifted by the scenario's re-plan interval, as a rolling re-plan starts),
both from the scenario's own planner settings with the budgets below replaced; plan value (the planner's objective),
F and the seconds per stage. Timing-sensitive: run alone. Writes bench/results/planner_speed.json.

Configurations:
  live       the planner of 5 Nov morning: Planner.LIVE and the scenario's own settings, full budget, no exact polish
  current    the scenario's planner now (scenario.json)
  quarter    a quarter of the live budget (relaxation, repair, ALM polish) and the exact polish (15 iterations)
  sections   the current planner with the exact section search after the GA (twins whose discrete stage is the GA)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

import torch

from plyxis.lang import compile_model
from plyxis.optim.cascade import Cascade
from plyxis.optim.exact import fitness
from plyxis.optim.problem import Problem
from plyxis.runtime.planner import Planner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def problem(sc: str):
    raw = json.load(open(os.path.join(ROOT, "scenarios", sc, "scenario.json"), encoding="utf-8"))
    tw = raw["twin"]
    m = compile_model([os.path.join(ROOT, "scenarios", sc, f) for f in tw["files"]], env=tw.get("env"), sim=tw.get("sim"))
    pl = {k: v for k, v in (raw.get("planner") or {}).items() if k not in ("cold_starts", "second_opinion")}
    base = dataclasses.replace(Planner.LIVE, **pl)
    hours = float(raw.get("time", {}).get("replan_every_h", 4.0))
    return Problem(m.program), base, hours


def configs(base, sc):
    """'live' is the planner of 5 Nov morning (Planner.LIVE and the scenario's own settings, no exact polish, the
    full budget); 'current' the scenario's planner now; 'quarter' a quarter of the cascade budget with the polish."""
    full = dict(relax_iters=Planner.LIVE.relax_iters, polish_iters=Planner.LIVE.polish_iters,
                repair_iters=Planner.LIVE.repair_iters)
    raw = json.load(open(os.path.join(ROOT, "scenarios", sc, "scenario.json"), encoding="utf-8")).get("planner") or {}
    # the steel group's scenario now holds the reduced budget; the others' scenario budgets are their live ones
    budget = full if sc == "aquila" else {k: raw.get(k, v) for k, v in full.items()}
    live = dataclasses.replace(base, slp_iters=0, **budget)
    out = {"live": live, "current": base}
    if base.discrete == "ga":                      # what the section search after the GA costs (the mine ran it until 6 Nov)
        out["sections"] = dataclasses.replace(base, discrete="ga+sections")
    out["quarter"] = dataclasses.replace(base, slp_iters=15, relax_iters=max(20, int(live.relax_iters * 0.25)),
                                         polish_iters=max(10, int(live.polish_iters * 0.25)),
                                         repair_iters=max(10, int(live.repair_iters * 0.25)))
    return out


def run(pb, cfg, warm=None) -> dict:
    t = time.perf_counter()
    r = Cascade(pb, config=cfg).solve(warm=warm)
    sec = time.perf_counter() - t
    F, mv, obj = fitness(pb, r.x, False)
    return {"seconds": sec, "objective": obj, "F": F, "max_violation": mv,
            "stages": {s["stage"]: round(s["seconds"], 2) for s in r.stages}, "x": r.x}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--twins", nargs="+", default=["kestrel", "corvus", "aquila"])
    ap.add_argument("--repeats", type=int, default=1)
    a = ap.parse_args()
    path = os.path.join(ROOT, "bench", "results", "planner_speed.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    out["threads"] = torch.get_num_threads()
    for sc in a.twins:
        pb, base, hours = problem(sc)
        base = dataclasses.replace(base, use_neural=False)
        res = {"replan_h": hours, "configs": {}}
        # warm-up, not timed: the first solve on a twin pays one-off costs (torch kernels, the linearizer's
        # schedule, HiGHS) that made the first configuration look 3-4 s slower than the same planner run second
        run(pb, dataclasses.replace(base, relax_iters=5, polish_iters=2, repair_iters=2, multistart=2, slp_iters=1))
        cold_ref = None
        for name, cfg in configs(base, sc).items():
            rows = []
            for rep in range(a.repeats):
                cfg_r = dataclasses.replace(cfg, seed=rep)
                c = run(pb, cfg_r)
                if cold_ref is None:
                    cold_ref = c["x"]
                warm = pb.space.shift(cold_ref.view(1, -1), hours)
                w = run(pb, cfg_r, warm=warm)
                rows.append({"cold": {k: v for k, v in c.items() if k != "x"}, "warm": {k: v for k, v in w.items() if k != "x"}})
                print(f"{sc:8s} {name:9s} rep {rep}: cold {c['seconds']:6.1f}s USD {c['objective']:,.0f} (F {c['F']:.5f}) | "
                      f"warm {w['seconds']:6.1f}s USD {w['objective']:,.0f} (F {w['F']:.5f})", flush=True)
            res["configs"][name] = rows
        out[sc] = res
        with open(path, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
