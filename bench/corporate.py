"""Corporate plan with plants at their own step: the Aquila group hourly and the Duisburg works machine by machine,
coordinated by prices (plyxis/lang/hierarchy.py, plyxis/optim/corporate.py). Nothing declares the plants: the detector
finds them from the blueprints' durations and buffers. With --reference also the same twin as one program at the
plants' step everywhere, solved directly.

    python -m bench.corporate [--rounds 3] [--reference]

Writes bench/results/corporate.json: what the detector found and why, the size of each program, per round the master's
value and every plant schedule's internal value, reduced cost and solve time, the committed plan (the group re-planned
around each plant's one schedule), each plant's schedule unit by unit (events of every on/off decision, ranges of
every rate) and how far its flows are from the master's mix. Nominal prices, no events: decision quality and time.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

import torch

from plyxis.lang import compile_model
from plyxis.lang import units as U
from plyxis.lang.blocks import render
from plyxis.lang.hierarchy import compile_hierarchy, describe
from plyxis.optim.cascade import Cascade
from plyxis.optim.corporate import CorporateConfig, CorporatePlanner
from plyxis.optim.problem import Problem
from plyxis.runtime.planner import Planner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = [os.path.join(ROOT, "scenarios", "aquila_works", "field.plx")]


def _bound_unit(ir, e):
    """The display unit of a decision without one: the unit of the first property in its upper bound."""
    from plyxis.lang.behavior import Op, Ref
    if isinstance(e, Ref) and e.kind == "param":
        return ir.param_units.get(e.name)
    if isinstance(e, Op):
        for a in e.args:
            u = _bound_unit(ir, a)
            if u:
                return u
    return None


def schedule_summary(prog, flat, problem, x) -> dict:
    """Every unit's decisions in the plan: on/off decisions as events (and starts of a run), rates as mean / min / max
    in their display unit."""
    D, _ = problem.space.decode(x[:1], problem.P, relax=False)
    blocks = {(b.gid, b.name): b for b in problem.space.blocks}
    out = {}
    for g in prog.groups:
        if g.is_env:
            continue
        for d, info in g.ir.decisions.items():
            v = D[g.gid][d][0]
            b = blocks.get((g.gid, d))
            for li, iid in enumerate(g.inst_ids):
                path = flat.instances[iid].path
                if path.startswith("PLX_"):
                    continue
                s = v[li]
                if info.domain == "bool":
                    on = s.round()
                    rec = {"on_share": round(float(on.mean()), 3), "events": int(on.sum()),
                           "starts": int(torch.relu(on[1:] - on[:-1]).sum() + on[0])}
                    if b is not None and b.hold > 1:
                        rec["events"] = int(on.sum()) // b.hold
                else:
                    unit = info.unit or _bound_unit(g.ir, info.hi)
                    f = (lambda z: U.from_canonical(float(z), unit)) if unit else float
                    rec = {"mean": round(f(s.mean()), 3), "min": round(f(s.min()), 3), "max": round(f(s.max()), 3),
                           "unit": unit}
                out.setdefault(path, {})[d] = rec
    return out


def toy() -> dict:
    """The toy of examples/plants_toy.plx: the decomposition against the whole model solved at once (optimum 37,100)."""
    from plyxis.optim.cascade import CascadeConfig
    src = {"toy.plx": open(os.path.join(ROOT, "examples", "plants_toy.plx"), encoding="utf-8").read()}
    E = {"peak": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32)}
    cfg = CascadeConfig(relax_iters=300, population=8, generations=1, repair_iters=60, polish_iters=200, multistart=4)
    t = time.perf_counter()
    ref = Cascade(Problem(compile_model(["toy.plx"], sources=src).program, E=E), config=cfg).solve()
    t_ref = time.perf_counter() - t
    t = time.perf_counter()
    plan = CorporatePlanner(compile_hierarchy(["toy.plx"], sources=src, k_max=6),
                            CorporateConfig(rounds=6, master=cfg, plant=cfg), E=E).solve()
    return {"optimum": 37100.0, "whole": {"objective": ref.objective, "feasible": bool(ref.feasible), "seconds": t_ref},
            "decomposed": {"objective": plan.objective, "feasible": plan.feasible, "mix_objective": plan.mix_objective,
                           "seconds": time.perf_counter() - t}}


WORKS_SIGNALS = (("DE_HSM", "rolled", "t/h"), ("DE_PP", "mw", "MW"), ("DE_PP", "flare", "GJ/h"), ("DE_PIG", "rate", "t/h"),
                 ("DE_GASNET", "made", "GJ/h"), ("DE_GRID", "load", "MW"),
                 ("DE_BF1", "made", "t/h"), ("DE_BF2", "made", "t/h"), ("DE_BF3", "made", "t/h"), ("DE_RF1", "fed", "t/h"),
                 ("DE_RF2", "fed", "t/h"), ("DE_CC1", "steel", "t/h"), ("DE_CC2", "steel", "t/h"))


def signals(prog, flat, problem, x, which=WORKS_SIGNALS) -> dict:
    """Mean of named signals (local nodes or decisions) of the plan, in display units."""
    with torch.no_grad():
        e = problem.evaluate(x[:1], relax=False)
    D, _ = problem.space.decode(x[:1], problem.P, relax=False)
    out = {}
    for path, name, unit in which:
        iid = flat.by_path.get(path)
        if iid is None:
            continue
        gid, li = prog.inst_group[iid]
        if name in prog.gmap[gid].ir.decisions:
            v = D[gid][name][0, li]
        else:
            v = e.raw[prog.node_var(gid, f"local:{name}")]
            v = v.reshape(v.shape[0], -1, v.shape[-1])[0, li if v.reshape(v.shape[0], -1, v.shape[-1]).shape[1] > 1 else 0]
        out[f"{path}.{name}"] = round(U.from_canonical(float(v.float().mean()), unit), 1)
    return out


def plant_ablation(h, master_cfg) -> dict:
    """Each plant alone at the group's prices as it stands: the GA discrete stage against the per-unit dynamic
    programs, at the live budget and a larger one."""
    large = dataclasses.replace(master_cfg, discrete="auto", relax_iters=400, repair_iters=150, polish_iters=400,
                                unit_passes=4)
    arms = {"ga, live budget": dataclasses.replace(master_cfg, discrete="ga"),
            "unit dp, live budget": dataclasses.replace(master_cfg, discrete="auto"),
            "unit dp, larger budget": large}
    out = {}
    for pname, p in h.plants.items():
        for name, cfg in arms.items():
            cp = CorporatePlanner(h, CorporateConfig(master=master_cfg, plant=cfg))
            pb = cp.master_problem()
            prices = cp.prices(pb, pb.space.default_u(1), None)[pname]
            t = time.perf_counter()
            col = cp.solve_plant(pname, prices)
            ev = col["eval"]
            sch = schedule_summary(p.program, p.flat, cp.pbp[pname], col["x"])
            pick = lambda path, d, k="mean": sch.get(path, {}).get(d, {}).get(k)  # noqa: E731
            out.setdefault(pname, {})[name] = {
                "seconds": time.perf_counter() - t, "value_at_prices": col["internal"] + float((prices * col["flows"]).sum()),
                "max_violation": col["max_violation"],
                "violations": {f"{p.program.gmap[g].ir.type.short}.{k}": round(float(v.max()), 4)
                               for (g, k), v in ev.viol.items() if float(v.max()) > 1e-3},
                "signals": signals(p.program, p.flat, col["problem"], col["x"]),
                "heats": sum(pick(f"DE_BOF{i}", "tap", "events") or 0 for i in (1, 2, 3)),
                "roll_changes": pick("DE_HSM", "change", "events")}
            print(pname, name, json.dumps(out[pname][name], default=float), flush=True)
    return out


def PLANT_CFG(master_cfg):
    return dataclasses.replace(master_cfg, discrete="auto", relax_iters=800, repair_iters=300, polish_iters=800,
                               restore_iters=200, unit_passes=4)


def _rate_unit(h, c) -> str:
    rt = h.model.lib.resources[c.resource]
    return rt.rate_unit or "t/h"


def regions(args) -> dict:
    """The Aquila group hourly (scenarios/aquila, no works detail): the whole twin as one program against its regional
    networks - found by the detector, each solved as its own program at the shared markets' prices, coordinated by
    column generation, the blocks side by side in worker processes."""
    files = [os.path.join(ROOT, "scenarios", "aquila", "field.plx")]
    cfg = dataclasses.replace(Planner.LIVE, use_neural=False, relax_tol=0.01)
    out: dict = {"scenario": "aquila", "rounds_max": args.rounds, "workers": args.workers}
    t = time.perf_counter()
    h = compile_hierarchy(files, k_max=6, block=8, split="regions")
    out["compile_seconds"] = time.perf_counter() - t
    out["detected"] = describe(h.found, h.model.flat).splitlines()
    print("\n".join(out["detected"]), flush=True)
    whole = Problem(h.model.program)
    out["decisions"] = {"whole": whole.space.N, "master": Problem(h.master).space.N,
                        **{n: Problem(p.program).space.N for n, p in h.plants.items()}}
    out["crossings"] = {n: len(p.crossings) for n, p in h.plants.items()}
    print(out["decisions"], flush=True)
    t = time.perf_counter()
    res = Cascade(whole, config=cfg).solve()
    out["whole"] = {"objective": res.objective, "feasible": bool(res.feasible), "max_violation": res.max_violation,
                    "seconds": time.perf_counter() - t}
    print("whole:", out["whole"], flush=True)
    cp = CorporatePlanner(h, CorporateConfig(rounds=args.rounds, master=cfg, plant=cfg, workers=args.workers,
                                             worker_threads=args.worker_threads))

    def log(rec):
        m = rec.get("master", {})
        vals = {pn: [round(c["internal"] / 1e6, 2) for c in cols] for pn, cols in rec.get("plants", {}).items()}
        rcs = [c["reduced_cost"] for cols in rec.get("plants", {}).values() for c in cols if c["reduced_cost"] is not None]
        print(f"round {rec['round']}: {len(vals)} blocks, max rc {max(rcs) / 1e6 if rcs else 0:.3f} M"
              + (f" | master {m['objective'] / 1e6:,.2f} M feasible {m['feasible']} viol {m['max_violation']:.3f} "
                 f"({m['seconds']:.0f}s)" if m else ""), flush=True)

    t = time.perf_counter()
    plan = cp.solve(callback=log)
    out["regions"] = {"objective": plan.objective, "feasible": plan.feasible, "master_feasible": plan.master_feasible,
                      "block_violation": plan.plant_violation, "mix_objective": plan.mix_objective,
                      "seconds": time.perf_counter() - t}
    out["rounds"] = plan.rounds
    print("regions:", out["regions"], flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--reference", action="store_true", help="also solve the twin as one program at 15 minutes")
    ap.add_argument("--tag", default="")
    ap.add_argument("--plant-ablation", action="store_true", help="only the plant alone: GA vs unit DP, live vs larger budget "
                                                                   "(writes corporate_plant.json)")
    ap.add_argument("--regions", action="store_true", help="the hourly group: whole vs its regional networks "
                                                           "(writes corporate_regions.json)")
    ap.add_argument("--workers", type=int, default=0, help="solve the blocks side by side in this many processes")
    ap.add_argument("--worker-threads", type=int, default=1)
    ap.add_argument("--split", default="auto", help="plants | regions | auto | both (see compile_hierarchy)")
    ap.add_argument("--reference-from", default="", help="take the reference (the whole twin, which no planner change "
                                                         "touches) from this result file instead of solving it again")
    args = ap.parse_args()
    if args.regions:
        res = regions(args)
        name = "corporate_regions" + (f"_{args.tag}" if args.tag else "") + ".json"
        with open(os.path.join(ROOT, "bench", "results", name), "w") as fh:
            json.dump(res, fh, indent=1, default=float)
        return
    if args.plant_ablation:
        h = compile_hierarchy(FILES, k_max=6, block=8)
        res = plant_ablation(h, dataclasses.replace(Planner.LIVE, use_neural=False, relax_tol=0.01))
        with open(os.path.join(ROOT, "bench", "results", "corporate_plant.json"), "w") as fh:
            json.dump(res, fh, indent=1, default=float)
        return
    out: dict = {"scenario": "aquila_works", "rounds_max": args.rounds}
    out["toy"] = toy()
    print("toy:", json.dumps(out["toy"], default=float), flush=True)
    t = time.perf_counter()
    h = compile_hierarchy(FILES, k_max=6, block=8, split=args.split)
    out["compile_seconds"] = time.perf_counter() - t
    out["split"] = args.split
    out["detected"] = (render(h.tree, h.model.flat) if getattr(h, "tree", None) is not None else
                       describe(h.found, h.model.flat)).splitlines()
    print("\n".join(out["detected"]), flush=True)
    out["programs"] = {"master": {**h.master.summary(), "steps": int(round(h.master.horizon / h.master.dt)),
                                  "step_h": U.from_canonical(h.master.dt, "h")}}
    for name, p in h.plants.items():
        out["programs"][name] = {**p.program.summary(), "steps": int(round(p.program.horizon / p.program.dt)),
                                 "step_h": U.from_canonical(p.dt, "h"),
                                 "crossings": [f"{len(c.srcs)} x {c.src.split('_')[0]}... -> {c.dst}.{c.dst_port} ({c.short})"
                                               for c in p.crossings]}
    master_cfg = dataclasses.replace(Planner.LIVE, use_neural=False, relax_tol=0.01)
    # the works' programs (hundreds of tightly buffered quarter-hour steps) get a larger budget than the group's live
    # one: at partial load (prices below the group's) the steel side's heat schedule is feasible only with it (violation
    # 0.005 against 0.018); the reference gets the same. discrete = "auto": the per-unit DPs wherever there is a rule
    plant_cfg = PLANT_CFG(master_cfg)
    cp = CorporatePlanner(h, CorporateConfig(rounds=args.rounds, master=master_cfg, plant=plant_cfg, workers=args.workers,
                                             worker_threads=args.worker_threads))
    out["decisions"] = {"master": cp.pbm.space.N, **{n: cp.pbp[n].space.N for n in h.plants},
                        **{f"{n}_discrete": int(cp.pbp[n].space.discrete_mask().sum()) for n in h.plants}}

    def log(rec):
        m = rec.get("master", {})
        parts = []
        for pn, cols in rec.get("plants", {}).items():
            parts += [f"{pn} {c['internal'] / 1e6:,.2f} M rc "
                      f"{'-' if c['reduced_cost'] is None else round(c['reduced_cost'] / 1e6, 3)} "
                      f"({c['seconds']:.0f}s, viol {c['max_violation']:.3f})" for c in cols]
        print(f"round {rec['round']}: " + "; ".join(parts)
              + (f" | master {m['objective'] / 1e6:,.2f} M feasible {m['feasible']} viol {m['max_violation']:.3f} "
                 f"({m['seconds']:.0f}s)" if m else ""), flush=True)

    t = time.perf_counter()
    plan = cp.solve(callback=log)
    out["seconds"] = time.perf_counter() - t
    out["plan"] = {"objective": plan.objective, "feasible": plan.feasible, "master_feasible": plan.master_feasible,
                   "plant_violation": plan.plant_violation, "mix_objective": plan.mix_objective,
                   "weights_mean": {n: w.mean(-1).tolist() for n, w in plan.weights.items()}}
    out["rounds"] = plan.rounds
    out["master_signals"] = signals(h.master, h.master_flat, cp.master_problem(), plan.master_x)
    out["plants"] = {}
    for name, p in h.plants.items():
        ev = plan.plant_eval[name]
        pbp = cp.pbp[name]
        out["plants"][name] = {
            "objective": float(ev.objective[0]), "max_violation": float(ev.max_violation()[0]),
            "violations": {f"{p.program.gmap[g].ir.type.short}.{k}": round(float(v.max()), 4)
                           for (g, k), v in ev.viol.items() if float(v.max()) > 1e-3},
            "flows_per_h": {f"{c.dst}.{c.dst_port}": round(U.from_canonical(float(plan.plant_flows[name][c.k].mean()),
                                                                              _rate_unit(h, c)), 1) for c in p.crossings},
            "mix_flows_per_h": {f"{c.dst}.{c.dst_port}": round(U.from_canonical(float(plan.mix_flows[name][c.k].mean()),
                                                                                  _rate_unit(h, c)), 1) for c in p.crossings},
            "schedule": schedule_summary(p.program, p.flat, pbp, plan.plant_x[name]),
            "signals": signals(p.program, p.flat, pbp, plan.plant_x[name])}
    print(json.dumps({k: out[k] for k in ("seconds", "plan", "master_signals")}, indent=1, default=float), flush=True)
    if args.reference_from:
        out["reference"] = json.load(open(args.reference_from))["reference"]
        out["reference"]["from"] = os.path.basename(args.reference_from)
    elif args.reference:
        t = time.perf_counter()
        whole = compile_model(FILES, sim="corp96_fine").program
        assert abs(whole.dt - min(p.dt for p in h.plants.values())) < 1e-9, "corp96_fine is not at the plants' step"
        pb = Problem(whole)
        out["reference"] = {"compile_seconds": time.perf_counter() - t, "decisions": pb.space.N,
                            "discrete": int(pb.space.discrete_mask().sum()), "steps": pb.T}
        t = time.perf_counter()
        res = Cascade(pb, config=plant_cfg).solve()
        out["reference"].update({"objective": res.objective, "feasible": bool(res.feasible),
                                 "max_violation": res.max_violation, "seconds": time.perf_counter() - t,
                                 "signals": signals(whole, whole.flat, pb, res.x)})
        print("reference:", out["reference"], flush=True)
    os.makedirs(os.path.join(ROOT, "bench", "results"), exist_ok=True)
    name = "corporate" + (f"_{args.tag}" if args.tag else "") + ".json"
    with open(os.path.join(ROOT, "bench", "results", name), "w") as fh:
        json.dump(out, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
