"""Run the full Kestrel showcase headless and record the outcome (A/B: PLYXIS vs manual operations).

    python -m bench.showcase_run --hours 72

Writes bench/results/showcase_run.json (KPIs per hour, timeline, plans, calibration accuracy).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from plyxis.lang import compile_model  # noqa: E402
from plyxis.lang import units as U  # noqa: E402
from plyxis.runtime.engine import Engine  # noqa: E402


# properties each case study's twin calibrates online, by asset type (the oil field's and the mine's keep their bare names)
CALIBRATED = {"esp_well": ("q_max", "wc", "gor"), "stope": ("grade", "rate_max"),
              "blast_furnace": ("rate_max", "coke_rate"), "eaf": ("kwh_t", "cap"), "scrap_yard": ("buy_cu",),
              "dri_plant": ("gj_t",), "hot_strip_mill": ("yield",)}


def calibration_error(eng, base) -> dict:
    """Mean absolute relative error of the calibrated properties vs the plant's actual values:
    engineering model (field.plx as delivered) vs the live twin after the run."""
    live, prog, pp = eng.twin.live, eng.twin.model.program, eng.plants["plyxis"]
    out = {}
    kinds = [k for k in CALIBRATED if any(g.ir.type.short == k for g in prog.groups)]
    for kind in kinds:
        if kind == "stope" and "esp_well" in kinds:
            continue
        for param in CALIBRATED[kind]:
            e0, e1 = [], []
            for g in prog.groups:
                if g.is_env or g.ir.type.short != kind:
                    continue
                for li, iid in enumerate(g.inst_ids):
                    path = prog.flat.instances[iid].path
                    bi = base.flat.by_path.get(path)
                    if path not in pp.path_index or bi is None:
                        continue
                    gp, lp = pp.path_index[path]
                    bg, bl = base.program.inst_group[bi]
                    truth = float(pp.P[gp][param].reshape(-1)[lp])
                    e0.append(abs(float(base.program.P0[bg][param].reshape(-1)[bl]) - truth) / abs(truth))
                    e1.append(abs(float(live.P[g.gid][param].reshape(-1)[li]) - truth) / abs(truth))
            if e0:
                key = param if kind in ("esp_well", "stope") else f"{kind}.{param}"
                out[key] = {"engineering": round(statistics.mean(e0), 4), "twin": round(statistics.mean(e1), 4)}
    return out


def protection(eng) -> dict:
    """What each world's plant protection had to cut (steps per asset and limit), and any material a stock's lower
    bound created (must be zero for a valid comparison)."""
    out = {}
    for w, p in eng.plants.items():
        cuts: dict = {}
        for _t, th in p.throttle_log:
            for c in th:
                k = f"{c['asset']}|{c['constraint']}"
                cuts[k] = cuts.get(k, 0) + 1
        out[w] = {"cuts": dict(sorted(cuts.items(), key=lambda kv: -kv[1])[:30]), "underflow": dict(p.underflow)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--scenario", default="kestrel")
    ap.add_argument("--no-neural", action="store_true")
    ap.add_argument("--skip-events", default="", help="comma list of assets / price names whose events are dropped "
                                                        "(ablation of scenario events), e.g. GASPLANT,power_price")
    ap.add_argument("--tag", default="", help="suffix of the result file, e.g. 'core' -> showcase_run_core.json")
    ap.add_argument("--no-branches", action="store_true", help="ablation: planner without the exact branch stage")
    ap.add_argument("--exact", default=None, help="exact-branch classes for this run, e.g. lp,slp,slp_quality (default: the scenario's)")
    ap.add_argument("--exact-margin", type=float, default=None, help="distance the exact stage keeps from state limits (share of their scale)")
    ap.add_argument("--discrete", default=None, help="discrete stage for this run: ga, sections or ga+sections (default: the scenario's)")
    ap.add_argument("--relax-tol", type=float, default=None, help="violation up to which the relaxation counts an iterate as feasible")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="any other CascadeConfig field for this run, e.g. population=4")
    ap.add_argument("--seed", type=int, default=0, help="scenario seed (prices, sensor and process noise; both worlds)")
    args = ap.parse_args()
    t0 = time.perf_counter()
    sdir = os.path.join(ROOT, "scenarios", args.scenario)
    eng = Engine(sdir, seed=args.seed, sync_plans=True, use_neural=not args.no_neural)
    args.hours = args.hours or eng.sc.end_hour
    import dataclasses
    if args.no_branches:
        eng.planner.cfg = dataclasses.replace(eng.planner.cfg, exact_branches=())
    if args.exact is not None:
        eng.planner.cfg = dataclasses.replace(eng.planner.cfg, exact_branches=tuple(c for c in args.exact.split(",") if c))
    if args.relax_tol is not None:
        eng.planner.cfg = dataclasses.replace(eng.planner.cfg, relax_tol=args.relax_tol)
    for item in args.set:
        name, _, value = item.partition("=")
        cur = getattr(eng.planner.cfg, name)
        conv = float if cur is None else type(cur)          # an option that is off by default (None) takes a number
        eng.planner.cfg = dataclasses.replace(eng.planner.cfg, **{name: conv(value)})
    if args.discrete is not None:
        eng.planner.cfg = dataclasses.replace(eng.planner.cfg, discrete=args.discrete)
    if args.exact_margin is not None:
        eng.planner.cfg = dataclasses.replace(eng.planner.cfg, exact_margin=args.exact_margin)
    skip = {s for s in args.skip_events.split(",") if s}
    if skip:
        eng.sc.events = [e for e in eng.sc.events if e.get("asset") not in skip and e.get("name") not in skip]
        print("events:", [(e["t"], e["kind"], e.get("asset") or e.get("name")) for e in eng.sc.events], flush=True)
    hourly = []

    def progress(e):
        k = e.kpis()
        p, l = k["plyxis"], k["legacy"]
        rec = {"t": e.t, "clock": e.clock_str(),
               "plyxis": {**{k2: round(v, 1) for k2, v in p["rate"].items()}, "cum_profit": round(p["cum"]["profit"])},
               "legacy": {**{k2: round(v, 1) for k2, v in l["rate"].items()}, "cum_profit": round(l["cum"]["profit"])},
               "delta_profit": round(k["delta"]["profit"]),
               # USD/h of every model term in the last step (+ = income), to see what each world did and when
               "terms_h": {w: {n: round(v, 1) for n, v in e.plants[w].kpi.term_rate.items()} for w in ("plyxis", "legacy")}}
        hourly.append(rec)
        main = [m for m in p["rate"] if m.startswith("m_")][:3] or ["oil_bpd", "flare_mscfd"]
        print(f"{rec['clock']}  " + " | ".join(f"{m} P {p['rate'].get(m, 0):8,.0f} L {l['rate'].get(m, 0):8,.0f}" for m in main)
              + f" | delta profit {k['delta']['profit']:11,.0f}  [{time.perf_counter() - t0:5.0f}s]", flush=True)
    eng.run_headless(args.hours, progress=progress)
    k = eng.kpis()
    out = {
        "scenario": args.scenario, "hours": args.hours, "wall_seconds": time.perf_counter() - t0, "neural": not args.no_neural,
        "exact_branches": list(eng.planner.cfg.exact_branches), "exact_margin": eng.planner.cfg.exact_margin, "seed": args.seed,
        "skipped_events": sorted(skip),
        "kpis": {"plyxis": k["plyxis"]["cum"], "legacy": k["legacy"]["cum"], "delta": k["delta"],
                 "value_per_day": k["value_per_day"]},
        "terms": {w: dict(eng.plants[w].kpi.terms) for w in ("plyxis", "legacy")},
        "inventory_terms": {w: dict(eng.plants[w].kpi.inv_terms) for w in ("plyxis", "legacy")},
        "bridge": k["bridge"],
        "calibration_error": calibration_error(eng, compile_model(eng.sc.twin_files, env=eng.sc.twin_env)),
        "protection": protection(eng),
        # offset-free corrections the twin estimated (e.g. the concentrator's analysed / computed head grade)
        "head_factor": {eng.twin.model.program.flat.instances[g.inst_ids[0]].path:
                        float(eng.twin.live.P[g.gid]["head_factor"].reshape(-1)[0])
                        for g in eng.twin.model.program.groups if "head_factor" in eng.twin.live.P.get(g.gid, {})},
        "margins": eng.margin_log,
        "plans": list(eng.plan_events),
        "runs": [{kk: v for kk, v in r.items() if kk != "history"} for r in eng.planner.runs],
        "timeline": eng.timeline, "hourly": hourly,
        "alerts": list(eng.alerts)[-150:],
        "commits": eng.twin.repo.log(50),
        "learn": eng.learn_log,
    }
    os.makedirs(os.path.join(ROOT, "bench", "results"), exist_ok=True)
    name = "showcase_run" + ("" if args.scenario == "kestrel" else f"_{args.scenario}") + \
        ("_noneural" if args.no_neural else "") + (f"_{args.tag}" if args.tag else "") + \
        (f"_s{args.seed}" if args.seed else "") + ".json"
    with open(os.path.join(ROOT, "bench", "results", name), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(json.dumps(out["kpis"], indent=1, default=str))
    print(json.dumps(out["calibration_error"], indent=1))


if __name__ == "__main__":
    main()
