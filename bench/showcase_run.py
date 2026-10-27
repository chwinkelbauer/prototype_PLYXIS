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


def calibration_error(eng, base) -> dict:
    """Mean absolute relative error of the well properties vs the plant's actual values:
    engineering model (field.plx as delivered) vs the live twin after the run."""
    live, prog, pp = eng.twin.live, eng.twin.model.program, eng.plants["plyxis"]
    out = {}
    for param in ("q_max", "wc", "gor"):
        e0, e1 = [], []
        for g in prog.groups:
            if g.is_env or g.ir.type.short != "esp_well":
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
        out[param] = {"engineering": round(statistics.mean(e0), 4), "twin": round(statistics.mean(e1), 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=72.0)
    ap.add_argument("--no-neural", action="store_true")
    ap.add_argument("--skip-events", default="", help="comma list of assets / price names whose events are dropped "
                                                        "(ablation of scenario events), e.g. GASPLANT,power_price")
    ap.add_argument("--tag", default="", help="suffix of the result file, e.g. 'core' -> showcase_run_core.json")
    args = ap.parse_args()
    t0 = time.perf_counter()
    eng = Engine(os.path.join(ROOT, "scenarios", "kestrel"), sync_plans=True, use_neural=not args.no_neural)
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
        print(f"{rec['clock']}  oil P {p['rate'].get('oil_bpd', 0):7,.0f} L {l['rate'].get('oil_bpd', 0):7,.0f} | "
              f"flare P {p['rate'].get('flare_mscfd', 0):6,.0f} L {l['rate'].get('flare_mscfd', 0):6,.0f} | "
              f"delta profit {k['delta']['profit']:11,.0f}  [{time.perf_counter() - t0:5.0f}s]", flush=True)
    eng.run_headless(args.hours, progress=progress)
    k = eng.kpis()
    out = {
        "hours": args.hours, "wall_seconds": time.perf_counter() - t0, "neural": not args.no_neural,
        "skipped_events": sorted(skip),
        "kpis": {"plyxis": k["plyxis"]["cum"], "legacy": k["legacy"]["cum"], "delta": k["delta"],
                 "value_per_day": k["value_per_day"]},
        "terms": {w: dict(eng.plants[w].kpi.terms) for w in ("plyxis", "legacy")},
        "bridge": k["bridge"],
        "calibration_error": calibration_error(eng, compile_model([os.path.join(ROOT, "scenarios", "kestrel", "field.plx")])),
        "plans": list(eng.plan_events),
        "runs": [{kk: v for kk, v in r.items() if kk != "history"} for r in eng.planner.runs],
        "timeline": eng.timeline, "hourly": hourly,
        "alerts": list(eng.alerts)[-150:],
        "commits": eng.twin.repo.log(50),
        "learn": eng.learn_log,
    }
    os.makedirs(os.path.join(ROOT, "bench", "results"), exist_ok=True)
    name = "showcase_run" + ("_noneural" if args.no_neural else "") + (f"_{args.tag}" if args.tag else "") + ".json"
    with open(os.path.join(ROOT, "bench", "results", name), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(json.dumps(out["kpis"], indent=1, default=str))
    print(json.dumps(out["calibration_error"], indent=1))


if __name__ == "__main__":
    main()
