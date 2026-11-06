"""The A/B as the console runs it: the plant clock in wall time at `--speed` (600x by default, as scripts/start.ps1),
plans computed in the background while the plant moves on (the showcase benchmark instead waits for every plan).

    python -m bench.console_run --scenario aquila [--speed 600] [--seed 0] [--tag x]

Boots like the server (initial plan synchronously, then the plant-clock thread), presses play, and logs every plant
hour: both worlds' cumulative profit, the plan the plant executes (id, the plant hour it was computed for) and how
far the plant has moved since. Writes bench/results/console_run_<scenario>[_tag].json. Not timing-sensitive in the
usual sense - it measures what wall time does to the plans - but it needs the machine to itself to mean anything.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from plyxis.runtime.engine import Engine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="aquila")
    ap.add_argument("--speed", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--no-neural", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.perf_counter()
    eng = Engine(os.path.join(ROOT, "scenarios", a.scenario), seed=a.seed, speed=a.speed, use_neural=not a.no_neural)
    eng._replan("initial plan")
    boot = time.perf_counter() - t0
    hours = a.hours or eng.sc.end_hour
    end = eng.sc.start_hour + hours
    eng.start()
    eng.running = True
    log, last_h = [], -1
    tw = time.perf_counter()
    while eng.t < end - 1e-6 and eng.running:
        time.sleep(0.2)
        h = int(eng.t)
        if h != last_h:
            last_h = h
            k = eng.kpis()
            plan = eng.planner.plan
            rec = {"t": round(eng.t, 3), "wall": round(time.perf_counter() - tw, 1),
                   "plyxis": round(k["plyxis"]["cum"]["profit"]), "legacy": round(k["legacy"]["cum"]["profit"]),
                   "delta": round(k["delta"]["profit"]), "plan": plan.pid if plan else None,
                   "plan_t": getattr(plan, "t0", None), "solving": bool(eng.planner.running)}
            log.append(rec)
            print(json.dumps(rec), flush=True)
    eng.stop()
    time.sleep(0.5)
    k = eng.kpis()
    out = {"scenario": a.scenario, "speed": a.speed, "seed": a.seed, "hours": hours, "boot_seconds": boot,
           "wall_seconds": time.perf_counter() - tw,
           "kpis": {"plyxis": k["plyxis"]["cum"], "legacy": k["legacy"]["cum"], "delta": k["delta"]},
           "bridge": k["bridge"], "plans": eng.plan_events, "hourly": log,
           "runs": [{kk: r.get(kk) for kk in ("id", "t", "trigger", "objective", "feasible", "seconds")}
                    for r in list(eng.planner.runs)]}
    path = os.path.join(ROOT, "bench", "results", f"console_run_{a.scenario}{('_' + a.tag) if a.tag else ''}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(json.dumps(out["kpis"]["delta"]), flush=True)


if __name__ == "__main__":
    main()
