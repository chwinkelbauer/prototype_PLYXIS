"""Twin accuracy: how well does the live twin mirror the (hidden) plant over time?

Every hour of plant time it compares the twin with the emulator's ground truth:
  * well properties q_max, GOR, water cut - mean / max absolute relative error
  * gas per pad header and per compressor station - twin one-step prediction vs plant

    python -m bench.twin_accuracy --hours 30            # cheap planner (fast)
    python -m bench.twin_accuracy --hours 30 --full     # production planner incl. neural sub-solvers

Writes bench/results/twin_accuracy.json.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from plyxis.optim.cascade import CascadeConfig  # noqa: E402
from plyxis.runtime.engine import Engine  # noqa: E402


def _sig(prog, raw, path, key):
    gid, li = prog.inst_group[prog.flat.by_path[path]]
    x = prog.signal(raw, gid, key)
    x = x.reshape(x.shape[0], -1, x.shape[-1])
    return float(x[0, li if x.shape[1] > 1 else 0, -1])


def measure(eng) -> dict:
    pp = eng.plants["plyxis"]
    prog = eng.twin.model.program
    errs = {"q_max": [], "gor": [], "wc": []}
    for g in prog.groups:
        if g.is_env or g.ir.type.short != "esp_well":
            continue
        for li, iid in enumerate(g.inst_ids):
            path = prog.flat.instances[iid].path
            gp = pp.path_index.get(path)
            if gp is None or path in pp.hidden:
                continue
            for k in errs:
                truth = float(pp.P[gp[0]][k].reshape(-1)[gp[1]])
                errs[k].append((float(eng.twin.live.P[g.gid][k].reshape(-1)[li]) - truth) / abs(truth))
    est = eng.twin.estimator
    D, R, E, S0, _ = est._window(list(est.live.buf)[-2:])
    phys = eng.twin.live.phys
    raw_t = phys.simulate(D, R, est.live.P, S0, E, T=1, dt=est.dt, t0=est.live.buf[-1]["t"] - est.dt)
    raw_p = pp.last_raw
    pads = [p.path for p in prog.flat.instances if p.type.short == "pad"]
    pad_err = [(_sig(pp.prog, raw_p, p, "node:out:header:gas") - _sig(phys, raw_t, p, "node:out:header:gas"))
               / max(abs(_sig(phys, raw_t, p, "node:out:header:gas")), 1e-6) for p in pads]
    stations = [p.path for p in prog.flat.instances if p.type.short == "compressor_station"]
    cs_err = {c: (_sig(pp.prog, raw_p, c, "node:local:suction_gas") - _sig(phys, raw_t, c, "node:local:suction_gas"))
              / max(abs(_sig(phys, raw_t, c, "node:local:suction_gas")), 1e-6) for c in stations}
    mean_abs = lambda xs: st.mean(abs(x) for x in xs)  # noqa: E731
    return {"t": round(eng.t, 2),
            **{f"{k}_mae": round(mean_abs(v), 4) for k, v in errs.items()},
            **{f"{k}_max": round(max(abs(x) for x in v), 4) for k, v in errs.items()},
            "gor_bias": round(st.mean(errs["gor"]), 4),
            "pad_gas_mae": round(mean_abs(pad_err), 4), "pad_gas_max": round(max(abs(x) for x in pad_err), 4),
            "station_gas": {c: round(v, 4) for c, v in cs_err.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=30.0)
    ap.add_argument("--full", action="store_true", help="production planner with neural sub-solvers")
    args = ap.parse_args()
    t0 = time.perf_counter()
    eng = Engine(os.path.join(ROOT, "scenarios", "kestrel"), sync_plans=True, use_neural=args.full,
                 online_learning=False)
    if not args.full:
        eng.planner.cfg = CascadeConfig(relax_iters=60, population=4, generations=1, repair_iters=30,
                                        polish_iters=15, use_neural=False, multistart=2)
    eng.run_headless(2.0 / 6.0)                 # the one-step comparison needs two data records
    rows = [measure(eng)]
    end = eng.t + args.hours
    while eng.t < end - 1e-9:
        eng.step()
        if eng.steps % 12 == 0:
            rows.append(measure(eng))
            r = rows[-1]
            print(f"t={r['t']:5.1f}  q_max {r['q_max_mae']:.3f}  gor {r['gor_mae']:.3f}  wc {r['wc_mae']:.3f}  "
                  f"pad gas {r['pad_gas_mae']:.3f}/{r['pad_gas_max']:.2f}  stations {r['station_gas']}", flush=True)
    out = {"hours": args.hours, "full_planner": args.full, "wall_seconds": time.perf_counter() - t0, "rows": rows}
    os.makedirs(os.path.join(ROOT, "bench", "results"), exist_ok=True)
    with open(os.path.join(ROOT, "bench", "results", "twin_accuracy.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
