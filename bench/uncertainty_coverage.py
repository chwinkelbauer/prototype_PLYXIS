"""Is the twin's uncertainty honest? Posterior intervals of the calibrated properties against the hidden truth.

    python -m bench.uncertainty_coverage --scenario kestrel --hours 48
    python -m bench.uncertainty_coverage --scenario corvus --hours 24

A headless run of the scenario (the PLYXIS world, with planning). Every hour, for every calibrated property
(and every directly tested one), z = log(truth / estimate) / sigma, sigma from the estimator's posterior
covariance (`Estimator.uncertainty_model`). If the uncertainty is honest, |z| <= 1 holds for about 68 % and
|z| <= 2 for about 95 %; the same is reported for the fixed engineering sigma (0.15) as the comparison.
Writes bench/results/uncertainty_coverage[_<scenario>].json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from plyxis.runtime.engine import Engine  # noqa: E402


def snapshot(eng) -> list:
    est = eng.twin.estimator
    unc = est.uncertainty_model()
    prog, live, pp = eng.twin.model.program, eng.twin.live, eng.plants["plyxis"]
    out = []
    for i, (gid, pname, li) in enumerate(unc.params):
        g = prog.gmap[gid]
        path = prog.flat.instances[g.inst_ids[li]].path
        if path not in pp.path_index:
            continue
        gp, lp = pp.path_index[path]
        if pname not in pp.P.get(gp, {}):
            continue
        truth = float(pp.P[gp][pname].reshape(-1)[lp])
        est_v = float(live.P[gid][pname].reshape(-1)[li])
        if truth <= 0 or est_v <= 0:
            continue
        sd = math.sqrt(max(float(unc.cov[i, i]), 1e-12))
        out.append((g.ir.type.short, pname, math.log(truth / est_v), sd))
    return out


def summary(recs: list, sd_key: str) -> dict:
    zs = [e / (s if sd_key == "posterior" else 0.15) for (_t, _p, e, s) in recs]
    n = len(zs)
    if not n:
        return {}
    return {"n": n, "within_1sd": sum(abs(z) <= 1 for z in zs) / n, "within_2sd": sum(abs(z) <= 2 for z in zs) / n,
            "z_rms": math.sqrt(sum(z * z for z in zs) / n)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="kestrel")
    ap.add_argument("--hours", type=float, default=48)
    ap.add_argument("--warmup", type=float, default=6, help="hours before the first evaluation (first calibrations)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--matching", action="store_true", help="covariance matching on (Estimator.cov_matching)")
    args = ap.parse_args()
    t0 = time.perf_counter()
    eng = Engine(os.path.join(ROOT, "scenarios", args.scenario), seed=args.seed, sync_plans=True)
    recs, hourly = [], []

    def progress(e):
        e.twin.estimator.cov_matching = args.matching      # (the estimator is rebuilt when the twin is rewritten)
        if e.t - e.t0 < args.warmup:
            return
        snap = snapshot(e)
        recs.extend(snap)
        h = {"t": e.t, "posterior": summary(snap, "posterior"), "prior": summary(snap, "prior"),
             "mean_sd": sum(s for *_r, s in snap) / max(len(snap), 1), "info": dict(e.twin.estimator.cov_info)}
        hourly.append(h)
        print(f"{e.clock_str()}  posterior: within 1 sd {h['posterior'].get('within_1sd', 0):.2f}, 2 sd {h['posterior'].get('within_2sd', 0):.2f}, "
              f"z rms {h['posterior'].get('z_rms', 0):.2f}, mean sd {h['mean_sd']:.3f} | prior 0.15: 1 sd {h['prior'].get('within_1sd', 0):.2f} "
              f"z rms {h['prior'].get('z_rms', 0):.2f} | variance factor {h['info'].get('variance_factor', 0):.2f}  [{time.perf_counter() - t0:.0f}s]", flush=True)
    eng.run_headless(args.hours, progress=progress)
    by = {}
    for r in recs:
        by.setdefault(f"{r[0]}.{r[1]}", []).append(r)
    res = {"scenario": args.scenario, "hours": args.hours, "seed": args.seed, "warmup": args.warmup, "matching": args.matching,
           "matching_factors": dict(eng.twin.estimator._cm),
           "posterior": summary(recs, "posterior"), "prior": summary(recs, "prior"),
           "by_property": {k: {"posterior": summary(v, "posterior"), "prior": summary(v, "prior"),
                               "mean_sd": sum(x[3] for x in v) / len(v), "rms_log_error": math.sqrt(sum(x[2] ** 2 for x in v) / len(v))}
                           for k, v in by.items()},
           "hourly": hourly, "wall_seconds": time.perf_counter() - t0}
    print(json.dumps({k: res[k] for k in ("posterior", "prior")}, indent=1))
    for k, v in res["by_property"].items():
        print(f"  {k:24s} posterior {v['posterior']} | prior {v['prior']} | mean sd {v['mean_sd']:.3f} rms error {v['rms_log_error']:.3f}")
    name = "uncertainty_coverage" + ("" if args.scenario == "kestrel" else f"_{args.scenario}") + ("_matching" if args.matching else "") + ".json"
    with open(os.path.join(ROOT, "bench", "results", name), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
