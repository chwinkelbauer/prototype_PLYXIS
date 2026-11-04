"""Column generation with exact DP sub-problems (plyxis/optim/colgen.py) against an exact MILP, by site size.

    python -m bench.colgen                       # ~8 min (the MILP runs to its time limit from about 30 assets)
    python -m bench.colgen --sizes 10 40 160 --milp-max 40

Wells-like units (rate linear in speed, power ~ speed^3, wear ~ speed^4, on/off every 6 h with restart costs,
speed per 2 h at 5 levels) share one gas limit per hour (55 % of what all of them could make), 24 h. Per size:
column generation (rounds, columns, seconds, value, proven bound, gap) and, up to --milp-max assets, the same
discrete model as one MILP (HiGHS, time limit): value, its own bound, whether it proved optimality. Timing:
run alone (bench notes in CLAUDE.md). Writes bench/results/colgen.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from plyxis.optim import colgen as CG  # noqa: E402


def learned(n: int = 60, train: int = 48, test: int = 16) -> dict:
    """Learned prices (plyxis/optim/learnprice.py): trained on column generation's own prices for `train` random
    situations (limit at 35-80 % of what all units could make, prices +-30 %), tested on `test` others: rounds,
    seconds and gap from a cold start, from the predicted prices, and from the exact prices (the best a start can do)."""
    import numpy as np
    from plyxis.optim import learnprice as LP
    rng = np.random.default_rng(7)

    def situation(seed):
        return CG.random_family(n, seed=seed, cap_frac=float(rng.uniform(0.35, 0.8)), price_scale=float(rng.uniform(0.7, 1.3)))
    t = time.perf_counter()
    data = []
    for s in range(train):
        fam, cap = situation(10_000 + s)
        data.append(LP.target(fam, cap, CG.solve(fam, cap).prices))
    net = LP.fit(data)
    train_s = time.perf_counter() - t
    rows = []
    for s in range(test):
        fam, cap = situation(20_000 + s)
        cold = CG.solve(fam, cap)
        warm = CG.solve(fam, cap, warm_pi=LP.predict(net, fam, cap))
        exact = CG.solve(fam, cap, warm_pi=cold.prices)
        rows.append({k: {"rounds": r.rounds, "seconds": r.seconds, "gap": r.gap, "value": r.value}
                     for k, r in (("cold", cold), ("learned", warm), ("exact_prices", exact))})
        print(f"   test {s:2d}: rounds cold {cold.rounds:3d} learned {warm.rounds:3d} exact {exact.rounds:3d} | gap "
              f"{100 * cold.gap:.3f} / {100 * warm.gap:.3f} / {100 * exact.gap:.3f} %", flush=True)
    mean = lambda k, f: float(np.mean([r[k][f] for r in rows]))  # noqa: E731
    out = {"assets": n, "train": train, "test": test, "train_seconds": train_s, "train_loss": net.train_loss,
           "mean": {k: {f: mean(k, f) for f in ("rounds", "seconds", "gap")} for k in ("cold", "learned", "exact_prices")},
           "tests": rows}
    print("learned prices:", json.dumps(out["mean"]), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="*", default=[10, 20, 40, 80, 160, 320, 640, 1280, 2560])
    ap.add_argument("--milp-max", type=int, default=80)
    ap.add_argument("--milp-seconds", type=float, default=60.0)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--learn", action="store_true", help="only the learned-prices experiment (written into colgen.json)")
    args = ap.parse_args()
    path = os.path.join(ROOT, "bench", "results", "colgen.json")
    if args.learn:
        old = json.load(open(path)) if os.path.exists(path) else {}
        old["learned_prices"] = learned()
        with open(path, "w") as fh:
            json.dump(old, fh, indent=1, default=float)
        return
    runs = []
    for n in args.sizes:
        for seed in range(args.seeds):
            fam, cap = CG.random_family(n, seed=1000 * seed + n)
            t = time.perf_counter()
            r = CG.solve(fam, cap)
            rec = {"assets": n, "seed": seed, "decisions": int(n * (fam.value.shape[1] // fam.hold + fam.value.shape[1] // fam.seg)),
                   "cg": {"seconds": time.perf_counter() - t, "rounds": r.rounds, "columns": r.columns, "value": r.value,
                          "bound": r.bound, "gap": r.gap, "violation": r.violation}}
            if n <= args.milp_max:
                ex = CG.exact_milp(fam, cap, seconds=args.milp_seconds)
                rec["milp"] = {k: ex.get(k) for k in ("seconds", "status", "value", "bound")}
                rec["milp"]["optimal"] = ex["status"] == 0
                if ex.get("value") is not None:
                    rec["cg"]["below_milp"] = (ex["value"] - r.value) / abs(ex["value"])
            runs.append(rec)
            m = rec.get("milp")
            print(f"{n:5d} assets seed {seed}: CG {rec['cg']['seconds']:6.2f} s, {r.rounds:3d} rounds, gap to bound {100 * r.gap:6.3f} %"
                  + (f" | MILP {m['seconds']:6.1f} s, {'optimal' if m['optimal'] else 'time limit'}, CG below MILP by "
                     f"{100 * rec['cg'].get('below_milp', float('nan')):.3f} %" if m else ""), flush=True)
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update({"runs": runs, "milp_seconds": args.milp_seconds})
    with open(path, "w") as fh:
        json.dump(old, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
