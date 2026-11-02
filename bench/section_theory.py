"""Section dynamic programming vs. exact MILP and local search on scheduling-like table models with known
structure - the evidence for the claims in docs/DECOMPOSITION.md.

    python -m bench.section_theory                    # scaling series + coupling series, ~15 min
    python -m bench.section_theory --quick

Instances: S sections x W shifts = blocks with 8 joint options (3 machines on/off); strong interactions
over time (a section in consecutive shifts), moderate ones along a random network tree (the same shift),
weak ones on random extra edges that close cycles; one shared crew capacity over the horizon (at most 30 %
of the machine-shifts). Methods at equal information:

    section DP      forest DP + price on the crew capacity + repair + coordinate ascent (sectiondp.solve),
                    with its certified upper bound
    coordinate      block coordinate ascent from the empty plan (exact per block, no DP)
    multi-start     coordinate ascent from 8 random feasible starts, best kept
    MILP            HiGHS on the exact linearisation (one-hot options, pairwise marginals, the crew knapsack),
                    time limit; its dual bound when it does not finish

Writes bench/results/section_theory.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from plyxis.optim import sectiondp as SD  # noqa: E402

Q = 8                         # joint options per block: 3 binary decisions


def instance(S: int, W: int = 6, extra: float = 0.1, weak: float = 0.1, seed: int = 0) -> SD.TableModel:
    rng = np.random.default_rng(seed)
    blk = lambda s, w: s * W + w  # noqa: E731
    on = np.array([bin(o).count("1") for o in range(Q)], dtype=float)
    unary = [rng.normal(size=Q) + 0.6 * on for _ in range(S * W)]           # running machines earns, mostly
    edges, pair = [], []

    def add(a, b, scale):
        edges.append((a, b))
        pair.append(scale * rng.normal(size=(Q, Q)))
    for s in range(S):                                                       # time: strong
        for w in range(W - 1):
            add(blk(s, w), blk(s, w + 1), 1.0)
    parent = [-1] + [int(rng.integers(0, s)) for s in range(1, S)]           # network tree: moderate
    for s in range(1, S):
        for w in range(W):
            add(blk(parent[s], w), blk(s, w), 0.5)
    for _ in range(int(extra * S * W)):                                      # random extra edges: weak
        a, b = rng.choice(S * W, 2, replace=False)
        add(int(a), int(b), weak)
    m = SD.TableModel(unary, edges, pair)
    m.usage = [[on.copy() for _ in range(S * W)]]
    m.cap = np.array([0.3 * 3 * S * W])
    return m


def milp_solve(m: SD.TableModel, time_limit: float) -> dict:
    n, E = m.n, len(m.edges)
    nx, ny = n * Q, E * Q * Q
    c = -np.concatenate([np.concatenate(m.unary), np.concatenate([P.reshape(-1) for P in m.pair])])
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0
    for b in range(n):                                       # one option per block
        rows += [r] * Q
        cols += list(range(b * Q, b * Q + Q))
        vals += [1.0] * Q
        lo.append(1.0)
        hi.append(1.0)
        r += 1
    for k, (a, b) in enumerate(m.edges):                     # marginals of the pairwise variables
        base = nx + k * Q * Q
        for o in range(Q):
            rows += [r] * (Q + 1)
            cols += [base + o * Q + p for p in range(Q)] + [a * Q + o]
            vals += [1.0] * Q + [-1.0]
            lo.append(0.0)
            hi.append(0.0)
            r += 1
        for p in range(Q):
            rows += [r] * (Q + 1)
            cols += [base + o * Q + p for o in range(Q)] + [b * Q + p]
            vals += [1.0] * Q + [-1.0]
            lo.append(0.0)
            hi.append(0.0)
            r += 1
    for h, use in enumerate(m.usage):                        # shared crew capacity
        rows += [r] * nx
        cols += list(range(nx))
        vals += list(np.concatenate(use))
        lo.append(-np.inf)
        hi.append(float(m.cap[h]))
        r += 1
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(r, nx + ny))
    integ = np.concatenate([np.ones(nx), np.zeros(ny)])
    t = time.perf_counter()
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=integ, bounds=Bounds(0, 1),
               options={"time_limit": time_limit, "disp": False, "mip_rel_gap": 1e-6})
    out = {"seconds": time.perf_counter() - t, "status": int(res.status), "message": res.message}
    if res.x is not None:
        z = res.x[:nx].reshape(n, Q).argmax(1)
        out["value"] = SD.value(m, z)
        out["feasible"] = SD.feasible(m, z)
    bound = getattr(res, "mip_dual_bound", None)
    out["bound"] = -float(bound) if bound is not None and np.isfinite(bound) else None
    out["optimal"] = res.status == 0
    return out


DUAL = "subgradient"


def run(m: SD.TableModel, milp_limit: float, rng) -> dict:
    t = time.perf_counter()
    s = SD.solve(m, dual=DUAL)
    rec = {"blocks": m.n, "edges": len(m.edges),
           "dp": {"value": s.value, "upper_bound": s.upper_bound, "certified_gap": s.gap, "feasible": s.feasible,
                  "seconds": time.perf_counter() - t, **s.info}}
    t = time.perf_counter()
    z = SD.coordinate_ascent(m, np.zeros(m.n, dtype=np.int64), sweeps=50)
    rec["coordinate"] = {"value": SD.value(m, z), "feasible": SD.feasible(m, z), "seconds": time.perf_counter() - t}
    t = time.perf_counter()
    best = -np.inf
    for _ in range(8):
        z0 = SD.repair(m, rng.integers(0, Q, size=m.n))
        z = SD.coordinate_ascent(m, z0, sweeps=50)
        if SD.feasible(m, z):
            best = max(best, SD.value(m, z))
    rec["multistart"] = {"value": best, "seconds": time.perf_counter() - t}
    if milp_limit > 0:
        rec["milp"] = milp_solve(m, milp_limit)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--milp-limit", type=float, default=120.0)
    ap.add_argument("--milp-max-blocks", type=int, default=9000)
    ap.add_argument("--dual", default="subgradient", choices=["subgradient", "smooth"],
                    help="certificate: projected subgradient or smoothed dual decomposition (L-BFGS)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    global DUAL
    DUAL = args.dual
    rng = np.random.default_rng(0)
    sizes = [20, 60, 200] if args.quick else [20, 60, 200, 600, 1500, 5000, 15000]
    out = {"scaling": [], "coupling": [], "dual": args.dual}
    for S in sizes:
        m = instance(S)
        lim = args.milp_limit if m.n <= args.milp_max_blocks else 0.0
        rec = run(m, lim, rng)
        rec["sections"] = S
        out["scaling"].append(rec)
        d, ml = rec["dp"], rec.get("milp", {})
        print(f"S {S:6d} blocks {m.n:7d} | DP {d['value']:12.2f} gap<= {100 * d['certified_gap']:.3f} % "
              f"{d['seconds']:7.2f}s | coord {rec['coordinate']['value']:12.2f} | multi {rec['multistart']['value']:12.2f} "
              f"{rec['multistart']['seconds']:6.1f}s | MILP {ml.get('value', float('nan')):12.2f} "
              f"bound {ml.get('bound') or float('nan'):12.2f} {ml.get('seconds', 0):6.1f}s opt {ml.get('optimal')}", flush=True)
    for weak in ([0.0, 0.3, 1.0] if args.quick else [0.0, 0.1, 0.3, 0.6, 1.0]):
        m = instance(200, extra=0.3, weak=weak, seed=1)
        rec = run(m, args.milp_limit, rng)
        rec["weak"] = weak
        out["coupling"].append(rec)
        d, ml = rec["dp"], rec.get("milp", {})
        print(f"weak {weak:4.2f} | DP {d['value']:10.2f} off-forest bound {d['off_forest_bound']:8.2f} "
              f"gap<= {100 * d['certified_gap']:.2f} % | MILP {ml.get('value', float('nan')):10.2f} opt {ml.get('optimal')}",
              flush=True)
    os.makedirs(os.path.join(ROOT, "bench", "results"), exist_ok=True)
    name = "section_theory" + (f"_{args.tag}" if args.tag else "") + ".json"
    with open(os.path.join(ROOT, "bench", "results", name), "w") as fh:
        json.dump(out, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
