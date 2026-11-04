"""Planner vs. the scenario's conventional policy in the twin's own world: the engineering model as the plant (no
hidden truth), constant nominal prices, no events, no noise. What is left is the decision quality alone - no
estimation error, no forecast error, no re-planning.

    python -m bench.policy_in_twin --scenario aquila [--hours 96]

Both are scored like the A/B: operating result (sales - cash cost + change of the inventories' value, planning-only
terms excluded) and after the model's internal carbon price. Writes bench/results/policy_in_twin_<scenario>.json.
Inventories at the start are valued as stock x unit cost (the steel and mining yards' terminal values).
`stocks` reports how much each kind of yard changed: one open-loop plan can end below its stock targets (it sells the
yards down at its horizon end), which a rolling plan never reaches - read it before quoting the difference.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

import torch

from plyxis.data.emulator import PlantEmulator, _Kpi
from plyxis.lang import compile_model
from plyxis.lang import units as U
from plyxis.lang.program import _ident
from plyxis.optim.cascade import Cascade
from plyxis.optim.problem import Problem
from plyxis.runtime.planner import Planner
from plyxis.runtime.scenario import Scenario

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def policy(sc, hours: float) -> dict:
    """The conventional rules (`legacy.py`) operating the engineering model, hour by hour."""
    emu = PlantEmulator(sc.twin_files, sc.twin_env, dt=1.0, seed=0, name="legacy", noise=0.0,
                        kpi_metrics=sc.raw.get("kpi", {}).get("metrics"))
    ops = sc.legacy_class()(emu, sc.legacy_cfg)
    t0 = time.perf_counter()
    while emu.t < hours - 1e-9:
        ops.control(emu.t)
        emu.step()
    c = emu.kpi.cum
    end = {p: float(emu.S[g]["stock"].reshape(-1)[i]) for p, (g, i) in emu.path_index.items() if "stock" in emu.S.get(g, {})}
    return {"profit": c["profit"], "profit_after_carbon": c["profit_after_carbon"], "emission_t": c["emission_kg"] / 1000.0,
            "metrics": {k: v for k, v in c.items() if k.startswith("m_")}, "seconds": time.perf_counter() - t0,
            "protection_cuts": len(emu.throttle_log), "underflow": dict(emu.underflow),
            "stocks": _stock_change(compile_model(sc.twin_files, env=sc.twin_env).program, lambda path: end.get(path))}


def _stock_change(prog, end_of) -> dict:
    """Change of every kind of yard over the run: t and USD at the yard's unit cost, by asset type."""
    out: dict = {}
    for g in prog.groups:
        P0, S0 = prog.P0.get(g.gid, {}), prog.S0.get(g.gid, {})
        if g.is_env or "stock" not in S0 or "unit_cost" not in P0:
            continue
        s0, uc = torch.as_tensor(S0["stock"]).reshape(-1), P0["unit_cost"].reshape(-1)
        a = out.setdefault(g.ir.type.short, {"t": 0.0, "usd": 0.0})
        for i, iid in enumerate(g.inst_ids):
            d = end_of(prog.flat.instances[iid].path) - float(s0[i])
            a["t"] += U.from_canonical(d, "t")
            a["usd"] += d * float(uc[i])
    return out


def plan(sc) -> dict:
    """One plan of the live cascade over the same horizon, valued with the same terms."""
    prog = compile_model(sc.twin_files, env=sc.twin_env).program
    pb = Problem(prog)
    t0 = time.perf_counter()
    cfg = dataclasses.replace(Planner.LIVE, **sc.raw.get("planner", {}))            # the scenario's own planner budget
    cfg = dataclasses.replace(cfg, use_neural=False, exact_branches=tuple(cfg.exact_branches))
    res = Cascade(pb, config=cfg).solve()
    secs = time.perf_counter() - t0
    with torch.no_grad():
        e = pb.evaluate(res.x[:1], relax=False)
    one = lambda v: float(torch.as_tensor(v).reshape(-1)[0])  # noqa: E731
    named = {k: one(v) for k, v in e.totals["named"].items()}
    running = {tr.name: tr.kind for tr in prog.terms if not tr.final}
    assert not set(running) & {tr.name for tr in prog.terms if tr.final}, "a running and a final term share a name"
    profit = 0.0
    for name, kind in running.items():                    # realised running terms, as the KPI counts them
        if name not in _Kpi.EXPECTATION_TERMS and kind in ("revenue", "cost"):
            profit += named.get(name, 0.0) * (1.0 if kind == "revenue" else -1.0)
    for tr in prog.terms:                                  # value of the inventories at the end ...
        if tr.final and tr.name not in _Kpi.EXPECTATION_TERMS and tr.kind in ("revenue", "cost"):
            v = float(torch.as_tensor(e.raw[f"F_{tr.gid}_{_ident(tr.key)}"]).reshape(1, -1).sum())
            profit += v if tr.kind == "revenue" else -v
    for g in prog.groups:                                  # ... less their value at the start
        S, P = pb.S0.get(g.gid, {}), pb.P.get(g.gid, {})
        if "stock" in S and "unit_cost" in P:
            profit -= float((torch.as_tensor(S["stock"]).reshape(-1) * P["unit_cost"].reshape(-1)).sum())
    em = one(e.totals["emission"])
    cp = float(prog.E0.get("carbon_price", 0.0))
    end = {}
    for g in prog.groups:
        if "stock" in pb.S0.get(g.gid, {}):
            st = e.raw[prog.node_var(g.gid, "state:stock")]
            st = st.reshape(st.shape[0], -1, st.shape[-1])[0, :, -1]
            end.update({prog.flat.instances[iid].path: float(st[i]) for i, iid in enumerate(g.inst_ids)})
    return {"profit": profit, "profit_after_carbon": profit - cp * em, "emission_t": em / 1000.0,
            "metrics": {f"m_{m['key']}": U.from_canonical(named.get(m["metric"], 0.0), m["unit"])
                        for m in sc.raw.get("kpi", {}).get("metrics", [])},
            "seconds": secs, "decisions": pb.space.N, "max_violation": float(res.max_violation),
            "exact_branches": list(cfg.exact_branches), "stocks": _stock_change(prog, end.get)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="aquila")
    ap.add_argument("--hours", type=float, default=None, help="policy hours (default: the twin's planning horizon)")
    args = ap.parse_args()
    sc = Scenario.load(os.path.join(ROOT, "scenarios", args.scenario))
    hours = args.hours or compile_model(sc.twin_files, env=sc.twin_env).program.horizon
    out = {"scenario": args.scenario, "hours": hours, "policy": policy(sc, hours), "plan": plan(sc)}
    p, q = out["plan"], out["policy"]
    out["delta"] = {k: p[k] - q[k] for k in ("profit", "profit_after_carbon", "emission_t")}
    os.makedirs(os.path.join(ROOT, "bench", "results"), exist_ok=True)
    with open(os.path.join(ROOT, "bench", "results", f"policy_in_twin_{args.scenario}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    for w in ("policy", "plan"):
        r = out[w]
        print(f"{w:7s} profit {r['profit']/1e6:9.2f} M  after carbon {r['profit_after_carbon']/1e6:8.2f} M  "
              f"CO2e {r['emission_t']/1e3:8.1f} kt  {r['seconds']:6.1f} s")
        print("        yards: " + ", ".join(f"{k} {v['t']:+,.0f} t (USD {v['usd']/1e6:+.1f} M)" for k, v in r["stocks"].items()))
    print("delta", {k: round(v / 1e6, 2) if k != "emission_t" else round(v / 1e3, 1) for k, v in out["delta"].items()})


if __name__ == "__main__":
    main()
