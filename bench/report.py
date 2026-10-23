"""Print markdown tables from bench/results/*.json (used to write docs/BENCHMARKS.md).

    python -m bench.report
"""
from __future__ import annotations

import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "bench", "results")


def _load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def showcase(d) -> None:
    k = d["kpis"]
    p, l = k["plyxis"], k["legacy"]
    rows = [("Operating profit (USD)", "profit", 1.0), ("Oil produced (bbl)", "oil_bbl", 1.0),
            ("Gas (Mscf)", "gas_mscf", 1.0), ("Flared gas (Mscf)", "flared_mscf", 1.0),
            ("Emissions (t CO2e)", "emission_kg", 1e-3), ("Revenue (USD)", "revenue", 1.0),
            ("Cash cost (USD)", "cost", 1.0), ("Inventory value change (USD)", "inventory_value", 1.0),
            ("Pump life value change (USD)", "equipment_value", 1.0),
            ("Carbon cost at internal price (USD)", "carbon_cost", 1.0),
            ("Profit after carbon cost (USD)", "profit_after_carbon", 1.0)]
    print(f"### Showcase run ({d['hours']:.0f} h, neural={d['neural']}, wall {d['wall_seconds']:.0f} s)\n")
    print("| KPI | PLYXIS | manual operations | difference |\n|---|---|---|---|")
    for label, key, f in rows:
        a, b = p.get(key, 0) * f, l.get(key, 0) * f
        rel = f"{(a - b) / abs(b) * 100:+.1f} %" if abs(b) > 1e-9 else ""
        print(f"| {label} | {a:,.0f} | {b:,.0f} | {a - b:+,.0f} ({rel}) |")
    print(f"\nValue per day: USD {k['value_per_day']:,.0f}\n")
    if "terms" in d:
        tp, tl = d["terms"]["plyxis"], d["terms"]["legacy"]
        print("| Term (USD, + = income) | PLYXIS | manual operations | difference |\n|---|---|---|---|")
        for name in sorted(set(tp) | set(tl), key=lambda n: -abs(tp.get(n, 0) - tl.get(n, 0))):
            print(f"| {name} | {tp.get(name, 0):,.0f} | {tl.get(name, 0):,.0f} | {tp.get(name, 0) - tl.get(name, 0):+,.0f} |")
        print()
    c = d["calibration_error"]
    print("| Property (mean abs. rel. error) | engineering model | live twin at end |\n|---|---|---|")
    for prm, v in c.items():
        print(f"| {prm} | {v['engineering'] * 100:.1f} % | {v['twin'] * 100:.1f} % |")
    plans = d["plans"]
    print(f"\nPlans: {len(plans)}, feasible {sum(1 for x in plans if x['feasible'])}, "
          f"solve time mean {st.mean(x['seconds'] for x in plans):.1f} s, max {max(x['seconds'] for x in plans):.1f} s\n")
    # value by phase from the hourly records
    hourly = d["hourly"]
    cuts = [("normal operation", None, 13.0), ("CS-1 trip and recovery", 13.0, 20.0), ("night / ESP wear", 20.0, 28.0),
            ("SWD-2 injectivity loss", 28.0, 36.0), ("negative gas price", 36.0, 54.0),
            ("CS-2 trip, end of window", 54.0, 60.0), ("recovery to end", 60.0, None)]
    print("| Period | delta profit (USD) | PLYXIS flare (Mscf) | manual flare (Mscf) |\n|---|---|---|---|")
    for name, a, b in cuts:
        sel = [h for h in hourly if (a is None or h["t"] > a + 1e-6) and (b is None or h["t"] <= b + 1e-6)]
        if not sel:
            continue
        prev = [h for h in hourly if a is not None and h["t"] <= a + 1e-6]
        d0 = prev[-1]["delta_profit"] if prev else 0
        fp = sum(h["plyxis"].get("flare_mscfd", 0) for h in sel) / 24
        fl = sum(h["legacy"].get("flare_mscfd", 0) for h in sel) / 24
        print(f"| {name} | {sel[-1]['delta_profit'] - d0:+,.0f} | {fp:,.0f} | {fl:,.0f} |")
    print()


def neural(d) -> None:
    cases = d["cases"]
    print(f"### Neural ablation ({len(cases)} held-out situations, {d['experts']} expert solves, "
          f"training {d['train_seconds']:.0f} s)\n")
    print("| Start / method | mean objective (USD) | feasible (max viol. < 0.01) | mean seconds |\n|---|---|---|---|")
    names = {"reuse_raw": "previous plan, as is", "neural_raw": "neural sub-solver plan, as is",
             "reuse_repair": "previous plan + gradient repair", "neural_repair": "neural plan + gradient repair",
             "cascade": "full cascade", "cascade_nn": "full cascade + neural start"}
    for k, label in names.items():
        objs = [c[k][0] for c in cases]
        feas = sum(1 for c in cases if c[k][1] < 0.01)
        secs = st.mean(c[k][2] for c in cases) if len(cases[0][k]) > 2 else st.mean(c["neural_ms"] for c in cases) / 1e3
        print(f"| {label} | {st.mean(objs):,.0f} | {feas}/{len(cases)} | {secs:.2f} |")
    gains = [(c["cascade_nn"][0] - c["cascade"][0]) / abs(c["cascade"][0]) * 100 for c in cases]
    wins = sum(1 for g in gains if g > 0.05)
    ties = sum(1 for g in gains if abs(g) <= 0.05)
    print(f"\nCascade + neural vs cascade: wins {wins}, ties {ties}, losses {len(gains) - wins - ties}; "
          f"median {st.median(gains):+.1f} %, mean {st.mean(gains):+.1f} %; per case "
          + ", ".join(f"{g:+.1f}" for g in gains) + "\n")


def solvers(d) -> None:
    print(f"### Solver comparison ({d['variables']:,} variables, {d['discrete']:,} discrete, budget {d['budget_s']:.0f} s)\n")
    print("| Method | objective (USD) | max violation | feasible | seconds |\n|---|---|---|---|---|")
    for r in d["table"]:
        print(f"| {r['method']} | {r['objective']:,.0f} | {r['max_violation']:.4f} | "
              f"{'yes' if r['max_violation'] <= 3e-3 else 'no'} | {r['seconds']:.1f} |")
    print()


def scaling(d) -> None:
    print("### Scaling (generated enterprise twins, N Kestrel fields sharing a gas plant and a terminal)\n")
    hdr = ("| Fields | wells | decisions | discrete | compile s | simulate ms (B=1 / 16) | gradient ms | code lines "
           "| solve s | objective (USD) | max viol. | GA same time: objective / max viol. |")
    print(hdr + "\n|" + "---|" * 12)
    for r in d["rows"]:
        ga = f"{r['ga_objective']:,.0f} / {r['ga_max_violation']:.3f}" if "ga_objective" in r else "-"
        print(f"| {r['fields']} | {r['wells']} | {r['decision_variables']:,} | {r['discrete']:,} | {r['compile_s']:.2f} | "
              f"{r['simulate_ms_B1']:.0f} / {r['simulate_ms_B16']:.0f} | {r['gradient_ms']:.0f} | "
              f"{r['generated_code_lines']} | {r.get('solve_s', 0):.0f} | {r.get('objective', 0):,.0f} | "
              f"{r.get('max_violation', 0):.4f} | {ga} |")
    print()


def accuracy(d) -> None:
    rows = d["rows"]
    print(f"### Twin accuracy ({d['hours']:.0f} h, full planner={d['full_planner']})\n")
    print("| t (h) | q_max | GOR | water cut | pad gas mean / max | stations |\n|---|---|---|---|---|---|")
    for r in rows[::2] + ([rows[-1]] if len(rows) % 2 == 0 else []):
        print(f"| {r['t']:.0f} | {r['q_max_mae'] * 100:.1f} % | {r['gor_mae'] * 100:.1f} % | {r['wc_mae'] * 100:.1f} % | "
              f"{r['pad_gas_mae'] * 100:.1f} % / {r['pad_gas_max'] * 100:.0f} % | "
              + ", ".join(f"{k} {v * 100:+.1f} %" for k, v in r["station_gas"].items()) + " |")
    print()


def main() -> None:
    for name, fn in (("showcase_run.json", showcase), ("neural_ablation.json", neural),
                     ("solver_comparison.json", solvers), ("scaling.json", scaling), ("twin_accuracy.json", accuracy)):
        d = _load(name)
        if d is not None:
            fn(d)


if __name__ == "__main__":
    main()
