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
            ("Flared above permit (Mscf)", "flared_over_permit_mscf", 1.0),
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
    if "bridge" in d:
        print("| Where the operating-profit difference comes from | USD |\n|---|---|")
        for r in d["bridge"]:
            print(f"| {r['label']} | {r['delta']:+,.0f} |")
        print(f"| **total** | **{sum(r['delta'] for r in d['bridge']):+,.0f}** |\n")
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
    cuts = [("normal operation", None, 13.0), ("CS-1 trip", 13.0, 19.0),
            ("gas plant upset (takeaway -1/3, 10 h), ESP wear", 19.0, 28.0),
            ("SWD-2 injectivity loss, new well", 28.0, 36.0), ("negative gas price", 36.0, 54.0),
            ("CS-2 trip during negative price", 54.0, 60.0), ("evening, ERCOT cap 17:00-19:00", 60.0, 67.0),
            ("recovery to end", 67.0, None)]
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


def variants() -> None:
    """Full showcase vs. ablation runs (bench.showcase_run --skip-events ... --tag ...)."""
    names = sorted(n for n in os.listdir(RES) if n.startswith("showcase_run") and n.endswith(".json"))
    runs = [(n, d) for n in names if (d := _load(n)) and d.get("scenario", "kestrel") == "kestrel"]
    if len(runs) < 2:
        return
    print("### Showcase variants (72 h, PLYXIS vs. manual operations)\n")
    print("| Run | events dropped | operating profit | profit after carbon | oil | flaring |\n|---|---|---|---|---|---|")
    for n, d in runs:
        p, l = d["kpis"]["plyxis"], d["kpis"]["legacy"]

        def rel(key):
            a, b = p.get(key, 0.0), l.get(key, 0.0)
            return f"{a - b:+,.0f} ({(a - b) / abs(b) * 100:+.1f} %)" if abs(b) > 1e-9 else "-"
        print(f"| {n} | {', '.join(d.get('skipped_events', [])) or 'none'} | {rel('profit')} | "
              f"{rel('profit_after_carbon')} | {rel('oil_bbl')} | {rel('flared_mscf')} |")
    print()


def _head_grade(copper_t: float, milled_t: float) -> float:
    """Head grade (%) behind copper in concentrate = milled x g / 100 x recovery(g), with the concentrator's
    recovery curve (stdlib mining.plx: pwl(g, [0.3, 0.8, 1.5], [0.70, 0.88, 0.92])); bisection."""
    def rec(g):
        xs, ys = (0.3, 0.8, 1.5), (0.70, 0.88, 0.92)
        if g <= xs[0]:
            return ys[0]
        if g >= xs[-1]:
            return ys[-1]
        i = 0 if g < xs[1] else 1
        return ys[i] + (g - xs[i]) / (xs[i + 1] - xs[i]) * (ys[i + 1] - ys[i])
    target, lo, hi = copper_t / max(milled_t, 1e-9) * 100.0, 0.0, 10.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mid * rec(mid) < target else (lo, mid)
    return 0.5 * (lo + hi)


def mine(d) -> None:
    """Showcase A/B of the Corvus underground mine (generic production metrics m_*)."""
    k = d["kpis"]
    p, l = k["plyxis"], k["legacy"]
    print(f"### Corvus mine showcase ({d['hours']:.0f} h, wall {d['wall_seconds']:.0f} s)\n")
    print("| KPI | PLYXIS | manual operations | difference |\n|---|---|---|---|")
    rows = [("Operating profit (USD)", "profit"), ("Profit after carbon cost (USD)", "profit_after_carbon"),
            ("Revenue (USD)", "revenue"), ("Cash operating cost (USD)", "cost"),
            ("Inventory value change (USD)", "inventory_value")]
    rows += [(f"{key[2:]} ({'t' if key != 'm_power' else 'MWh'})", key) for key in p if key.startswith("m_")]
    rows += [("Emissions (t CO2e)", "emission_kg")]
    if "overflow_m3" in p or "overflow_m3" in l:
        rows += [("Sump overflow (m3)", "overflow_m3")]
    for label, key in rows:
        f = 1e-3 if key == "emission_kg" else 1.0
        a, b = p.get(key, 0) * f, l.get(key, 0) * f
        rel = f"{(a - b) / abs(b) * 100:+.1f} %" if abs(b) > 1e-9 else ""
        print(f"| {label} | {a:,.0f} | {b:,.0f} | {a - b:+,.0f} ({rel}) |")
    for w, lab in ((p, "PLYXIS"), (l, "manual")):
        if w.get("m_milled"):
            print(f"\n{lab}: cash cost per tonne milled USD {w['cost'] / w['m_milled']:.2f}, "
                  f"annualised cash cost USD {w['cost'] / d['hours'] * 8760 / 1e6:,.0f} M")
        if w.get("m_copper"):
            print(f"{lab}: mean head grade {_head_grade(w['m_copper'], w['m_milled']):.3f} % Cu")
            # unit cost of production (C1 style): cash cost plus ore inventories drawn down (at cost), per t of copper
            prod = w["cost"] - w.get("inventory_value", 0.0)
            print(f"{lab}: production cost USD {prod:,.0f} = USD {prod / w['m_copper']:,.0f} per t copper "
                  f"(cash cost USD {w['cost'] / w['m_copper']:,.0f} per t)")
    if "bridge" in d:
        print("\n| Where the operating-profit difference comes from | USD |\n|---|---|")
        for r in d["bridge"]:
            print(f"| {r['label']} | {r['delta']:+,.0f} |")
    inv = d.get("inventory_terms")
    if inv:
        print("\n| Inventory change by item (at cost) | PLYXIS | manual operations | difference |\n|---|---|---|---|")
        for n in sorted(set(inv["plyxis"]) | set(inv["legacy"])):
            a, b = inv["plyxis"].get(n, 0.0), inv["legacy"].get(n, 0.0)
            print(f"| {n} | {a:+,.0f} | {b:+,.0f} | {a - b:+,.0f} |")
    plans = d["plans"]
    print(f"\nPlans: {len(plans)}, feasible {sum(1 for x in plans if x['feasible'])}, solve time mean "
          f"{st.mean(x['seconds'] for x in plans):.1f} s, max {max(x['seconds'] for x in plans):.1f} s")
    for prm, v in d.get("calibration_error", {}).items():
        print(f"- {prm}: engineering model {v['engineering'] * 100:.1f} % -> live twin {v['twin'] * 100:.1f} %")
    print()


def decomposition(d) -> None:
    print(f"### Deterministic sections (native kernels available: {d['native']})\n")
    print("| Case study | decisions (discrete) | method | mean objective (USD) | vs GA | feasible | mean seconds |"
          "\n|---|---|---|---|---|---|---|")
    for scen, r in d["scenarios"].items():
        for m, v in r["summary"].items():
            print(f"| {scen} | {r['variables']:,} ({r['discrete']:,}) | {m} | {v['mean_objective']:,.0f} | "
                  f"{v.get('vs_ga_pct', 0):+.1f} % | {v['feasible']}/{v['runs']} | {v['mean_seconds']:.1f} |")
    print("\n| Case study | sections (largest) | colours | filter Spearman | filter forward | probing all couplings "
          "| section sweep native / NumPy |\n|---|---|---|---|---|---|---|")
    for scen, r in d["scenarios"].items():
        sx, k = r["sections"], r["section_sweep_s"]
        print(f"| {scen} | {sx['sections']} ({sx['largest_section']}) | {sx['colours']} | {sx['filter_spearman']:.2f} | "
              f"{r['filter_forward_ms']:.1f} ms | {r['probe_all_couplings_s']:.1f} s ({r['candidate_couplings']} pairs) | "
              f"{k['native']:.2f} s / {k['numpy']:.2f} s |")
    if d.get("scaling"):
        print("\n| Enterprise twin | decisions | discrete units | sections (largest) | colours | one exact section sweep "
              "| one simulation |\n|---|---|---|---|---|---|---|")
        for r in d["scaling"]:
            print(f"| {r['fields']} field(s) | {r['variables']:,} | {r['discrete_units']:,} | {r['sections']} "
                  f"({r['largest_section']}) | {r['colours']} | {r['sweep_s']:.1f} s | {r['simulate_s'] * 1000:.0f} ms |")
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


def budget(d) -> None:
    n = d["summary"][0]["cases"]
    print(f"### Solver budget ({n} sampled event situations, {d['variables']:,} variables, warm start = nominal plan)\n")
    print("| Configuration | relax / starts / discrete / repair / polish | mean objective (USD) | vs previous live "
          "| feasible | mean seconds |\n|---|---|---|---|---|---|")
    for s in d["summary"]:
        k = s["settings"]
        print(f"| {s['config']} | {k['relax_iters']} / {k['multistart']} / {k['population']}x{k['generations']} / "
              f"{k['repair_iters']} / {k['polish_iters']} | {s['mean_objective']:,.0f} | {s['vs_previous_pct']:+.1f} % | "
              f"{s['feasible']}/{s['cases']} | {s['mean_seconds']:.1f} |")
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
    for name, fn in (("showcase_run.json", showcase), ("showcase_run_corvus.json", mine),
                     ("decomposition.json", decomposition), ("neural_ablation.json", neural),
                     ("solver_comparison.json", solvers), ("solver_budget.json", budget), ("scaling.json", scaling),
                     ("twin_accuracy.json", accuracy)):
        d = _load(name)
        if d is not None:
            fn(d)
        if name == "showcase_run.json":
            variants()


if __name__ == "__main__":
    main()
