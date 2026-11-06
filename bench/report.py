"""Print markdown tables from bench/results/*.json (used to write docs/BENCHMARKS.md).

    python -m bench.report
"""
from __future__ import annotations

import json
import os
import re
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


def _branch_stats(d) -> str:
    stg = [s for r in d.get("runs", []) for s in r.get("stages", []) if s.get("stage") == "branches"]
    if not stg:
        return "no exact-branch stage"
    return (f"exact-branch stage in {len(stg)} plans, mean {st.mean(s['seconds'] for s in stg):.1f} s, "
            f"improved {sum(1 for s in stg if s.get('gain', 0) > 0)}")


def seed_summary(scenario: str) -> None:
    """Exact branches on vs. off over the seeds that were run (full event set only)."""
    # only the main runs and their branch arms: a tagged variant (e.g. the capped grade window) is another scenario
    base = "showcase_run" + ("" if scenario == "kestrel" else f"_{scenario}")
    main = re.compile(rf"^{base}(_branches|_nobranches)?(_s\d+)?\.json$")
    names = sorted(n for n in os.listdir(RES) if main.match(n))
    groups: dict = {}
    for n in names:
        d = _load(n)
        if not d or d.get("scenario", "kestrel") != scenario or d.get("skipped_events") or not d.get("neural", True) \
                and scenario == "kestrel":
            continue
        if "exact_branches" not in d:
            continue
        on = bool(d.get("exact_branches"))
        p, l = d["kpis"]["plyxis"], d["kpis"]["legacy"]
        groups.setdefault(d.get("seed", 0), {})[on] = (p["profit"] - l["profit"], (p["profit"] - l["profit"]) / abs(l["profit"]),
                                                      st.mean(x["seconds"] for x in d["plans"]))
    rows = [(s, g) for s, g in sorted(groups.items()) if True in g and False in g]
    if not rows:
        return
    print(f"### Exact branches on / off over seeds ({scenario})\n")
    print("| seed | with exact branches | without | effect | mean plan seconds (with / without) |\n|---|---|---|---|---|")
    eff = []
    for s, g in rows:
        (a, ar, at), (b, br, bt) = g[True], g[False]
        eff.append(a - b)
        print(f"| {s} | {a:+,.0f} ({100 * ar:+.1f} %) | {b:+,.0f} ({100 * br:+.1f} %) | {a - b:+,.0f} | {at:.1f} / {bt:.1f} |")
    print(f"\nEffect over {len(eff)} seeds: mean {st.mean(eff):+,.0f}, min {min(eff):+,.0f}, max {max(eff):+,.0f}\n")


def variants(scenario: str = "kestrel") -> None:
    """Full showcase vs. ablation runs (bench.showcase_run --skip-events / --no-branches ... --tag ...)."""
    names = sorted(n for n in os.listdir(RES) if n.startswith("showcase_run") and n.endswith(".json"))
    runs = [(n, d) for n in names if (d := _load(n)) and d.get("scenario", "kestrel") == scenario]
    if len(runs) < 2:
        return
    print(f"### Showcase variants ({scenario}, PLYXIS vs. manual operations)\n")
    print("| Run | exact branches | events dropped | operating profit | profit after carbon | oil | flaring |"
          "\n|---|---|---|---|---|---|---|")
    for n, d in runs:
        p, l = d["kpis"]["plyxis"], d["kpis"]["legacy"]

        def rel(key):
            a, b = p.get(key, 0.0), l.get(key, 0.0)
            return f"{a - b:+,.0f} ({(a - b) / abs(b) * 100:+.1f} %)" if abs(b) > 1e-9 else "-"
        print(f"| {n} | {', '.join(d.get('exact_branches', [])) or 'off'} | "
              f"{', '.join(d.get('skipped_events', [])) or 'none'} | {rel('profit')} | "
              f"{rel('profit_after_carbon')} | {rel('oil_bbl')} | {rel('flared_mscf')} |")
        print(f"|   | {_branch_stats(d)} | | manual ops USD {l.get('profit', 0):,.0f} | | | |")
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


def steel(d) -> None:
    """Showcase A/B of the Aquila steel group: KPIs, unit cost and intensity per t shipped, bridge, plant protection."""
    k = d["kpis"]
    p, l = k["plyxis"], k["legacy"]
    print(f"### Aquila steel group showcase ({d['hours']:.0f} h, seed {d.get('seed', 0)}, wall {d['wall_seconds']:.0f} s)\n")
    print("| KPI | PLYXIS | conventional planning | difference |\n|---|---|---|---|")
    rows = [("Operating profit (USD)", "profit"), ("Profit after carbon cost (USD)", "profit_after_carbon"),
            ("Revenue (USD)", "revenue"), ("Cash operating cost (USD)", "cost"),
            ("Inventory value change (USD)", "inventory_value"), ("Carbon cost at internal price (USD)", "carbon_cost")]
    rows += [(f"{key[2:]} ({'MWh' if key == 'm_power' else 't'})", key) for key in p if key.startswith("m_")]
    rows += [("Emissions (t CO2e)", "emission_kg")]
    for label, key in rows:
        f = 1e-3 if key == "emission_kg" else 1.0
        a_, b_ = p.get(key, 0) * f, l.get(key, 0) * f
        rel = f"{(a_ - b_) / abs(b_) * 100:+.1f} %" if abs(b_) > 1e-9 else ""
        print(f"| {label} | {a_:,.0f} | {b_:,.0f} | {a_ - b_:+,.0f} ({rel}) |")
    for w, lab in ((p, "PLYXIS"), (l, "conventional")):
        if w.get("m_shipped"):
            print(f"\n{lab}: cash cost USD {w['cost'] / w['m_shipped']:.1f} per t shipped, "
                  f"{w['emission_kg'] / w['m_shipped']:.0f} kg CO2e per t shipped, "
                  f"operating profit USD {w['profit'] / d['hours'] * 8760 / 1e9:.2f} bn/a")
    if "bridge" in d:
        print("\n| Where the operating-profit difference comes from | USD |\n|---|---|")
        for r in d["bridge"]:
            print(f"| {r['label']} | {r['delta']:+,.0f} |")
    plans = d["plans"]
    print(f"\nPlans: {len(plans)}, feasible {sum(1 for x in plans if x['feasible'])}, solve time mean "
          f"{st.mean(x['seconds'] for x in plans):.1f} s, max {max(x['seconds'] for x in plans):.1f} s")
    for w, v in d.get("protection", {}).items():
        print(f"- plant protection, {w}: {sum(v['cuts'].values())} cuts, underflow {v['underflow'] or 'none'}")
    for prm, v in d.get("calibration_error", {}).items():
        print(f"- {prm}: engineering model {v['engineering'] * 100:.1f} % -> live twin {v['twin'] * 100:.1f} %")
    print()


def steel_seeds() -> None:
    """The Aquila A/B over every seed that ran with the full scenario (no tag)."""
    rows = []
    for n in sorted(os.listdir(RES)):
        if not (n == "showcase_run_aquila.json" or (n.startswith("showcase_run_aquila_s") and n[21:-5].isdigit())):
            continue
        d = _load(n)
        p, l = d["kpis"]["plyxis"], d["kpis"]["legacy"]
        rows.append((d.get("seed", 0), p, l, st.mean(x["seconds"] for x in d["plans"]), len(d["plans"])))
    if not rows:
        return
    print("### Aquila over seeds (96 h, PLYXIS - conventional planning)\n")
    print("| seed | operating profit | after carbon | CO2e (t) | shipped | crude steel | plans, mean s |\n|---|---|---|---|---|---|---|")
    dp, dc = [], []
    for seed, p, l, secs, n in sorted(rows, key=lambda r: r[0]):
        rel = lambda k: (p[k] - l[k]) / abs(l[k]) * 100  # noqa: E731
        dp.append(p["profit"] - l["profit"])
        dc.append(p["profit_after_carbon"] - l["profit_after_carbon"])
        print(f"| {seed} | {dp[-1]:+,.0f} ({rel('profit'):+.2f} %) | {dc[-1]:+,.0f} | "
              f"{(p['emission_kg'] - l['emission_kg']) / 1e3:+,.0f} ({rel('emission_kg'):+.1f} %) | {rel('m_shipped'):+.2f} % | "
              f"{rel('m_crude'):+.2f} % | {n}, {secs:.0f} |")
    print(f"\nMean over {len(rows)} seeds: operating profit {st.mean(dp):+,.0f} (min {min(dp):+,.0f}, max {max(dp):+,.0f}), "
          f"after carbon {st.mean(dc):+,.0f} (min {min(dc):+,.0f}, max {max(dc):+,.0f})\n")


def policy_twin(d) -> None:
    print(f"### Plan vs. conventional rules in the twin's own world ({d['scenario']}, {d['hours']:.0f} h, nominal, no events)\n")
    print("| | operating profit (USD) | after carbon (USD) | CO2e (t) | seconds |\n|---|---|---|---|---|")
    for w in ("policy", "plan"):
        r = d[w]
        print(f"| {w} | {r['profit']:,.0f} | {r['profit_after_carbon']:,.0f} | {r['emission_t']:,.0f} | {r['seconds']:.0f} |")
    x = d["delta"]
    print(f"| difference | {x['profit']:+,.0f} | {x['profit_after_carbon']:+,.0f} | {x['emission_t']:+,.0f} | |\n")
    if "stocks" in d["plan"]:
        print("| Yards, change over the run | rules (t) | plan (t) | plan - rules (USD at unit cost) |\n|---|---|---|---|")
        for k in d["plan"]["stocks"]:
            a, b = d["policy"]["stocks"].get(k, {"t": 0.0, "usd": 0.0}), d["plan"]["stocks"][k]
            print(f"| {k} | {a['t']:+,.0f} | {b['t']:+,.0f} | {b['usd'] - a['usd']:+,.0f} |")
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


def section_theory(d) -> None:
    print("### Section DP vs. exact MILP (scheduling models with known structure)\n")
    print("| blocks | section DP | its certified gap | DP seconds | coordinate ascent | multi-start | MILP (HiGHS) best "
          "| MILP bound | MILP seconds | MILP optimal |\n|---|---|---|---|---|---|---|---|---|---|")
    for r in d["scaling"]:
        p, ml = r["dp"], r.get("milp", {})
        f = lambda v: "-" if v is None else f"{v:,.1f}"  # noqa: E731
        print(f"| {r['blocks']:,} | {p['value']:,.1f} | {100 * p['certified_gap']:.1f} % | {p['seconds']:.2f} | "
              f"{r['coordinate']['value']:,.1f} | {r['multistart']['value']:,.1f} | {f(ml.get('value'))} | "
              f"{f(ml.get('bound'))} | {ml.get('seconds', 0):.0f} | {ml.get('optimal', '-')} |")
    print("\n| weak-edge scale | section DP | certified gap | MILP best (120 s) | MILP bound |\n|---|---|---|---|---|")
    for r in d["coupling"]:
        p, ml = r["dp"], r.get("milp", {})
        print(f"| {r['weak']} | {p['value']:,.1f} | {100 * p['certified_gap']:.1f} % | {ml.get('value', float('nan')):,.1f} | "
              f"{(ml.get('bound') or float('nan')):,.1f} |")
    print()


def exact_branches(d) -> None:
    print("### Structure recognition and exact branches\n")
    print("| twin | decisions | LP | sequential LP (given discrete choices, frozen qualities) | convex (same, prices at their sign) | nonlinear |"
          "\n|---|---|---|---|---|---|")
    for r in d["runs"]:
        sh = r["structure"]["share_by_class"]
        print(f"| {r['scenario']} | {r['structure']['decisions']:,} | {100 * sh.get('lp', 0):.1f} % | "
              f"{100 * (sh.get('slp', 0) + sh.get('slp_quality', 0)):.1f} % | {100 * sh.get('convex', 0):.1f} % | {100 * sh.get('nonlinear', 0):.1f} % |")
    print("\n| twin | cascade plan (USD, F) | + LP branches | + sequential LP over the layer | "
          "+ polish for the same time |\n|---|---|---|---|---|")
    for r in d["runs"]:
        c, e, s, p = r["cascade"], r["exact_branches"], r["slp_layer"], r["slp_layer_polish_equal_time"]
        dv = lambda a: f"{a['objective'] - c['objective']:+,.0f} (F {a['F'] - c['F']:+.4f})"  # noqa: E731
        print(f"| {r['scenario']} | {c['objective']:,.0f} (F {c['F']:.4f}) | {dv(e)} in {e['seconds']:.1f} s, "
              f"{e['variables']} vars | {dv(s)} in {s['seconds']:.1f} s, {s['variables']} vars, {s['rows']:,} rows, "
              f"{s['iterations']} it (Jacobian {s['jacobian_seconds']:.1f} s, LP {s['lp_seconds']:.1f} s), "
              f"violation {s['max_violation']:.4f} | {dv(p)} in {p['seconds']:.1f} s, violation {p['max_violation']:.4f} |")
    print("\n| twin | full cascade | half the cascade's iterations + LP branches |\n|---|---|---|")
    for r in d["runs"]:
        h = r["half_cascade_plus_branches"]
        print(f"| {r['scenario']} | {r['cascade']['objective']:,.0f} in {r['cascade_seconds']:.1f} s | "
              f"{h['objective']:,.0f} in {h['seconds']:.1f} s |")
    print()
    rows = [r for r in d["runs"] if "slp_with_convex" in r]
    if rows:
        print("| twin | + sequential LP including the convex class | tangent gap left |\n|---|---|---|")
        for r in rows:
            c, v = r["cascade"], r["slp_with_convex"]
            print(f"| {r['scenario']} | {v['objective'] - c['objective']:+,.0f} (F {v['F'] - c['F']:+.4f}) in {v['seconds']:.1f} s, {v['variables']} vars, "
                  f"{v['rows']:,} rows, {v['iterations']} it (Jacobian {v['jacobian_seconds']:.1f} s, LP {v['lp_seconds']:.1f} s), violation {v['max_violation']:.4f} | "
                  f"USD {v['tangent_gap_usd']:,.0f} |")
        print()


def backtest(d) -> None:
    print(f"### Backtest, gate G1 ({d['scenario']}: calibrated on {d['train_h']:.0f} h, frozen, predicting {d['test_h']:.0f} h)\n")
    print("| Signal | measure | value | limit | result | worst tags |\n|---|---|---|---|---|---|")
    for r in d["signals"]:
        worst = ", ".join(f"{w['tag']} {w['mape']:.1f}" for w in r.get("worst", [])[:2])
        print(f"| {r['signal']} | {r['measure']} | {r['value']:.2f} | {r['limit']:.0f} | {'pass' if r['pass'] else 'FAIL'} | {worst} |")
    print(f"\nAssets discovered from tags: {', '.join(d['discovered_assets']) or 'none'}; unmapped tags left: {len(d['tags_unmapped'])}; "
          f"gate {'passed' if d['gate_passed'] else 'not passed'}.\n")


def risk(d) -> None:
    print(f"### Outcome distributions: sigma points vs Monte Carlo (property sigma {d['property_sigma']}, "
          f"price sigma {d['price_sigma_per_sqrt_h']} per sqrt h)\n")
    print("| twin | inputs | factors | limits: spread ratio median (p10-p90) | P(violation) error | plan value: reference std | "
          "sigma points only | + 128 Sobol: error of mean / std | Monte Carlo, same total: error of mean / std |\n|---|---|---|---|---|---|---|---|---|")
    for r in d.get("runs", []):
        for name in ("properties", "properties_and_prices"):
            x = r.get(name)
            if not x:
                continue
            L, V, R = x["limits"], x["value"], x["reference"]
            print(f"| {r['scenario']} | {name.replace('_', ' ')} | {L['factors']} | {L['std_ratio_median']:.3f} ({L['std_ratio_p10']:.3f}-{L['std_ratio_p90']:.3f}) | "
                  f"{L['p_violation_abs_error_mean']:.3f} | {R['std']:,.0f} | {V['sigma_points_only']['std']['estimate']:,.0f} | "
                  f"{V['sigma_points_and_sobol']['mean']['rms_error']:,.0f} / {V['sigma_points_and_sobol']['std']['rms_error']:,.0f} | "
                  f"{V['monte_carlo_same_cost']['mean']['rms_error']:,.0f} / {V['monte_carlo_same_cost']['std']['rms_error']:,.0f} |")
    for f in d.get("frontier", []):
        print(f"\n| {f['scenario']}: cap on the value's std | expected value | std | 5 % quantile | max violation |\n|---|---|---|---|---|")
        for p in f["points"]:
            print(f"| {p['cap_fraction']:.2f} x uncapped | {p['mean']:,.0f} | {p['std']:,.0f} | {p['q05']:,.0f} | {p['max_violation']:.4f} |")
    print()


def coverage(d) -> None:
    print(f"### Posterior coverage ({d['scenario']}, {d['hours']:.0f} h after {d['warmup']:.0f} h warm-up)\n")
    print("| | within 1 sd (honest: 0.68) | within 2 sd (0.95) | z rms (1.0) | n |\n|---|---|---|---|---|")
    for k in ("posterior", "prior"):
        s = d[k]
        print(f"| {'posterior covariance' if k == 'posterior' else 'fixed engineering sigma 0.15'} | {s['within_1sd']:.2f} | "
              f"{s['within_2sd']:.2f} | {s['z_rms']:.2f} | {s['n']} |")
    print("\n| property | posterior 1 sd / 2 sd / z rms | mean posterior sd | rms log error |\n|---|---|---|---|")
    for k, v in d["by_property"].items():
        p = v["posterior"]
        print(f"| {k} | {p['within_1sd']:.2f} / {p['within_2sd']:.2f} / {p['z_rms']:.2f} | {v['mean_sd']:.3f} | {v['rms_log_error']:.3f} |")
    print()


def colgen(d) -> None:
    print("### Column generation vs exact MILP (wells-like units, one gas limit per hour, 24 h)\n")
    print("| assets | decisions | CG seconds | rounds | gap to proven bound | MILP seconds | MILP | CG below MILP |\n|---|---|---|---|---|---|---|---|")
    for r in d.get("runs", []):
        c, m = r["cg"], r.get("milp")
        ms = f"{m['seconds']:.1f} | {'optimal' if m['optimal'] else 'time limit'} | {100 * c.get('below_milp', float('nan')):.2f} %" if m else "- | - | -"
        print(f"| {r['assets']} | {r['decisions']:,} | {c['seconds']:.2f} | {c['rounds']} | {100 * c['gap']:.3f} % | {ms} |")
    lp = d.get("learned_prices")
    if lp:
        print(f"\nLearned prices ({lp['test']} test situations, {lp['assets']} assets, trained on {lp['train']}):\n")
        print("| start | rounds | seconds | gap |\n|---|---|---|---|")
        for k in ("cold", "learned", "exact_prices"):
            v = lp["mean"][k]
            print(f"| {k.replace('_', ' ')} | {v['rounds']:.1f} | {v['seconds']:.2f} | {100 * v['gap']:.3f} % |")
    print()


def corporate(d) -> None:
    """One twin, several time scales: the steel group hourly, the Duisburg works' plants found by the detector at their
    own step, coordinated by prices (bench.corporate)."""
    pr = d["programs"]
    blocks = [k for k in pr if k != "master"]
    print(f"### One twin, plants found in it: the steel group hourly, the Duisburg works' plants at their own step "
          f"(aquila_works, split {d.get('split', 'auto')})\n")
    for line in d.get("detected", []):
        if not line.startswith("  "):
            print(f"- {line}")
    ref = d.get("reference", {})
    print("\n| | " + " | ".join(["group (master)"] + blocks + ["whole twin, one program (reference)"]) + " |")
    print("|---|" + "---|" * (len(blocks) + 2))
    print("| instances | " + " | ".join([str(pr["master"]["instances"])] + [str(pr[b]["instances"]) for b in blocks]
                                         + ["-"]) + " |")
    print("| steps | " + " | ".join([f"{pr[k]['steps']} x {pr[k]['step_h'] * 60:g} min" for k in ["master"] + blocks]
                                     + [f"{ref.get('steps', '-')}"]) + " |")
    dec = d["decisions"]
    print("| decisions (discrete) | " + " | ".join([f"{dec['master']:,}"] + [f"{dec[b]:,} ({dec.get(b + '_discrete', 0):,})"
                                                                          for b in blocks]
                                                 + [f"{ref.get('decisions', 0):,} ({ref.get('discrete', 0):,})"]) + " |")
    print("| crossings | - | " + " | ".join(str(len(pr[b].get("crossings", []))) for b in blocks) + " | - |")
    pl = d["plan"]
    if "plant_violation" in pl:
        print("\nCommitted schedules' own limits (max violation): "
              + ", ".join(f"{k} {v:.4f}" for k, v in pl["plant_violation"].items())
              + f"; the group's program feasible {pl.get('master_feasible')}.")
    print(f"\nCommitted plan USD {pl['objective']:,.0f}, feasible {pl['feasible']}, {d['seconds']:.0f} s (all rounds); "
          f"reference USD {ref.get('objective', float('nan')):,.0f}, feasible {ref.get('feasible')} (max violation "
          f"{ref.get('max_violation', 0):.4f}), {ref.get('seconds', 0):.0f} s.\n")
    print("| round | block schedules (block: internal value USD M / reduced cost USD M / max violation / seconds) | master (USD M, feasible, seconds) |\n|---|---|---|")
    for r in d["rounds"]:
        parts = []
        for b, cols in r.get("plants", {}).items():
            for c in (cols if isinstance(cols, list) else []):
                rc = "-" if c["reduced_cost"] is None else f"{c['reduced_cost'] / 1e6:,.3f}"
                parts.append(f"{b}: {c['internal'] / 1e6:,.2f} / {rc} / {c['max_violation']:.4f} / {c['seconds']:.0f}")
        m = r.get("master")
        ms = f"{m['objective'] / 1e6:,.2f}, {m['feasible']}, {m['seconds']:.0f}" if m else ""
        print(f"| {r['round']} | {'; '.join(parts)} | {ms} |")
    for b, p in d.get("plants", {}).items():
        print(f"\n{b}: committed schedule max violation {p['max_violation']:.4f}; flows at the fence per hour (the "
              f"schedule / the master's mix): " + ", ".join(f"{k} {v:,.0f} / {p['mix_flows_per_h'][k]:,.0f}"
                                                            for k, v in p["flows_per_h"].items()))
        for path, dec_ in sorted(p["schedule"].items()):
            for name, v in dec_.items():
                if "events" in v:
                    print(f"- {path}.{name}: {v['events']} events, {v['starts']} starts, on {100 * v['on_share']:.0f} % of the time")
        if p.get("signals"):
            print("- means: " + ", ".join(f"{k} {v:,.1f}" for k, v in p["signals"].items()))
    if d.get("master_signals"):
        print("- group (master) means: " + ", ".join(f"{k} {v:,.1f}" for k, v in d["master_signals"].items()))
    if ref.get("signals"):
        print("- reference means: " + ", ".join(f"{k} {v:,.1f}" for k, v in ref["signals"].items()))
    t = d.get("toy")
    if t:
        print(f"\nToy (examples/plants_toy.plx, optimum {t['optimum']:,.0f}): whole model solved at once {t['whole']['objective']:,.0f} "
              f"({t['whole']['seconds']:.1f} s), decomposed {t['decomposed']['objective']:,.0f} committed, feasible "
              f"{t['decomposed']['feasible']} ({t['decomposed']['seconds']:.1f} s)")
    print()


def corporate_regions(d) -> None:
    """The steel group hourly: the whole twin against its regional networks coordinated by prices."""
    print("### Regional networks found in the steel group (aquila, hourly)\n")
    for line in d.get("detected", []):
        if not line.startswith("  "):
            print(f"- {line}")
    dec = d["decisions"]
    w, r = d["whole"], d["regions"]
    print(f"\nDecisions: whole {dec['whole']:,}; master {dec['master']:,} (column weights); blocks "
          + ", ".join(f"{k} {v:,}" for k, v in dec.items() if k not in ("whole", "master")) + ".\n")
    print("| | whole twin, one program | regional networks, coordinated by prices |\n|---|---|---|")
    print(f"| plan value (USD) | {w['objective']:,.0f} (feasible {w['feasible']}) | {r['objective']:,.0f} committed "
          f"(feasible {r['feasible']}; mix {r['mix_objective']:,.0f}) |")
    print(f"| seconds | {w['seconds']:.0f} | {r['seconds']:.0f} ({d.get('workers', 0)} workers) |")
    for rr in d.get("rounds", []):
        m = rr.get("master")
        rcs = [c["reduced_cost"] for cols in rr.get("plants", {}).values() for c in cols if c.get("reduced_cost") is not None]
        if m or rcs:
            print(f"- round {rr['round']}: " + (f"max reduced cost USD {max(rcs) / 1e6:,.3f} M; " if rcs else "")
                  + (f"master USD {m['objective'] / 1e6:,.2f} M, feasible {m['feasible']}, {m['seconds']:.0f} s" if m else ""))
    print()


def reduction(d) -> None:
    """The exact layer: reduction engine vs whole-model MILP as the kiln group grows; filters on the twins; SLP."""
    print("### Reduction engine vs the whole-model MILP (N kiln sites, one market)\n")
    print("| sites | binaries (decisions + indicators) | whole MILP s | whole F | engine s | engine F | bound | gap to bound | rounds | blocks |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for c in d.get("scaling", []):
        w, e = c["whole"], c["engine"]
        print(f"| {c['sites']} | {c['model']['discrete']:,} | {w['seconds']:.1f} ({w['status']}) | {w['F']:.5f} | "
              f"{e['seconds']:.1f} | {e['F']:.5f} | {e['bound']:.5f} | {e['gap']:.2%} | {e['rounds']} | {e['blocks']} |")
    st = d.get("structure", {})
    if st:
        print("\n| twin | linearise s | rows | nonzeros | filters s | blocks | largest block (integers) | master columns |")
        print("|---|---|---|---|---|---|---|---|")
        for k, v in st.items():
            big = max((b["ints"] for b in v["blocks"]), default=0)
            print(f"| {k} | {v['linearize_s']:.2f} | {v['model']['rows']:,} | {v['model']['nonzeros']:,} | {v['contract_s']:.2f} | "
                  f"{len(v['blocks'])} | {big} | {v['master_columns']:,} |")
    for r in d.get("slp_vs_cascade", []):
        print(f"\n{r['twin']}: start F {r['start']['F']:.4f}; sequential LP F {r['slp']['F']:.5f} (USD {r['slp']['objective']:,.0f}, "
              f"violation {r['slp']['violation']:.2g}) in {r['slp']['seconds']:.0f} s; cascade F {r['cascade']['F']:.5f} "
              f"(USD {r['cascade']['objective']:,.0f}, violation {r['cascade']['violation']:.2g}) in {r['cascade']['seconds']:.0f} s")
    print()


def plan_quality(d) -> None:
    """One steel plan: the cascade's against a better stored one, restarts, the exact polish, where a step goes wrong."""
    print("### One steel-group plan: how good is it (plan_quality_aquila.json)\n")
    print("| plan | value (USD) | shipped (t) | spot (t) |\n|---|---|---|---|")
    for name, r in d.get("stored", {}).items():
        print(f"| stored: {name} | {r['objective']:,.0f} | {r['shipped_kg'] / 1e3:,.0f} | {r['spot_kg'] / 1e3:,.0f} |")
    for r in d.get("cold", []):
        print(f"| cold cascade, {r['multistart']} restarts, seed {r['seed']} | {r['objective']:,.0f} | "
              f"{r['shipped_kg'] / 1e3:,.0f} | {r['spot_kg'] / 1e3:,.0f} |")
    for r in d.get("polish") or []:
        print(f"| sequential LP from the cascade's plan, radius {r['radius']} (floor {r['min_radius']}), {r['iterations']} it | "
              f"{r['objective']:,.0f} | {r['shipped_kg'] / 1e3:,.0f} | {r['spot_kg'] / 1e3:,.0f} |")
    st = d.get("step")
    if st:
        print(f"\nOne LP step of {st['radius']} from the cascade's plan: twin USD {st['objective_before']:,.0f} -> "
              f"{st['objective_after']:,.0f}; broken: {st['broken_limits']}")
        print("\n| term | model change | twin change |\n|---|---|---|")
        for t in st["terms"][:8]:
            print(f"| {t['term']} ({t['kind']}) | {t['model']:+,.0f} | {t['twin']:+,.0f} |")
    print()


def console_run(d) -> None:
    """The A/B as the console runs it."""
    k = d["kpis"]["delta"]
    plans = sorted({h["plan"] for h in d.get("hourly", []) if h.get("plan") is not None})
    print(f"### Console run ({d['scenario']}, {d['speed']:.0f}x, seed {d['seed']}): operating {k['profit']:+,.0f} USD, after carbon "
          f"{k.get('profit_after_carbon', k['profit']):+,.0f} USD; the plant executed {len(plans)} plans\n")


def planner_speed(d) -> None:
    """Plan time and value per twin: the morning's planner, the current one, a quarter budget with the exact polish."""
    print(f"### Planner speed ({d.get('threads')} threads, default problem, one plan each)\n")
    print("| twin | configuration | cold s | cold value (USD) | warm s | warm value (USD) |\n|---|---|---|---|---|---|")
    for sc in ("kestrel", "corvus", "aquila"):
        if sc not in d:
            continue
        for name, rows in d[sc]["configs"].items():
            for r in rows:
                c, w = r["cold"], r["warm"]
                print(f"| {sc} | {name} | {c['seconds']:.1f} | {c['objective']:,.0f} | {w['seconds']:.1f} | {w['objective']:,.0f} |")
    print()


def main() -> None:
    for name, fn in (("risk.json", risk), ("uncertainty_coverage.json", coverage), ("uncertainty_coverage_matching.json", coverage),
                     ("colgen.json", colgen), ("reduction.json", reduction), ("plan_quality_aquila.json", plan_quality),
                     ("console_run_aquila.json", console_run), ("planner_speed.json", planner_speed),
                     ("showcase_run.json", showcase), ("showcase_run_corvus.json", mine),
                     ("showcase_run_aquila.json", steel), ("policy_in_twin_aquila.json", policy_twin),
                     ("corporate.json", corporate), ("corporate_regions.json", corporate_regions),
                     ("backtest_kestrel.json", backtest), ("backtest_corvus.json", backtest),
                     ("decomposition.json", decomposition), ("section_theory.json", section_theory),
                     ("exact_branches.json", exact_branches), ("neural_ablation.json", neural),
                     ("solver_comparison.json", solvers), ("solver_budget.json", budget), ("scaling.json", scaling),
                     ("twin_accuracy.json", accuracy)):
        d = _load(name)
        if d is not None:
            fn(d)
        if name == "showcase_run.json":
            variants("kestrel")
            seed_summary("kestrel")
        if name == "showcase_run_corvus.json":
            variants("corvus")
            seed_summary("corvus")
        if name == "showcase_run_aquila.json":
            steel_seeds()


if __name__ == "__main__":
    main()
