"""Backtest (pilot gate G1, docs/PILOT_PLAN_OIL_GAS.md): can a twin calibrated on history predict unseen history?

Two separate steps, as in a pilot:

    export    the plant runs under manual operations and writes a raw historian export (data.historian)
    replay    a fresh twin sees only that file: it is calibrated on the training part, then frozen; on the unseen
              part it predicts every signal one step ahead and every level 3 h ahead from the state 3 h earlier

    python -m bench.backtest --scenario kestrel
    python -m bench.backtest --scenario corvus

The gate's criteria are the pilot plan's for the oil field; for the mine the same limits are applied to the
corresponding signals (no pilot plan exists for a mine yet). Writes bench/results/backtest_<scenario>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from plyxis.data.historian import HistoryWriter, read_history  # noqa: E402
from plyxis.lang import units as U  # noqa: E402
from plyxis.runtime.engine import Engine  # noqa: E402
from plyxis.runtime.scenario import Scenario  # noqa: E402
from plyxis.runtime.twin_manager import TwinManager  # noqa: E402

GATES = {
    "kestrel": {"train_h": 36.0, "test_h": 36.0,
                "signals": [("Well liquid rates", ".FLOW.LIQ", 8.0), ("Compression suction gas", ".SUC.FLOW", 5.0),
                            ("Battery separator gas", ".SEP.GAS", 5.0), ("Disposal injection pressure", ".INJ.PRESS", 10.0)],
                "levels": [("Oil tank level, 3 h ahead", "oil_lvl", "oil_tank", 10.0),
                           ("Water tank level, 3 h ahead", "wat_lvl", "wat_tank", 10.0)]},
    "corvus": {"train_h": 24.0, "test_h": 24.0,
               "signals": [("Stope mucking rates", ".LHD.TPH", 8.0), ("Crusher throughput", ".CR.TPH", 5.0),
                           ("Hoist throughput", ".HOIST.TPH", 5.0), ("Head grade (on-stream analyser)", ".OSA.HEAD", 5.0)],
                "levels": [("Sump level, 3 h ahead", "level", "sump", 10.0), ("Shaft bin level, 3 h ahead", "bin", "bin_cap", 10.0),
                           ("ROM pad level, 3 h ahead", "tonnes", "cap", 10.0), ("Ore pass level, 3 h ahead", "stock", "cap", 10.0)]},
}


def export_history(sdir: str, hours: float, path: str, seed: int = 0) -> dict:
    """Run the plant under manual operations and write what its systems would export. Nothing else leaves this."""
    e = Engine(sdir, seed=seed, sync_plans=True, use_neural=False, online_learning=False)
    e._replan = lambda trigger: None
    e.last_plan_t = 1e12
    e.last_calib = 1e12
    e.advisory.mode = "shadow"
    w = HistoryWriter(path)
    push = e.scada.push

    def capture(msgs):
        w.telemetry(msgs)
        push(msgs)
    e.scada.push = capture
    env = sorted(e.twin.model.program.env_inputs)
    pp = e.plants["plyxis"]
    end = e.t + hours
    n = 0
    while e.t < end - 1e-9:
        t = e.t
        w.prices(t + e.sc.dt_emu, {k: e.prices.canonical(k, t + e.sc.dt_emu) for k in env})
        e.step()
        vals, routes = pp.readback()
        w.controls(t + e.sc.dt_emu, vals, routes)
        n += 1
    w.close()
    return {"steps": n, "bytes": os.path.getsize(path)}


def _values(buf_rec, gid: str, state: str, li: int) -> float:
    return float(buf_rec["S"][gid][state].reshape(-1)[li])


def replay(sdir: str, path: str, train_h: float, gate: dict, progress=None) -> dict:
    """A fresh twin on the file alone."""
    sc = Scenario.load(sdir)
    twin = TwinManager(sc.twin_files, sc.twin_env, sc.twin_sim, dt_data=sc.dt_emu, test_sigma=(sc.raw.get("assays") or {}).get("sigma"))
    t_first = None
    sums: dict = {name: [0.0, 0] for name, _, _ in gate["signals"]}
    per_tag: dict = {}
    peak: dict = {}
    levels: dict = {name: [0.0, 0] for name, *_ in gate["levels"]}
    last_cal, discovered, unmapped = None, [], set()
    for step in read_history(path):
        t = step.t
        t_first = t if t_first is None else t_first
        last_cal = t if last_cal is None else last_cal
        testing = t - t_first >= train_h - 1e-9
        for b in step.births:
            twin.discovery.note_birth(b.prefix, b.meta)
        tele = []
        for m in step.telemetry:
            if twin.tags.get(m.tag) is None:
                prop = twin.discovery.observe(m.tag, t) if m.quality != "test" else None
                unmapped.add(m.tag)
                if prop is not None and twin.apply_discovery(prop, t).get("ok"):
                    discovered.extend(prop.assets)
                continue
            if testing and m.quality == "test":        # frozen: no new direct samples either
                continue
            tele.append(m)
        est = twin.estimator
        D, R = twin.live.controls_from_paths(step.controls, step.routes)
        E_now = {k: torch.tensor([step.prices.get(k, float(twin.model.program.E0[k]))]) for k in twin.model.program.env_inputs}
        est.ingest(t, tele, D, R, E_now)
        est.propagate()
        if not testing:
            if t - last_cal >= sc.calibrate_every - 1e-9:
                last_cal = t
                est.calibrate(iters=25)
            continue
        preds = est.predict_last()
        for m in tele:
            info = twin.tags.get(m.tag)
            if info is None or not info.key.startswith("node:"):
                continue
            p = preds.get((info.gid, info.li, info.key))
            if p is None:
                continue
            pv = U.from_canonical(p, info.unit) if info.unit else p
            for name, suffix, _ in gate["signals"]:
                if m.tag.endswith(suffix):
                    peak[name] = max(peak.get(name, 0.0), abs(m.value))
                    if abs(m.value) < max(1e-6, 0.02 * peak[name]):
                        continue                         # stopped equipment has no relative error: noise around zero
                    sums[name][0] += abs(pv - m.value) / abs(m.value)
                    sums[name][1] += 1
                    row = per_tag.setdefault((name, m.tag), [0.0, 0])
                    row[0] += abs(pv - m.value) / abs(m.value)
                    row[1] += 1
        buf = list(est.live.buf)
        if abs(round(t) - t) < 1e-6 and len(buf) >= est.live.buf.maxlen:   # hourly, from the state 3 h earlier
            Dw, Rw, Ew, S0, W = est._window(buf)
            phys = est.live.phys
            with torch.no_grad():
                fin = phys.final_states(phys.simulate(Dw, Rw, est.live.P, S0, Ew, T=W, dt=est.dt, t0=buf[1]["t"] - est.dt))
            for name, state, cap, _ in gate["levels"]:
                for g in phys.groups:
                    if state not in g.ir.states or cap not in est.live.P.get(g.gid, {}) or (g.gid, f"state:{state}") not in est.measured_states:
                        continue
                    pred = fin[g.gid][state].reshape(-1)
                    meas = buf[-1]["S"][g.gid][state].reshape(-1)
                    span = est.live.P[g.gid][cap].reshape(-1)
                    span = span.expand_as(pred) if span.numel() == 1 else span
                    levels[name][0] += float(((pred - meas).abs() / span.clamp(min=1e-6)).sum())
                    levels[name][1] += pred.numel()
        if progress:
            progress(t)
    rows = []
    for name, _, limit in gate["signals"]:
        tot, n = sums[name]
        v = 100.0 * tot / max(n, 1)
        worst = sorted(((round(100.0 * a / max(b, 1), 2), tag) for (nm, tag), (a, b) in per_tag.items() if nm == name), reverse=True)[:3]
        rows.append({"signal": name, "measure": "MAPE, 1 step ahead (%)", "value": round(v, 2), "limit": limit, "n": n,
                     "pass": n > 0 and v <= limit, "worst": [{"tag": tag, "mape": a} for a, tag in worst]})
    for name, _, _, limit in gate["levels"]:
        tot, n = levels[name]
        v = 100.0 * tot / max(n, 1)
        rows.append({"signal": name, "measure": "MAE, % of capacity", "value": round(v, 2), "limit": limit, "n": n, "pass": n > 0 and v <= limit})
    known = set(twin.tags.by_tag)
    still_unmapped = sorted(t for t in unmapped if t not in known)
    return {"signals": rows, "discovered_assets": discovered, "tags_mapped": len(known), "tags_unmapped": still_unmapped[:20],
            "all_assets_represented": not still_unmapped, "gate_passed": all(r["pass"] for r in rows) and not still_unmapped,
            "twin_version": twin.version}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="kestrel", choices=sorted(GATES))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    gate = GATES[args.scenario]
    sdir = os.path.join(ROOT, "scenarios", args.scenario)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "history.csv.gz")
        ex = export_history(sdir, gate["train_h"] + gate["test_h"], path, args.seed)
        t1 = time.perf_counter()
        print(f"export: {ex['steps']} steps, {ex['bytes'] / 1e6:.1f} MB compressed, {t1 - t0:.0f} s", flush=True)
        res = replay(sdir, path, gate["train_h"], gate)
    res.update(scenario=args.scenario, seed=args.seed, train_h=gate["train_h"], test_h=gate["test_h"], export=ex,
               wall_seconds=round(time.perf_counter() - t0, 1))
    os.makedirs(os.path.join(ROOT, "bench", "results"), exist_ok=True)
    with open(os.path.join(ROOT, "bench", "results", f"backtest_{args.scenario}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    for r in res["signals"]:
        print(f"{r['signal']:42s} {r['value']:7.2f}  limit {r['limit']:5.1f}  n={r['n']:6d}  {'pass' if r['pass'] else 'FAIL'}  {r.get('worst', '')}")
    print("assets discovered:", res["discovered_assets"], "| unmapped tags:", res["tags_unmapped"], "| gate", "PASSED" if res["gate_passed"] else "NOT PASSED")


if __name__ == "__main__":
    main()
