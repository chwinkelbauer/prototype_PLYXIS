"""Shift handover report: what happened, what the plan does next, what is open. Deterministic templates only."""
from __future__ import annotations

import csv
import io
from typing import Optional


def _usd(v: float) -> str:
    s = "-" if v < 0 else ""
    a = abs(v)
    return f"{s}USD {a / 1e6:,.2f} M" if a >= 1e6 else f"{s}USD {a:,.0f}"


def shift_report(e, hours: float = 12.0) -> dict:
    """Facts for the handover of the shift ending now. Every number comes from the engine's own accounts."""
    t1, t0 = e.t, max(e.sc.start_hour, e.t - hours)
    k = e.kpis()
    pc, lc = k["plyxis"]["cum"], k["legacy"]["cum"]
    hist = [p for p in e.kpi_history if p["t"] >= t0]
    first = hist[0] if hist else {"pc": 0.0, "lc": 0.0}
    shift_p, shift_l = pc["profit"] - first["pc"], lc["profit"] - first["lc"]
    metrics = []
    for m in e.sc.raw.get("kpi", {}).get("metrics", []) or []:
        key = f"m_{m['key']}"
        if key in pc:
            metrics.append({"label": m["label"], "unit": m["unit"], "plyxis": pc[key], "manual": lc.get(key, 0.0),
                            "better": m.get("better", "higher")})
    if not metrics:
        for key, label, unit, better in (("oil_bbl", "Oil", "bbl", "higher"), ("gas_mscf", "Gas", "Mscf", "higher"),
                                         ("water_bbl", "Water", "bbl", "lower"), ("flared_mscf", "Flared", "Mscf", "lower")):
            if key in pc:
                metrics.append({"label": label, "unit": unit, "plyxis": pc[key], "manual": lc.get(key, 0.0), "better": better})
    events = [{"clock": r["clock"], "kind": r["kind"], "title": r["title"], "text": r.get("text", "")}
              for r in e.timeline if r["t"] >= t0 and r["kind"] in ("event", "failure", "market", "maintenance", "twin")]
    alerts = [a for a in list(e.alerts) if a.get("t", 0.0) >= t0]      # the plant thread adds while the server reads
    sev = {s: sum(1 for a in alerts if a.get("severity") == s) for s in ("critical", "warning", "info")}
    plan = e.planner.plan
    pl = None
    if plan is not None:
        pl = {"id": plan.pid, "made": e.clock_str(plan.t0), "trigger": plan.trigger, "feasible": plan.feasible,
              "objective": plan.objective, "decisions": plan.solver.get("variables"), "seconds": plan.solver.get("seconds"),
              "valid": e.advisory.plan_valid(e.t),
              "bottlenecks": [{"label": b.get("label"), "hours": b.get("hours_binding"),
                               "value_10pct_per_day": b.get("value_10pct_per_day")}
                              for b in (plan.bottlenecks or [])[:6]]}
    plans = [p for p in list(e.plan_events) if p["t"] >= t0]
    view = e.advisory.view()
    decided = [r for r in view["recent_decisions"] if r.get("decided_at", -1) >= t0]
    jobs = [{**j["job"], "status": j["status"]} for j in list(e.jobs.values()) if j["status"] in ("scheduled", "active")]
    commits = [c for c in e.twin.repo.log(50) if (c.get("t") or 0.0) >= t0]
    return {"site": e.sc.raw.get("title", e.sc.name), "from": e.clock_str(t0), "to": e.clock_str(t1), "hours": t1 - t0,
            "mode": e.advisory.mode, "twin_version": e.twin.version,
            "profit": {"shift_plyxis": shift_p, "shift_manual": shift_l, "cum_plyxis": pc["profit"], "cum_manual": lc["profit"]},
            "metrics": metrics, "events": events, "alerts": sev,
            "top_alerts": [{"clock": a.get("clock", ""), "severity": a.get("severity"), "message": a.get("message")}
                           for a in alerts if a.get("severity") in ("critical", "warning")][-8:],
            "plan": pl, "plans_in_shift": len(plans),
            "recommendations": {"open": sum(1 for r in view["recommendations"] if r["status"] == "pending"),
                                "next": [{"label": r["label"], "decision": r["decision"], "from": r["from_display"],
                                          "to": r["to_display"], "status": r["status"]} for r in view["recommendations"][:10]]},
            "decisions": [{"label": r["label"], "decision": r["decision"], "status": r["status"], "by": r.get("decided_by"),
                           "reason": r.get("reason", "")} for r in decided],
            "overrides": view["overrides"], "jobs": jobs, "commits": [{"id": c["id"], "message": c["message"]} for c in commits],
            "data_quality": e.quality.summary()}


def work_orders_csv(e) -> str:
    """Scheduled and running jobs as a flat file for an import into SAP PM or Maximo (their field mapping is the site's)."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["id", "asset", "description", "kind", "resource", "status", "start_h", "end_h", "travel_h", "duration_h", "value_usd_per_h", "reason"])
    for rec in list(e.jobs.values()):
        j = rec["job"]
        w.writerow([j["id"], j["asset"], j["label"], j["kind"], j["resource_label"], rec["status"], round(j["start"], 2), round(j["end"], 2),
                    round(j.get("travel_h", 0.0), 2), round(j.get("hours", 0.0), 2), round(j.get("value_per_h", 0.0), 2), j.get("reason", "")])
    return buf.getvalue()


def to_markdown(r: dict, delta_label: Optional[str] = "manual operations") -> str:
    out = [f"# Shift report: {r['site']}", "", f"{r['from']} to {r['to']} ({r['hours']:.0f} h). Mode: {r['mode']}. "
           f"Twin version {r['twin_version']}.", ""]
    p = r["profit"]
    out += ["## Result", "", f"Operating profit this shift: {_usd(p['shift_plyxis'])} ({delta_label}: {_usd(p['shift_manual'])}). "
            f"Since the start: {_usd(p['cum_plyxis'])} ({delta_label}: {_usd(p['cum_manual'])}).", ""]
    if r["metrics"]:
        out += ["| Since the start | PLYXIS | " + str(delta_label) + " |", "|---|---:|---:|"]
        out += [f"| {m['label']} ({m['unit']}) | {m['plyxis']:,.0f} | {m['manual']:,.0f} |" for m in r["metrics"]]
        out.append("")
    out += ["## What happened", ""]
    out += [f"- {ev['clock']}: {ev['title']}" for ev in r["events"]] or ["- No scripted events, failures or twin changes in this shift."]
    a = r["alerts"]
    out += ["", f"Alerts: {a['critical']} critical, {a['warning']} warnings, {a['info']} information.", ""]
    out += [f"- {x['clock']} [{x['severity']}] {x['message']}" for x in r["top_alerts"]]
    if r["top_alerts"]:
        out.append("")
    pl = r["plan"]
    out += ["## The current plan", ""]
    if pl is None:
        out += ["No plan yet.", ""]
    else:
        valid = "" if pl["valid"] else " It has expired: no recommendation, the plant holds its last set points."
        out += [f"Plan #{pl['id']}, made {pl['made']} ({pl['trigger']}): {'feasible' if pl['feasible'] else 'NOT feasible'}, "
                f"plan value {_usd(pl['objective'])} over the horizon, {pl['decisions']:,} decisions in {pl['seconds']:.0f} s. "
                f"{r['plans_in_shift']} plan{'' if r['plans_in_shift'] == 1 else 's'} in this shift.{valid}", ""]
        if pl["bottlenecks"]:
            out += ["What limits the plan (from the shadow prices; value of 10 % more capacity per day):", ""]
            out += [f"- {b['label']}: binding {b['hours']:g} h, {_usd(b['value_10pct_per_day'])} per day" for b in pl["bottlenecks"]]
            out.append("")
    rec = r["recommendations"]
    out += ["## Next shift", "", f"{rec['open']} recommendations wait for a decision." if r["mode"] == "advisory" else
            "Largest planned changes in the next shift:", ""]
    out += [f"- {x['label']}, {x['decision']}: {x['from']} to {x['to']} ({x['status']})" for x in rec["next"]] or ["- No material change planned."]
    out.append("")
    if r["decisions"]:
        out += ["## Operator decisions", ""]
        out += [f"- {d['label']}, {d['decision']}: {d['status']} by {d['by']}" + (f" ({d['reason']})" if d["reason"] else "")
                for d in r["decisions"]]
        out.append("")
    if r["jobs"]:
        out += ["## Work orders", ""]
        out += [f"- {j['label']}: {j['status']}, {j['resource_label']}" for j in r["jobs"]]
        out.append("")
    if r["commits"]:
        out += ["## Twin changes", ""]
        out += [f"- {c['id']}: {c['message']}" for c in r["commits"]]
        out.append("")
    q = r["data_quality"]
    out += ["## Data quality", "", f"{q['bad_now']} of {q['tags']} tags are flagged now; {q['flagged_total']} readings were held back from "
            "the estimator so far.", ""]
    out += [f"- {i['tag']}: {i['issue']} since {i['since']:.1f} h" for i in q["issues"][:10]]
    return "\n".join(out).rstrip() + "\n"
