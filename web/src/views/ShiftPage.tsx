import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, Check, ClipboardList, Download, Eye, FileText, Hand, Pencil, ShieldCheck, X, Zap } from "lucide-react";
import type { AdvisoryView, Recommendation } from "../api";
import { api, fmt } from "../api";
import { useStore } from "../store";

const MODES = [
  { id: "autonomous", label: "Autonomous", icon: Zap, text: "The plan reaches the plant. The list shows what it changes next." },
  { id: "advisory", label: "Advisory", icon: Hand, text: "Nothing changes until an operator accepts it. Rejected and modified decisions are held; the plan re-optimises around them." },
  { id: "shadow", label: "Shadow", icon: Eye, text: "The plant runs on manual operations. Plans are computed and valued, nothing is executed." },
] as const;

const STATUS: Record<string, string> = {
  pending: "text-amber", accepted: "text-good", modified: "text-amber", rejected: "text-crit",
  automatic: "text-muted", shadow: "text-muted", superseded: "text-dim",
};

function Row({ r, advisory, onDone }: { r: Recommendation; advisory: boolean; onDone: (msg?: string) => void }) {
  const [open, setOpen] = useState<null | "reject" | "modify">(null);
  const [reason, setReason] = useState("");
  const [value, setValue] = useState(String(r.to ?? ""));
  const [why, setWhy] = useState<string | null>(null);
  const decide = async (action: "accept" | "reject" | "modify") => {
    const res = await api.decide([r.id], action, reason, action === "modify" ? Number(value) : undefined);
    if (!res.ok) { onDone(res.error ?? "not possible"); return; }
    setOpen(null);
    onDone();
  };
  const explain = () => {
    if (why !== null) { setWhy(null); return; }
    if (r.kind === "lineup") {
      setWhy("A line-up follows from the plan as a whole, not from one asset. The shadow prices on the Optimizer page show what limits the receiving facilities.");
      return;
    }
    setWhy("computing reasons…");
    api.explain(r.asset).then((w) => {
      const d = w.decisions?.find((x: any) => x.name === r.decision);
      setWhy(d?.blocks?.[0]?.summary ?? "No reason available for this decision.");
    }).catch(() => setWhy("Reasons are not available right now."));
  };
  const pending = r.status === "pending" && advisory;
  return (
    <>
      <tr>
        <td className="max-w-[240px] truncate" title={r.asset}>{r.label}</td>
        <td className="text-muted">{r.decision.replace(/_/g, " ")}</td>
        <td className="num text-muted max-w-[220px] truncate" title={r.from_display}>{r.from_display}</td>
        <td className="num max-w-[220px] truncate" title={r.to_display}>{r.to_display}</td>
        <td className="num text-muted">{fmt.hhmm(r.start)}</td>
        <td className={`text-[11.5px] ${STATUS[r.status] ?? "text-muted"}`}>{r.status}</td>
        <td className="whitespace-nowrap">
          <div className="flex items-center gap-1 justify-end">
            <button className="btn !py-0.5 !px-1.5" title="Why does the plan want this?" onClick={explain}>Why</button>
            {pending && <>
              <button className="btn !py-0.5 !px-1.5" title="Accept" onClick={() => decide("accept")}><Check size={12} /></button>
              {r.kind === "setpoint" && <button className="btn !py-0.5 !px-1.5" title="Modify" onClick={() => setOpen(open === "modify" ? null : "modify")}><Pencil size={12} /></button>}
              <button className="btn !py-0.5 !px-1.5" title="Reject" onClick={() => setOpen(open === "reject" ? null : "reject")}><X size={12} /></button>
            </>}
          </div>
        </td>
      </tr>
      {why !== null && <tr><td colSpan={7} className="text-[11.5px] text-muted !whitespace-normal bg-panel2">{why}</td></tr>}
      {open && (
        <tr><td colSpan={7} className="bg-panel2">
          <div className="flex flex-wrap items-center gap-2 py-1">
            {open === "modify" && <label className="text-[11.5px] text-muted flex items-center gap-1.5">set point
              <input value={value} onChange={(e) => setValue(e.target.value)} className="num w-24 bg-panel border border-line2 rounded px-1.5 py-0.5 text-ink text-[12px]" /></label>}
            <input value={reason} onChange={(e) => setReason(e.target.value)} placeholder="Reason (needed: it shows the twin what it misses)"
              className="flex-1 min-w-[200px] bg-panel border border-line2 rounded px-2 py-0.5 text-ink text-[12px]" />
            <button className="btn btn-primary !py-0.5" disabled={!reason.trim()} onClick={() => decide(open)}>{open === "modify" ? "Modify" : "Reject"}</button>
          </div>
        </td></tr>
      )}
    </>
  );
}

/** The report's small markdown subset: headings, lists, pipe tables, paragraphs. */
function Markdown({ text }: { text: string }) {
  const lines = text.split("\n");
  const out: JSX.Element[] = [];
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    if (!l.trim()) continue;
    if (l.startsWith("# ")) out.push(<h1 key={i} className="display text-[20px] font-semibold text-ink mb-1">{l.slice(2)}</h1>);
    else if (l.startsWith("## ")) out.push(<h2 key={i} className="text-[11px] font-semibold uppercase tracking-wider text-muted mt-5 mb-1.5">{l.slice(3)}</h2>);
    else if (l.startsWith("- ")) {
      const items: string[] = [];
      while (i < lines.length && lines[i].startsWith("- ")) items.push(lines[i++].slice(2));
      i--;
      out.push(<ul key={i} className="list-disc pl-5 space-y-0.5 text-[12.5px] text-ink">{items.map((t, k) => <li key={k}>{t}</li>)}</ul>);
    } else if (l.startsWith("|")) {
      const rows: string[][] = [];
      while (i < lines.length && lines[i].startsWith("|")) {
        if (!/^\|[-:| ]+\|$/.test(lines[i])) rows.push(lines[i].split("|").slice(1, -1).map((c) => c.trim()));
        i++;
      }
      i--;
      out.push(
        <table key={i} className="tbl !w-auto my-1.5">
          <thead><tr>{rows[0].map((c, k) => <th key={k} className={k ? "!text-right" : ""}>{c}</th>)}</tr></thead>
          <tbody>{rows.slice(1).map((r, n) => <tr key={n}>{r.map((c, k) => <td key={k} className={k ? "num text-right" : ""}>{c}</td>)}</tr>)}</tbody>
        </table>);
    } else out.push(<p key={i} className="text-[12.5px] leading-relaxed text-ink my-1">{l}</p>);
  }
  return <div className="max-w-[900px]">{out}</div>;
}

export function ShiftPage() {
  const planVersion = useStore((s) => s.planVersion);
  const snapMode = useStore((s) => s.snap?.advisory?.mode);
  const [d, setD] = useState<AdvisoryView | null>(null);
  const [audit, setAudit] = useState<any[]>([]);
  const [report, setReport] = useState("");
  const [tab, setTab] = useState<"changes" | "report" | "audit">("changes");
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = () => {
    api.advisory().then(setD).catch(() => {});
    api.audit().then(setAudit).catch(() => {});
  };
  useEffect(() => {
    load();
    const id = setInterval(load, 4000);
    return () => clearInterval(id);
  }, [planVersion, snapMode]);
  useEffect(() => {
    if (tab !== "report") return;
    let alive = true;
    const get = () => api.reportMd().then((t) => alive && setReport(t)).catch((e) => alive && setReport(String(e)));
    get();
    const id = setInterval(get, 8000);
    return () => { alive = false; clearInterval(id); };
  }, [tab]);

  const setMode = async (mode: string) => {
    setBusy(true);
    try { const r = await api.setMode(mode); if (!r.ok) setMsg(r.error ?? "not possible"); else setMsg(null); load(); } finally { setBusy(false); }
  };
  const acceptAll = async () => {
    setBusy(true);
    try { const r = await api.decide("all", "accept", ""); setMsg(r.ok ? null : r.error ?? "not possible"); load(); } finally { setBusy(false); }
  };
  const download = () => {
    const url = URL.createObjectURL(new Blob([report], { type: "text/markdown" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = "plyxis-shift-report.md";
    a.click();
    URL.revokeObjectURL(url);
  };

  const mode = d?.mode ?? "autonomous";
  const recs = d?.recommendations ?? [];
  const pending = recs.filter((r) => r.status === "pending").length;
  const info = MODES.find((m) => m.id === mode);
  const shadow = d?.shadow ?? [];
  const lastShadow = shadow[shadow.length - 1];
  const auditRows = useMemo(() => audit.slice().reverse().slice(0, 120), [audit]);

  return (
    <div className="h-full overflow-auto p-3 flex flex-col gap-3">
      <div className="panel px-3 py-2.5 flex flex-wrap items-center gap-3">
        <div className="flex rounded-md border border-line2 overflow-hidden shrink-0">
          {MODES.map(({ id, label, icon: Icon }) => (
            <button key={id} disabled={busy} onClick={() => setMode(id)}
              className={`px-2.5 py-1 text-[12px] flex items-center gap-1.5 ${mode === id ? "seg-active" : "bg-panel2 text-muted hover:text-ink"}`}>
              <Icon size={13} />{label}
            </button>
          ))}
        </div>
        <div className="flex-1 min-w-[260px] text-[12px] text-muted">{info?.text}</div>
        {d && !d.plan_valid && <div className="flex items-center gap-1.5 text-[12px] text-crit"><AlertTriangle size={14} />The last plan has expired: no recommendation</div>}
        {d?.valid_until != null && d.plan_valid && <div className="num text-[11.5px] text-dim shrink-0">plan #{d.plan} valid until {fmt.clock(d.valid_until)}</div>}
      </div>
      {msg && <div className="panel px-3 py-2 text-[12px] text-amber flex items-center gap-2"><AlertTriangle size={14} />{msg}</div>}

      <div className="flex items-center gap-1 shrink-0">
        {([["changes", "Next shift", ClipboardList], ["report", "Shift report", FileText], ["audit", "Audit trail", ShieldCheck]] as const).map(([id, label, Icon]) => (
          <button key={id} onClick={() => setTab(id)} className={`btn ${tab === id ? "btn-primary" : ""}`}><Icon size={13} />{label}</button>
        ))}
        <div className="flex-1" />
        {tab === "changes" && mode === "advisory" && <button className="btn btn-primary" disabled={busy || !pending} onClick={acceptAll}><Check size={13} />Accept all {pending ? `(${pending})` : ""}</button>}
        {tab === "report" && <button className="btn" disabled={!report} onClick={download}><Download size={13} />Download</button>}
      </div>

      {tab === "changes" && (
        <div className="grid grid-cols-12 gap-3 flex-1 min-h-0">
          <div className="col-span-12 xl:col-span-8 panel flex flex-col min-h-[320px]">
            <div className="panel-h"><span>Changes the plan asks for in the next {d?.shift_h ?? 12} h</span><span className="num normal-case font-normal">{d?.total ?? 0} changes</span></div>
            <div className="flex-1 overflow-auto">
              <table className="tbl">
                <thead><tr><th>asset</th><th>decision</th><th>now</th><th>plan</th><th>at</th><th>status</th><th /></tr></thead>
                <tbody>
                  {recs.map((r) => <Row key={r.id} r={r} advisory={mode === "advisory"} onDone={(m) => { setMsg(m ?? null); load(); }} />)}
                  {!recs.length && <tr><td colSpan={7} className="text-dim">The plan keeps everything as it runs now.</td></tr>}
                </tbody>
              </table>
            </div>
          </div>
          <div className="col-span-12 xl:col-span-4 flex flex-col gap-3 min-w-0">
            {mode === "shadow" && (
              <div className="panel">
                <div className="panel-h">Shadow value</div>
                <div className="p-3 text-[12px] text-muted space-y-1.5">
                  {lastShadow ? <>
                    <div className="display text-[22px] text-ink">{fmt.usd(lastShadow.gap)}</div>
                    <div>The latest plan against holding what the operators run now, over the plan's horizon, on the calibrated twin.</div>
                    <div className="num text-[11.5px]">plan {fmt.usd(lastShadow.plan_value)} · hold {fmt.usd(lastShadow.hold_value)}</div>
                  </> : "The first comparison appears with the next plan."}
                </div>
              </div>
            )}
            <div className="panel">
              <div className="panel-h">Held for the planner</div>
              <div className="p-2 text-[12px]">
                {(d?.overrides ?? []).map((o) => (
                  <div key={`${o.asset}-${o.decision}`} className="flex justify-between gap-2 py-0.5"><span className="truncate">{o.asset} · {o.decision}</span><span className="num text-muted shrink-0">until {fmt.clock(o.until)}</span></div>
                ))}
                {!(d?.overrides ?? []).length && <div className="text-dim px-1">No decision is held.</div>}
              </div>
            </div>
            <div className="panel flex-1 min-h-[140px] flex flex-col">
              <div className="panel-h">Recent decisions</div>
              <div className="p-2 text-[12px] overflow-auto">
                {(d?.recent_decisions ?? []).map((r) => (
                  <div key={r.id} className="py-1 border-b border-line last:border-0">
                    <div className="flex justify-between gap-2"><span className="truncate">{r.label} · {r.decision}</span><span className={`shrink-0 ${STATUS[r.status] ?? ""}`}>{r.status}</span></div>
                    <div className="text-dim text-[11px]">{r.decided_by}{r.reason ? `: ${r.reason}` : ""}</div>
                  </div>
                ))}
                {!(d?.recent_decisions ?? []).length && <div className="text-dim px-1">No decision yet.</div>}
              </div>
            </div>
          </div>
        </div>
      )}

      {tab === "report" && (
        <div className="panel flex-1 min-h-[320px] overflow-auto px-5 py-4">
          {report ? <Markdown text={report} /> : <div className="text-[12.5px] text-dim">Building the report…</div>}
        </div>
      )}

      {tab === "audit" && (
        <div className="panel flex-1 min-h-[320px] overflow-auto">
          <table className="tbl">
            <thead><tr><th>plant time</th><th>what</th><th>who</th><th>details</th></tr></thead>
            <tbody>
              {auditRows.map((a, i) => {
                const { wall: _w, t, kind, actor, ...rest } = a;
                return (
                  <tr key={i}><td className="num text-muted">{fmt.clock(t)}</td><td>{kind.replace(/_/g, " ")}</td><td className="text-muted">{actor}</td>
                    <td className="text-[11.5px] text-muted !whitespace-normal">{Object.entries(rest).map(([k, v]) => `${k}: ${String(v)}`).join(" · ")}</td></tr>
                );
              })}
              {!auditRows.length && <tr><td colSpan={4} className="text-dim">Nothing recorded yet.</td></tr>}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
