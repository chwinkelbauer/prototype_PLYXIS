import { useEffect, useMemo, useState } from "react";
import { Cable, Check, GitPullRequestArrow, Radar, Search, X } from "lucide-react";
import { api, fmt } from "../api";
import { base, C, EChart } from "../components/Chart";
import { useStore } from "../store";

export function DataPage() {
  const snap = useStore((s) => s.snap);
  const alerts = useStore((s) => s.alerts);
  const set = useStore((s) => s.set);
  const [conns, setConns] = useState<any[]>([]);
  const [tags, setTags] = useState<any[]>([]);
  const [q, setQ] = useState("");
  const [sel, setSel] = useState<string | null>(null);
  const [hist, setHist] = useState<[number, number][]>([]);
  const [cal, setCal] = useState<any>(null);
  const [disc, setDisc] = useState<any[]>([]);

  useEffect(() => {
    let alive = true;
    const load = () => {
      api.connectors().then((r) => alive && setConns(r)).catch(() => {});
      api.tags(q).then((r) => alive && setTags(r)).catch(() => {});
      api.calibration().then((r) => alive && setCal(r)).catch(() => {});
      api.discovery().then((r) => alive && setDisc(r)).catch(() => {});
    };
    load();
    const id = setInterval(load, 3000);
    return () => { alive = false; clearInterval(id); };
  }, [q]);
  useEffect(() => {
    if (!sel) return;
    let alive = true;
    const load = () => api.tag(sel).then((r) => alive && setHist(r.history)).catch(() => {});
    load();
    const id = setInterval(load, 2000);
    return () => { alive = false; clearInterval(id); };
  }, [sel]);

  const anomalies = alerts.filter((a) => ["mismatch", "trend", "threshold", "calibration"].includes(a.kind)).slice(-40).reverse();
  const lastCal = cal?.log?.[cal.log.length - 1];
  const changes = useMemo(() => (lastCal?.items ?? []).slice().sort((a: any, b: any) => Math.abs(b.change_pct ?? 0) - Math.abs(a.change_pct ?? 0)), [lastCal]);

  return (
    <div className="h-full overflow-auto p-3">
      <div className="grid grid-cols-12 gap-3">
        <div className="col-span-12 grid grid-cols-3 gap-3">
          {conns.map((c) => (
            <div key={c.name} className="panel px-3 py-2.5 flex items-start gap-3">
              <Cable size={18} className="text-teal mt-0.5" />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2"><span className="text-[13px] font-semibold">{c.name}</span>
                  <span className="chip !text-plx !border-plx/40">{c.status}</span></div>
                <div className="text-[11.5px] text-dim truncate">{c.description}</div>
                <div className="num text-[11.5px] text-muted mt-1">{fmt.n(c.messages)} messages · {fmt.n(c.rate_per_s, 1)}/s · {c.kind}</div>
              </div>
            </div>
          ))}
        </div>

        <div className="col-span-7 panel flex flex-col" style={{ height: 470 }}>
          <div className="panel-h"><span>Tag browser</span>
            <span className="flex items-center gap-1.5 normal-case tracking-normal"><Search size={12} />
              <input className="inp w-[200px] !py-0.5" placeholder="filter by prefix, e.g. KST.P07" value={q} onChange={(e) => setQ(e.target.value)} /></span></div>
          <div className="flex-1 overflow-auto">
            <table className="tbl">
              <thead><tr><th>tag</th><th className="!text-right">value</th><th>unit</th><th>twin signal</th></tr></thead>
              <tbody>
                {tags.map((t) => (
                  <tr key={t.tag} onClick={() => setSel(t.tag)} className={`cursor-pointer ${sel === t.tag ? "bg-panel2" : ""}`}>
                    <td className="num text-[11.5px]">{t.mapped ? "" : <span className="text-warn mr-1">●</span>}{t.tag}</td>
                    <td className="num text-right">{fmt.n(t.value, Math.abs(t.value) < 10 ? 3 : 1)}</td>
                    <td className="text-dim">{t.unit}</td>
                    <td className="text-[11px] text-muted truncate max-w-[200px]">{t.asset ? `${t.asset} · ${String(t.signal).split(":").pop()}` : t.tag.startsWith("MKT") ? "market forecast input" : "unmapped"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div className="col-span-5 flex flex-col gap-3" style={{ height: 470 }}>
          <div className="panel">
            <div className="panel-h"><span className="num normal-case tracking-normal">{sel ?? "select a tag"}</span></div>
            <EChart height={170} option={base({
              grid: { left: 52, right: 10, top: 10, bottom: 22 }, legend: { show: false },
              xAxis: { type: "value", min: "dataMin", max: "dataMax", axisLabel: { color: C.axis, fontSize: 10, formatter: (v: number) => fmt.clock(v) }, splitLine: { show: false } },
              series: [{ type: "line", showSymbol: false, data: hist, lineStyle: { width: 1.3, color: C.teal }, areaStyle: { color: "#2dd4bf14" } }],
            })} />
          </div>
          <div className="panel flex-1 min-h-0 flex flex-col">
            <div className="panel-h"><span className="flex items-center gap-1.5"><Radar size={12} />Anomalies & predictive maintenance</span></div>
            <div className="flex-1 overflow-auto">
              {anomalies.length === 0 && <div className="p-3 text-dim text-[12px]">No anomalies.</div>}
              {anomalies.map((a, i) => (
                <div key={i} className="px-2.5 py-1.5 border-b border-[#131c26] text-[12px] cursor-pointer hover:bg-panel2" onClick={() => a.asset && set({ selected: a.asset })}>
                  <span className={a.kind === "trend" ? "text-opt" : a.severity === "critical" ? "text-crit" : "text-warn"}>{a.kind}</span> · {a.message}
                  <div className="text-[10.5px] text-dim num">{a.clock}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="col-span-7 panel flex flex-col" style={{ maxHeight: 420 }}>
          <div className="panel-h"><span>Calibration · moving-horizon estimation</span>
            <span className="normal-case tracking-normal font-normal">{lastCal ? `last run ${fmt.clock(lastCal.t)} · ${lastCal.changes} properties · commit ${lastCal.commit ?? "–"}` : "waiting for data"}</span></div>
          <div className="px-3 py-2 text-[11.5px] text-dim leading-snug">
            Every hour the twin is replayed through the last hours with the controls actually applied in the plant and its
            calibratable properties are re-estimated ({(cal?.calibrated ?? []).slice(0, 6).join(", ")}…). Properties with their
            own measurement are updated directly from it ({(cal?.direct ?? []).length ? "well tests: wc, gor" : "–"}).
          </div>
          <div className="flex-1 overflow-auto">
            <table className="tbl">
              <thead><tr><th>asset</th><th>property</th><th className="!text-right">change</th><th>method</th></tr></thead>
              <tbody>
                {changes.slice(0, 60).map((c: any, i: number) => (
                  <tr key={i} className="cursor-pointer" onClick={() => set({ selected: c.asset })}>
                    <td>{c.asset}</td><td className="text-muted">{c.property}</td>
                    <td className={`num text-right ${c.change_pct > 0 ? "text-plx" : "text-leg"}`}>{c.change_pct === null ? "–" : fmt.pct(c.change_pct)}</td>
                    <td className="text-dim text-[11px]">{c.how}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div className="col-span-5 panel flex flex-col" style={{ maxHeight: 420 }}>
          <div className="panel-h"><span className="flex items-center gap-1.5"><GitPullRequestArrow size={12} />Topology discovery</span></div>
          <div className="flex-1 overflow-auto p-2.5 space-y-2.5">
            {disc.length === 0 && <div className="text-dim text-[12px]">No unknown devices so far. When tags of an unregistered asset appear, a discovery rule turns them into PLX code here.</div>}
            {disc.slice().reverse().map((p) => (
              <div key={p.id} className="rounded border border-line2 bg-panel2">
                <div className="flex items-center gap-2 px-2.5 py-1.5 border-b border-line">
                  <span className="text-[12.5px] font-semibold">{p.assets.join(", ")}</span>
                  <span className="text-[11px] text-dim">rule {p.rule} · {fmt.clock(p.t)}</span>
                  <span className={`ml-auto chip ${p.status === "applied" ? "!text-plx !border-plx/40" : p.status === "rejected" ? "!text-crit" : "!text-warn"}`}>{p.status}</span>
                </div>
                <div className="px-2.5 py-1 text-[11px] text-dim">from tag <span className="num">{p.tag}</span></div>
                <pre className="num text-[11px] leading-snug px-2.5 pb-2 text-[#c3e88d] whitespace-pre-wrap">{p.code}</pre>
                {p.status === "pending" && (
                  <div className="flex gap-2 px-2.5 pb-2">
                    <button className="btn btn-primary" onClick={() => api.discoveryAction(p.id, "approve")}><Check size={12} />Approve & compile</button>
                    <button className="btn" onClick={() => api.discoveryAction(p.id, "reject")}><X size={12} />Reject</button>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
      <span className="hidden">{snap?.t}</span>
    </div>
  );
}
