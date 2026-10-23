import { useEffect, useMemo, useState } from "react";
import { HelpCircle, Zap, X } from "lucide-react";
import { api, fmt } from "../api";
import { useStore } from "../store";
import { base, C, EChart } from "./Chart";

const TABS = ["Live", "Twin", "Plan", "Why"] as const;

export function AssetDrawer() {
  const selected = useStore((s) => s.selected);
  const set = useStore((s) => s.set);
  const planVersion = useStore((s) => s.planVersion);
  const snap = useStore((s) => s.snap);
  const [tab, setTab] = useState<(typeof TABS)[number]>("Live");
  const [d, setD] = useState<any>(null);
  const [why, setWhy] = useState<any>(null);
  const [whyLoading, setWhyLoading] = useState(false);

  useEffect(() => {
    if (!selected) return;
    let alive = true;
    const load = () => api.asset(selected).then((r) => alive && setD(r)).catch(() => {});
    load();
    const id = setInterval(load, 4000);
    return () => { alive = false; clearInterval(id); };
  }, [selected, planVersion]);

  useEffect(() => { setWhy(null); }, [selected, planVersion]);
  useEffect(() => {
    if (tab !== "Why" || !selected || why) return;
    setWhyLoading(true);
    api.explain(selected).then(setWhy).catch(() => setWhy({ error: "explanation unavailable" })).finally(() => setWhyLoading(false));
  }, [tab, selected, why]);

  if (!selected) return null;
  return (
    <aside className="w-[420px] shrink-0 border-l border-line bg-panel flex flex-col fade-in">
      <div className="flex items-start gap-2 px-3 py-2.5 border-b border-line">
        <div className="min-w-0 flex-1">
          <div className="text-[14px] font-semibold truncate">{d?.label ?? selected}</div>
          <div className="text-[11px] text-dim truncate">{d?.type?.replace(/_/g, " ")} · <span className="num">{d?.tag || selected}</span></div>
          {d?.doc && <div className="text-[11px] text-muted mt-1 leading-snug">{d.doc}</div>}
        </div>
        <button className="btn btn-ghost !p-1" onClick={() => set({ selected: null })}><X size={15} /></button>
      </div>
      <div className="flex border-b border-line px-1">
        {TABS.map((t) => <div key={t} className={`tab ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>{t}</div>)}
      </div>
      <div className="flex-1 overflow-auto p-3 space-y-3">
        {!d && <div className="text-dim">Loading…</div>}
        {d && tab === "Live" && <Live d={d} />}
        {d && tab === "Twin" && <Twin d={d} />}
        {d && tab === "Plan" && <Plan d={d} now={snap?.t ?? 0} />}
        {tab === "Why" && <Why why={why} loading={whyLoading} now={snap?.t ?? 0} />}
      </div>
      {d && <Inject path={selected} d={d} />}
    </aside>
  );
}

/* Live-demo disruptions: applied to the emulated plant of BOTH worlds (PLYXIS and manual operations). */
const DISRUPTIONS: Record<string, { label: string; ev: (path: string, label: string) => Record<string, unknown> }[]> = {
  compressor_station: [{ label: "Trip one unit", ev: (a, l) => ({ kind: "trip", asset: a, units: 1, title: `Compressor unit trip at ${l} (live)` }) }],
  esp_well: [{ label: "Fail the ESP", ev: (a, l) => ({ kind: "fail", asset: a, title: `ESP failure at ${l} (live)` }) }],
  rod_pump_well: [{ label: "Fail the pump", ev: (a, l) => ({ kind: "fail", asset: a, title: `Pump failure at ${l} (live)` }) }],
  swd_well: [{ label: "Lose 50 % injectivity", ev: (a, l) => ({ kind: "capacity", asset: a, param: "inj_cap", factor: 0.5, title: `Injectivity loss at ${l} (live)` }) }],
};

function Inject({ path, d }: { path: string; d: any }) {
  const [done, setDone] = useState<string | null>(null);
  const actions = DISRUPTIONS[d?.type] ?? [];
  useEffect(() => { setDone(null); }, [path]);
  if (!actions.length) return null;
  return (
    <div className="border-t border-line px-3 py-2 flex items-center gap-2">
      <Zap size={13} className="text-warn shrink-0" />
      <div className="text-[11px] text-dim flex-1 leading-tight">Live demo: disrupt the plant (both worlds)</div>
      {actions.map((a) => (
        <button key={a.label} className="btn" disabled={done === a.label}
          onClick={() => api.event(a.ev(path, d.label ?? path)).then(() => setDone(a.label)).catch(() => setDone(null))}>
          {done === a.label ? "Injected" : a.label}
        </button>
      ))}
    </div>
  );
}

function Live({ d }: { d: any }) {
  const tags = Object.entries(d.tags ?? {}) as [string, any][];
  return (
    <>
      {tags.length === 0 && <div className="text-dim text-[12px]">This asset publishes no telemetry.</div>}
      {tags.map(([tag, info]) => {
        const h: [number, number][] = info.history ?? [];
        const last = h.length ? h[h.length - 1][1] : undefined;
        return (
          <div key={tag} className="panel">
            <div className="panel-h !normal-case !tracking-normal">
              <span className="num text-[11px] text-muted">{tag}</span>
              <span className="num text-ink text-[13px]">{last === undefined ? "–" : fmt.n(last, last < 10 ? 2 : 0)} <span className="text-dim text-[10px]">{info.unit}</span></span>
            </div>
            <EChart height={90} option={base({
              grid: { left: 44, right: 8, top: 8, bottom: 18 }, legend: { show: false },
              xAxis: { type: "value", min: "dataMin", max: "dataMax", axisLabel: { color: C.axis, fontSize: 9, formatter: (v: number) => fmt.hhmm(v) }, splitLine: { show: false } },
              series: [{ type: "line", data: h, showSymbol: false, lineStyle: { width: 1.2, color: C.teal }, areaStyle: { color: "#2dd4bf14" } }],
            })} />
          </div>
        );
      })}
      {(d.alerts ?? []).length > 0 && (
        <div className="panel">
          <div className="panel-h">Alerts</div>
          {d.alerts.slice().reverse().map((a: any, i: number) => (
            <div key={i} className="px-2.5 py-1.5 border-b border-[#131c26] text-[12px]">
              <span className={a.severity === "critical" ? "text-crit" : a.severity === "warning" ? "text-warn" : "text-info"}>●</span> {a.message}
              <div className="text-[10.5px] text-dim num">{a.clock}</div>
            </div>
          ))}
        </div>
      )}
    </>
  );
}

function Twin({ d }: { d: any }) {
  const params = Object.entries(d.params ?? {}) as [string, any][];
  const states = Object.entries(d.states ?? {}) as [string, any][];
  return (
    <>
      <div className="panel">
        <div className="panel-h">Properties <span className="normal-case tracking-normal font-normal">engineering → live estimate</span></div>
        <table className="tbl">
          <tbody>
            {params.map(([k, p]) => {
              const ch = p.engineering ? ((p.value - p.engineering) / Math.abs(p.engineering)) * 100 : 0;
              return (
                <tr key={k}>
                  <td className="text-muted">{k}{p.calibrated && <span className="ml-1 text-[9px] text-teal border border-teal/40 rounded px-1">calibrated</span>}</td>
                  <td className="num text-right text-dim">{fmt.n(p.engineering, p.engineering < 10 ? 3 : 0)}</td>
                  <td className="num text-right">{fmt.n(p.value, p.value < 10 ? 3 : 0)}</td>
                  <td className="num text-right text-[11px]">{Math.abs(ch) >= 0.5 ? <span className={ch > 0 ? "text-plx" : "text-leg"}>{fmt.pct(ch)}</span> : ""}</td>
                  <td className="text-dim text-[10.5px]">{p.unit}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="panel">
        <div className="panel-h">State estimate</div>
        <table className="tbl"><tbody>
          {states.map(([k, s]) => (
            <tr key={k}><td className="text-muted">{k}</td><td className="num text-right">{fmt.n(s.value, Math.abs(s.value) < 10 ? 3 : 0)}</td>
              <td className="text-dim text-[10.5px]">{s.unit}</td></tr>
          ))}
        </tbody></table>
      </div>
    </>
  );
}

function Plan({ d, now }: { d: any; now: number }) {
  const series = Object.entries(d.plan ?? {}) as [string, any][];
  const t0 = d.plan_t0 ?? 0, dt = d.plan_dt ?? 1;
  const interesting = series.filter(([k]) => !k.startsWith("term:") || k.includes("revenue") || k.includes("cost:power")).slice(0, 8);
  return (
    <>
      <div className="panel">
        <div className="panel-h">Operating instructions</div>
        <table className="tbl"><tbody>
          {(d.instructions ?? []).slice(0, 40).map((i: any, k: number) => (
            <tr key={k} className={i.start <= now && now < i.end ? "bg-[#0f2e2b55]" : ""}>
              <td className="num text-dim">{fmt.clock(i.start)}–{fmt.hhmm(i.end)}</td>
              <td className="text-muted">{i.decision}</td><td className="num text-right">{i.display}</td>
            </tr>
          ))}
        </tbody></table>
      </div>
      {interesting.map(([k, s]) => (
        <div key={k} className="panel">
          <div className="panel-h !normal-case !tracking-normal"><span className="num text-[11px]">{k}</span><span className="text-dim">{s.unit}</span></div>
          <EChart height={80} option={base({
            grid: { left: 48, right: 8, top: 6, bottom: 18 }, legend: { show: false },
            xAxis: { type: "value", min: t0, max: t0 + s.values.length * dt, axisLabel: { color: C.axis, fontSize: 9, formatter: (v: number) => fmt.hhmm(v) }, splitLine: { show: false } },
            series: [{ type: "line", step: "end", showSymbol: false, data: s.values.map((v: number, i: number) => [t0 + i * dt, v]),
              lineStyle: { width: 1.2, color: C.opt }, areaStyle: { color: "#a78bfa12" },
              markLine: { symbol: "none", silent: true, lineStyle: { color: "#f59e0b", type: "dashed" }, data: [{ xAxis: now }], label: { show: false } } }],
          })} />
        </div>
      ))}
    </>
  );
}

function Why({ why, loading, now }: { why: any; loading: boolean; now: number }) {
  const blocks = useMemo(() => {
    if (!why?.decisions) return [];
    return why.decisions.map((dec: any) => {
      const b = dec.blocks.find((b: any) => b.t_start <= (now - (why.t0 ?? 0)) && (now - (why.t0 ?? 0)) < b.t_end) ?? dec.blocks[0];
      return { dec, b };
    });
  }, [why, now]);
  if (loading) return <div className="text-dim">Differentiating the plan with respect to this asset…</div>;
  if (!why) return null;
  if (why.error) return <div className="text-dim">{why.error}</div>;
  if (!blocks.length) return <div className="text-dim text-[12px]">This asset has no decisions of its own; it is driven by its neighbours.</div>;
  return (
    <>
      <div className="text-[11.5px] text-muted leading-snug flex gap-2">
        <HelpCircle size={14} className="shrink-0 mt-0.5 text-teal" />
        Reasons are computed from the twin: the marginal value of each decision split into objective terms and the shadow
        prices of binding constraints (per +10 % of the decision range).
      </div>
      {blocks.map(({ dec, b }: any) => (
        <div key={dec.name} className="panel">
          <div className="panel-h"><span>{dec.name.replace(/_/g, " ")}</span><span className="num text-ink normal-case">{b.display}</span></div>
          <div className="px-3 py-2 text-[12.5px] leading-snug">{b.summary}</div>
          <div className="px-3 pb-2 space-y-1">
            {b.drivers.map((dr: any, i: number) => {
              const w = Math.min(100, Math.abs(dr.usd_per_h) / Math.max(1, ...b.drivers.map((x: any) => Math.abs(x.usd_per_h))) * 100);
              return (
                <div key={i} className="flex items-center gap-2 text-[11.5px]">
                  <div className="w-[170px] truncate text-muted" title={dr.label}>{dr.label}</div>
                  <div className="flex-1 h-2 bg-panel2 rounded relative overflow-hidden">
                    <div className={`absolute top-0 bottom-0 ${dr.usd_per_h >= 0 ? "bg-plx/70" : "bg-crit/70"}`} style={{ width: `${w}%` }} />
                  </div>
                  <div className={`num w-[78px] text-right ${dr.usd_per_h >= 0 ? "text-plx" : "text-crit"}`}>{dr.usd_per_h >= 0 ? "+" : "−"}${fmt.n(Math.abs(dr.usd_per_h))}/h</div>
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </>
  );
}
