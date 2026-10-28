import { useEffect, useMemo, useState } from "react";
import { Leaf, Scale, TrendingUp, Waypoints } from "lucide-react";
import type { KpiConfig } from "../api";
import { api, fmt } from "../api";
import { base, C, EChart } from "../components/Chart";
import { BridgeChart } from "../components/Bridge";
import { AlertFeed, Timeline } from "../components/Feed";
import { KpiAB } from "../components/Kpi";
import { useStore } from "../store";

const rel = (a: number, b: number) => (Math.abs(b) > 1e-9 ? ((a - b) / Math.abs(b)) * 100 : 0);

/** Relative difference with its label; status colour only when PLYXIS is better or worse. */
function Delta({ pct, label, better = "higher" }: { pct: number; label: string; better?: "higher" | "lower" }) {
  const good = better === "higher" ? pct >= 0 : pct <= 0;
  return (
    <span className="whitespace-nowrap">
      <span className={`num font-semibold ${good ? "text-good" : "text-crit"}`}>{pct >= 0 ? "▲" : "▼"} {Math.abs(pct).toFixed(1)} %</span>
      <span className="text-muted"> {label}</span>
    </span>
  );
}

function OverviewOilGas() {
  const snap = useStore((s) => s.snap)!;
  const hist = useStore((s) => s.kpiHistory);
  const planVersion = useStore((s) => s.planVersion);
  const set = useStore((s) => s.set);
  const [plan, setPlan] = useState<any>(null);
  useEffect(() => { api.plan().then(setPlan).catch(() => {}); }, [planVersion]);

  const k = snap.kpis;
  const P = k.plyxis, L = k.legacy;
  const boe = (w: typeof P) => w.cum.oil_bbl + w.cum.gas_mscf / 6;
  const intensity = (w: typeof P) => (boe(w) > 1 ? w.cum.emission_kg / boe(w) : NaN);
  const perBbl = (w: typeof P) => (w.cum.oil_bbl > 1 ? w.cum.cost / w.cum.oil_bbl : NaN);

  const prodOption = useMemo(() => {
    const series = plan?.plan?.series;
    const fc = series?.field_oil?.values?.map((v: number, i: number) => [series.hours[i], v]) ?? [];
    return base({
      grid: { left: 54, right: 16, top: 30, bottom: 24 },
      xAxis: { type: "value", min: "dataMin", axisLabel: { color: C.axis, fontSize: 10, formatter: (v: number) => fmt.clock(v) }, splitLine: { show: false } },
      yAxis: { type: "value", scale: true, axisLabel: { color: C.axis, fontSize: 10 }, splitLine: { lineStyle: { color: C.grid } } },
      tooltip: { trigger: "axis", valueFormatter: (v: any) => fmt.n(Number(v)) },
      series: [
        { name: "PLYXIS (actual)", type: "line", showSymbol: false, data: hist.map((h) => [h.t, h.p.oil_bpd ?? null]), lineStyle: { width: 1.6, color: C.plx }, itemStyle: { color: C.plx } },
        { name: "Manual ops (actual)", type: "line", showSymbol: false, data: hist.map((h) => [h.t, h.l.oil_bpd ?? null]), lineStyle: { width: 1.4, color: C.leg }, itemStyle: { color: C.leg } },
        { name: "PLYXIS plan (72 h)", type: "line", showSymbol: false, data: fc, lineStyle: { width: 1.4, color: C.opt, type: "dashed", opacity: 0.85 }, itemStyle: { color: C.opt } },
      ],
    });
  }, [hist, plan]);

  const valueOption = useMemo(() => base({
    grid: { left: 60, right: 16, top: 30, bottom: 24 },
    xAxis: { type: "value", min: "dataMin", axisLabel: { color: C.axis, fontSize: 10, formatter: (v: number) => fmt.clock(v) }, splitLine: { show: false } },
    yAxis: { type: "value", axisLabel: { color: C.axis, fontSize: 10, formatter: (v: number) => fmt.usd(v) }, splitLine: { lineStyle: { color: C.grid } } },
    tooltip: { trigger: "axis", valueFormatter: (v: any) => fmt.usd(Number(v)) },
    series: [
      { name: "Value captured (cumulative profit difference)", type: "line", showSymbol: false, data: hist.map((h) => [h.t, h.pc - h.lc]),
        lineStyle: { width: 2, color: C.plx }, areaStyle: { color: { type: "linear", x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: "#fe026540" }, { offset: 1, color: "#fe026500" }] } }, itemStyle: { color: C.plx } },
    ],
  }), [hist]);

  // value bridge: where the operating-profit difference comes from (realised terms, sums to the total)
  const bridge = k.bridge ?? [];

  const flareOption = useMemo(() => base({
    grid: { left: 54, right: 16, top: 30, bottom: 24 },
    xAxis: { type: "value", min: "dataMin", axisLabel: { color: C.axis, fontSize: 10, formatter: (v: number) => fmt.clock(v) }, splitLine: { show: false } },
    yAxis: { type: "value", axisLabel: { color: C.axis, fontSize: 10 }, splitLine: { lineStyle: { color: C.grid } } },
    tooltip: { trigger: "axis", valueFormatter: (v: any) => fmt.n(Number(v)) },
    series: [
      { name: "PLYXIS flaring", type: "line", showSymbol: false, data: hist.map((h) => [h.t, h.p.flare_mscfd ?? 0]), lineStyle: { width: 1.4, color: C.plx }, itemStyle: { color: C.plx }, areaStyle: { color: "#fe026514" } },
      { name: "Manual ops flaring", type: "line", showSymbol: false, data: hist.map((h) => [h.t, h.l.flare_mscfd ?? 0]), lineStyle: { width: 1.4, color: C.leg }, itemStyle: { color: C.leg }, areaStyle: { color: "#8b93a714" } },
    ],
  }), [hist]);

  const bn = plan?.plan?.bottlenecks ?? [];
  return (
    <div className="h-full overflow-auto p-3">
      <div className="grid grid-cols-12 gap-3">
        {/* value hero */}
        <div className="col-span-4 panel hero p-4 flex flex-col justify-between">
          <div>
            <div className="text-[10.5px] uppercase tracking-wider text-brand2 font-semibold flex items-center gap-1.5"><TrendingUp size={13} />Value captured vs. manual operations</div>
            <div className="display text-[44px] leading-none font-bold mt-2 brand-text tabular-nums">{fmt.usd(k.delta.profit_after_carbon ?? k.delta.profit)}</div>
            <div className="text-[12px] text-muted mt-1.5">profit after carbon cost · same field, events and physics, over <span className="num">{fmt.n(k.hours, 1)}</span> h</div>
            <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2.5 text-[12px]">
              <Delta pct={rel(P.cum.profit, L.cum.profit)} label="operating profit" />
              <Delta pct={rel(P.cum.cost, L.cum.cost)} label="operating cost" better="lower" />
              <Delta pct={rel(P.cum.flared_mscf, L.cum.flared_mscf)} label="flaring" better="lower" />
            </div>
          </div>
          <div className="grid grid-cols-3 gap-2 mt-4">
            <div><div className="text-[10px] text-dim uppercase">operating profit</div><div className="num text-[16px] font-semibold text-plx">{fmt.usd(k.delta.profit)}</div></div>
            <div><div className="text-[10px] text-dim uppercase">CO2e avoided</div><div className="num text-[16px] font-semibold text-plx">{fmt.n(-k.delta.emission_t, 0)} t</div></div>
            <div><div className="text-[10px] text-dim uppercase">value per day</div><div className="num text-[16px] font-semibold text-plx">{fmt.usd(k.value_per_day)}</div></div>
          </div>
        </div>
        {/* cumulative since start: these converge instead of jumping with every 10-minute step */}
        <div className="col-span-8 grid grid-cols-4 gap-3">
          <KpiAB label="Operating profit" plx={P.cum.profit} leg={L.cum.profit} format={(v) => fmt.usd(v)} hint="since start: sales − cash cost + change in inventory and pump life" />
          <KpiAB label="Operating cost" plx={P.cum.cost} leg={L.cum.cost} better="lower" format={(v) => fmt.usd(v)} hint="since start: power, trucking, disposal, treating, lease, restarts, failures" />
          <KpiAB label="Cost per barrel" plx={perBbl(P)} leg={perBbl(L)} unit="$/bbl" better="lower" digits={2} hint="cash operating cost per barrel of oil produced, since start" />
          <KpiAB label="Revenue" plx={P.cum.revenue} leg={L.cum.revenue} format={(v) => fmt.usd(v)} hint="oil and gas sales since start" />
          <KpiAB label="Oil production" plx={P.rate.oil_bpd} leg={L.rate.oil_bpd} unit="bopd" hint="current rate, smoothed over ~1.5 h" />
          <KpiAB label="Oil produced" plx={P.cum.oil_bbl} leg={L.cum.oil_bbl} unit="bbl" />
          <KpiAB label="Flaring" plx={P.cum.flared_mscf} leg={L.cum.flared_mscf} unit="Mscf" better="lower" hint="gas flared since start" />
          <KpiAB label="Emissions intensity" plx={intensity(P)} leg={intensity(L)} unit="kgCO2e/boe" better="lower" digits={1} />
        </div>

        <div className="col-span-8 panel">
          <div className="panel-h">Oil production, bopd: actual (both worlds) and current 72 h plan</div>
          <EChart height={250} option={prodOption} />
        </div>
        <div className="col-span-4 panel flex flex-col" style={{ height: 286 }}>
          <div className="panel-h">Autonomous decisions & events</div>
          <div className="flex-1 min-h-0"><Timeline limit={80} /></div>
        </div>

        <div className="col-span-4 panel">
          <div className="panel-h">Value captured over time, USD</div>
          <EChart height={220} option={valueOption} />
        </div>
        <div className="col-span-4 panel">
          <div className="panel-h"><span className="flex items-center gap-1.5"><Waypoints size={12} />Where the value comes from</span>
            <span className="normal-case tracking-normal font-normal text-dim">PLYXIS − manual</span></div>
          <BridgeChart rows={bridge} height={220} />
        </div>
        <div className="col-span-4 panel">
          <div className="panel-h flex"><span className="flex items-center gap-1.5"><Leaf size={12} />Flaring, Mscf/d</span></div>
          <EChart height={220} option={flareOption} />
        </div>

        <LowerRow plan={plan} />
      </div>
    </div>
  );
}

/** Bottlenecks, alerts and maintenance dispatch (same for every case study). */
function LowerRow({ plan }: { plan: any }) {
  const snap = useStore((s) => s.snap)!;
  const set = useStore((s) => s.set);
  const bn = plan?.plan?.bottlenecks ?? [];
  return (
    <>
        <div className="col-span-4 panel flex flex-col" style={{ height: 240 }}>
        <div className="panel-h"><span className="flex items-center gap-1.5"><Scale size={12} />Bottlenecks & shadow prices</span>
          <button className="btn btn-ghost !py-0 !text-[10.5px]" onClick={() => set({ view: "optimizer" })}>details</button></div>
        <div className="flex-1 overflow-auto">
          <table className="tbl">
            <thead><tr><th>constraint</th><th className="!text-right">binding</th><th className="!text-right">+10 % worth</th></tr></thead>
            <tbody>
              {bn.length === 0 && <tr><td colSpan={3} className="text-dim">No binding constraints in the current plan.</td></tr>}
              {bn.map((b: any, i: number) => (
                <tr key={i} title={b.max_price ?? ""}>
                  <td className="truncate max-w-[210px]">{b.label}</td>
                  <td className="num text-right text-muted">{fmt.n(b.hours_binding)} h</td>
                  <td className="num text-right text-plx">{fmt.usd(b.value_10pct_per_day)}/d</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="col-span-4 panel flex flex-col" style={{ height: 240 }}>
        <div className="panel-h">Alerts</div>
        <div className="flex-1 min-h-0"><AlertFeed limit={60} /></div>
      </div>
      <div className="col-span-4 panel flex flex-col" style={{ height: 240 }}>
        <div className="panel-h">Maintenance dispatch</div>
        <div className="flex-1 overflow-auto">
          <table className="tbl">
            <thead><tr><th>asset</th><th>window</th><th className="!text-right">value</th><th>status</th></tr></thead>
            <tbody>
              {snap.jobs.length === 0 && <tr><td colSpan={4} className="text-dim">No maintenance jobs.</td></tr>}
              {snap.jobs.slice().sort((a, b) => a.start - b.start).map((j) => (
                <tr key={j.id} className="cursor-pointer" onClick={() => set({ selected: j.asset })} title={j.reason}>
                  <td>{j.label}<div className="text-[10.5px] text-dim truncate max-w-[200px]">{j.resource_label} · {j.reason}</div></td>
                  <td className="num text-muted">{fmt.clock(j.start)}–{fmt.hhmm(j.end)}</td>
                  <td className="num text-right">{fmt.usd(j.value_per_h)}/h</td>
                  <td><span className={`chip ${j.status === "active" ? "!text-info !border-info/40" : j.status === "done" ? "!text-good !border-good/40" : ""}`}>{j.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

export function Overview() {
  const cfg = useStore((s) => s.meta?.kpi);
  return cfg ? <OverviewGeneric cfg={cfg} /> : <OverviewOilGas />;
}

/** Overview driven by the scenario's KPI configuration (production metrics, unit cost, secondary metric). */
function OverviewGeneric({ cfg }: { cfg: KpiConfig }) {
  const snap = useStore((s) => s.snap)!;
  const hist = useStore((s) => s.kpiHistory);
  const planVersion = useStore((s) => s.planVersion);
  const [plan, setPlan] = useState<any>(null);
  useEffect(() => { api.plan().then(setPlan).catch(() => {}); }, [planVersion]);
  const k = snap.kpis;
  const P = k.plyxis, L = k.legacy;
  const M = (key?: string) => cfg.metrics.find((m) => m.key === key);
  const prim = M(cfg.primary) ?? cfg.metrics[0];
  const prod = M(cfg.product);
  const sec = M(cfg.secondary);
  const cum = (w: typeof P, key: string) => (w.cum as any)[`m_${key}`] ?? NaN;
  const rate = (w: typeof P, key: string) => (w.rate as any)[`m_${key}`] ?? NaN;
  const perUnit = (w: typeof P) => (cum(w, prim.key) > 1 ? w.cum.cost / cum(w, prim.key) : NaN);
  const intensity = (w: typeof P) => (cum(w, prim.key) > 1 ? w.cum.emission_kg / cum(w, prim.key) : NaN);
  const digits = (u: string) => (/t\/h|t\/d|MW/.test(u) ? 1 : 0);

  const prodOption = useMemo(() => {
    const series = plan?.plan?.series;
    const ps = cfg.plan_series ? series?.[cfg.plan_series] : null;
    const fc = ps?.values?.map((v: number, i: number) => [series.hours[i], v]) ?? [];
    const key = `m_${prim.key}`;
    return base({
      grid: { left: 54, right: 16, top: 30, bottom: 24 },
      xAxis: { type: "value", min: "dataMin", axisLabel: { color: C.axis, fontSize: 10, formatter: (v: number) => fmt.clock(v) }, splitLine: { show: false } },
      yAxis: { type: "value", scale: true, axisLabel: { color: C.axis, fontSize: 10 }, splitLine: { lineStyle: { color: C.grid } } },
      tooltip: { trigger: "axis", valueFormatter: (v: any) => fmt.n(Number(v)) },
      series: [
        { name: "PLYXIS (actual)", type: "line", showSymbol: false, data: hist.map((h) => [h.t, (h.p as any)[key] ?? null]), lineStyle: { width: 1.6, color: C.plx }, itemStyle: { color: C.plx } },
        { name: "Manual ops (actual)", type: "line", showSymbol: false, data: hist.map((h) => [h.t, (h.l as any)[key] ?? null]), lineStyle: { width: 1.4, color: C.leg }, itemStyle: { color: C.leg } },
        { name: `PLYXIS plan (${fc.length} h)`, type: "line", showSymbol: false, data: fc, lineStyle: { width: 1.4, color: C.opt, type: "dashed", opacity: 0.85 }, itemStyle: { color: C.opt } },
      ],
    });
  }, [hist, plan, prim.key, cfg.plan_series]);

  const valueOption = useMemo(() => base({
    grid: { left: 60, right: 16, top: 30, bottom: 24 },
    xAxis: { type: "value", min: "dataMin", axisLabel: { color: C.axis, fontSize: 10, formatter: (v: number) => fmt.clock(v) }, splitLine: { show: false } },
    yAxis: { type: "value", axisLabel: { color: C.axis, fontSize: 10, formatter: (v: number) => fmt.usd(v) }, splitLine: { lineStyle: { color: C.grid } } },
    tooltip: { trigger: "axis", valueFormatter: (v: any) => fmt.usd(Number(v)) },
    series: [
      { name: "Value captured (cumulative profit difference)", type: "line", showSymbol: false, data: hist.map((h) => [h.t, h.pc - h.lc]),
        lineStyle: { width: 2, color: C.plx }, areaStyle: { color: { type: "linear", x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: "#fe026540" }, { offset: 1, color: "#fe026500" }] } }, itemStyle: { color: C.plx } },
    ],
  }), [hist]);

  const secOption = useMemo(() => {
    if (!sec) return null;
    const key = `m_${sec.key}`;
    return base({
      grid: { left: 54, right: 16, top: 30, bottom: 24 },
      xAxis: { type: "value", min: "dataMin", axisLabel: { color: C.axis, fontSize: 10, formatter: (v: number) => fmt.clock(v) }, splitLine: { show: false } },
      yAxis: { type: "value", scale: true, axisLabel: { color: C.axis, fontSize: 10 }, splitLine: { lineStyle: { color: C.grid } } },
      tooltip: { trigger: "axis", valueFormatter: (v: any) => fmt.n(Number(v), 1) },
      series: [
        { name: `PLYXIS`, type: "line", showSymbol: false, data: hist.map((h) => [h.t, (h.p as any)[key] ?? null]), lineStyle: { width: 1.4, color: C.plx }, itemStyle: { color: C.plx } },
        { name: `Manual ops`, type: "line", showSymbol: false, data: hist.map((h) => [h.t, (h.l as any)[key] ?? null]), lineStyle: { width: 1.4, color: C.leg }, itemStyle: { color: C.leg } },
      ],
    });
  }, [hist, sec?.key]);

  return (
    <div className="h-full overflow-auto p-3">
      <div className="grid grid-cols-12 gap-3">
        <div className="col-span-4 panel hero p-4 flex flex-col justify-between">
          <div>
            <div className="text-[10.5px] uppercase tracking-wider text-brand2 font-semibold flex items-center gap-1.5"><TrendingUp size={13} />Value captured vs. manual operations</div>
            <div className="display text-[44px] leading-none font-bold mt-2 brand-text tabular-nums">{fmt.usd(k.delta.profit_after_carbon ?? k.delta.profit)}</div>
            <div className="text-[12px] text-muted mt-1.5">profit after carbon cost · same site, events and physics, over <span className="num">{fmt.n(k.hours, 1)}</span> h</div>
            <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2.5 text-[12px]">
              <Delta pct={rel(P.cum.profit, L.cum.profit)} label="operating profit" />
              <Delta pct={rel(P.cum.cost, L.cum.cost)} label="operating cost" better="lower" />
              <Delta pct={rel(cum(P, prim.key), cum(L, prim.key))} label={prim.label.toLowerCase()} />
            </div>
          </div>
          <div className="grid grid-cols-3 gap-2 mt-4">
            <div><div className="text-[10px] text-dim uppercase">operating profit</div><div className="num text-[16px] font-semibold text-plx">{fmt.usd(k.delta.profit)}</div></div>
            <div><div className="text-[10px] text-dim uppercase">CO2e avoided</div><div className="num text-[16px] font-semibold text-plx">{fmt.n(-k.delta.emission_t, 0)} t</div></div>
            <div><div className="text-[10px] text-dim uppercase">value per day</div><div className="num text-[16px] font-semibold text-plx">{fmt.usd(k.value_per_day)}</div></div>
          </div>
        </div>
        <div className="col-span-8 grid grid-cols-4 gap-3">
          <KpiAB label="Operating profit" plx={P.cum.profit} leg={L.cum.profit} format={(v) => fmt.usd(v)} hint="since start: sales − cash cost + change in inventories" />
          <KpiAB label="Operating cost" plx={P.cum.cost} leg={L.cum.cost} better="lower" format={(v) => fmt.usd(v)} hint="since start: all cash operating costs incl. labour and overheads" />
          <KpiAB label={cfg.unit_cost_label ?? `Cost per ${prim.unit}`} plx={perUnit(P)} leg={perUnit(L)} unit={`$/${prim.unit}`} better="lower" digits={2} hint="cash operating cost per unit of the primary product, since start" />
          <KpiAB label="Revenue" plx={P.cum.revenue} leg={L.cum.revenue} format={(v) => fmt.usd(v)} />
          <KpiAB label={prim.label} plx={rate(P, prim.key)} leg={rate(L, prim.key)} unit={prim.rate_unit} digits={0} hint="current rate, smoothed over ~1.5 h" />
          {prod ? <KpiAB label={prod.label} plx={cum(P, prod.key)} leg={cum(L, prod.key)} unit={prod.unit} digits={digits(prod.unit)} hint="since start" />
            : <KpiAB label={`${prim.label} (total)`} plx={cum(P, prim.key)} leg={cum(L, prim.key)} unit={prim.unit} />}
          {sec ? <KpiAB label={sec.label} plx={cum(P, sec.key)} leg={cum(L, sec.key)} unit={sec.unit} better={sec.better ?? "higher"} hint="since start" />
            : <KpiAB label={`${prim.label} (total)`} plx={cum(P, prim.key)} leg={cum(L, prim.key)} unit={prim.unit} />}
          <KpiAB label="Emissions intensity" plx={intensity(P)} leg={intensity(L)} unit={`kgCO2e/${prim.unit}`} better="lower" digits={1} />
        </div>

        <div className="col-span-8 panel">
          <div className="panel-h">{prim.label}, {prim.rate_unit}: actual (both worlds) and current plan</div>
          <EChart height={250} option={prodOption} />
        </div>
        <div className="col-span-4 panel flex flex-col" style={{ height: 286 }}>
          <div className="panel-h">Autonomous decisions & events</div>
          <div className="flex-1 min-h-0"><Timeline limit={80} /></div>
        </div>

        <div className="col-span-4 panel">
          <div className="panel-h">Value captured over time, USD</div>
          <EChart height={220} option={valueOption} />
        </div>
        <div className="col-span-4 panel">
          <div className="panel-h"><span className="flex items-center gap-1.5"><Waypoints size={12} />Where the value comes from</span>
            <span className="normal-case tracking-normal font-normal text-dim">PLYXIS − manual</span></div>
          <BridgeChart rows={k.bridge ?? []} height={220} />
        </div>
        <div className="col-span-4 panel">
          <div className="panel-h">{sec ? `${sec.label}, ${sec.rate_unit}` : "Production"}</div>
          {secOption ? <EChart height={220} option={secOption} /> : <div className="h-[220px]" />}
        </div>
        <LowerRow plan={plan} />
      </div>
    </div>
  );
}
