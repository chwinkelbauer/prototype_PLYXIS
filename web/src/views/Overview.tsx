import { useEffect, useMemo, useState } from "react";
import { Leaf, Scale, TrendingUp } from "lucide-react";
import { api, fmt } from "../api";
import { base, C, EChart } from "../components/Chart";
import { AlertFeed, Timeline } from "../components/Feed";
import { KpiAB } from "../components/Kpi";
import { useStore } from "../store";

export function Overview() {
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

  const prodOption = useMemo(() => {
    const series = plan?.plan?.series;
    const fc = series?.field_oil?.values?.map((v: number, i: number) => [series.hours[i], v]) ?? [];
    return base({
      grid: { left: 54, right: 16, top: 30, bottom: 24 },
      xAxis: { type: "value", min: "dataMin", axisLabel: { color: C.axis, fontSize: 10, formatter: (v: number) => fmt.clock(v) }, splitLine: { show: false } },
      yAxis: { type: "value", scale: true, name: "bopd", nameTextStyle: { color: C.axis, fontSize: 10 }, axisLabel: { color: C.axis, fontSize: 10 }, splitLine: { lineStyle: { color: C.grid } } },
      tooltip: { trigger: "axis", valueFormatter: (v: any) => fmt.n(Number(v)) },
      series: [
        { name: "PLYXIS (actual)", type: "line", showSymbol: false, data: hist.map((h) => [h.t, h.p.oil_bpd ?? null]), lineStyle: { width: 1.6, color: C.plx }, itemStyle: { color: C.plx } },
        { name: "Manual ops (actual)", type: "line", showSymbol: false, data: hist.map((h) => [h.t, h.l.oil_bpd ?? null]), lineStyle: { width: 1.4, color: C.leg }, itemStyle: { color: C.leg } },
        { name: "PLYXIS plan (72 h)", type: "line", showSymbol: false, data: fc, lineStyle: { width: 1.2, color: C.plx, type: "dashed", opacity: 0.6 }, itemStyle: { color: C.plx } },
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
        lineStyle: { width: 1.8, color: C.teal }, areaStyle: { color: "#2dd4bf18" }, itemStyle: { color: C.teal } },
    ],
  }), [hist]);

  const flareOption = useMemo(() => base({
    grid: { left: 54, right: 16, top: 30, bottom: 24 },
    xAxis: { type: "value", min: "dataMin", axisLabel: { color: C.axis, fontSize: 10, formatter: (v: number) => fmt.clock(v) }, splitLine: { show: false } },
    yAxis: { type: "value", name: "Mscf/d", nameTextStyle: { color: C.axis, fontSize: 10 }, axisLabel: { color: C.axis, fontSize: 10 }, splitLine: { lineStyle: { color: C.grid } } },
    tooltip: { trigger: "axis", valueFormatter: (v: any) => fmt.n(Number(v)) },
    series: [
      { name: "PLYXIS flaring", type: "line", showSymbol: false, data: hist.map((h) => [h.t, h.p.flare_mscfd ?? 0]), lineStyle: { width: 1.4, color: C.plx }, itemStyle: { color: C.plx }, areaStyle: { color: "#34d39914" } },
      { name: "Manual ops flaring", type: "line", showSymbol: false, data: hist.map((h) => [h.t, h.l.flare_mscfd ?? 0]), lineStyle: { width: 1.4, color: C.leg }, itemStyle: { color: C.leg }, areaStyle: { color: "#f59e0b14" } },
    ],
  }), [hist]);

  const bn = plan?.plan?.bottlenecks ?? [];
  return (
    <div className="h-full overflow-auto p-3">
      <div className="grid grid-cols-12 gap-3">
        {/* value hero */}
        <div className="col-span-4 panel p-4 flex flex-col justify-between" style={{ background: "linear-gradient(135deg,#0d2a28 0%,#0c131b 60%)" }}>
          <div>
            <div className="text-[10.5px] uppercase tracking-wider text-teal font-semibold flex items-center gap-1.5"><TrendingUp size={13} />Value captured vs. manual operations</div>
            <div className="num text-[40px] leading-none font-bold mt-2 text-ink">{fmt.usd(k.delta.profit_after_carbon ?? k.delta.profit)}</div>
            <div className="text-[12px] text-muted mt-1.5">profit after carbon cost · same field, events and physics, over <span className="num">{fmt.n(k.hours, 1)}</span> h</div>
          </div>
          <div className="grid grid-cols-3 gap-2 mt-4">
            <div><div className="text-[10px] text-dim uppercase">operating profit</div><div className="num text-[16px] font-semibold text-plx">{fmt.usd(k.delta.profit)}</div></div>
            <div><div className="text-[10px] text-dim uppercase">CO2e avoided</div><div className="num text-[16px] font-semibold text-plx">{fmt.n(-k.delta.emission_t, 0)} t</div></div>
            <div><div className="text-[10px] text-dim uppercase">extra oil</div><div className="num text-[16px] font-semibold text-plx">{fmt.signed(k.delta.oil_bbl, 0)} bbl</div></div>
          </div>
        </div>
        <div className="col-span-8 grid grid-cols-4 gap-3">
          <KpiAB label="Oil production" plx={P.rate.oil_bpd} leg={L.rate.oil_bpd} unit="bopd" />
          <KpiAB label="Operating profit" plx={P.rate.profit_usd_h} leg={L.rate.profit_usd_h} unit="$/h" format={(v) => fmt.n(v)} />
          <KpiAB label="Flaring" plx={P.rate.flare_mscfd} leg={L.rate.flare_mscfd} unit="Mscf/d" better="lower" />
          <KpiAB label="Emissions intensity" plx={intensity(P)} leg={intensity(L)} unit="kgCO2e/boe" better="lower" digits={1} />
          <KpiAB label="Gas sold" plx={P.rate.gas_mscfd} leg={L.rate.gas_mscfd} unit="Mscf/d" />
          <KpiAB label="Operating cost" plx={P.rate.cost_usd_h} leg={L.rate.cost_usd_h} unit="$/h" better="lower" />
          <KpiAB label="Cumulative oil" plx={P.cum.oil_bbl} leg={L.cum.oil_bbl} unit="bbl" />
          <KpiAB label="Cumulative flaring" plx={P.cum.flared_mscf} leg={L.cum.flared_mscf} unit="Mscf" better="lower" />
        </div>

        <div className="col-span-8 panel">
          <div className="panel-h">Oil production: actual (both worlds) and current 72 h plan</div>
          <EChart height={250} option={prodOption} />
        </div>
        <div className="col-span-4 panel flex flex-col" style={{ height: 286 }}>
          <div className="panel-h">Autonomous decisions & events</div>
          <div className="flex-1 min-h-0"><Timeline limit={80} /></div>
        </div>

        <div className="col-span-4 panel">
          <div className="panel-h">Value captured over time</div>
          <EChart height={200} option={valueOption} />
        </div>
        <div className="col-span-4 panel">
          <div className="panel-h flex"><span className="flex items-center gap-1.5"><Leaf size={12} />Flaring</span></div>
          <EChart height={200} option={flareOption} />
        </div>
        <div className="col-span-4 panel flex flex-col" style={{ height: 236 }}>
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

        <div className="col-span-6 panel flex flex-col" style={{ height: 240 }}>
          <div className="panel-h">Alerts</div>
          <div className="flex-1 min-h-0"><AlertFeed limit={60} /></div>
        </div>
        <div className="col-span-6 panel flex flex-col" style={{ height: 240 }}>
          <div className="panel-h">Maintenance dispatch</div>
          <div className="flex-1 overflow-auto">
            <table className="tbl">
              <thead><tr><th>asset</th><th>crew</th><th>window</th><th className="!text-right">value</th><th>status</th></tr></thead>
              <tbody>
                {snap.jobs.length === 0 && <tr><td colSpan={5} className="text-dim">No maintenance jobs.</td></tr>}
                {snap.jobs.slice().sort((a, b) => a.start - b.start).map((j) => (
                  <tr key={j.id} className="cursor-pointer" onClick={() => set({ selected: j.asset })} title={j.reason}>
                    <td>{j.label}<div className="text-[10.5px] text-dim truncate max-w-[260px]">{j.reason}</div></td>
                    <td className="text-muted">{j.resource_label}</td>
                    <td className="num text-muted">{fmt.clock(j.start)}–{fmt.hhmm(j.end)}</td>
                    <td className="num text-right">{fmt.usd(j.value_per_h)}/h</td>
                    <td><span className={`chip ${j.status === "active" ? "!text-info !border-info/40" : j.status === "done" ? "!text-plx !border-plx/40" : ""}`}>{j.status}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
