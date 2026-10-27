import { useEffect, useMemo, useState } from "react";
import { BrainCircuit, Cpu, Layers3, RefreshCw, Scale, Workflow } from "lucide-react";
import { api, fmt } from "../api";
import { base, C, EChart } from "../components/Chart";
import { Stat } from "../components/Kpi";
import { useStore } from "../store";

const STAGE_COLOR: Record<string, string> = {
  neural: "#fe0265", relaxation: "#3987e5", discrete: "#c97a14", polish: "#8b5cf6", done: "#199e70", compile: "#71717a",
};
const STAGES = [
  { id: "compile", title: "Compile & decompose", text: "PLX netlist → decision space, subsystems, constraint scales" },
  { id: "neural", title: "Neural sub-solvers", text: "graph network proposes every asset's decisions (ms), price-coordinated" },
  { id: "relaxation", title: "Coordinated relaxation", text: "multi-start augmented Lagrangian on the differentiable twin" },
  { id: "discrete", title: "Discrete search", text: "memetic GA: randomised roundings, cooperative recombination, batched repair" },
  { id: "polish", title: "Polish & verify", text: "continuous refinement, exact simulation, shadow prices" },
];

export function OptimizerPage() {
  const planVersion = useStore((s) => s.planVersion);
  const optRun = useStore((s) => s.optRun);
  const [d, setD] = useState<any>(null);
  useEffect(() => {
    let alive = true;
    const load = () => api.optimizer().then((r) => alive && setD(r)).catch(() => {});
    load();
    const id = setInterval(load, 5000);
    return () => { alive = false; clearInterval(id); };
  }, [planVersion]);

  const last = d?.runs?.[d.runs.length - 1];
  const recs = optRun && (optRun.running || !last) ? optRun.recs : (last?.history ?? optRun?.recs ?? []);
  const conv = useMemo(() => {
    const byStage: Record<string, [number, number][]> = {};
    const viol: [number, number][] = [];
    for (const r of recs) {
      if (r.objective === undefined || r.t_total === undefined || r.event) continue;
      (byStage[r.stage] ??= []).push([r.t_total as number, r.objective as number]);
      if (r.max_violation !== undefined) viol.push([r.t_total as number, Math.max(1e-5, r.max_violation as number)]);
    }
    return base({
      grid: { left: 64, right: 56, top: 30, bottom: 30 },
      xAxis: { type: "value", name: "seconds", nameLocation: "end", nameTextStyle: { color: C.axis, fontSize: 10 }, axisLabel: { color: C.axis, fontSize: 10 }, splitLine: { show: false } },
      yAxis: [
        { type: "value", scale: true, axisLabel: { color: C.axis, fontSize: 10, formatter: (v: number) => fmt.usd(v) }, splitLine: { lineStyle: { color: C.grid } } },
        { type: "log", min: 1e-5, axisLabel: { color: C.axis, fontSize: 10 }, splitLine: { show: false }, name: "max violation", nameTextStyle: { color: C.axis, fontSize: 10 } },
      ],
      tooltip: { trigger: "axis" },
      series: [
        ...Object.entries(byStage).map(([st, data]) => ({ name: st, type: "scatter" as const, symbolSize: 4, data, itemStyle: { color: STAGE_COLOR[st] ?? C.teal } })),
        { name: "max violation", type: "line" as const, yAxisIndex: 1, showSymbol: false, data: viol, lineStyle: { width: 1, color: "#ef4444", opacity: 0.7 } },
      ],
    });
  }, [recs]);

  const sizes = d?.space;
  const subs: any[] = d?.subsystems ?? [];
  const maxSize = Math.max(1, ...subs.map((s) => s.size));
  return (
    <div className="h-full overflow-auto p-3">
      <div className="grid grid-cols-12 gap-3">
        <div className="col-span-12 grid grid-cols-6 gap-3">
          <Stat label="Decision variables" icon={<Cpu size={12} />} value={fmt.n(sizes?.variables)} sub={`${fmt.n(sizes?.discrete)} integer / boolean / line-up`} />
          <Stat label="Subsystems" icon={<Layers3 size={12} />} value={fmt.n(subs.length)} sub="automatic decomposition" />
          <Stat label="Last solve" icon={<Workflow size={12} />} value={last ? `${fmt.n(last.seconds, 1)} s` : "–"} sub={last ? `${fmt.n(last.evals)} twin simulations` : ""} />
          <Stat label="Plans computed" icon={<RefreshCw size={12} />} value={fmt.n(d?.runs?.length ?? 0)} sub={last ? `last trigger: ${last.trigger}` : ""} />
          <Stat label="Neural sub-solvers" icon={<BrainCircuit size={12} />} color="text-opt"
            value={d?.neural?.loaded ? "active" : "off"} sub={d?.neural?.loaded ? `${d.neural.updates ?? 0} online updates · ${d.neural.trained_on ?? 0} plans learned` : "fallback: multi-start"} />
          <Stat label="Last objective" icon={<Scale size={12} />} value={last ? fmt.usd(last.objective) : "–"} sub={last ? (last.feasible ? "feasible plan (72 h)" : "best compromise (soft)") : ""} />
        </div>

        <div className="col-span-8 panel">
          <div className="panel-h"><span>Convergence {optRun?.running ? <span className="text-opt pulse normal-case">· solving ({optRun.trigger})</span> : last ? <span className="normal-case">· plan #{last.id} ({last.trigger})</span> : ""}</span>
            <button className="btn !py-0.5" onClick={() => api.replan("manual request from optimizer console")}><RefreshCw size={12} />Re-plan</button></div>
          <EChart height={290} option={conv} />
        </div>
        <div className="col-span-4 panel">
          <div className="panel-h">The PLYXIS cascade</div>
          <div className="p-2.5 space-y-1.5">
            {STAGES.map((s, i) => {
              const st = last?.stages?.find((x: any) => x.stage === s.id);
              return (
                <div key={s.id} className="flex gap-2.5 items-start">
                  <div className="w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold shrink-0 mt-0.5" style={{ background: `${STAGE_COLOR[s.id]}22`, color: STAGE_COLOR[s.id], border: `1px solid ${STAGE_COLOR[s.id]}66` }}>{i}</div>
                  <div className="min-w-0 flex-1">
                    <div className="flex justify-between gap-2"><span className="text-[12.5px] font-semibold">{s.title}</span>
                      {st && <span className="num text-[11px] text-muted">{fmt.n(st.seconds, 2)} s{st.objective !== undefined ? ` · ${fmt.usd(st.objective)}` : ""}</span>}</div>
                    <div className="text-[11.5px] text-dim leading-snug">{s.text}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="col-span-7 panel">
          <div className="panel-h">Decomposition · subsystems coordinated by prices</div>
          <div className="p-2.5 flex flex-wrap gap-1.5">
            {subs.filter((s) => s.size > 0).map((s) => (
              <div key={s.id} className="rounded border border-line2 bg-panel2 px-2 py-1.5" style={{ width: 70 + 80 * (s.size / maxSize) }}
                title={Object.entries(s.types).map(([k, v]) => `${v}× ${k}`).join(", ")}>
                <div className="text-[11.5px] font-semibold truncate">{s.name}</div>
                <div className="text-[10.5px] text-dim truncate">{s.kind.replace(/_/g, " ")} · {s.size}</div>
              </div>
            ))}
          </div>
        </div>
        <div className="col-span-5 panel flex flex-col" style={{ maxHeight: 360 }}>
          <div className="panel-h">Shadow prices of binding constraints</div>
          <div className="flex-1 overflow-auto">
            <table className="tbl">
              <thead><tr><th>bottleneck</th><th className="!text-right">binding</th><th>marginal value</th><th className="!text-right">+10 %/day</th></tr></thead>
              <tbody>
                {(d?.bottlenecks ?? []).map((b: any, i: number) => (
                  <tr key={i}><td>{b.label}</td><td className="num text-right text-muted">{fmt.n(b.hours_binding)} h</td>
                    <td className="num text-[11px] text-muted">{b.max_price ?? "–"}</td><td className="num text-right text-plx">{fmt.usd(b.value_10pct_per_day)}</td></tr>
                ))}
                {!(d?.bottlenecks ?? []).length && <tr><td colSpan={4} className="text-dim">No binding constraints.</td></tr>}
              </tbody>
            </table>
          </div>
        </div>

        <div className="col-span-12 panel">
          <div className="panel-h">Re-planning history</div>
          <table className="tbl">
            <thead><tr><th>#</th><th>time</th><th>trigger</th><th className="!text-right">objective (72 h)</th><th>feasible</th><th className="!text-right">solve</th><th className="!text-right">simulations</th><th>stages</th></tr></thead>
            <tbody>
              {(d?.runs ?? []).slice().reverse().map((r: any) => (
                <tr key={r.id}><td className="num">{r.id}</td><td className="num text-muted">{fmt.clock(r.t)}</td><td>{r.trigger}</td>
                  <td className="num text-right">{fmt.usd(r.objective)}</td><td>{r.feasible ? <span className="text-good">yes</span> : <span className="text-warn">soft</span>}</td>
                  <td className="num text-right">{fmt.n(r.seconds, 1)} s</td><td className="num text-right text-muted">{fmt.n(r.evals)}</td>
                  <td className="text-[11px] text-dim">{(r.stages ?? []).map((s: any) => `${s.stage} ${fmt.n(s.seconds, 1)}s`).join(" · ")}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
