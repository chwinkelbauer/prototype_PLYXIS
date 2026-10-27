import { Activity, BrainCircuit, Cpu, Database, GitCommitHorizontal, Pause, Play, Presentation, RotateCcw, Zap } from "lucide-react";
import { api, fmt } from "../api";
import { useStore } from "../store";

const SPEEDS = [60, 300, 600, 1800, 3600];

export function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <img src="/logo.svg" alt="" width={30} height={27} className="block" />
      <div className="leading-none">
        <div className="display text-[16px] font-bold tracking-[0.22em] text-ink">PLYXIS</div>
        <div className="text-[9px] tracking-[0.16em] text-dim uppercase mt-1">Operating system for industries</div>
      </div>
    </div>
  );
}

export function TopBar() {
  const snap = useStore((s) => s.snap);
  const connected = useStore((s) => s.connected);
  const optRun = useStore((s) => s.optRun);
  const meta = useStore((s) => s.meta);
  const set = useStore((s) => s.set);
  const demoOpen = useStore((s) => s.demoOpen);
  const running = snap?.running ?? false;
  const solving = snap?.optimizer.running || optRun?.running;
  const lastStage = optRun?.recs[optRun.recs.length - 1]?.stage;
  const crit = useStore((s) => s.alerts.filter((a) => a.severity === "critical" && snap && snap.t - a.t < 2).length);
  return (
    <header className="relative h-12 shrink-0 flex items-center gap-4 px-3 bg-panel">
      <div className="brand-line absolute left-0 right-0 bottom-0" />
      <Logo />
      <div className="h-6 w-px bg-line" />
      <div className="min-w-0">
        <div className="text-[12.5px] font-semibold truncate">Kestrel Field <span className="text-dim font-normal">· Delaware Basin</span></div>
        <div className="text-[10.5px] text-dim truncate">synthetic demonstration asset · 96 wells · 3 batteries · 2 compressor stations</div>
      </div>
      <div className="flex-1" />
      <div className="flex items-center gap-1.5">
        <div className="display text-[16px] font-semibold text-ink w-[110px] text-right tabular-nums">{snap ? snap.clock : "—"}</div>
        <button className="btn" onClick={() => api.control(running ? "pause" : "play")} title={running ? "Pause" : "Run"}>
          {running ? <Pause size={14} /> : <Play size={14} />}
        </button>
        <div className="flex rounded-md border border-line2 overflow-hidden">
          {SPEEDS.map((s) => (
            <button key={s} onClick={() => api.control("speed", s)}
              className={`px-2 py-1 text-[11px] num ${snap?.speed === s ? "seg-active" : "bg-panel2 text-muted hover:text-ink"}`}>
              {s >= 3600 ? `${s / 3600}h/s` : `${s / 60}m/s`}
            </button>
          ))}
        </div>
        <button className="btn btn-ghost" title="Reset scenario" onClick={() => api.control("reset")}><RotateCcw size={14} /></button>
      </div>
      <div className="h-6 w-px bg-line" />
      <div className="flex items-center gap-1.5">
        <span className="chip !border-brand/50 !text-[#ffd1e3]"><Zap size={11} className="text-brand" />Autonomous</span>
        <button className="chip hover:text-ink" onClick={() => set({ view: "optimizer" })} title="Optimizer">
          <Cpu size={11} className={solving ? "text-opt pulse" : ""} />
          {solving ? <span className="text-opt">solving{lastStage ? ` · ${lastStage}` : ""}</span> : <>plan #{snap?.optimizer.plan_id ?? "–"}</>}
        </button>
        <button className="chip hover:text-ink" onClick={() => set({ view: "studio" })} title="Twin version">
          <GitCommitHorizontal size={11} />twin v{snap?.twin.version ?? "–"}
        </button>
        <button className="chip hover:text-ink" onClick={() => set({ view: "data" })} title="Data integration">
          <Database size={11} className={connected ? "text-good" : "text-crit"} />{connected ? "live" : "offline"}
        </button>
        {(snap?.optimizer.neural as any)?.loaded && (
          <span className="chip" title="Neural sub-solvers loaded"><BrainCircuit size={11} className="text-opt" />neural</span>
        )}
        {crit > 0 && <span className="chip !border-crit/50 !text-crit pulse"><Activity size={11} />{crit} critical</span>}
        <button className={`btn ${demoOpen ? "btn-primary" : ""}`} onClick={() => set({ demoOpen: !demoOpen })} title="Guided demo">
          <Presentation size={14} />Demo
        </button>
      </div>
      {meta && null}
      <span className="hidden">{fmt.n(0)}</span>
    </header>
  );
}
