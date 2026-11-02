import { useEffect, useRef, useState } from "react";
import { Activity, BrainCircuit, Check, ChevronDown, Cpu, Database, GitCommitHorizontal, Loader2, Pause, Play, Presentation, RotateCcw, Zap } from "lucide-react";
import type { ScenarioInfo } from "../api";
import { api } from "../api";
import { useStore } from "../store";

const SPEEDS = [60, 300, 600, 1800, 3600];

export function Logo() {
  return (
    <div className="flex items-center gap-2.5 shrink-0">
      <img src="/logo.svg" alt="PLYXIS" width={30} height={27} className="block" />
      <div className="leading-none max-[1180px]:hidden">
        <div className="display text-[16px] font-bold tracking-[0.22em] text-ink">PLYXIS</div>
        <div className="text-[9px] tracking-[0.16em] text-dim uppercase mt-1 whitespace-nowrap hidden min-[1800px]:block">Operating system for industries</div>
      </div>
    </div>
  );
}

/**
 * Three zones. Only the left one (logo, case study) gives way when the window is narrow: its text truncates. The clock
 * zone and the status zone never shrink, they lose labels instead, in this order: speed buttons -> a menu, chip labels ->
 * their icons. So nothing can slide under the clock, however narrow the window (or high the display scaling).
 */
export function TopBar() {
  const snap = useStore((s) => s.snap);
  const connected = useStore((s) => s.connected);
  const optRun = useStore((s) => s.optRun);
  const set = useStore((s) => s.set);
  const demoOpen = useStore((s) => s.demoOpen);
  const running = snap?.running ?? false;
  const solving = snap?.optimizer.running || optRun?.running;
  const lastStage = optRun?.recs[optRun.recs.length - 1]?.stage;
  const crit = useStore((s) => s.alerts.filter((a) => a.severity === "critical" && snap && snap.t - a.t < 2).length);
  return (
    <header className="relative h-12 shrink-0 flex items-center gap-3 min-[1280px]:gap-4 px-3 bg-panel">
      <div className="brand-line absolute left-0 right-0 bottom-0" />
      <div className="flex-1 min-w-0 flex items-center gap-3 min-[1280px]:gap-4">
        <Logo />
        <div className="h-6 w-px bg-line shrink-0" />
        <ScenarioSwitch />
      </div>
      <div className="shrink-0 flex items-center gap-1.5">
        <div className="display text-[16px] font-semibold text-ink whitespace-nowrap min-w-[96px] text-right tabular-nums">{snap ? snap.clock : "—"}</div>
        <button className="btn" onClick={() => api.control(running ? "pause" : "play")} title={running ? "Pause" : "Run"}>
          {running ? <Pause size={14} /> : <Play size={14} />}
        </button>
        <div className="flex rounded-md border border-line2 overflow-hidden max-[1320px]:hidden">
          {SPEEDS.map((s) => (
            <button key={s} onClick={() => api.control("speed", s)}
              className={`px-2 py-1 text-[11px] num ${snap?.speed === s ? "seg-active" : "bg-panel2 text-muted hover:text-ink"}`}>
              {speedLabel(s)}
            </button>
          ))}
        </div>
        <select className="inp num text-[11px] !py-1 min-[1321px]:hidden" title="Plant clock speed" value={snap?.speed ?? 600}
          onChange={(e) => api.control("speed", Number(e.target.value))}>
          {SPEEDS.map((s) => <option key={s} value={s}>{speedLabel(s)}</option>)}
        </select>
        <button className="btn btn-ghost" title="Reset scenario" onClick={() => api.control("reset")}><RotateCcw size={14} /></button>
      </div>
      <div className="h-6 w-px bg-line shrink-0 max-[900px]:hidden" />
      <div className="shrink-0 flex items-center gap-1.5">
        <button className="chip !border-brand/50 !text-[#ffd1e3] capitalize" title={`Operating mode: ${snap?.advisory?.mode ?? "autonomous"}`} onClick={() => set({ view: "shift" })}>
          <Zap size={11} className="text-brand" /><span className="max-[1500px]:hidden">{snap?.advisory?.mode ?? "autonomous"}</span>
        </button>
        <button className="chip hover:text-ink" onClick={() => set({ view: "optimizer" })} title={solving ? `Solving${lastStage ? `: ${lastStage}` : ""}` : `Optimizer: plan #${snap?.optimizer.plan_id ?? "–"}`}>
          <Cpu size={11} className={solving ? "text-opt pulse" : ""} />
          {solving ? <span className="text-opt max-[1500px]:hidden">solving{lastStage ? ` · ${lastStage}` : ""}</span> : <span className="max-[1100px]:hidden">plan #{snap?.optimizer.plan_id ?? "–"}</span>}
        </button>
        <button className="chip hover:text-ink max-[780px]:hidden" onClick={() => set({ view: "studio" })} title={`Twin version ${snap?.twin.version ?? "–"}`}>
          <GitCommitHorizontal size={11} /><span className="max-[1100px]:hidden">twin v{snap?.twin.version ?? "–"}</span>
        </button>
        <button className="chip hover:text-ink" onClick={() => set({ view: "data" })} title={connected ? "Data integration: live" : "Data integration: offline"}>
          <Database size={11} className={connected ? "text-good" : "text-crit"} /><span className="max-[1500px]:hidden">{connected ? "live" : "offline"}</span>
        </button>
        {(snap?.optimizer.neural as any)?.loaded && (
          <span className="chip max-[1500px]:hidden" title="Neural sub-solvers loaded"><BrainCircuit size={11} className="text-opt" />neural</span>
        )}
        {crit > 0 && <span className="chip !border-crit/50 !text-crit pulse" title={`${crit} critical alerts`}><Activity size={11} />{crit}<span className="max-[1500px]:hidden"> critical</span></span>}
        <button className={`btn ${demoOpen ? "btn-primary" : ""}`} onClick={() => set({ demoOpen: !demoOpen })} title="Guided demo">
          <Presentation size={14} /><span className="max-[1100px]:hidden">Demo</span>
        </button>
      </div>
    </header>
  );
}

const speedLabel = (s: number) => (s >= 3600 ? `${s / 3600}h/s` : `${s / 60}m/s`);

/** Case-study selector: the server stops the running twin and boots the chosen one. */
function ScenarioSwitch() {
  const meta = useStore((s) => s.meta);
  const [open, setOpen] = useState(false);
  const [list, setList] = useState<ScenarioInfo[]>([]);
  const [switching, setSwitching] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => { if (open) api.scenarios().then(setList).catch(() => {}); }, [open]);
  useEffect(() => {
    const close = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    window.addEventListener("mousedown", close);
    return () => window.removeEventListener("mousedown", close);
  }, []);
  const pick = async (sc: ScenarioInfo) => {
    if (sc.active) { setOpen(false); return; }
    setSwitching(sc.title);
    try { await api.selectScenario(sc.id); } catch { setSwitching(null); return; }
    setTimeout(() => location.reload(), 300);
  };
  return (
    <div ref={ref} className="relative min-w-[64px]">
      <button className="w-full max-w-[460px] text-left rounded-md px-1.5 py-0.5 hover:bg-panel2 flex items-center gap-2" onClick={() => setOpen(!open)}
        title={`Switch case study (${meta?.title ?? ""})`}>
        <div className="min-w-0 flex-1 overflow-hidden">
          <div className="text-[12.5px] font-semibold truncate">{meta?.title ?? "…"} {meta?.region && <span className="text-dim font-normal">· {meta.region}</span>}</div>
          <div className="text-[10.5px] text-dim truncate">{meta?.subtitle ?? ""}</div>
        </div>
        {switching ? <Loader2 size={13} className="animate-spin text-brand shrink-0" /> : <ChevronDown size={13} className="text-dim shrink-0" />}
      </button>
      {open && (
        <div className="absolute left-0 top-[calc(100%+6px)] z-50 w-[440px] max-w-[calc(100vw-24px)] panel shadow-2xl shadow-black/70 fade-in" style={{ background: "#12121af8" }}>
          <div className="panel-h">Case study</div>
          {list.map((sc) => (
            <button key={sc.id} onClick={() => pick(sc)} disabled={switching !== null}
              className={`w-full text-left px-3 py-2.5 border-b border-line last:border-0 hover:bg-panel2 flex items-start gap-2.5 ${sc.active ? "bg-panel2" : ""}`}>
              <span className="w-4 pt-0.5 shrink-0">{sc.active && <Check size={14} className="text-brand" />}</span>
              <span className="min-w-0">
                <span className="block text-[13px] font-semibold">{sc.title} <span className="text-dim font-normal">· {sc.region}</span></span>
                <span className="block text-[11px] text-dim leading-snug mt-0.5">{sc.subtitle}</span>
              </span>
            </button>
          ))}
          {switching && <div className="px-3 py-2 text-[11.5px] text-muted flex items-center gap-2"><Loader2 size={12} className="animate-spin" />starting {switching} - compiling the twin and the plant…</div>}
        </div>
      )}
    </div>
  );
}
