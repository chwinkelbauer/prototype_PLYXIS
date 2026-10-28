import { useEffect, useRef, useState } from "react";
import { Activity, BrainCircuit, Check, ChevronDown, Cpu, Database, GitCommitHorizontal, Loader2, Pause, Play, Presentation, RotateCcw, Zap } from "lucide-react";
import type { ScenarioInfo } from "../api";
import { api, fmt } from "../api";
import { useStore } from "../store";

const SPEEDS = [60, 300, 600, 1800, 3600];

export function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <img src="/logo.svg" alt="" width={30} height={27} className="block" />
      <div className="leading-none">
        <div className="display text-[16px] font-bold tracking-[0.22em] text-ink">PLYXIS</div>
        <div className="text-[9px] tracking-[0.16em] text-dim uppercase mt-1 whitespace-nowrap hidden 2xl:block">Operating system for industries</div>
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
      <ScenarioSwitch />
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
    <div ref={ref} className="relative min-w-0">
      <button className="min-w-0 text-left rounded-md px-1.5 py-0.5 -mx-1.5 hover:bg-panel2 flex items-center gap-2" onClick={() => setOpen(!open)}
        title="Switch case study">
        <div className="min-w-0">
          <div className="text-[12.5px] font-semibold truncate">{meta?.title ?? "…"} {meta?.region && <span className="text-dim font-normal">· {meta.region}</span>}</div>
          <div className="text-[10.5px] text-dim truncate max-w-[420px]">{meta?.subtitle ?? ""}</div>
        </div>
        {switching ? <Loader2 size={13} className="animate-spin text-brand shrink-0" /> : <ChevronDown size={13} className="text-dim shrink-0" />}
      </button>
      {open && (
        <div className="absolute left-0 top-[calc(100%+6px)] z-50 w-[440px] panel shadow-2xl shadow-black/70 fade-in" style={{ background: "#12121af8" }}>
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
