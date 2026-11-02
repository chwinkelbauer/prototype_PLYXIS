import { useEffect, useState } from "react";
import { AlertTriangle, Loader2, Play, RotateCcw } from "lucide-react";
import type { ScenarioInfo } from "./api";
import { api } from "./api";
import { AssetDrawer } from "./components/AssetDrawer";
import { DemoGuide } from "./components/DemoGuide";
import { SideNav } from "./components/SideNav";
import { TopBar } from "./components/TopBar";
import type { View } from "./store";
import { useStore } from "./store";
import { BenchPage } from "./views/BenchPage";
import { DataPage } from "./views/DataPage";
import { MapPage } from "./views/MapPage";
import { Mine3DPage } from "./views/Mine3DPage";
import { OptimizerPage } from "./views/OptimizerPage";
import { Overview } from "./views/Overview";
import { SchedulePage } from "./views/SchedulePage";
import { ShiftPage } from "./views/ShiftPage";
import { StudioPage } from "./views/StudioPage";

function HoldBanner() {
  const optRun = useStore((s) => s.optRun);
  const recs = optRun?.recs ?? [];
  const last = recs[recs.length - 1];
  return (
    <div className="absolute top-3 left-1/2 -translate-x-1/2 z-40 panel px-4 py-2 flex items-center gap-3 shadow-xl shadow-black/60 fade-in"
      style={{ background: "#16101ff2", borderColor: "#fe026566" }}>
      <Loader2 size={15} className="animate-spin text-brand" />
      <div className="text-[12.5px]">
        <span className="font-semibold text-ink">Re-planning the whole field</span>
        <span className="text-muted"> · {optRun?.trigger ?? "disruption"} · stage {String(last?.stage ?? "…")}</span>
        <div className="text-[11px] text-dim">plant clock held - a {">"}10 s solve is instantaneous for the field, not at 3600x time compression</div>
      </div>
    </div>
  );
}

/** The scenario opens paused so the guided demo can start at its first step: say so, zeros alone read as broken. */
function StartBar() {
  const snap = useStore((s) => s.snap);
  const start = useStore((s) => s.meta?.start_hour ?? 0);
  if (!snap || snap.running || snap.t > start + 1e-6) return null;
  return (
    <div className="shrink-0 border-b border-line bg-panel2 px-3 py-2 flex items-center gap-3">
      <div className="min-w-0 text-[12.5px]">
        <span className="font-semibold text-ink">Paused at {snap.clock}.</span>
        <span className="text-muted"> The first plan is ready. Values build up in both worlds as soon as the plant clock runs.</span>
      </div>
      <button className="btn btn-primary shrink-0" onClick={() => api.control("play")}><Play size={14} />Start</button>
    </div>
  );
}

/** A start that failed: the reason, and a way out that does not need a server restart. */
function BootFailed({ message }: { message: string }) {
  const [list, setList] = useState<ScenarioInfo[]>([]);
  useEffect(() => { api.scenarios().then(setList).catch(() => {}); }, []);
  const start = async (id: string) => {
    try { await api.selectScenario(id); } finally { location.reload(); }
  };
  return (
    <div className="h-full flex flex-col items-center justify-center gap-3 px-4">
      <img src="/logo.svg" alt="" width={44} height={40} />
      <div className="flex items-center gap-2 text-[14px] font-semibold text-ink"><AlertTriangle size={16} className="text-crit" />PLYXIS could not start</div>
      <div className="panel px-3 py-2 num text-[12px] text-muted max-w-2xl break-words">{message}</div>
      <div className="text-[11px] text-dim">The full traceback is in the terminal running the server.</div>
      <div className="flex flex-wrap justify-center gap-2 mt-1">
        {list.map((sc) => (
          <button key={sc.id} className={`btn ${sc.active ? "btn-primary" : ""}`} onClick={() => start(sc.id)}>
            <RotateCcw size={13} />{sc.active ? "Retry" : "Start"} {sc.title}
          </button>
        ))}
      </div>
    </div>
  );
}

function StartingScreen() {
  const booting = useStore((s) => s.booting);
  const bootError = useStore((s) => s.bootError);
  const loadError = useStore((s) => s.loadError);
  const connected = useStore((s) => s.connected);
  const optRun = useStore((s) => s.optRun);
  const [waited, setWaited] = useState(0);
  useEffect(() => {
    const t0 = Date.now();
    const id = setInterval(() => setWaited(Math.round((Date.now() - t0) / 1000)), 1000);
    return () => clearInterval(id);
  }, []);
  if (bootError) return <BootFailed message={bootError} />;
  const stage = optRun?.recs[optRun.recs.length - 1]?.stage;
  const notice = !connected && waited >= 5
    ? <>No connection to the PLYXIS server at <span className="num">{location.host}</span>. Start it with <span className="num">python -m plyxis serve --port 8000</span>; this page reconnects by itself.</>
    : connected && !booting && waited >= 10
      ? <>The server is connected but no data arrives{loadError ? <>: <span className="num">{loadError}</span></> : ""}. Its terminal shows more.</>
      : null;
  return (
    <div className="h-full flex flex-col items-center justify-center gap-3 text-muted px-4">
      <img src="/logo.svg" alt="" width={44} height={40} />
      <Loader2 className="animate-spin text-brand" size={22} />
      <div className="text-[13px] text-center">PLYXIS is starting: {booting ?? "waiting for first snapshot"}{stage ? ` · stage ${stage}` : ""}… <span className="num text-dim">{waited} s</span></div>
      <div className="text-[11px] text-dim max-w-md text-center">Compiling the PLX twin, building the plant emulators,
        loading neural sub-solvers and computing the first plan.</div>
      {notice && <div className="text-[12px] text-muted max-w-xl text-center flex items-start gap-2"><AlertTriangle size={14} className="text-warn shrink-0 mt-0.5" /><span>{notice}</span></div>}
    </div>
  );
}

export default function App() {
  const view = useStore((s) => s.view);
  const booting = useStore((s) => s.booting);
  const bootError = useStore((s) => s.bootError);
  const snap = useStore((s) => s.snap);
  const meta = useStore((s) => s.meta);
  const structureVersion = useStore((s) => s.structureVersion);
  const set = useStore((s) => s.set);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const h = await api.health();
        if (!alive) return;
        if (h.error) { set({ booting: null, bootError: h.error }); setTimeout(poll, 2000); return; }
        if (h.booting) { set({ booting: h.message, bootError: null }); setTimeout(poll, 1200); return; }
        const [meta, structure] = await Promise.all([api.meta(), api.structure()]);
        if (!alive) return;
        set({ meta, structure, booting: null, bootError: null, loadError: null });
      } catch (e) {
        if (alive) { set({ loadError: String(e) }); setTimeout(poll, 1500); }
      }
    };
    poll();
    return () => { alive = false; };
  }, [set]);

  // a link to a page (#shift, #data, ...) works on an open console too
  useEffect(() => {
    const onHash = () => {
      const v = location.hash.slice(1);
      if (v && v !== useStore.getState().view) set({ view: v as View });
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, [set]);

  useEffect(() => {
    if (structureVersion === 0) return;
    // also after a server restart behind an open page, which may run another case study
    Promise.all([api.meta(), api.structure()]).then(([meta, structure]) => set({ meta, structure })).catch(() => {});
  }, [structureVersion, set]);

  // the push channel can fail on its own (a blocked WebSocket, a dead ticker): keep the numbers coming over REST
  useEffect(() => {
    const id = setInterval(async () => {
      const s = useStore.getState();
      if (s.booting || s.bootError || !s.meta || Date.now() - s.lastSnapAt < 3000) return;
      try { s.ingest(await api.snapshot(!s.snap)); } catch (e) { s.set({ loadError: String(e) }); }
    }, 1000);
    return () => clearInterval(id);
  }, []);

  if (bootError || booting || !snap || !meta) return <StartingScreen />;
  return (
    <div className="h-full flex flex-col">
      <TopBar />
      <div className="flex-1 flex min-h-0">
        <SideNav />
        <main className="flex-1 min-w-0 flex flex-col overflow-hidden">
          <StartBar />
          <div className="flex-1 min-h-0 relative overflow-hidden">
            {view === "overview" && <Overview />}
            {view === "map" && <MapPage />}
            {view === "mine3d" && <Mine3DPage />}
            {view === "schedule" && <SchedulePage />}
            {view === "shift" && <ShiftPage />}
            {view === "optimizer" && <OptimizerPage />}
            {view === "studio" && <StudioPage />}
            {view === "data" && <DataPage />}
            {view === "bench" && <BenchPage />}
            {snap.holding && <HoldBanner />}
          </div>
          <DemoGuide />
        </main>
        <AssetDrawer />
      </div>
    </div>
  );
}
