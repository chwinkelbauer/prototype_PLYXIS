import { useEffect } from "react";
import { Loader2 } from "lucide-react";
import { api } from "./api";
import { AssetDrawer } from "./components/AssetDrawer";
import { DemoGuide } from "./components/DemoGuide";
import { SideNav } from "./components/SideNav";
import { TopBar } from "./components/TopBar";
import { useStore } from "./store";
import { BenchPage } from "./views/BenchPage";
import { DataPage } from "./views/DataPage";
import { MapPage } from "./views/MapPage";
import { Mine3DPage } from "./views/Mine3DPage";
import { OptimizerPage } from "./views/OptimizerPage";
import { Overview } from "./views/Overview";
import { SchedulePage } from "./views/SchedulePage";
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

export default function App() {
  const view = useStore((s) => s.view);
  const booting = useStore((s) => s.booting);
  const snap = useStore((s) => s.snap);
  const structureVersion = useStore((s) => s.structureVersion);
  const set = useStore((s) => s.set);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const h = await api.health();
        if (!alive) return;
        if (h.booting) { set({ booting: h.message }); setTimeout(poll, 1200); return; }
        const [meta, structure] = await Promise.all([api.meta(), api.structure()]);
        if (!alive) return;
        set({ meta, structure, booting: null });
      } catch {
        if (alive) setTimeout(poll, 1500);
      }
    };
    poll();
    return () => { alive = false; };
  }, [set]);

  useEffect(() => {
    if (structureVersion === 0) return;
    api.structure().then((structure) => set({ structure })).catch(() => {});
  }, [structureVersion, set]);

  if (booting || !snap) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-3 text-muted">
        <Loader2 className="animate-spin text-teal" size={28} />
        <div className="text-[13px]">PLYXIS is starting: {booting ?? "waiting for first snapshot"}…</div>
        <div className="text-[11px] text-dim max-w-md text-center">Compiling the PLX twin, building the plant emulators,
          loading neural sub-solvers and computing the first plan.</div>
      </div>
    );
  }
  return (
    <div className="h-full flex flex-col">
      <TopBar />
      <div className="flex-1 flex min-h-0">
        <SideNav />
        <main className="flex-1 min-w-0 flex flex-col overflow-hidden">
          <div className="flex-1 min-h-0 relative overflow-hidden">
            {view === "overview" && <Overview />}
            {view === "map" && <MapPage />}
            {view === "mine3d" && <Mine3DPage />}
            {view === "schedule" && <SchedulePage />}
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
