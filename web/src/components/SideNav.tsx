import { BarChart3, Box, Code2, Cpu, Database, GanttChartSquare, Gauge, Map } from "lucide-react";
import type { View } from "../store";
import { useStore } from "../store";

const ITEMS: { id: View; label: string; icon: any; hint: string }[] = [
  { id: "overview", label: "Overview", icon: Gauge, hint: "Executive cockpit: value, production, bottlenecks" },
  { id: "map", label: "Live map", icon: Map, hint: "GIS view of the digital twin" },
  { id: "mine3d", label: "3D twin", icon: Box, hint: "Three-dimensional view of the twin (underground assets at true depth)" },
  { id: "schedule", label: "Schedule", icon: GanttChartSquare, hint: "Plan: set points, line-ups, trucks, crews" },
  { id: "optimizer", label: "Optimizer", icon: Cpu, hint: "Solver cascade, decomposition, shadow prices" },
  { id: "studio", label: "Twin studio", icon: Code2, hint: "PLX source of the twin, versions, diagnostics" },
  { id: "data", label: "Data", icon: Database, hint: "Connectors, tags, calibration, discovery" },
  { id: "bench", label: "Evidence", icon: BarChart3, hint: "Benchmarks and A/B results" },
];

export function SideNav() {
  const view = useStore((s) => s.view);
  const set = useStore((s) => s.set);
  const has3d = useStore((s) => (s.structure?.assets ?? []).some((a) => (a.at?.length ?? 0) > 2 || (a.line?.[0]?.length ?? 0) > 2));
  return (
    <nav className="w-[68px] shrink-0 border-r border-line bg-panel flex flex-col items-center py-2 gap-1">
      {ITEMS.filter((it) => it.id !== "mine3d" || has3d).map(({ id, label, icon: Icon, hint }) => (
        <button key={id} title={hint} onClick={() => set({ view: id })}
          className={`w-[60px] py-2 rounded-md flex flex-col items-center gap-1 transition-colors ${view === id ? "nav-active" : "text-muted hover:text-ink hover:bg-panel2"}`}>
          <Icon size={18} strokeWidth={1.7} />
          <span className="text-[9.5px] leading-none">{label}</span>
        </button>
      ))}
    </nav>
  );
}
