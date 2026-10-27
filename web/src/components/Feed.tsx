import { AlertTriangle, CheckCircle2, Cpu, GitCommitHorizontal, Info, Siren, TrendingUp, Wrench } from "lucide-react";
import type { Alert, TimelineEvent } from "../api";
import { fmt } from "../api";
import { useStore } from "../store";

const SEV: Record<string, string> = { critical: "text-crit", warning: "text-warn", info: "text-info" };

export function AlertFeed({ limit = 40, compact = false }: { limit?: number; compact?: boolean }) {
  const alerts = useStore((s) => s.alerts);
  const set = useStore((s) => s.set);
  const rows = alerts.slice(-limit).reverse();
  return (
    <div className="overflow-auto h-full">
      {rows.length === 0 && <div className="text-dim text-[12px] p-3">No alerts.</div>}
      {rows.map((a: Alert, i) => (
        <div key={i} className="flex gap-2 px-2.5 py-1.5 border-b border-[#1a1a26] hover:bg-panel2 cursor-pointer fade-in"
          onClick={() => a.asset && set({ selected: a.asset })}>
          <span className={`mt-0.5 ${SEV[a.severity] ?? "text-muted"}`}>
            {a.severity === "critical" ? <Siren size={13} /> : a.kind === "trend" ? <TrendingUp size={13} /> :
              a.severity === "warning" ? <AlertTriangle size={13} /> : <Info size={13} />}
          </span>
          <div className="min-w-0">
            <div className={`text-[12px] leading-snug ${compact ? "truncate" : ""}`}>{a.message}</div>
            <div className="text-[10.5px] text-dim num">{a.clock ?? fmt.clock(a.t)} · {a.kind}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

const KIND_ICON: Record<string, any> = {
  plan: Cpu, event: AlertTriangle, failure: Siren, maintenance: Wrench, twin: GitCommitHorizontal, market: TrendingUp,
};
const KIND_COLOR: Record<string, string> = {
  plan: "text-opt", event: "text-warn", failure: "text-crit", maintenance: "text-info", twin: "text-teal", market: "text-leg",
};

export function Timeline({ limit = 60, kinds }: { limit?: number; kinds?: string[] }) {
  const tl = useStore((s) => s.timeline);
  const rows = tl.filter((e) => !kinds || kinds.includes(e.kind)).slice(-limit).reverse();
  return (
    <div className="overflow-auto h-full">
      {rows.length === 0 && <div className="text-dim text-[12px] p-3">Nothing yet.</div>}
      {rows.map((e: TimelineEvent, i) => {
        const Icon = KIND_ICON[e.kind] ?? CheckCircle2;
        return (
          <div key={i} className="flex gap-2 px-2.5 py-1.5 border-b border-[#1a1a26] fade-in">
            <Icon size={13} className={`mt-0.5 shrink-0 ${KIND_COLOR[e.kind] ?? "text-muted"}`} />
            <div className="min-w-0">
              <div className="text-[12px] leading-snug">{e.title}</div>
              {e.text && <div className="text-[11px] text-dim leading-snug line-clamp-2 whitespace-pre-line">{e.text}</div>}
              <div className="text-[10.5px] text-dim num">{e.clock}</div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
