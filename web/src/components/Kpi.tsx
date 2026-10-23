import type { ReactNode } from "react";

export function KpiAB({ label, plx, leg, unit, digits = 0, better = "higher", hint, format }: {
  label: string; plx?: number; leg?: number; unit?: string; digits?: number; better?: "higher" | "lower";
  hint?: string; format?: (v: number) => string;
}) {
  const f = format ?? ((v: number) => v.toLocaleString("en-US", { maximumFractionDigits: digits, minimumFractionDigits: digits }));
  const has = plx !== undefined && leg !== undefined && !Number.isNaN(plx) && !Number.isNaN(leg);
  const diff = has ? (plx! - leg!) : 0;
  const pct = has && Math.abs(leg!) > 1e-9 ? (diff / Math.abs(leg!)) * 100 : 0;
  const good = better === "higher" ? diff >= 0 : diff <= 0;
  return (
    <div className="panel px-3 py-2.5 min-w-0" title={hint}>
      <div className="text-[10.5px] uppercase tracking-wider text-muted font-semibold truncate">{label}</div>
      <div className="flex items-baseline gap-1.5 mt-1">
        <span className="num text-[22px] leading-none font-semibold text-ink">{plx === undefined ? "–" : f(plx)}</span>
        {unit && <span className="text-[11px] text-dim">{unit}</span>}
      </div>
      <div className="flex items-center justify-between mt-1.5 text-[11px]">
        <span className="text-dim">manual <span className="num text-leg/90">{leg === undefined ? "–" : f(leg)}</span></span>
        {has && Math.abs(diff) > 1e-9 && (
          <span className={`num font-semibold ${good ? "text-plx" : "text-crit"}`}>
            {diff >= 0 ? "▲" : "▼"} {Math.abs(pct) >= 0.05 ? `${Math.abs(pct).toFixed(1)} %` : f(Math.abs(diff))}
          </span>
        )}
      </div>
    </div>
  );
}

export function Stat({ label, value, sub, color = "text-ink", icon }: { label: string; value: ReactNode; sub?: ReactNode;
  color?: string; icon?: ReactNode }) {
  return (
    <div className="panel px-3 py-2.5 min-w-0">
      <div className="flex items-center gap-1.5 text-[10.5px] uppercase tracking-wider text-muted font-semibold">{icon}{label}</div>
      <div className={`num text-[20px] leading-tight font-semibold mt-1 ${color}`}>{value}</div>
      {sub && <div className="text-[11px] text-dim mt-0.5">{sub}</div>}
    </div>
  );
}
