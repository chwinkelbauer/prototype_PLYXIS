import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, ChevronRight, RefreshCw } from "lucide-react";
import type { Gantt, GanttRow, GanttSegment } from "../api";
import { api, fmt } from "../api";
import { useStore } from "../store";

const ROW_H = 20;
const LABEL_W = 250;
const GROUP_TITLES: Record<string, string> = {
  compressor_station: "Compression", battery: "Tank batteries · pump-out", routing: "Line-ups & routing",
  truck: "Trucks · trips", crew: "Rigs & crews · jobs", pad: "Pads", truck_lane: "Truck lanes",
};

function color(row: GanttRow, s: GanttSegment, rowMax: number): string {
  if (row.kind === "trips") return s.cargo === "water" ? "#1d6fa5" : "#b7791f";
  if (row.kind === "jobs") return s.status === "done" ? "#1f6f55" : s.status === "active" ? "#1d6fa5" : "#3b4b5e";
  if (row.kind === "lineup") {
    const hues = ["#0f766e", "#7c3aed", "#b45309", "#1d4ed8"];
    let h = 0;
    for (const ch of s.label) h = (h * 31 + ch.charCodeAt(0)) % 997;
    return hues[h % hues.length];
  }
  if (row.decision === "on") return s.value >= 0.5 ? "#14532d" : "#7f1d1d";
  const f = rowMax > 0 ? Math.max(0, Math.min(1, s.value / rowMax)) : 0;
  const a = 0.25 + 0.75 * f;
  return `rgba(45, 212, 191, ${a.toFixed(2)})`;
}

export function SchedulePage() {
  const planVersion = useStore((s) => s.planVersion);
  const snap = useStore((s) => s.snap);
  const set = useStore((s) => s.set);
  const [g, setG] = useState<Gantt | null>(null);
  const [open, setOpen] = useState<Record<string, boolean>>({ compressor_station: true, battery: true, routing: true, truck: true, crew: true });
  const [hover, setHover] = useState<{ x: number; y: number; row: GanttRow; seg: GanttSegment } | null>(null);
  const [why, setWhy] = useState<Record<string, any>>({});
  const box = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(1200);

  useEffect(() => { api.gantt().then(setG).catch(() => {}); }, [planVersion]);
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const groups = useMemo(() => {
    if (!g) return [];
    const out: { key: string; title: string; rows: GanttRow[] }[] = [];
    const add = (key: string, title: string, row: GanttRow) => {
      let grp = out.find((x) => x.key === key);
      if (!grp) { grp = { key, title, rows: [] }; out.push(grp); }
      grp.rows.push(row);
    };
    for (const r of g.rows) {
      if (r.group === "esp_well" || r.group === "rod_pump_well") {
        const pad = r.label.split(" W-")[0];
        add(`wells:${pad}`, `Wells · ${pad}`, r);
      } else add(r.group, GROUP_TITLES[r.group] ?? r.group.replace(/_/g, " "), r);
    }
    const order = ["compressor_station", "battery", "routing", "truck", "crew"];
    out.sort((a, b) => {
      const ia = order.indexOf(a.key), ib = order.indexOf(b.key);
      return (ia < 0 ? 50 : ia) - (ib < 0 ? 50 : ib) || a.key.localeCompare(b.key);
    });
    return out;
  }, [g]);

  if (!g) return <div className="p-4 text-dim">Loading plan…</div>;
  const t0 = g.t0, t1 = g.t1;
  const W = Math.max(400, width - LABEL_W - 24);
  const x = (t: number) => ((t - t0) / (t1 - t0)) * W;
  const ticks: number[] = [];
  for (let t = Math.ceil(t0 / 6) * 6; t <= t1; t += 6) ticks.push(t);
  const now = snap?.t ?? g.now;

  const fetchWhy = (asset: string) => {
    if (why[asset]) return;
    setWhy((w) => ({ ...w, [asset]: { loading: true } }));
    api.explain(asset).then((r) => setWhy((w) => ({ ...w, [asset]: r }))).catch(() => {});
  };
  const whyText = (row: GanttRow, seg: GanttSegment): string | null => {
    const w = why[row.asset];
    if (!w || w.loading || !w.decisions) return w?.loading ? "computing reasons…" : null;
    const dec = w.decisions.find((d: any) => d.name === row.decision);
    const rel = (seg.start + seg.end) / 2 - (w.t0 ?? t0);
    const b = dec?.blocks.find((b: any) => b.t_start <= rel && rel < b.t_end);
    return b?.summary ?? null;
  };

  let y = 0;
  const rowsSvg: JSX.Element[] = [];
  const labels: JSX.Element[] = [];
  for (const grp of groups) {
    const isOpen = open[grp.key] ?? false;
    labels.push(
      <div key={`g-${grp.key}`} style={{ top: y, height: ROW_H + 4 }} className="absolute left-0 right-0 flex items-center gap-1 px-2 text-[11px] font-semibold uppercase tracking-wider text-muted bg-panel2 border-b border-line cursor-pointer"
        onClick={() => setOpen({ ...open, [grp.key]: !isOpen })}>
        {isOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}{grp.title}<span className="text-dim font-normal ml-1">({grp.rows.length})</span>
      </div>);
    rowsSvg.push(<rect key={`gb-${grp.key}`} x={0} y={y} width={W} height={ROW_H + 4} fill="#101a24" />);
    y += ROW_H + 4;
    if (!isOpen) continue;
    for (const row of grp.rows) {
      const rowMax = Math.max(1e-9, ...row.segments.map((s) => s.value));
      labels.push(
        <div key={`l-${row.asset}-${row.decision}`} style={{ top: y, height: ROW_H }} className="absolute left-0 right-0 flex items-center px-2 pl-6 text-[11.5px] border-b border-[#121b25] hover:bg-panel2 cursor-pointer truncate"
          onClick={() => set({ selected: row.kind === "trips" || row.kind === "jobs" ? null : row.asset })}>
          <span className="truncate">{row.label}</span><span className="ml-auto text-dim text-[10.5px] pl-2">{row.decision.replace(/_/g, " ")}</span>
        </div>);
      rowsSvg.push(<line key={`rl-${row.asset}-${row.decision}`} x1={0} x2={W} y1={y + ROW_H} y2={y + ROW_H} stroke="#121b25" />);
      for (const [i, s] of row.segments.entries()) {
        const xa = Math.max(0, x(s.start)), xb = Math.min(W, x(s.end));
        if (xb <= xa) continue;
        const yy = y;
        rowsSvg.push(
          <g key={`s-${row.asset}-${row.decision}-${i}`} className="cursor-pointer"
            onMouseEnter={(e) => { setHover({ x: e.clientX, y: e.clientY, row, seg: s }); if (row.kind === "setpoint") fetchWhy(row.asset); }}
            onMouseMove={(e) => setHover((h) => (h ? { ...h, x: e.clientX, y: e.clientY } : h))}
            onMouseLeave={() => setHover(null)}
            onClick={() => row.kind !== "trips" && set({ selected: row.kind === "jobs" ? null : row.asset })}>
            <rect x={xa + 0.5} y={yy + 3} width={Math.max(1, xb - xa - 1)} height={ROW_H - 6} rx={2} fill={color(row, s, rowMax)} />
            {xb - xa > 34 && (
              <text x={xa + 4} y={yy + ROW_H / 2 + 3.5} fontSize={10} fill="#e6edf3" fontFamily="JetBrains Mono" pointerEvents="none">
                {s.label.length * 6 > xb - xa - 6 ? s.label.slice(0, Math.max(2, Math.floor((xb - xa - 8) / 6))) : s.label}
              </text>
            )}
          </g>);
      }
      y += ROW_H;
    }
  }
  const H = y;

  return (
    <div className="h-full flex flex-col p-3 gap-2">
      <div className="flex items-center gap-3">
        <div className="text-[13px] font-semibold">Operating plan #{g.plan_id}</div>
        <div className="text-[11.5px] text-dim">horizon {fmt.clock(t0)} – {fmt.clock(t1)} · re-optimised on every disruption and every 3 h</div>
        <div className="flex-1" />
        <button className="btn" onClick={() => api.replan("manual request from schedule")}><RefreshCw size={13} />Re-plan now</button>
      </div>
      <div ref={box} className="panel flex-1 min-h-0 flex flex-col overflow-hidden">
        <div className="flex border-b border-line" style={{ height: 26 }}>
          <div style={{ width: LABEL_W }} className="shrink-0 px-2 text-[10.5px] uppercase tracking-wider text-dim flex items-center">asset · decision</div>
          <svg width={W} height={26} className="shrink-0">
            {ticks.map((t) => (
              <g key={t}><line x1={x(t)} x2={x(t)} y1={18} y2={26} stroke="#34475a" />
                <text x={x(t) + 3} y={15} fontSize={10} fill="#8293a6" fontFamily="JetBrains Mono">{fmt.clock(t)}</text></g>
            ))}
          </svg>
        </div>
        <div className="flex-1 overflow-auto">
          <div className="flex relative" style={{ height: H }}>
            <div style={{ width: LABEL_W, height: H }} className="shrink-0 relative border-r border-line">{labels}</div>
            <svg width={W} height={H} className="shrink-0">
              {ticks.map((t) => <line key={t} x1={x(t)} x2={x(t)} y1={0} y2={H} stroke="#16212c" />)}
              {rowsSvg}
              {now >= t0 && now <= t1 && <line x1={x(now)} x2={x(now)} y1={0} y2={H} stroke="#f59e0b" strokeWidth={1.5} strokeDasharray="4 3" />}
            </svg>
          </div>
        </div>
      </div>
      {hover && (
        <div className="fixed z-50 panel px-3 py-2 text-[12px] pointer-events-none max-w-[380px] shadow-xl shadow-black/50"
          style={{ left: Math.min(hover.x + 14, window.innerWidth - 400), top: hover.y + 14, background: "#0b1219f5" }}>
          <div className="font-semibold">{hover.row.label} · {hover.row.decision.replace(/_/g, " ")}</div>
          <div className="num text-muted">{fmt.clock(hover.seg.start)} – {fmt.clock(hover.seg.end)} · <span className="text-ink">{hover.seg.label}</span></div>
          {hover.seg.reason && <div className="text-muted mt-1">{hover.seg.reason}</div>}
          {hover.seg.shares && <div className="text-muted mt-1">{Object.entries(hover.seg.shares).map(([k, v]) => `${k} ${(v * 100).toFixed(0)} %`).join(" · ")}</div>}
          {hover.row.kind === "setpoint" && <div className="mt-1.5 text-[11.5px] text-ink leading-snug border-t border-line pt-1.5">{whyText(hover.row, hover.seg) ?? "hover to compute reasons"}</div>}
        </div>
      )}
    </div>
  );
}
