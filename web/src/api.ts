// Typed client for the PLYXIS server API.

export type Rate = Partial<Record<
  "oil_bpd" | "gas_mscfd" | "water_bpd" | "flare_mscfd" | "oil_sold_bpd" | "profit_usd_h" | "revenue_usd_h" |
  "cost_usd_h" | "emission_t_h", number>>;

export interface WorldKpi {
  cum: { revenue: number; cost: number; profit: number; emission_kg: number; oil_bbl: number; gas_mscf: number;
    water_bbl: number; flared_mscf: number; spilled_bbl: number; hours: number; inventory_value?: number;
    equipment_value?: number; carbon_cost?: number; profit_after_carbon?: number; flared_over_permit_mscf?: number };
  rate: Rate;
}

export interface Kpis {
  plyxis: WorldKpi;
  legacy: WorldKpi;
  delta: { profit: number; profit_after_carbon?: number; oil_bbl: number; flared_mscf: number; emission_t: number };
  hours: number;
  value_per_day: number;
  carbon_price_usd_t: number;
  bridge?: { key: string; label: string; delta: number }[];   // operating-profit difference by realised term
}

export interface Alert {
  t: number; clock?: string; severity: "info" | "warning" | "critical"; kind: string; message: string;
  asset?: string | null; label?: string; tag?: string; data?: Record<string, unknown>;
}

export interface TimelineEvent { t: number; clock: string; kind: string; title: string; text: string; asset?: string }

export interface Truck { truck: string; lat: number; lon: number; state: string; lane: string | null; cargo: string | null }

export interface Job {
  id: string; asset: string; label: string; kind: string; resource: string; resource_label: string;
  start: number; end: number; travel_h: number; hours: number; value_per_h: number; reason: string;
  at: [number, number] | null; status: string;
}

export interface KpiPoint { t: number; p: Rate; l: Rate; pc: number; lc: number }

export interface Snapshot {
  type: "snapshot"; t: number; clock: string; running: boolean; holding?: boolean; speed: number; kpis: Kpis; trucks: Truck[];
  assets: Record<string, Record<string, number>>;
  optimizer: { running: boolean; plan_id: number | null; last_plan_t: number; neural: Record<string, unknown>;
    progress_tail: Record<string, unknown>[] };
  twin: { version: number; commits: number };
  jobs: Job[];
  prices: Record<string, number>;
  alerts_tail: Alert[];
  kpi_history?: KpiPoint[];
  timeline?: TimelineEvent[];
  alerts?: Alert[];
}

export interface Asset {
  path: string; label: string; type: string; kind: "point" | "line"; layer: string; icon: string;
  at: [number, number] | null; line: [number, number][] | null; tag: string; parent: string | null;
  is_link: boolean; subsystem: number | null; doc: string; resources?: string[];
  params: Record<string, { value: number; engineering: number; unit: string; calibrated: boolean; doc: string }>;
}

export interface LibraryType {
  name: string; qualified: string; doc: string; base: string | null; package: string; count: number;
  params: { name: string; unit: string | null; calibrate: boolean; doc: string }[];
  states: { name: string; unit: string | null }[];
  decisions: { name: string; domain: string; unit: string | null }[];
  ports: { name: string; direction: string; resources: string[]; exclusive: boolean }[];
  telemetry: { signal: string; tag: string; unit: string | null }[];
}

export interface Structure {
  assets: Asset[]; edges: { src: string; dst: string; resources: string[]; line: [number, number][] }[];
  layers: string[]; origin: [number, number] | null; env: string; version: number;
  subsystems: { id: number; name: string; kind: string; size: number; types: Record<string, number> }[];
  summary: Record<string, unknown>; library: LibraryType[];
}

export interface DemoStep { id: string; t: number; title: string; view: string; text: string }

export interface Meta {
  name: string; description: string; start_hour: number; events: Record<string, unknown>[]; demo: DemoStep[];
  speed: number; dt_emulator_h: number; replan_every_h: number;
}

export interface GanttSegment { start: number; end: number; value: number; label: string; status?: string;
  reason?: string; kind?: string; cargo?: string; shares?: Record<string, number> }
export interface GanttRow { group: string; asset: string; label: string; decision: string; kind: string;
  segments: GanttSegment[] }
export interface Gantt { rows: GanttRow[]; t0: number; t1: number; now: number; plan_id?: number }

export interface Diagnostic { severity: string; message: string; code: string; hint: string; file: string;
  line: number; col: number; end_line: number; end_col: number }

async function j<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

const post = <T>(url: string, body: unknown) =>
  j<T>(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export const api = {
  health: () => j<{ ok: boolean; booting: boolean; message: string }>("/api/health"),
  meta: () => j<Meta>("/api/meta"),
  structure: () => j<Structure>("/api/structure"),
  snapshot: (full = false) => j<Snapshot>(`/api/snapshot?full=${full ? 1 : 0}`),
  control: (action: string, value?: number) => post<{ running: boolean; speed: number; t: number }>("/api/control", { action, value }),
  replan: (trigger = "manual request") => post("/api/replan", { trigger }),
  event: (ev: Record<string, unknown>) => post("/api/event", ev),
  plan: () => j<any>("/api/plan"),
  gantt: () => j<Gantt>("/api/plan/gantt"),
  explain: (path: string) => j<any>(`/api/explain/${encodeURIComponent(path)}`),
  asset: (path: string) => j<any>(`/api/asset/${encodeURIComponent(path)}`),
  optimizer: () => j<any>("/api/optimizer"),
  twinFiles: () => j<{ files: Record<string, string>; version: number; env: string; stdlib: Record<string, string> }>("/api/twin/files"),
  twinGenerated: () => j<{ simulate: string; bounds: string; summary: Record<string, unknown> }>("/api/twin/generated"),
  twinCheck: (files: Record<string, string>) => post<{ diagnostics: Diagnostic[] }>("/api/twin/check", { files }),
  twinApply: (files: Record<string, string>, message: string) => post<any>("/api/twin/apply", { files, message }),
  commits: () => j<any[]>("/api/twin/commits"),
  commit: (id: string) => j<any>(`/api/twin/commit/${id}`),
  connectors: () => j<any[]>("/api/data/connectors"),
  tags: (prefix = "") => j<any[]>(`/api/data/tags?prefix=${encodeURIComponent(prefix)}`),
  tag: (tag: string) => j<{ tag: string; history: [number, number][] }>(`/api/data/tag?tag=${encodeURIComponent(tag)}`),
  calibration: () => j<any>("/api/data/calibration"),
  discovery: () => j<any[]>("/api/data/discovery"),
  discoveryAction: (id: string, action: "approve" | "reject") => post<any>(`/api/data/discovery/${id}/${action}`, {}),
  alerts: () => j<Alert[]>("/api/alerts"),
  timeline: () => j<TimelineEvent[]>("/api/timeline"),
  bench: () => j<Record<string, any>>("/api/bench"),
};

// ---------------------------------------------------------------- formatting helpers
export const fmt = {
  n: (v: number | undefined | null, d = 0) =>
    v === undefined || v === null || Number.isNaN(v) ? "–" : v.toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d }),
  usd: (v: number | undefined | null, d = 0) => {
    if (v === undefined || v === null || Number.isNaN(v)) return "–";
    const a = Math.abs(v);
    const s = v < 0 ? "−" : "";
    if (a >= 1e6) return `${s}$${(a / 1e6).toFixed(2)}M`;
    if (a >= 1e4) return `${s}$${(a / 1e3).toFixed(1)}k`;
    return `${s}$${a.toLocaleString("en-US", { maximumFractionDigits: d })}`;
  },
  signed: (v: number, d = 0, unit = "") => `${v >= 0 ? "+" : "−"}${Math.abs(v).toLocaleString("en-US", { maximumFractionDigits: d })}${unit}`,
  pct: (v: number, d = 1) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)} %`,
  clock: (t: number) => {
    const day = Math.floor(t / 24) + 1;
    const h = t % 24;
    const hh = Math.floor(h);
    const mm = Math.round((h - hh) * 60) % 60;
    return `D${day} ${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")}`;
  },
  hhmm: (t: number) => {
    const h = ((t % 24) + 24) % 24;
    const hh = Math.floor(h);
    return `${String(hh).padStart(2, "0")}:${String(Math.round((h - hh) * 60) % 60).padStart(2, "0")}`;
  },
};
