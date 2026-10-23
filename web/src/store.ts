import { create } from "zustand";
import type { Alert, KpiPoint, Meta, Snapshot, Structure, TimelineEvent } from "./api";

export type View = "overview" | "map" | "schedule" | "optimizer" | "studio" | "data" | "bench";

export interface OptRec { stage: string; iter?: number; objective?: number; max_violation?: number; t?: number;
  t_total?: number; event?: string; generation?: number; [k: string]: unknown }

interface State {
  connected: boolean;
  booting: string | null;
  meta: Meta | null;
  structure: Structure | null;
  snap: Snapshot | null;
  kpiHistory: KpiPoint[];
  alerts: Alert[];
  timeline: TimelineEvent[];
  optRun: { trigger: string; t: number; recs: OptRec[]; running: boolean } | null;
  planVersion: number;
  structureVersion: number;
  view: View;
  selected: string | null;
  demoOpen: boolean;
  demoStep: number;
  set: (p: Partial<State>) => void;
  ingest: (msg: any) => void;
}

export const useStore = create<State>((set, get) => ({
  connected: false,
  booting: "connecting",
  meta: null,
  structure: null,
  snap: null,
  kpiHistory: [],
  alerts: [],
  timeline: [],
  optRun: null,
  planVersion: 0,
  structureVersion: 0,
  view: (["overview", "map", "schedule", "optimizer", "studio", "data", "bench"].includes(location.hash.slice(1)) ?
    location.hash.slice(1) : "overview") as View,
  selected: null,
  demoOpen: true,
  demoStep: 0,
  set: (p) => {
    if (p.view && location.hash.slice(1) !== p.view) history.replaceState(null, "", `#${p.view}`);
    set(p);
  },
  ingest: (msg) => {
    const s = get();
    switch (msg.type) {
      case "booting":
        set({ booting: msg.message });
        break;
      case "ready":
        set({ booting: null, planVersion: s.planVersion + 1, structureVersion: s.structureVersion + 1 });
        break;
      case "reset":
        set({ kpiHistory: [], alerts: [], timeline: [], optRun: null, planVersion: s.planVersion + 1 });
        break;
      case "snapshot": {
        const upd: Partial<State> = { snap: msg as Snapshot, booting: null };
        if (msg.kpi_history) upd.kpiHistory = msg.kpi_history;
        else {
          const k = msg.kpis;
          const hist = s.kpiHistory;
          const last = hist[hist.length - 1];
          if (!last || msg.t - last.t >= 0.49) {
            const p: KpiPoint = { t: msg.t, p: k.plyxis.rate, l: k.legacy.rate, pc: k.plyxis.cum.profit, lc: k.legacy.cum.profit };
            upd.kpiHistory = [...hist.slice(-2400), p];
          }
        }
        if (msg.timeline) upd.timeline = msg.timeline;
        if (msg.alerts) upd.alerts = msg.alerts;
        set(upd);
        break;
      }
      case "alert":
        set({ alerts: [...s.alerts.slice(-299), msg.alert] });
        break;
      case "timeline":
        set({ timeline: [...s.timeline.slice(-399), msg.event] });
        break;
      case "optimizer":
        if (msg.event === "start") set({ optRun: { trigger: msg.trigger, t: msg.t, recs: [], running: true } });
        else if (msg.event === "progress" && s.optRun) {
          const recs = [...s.optRun.recs, msg.rec];
          set({ optRun: { ...s.optRun, recs: recs.slice(-1500), running: msg.rec.stage !== "done" } });
        }
        break;
      case "plan":
        set({ planVersion: s.planVersion + 1, optRun: s.optRun ? { ...s.optRun, running: false } : null });
        break;
      case "twin":
        if (msg.structure_changed) set({ structureVersion: s.structureVersion + 1 });
        break;
      default:
        break;
    }
  },
}));

let ws: WebSocket | null = null;
export function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => useStore.getState().set({ connected: true });
  ws.onclose = () => {
    useStore.getState().set({ connected: false });
    setTimeout(connect, 1500);
  };
  ws.onmessage = (ev) => {
    try {
      useStore.getState().ingest(JSON.parse(ev.data));
    } catch (e) {
      console.error(e);
    }
  };
}
