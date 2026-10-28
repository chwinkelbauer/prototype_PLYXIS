import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, ChevronLeft, ChevronRight, ChevronUp, FastForward, X } from "lucide-react";
import { api } from "../api";
import type { View } from "../store";
import { useStore } from "../store";

/** Guided demo, docked under the views (part of the layout, never on top of content).
 *  Keys: ← / → previous / next step, Enter = go to the step (fast-forward) or show it. */
export function DemoGuide() {
  const meta = useStore((s) => s.meta);
  const snap = useStore((s) => s.snap);
  const open = useStore((s) => s.demoOpen);
  const step = useStore((s) => s.demoStep);
  const set = useStore((s) => s.set);
  const [seeking, setSeeking] = useState<number | null>(null);
  const [expanded, setExpanded] = useState(true);
  const prevSpeed = useRef<number>(600);

  const steps = meta?.demo ?? [];
  const start = meta?.start_hour ?? 0;
  const cur = steps[step];
  const target = cur ? start + cur.t : 0;
  const reached = snap && cur ? snap.t >= target - 0.05 : false;

  useEffect(() => {
    if (seeking === null || !snap) return;
    if (snap.t >= seeking - 1e-6) {
      api.control("pause").then(() => api.control("speed", prevSpeed.current));
      setSeeking(null);
    }
  }, [snap?.t, seeking]);

  const go = useCallback(async () => {
    if (!cur) return;
    set({ view: cur.view as View });
    if (!snap) return;
    if (snap.t < target - 0.05) {
      prevSpeed.current = snap.speed;
      setSeeking(target);
      await api.control("speed", 3600);
      await api.control("play");
    }
  }, [cur, snap, target, set]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName) || el.closest(".monaco-editor"))) return;
      if (e.key === "ArrowRight" && step < steps.length - 1) { set({ demoStep: step + 1 }); e.preventDefault(); }
      else if (e.key === "ArrowLeft" && step > 0) { set({ demoStep: step - 1 }); e.preventDefault(); }
      else if (e.key === "Enter" && seeking === null) { go(); e.preventDefault(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, step, steps.length, seeking, go, set]);

  if (!open || !cur) return null;
  const when = `Day ${Math.floor(target / 24) + 1} ${String(Math.floor(target % 24)).padStart(2, "0")}:${String(Math.round((target % 1) * 60)).padStart(2, "0")}`;
  return (
    <div className="shrink-0 border-t border-line bg-panel/95 px-3 py-2" style={{ boxShadow: "0 -8px 24px #0006" }}>
      <div className="flex items-center gap-3 min-w-0">
        <div className="text-[10.5px] uppercase tracking-wider font-semibold text-brand2 whitespace-nowrap">Guided demo {step + 1}/{steps.length}</div>
        <div className="flex gap-0.5 shrink-0" aria-hidden>
          {steps.map((s, i) => (
            <button key={s.id} onClick={() => set({ demoStep: i })} title={s.title}
              className={`h-1.5 w-4 rounded ${i === step ? "bg-brand" : snap && snap.t >= start + s.t ? "bg-[#7a1a45]" : "bg-line2"}`} />
          ))}
        </div>
        <div className="flex-1 min-w-0 text-[13.5px] font-semibold truncate" title={cur.title}>{cur.title}</div>
        <div className="flex items-center gap-1.5 shrink-0">
          <button className="btn !px-2" disabled={step === 0} onClick={() => set({ demoStep: step - 1 })} title="Previous step (←)"><ChevronLeft size={14} /></button>
          <button className="btn btn-primary min-w-[180px] justify-center" onClick={go} disabled={seeking !== null} title="Go to this step (Enter)">
            {seeking !== null ? <><FastForward size={14} className="pulse" />fast-forwarding…</> :
              reached ? <>Show</> : <><FastForward size={14} />Go to {when}</>}
          </button>
          <button className="btn !px-2" disabled={step >= steps.length - 1} onClick={() => set({ demoStep: step + 1 })} title="Next step (→)"><ChevronRight size={14} /></button>
          <button className="btn btn-ghost !px-1.5" onClick={() => setExpanded(!expanded)} title={expanded ? "Hide the narrative" : "Show the narrative"}>
            {expanded ? <ChevronDown size={14} /> : <ChevronUp size={14} />}</button>
          <button className="btn btn-ghost !px-1.5" onClick={() => set({ demoOpen: false })} title="Close (Demo button in the top bar reopens)"><X size={14} /></button>
        </div>
      </div>
      {expanded && <div className="text-[12.5px] text-muted leading-relaxed mt-1.5 max-w-[1100px] line-clamp-2" title={cur.text}>{cur.text}</div>}
    </div>
  );
}
