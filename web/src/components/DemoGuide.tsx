import { useEffect, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, FastForward, X } from "lucide-react";
import { api } from "../api";
import type { View } from "../store";
import { useStore } from "../store";

export function DemoGuide() {
  const meta = useStore((s) => s.meta);
  const snap = useStore((s) => s.snap);
  const open = useStore((s) => s.demoOpen);
  const step = useStore((s) => s.demoStep);
  const set = useStore((s) => s.set);
  const [seeking, setSeeking] = useState<number | null>(null);
  const [mini, setMini] = useState(false);
  const prevSpeed = useRef<number>(600);

  const steps = meta?.demo ?? [];
  const start = meta?.start_hour ?? 0;
  const cur = steps[step];

  useEffect(() => {
    if (seeking === null || !snap) return;
    if (snap.t >= seeking - 1e-6) {
      api.control("pause").then(() => api.control("speed", prevSpeed.current));
      setSeeking(null);
    }
  }, [snap?.t, seeking]);

  if (!open || !cur) return null;
  const target = start + cur.t;
  const reached = snap ? snap.t >= target - 0.05 : false;

  const go = async () => {
    set({ view: cur.view as View });
    if (!snap) return;
    if (snap.t < target - 0.05) {
      prevSpeed.current = snap.speed;
      setSeeking(target);
      await api.control("speed", 3600);
      await api.control("play");
    }
  };

  if (mini) {
    return (
      <div className="absolute left-3 bottom-3 z-30 panel shadow-xl shadow-black/60 flex items-center gap-2 px-2.5 py-1.5 fade-in" style={{ background: "#0b1219f2" }}>
        <button className="text-[11.5px] text-teal font-semibold" onClick={() => setMini(false)} title="Expand">Demo {step + 1}/{steps.length}</button>
        <span className="text-[12px] max-w-[260px] truncate">{cur.title}</span>
        <button className="btn !py-0.5" onClick={go} disabled={seeking !== null}><FastForward size={12} /></button>
        <button className="btn !py-0.5" disabled={step >= steps.length - 1} onClick={() => set({ demoStep: step + 1 })}><ChevronRight size={12} /></button>
      </div>
    );
  }
  return (
    <div className="absolute left-3 bottom-3 z-30 w-[420px] panel shadow-2xl shadow-black/60 fade-in" style={{ background: "#0b1219f2" }}>
      <div className="panel-h">
        <span className="text-teal">Guided demo · {step + 1}/{steps.length}</span>
        <span className="flex gap-1">
          <button className="btn btn-ghost !p-0.5 !text-[10px]" onClick={() => setMini(true)} title="Minimise">—</button>
          <button className="btn btn-ghost !p-0.5" onClick={() => set({ demoOpen: false })}><X size={14} /></button>
        </span>
      </div>
      <div className="px-3.5 py-3">
        <div className="text-[15px] font-semibold leading-snug">{cur.title}</div>
        <div className="text-[12.5px] text-muted leading-relaxed mt-1.5">{cur.text}</div>
        <div className="flex items-center gap-2 mt-3">
          <button className="btn" disabled={step === 0} onClick={() => set({ demoStep: step - 1 })}><ChevronLeft size={14} /></button>
          <button className="btn btn-primary flex-1 justify-center" onClick={go} disabled={seeking !== null}>
            {seeking !== null ? <><FastForward size={14} className="pulse" />fast-forwarding…</> :
              reached ? <>Show</> : <><FastForward size={14} />Go to {cur.view} @ Day {Math.floor(target / 24) + 1} {String(Math.floor(target % 24)).padStart(2, "0")}:{String(Math.round((target % 1) * 60)).padStart(2, "0")}</>}
          </button>
          <button className="btn" disabled={step >= steps.length - 1} onClick={() => set({ demoStep: step + 1 })}><ChevronRight size={14} /></button>
        </div>
        <div className="flex gap-1 mt-3">
          {steps.map((s, i) => (
            <button key={s.id} onClick={() => set({ demoStep: i })} title={s.title}
              className={`h-1.5 flex-1 rounded ${i === step ? "bg-teal" : snap && snap.t >= start + s.t ? "bg-[#1f5f57]" : "bg-line2"}`} />
          ))}
        </div>
      </div>
    </div>
  );
}
