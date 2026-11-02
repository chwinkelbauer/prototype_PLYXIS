import { useMemo, useState } from "react";
import { FlaskConical, Loader2 } from "lucide-react";
import { api, fmt } from "../api";
import { useStore } from "../store";

const AVAIL = /fan|hoist|station|compressor|crusher|swd|pump/i;

/** What would an outage or a price shock cost, and what would the plan do about it? Nothing is executed. */
export function WhatIfPanel() {
  const structure = useStore((s) => s.structure);
  const meta = useStore((s) => s.meta);
  const assets = useMemo(() => structure?.assets ?? [], [structure]);
  const prices = useMemo(() => Object.keys(meta?.prices ?? {}), [meta]);
  const [kind, setKind] = useState<"outage" | "price">("outage");
  const [asset, setAsset] = useState("");
  const [hours, setHours] = useState(6);
  const [price, setPrice] = useState("");
  const [factor, setFactor] = useState(2);
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<any>(null);
  const candidates = useMemo(() => assets.filter((a: any) => AVAIL.test(a.type ?? "")).slice(0, 200), [assets]);
  const chosenAsset = asset || (candidates[0] as any)?.path || "";
  const chosenPrice = price || prices[0] || "";
  const run = async () => {
    setBusy(true);
    setRes(null);
    try {
      const change = kind === "outage" ? { kind, asset: chosenAsset, hours } : { kind, name: chosenPrice, factor, hours };
      setRes(await api.whatIf([change], kind));
    } catch (e) { setRes({ ok: false, error: String(e) }); } finally { setBusy(false); }
  };
  const input = "num bg-panel2 border border-line2 rounded px-1.5 py-1 text-ink text-[12px]";
  return (
    <div className="col-span-12 panel">
      <div className="panel-h"><span className="flex items-center gap-1.5"><FlaskConical size={12} />What if · on the live twin, nothing is executed</span></div>
      <div className="p-3 grid grid-cols-12 gap-3 items-start">
        <div className="col-span-12 lg:col-span-5 flex flex-wrap items-center gap-2 text-[12px] text-muted">
          <select value={kind} onChange={(e) => setKind(e.target.value as "outage" | "price")} className={input}>
            <option value="outage">Asset out of service</option><option value="price">Price change</option>
          </select>
          {kind === "outage"
            ? <select value={chosenAsset} onChange={(e) => setAsset(e.target.value)} className={`${input} max-w-[240px]`}>
                {candidates.map((a: any) => <option key={a.path} value={a.path}>{a.label ?? a.path}</option>)}
              </select>
            : <>
                <select value={chosenPrice} onChange={(e) => setPrice(e.target.value)} className={input}>
                  {prices.map((p) => <option key={p} value={p}>{p.replace(/_/g, " ")}</option>)}
                </select>
                <label className="flex items-center gap-1">x<input type="number" step="0.1" value={factor} onChange={(e) => setFactor(Number(e.target.value))} className={`${input} w-16`} /></label>
              </>}
          <label className="flex items-center gap-1">for<input type="number" min={1} value={hours} onChange={(e) => setHours(Number(e.target.value))} className={`${input} w-16`} />h</label>
          <button className="btn btn-primary" disabled={busy || (kind === "outage" && !chosenAsset)} onClick={run}>
            {busy ? <Loader2 size={13} className="animate-spin" /> : <FlaskConical size={13} />}{busy ? "Re-planning…" : "Run"}
          </button>
        </div>
        <div className="col-span-12 lg:col-span-7 min-w-0">
          {!res && !busy && <div className="text-[12px] text-dim">Three values over the plan's horizon: a new plan as forecast, the current plan under the change, and a new plan for the change.</div>}
          {res && !res.ok && <div className="text-[12px] text-amber">{res.error}</div>}
          {res?.ok && (
            <div className="space-y-2">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                {[["The event costs", fmt.usd(res.cost_of_event), "after re-planning"], ["Reacting is worth", fmt.usd(res.value_of_reacting), "against keeping the plan"],
                  ["No reaction", fmt.usd(res.no_reaction), `max violation ${fmt.n(res.no_reaction_violation, 3)}`], ["Re-planned", fmt.usd(res.replanned), `${res.feasible ? "feasible" : "not feasible"} · ${fmt.n(res.seconds, 1)} s for both plans`]].map(([l, v, s]) => (
                  <div key={l} className="rounded border border-line2 bg-panel2 px-2.5 py-1.5"><div className="text-[10.5px] text-dim uppercase tracking-wider">{l}</div>
                    <div className="display text-[17px]">{v}</div><div className="text-[10.5px] text-dim">{s}</div></div>
                ))}
              </div>
              {res.decisions?.length > 0 && <div className="text-[11.5px] text-muted">Moves most: {res.decisions.slice(0, 5).map((d: any) => `${d.label} ${d.decision} ${d.change_share > 0 ? "+" : "−"}${Math.abs(d.change_share * 100).toFixed(0)} %`).join(" · ")}</div>}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
