import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle, CheckCircle2, FileCode2, GitCommitHorizontal, Library, Lock, Rocket, TriangleAlert } from "lucide-react";
import type { Diagnostic } from "../api";
import { api, fmt } from "../api";
import { CodeEditor } from "../components/CodeEditor";
import { useStore } from "../store";

type Tab = "library" | "commits" | "generated";

export function StudioPage() {
  const structure = useStore((s) => s.structure);
  const snap = useStore((s) => s.snap);
  const [files, setFiles] = useState<Record<string, string>>({});
  const [stdlib, setStdlib] = useState<Record<string, string>>({});
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [cur, setCur] = useState("field.plx");
  const [diags, setDiags] = useState<Diagnostic[]>([]);
  const [checking, setChecking] = useState(false);
  const [tab, setTab] = useState<Tab>("library");
  const [commits, setCommits] = useState<any[]>([]);
  const [commit, setCommit] = useState<any>(null);
  const [gen, setGen] = useState<string>("");
  const [msg, setMsg] = useState("");
  const [applyRes, setApplyRes] = useState<any>(null);
  const [reveal, setReveal] = useState<number | null>(null);
  const timer = useRef<number | null>(null);

  const load = useCallback(() => {
    api.twinFiles().then((r) => { setFiles(r.files); setStdlib(r.stdlib); });
    api.commits().then(setCommits);
  }, []);
  useEffect(() => { load(); }, [load, snap?.twin.version, snap?.twin.commits]);
  useEffect(() => { if (tab === "generated" && !gen) api.twinGenerated().then((g) => setGen(g.simulate)); }, [tab, gen]);

  const isStd = cur in stdlib && !(cur in files);
  const text = edits[cur] ?? files[cur] ?? stdlib[cur] ?? "";
  const dirty = Object.keys(edits).filter((k) => edits[k] !== files[k]);

  const check = useCallback((ed: Record<string, string>) => {
    setChecking(true);
    api.twinCheck(ed).then((r) => setDiags(r.diagnostics)).finally(() => setChecking(false));
  }, []);
  const onChange = (v: string) => {
    if (isStd) return;
    const ed = { ...edits, [cur]: v };
    setEdits(ed);
    if (timer.current) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => check(ed), 700);
  };
  const apply = async () => {
    const changed: Record<string, string> = {};
    for (const k of dirty) changed[k] = edits[k];
    const r = await api.twinApply(changed, msg || `edit ${dirty.join(", ")}`);
    setApplyRes(r);
    if (r.ok) { setEdits({}); setMsg(""); load(); } else if (r.diagnostics) setDiags(r.diagnostics);
  };

  const errs = diags.filter((d) => d.severity === "error"), warns = diags.filter((d) => d.severity === "warning");
  const fileList = [...Object.keys(files), ...Object.keys(stdlib).filter((k) => !(k in files))];

  return (
    <div className="h-full flex">
      <div className="w-[190px] shrink-0 border-r border-line bg-panel flex flex-col">
        <div className="panel-h">Twin sources</div>
        <div className="flex-1 overflow-auto py-1">
          {fileList.map((f) => {
            const ro = f in stdlib && !(f in files);
            const gen = f === "live.plx" || f === "calibration.plx";
            return (
              <div key={f} onClick={() => setCur(f)}
                className={`flex items-center gap-1.5 px-3 py-1.5 text-[12px] cursor-pointer ${cur === f ? "bg-[#0f2e2b] text-[#9ff3e6]" : "text-muted hover:text-ink hover:bg-panel2"}`}>
                {ro ? <Lock size={12} /> : <FileCode2 size={12} />}<span className="truncate flex-1">{f}</span>
                {dirty.includes(f) && <span className="w-1.5 h-1.5 rounded-full bg-warn" />}
                {gen && <span className="text-[9px] text-teal border border-teal/40 rounded px-1">auto</span>}
              </div>
            );
          })}
          <div className="px-3 pt-3 text-[10.5px] text-dim leading-snug">
            <b className="text-muted">field.plx</b> is the engineered twin. <b className="text-muted">live.plx</b> and <b className="text-muted">calibration.plx</b> are written by the data-integration layer. Standard-library packages are read-only here.
          </div>
        </div>
      </div>

      <div className="flex-1 min-w-0 flex flex-col">
        <div className="h-10 shrink-0 border-b border-line flex items-center gap-2 px-3">
          <span className="text-[12.5px] font-semibold">{cur}</span>
          {isStd && <span className="chip"><Lock size={10} />standard library</span>}
          <div className="flex-1" />
          <span className="text-[11.5px] flex items-center gap-1.5">
            {checking ? <span className="text-dim">checking…</span> :
              errs.length ? <span className="text-crit flex items-center gap-1"><AlertCircle size={13} />{errs.length} error(s)</span> :
                <span className="text-plx flex items-center gap-1"><CheckCircle2 size={13} />compiles</span>}
            {warns.length > 0 && <span className="text-warn flex items-center gap-1 ml-2"><TriangleAlert size={13} />{warns.length}</span>}
          </span>
          <input className="inp w-[260px]" placeholder="commit message" value={msg} onChange={(e) => setMsg(e.target.value)} />
          <button className="btn btn-primary" disabled={!dirty.length || errs.length > 0} onClick={apply}><Rocket size={13} />Apply to live twin</button>
        </div>
        <div className="flex-1 min-h-0">
          <CodeEditor value={text} onChange={onChange} readOnly={isStd} diagnostics={diags} file={cur} revealLine={reveal} />
        </div>
        <div className="h-[150px] shrink-0 border-t border-line overflow-auto bg-panel">
          <div className="panel-h">Diagnostics {applyRes && (applyRes.ok ? <span className="text-plx normal-case">applied · twin v{applyRes.version} · {applyRes.commit?.id}</span> : <span className="text-crit normal-case">not applied</span>)}</div>
          {diags.length === 0 && <div className="px-3 py-2 text-dim text-[12px]">No diagnostics. Edit the code - the PLX compiler checks it as you type (lexer, parser, units, elaboration, scheduling).</div>}
          {diags.map((d, i) => (
            <div key={i} className="px-3 py-1 text-[12px] flex gap-2 cursor-pointer hover:bg-panel2" onClick={() => { const f = d.file.split(/[\\/]/).pop()!; if (fileList.includes(f)) setCur(f); setReveal(d.line); }}>
              <span className={d.severity === "error" ? "text-crit" : d.severity === "warning" ? "text-warn" : "text-info"}>{d.severity}</span>
              <span className="num text-dim">{d.file.split(/[\\/]/).pop()}:{d.line}:{d.col}</span>
              <span>{d.message}</span>{d.hint && <span className="text-dim">— {d.hint}</span>}
            </div>
          ))}
        </div>
      </div>

      <div className="w-[400px] shrink-0 border-l border-line bg-panel flex flex-col">
        <div className="flex border-b border-line px-1">
          {([["library", "Library"], ["commits", "Versions"], ["generated", "Generated code"]] as [Tab, string][]).map(([t, l]) =>
            <div key={t} className={`tab ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>{l}</div>)}
        </div>
        <div className="flex-1 min-h-0 overflow-auto">
          {tab === "library" && <LibraryPanel />}
          {tab === "commits" && (
            <div>
              {commits.map((c) => (
                <div key={c.id} onClick={() => api.commit(c.id).then(setCommit)}
                  className={`px-3 py-2 border-b border-[#131c26] cursor-pointer hover:bg-panel2 ${commit?.id === c.id ? "bg-panel2" : ""}`}>
                  <div className="flex items-center gap-1.5 text-[12px]"><GitCommitHorizontal size={12} className={c.author.startsWith("auto") ? "text-teal" : "text-leg"} />
                    <span className="num text-dim">{c.id}</span><span className="truncate">{c.message}</span></div>
                  <div className="text-[10.5px] text-dim num">{c.author} · {fmt.clock(c.t)} · +{c.stats?.added ?? 0} −{c.stats?.removed ?? 0}</div>
                </div>
              ))}
              {commit && (
                <pre className="text-[11px] leading-snug p-2 m-2 rounded bg-[#0a1017] border border-line overflow-auto max-h-[420px] num">
                  {commit.diff.split("\n").map((l: string, i: number) => (
                    <div key={i} className={l.startsWith("+") && !l.startsWith("+++") ? "text-plx" : l.startsWith("-") && !l.startsWith("---") ? "text-crit" : l.startsWith("@@") ? "text-info" : "text-muted"}>{l || " "}</div>
                  ))}
                </pre>
              )}
            </div>
          )}
          {tab === "generated" && (
            <div className="h-full flex flex-col">
              <div className="px-3 py-2 text-[11.5px] text-dim leading-snug">What the PLX compiler emits: a time-vectorised, differentiable simulator (Python + torch). Every behavior statement runs once per asset group, over all instances, time steps and candidate plans at once.</div>
              <div className="flex-1 min-h-[400px]"><CodeEditor value={gen} language="python" readOnly /></div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function LibraryPanel() {
  const structure = useStore((s) => s.structure);
  const [open, setOpen] = useState<string | null>(null);
  const lib = useMemo(() => structure?.library ?? [], [structure]);
  return (
    <div>
      <div className="px-3 py-2 text-[11.5px] text-dim flex items-center gap-1.5"><Library size={13} />Entity blueprints used by this twin (all defined in PLX)</div>
      {lib.map((t) => (
        <div key={t.qualified} className="border-b border-[#131c26]">
          <div className="px-3 py-1.5 flex items-center gap-2 cursor-pointer hover:bg-panel2" onClick={() => setOpen(open === t.qualified ? null : t.qualified)}>
            <span className="text-[12.5px] font-semibold">{t.name}</span><span className="text-[10.5px] text-dim">{t.package}</span>
            <span className="ml-auto num text-[11px] text-muted">×{t.count}</span>
          </div>
          {open === t.qualified && (
            <div className="px-3 pb-2 text-[11.5px] space-y-1.5">
              {t.doc && <div className="text-muted leading-snug">{t.doc}</div>}
              <div><span className="text-dim">ports </span>{t.ports.map((p) => `${p.direction} ${p.name} ⟨${p.resources.map((r) => r.split(".").pop()).join(", ")}⟩${p.exclusive ? " exclusive" : ""}`).join(" · ") || "–"}</div>
              <div><span className="text-dim">decisions </span>{t.decisions.map((d) => `${d.name}:${d.domain}`).join(" · ") || "–"}</div>
              <div><span className="text-dim">states </span>{t.states.map((s) => s.name).join(" · ") || "–"}</div>
              <div><span className="text-dim">properties </span>{t.params.map((p) => `${p.name}${p.calibrate ? "*" : ""}${p.unit ? ` [${p.unit}]` : ""}`).join(" · ")}</div>
              <div><span className="text-dim">telemetry </span>{t.telemetry.map((x) => `${x.signal} ← ${x.tag}`).join(" · ") || "–"}</div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
