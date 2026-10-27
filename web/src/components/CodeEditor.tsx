import Editor, { loader, type OnMount } from "@monaco-editor/react";
import * as monaco from "monaco-editor";
import EditorWorker from "monaco-editor/esm/vs/editor/editor.worker?worker";
import { useEffect, useRef } from "react";
import type { Diagnostic } from "../api";

(self as any).MonacoEnvironment = { getWorker: () => new EditorWorker() };
loader.config({ monaco });

let registered = false;
function register() {
  if (registered) return;
  registered = true;
  monaco.languages.register({ id: "plx" });
  monaco.languages.setMonarchTokensProvider("plx", {
    keywords: ["define", "as", "use", "set", "init", "connect", "via", "for", "in", "if", "else", "rule", "on", "when",
      "constraint", "soft", "final", "cost", "revenue", "emission", "metric", "maximize", "minimize", "default", "hold",
      "calibrate", "input", "exclusive", "extends", "and", "or", "not", "true", "false", "TRUE", "FALSE", "load"],
    kinds: ["pkg", "ent", "env", "sim", "grp", "res", "obj", "bool", "int", "real"],
    sections: ["properties", "state", "decision", "nodes", "behavior", "behaviour", "contains", "geom", "telemetry",
      "discover", "objective", "solver", "inputs", "attr", "constraints", "env"],
    builtins: ["min", "max", "abs", "sqrt", "exp", "log", "pos", "clamp", "if", "pwl", "delay", "prev", "sum", "mean", "sat",
      "smin", "smax", "table", "geojson", "distance", "concat", "zpad", "dt", "t", "time", "hour", "self"],
    tokenizer: {
      root: [
        [/##.*$/, "comment.doc"],
        [/#.*$/, "comment"],
        [/"([^"\\]|\\.)*"/, "string"],
        [/\d+(\.\d+)?([eE][-+]?\d+)?(?=\s*\[)/, "number", "@unit"],
        [/\d+(\.\d+)?([eE][-+]?\d+)?/, "number"],
        [/[a-zA-Z_]\w*(?=\s*:(?!:))/, { cases: { "@sections": "keyword.section", "@default": "identifier" } }],
        [/[a-zA-Z_]\w*/, { cases: { "@keywords": "keyword", "@kinds": "type", "@builtins": "predefined", "@default": "identifier" } }],
        [/<-|<=|>=|==|!=|->|\.\./, "operator"],
        [/[{}()[\];,.]/, "delimiter"],
        [/[+\-*/^%<>=!]/, "operator"],
      ],
      unit: [[/\s*\[[^\]\n]*\]/, "type.unit", "@pop"], [/./, { token: "", next: "@pop", goBack: 1 }]],
    },
  } as any);
  monaco.languages.setLanguageConfiguration("plx", {
    comments: { lineComment: "#" },
    brackets: [["{", "}"], ["[", "]"], ["(", ")"]],
    autoClosingPairs: [{ open: "{", close: "}" }, { open: "[", close: "]" }, { open: "(", close: ")" }, { open: '"', close: '"' }],
  });
  monaco.editor.defineTheme("plyxis", {
    base: "vs-dark", inherit: true,
    rules: [
      { token: "comment", foreground: "5c5c70" }, { token: "comment.doc", foreground: "9d93b8", fontStyle: "italic" },
      { token: "keyword", foreground: "ff5c9a" }, { token: "keyword.section", foreground: "a78bfa", fontStyle: "bold" },
      { token: "type", foreground: "f5b14a" }, { token: "type.unit", foreground: "7dd3c0" },
      { token: "predefined", foreground: "7cb3f2" }, { token: "number", foreground: "f0a05a" },
      { token: "string", foreground: "a3e3b5" }, { token: "operator", foreground: "c4c4d4" },
    ],
    colors: { "editor.background": "#0c0c12", "editor.lineHighlightBackground": "#16161f", "editorGutter.background": "#0c0c12",
      "editorLineNumber.foreground": "#3a3a52", "editorLineNumber.activeForeground": "#a1a1aa",
      "editor.selectionBackground": "#fe026540", "editorCursor.foreground": "#fe0265" },
  });
}

export function CodeEditor({ value, onChange, language = "plx", readOnly = false, diagnostics = [], file, revealLine }: {
  value: string; onChange?: (v: string) => void; language?: string; readOnly?: boolean; diagnostics?: Diagnostic[];
  file?: string; revealLine?: number | null;
}) {
  const ed = useRef<monaco.editor.IStandaloneCodeEditor | null>(null);
  register();
  const onMount: OnMount = (editor) => { ed.current = editor as any; };
  useEffect(() => {
    const model = ed.current?.getModel();
    if (!model) return;
    const marks = diagnostics.filter((d) => !file || d.file.endsWith(file)).map((d) => ({
      severity: d.severity === "error" ? monaco.MarkerSeverity.Error : d.severity === "warning" ? monaco.MarkerSeverity.Warning : monaco.MarkerSeverity.Info,
      message: d.message + (d.hint ? `\nhint: ${d.hint}` : ""), startLineNumber: Math.max(1, d.line), startColumn: Math.max(1, d.col),
      endLineNumber: Math.max(1, d.end_line || d.line), endColumn: Math.max(d.col + 1, d.end_col || d.col + 1),
    }));
    monaco.editor.setModelMarkers(model, "plx", marks);
  }, [diagnostics, file, value]);
  useEffect(() => {
    if (revealLine && ed.current) {
      ed.current.revealLineInCenter(revealLine);
      ed.current.setPosition({ lineNumber: revealLine, column: 1 });
      ed.current.focus();
    }
  }, [revealLine]);
  return (
    <Editor height="100%" language={language} theme="plyxis" value={value} onMount={onMount}
      onChange={(v) => onChange?.(v ?? "")}
      options={{ readOnly, fontSize: 12.5, fontFamily: "JetBrains Mono", minimap: { enabled: false }, scrollBeyondLastLine: false,
        renderWhitespace: "none", lineNumbersMinChars: 3, tabSize: 2, automaticLayout: true, smoothScrolling: true }} />
  );
}
