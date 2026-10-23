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
      { token: "comment", foreground: "566677" }, { token: "comment.doc", foreground: "7aa2a0", fontStyle: "italic" },
      { token: "keyword", foreground: "c792ea" }, { token: "keyword.section", foreground: "2dd4bf", fontStyle: "bold" },
      { token: "type", foreground: "f59e0b" }, { token: "type.unit", foreground: "8bd5ca" },
      { token: "predefined", foreground: "82aaff" }, { token: "number", foreground: "f78c6c" },
      { token: "string", foreground: "c3e88d" }, { token: "operator", foreground: "89ddff" },
    ],
    colors: { "editor.background": "#0a1017", "editor.lineHighlightBackground": "#101a24", "editorGutter.background": "#0a1017",
      "editorLineNumber.foreground": "#2c3d50", "editor.selectionBackground": "#1f5f5755" },
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
