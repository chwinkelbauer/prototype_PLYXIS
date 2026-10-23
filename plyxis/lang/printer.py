"""PLX pretty printer (AST -> source).

Used by the data-integration layer: twin changes discovered from live data are
written back as ordinary PLX statements, reviewable like any code change.
"""
from __future__ import annotations

from typing import Optional

from . import ast as A
from .consteval import interp_str, template_str

PREC = {"or": 1, "and": 2, "+": 6, "-": 6, "*": 7, "/": 7, "%": 7, "^": 9}


def expr(e, scope: Optional[dict] = None) -> str:
    if isinstance(e, A.Num):
        v = e.value
        s = str(int(v)) if float(v).is_integer() and abs(v) < 1e15 else repr(float(v))
        return f"{s} [{e.unit}]" if e.unit else s
    if isinstance(e, A.Str):
        return '"' + interp_str(e.value, scope).replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(e, A.Bool):
        return "true" if e.value else "false"
    if isinstance(e, A.Name):
        return e.id
    if isinstance(e, A.Template):
        if scope is not None:
            return template_str(e, scope)
        return "".join(p if isinstance(p, str) else "{" + expr(p) + "}" for p in e.parts)
    if isinstance(e, A.Member):
        attr = e.attr if isinstance(e.attr, str) else expr(e.attr, scope)
        return f"{expr(e.obj, scope)}.{attr}"
    if isinstance(e, A.Call):
        args = [expr(a, scope) for a in e.args] + [f"{k} = {expr(v, scope)}" for k, v in e.kwargs.items()]
        return f"{e.func}({', '.join(args)})"
    if isinstance(e, A.Unary):
        return f"-{_wrap(e.operand, 8, scope)}" if e.op == "-" else f"not {_wrap(e.operand, 3, scope)}"
    if isinstance(e, A.Binary):
        p = PREC.get(e.op, 5)
        return f"{_wrap(e.left, p, scope)} {e.op} {_wrap(e.right, p + 1, scope)}"
    if isinstance(e, A.Compare):
        out = expr(e.operands[0], scope)
        for op, o in zip(e.ops, e.operands[1:]):
            out += f" {op} {expr(o, scope)}"
        return out
    if isinstance(e, A.TupleExpr):
        return "(" + ", ".join(expr(x, scope) for x in e.items) + ")"
    if isinstance(e, A.ListExpr):
        return "[" + ", ".join(expr(x, scope) for x in e.items) + "]"
    if isinstance(e, A.Range):
        return f"{expr(e.start, scope)}..{expr(e.end, scope)}"
    if isinstance(e, A.Wildcard):
        return "*"
    return f"<{type(e).__name__}>"


def _wrap(e, prec: int, scope) -> str:
    s = expr(e, scope)
    if isinstance(e, A.Binary) and PREC.get(e.op, 5) < prec:
        return f"({s})"
    return s


def stmt(s, indent: int = 4, scope: Optional[dict] = None) -> str:
    pad = " " * indent
    if isinstance(s, A.InitStmt):
        name = expr(s.name, scope)
        typ = expr(s.type, scope)
        ov = ""
        if s.overrides:
            ov = " { " + " ".join(f"{o.name} = {expr(o.value, scope)}{' [' + o.unit + ']' if o.unit else ''};"
                                   for o in s.overrides) + " }"
        return f"{pad}init {name} as {typ}{ov};"
    if isinstance(s, A.ConnectStmt):
        via = ""
        if s.via is not None:
            via = f" via {expr(s.via, scope)}"
            if s.overrides:
                via += " { " + " ".join(f"{o.name} = {expr(o.value, scope)};" for o in s.overrides) + " }"
        name = f" as {expr(s.name, scope)}" if s.name is not None else ""
        return f"{pad}connect({expr(s.src, scope)}, {expr(s.dst, scope)}){via}{name};"
    if isinstance(s, A.SetStmt):
        return f"{pad}set {expr(s.path, scope)} = {expr(s.value, scope)}{' [' + s.unit + ']' if s.unit else ''};"
    if isinstance(s, A.Define):
        return f"{pad}{expr(s.target, scope)} = {expr(s.value, scope)}{' [' + s.unit + ']' if s.unit else ''};"
    if isinstance(s, A.ForStmt):
        body = "\n".join(stmt(b, indent + 2, scope) for b in s.body)
        return f"{pad}for {s.var} in {expr(s.iterable, scope)} {{\n{body}\n{pad}}}"
    return f"{pad}# <{type(s).__name__}>"


def set_line(env: str, path: str, prop: str, value: float, unit: Optional[str] = None, comment: str = "") -> str:
    v = f"{value:.6g}"
    u = f" [{unit}]" if unit else ""
    c = f"   ## {comment}" if comment else ""
    return f"set {env}.{path}.{prop} = {v}{u};{c}"
