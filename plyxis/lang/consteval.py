"""Elaboration-time (constant) expression evaluation.

Used for parameters, overrides, generate-loops, name templates and data tables.
Values are plain Python objects. Numbers carry a *canonical* flag so that the
elaborator can decide whether a declaration's unit must be applied:

    q_max = 900            -> raw number, converted with the declared unit
    q_max = r.q_max        -> raw data from a table, converted with the declared unit
    level = 0.5 * tank_cap -> derived from a canonical parameter, no conversion
"""
from __future__ import annotations

import csv
import json
import math
import os
from typing import Any, Callable, Optional

from . import ast as A


class EvalError(Exception):
    def __init__(self, msg: str, loc=None):
        super().__init__(msg)
        self.loc = loc


class Canon(float):
    """A float that is already expressed in canonical (internal) units."""
    __slots__ = ()


def is_canon(v) -> bool:
    return isinstance(v, Canon)


def _num(v, loc=None) -> float:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            pass
    raise EvalError(f"expected a number but got {type(v).__name__} '{v}'", loc)


def _wrap(result: float, *operands) -> float:
    return Canon(result) if any(is_canon(o) for o in operands) else float(result)


_TABLE_CACHE: dict = {}


def _coerce(s: str):
    s2 = s.strip()
    if s2 == "":
        return None
    low = s2.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        f = float(s2)
        return int(f) if f.is_integer() and "." not in s2 and "e" not in low else f
    except ValueError:
        return s2


def load_table(path: str) -> list:
    key = os.path.abspath(path)
    mtime = os.path.getmtime(key)
    hit = _TABLE_CACHE.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    with open(key, encoding="utf-8-sig", newline="") as fh:
        rows = [{k.strip(): _coerce(v or "") for k, v in r.items() if k} for r in csv.DictReader(fh)]
    _TABLE_CACHE[key] = (mtime, rows)
    return rows


def load_geojson(path: str) -> list:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    out = []
    for f in data.get("features", []):
        row = dict(f.get("properties") or {})
        g = f.get("geometry") or {}
        if g.get("type") == "Point":
            lon, lat = g["coordinates"][:2]
            row.setdefault("lat", lat)
            row.setdefault("lon", lon)
        elif g.get("type") == "LineString":
            row["path"] = tuple((c[1], c[0]) for c in g["coordinates"])
        elif g.get("type") == "Polygon":
            row["polygon"] = tuple((c[1], c[0]) for c in g["coordinates"][0])
        out.append(row)
    return out


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


class Scope:
    """Name lookup chain for constant evaluation."""

    def __init__(self, values: Optional[dict] = None, parent: Optional["Scope"] = None,
                 resolver: Optional[Callable[[str], Any]] = None):
        self.values = values or {}
        self.parent = parent
        self.resolver = resolver

    def lookup(self, name: str):
        s = self
        while s is not None:
            if name in s.values:
                return s.values[name]
            if s.resolver is not None:
                v = s.resolver(name)
                if v is not _MISSING:
                    return v
            s = s.parent
        raise KeyError(name)

    def child(self, values: Optional[dict] = None, resolver=None) -> "Scope":
        return Scope(values or {}, self, resolver)


_MISSING = object()
MISSING = _MISSING


def zpad(v, width=2) -> str:
    return str(int(_num(v))).zfill(int(_num(width)))


BUILTINS: dict[str, Callable] = {
    "min": lambda *a: _wrap(min(_num(x) for x in (a[0] if len(a) == 1 and isinstance(a[0], (list, tuple)) else a)), *a),
    "max": lambda *a: _wrap(max(_num(x) for x in (a[0] if len(a) == 1 and isinstance(a[0], (list, tuple)) else a)), *a),
    "abs": lambda x: _wrap(abs(_num(x)), x),
    "sqrt": lambda x: _wrap(math.sqrt(_num(x)), x),
    "exp": lambda x: math.exp(_num(x)),
    "log": lambda x: math.log(_num(x)),
    "log10": lambda x: math.log10(_num(x)),
    "floor": lambda x: _wrap(math.floor(_num(x)), x),
    "ceil": lambda x: _wrap(math.ceil(_num(x)), x),
    "round": lambda x, n=0: _wrap(round(_num(x), int(_num(n))), x),
    "int": lambda x: int(_num(x)),
    "float": lambda x: _num(x),
    "str": lambda x: (str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)),
    "pow": lambda a, b: _wrap(_num(a) ** _num(b), a),
    "sin": lambda x: math.sin(_num(x)), "cos": lambda x: math.cos(_num(x)), "tan": lambda x: math.tan(_num(x)),
    "atan2": lambda a, b: math.atan2(_num(a), _num(b)),
    "zpad": zpad,
    "len": lambda x: len(x),
    "c": lambda *a: tuple(a),
    "list": lambda *a: list(a),
    "sum": lambda x: sum(_num(v) for v in x),
    "mean": lambda x: sum(_num(v) for v in x) / max(1, len(x)),
    "upper": lambda s: str(s).upper(),
    "lower": lambda s: str(s).lower(),
    "concat": lambda *a: "".join(str(x) for x in a),
    "haversine": lambda a, b, c, d: haversine_km(_num(a), _num(b), _num(c), _num(d)),
    "pi": lambda: math.pi,
}


def const_eval(e: A.Expr, scope, *, base_dir: str = ".", extra: Optional[dict] = None):
    """Evaluate an expression at elaboration time."""
    if isinstance(scope, dict):
        scope = Scope(scope)
    ev = lambda x: const_eval(x, scope, base_dir=base_dir, extra=extra)  # noqa: E731
    if isinstance(e, A.Num):
        if e.unit:
            from .units import parse_unit
            return Canon(float(e.value) * parse_unit(e.unit)[0])
        return float(e.value)
    if isinstance(e, A.Str):
        return e.value
    if isinstance(e, A.Bool):
        return e.value
    if isinstance(e, A.Name):
        try:
            return scope.lookup(e.id)
        except KeyError:
            if e.id == "pi":
                return math.pi
            if e.id in ("inf", "Inf"):
                return math.inf
            raise EvalError(f"unknown name '{e.id}' in constant expression", e.loc)
    if isinstance(e, A.Template):
        return template_str(e, scope, base_dir=base_dir, extra=extra)
    if isinstance(e, A.Member):
        obj = ev(e.obj)
        attr = e.attr if isinstance(e.attr, str) else template_str(e.attr, scope, base_dir=base_dir, extra=extra)
        if isinstance(obj, dict):
            if attr in obj:
                return obj[attr]
            raise EvalError(f"no field '{attr}'", e.loc)
        if hasattr(obj, "lookup_attr"):
            return obj.lookup_attr(attr, e.loc)
        if isinstance(obj, (tuple, list)) and attr in ("lat", "lon", "x", "y"):
            return obj[0 if attr in ("lat", "x") else 1]
        raise EvalError(f"cannot access '.{attr}' on {type(obj).__name__}", e.loc)
    if isinstance(e, A.TupleExpr):
        return tuple(ev(x) for x in e.items)
    if isinstance(e, A.ListExpr):
        return [ev(x) for x in e.items]
    if isinstance(e, A.Range):
        a, b = int(_num(ev(e.start), e.loc)), int(_num(ev(e.end), e.loc))
        return list(range(a, b + 1))
    if isinstance(e, A.Unary):
        v = ev(e.operand)
        if e.op == "-":
            return _wrap(-_num(v, e.loc), v)
        if e.op == "not":
            return not bool(v)
    if isinstance(e, A.Binary):
        if e.op == "and":
            return bool(ev(e.left)) and bool(ev(e.right))
        if e.op == "or":
            return bool(ev(e.left)) or bool(ev(e.right))
        lv, rv = ev(e.left), ev(e.right)
        if e.op == "+" and (isinstance(lv, str) or isinstance(rv, str)):
            return f"{lv}{rv}"
        a, b = _num(lv, e.loc), _num(rv, e.loc)
        if e.op == "+":
            return _wrap(a + b, lv, rv)
        if e.op == "-":
            return _wrap(a - b, lv, rv)
        if e.op == "*":
            return _wrap(a * b, lv, rv)
        if e.op == "/":
            if b == 0:
                raise EvalError("division by zero", e.loc)
            return _wrap(a / b, lv, rv)
        if e.op == "%":
            return _wrap(a % b, lv, rv)
        if e.op == "^":
            return _wrap(a ** b, lv)
    if isinstance(e, A.Compare):
        vals = [ev(x) for x in e.operands]
        for op, a, b in zip(e.ops, vals, vals[1:]):
            if isinstance(a, str) or isinstance(b, str):
                ok = {"==": a == b, "!=": a != b}.get(op)
                if ok is None:
                    raise EvalError(f"cannot compare strings with '{op}'", e.loc)
            else:
                a, b = _num(a), _num(b)
                ok = {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b, "==": a == b, "!=": a != b}[op]
            if not ok:
                return False
        return True
    if isinstance(e, A.Call):
        f = e.func
        if f == "if":
            if len(e.args) != 3:
                raise EvalError("if(cond, a, b) takes 3 arguments", e.loc)
            return ev(e.args[1]) if bool(ev(e.args[0])) else ev(e.args[2])
        if f == "table":
            path = _resolve_path(str(ev(e.args[0])), base_dir)
            rows = load_table(path)
            where = e.kwargs.get("where")
            if where is not None:
                rows = [r for r in rows if const_eval(where, scope.child(dict(r)) if isinstance(scope, Scope) else r,
                                                      base_dir=base_dir, extra=extra)]
            return rows
        if f == "geojson":
            return load_geojson(_resolve_path(str(ev(e.args[0])), base_dir))
        if extra and f in extra:
            return extra[f](*[ev(a) for a in e.args], **{k: ev(v) for k, v in e.kwargs.items()})
        if f in BUILTINS:
            args = [ev(a) for a in e.args]
            try:
                return BUILTINS[f](*args)
            except EvalError:
                raise
            except Exception as ex:  # noqa: BLE001
                raise EvalError(f"{f}(): {ex}", e.loc)
        raise EvalError(f"function '{f}' is not available at elaboration time", e.loc)
    if isinstance(e, A.Wildcard):
        return "*"
    raise EvalError(f"cannot evaluate {type(e).__name__} as a constant", getattr(e, "loc", None))


def template_str(t: A.Template, scope, *, base_dir=".", extra=None) -> str:
    out = []
    for p in t.parts:
        if isinstance(p, str):
            out.append(p)
        else:
            v = const_eval(p, scope, base_dir=base_dir, extra=extra)
            if isinstance(v, float) and v.is_integer():
                v = int(v)
            out.append(str(v))
    return "".join(out)


def _resolve_path(p: str, base_dir: str) -> str:
    if os.path.isabs(p):
        return p
    cand = os.path.join(base_dir, p)
    if os.path.exists(cand):
        return cand
    raise EvalError(f"data file not found: {cand}")


def refs_names(e) -> set:
    """Names referenced by an expression (for dependency ordering)."""
    out = set()

    def walk(x):
        if isinstance(x, A.Name):
            out.add(x.id)
        elif isinstance(x, A.Member):
            walk(x.obj)
        elif isinstance(x, A.Call):
            for a in x.args:
                walk(a)
            for a in x.kwargs.values():
                walk(a)
        elif isinstance(x, A.Unary):
            walk(x.operand)
        elif isinstance(x, A.Binary):
            walk(x.left); walk(x.right)
        elif isinstance(x, A.Compare):
            for o in x.operands:
                walk(o)
        elif isinstance(x, (A.TupleExpr, A.ListExpr)):
            for o in x.items:
                walk(o)
        elif isinstance(x, A.Range):
            walk(x.start); walk(x.end)
        elif isinstance(x, A.Template):
            for p in x.parts:
                if not isinstance(p, str):
                    walk(p)
    walk(e)
    return out
