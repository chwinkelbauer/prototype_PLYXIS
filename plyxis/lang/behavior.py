"""Behavior compiler: entity statements -> typed dataflow IR.

PLX behavior has *synchronous dataflow* semantics (like Lustre or VHDL
processes):

    x = expr;          local signal, same time step
    port.res = expr;   out-port flow (rate), same time step
    s <- expr;         next value of state s   (s[t+1] = expr[t])
    a <= b;            constraint (hard unless soft(w))
    cost c = expr;     objective term (rate per hour; revenue / emission / metric alike)

The result is a small graph per (specialised) entity type: computed nodes with
their IR expressions and their instant dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from . import ast as A
from . import units as U
from .errors import Diagnostics, Loc, NOLOC, suggest
from .library import EntityType, Library


# ----------------------------------------------------------------------------- IR
@dataclass
class IR:
    pass


@dataclass
class C(IR):
    value: float


@dataclass
class Ref(IR):
    kind: str        # param | dec | state | in | node | env | b | agg
    name: str


@dataclass
class Op(IR):
    op: str
    args: list
    extra: Any = None


UNARY = {"abs", "sqrt", "exp", "log", "log10", "sin", "cos", "tanh", "sigmoid", "floor", "ceil", "round",
         "sign", "step", "pos", "negpart"}
NARY = {"min", "max", "sum", "mean"}
BUILTIN_NAMES = {"dt", "t", "time", "hour", "dow", "day"}
EPS = 1e-9


def walk_refs(e: IR, out: Optional[set] = None) -> set:
    out = set() if out is None else out
    if isinstance(e, Ref):
        out.add((e.kind, e.name))
    elif isinstance(e, Op):
        for a in e.args:
            walk_refs(a, out)
        if e.op == "delay" and isinstance(e.extra, dict):
            k = e.extra.get("k")
            if isinstance(k, IR):
                walk_refs(k, out)
    return out


def delayed_refs(e: IR, out: Optional[set] = None, inside: bool = False) -> set:
    """Refs that are read through delay() (not instant dependencies)."""
    out = set() if out is None else out
    if isinstance(e, Ref) and inside:
        out.add((e.kind, e.name))
    elif isinstance(e, Op):
        for a in e.args:
            delayed_refs(a, out, inside or e.op == "delay")
    return out


def instant_refs(e: IR, out: Optional[set] = None) -> set:
    out = set() if out is None else out
    if isinstance(e, Ref):
        out.add((e.kind, e.name))
    elif isinstance(e, Op):
        if e.op == "delay":
            k = e.extra.get("k") if isinstance(e.extra, dict) else None
            if isinstance(k, IR):
                instant_refs(k, out)
            return out
        for a in e.args:
            instant_refs(a, out)
    return out


def is_const(e: IR) -> bool:
    return isinstance(e, C)


# ----------------------------------------------------------------------------- type IR
@dataclass
class PortInfo:
    name: str
    direction: str
    resources: list
    channels: list           # list[(res, attr|None)]
    exclusive: bool = False


@dataclass
class DecInfo:
    name: str
    domain: str
    lo: IR
    hi: IR
    default: Optional[IR]
    hold: float               # in steps (if hold_hours is None) else hours
    hold_hours: Optional[float]
    unit: Optional[str]
    doc: str = ""


@dataclass
class StateInfo:
    name: str
    unit: Optional[str]
    lo: Optional[IR] = None
    hi: Optional[IR] = None
    has_update: bool = False
    doc: str = ""


@dataclass
class ConInfo:
    name: str
    kind: str                 # le | ge | eq
    lhs: IR
    rhs: IR
    soft: Optional[IR] = None
    loc: Loc = NOLOC
    is_bound: bool = False


@dataclass
class TermInfo:
    kind: str
    name: str
    final: bool = False


@dataclass
class TypeIR:
    key: str
    type: EntityType
    ports: dict = field(default_factory=dict)
    params: list = field(default_factory=list)
    param_units: dict = field(default_factory=dict)
    states: dict = field(default_factory=dict)
    decisions: dict = field(default_factory=dict)
    nodes: dict = field(default_factory=dict)        # key -> IR
    order: list = field(default_factory=list)
    cons: dict = field(default_factory=dict)         # con key -> ConInfo
    terms: dict = field(default_factory=dict)        # term key -> TermInfo
    finals: dict = field(default_factory=dict)       # final term key -> IR (refs to final states)
    units: dict = field(default_factory=dict)        # node key -> display unit
    feed: dict = field(default_factory=dict)         # out port -> set(in ports)
    telemetry: list = field(default_factory=list)    # (signal key, tag suffix, unit, options)
    aggs: dict = field(default_factory=dict)         # env only: agg name -> (selector, signal path list, how)
    local_names: list = field(default_factory=list)

    def in_channels(self):
        for p in self.ports.values():
            if p.direction == "in":
                for (r, a) in p.channels:
                    yield p.name, r, a

    def out_channels(self):
        for p in self.ports.values():
            if p.direction == "out":
                for (r, a) in p.channels:
                    yield p.name, r, a


def chan_key(port: str, res: str, attr: Optional[str]) -> str:
    r = res.split(".")[-1]
    return f"{port}:{r}" + (f":{attr}" if attr else "")


# ----------------------------------------------------------------------------- compiler
class BehaviorCompiler:
    def __init__(self, lib: Library, diags: Diagnostics):
        self.lib = lib
        self.diags = diags

    def compile(self, key: str, typ: EntityType, port_resources: dict, is_env: bool = False,
                env_params: Optional[dict] = None, env_units: Optional[dict] = None,
                agg_resolver=None) -> TypeIR:
        ir = TypeIR(key=key, type=typ)
        self.ir = ir
        self.typ = typ
        self.is_env = is_env
        self.env_params = env_params or {}
        self.env_units = env_units or {}
        self.agg_resolver = agg_resolver
        # ports & channels
        for pname, decl in typ.ports.items():
            if is_env:
                continue
            res = port_resources.get(pname, [])
            chans = []
            for r in res:
                chans.append((r, None))
                rt = self.lib.resources.get(r)
                if rt:
                    for a in rt.attrs:
                        chans.append((r, a))
            ir.ports[pname] = PortInfo(pname, decl.direction, list(res), chans, decl.exclusive)
        ir.params = list(typ.params)
        ir.param_units = {n: p.unit for n, p in typ.params.items()}
        for n, s in typ.states.items():
            ir.states[n] = StateInfo(n, s.unit, doc=s.doc)
        # pass 1 - collect definitions
        local_defs: dict[str, A.Define] = {}
        out_defs: dict[str, tuple] = {}         # chan key -> (expr AST or IR, loc)
        port_copies: list = []
        updates: dict[str, A.Update] = {}
        for st in typ.behavior:
            if isinstance(st, A.Define):
                tgt = st.target
                if isinstance(tgt, A.Name):
                    n = tgt.id
                    if n in ir.ports and ir.ports[n].direction == "out":
                        port_copies.append(st)
                    elif n in typ.params or n in typ.states or n in typ.decisions:
                        kind = "property" if n in typ.params else ("state" if n in typ.states else "decision")
                        hint = f"use '{n} <- ...;' for state updates" if kind == "state" else ""
                        self.diags.error(f"cannot assign to {kind} '{n}' in behavior", st.loc, "E-ASSIGN", hint)
                    elif n in ir.ports:
                        self.diags.error(f"cannot assign to in-port '{n}'", st.loc, "E-ASSIGN")
                    elif n in local_defs:
                        self.diags.error(f"'{n}' is defined twice (signals are single-assignment)", st.loc,
                                         "E-ASSIGN", hint=f"first definition at line {local_defs[n].loc.line}")
                    else:
                        local_defs[n] = st
                elif isinstance(tgt, A.Member):
                    path = _member_path(tgt)
                    if path is None or path[0] not in ir.ports:
                        self.diags.error("assignment target must be a local name or an out-port channel "
                                         "(port.resource)", st.loc, "E-ASSIGN")
                        continue
                    p = ir.ports[path[0]]
                    if p.direction != "out":
                        self.diags.error(f"cannot assign to in-port '{path[0]}'", st.loc, "E-ASSIGN")
                        continue
                    if len(path) not in (2, 3):
                        self.diags.error("expected port.resource or port.resource.attribute", st.loc, "E-ASSIGN")
                        continue
                    res = self._port_res(p, path[1])
                    if res is None:
                        if p.resources and self.typ.ports[p.name].resources != ["*"]:
                            self.diags.error(f"port '{p.name}' does not carry '{path[1]}'", st.loc, "E-ASSIGN",
                                             hint=suggest(path[1], [r.split('.')[-1] for r in p.resources]))
                        continue          # wildcard specialisation without this resource
                    attr = path[2] if len(path) == 3 else None
                    if attr and (res, attr) not in p.channels:
                        self.diags.error(f"resource '{path[1]}' has no attribute '{attr}'", st.loc, "E-ASSIGN")
                        continue
                    k = chan_key(p.name, res, attr)
                    if k in out_defs:
                        self.diags.error(f"'{p.name}.{path[1]}{'.' + attr if attr else ''}' assigned twice",
                                         st.loc, "E-ASSIGN")
                        continue
                    out_defs[k] = (st.value, st.loc, attr is not None, res)
                else:
                    self.diags.error("invalid assignment target", st.loc, "E-ASSIGN")
            elif isinstance(st, A.Update):
                if not isinstance(st.target, A.Name):
                    self.diags.error("state updates need a state name on the left of '<-'", st.loc, "E-ASSIGN")
                    continue
                n = st.target.id
                if n not in typ.states:
                    if n in ("costs", "cost", "revenues", "revenue"):
                        self.diags.error(f"legacy '{n} <- ...' : declare an objective term instead",
                                         st.loc, "E-LEGACY", hint=f"cost <name> = expr;")
                    else:
                        self.diags.error(f"'{n}' is not a declared state", st.loc, "E-ASSIGN",
                                         hint=f"declare it under 'state:' (e.g. {n} = 0;) or use '=' for a "
                                              "same-step signal")
                    continue
                if n in updates:
                    self.diags.error(f"state '{n}' updated twice", st.loc, "E-ASSIGN")
                    continue
                updates[n] = st
            elif isinstance(st, A.DecisionStmtLegacy):
                self.diags.warning(f"legacy decision statement '{st.text}' ignored: declare decisions under "
                                   "'decision:' with bounds", st.loc, "W-LEGACY")
            elif isinstance(st, A.ExprStmt):
                self.diags.warning("expression statement has no effect", st.loc, "W-EXPR")
        ir.local_names = list(local_defs)
        self.local_defs = local_defs
        # decisions
        for n, d in typ.decisions.items():
            lo = self.expr(d.lo, static=True) if d.lo is not None else C(0.0)
            hi = self.expr(d.hi, static=True) if d.hi is not None else C(1.0)
            fac = U.factor(d.unit) if d.unit else 1.0
            if fac != 1.0:
                lo = _scale_raw(lo, fac, d.lo)
                hi = _scale_raw(hi, fac, d.hi)
            default = None
            if d.default is not None:
                default = self.expr(d.default, static=True)
                dfac = U.factor(d.default_unit) if d.default_unit else fac
                if dfac != 1.0:
                    default = _scale_raw(default, dfac, d.default)
            hold, hold_h = 1.0, None
            if d.hold is not None:
                hv = self.expr(d.hold, static=True)
                if not isinstance(hv, C):
                    self.diags.error("'hold' must be a constant", d.loc, "E-DEC")
                else:
                    if d.hold_unit:
                        hold_h = hv.value * U.factor(d.hold_unit)
                    else:
                        hold = max(1.0, round(hv.value))
            ir.decisions[n] = DecInfo(n, d.domain, lo, hi, default, hold, hold_h, d.unit, d.doc)
        # state bounds
        for n, s in typ.states.items():
            if s.bounds is not None:
                lo = self.expr(s.bounds[0], static=True)
                hi = self.expr(s.bounds[1], static=True)
                fac = U.factor(s.unit) if s.unit else 1.0
                if fac != 1.0:
                    lo, hi = _scale_raw(lo, fac, s.bounds[0]), _scale_raw(hi, fac, s.bounds[1])
                ir.states[n].lo, ir.states[n].hi = lo, hi
        # pass 2 - expressions
        self._resolving: list = []
        for n, st in local_defs.items():
            self._local(n)
        for k, (expr, loc, is_attr, res) in out_defs.items():
            v = self.expr(expr)
            if is_attr:
                base_k = k.rsplit(":", 1)[0]
                v = Op("mul", [Ref("node", f"out:{base_k}"), v])
            ir.nodes[f"out:{k}"] = v
        for st in port_copies:
            src = st.value
            if not (isinstance(src, A.Name) and src.id in ir.ports and ir.ports[src.id].direction == "in"):
                self.diags.error("port assignment 'out = in' needs an in-port on the right", st.loc, "E-ASSIGN")
                continue
            sp, dp = ir.ports[src.id], ir.ports[st.target.id]
            for (r, a) in dp.channels:
                k = chan_key(dp.name, r, a)
                if f"out:{k}" in ir.nodes:
                    continue
                if (r, a) in sp.channels:
                    ir.nodes[f"out:{k}"] = Ref("in", chan_key(sp.name, r, a))
                else:
                    ir.nodes[f"out:{k}"] = C(0.0)
        # unassigned out channels
        for pname, r, a in list(ir.out_channels()):
            k = chan_key(pname, r, a)
            if f"out:{k}" in ir.nodes:
                continue
            if a is None:
                if not is_env:
                    self.diags.warning(f"'{typ.short}': output {pname}.{r.split('.')[-1]} is never assigned "
                                       "(flow is zero)", typ.loc, "W-OUT")
                ir.nodes[f"out:{k}"] = C(0.0)
            else:
                ir.nodes[f"out:{k}"] = self._attr_default(pname, r, a)
        # state updates
        for n, st in updates.items():
            ir.nodes[f"next:{n}"] = self.expr(st.value)
            ir.states[n].has_update = True
        # constraints, terms
        cidx = 0
        for st in typ.behavior + (typ.constraints if is_env else []):
            if isinstance(st, A.ConstraintStmt):
                cmp = st.expr
                pairs = list(zip(cmp.ops, cmp.operands, cmp.operands[1:]))
                soft = self.expr(st.soft, static=True) if st.soft is not None else None
                for j, (op, l, r) in enumerate(pairs):
                    if op == "!=":
                        self.diags.error("'!=' cannot be used as a constraint", st.loc, "E-CON")
                        continue
                    base = st.name or _auto_con_name(l, r, cidx)
                    name = base if len(pairs) == 1 else f"{base}.{'lo' if j == 0 else 'hi'}"
                    while f"con:{name}" in ir.cons:
                        cidx += 1
                        name = f"{base}_{cidx}"
                    cidx += 1
                    kind = {"<": "le", "<=": "le", ">": "ge", ">=": "ge", "==": "eq"}[op]
                    lhs, rhs = self.expr(l), self.expr(r)
                    self._check_dims(lhs, rhs, st.loc, f"constraint '{name}'")
                    ir.cons[f"con:{name}"] = ConInfo(name, kind, lhs, rhs, soft, st.loc)
                    ir.nodes[f"con:{name}"] = _violation(kind, lhs, rhs)
            elif isinstance(st, A.TermStmt):
                key = f"term:{st.kind}:{st.name}"
                if key in ir.terms or key.replace("term:", "final:") in ir.finals:
                    self.diags.error(f"{st.kind} '{st.name}' defined twice", st.loc, "E-TERM")
                    continue
                if st.final:
                    ir.finals[key.replace("term:", "final:")] = self.expr(st.expr)
                    ir.terms[key.replace("term:", "final:")] = TermInfo(st.kind, st.name, True)
                else:
                    ir.nodes[key] = self.expr(st.expr)
                    ir.terms[key] = TermInfo(st.kind, st.name, False)
        # state bounds -> constraints on next values (x[t+1])
        for n, s in ir.states.items():
            if s.lo is None:
                continue
            target = Ref("node", f"next:{n}") if s.has_update else Ref("state", n)
            ir.cons[f"con:{n}.lo"] = ConInfo(f"{n}.lo", "ge", target, s.lo, None, typ.states[n].loc, True)
            ir.nodes[f"con:{n}.lo"] = _violation("ge", target, s.lo)
            ir.cons[f"con:{n}.hi"] = ConInfo(f"{n}.hi", "le", target, s.hi, None, typ.states[n].loc, True)
            ir.nodes[f"con:{n}.hi"] = _violation("le", target, s.hi)
        # telemetry
        for tdecl in typ.telemetry:
            sk = self._signal_key(tdecl.signal, tdecl.loc)
            if sk is not None:
                ir.telemetry.append((sk, tdecl.tag, tdecl.unit, tdecl.options))
        self._order_and_feed()
        return ir

    # ------------------------------------------------------------------ helpers
    def _port_res(self, p: PortInfo, short: str) -> Optional[str]:
        for r in p.resources:
            if r == short or r.split(".")[-1] == short:
                return r
        return None

    def _attr_default(self, pname: str, res: str, attr: str) -> IR:
        flow = Ref("node", f"out:{chan_key(pname, res, None)}")
        mass, tot = [], []
        for p in self.ir.ports.values():
            if p.direction == "in" and (res, attr) in p.channels:
                mass.append(Ref("in", chan_key(p.name, res, attr)))
                tot.append(Ref("in", chan_key(p.name, res, None)))
        if mass:
            m = mass[0] if len(mass) == 1 else Op("sum", mass)
            q = tot[0] if len(tot) == 1 else Op("sum", tot)
            return Op("mul", [flow, Op("safediv", [m, q])])
        rt = self.lib.resources.get(res)
        return Op("mul", [flow, C(rt.attrs.get(attr, 0.0) if rt else 0.0)])

    def _signal_key(self, name: str, loc: Loc) -> Optional[str]:
        parts = name.split(".")
        n = parts[0]
        if len(parts) == 1:
            if n in self.local_defs:
                return f"node:local:{n}"
            if n in self.typ.states:
                return f"state:{n}"
            if n in self.typ.decisions:
                return f"dec:{n}"
            if n in self.typ.params:
                return f"param:{n}"
        elif n in self.ir.ports:
            p = self.ir.ports[n]
            res = self._port_res(p, parts[1])
            if res is not None:
                k = chan_key(n, res, parts[2] if len(parts) > 2 else None)
                return f"node:out:{k}" if p.direction == "out" else f"in:{k}"
            if self.typ.ports[n].resources == ["*"]:
                return None
        self.diags.error(f"telemetry binds unknown signal '{name}' in '{self.typ.short}'", loc, "E-TELE",
                         hint=suggest(n, list(self.local_defs) + list(self.typ.states)))
        return None

    def _local(self, n: str) -> IR:
        key = f"local:{n}"
        if key in self.ir.nodes:
            return Ref("node", key)
        if n in self._resolving:
            cyc = self._resolving[self._resolving.index(n):] + [n]
            self.diags.error(f"algebraic loop between signals: {' -> '.join(cyc)}",
                             self.local_defs[n].loc, "E-LOOP",
                             hint="break the loop with a state (s <- ...) or delay()")
            self.ir.nodes[key] = C(0.0)
            return Ref("node", key)
        self._resolving.append(n)
        st = self.local_defs[n]
        v = self.expr(st.value)
        if st.unit:
            if is_const(v) and not _has_unit_literal(st.value):
                v = C(v.value * U.factor(st.unit))
            self.ir.units[key] = st.unit
        self._resolving.pop()
        self.ir.nodes[key] = v
        return Ref("node", key)

    # ------------------------------------------------------------------ expressions
    def expr(self, e: A.Expr, static: bool = False, fn_scope: Optional[dict] = None) -> IR:
        ex = lambda x: self.expr(x, static, fn_scope)  # noqa: E731
        if isinstance(e, A.Num):
            if e.unit:
                try:
                    return C(float(e.value) * U.parse_unit(e.unit)[0])
                except U.UnitError as ex:
                    self.diags.warning(str(ex), e.loc, "W-UNIT")
            return C(float(e.value))
        if isinstance(e, A.Bool):
            return C(1.0 if e.value else 0.0)
        if isinstance(e, A.Str):
            self.diags.error("strings are not allowed in behavior expressions", e.loc, "E-EXPR")
            return C(0.0)
        if isinstance(e, A.Name):
            return self._name(e.id, e.loc, static, fn_scope)
        if isinstance(e, A.Member):
            return self._member(e, static)
        if isinstance(e, A.Unary):
            v = ex(e.operand)
            if e.op == "-":
                return C(-v.value) if is_const(v) else Op("neg", [v])
            if e.op == "not":
                return Op("not", [v])
        if isinstance(e, A.Binary):
            l, r = ex(e.left), ex(e.right)
            op = {"+": "add", "-": "sub", "*": "mul", "/": "div", "^": "pow", "%": "mod",
                  "and": "and", "or": "or"}[e.op]
            if is_const(l) and is_const(r):
                return C(_fold(op, l.value, r.value))
            if op in ("add", "sub"):
                self._check_dims(l, r, e.loc, f"'{e.op}'")
            return Op(op, [l, r])
        if isinstance(e, A.Compare):
            vals = [ex(x) for x in e.operands]
            parts = []
            for op, a, b in zip(e.ops, vals, vals[1:]):
                parts.append(Op({"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}[op], [a, b]))
            out = parts[0]
            for p in parts[1:]:
                out = Op("and", [out, p])
            return out
        if isinstance(e, A.Call):
            return self._call(e, static, fn_scope)
        if isinstance(e, (A.TupleExpr, A.ListExpr)):
            self.diags.error("tuples/lists are only allowed as pwl() tables", e.loc, "E-EXPR")
            return C(0.0)
        self.diags.error(f"unsupported expression ({type(e).__name__})", getattr(e, "loc", NOLOC), "E-EXPR")
        return C(0.0)

    def _name(self, n: str, loc: Loc, static: bool, fn_scope) -> IR:
        if fn_scope and n in fn_scope:
            return fn_scope[n]
        if n in getattr(self, "local_defs", {}) and not static:
            return self._local(n)
        t = self.typ
        if n in t.params:
            return Ref("param", n)
        if static:
            if n in BUILTIN_NAMES and n == "dt":
                return Ref("b", "dt")
            self.diags.error(f"'{n}' cannot be used here (only properties and constants)", loc, "E-STATIC")
            return C(0.0)
        if n in t.states:
            return Ref("state", n)
        if n in t.decisions:
            return Ref("dec", n)
        if n in BUILTIN_NAMES:
            return Ref("b", {"day": "dow"}.get(n, n))
        if n == "pi":
            return C(3.141592653589793)
        if n in self.ir.ports:
            self.diags.error(f"port '{n}' needs a resource, e.g. {n}.{(self.ir.ports[n].resources or ['oil'])[0].split('.')[-1]}"
                             " (or sum(" + n + ") for the total)", loc, "E-PORT")
            return C(0.0)
        if self.is_env and self.agg_resolver is not None:
            r = self.agg_resolver(self, [n], "value", loc)
            if r is not None:
                return r
        cands = list(t.params) + list(t.states) + list(t.decisions) + list(getattr(self, "local_defs", {}))
        self.diags.error(f"unknown name '{n}' in '{t.short}'", loc, "E-NAME", hint=suggest(n, cands))
        return C(0.0)

    def _member(self, e: A.Member, static: bool) -> IR:
        path = _member_path(e)
        if path is None:
            self.diags.error("invalid member expression", e.loc, "E-EXPR")
            return C(0.0)
        head = path[0]
        if head == "env":
            if len(path) != 2:
                self.diags.error("use env.<property>", e.loc, "E-ENV")
                return C(0.0)
            if path[1] not in self.env_params:
                self.diags.error(f"environment has no property '{path[1]}'", e.loc, "E-ENV",
                                 hint=suggest(path[1], list(self.env_params)))
                return C(0.0)
            return Ref("env", path[1])
        if head == "self" and len(path) >= 2:
            return self.expr(_rebuild(path[1:], e.loc), static)
        if head in self.ir.ports and not static:
            p = self.ir.ports[head]
            if len(path) < 2:
                return C(0.0)
            res = self._port_res(p, path[1])
            if res is None:
                if self.typ.ports[head].resources == ["*"]:
                    return C(0.0)       # specialisation without this resource
                self.diags.error(f"port '{head}' does not carry '{path[1]}'", e.loc, "E-PORT",
                                 hint=suggest(path[1], [r.split('.')[-1] for r in p.resources]))
                return C(0.0)
            if len(path) == 2:
                k = chan_key(head, res, None)
                return Ref("in", k) if p.direction == "in" else Ref("node", f"out:{k}")
            attr = path[2]
            if (res, attr) not in p.channels:
                self.diags.error(f"resource '{path[1]}' has no attribute '{attr}'", e.loc, "E-PORT")
                return C(0.0)
            if p.direction == "in":
                return Op("safediv", [Ref("in", chan_key(head, res, attr)), Ref("in", chan_key(head, res, None))])
            return Op("safediv", [Ref("node", f"out:{chan_key(head, res, attr)}"),
                                  Ref("node", f"out:{chan_key(head, res, None)}")])
        if self.is_env and self.agg_resolver is not None and not static:
            r = self.agg_resolver(self, path, "value", e.loc)
            if r is not None:
                return r
        self.diags.error(f"unknown reference '{'.'.join(path)}'", e.loc, "E-NAME")
        return C(0.0)

    def _call(self, e: A.Call, static: bool, fn_scope) -> IR:
        f = e.func
        args = e.args
        ex = lambda x: self.expr(x, static, fn_scope)  # noqa: E731
        uf = self.lib.function(f, self.typ.package)
        if uf is not None:
            if len(args) != len(uf.args):
                self.diags.error(f"{f}() takes {len(uf.args)} arguments", e.loc, "E-CALL")
                return C(0.0)
            scope = dict(fn_scope or {})
            for a_name, a_expr in zip(uf.args, args):
                scope[a_name] = ex(a_expr)
            return self.expr(uf.body, static, scope)
        if f == "sum" and len(args) == 1 and isinstance(args[0], A.Name) and args[0].id in self.ir.ports:
            p = self.ir.ports[args[0].id]
            refs = []
            for (r, a) in p.channels:
                if a is None:
                    k = chan_key(p.name, r, None)
                    refs.append(Ref("in", k) if p.direction == "in" else Ref("node", f"out:{k}"))
            if not refs:
                return C(0.0)
            return refs[0] if len(refs) == 1 else Op("sum", refs)
        if self.is_env and f in ("sum", "mean", "max", "min", "count") and len(args) == 1 and \
                isinstance(args[0], (A.Member, A.Name)) and self.agg_resolver is not None:
            path = _member_path(args[0]) if isinstance(args[0], A.Member) else [args[0].id]
            r = self.agg_resolver(self, path, f, e.loc)
            if r is not None:
                return r
        if f in UNARY:
            if len(args) != 1:
                self.diags.error(f"{f}() takes 1 argument", e.loc, "E-CALL")
                return C(0.0)
            v = ex(args[0])
            if is_const(v):
                return C(_fold1(f, v.value))
            return Op(f, [v])
        if f in NARY:
            vs = [ex(a) for a in args]
            if not vs:
                self.diags.error(f"{f}() needs arguments", e.loc, "E-CALL")
                return C(0.0)
            if len(vs) == 1:
                return vs[0]
            if f in ("min", "max"):
                for v in vs[1:]:
                    self._check_dims(vs[0], v, e.loc, f"{f}()")
            return Op(f, vs)
        if f == "pow":
            return Op("pow", [ex(args[0]), ex(args[1])])
        if f == "sat":
            if len(args) != 2:
                self.diags.error("sat(rate, available) takes 2 arguments", e.loc, "E-CALL")
                return C(0.0)
            return Op("sat", [ex(args[0]), ex(args[1])])
        if f == "clamp":
            if len(args) != 3:
                self.diags.error("clamp(x, lo, hi) takes 3 arguments", e.loc, "E-CALL")
                return C(0.0)
            return Op("clamp", [ex(a) for a in args])
        if f == "if":
            if len(args) != 3:
                self.diags.error("if(cond, a, b) takes 3 arguments", e.loc, "E-CALL")
                return C(0.0)
            return Op("where", [ex(a) for a in args])
        if f in ("smin", "smax"):
            k = ex(e.kwargs["k"]) if "k" in e.kwargs else (ex(args[2]) if len(args) > 2 else C(10.0))
            return Op(f, [ex(args[0]), ex(args[1]), k])
        if f == "pwl":
            if len(args) != 3 or not isinstance(args[1], (A.ListExpr, A.TupleExpr)) or \
                    not isinstance(args[2], (A.ListExpr, A.TupleExpr)):
                self.diags.error("pwl(x, [x0, x1, ...], [y0, y1, ...]) needs constant lists", e.loc, "E-CALL")
                return C(0.0)
            xs = [self.expr(v, True) for v in args[1].items]
            ys = [self.expr(v, True) for v in args[2].items]
            if len(xs) != len(ys) or len(xs) < 2:
                self.diags.error("pwl tables need >= 2 points of equal length", e.loc, "E-CALL")
                return C(0.0)
            if not all(is_const(v) for v in xs):
                self.diags.error("pwl x-breakpoints must be constants", e.loc, "E-CALL")
                return C(0.0)
            return Op("pwl", [ex(args[0])] + ys, extra={"xs": [v.value for v in xs]})
        if f in ("delay", "prev"):
            if static:
                self.diags.error(f"{f}() is not allowed here", e.loc, "E-STATIC")
                return C(0.0)
            x = ex(args[0])
            if f == "prev":
                k = C(1.0)
                init = ex(args[1]) if len(args) > 1 else (ex(e.kwargs["init"]) if "init" in e.kwargs else C(0.0))
            else:
                k = self.expr(args[1], True) if len(args) > 1 else C(1.0)
                init = ex(args[2]) if len(args) > 2 else (ex(e.kwargs["init"]) if "init" in e.kwargs else C(0.0))
            if is_const(k) and k.value < 1:
                self.diags.error("delay must be >= 1 step", e.loc, "E-CALL")
                k = C(1.0)
            if not is_const(init) and not (isinstance(init, Ref) and init.kind in ("param", "env")):
                self.diags.error("delay initial value must be a constant or property", e.loc, "E-CALL")
                init = C(0.0)
            self._ndelay = getattr(self, "_ndelay", 0) + 1
            n = self._ndelay
            self.ir.nodes[f"dsrc:{n}"] = x
            self.ir.nodes[f"delay:{n}"] = Op("delay", [Ref("node", f"dsrc:{n}")], extra={"k": k, "init": init})
            return Ref("node", f"delay:{n}")
        if f in ("hour", "dow", "time"):
            return Ref("b", f)
        if f == "c":
            self.diags.error("vectors c(...) are not supported in behavior", e.loc, "E-CALL")
            return C(0.0)
        self.diags.error(f"unknown function '{f}'", e.loc, "E-CALL",
                         hint=suggest(f, sorted(UNARY | NARY | {"pow", "clamp", "if", "pwl", "delay", "prev",
                                                               "smin", "smax"})))
        return C(0.0)

    # ------------------------------------------------------------------ dims (warnings only)
    def dims(self, e: IR):
        if isinstance(e, C):
            return None
        if isinstance(e, Ref):
            if e.kind == "param":
                return U.dims_of(self.ir.param_units.get(e.name))
            if e.kind == "state":
                s = self.ir.states.get(e.name)
                return U.dims_of(s.unit) if s else None
            if e.kind == "dec":
                d = self.ir.decisions.get(e.name)
                return U.dims_of(d.unit) if d else None
            if e.kind == "env":
                return U.dims_of(self.env_units.get(e.name))
            if e.kind == "b":
                return U.dims_of("h") if e.name in ("dt", "time") else U.ZERO
            if e.kind in ("in", "node"):
                k = e.name.split(":")
                if (e.kind == "in" and len(k) == 2) or (e.kind == "node" and k[0] == "out" and len(k) == 3):
                    port, res = (k[0], k[1]) if e.kind == "in" else (k[1], k[2])
                    p = self.ir.ports.get(port)
                    if p:
                        r = self._port_res(p, res)
                        rt = self.lib.resources.get(r) if r else None
                        if rt and rt.unit:
                            return U.dims_div(U.dims_of(rt.unit), U.dims_of("h"))
                if e.kind == "node" and k[0] == "local":
                    u = self.ir.units.get(e.name)
                    if u:
                        return U.dims_of(u)
                    v = self.ir.nodes.get(e.name)
                    return self.dims(v) if v is not None else None
            return None
        if isinstance(e, Op):
            if e.op in ("add", "sub", "min", "max", "sum", "neg", "abs", "clamp", "pos", "negpart", "floor",
                        "ceil", "round", "smin", "smax", "mean"):
                for a in e.args:
                    d = self.dims(a)
                    if d is not None:
                        return d
                return None
            if e.op == "mul":
                a, b = self.dims(e.args[0]), self.dims(e.args[1])
                if a is None and b is None:
                    return None
                return U.dims_mul(a or U.ZERO, b or U.ZERO) if (a is None) != (b is None) or (a and b) else None
            if e.op == "div":
                a, b = self.dims(e.args[0]), self.dims(e.args[1])
                if a is None and b is None:
                    return None
                return U.dims_div(a or U.ZERO, b or U.ZERO)
            if e.op == "where":
                return self.dims(e.args[1]) or self.dims(e.args[2])
            if e.op == "delay":
                return self.dims(e.args[0])
            return None
        return None

    def _check_dims(self, a: IR, b: IR, loc: Loc, what: str) -> None:
        if is_const(a) or is_const(b):
            return
        da, db = self.dims(a), self.dims(b)
        if da is not None and db is not None and not U.dims_equal(da, db):
            self.diags.warning(f"unit mismatch in {what}: {U.fmt_dims(da)} vs {U.fmt_dims(db)}", loc, "W-UNIT")

    # ------------------------------------------------------------------ ordering & feedthrough
    def _order_and_feed(self) -> None:
        ir = self.ir
        deps = {k: {f"{kind}:{n}" if kind != "node" else n for kind, n in instant_refs(v) if kind in ("node",)}
                for k, v in ir.nodes.items()}
        order, state = [], {}

        def visit(k, stack):
            st = state.get(k)
            if st == 2:
                return
            if st == 1:
                cyc = stack[stack.index(k):] + [k]
                self.diags.error(f"algebraic loop in '{self.typ.short}': {' -> '.join(cyc)}", self.typ.loc, "E-LOOP",
                                 hint="break it with a state or delay()")
                return
            state[k] = 1
            for d in deps.get(k, ()):
                if d in ir.nodes:
                    visit(d, stack + [k])
            state[k] = 2
            order.append(k)

        for k in ir.nodes:
            visit(k, [])
        ir.order = order
        # instant dependence of each node on in-ports (transitive)
        memo: dict = {}

        def ins_of(k, guard=()):
            if k in memo:
                return memo[k]
            out = set()
            v = ir.nodes.get(k)
            if v is not None and k not in guard:
                for kind, n in instant_refs(v):
                    if kind == "in":
                        out.add(n.split(":")[0])
                    elif kind == "node":
                        out |= ins_of(n, guard + (k,))
            memo[k] = out
            return out

        for p in ir.ports.values():
            if p.direction != "out":
                continue
            s = set()
            for (r, a) in p.channels:
                s |= ins_of(f"out:{chan_key(p.name, r, a)}")
            ir.feed[p.name] = s


def _member_path(e) -> Optional[list]:
    if isinstance(e, A.Name):
        return [e.id]
    if isinstance(e, A.Member):
        head = _member_path(e.obj)
        if head is None or not isinstance(e.attr, str):
            return None
        return head + [e.attr]
    return None


def _rebuild(path: list, loc) -> A.Expr:
    e: A.Expr = A.Name(path[0], loc=loc)
    for p in path[1:]:
        e = A.Member(e, p, loc=loc)
    return e


def _violation(kind: str, lhs: IR, rhs: IR) -> IR:
    """Signed residual h: feasible iff h <= 0 (le/ge) or h == 0 (eq). Violation = relu(h) / |h|."""
    if kind == "ge":
        return Op("sub", [rhs, lhs])
    return Op("sub", [lhs, rhs])


def _auto_con_name(l: A.Expr, r: A.Expr, idx: int) -> str:
    def nm(x):
        if isinstance(x, A.Name):
            return x.id
        if isinstance(x, A.Member):
            p = _member_path(x)
            return ".".join(p) if p else None
        return None
    a, b = nm(l), nm(r)
    if a and b:
        return f"{a}_vs_{b}"
    if a:
        return f"{a}_limit"
    if b:
        return f"{b}_limit"
    return f"c{idx}"


def _scale_raw(e: IR, fac: float, src=None) -> IR:
    """Apply a declared unit to literal-only expressions (not to unit-annotated literals)."""
    if isinstance(e, C) and not (src is not None and _has_unit_literal(src)):
        return C(e.value * fac)
    return e


def _has_unit_literal(e) -> bool:
    if isinstance(e, A.Num):
        return bool(e.unit)
    if isinstance(e, A.Unary):
        return _has_unit_literal(e.operand)
    if isinstance(e, A.Binary):
        return _has_unit_literal(e.left) or _has_unit_literal(e.right)
    if isinstance(e, A.Call):
        return any(_has_unit_literal(a) for a in e.args)
    return False


def _fold(op: str, a: float, b: float) -> float:
    import math
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "div":
        return a / b if b != 0 else math.copysign(math.inf, a) if a else 0.0
    if op == "pow":
        return a ** b
    if op == "mod":
        return a % b
    if op == "and":
        return 1.0 if (a and b) else 0.0
    if op == "or":
        return 1.0 if (a or b) else 0.0
    raise ValueError(op)


def _fold1(f: str, x: float) -> float:
    import math
    return {
        "abs": abs, "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "log10": math.log10,
        "sin": math.sin, "cos": math.cos, "tanh": math.tanh, "sigmoid": lambda v: 1 / (1 + math.exp(-v)),
        "floor": math.floor, "ceil": math.ceil, "round": round, "sign": lambda v: (v > 0) - (v < 0),
        "step": lambda v: 1.0 if v > 0 else 0.0, "pos": lambda v: max(v, 0.0), "negpart": lambda v: min(v, 0.0),
    }[f](x)
