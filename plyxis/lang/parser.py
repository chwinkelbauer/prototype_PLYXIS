"""Recursive-descent parser for PLX (with Pratt expression parsing).

The parser recovers from errors at statement granularity so the editor can
show all problems of a file at once.
"""
from __future__ import annotations

from typing import Optional

from . import ast as A
from .errors import Diagnostics, Loc
from .lexer import Token, tokenize

KIND_ALIASES = {
    "pkg": "pkg", "package": "pkg",
    "ent": "ent", "entity": "ent",
    "obj": "obj", "object": "obj",
    "env": "env", "environment": "env", "site": "env",
    "sim": "sim", "simulation": "sim", "scenario": "sim",
    "grp": "grp", "group": "grp",
    "res": "res", "resource": "res",
}

SECTION_ALIASES = {
    "properties": "properties", "props": "properties", "parameters": "properties", "params": "properties",
    "state": "state", "states": "state",
    "decision": "decision", "decisions": "decision",
    "nodes": "nodes", "ports": "nodes",
    "behavior": "behavior", "behaviour": "behavior",
    "contains": "contains",
    "connect": "contains", "connections": "contains",
    "geom": "geom", "geometry": "geom",
    "telemetry": "telemetry", "signals": "telemetry",
    "discover": "discover",
    "attr": "attr", "attributes": "attr",
    "constraints": "constraints",
    "objective": "objective",
    "inputs": "inputs",
    "solver": "solver",
}

TERM_KINDS = ("cost", "revenue", "emission", "metric")
COMPARE_OPS = ("<", "<=", ">", ">=", "==", "!=")
BINARY_PREC = {
    "or": 1, "||": 1,
    "and": 2, "&&": 2,
    "<": 4, "<=": 4, ">": 4, ">=": 4, "==": 4, "!=": 4,
    "..": 5,
    "+": 6, "-": 6,
    "*": 7, "/": 7, "%": 7,
    "^": 9, "**": 9,
}
RESERVED = {"define", "as", "init", "connect", "via", "for", "in", "if", "else", "set", "use",
            "and", "or", "not", "true", "false", "TRUE", "FALSE"}


class ParseError(Exception):
    pass


class Parser:
    def __init__(self, text: str, file: str = "<input>", diags: Optional[Diagnostics] = None):
        self.text = text
        self.file = file
        self.diags = diags if diags is not None else Diagnostics()
        self.diags.sources[file] = text
        self.toks, self.docs = tokenize(text, file, self.diags)
        self.i = 0
        self.no_brace_template = 0
        self._last_expr = None

    # ------------------------------------------------------------------ token helpers
    @property
    def tok(self) -> Token:
        return self.toks[self.i]

    def peek(self, k: int = 1) -> Token:
        j = min(self.i + k, len(self.toks) - 1)
        return self.toks[j]

    def next(self) -> Token:
        t = self.toks[self.i]
        if self.i < len(self.toks) - 1:
            self.i += 1
        return t

    def at_op(self, *ops: str) -> bool:
        return self.tok.is_op(*ops)

    def at_ident(self, *names: str) -> bool:
        return self.tok.is_ident(*names)

    def accept_op(self, op: str) -> Optional[Token]:
        if self.tok.is_op(op):
            return self.next()
        return None

    def accept_ident(self, *names: str) -> Optional[Token]:
        if self.tok.is_ident(*names):
            return self.next()
        return None

    def expect_op(self, op: str, what: str = "") -> Token:
        if self.tok.is_op(op):
            return self.next()
        self.error(f"expected '{op}'{(' ' + what) if what else ''} but found {self.describe(self.tok)}")
        raise ParseError()

    def expect_ident(self, what: str = "identifier") -> Token:
        if self.tok.kind == "ident":
            return self.next()
        self.error(f"expected {what} but found {self.describe(self.tok)}")
        raise ParseError()

    @staticmethod
    def describe(t: Token) -> str:
        if t.kind == "eof":
            return "end of file"
        return f"'{t.value}'"

    def error(self, msg: str, loc: Optional[Loc] = None, hint: str = "") -> None:
        self.diags.error(msg, loc or self.tok.loc, "E-PARSE", hint)

    def adjacent(self, a: Token, b: Token) -> bool:
        return a.loc.end_offset == b.loc.offset

    def doc_before(self, line: int) -> str:
        lines = []
        ln = line - 1
        while ln in self.docs:
            lines.insert(0, self.docs[ln])
            ln -= 1
        return " ".join(lines)

    def semi(self, lenient: bool = True) -> None:
        """Statement terminator; tolerate a missing ';' before '}' or a section header."""
        if self.accept_op(";"):
            return
        if lenient and (self.at_op("}") or self.at_section_header() or self.tok.kind == "eof"):
            self.diags.warning("missing ';'", self.toks[self.i - 1].loc, "W-SEMI")
            return
        self.error(f"expected ';' but found {self.describe(self.tok)}")
        raise ParseError()

    def sync(self) -> None:
        """Error recovery: skip to the end of the current statement."""
        depth = 0
        while self.tok.kind != "eof":
            if self.at_op("{"):
                depth += 1
            elif self.at_op("}"):
                if depth == 0:
                    return
                depth -= 1
                if depth == 0:
                    self.next()
                    self.accept_op(";")
                    return
            elif self.at_op(";") and depth == 0:
                self.next()
                return
            elif depth == 0 and self.at_section_header():
                return
            self.next()

    def at_section_header(self) -> bool:
        return self.tok.kind == "ident" and self.tok.value in SECTION_ALIASES and self.peek().is_op(":") \
            and not self.peek(2).is_op(":")

    def raw_until(self, end_ops=(";",)) -> str:
        start = self.tok.loc.offset
        end = start
        depth = 0
        while self.tok.kind != "eof":
            if depth == 0 and self.tok.kind == "op" and self.tok.value in end_ops:
                break
            if self.at_op("(", "[", "{"):
                depth += 1
            elif self.at_op(")", "]", "}"):
                if depth == 0:
                    break
                depth -= 1
            end = self.tok.loc.end_offset
            self.next()
        return self.text[start:end].strip()

    # ------------------------------------------------------------------ program
    def parse_program(self) -> A.Program:
        items = []
        while self.tok.kind != "eof":
            start = self.i
            try:
                it = self.parse_item()
                if it is not None:
                    items.append(it)
            except ParseError:
                self.sync()
                if self.at_op("}"):
                    self.next()
            if self.i == start:      # guarantee progress
                self.next()
        return A.Program(items, self.file, self.docs)

    def parse_item(self):
        t = self.tok
        if t.is_ident("define"):
            return self.parse_define()
        if t.is_ident("use"):
            self.next()
            path = self.parse_dotted()
            self.semi()
            return A.UseDef(name=path, path=path, loc=t.loc)
        if t.is_ident("load") and self.peek().is_op("("):
            self.next()
            self.expect_op("(")
            name = self.expect_ident("package name").value
            self.raw_until((")",))
            self.expect_op(")")
            self.semi()
            return A.UseDef(name=name, path=name, loc=t.loc)
        if t.is_ident("set"):
            return self.parse_set()
        self.error(f"expected a definition ('define ...'), 'use' or 'set' but found {self.describe(t)}")
        raise ParseError()

    def parse_dotted(self) -> str:
        parts = [self.expect_ident("name").value]
        while self.accept_op("."):
            if self.accept_op("*"):
                parts.append("*")
                break
            parts.append(self.expect_ident("name").value)
        return ".".join(parts)

    # ------------------------------------------------------------------ definitions
    def parse_define(self):
        dt = self.next()  # 'define'
        doc = self.doc_before(dt.loc.line)
        name_tok = self.expect_ident("definition name")
        name = name_tok.value
        # function: define f(a, b) = expr;
        if self.at_op("("):
            self.next()
            args = []
            if not self.at_op(")"):
                args.append(self.expect_ident("argument").value)
                while self.accept_op(","):
                    args.append(self.expect_ident("argument").value)
            self.expect_op(")")
            self.expect_op("=")
            body = self.parse_expr()
            self.semi()
            return A.FunctionDef(name, args=args, body=body, loc=dt.loc, doc=doc)
        if not self.accept_ident("as"):
            self.error(f"expected 'as' after 'define {name}'", hint="syntax: define NAME as KIND { ... };")
            raise ParseError()
        if name in KIND_ALIASES:
            self.diags.warning(f"'{name}' is a definition kind keyword: 'define x as {name}' declares a new "
                               f"{KIND_ALIASES[name]} instead of extending this definition", name_tok.loc, "W-NAME",
                               hint=f"rename it, or extend it explicitly with 'define x as env extends {name}'")
        kind_tok = self.expect_ident("kind (ent, pkg, env, sim, grp, res) or a base type")
        kind = KIND_ALIASES.get(kind_tok.value)
        base = None
        if kind is None:
            base = kind_tok.value          # inheritance: define esp_well as well { ... }
            kind = "ent"
        if self.accept_ident("extends"):
            base = self.expect_ident("base type").value
        if kind == "pkg":
            return self.parse_package(name, dt, doc)
        if kind == "grp":
            return self.parse_group(name, dt, doc)
        if kind == "res":
            return self.parse_resource(name, dt, doc)
        if kind == "sim":
            return self.parse_sim(name, dt, doc)
        ent = A.EntityDef(name, kind=kind, base=base, loc=dt.loc, doc=doc)
        if self.at_op("{"):
            self.parse_entity_body(ent)
        self.accept_op(";")
        return ent

    def parse_package(self, name, dt, doc):
        pkg = A.PackageDef(name, loc=dt.loc, doc=doc)
        self.expect_op("{")
        while not self.at_op("}") and self.tok.kind != "eof":
            start = self.i
            try:
                it = self.parse_item()
                if it is not None:
                    pkg.items.append(it)
            except ParseError:
                self.sync()
            if self.i == start:
                self.next()
        self.expect_op("}")
        self.accept_op(";")
        return pkg

    def parse_group(self, name, dt, doc):
        g = A.GroupDef(name, loc=dt.loc, doc=doc)
        self.expect_op("{")
        while not self.at_op("}") and self.tok.kind != "eof":
            if self.accept_op(";") or self.accept_op(","):
                continue
            g.members.append(self.parse_name_template_text())
        self.expect_op("}")
        self.accept_op(";")
        return g

    def parse_resource(self, name, dt, doc):
        r = A.ResourceDef(name, loc=dt.loc, doc=doc)
        if self.at_op("["):
            r.unit = self.parse_unit()
        if self.accept_op("{"):
            section = None
            while not self.at_op("}") and self.tok.kind != "eof":
                try:
                    if self.at_section_header():
                        section = SECTION_ALIASES[self.next().value]
                        self.next()
                        continue
                    key = self.expect_ident("attribute").value
                    self.expect_op("=")
                    val = self.parse_expr()
                    unit = self._decl_unit()
                    self.semi()
                    if section in ("attr",):
                        r.attrs[key] = val
                    elif key == "unit" and isinstance(val, A.Str):
                        r.unit = val.value
                    elif key == "rate_unit" and isinstance(val, A.Str):
                        r.rate_unit = val.value
                    else:
                        r.props[key] = (val, unit)
                except ParseError:
                    self.sync()
            self.expect_op("}")
        self.accept_op(";")
        return r

    def parse_entity_body(self, ent: A.EntityDef) -> None:
        self.expect_op("{")
        section = None
        while not self.at_op("}") and self.tok.kind != "eof":
            start = self.i
            try:
                if self.at_section_header():
                    sec_tok = self.next()
                    section = SECTION_ALIASES[sec_tok.value]
                    self.next()  # ':'
                    continue
                if self.accept_op(";"):
                    continue
                if section is None:
                    self.parse_entity_attr(ent)
                else:
                    self.parse_section_item(ent, section)
            except ParseError:
                self.sync()
            if self.i == start:
                self.next()
        self.expect_op("}")

    def parse_entity_attr(self, ent: A.EntityDef) -> None:
        # flags such as part=TRUE; lat_lon_grid=TRUE; coord = c(1,2)
        if self.at_ident("init", "connect", "for", "if", "define"):
            # contains-statements without a section header (lenient)
            ent.contains.append(self.parse_stmt("contains"))
            return
        key = self.expect_ident("attribute or section").value
        self.expect_op("=")
        val = self.parse_expr()
        unit = self._decl_unit()
        self.semi()
        ent.attrs[key] = (val, unit)

    def parse_section_item(self, ent: A.EntityDef, section: str) -> None:
        t = self.tok
        doc = self.doc_before(t.loc.line)
        if section == "properties":
            # tolerate old-style 'decision numeric(1) x(t);' inside properties (app.R)
            if t.is_ident("decision"):
                ent.behavior.append(A.DecisionStmtLegacy(self.raw_until(), loc=t.loc))
                self.semi()
                return
            name = self.expect_ident("property name").value
            default = None
            unit = None
            if self.accept_op("="):
                default = self.parse_expr()
                unit = self._decl_unit()
            elif self.at_op("["):
                unit = self.parse_unit()
            calib = None
            is_input = False
            while True:
                if self.accept_ident("calibrate"):
                    if self.accept_ident("in"):
                        lo, hi = self.parse_bounds()
                        calib = (lo, hi)
                    else:
                        calib = (None, None)
                    continue
                if self.accept_ident("input"):
                    is_input = True
                    continue
                break
            self.semi()
            ent.params.append(A.ParamDecl(name, default, unit, calib, is_input, doc, loc=t.loc))
        elif section == "state":
            name = self.expect_ident("state name").value
            self.expect_op("=")
            init = self.parse_expr()
            unit = self._decl_unit()
            bounds = None
            if self.accept_ident("in"):
                bounds = self.parse_bounds()
            self.semi()
            ent.states.append(A.StateDecl(name, init, unit, bounds, doc, loc=t.loc))
        elif section == "decision":
            ent.decisions.append(self.parse_decision(doc))
        elif section == "nodes":
            p = self.parse_port(doc)
            if p is not None:
                ent.ports.append(p)
        elif section == "behavior":
            st = self.parse_stmt("behavior")
            if st is not None:
                if isinstance(st, A.ConnectStmt) or isinstance(st, A.InitStmt) or isinstance(st, A.Definition):
                    ent.contains.append(st)      # old style: connect() inside behavior
                else:
                    ent.behavior.append(st)
        elif section == "contains":
            st = self.parse_stmt("contains")
            if st is not None:
                ent.contains.append(st)
        elif section == "geom":
            name = self.expect_ident("geometry item").value
            if self.accept_op("="):
                val = self.parse_expr()
                ent.geom[name] = val
            else:
                ent.geom["kind"] = A.Str(name, loc=t.loc)
            self.semi()
        elif section == "telemetry":
            sig = self.expect_ident("signal name").value
            if not (self.accept_op("<-") or self.accept_op("=")):
                self.error("expected '<-' in telemetry binding", hint='signal <- "TAG.SUFFIX" [unit];')
                raise ParseError()
            tag_tok = self.tok
            if tag_tok.kind != "string":
                self.error("expected tag string in telemetry binding")
                raise ParseError()
            self.next()
            unit = self._decl_unit()
            opts = {}
            if self.accept_op("{"):
                while not self.at_op("}"):
                    k = self.expect_ident("option").value
                    self.expect_op("=")
                    opts[k] = self.parse_expr()
                    self.accept_op(";") or self.accept_op(",")
                self.expect_op("}")
            self.semi()
            ent.telemetry.append(A.TelemetryDecl(sig, tag_tok.value, unit, opts, loc=t.loc))
        elif section == "discover":
            ent.rules.append(self.parse_rule())
        elif section == "constraints":
            st = self.parse_stmt("behavior")
            if isinstance(st, A.ConstraintStmt):
                ent.constraints.append(st)
            elif st is not None:
                self.error("only constraints are allowed in a 'constraints:' section", st.loc)
        elif section == "attr":
            self.parse_entity_attr(ent)
        else:
            self.error(f"section '{section}:' is not valid inside an entity")
            raise ParseError()

    def parse_bounds(self):
        self.expect_op("[")
        lo = self.parse_expr()
        self.expect_op(",")
        hi = self.parse_expr()
        self.expect_op("]")
        return lo, hi

    def parse_decision(self, doc: str) -> A.DecisionDecl:
        t = self.tok
        name = self.expect_ident("decision name").value
        d = A.DecisionDecl(name, doc=doc, loc=t.loc)
        if self.accept_op(":"):
            dom = self.expect_ident("domain (real, int, bool)").value
            if dom in ("bool", "binary"):
                d.domain = "bool"
                d.lo, d.hi = A.Num(0.0), A.Num(1.0)
            elif dom in ("int", "integer"):
                d.domain = "int"
            elif dom in ("real", "float", "numeric"):
                d.domain = "real"
            else:
                self.error(f"unknown decision domain '{dom}'", hint="use real, int or bool")
        if self.accept_ident("in"):
            d.lo, d.hi = self.parse_bounds()
        if self.at_op("["):
            d.unit = self.parse_unit()
        while True:
            if self.accept_ident("default"):
                d.default = self.parse_expr()
                if self.at_op("["):
                    d.default_unit = self.parse_unit()
                continue
            if self.accept_ident("hold"):
                d.hold = self.parse_expr()
                if self.at_op("["):
                    d.hold_unit = self.parse_unit()
                continue
            break
        if d.lo is None and d.domain != "bool":
            self.error(f"decision '{name}' needs bounds", t.loc, hint=f"{name} in [lo, hi];")
        self.semi()
        return d

    def parse_port(self, doc: str) -> Optional[A.PortDecl]:
        t = self.tok
        direction = None
        if self.at_ident("in", "out") and self.peek().kind == "ident":
            direction = self.next().value
        name = self.expect_ident("port name").value
        if direction is None:
            low = name.lower()
            if low.startswith("in"):
                direction = "in"
            elif low.startswith("out"):
                direction = "out"
            else:
                self.error(f"cannot infer direction of port '{name}'", t.loc,
                           hint=f"write 'in {name} <- ...;' or 'out {name} <- ...;'")
                direction = "in"
        if self.at_op(";"):
            self.next()
            return A.PortDecl(direction, name, ["*"], False, doc, loc=t.loc)
        if not (self.accept_op("<-") or self.accept_op(":") or self.accept_op("->")):
            self.error(f"expected '<-' after port '{name}'")
            raise ParseError()
        resources = []
        exclusive = False
        while True:
            if self.accept_op(".") or self.accept_op("*"):
                resources.append("*")
            else:
                r = self.expect_ident("resource name").value
                if self.at_op("("):          # legacy aggregate(type=Sand)
                    self.raw_until((";",))
                    self.diags.warning(f"resource filters like '{r}(...)' are ignored", t.loc, "W-LEGACY")
                resources.append(r)
            if not self.accept_op(","):
                break
        if self.accept_ident("exclusive"):
            exclusive = True
        self.semi()
        return A.PortDecl(direction, name, resources, exclusive, doc, loc=t.loc)

    def parse_rule(self) -> A.RuleDecl:
        t = self.tok
        if not self.accept_ident("rule"):
            self.error("expected 'rule' in discover section", hint='rule NAME: on "PATTERN" { ... };')
            raise ParseError()
        name = self.expect_ident("rule name").value
        self.expect_op(":")
        if not self.accept_ident("on"):
            self.error("expected 'on \"TAG-PATTERN\"'")
            raise ParseError()
        if self.tok.kind != "string":
            self.error("expected tag pattern string")
            raise ParseError()
        pattern = self.next().value
        when = None
        if self.accept_ident("when"):
            self.no_brace_template += 1
            try:
                when = self.parse_expr()
            finally:
                self.no_brace_template -= 1
        body = self.parse_block("contains")
        self.accept_op(";")
        return A.RuleDecl(name, pattern, body, when, loc=t.loc)

    def parse_sim(self, name, dt, doc) -> A.SimDef:
        sim = A.SimDef(name, loc=dt.loc, doc=doc)
        self.expect_op("{")
        section = None
        while not self.at_op("}") and self.tok.kind != "eof":
            start = self.i
            try:
                if self.at_ident("env") and self.peek().is_op(":", "="):
                    self.next(); self.next()
                    sim.env = self.expect_ident("environment name").value
                    self.semi()
                    continue
                if self.at_section_header():
                    section = SECTION_ALIASES[self.next().value]
                    self.next()
                    continue
                if self.accept_op(";"):
                    continue
                if section == "objective":
                    t = self.tok
                    sense = None
                    if self.accept_ident("maximize", "max", "maximise"):
                        sense = "max"
                    elif self.accept_ident("minimize", "min", "minimise"):
                        sense = "min"
                    else:
                        self.error("expected 'maximize' or 'minimize'")
                        raise ParseError()
                    if self.at_op("."):        # legacy: max .all$revenues - .all$costs;
                        raw = self.raw_until()
                        self.diags.warning(f"legacy objective '{raw}' interpreted as "
                                           "sum(revenue) - sum(cost)", t.loc, "W-LEGACY")
                        expr = A.Binary("-", A.Call("sum", [A.Name("revenue")]), A.Call("sum", [A.Name("cost")]))
                    else:
                        expr = self.parse_expr()
                    self.semi()
                    sim.objective = (sense, expr)
                elif section == "behavior":
                    raw = self.raw_until()
                    self.semi()
                    sim.legacy_behavior.append(raw)
                    self.diags.warning("sim 'behavior:' blocks are legacy; model external inflows "
                                       "as entities with input series", self.tok.loc, "W-LEGACY")
                else:
                    t = self.tok
                    key = self.expect_ident("setting").value
                    self.expect_op("=")
                    val = self.parse_expr()
                    unit = self._decl_unit()
                    self.semi()
                    if section == "inputs":
                        sim.inputs.append(A.Define(A.Name(key, loc=t.loc), val, unit, loc=t.loc))
                    elif section == "solver":
                        sim.solver[key] = (val, unit)
                    else:
                        sim.settings[key] = (val, unit)
            except ParseError:
                self.sync()
            if self.i == start:
                self.next()
        self.expect_op("}")
        self.accept_op(";")
        return sim

    # ------------------------------------------------------------------ statements
    def parse_block(self, ctx: str) -> list:
        self.expect_op("{")
        body = []
        while not self.at_op("}") and self.tok.kind != "eof":
            start = self.i
            try:
                st = self.parse_stmt(ctx)
                if st is not None:
                    body.append(st)
            except ParseError:
                self.sync()
            if self.i == start:
                self.next()
        self.expect_op("}")
        return body

    def parse_set(self) -> A.SetStmt:
        t = self.next()
        path = self.parse_postfix(self.parse_atom())
        self.expect_op("=")
        val = self.parse_expr()
        unit = self._decl_unit()
        self.semi()
        return A.SetStmt(path, val, unit, loc=t.loc)

    def parse_stmt(self, ctx: str):
        t = self.tok
        if t.kind == "ident":
            v = t.value
            if v == "final" and self.peek().is_ident(*TERM_KINDS):
                self.next()
                st = self.parse_term()
                st.final = True
                return st
            if v in TERM_KINDS and self.peek().kind == "ident" and self.peek(2).is_op("="):
                return self.parse_term()
            if v == "constraint":
                self.next()
                return self.parse_constraint(t)
            if v == "soft" and self.peek().is_op("("):
                return self.parse_constraint(t)
            if v == "connect":
                return self.parse_connect()
            if v == "init":
                return self.parse_init()
            if v == "for":
                return self.parse_for(ctx)
            if v == "if" and ctx != "behavior":
                return self.parse_if(ctx)
            if v == "if" and not self.peek().is_op("("):
                self.error("conditional statements are not allowed in behavior", t.loc,
                           hint="use the expression form: x = if(cond, a, b);")
                raise ParseError()
            if v == "set":
                return self.parse_set()
            if v == "decision":
                raw = self.raw_until()
                self.semi()
                return A.DecisionStmtLegacy(raw, loc=t.loc)
            if v == "define":
                return self.parse_define()
            if (v == "load" and self.peek().is_op("(")) or (v == "use" and self.peek().kind == "ident"):
                self.parse_item()
                return None
        lhs = self.parse_expr()
        if self.accept_op("="):
            val = self.parse_expr()
            unit = self._decl_unit()
            self.semi()
            return A.Define(lhs, val, unit, loc=t.loc)
        if self.accept_op("<-"):
            val = self.parse_expr()
            self.semi()
            return A.Update(lhs, val, loc=t.loc)
        if isinstance(lhs, A.Compare):
            self.semi()
            return A.ConstraintStmt(None, lhs, None, loc=t.loc)
        if isinstance(lhs, A.Call):
            self.semi()
            return A.ExprStmt(lhs, loc=t.loc)
        self.error("expected '=', '<-' or a comparison (constraint) in statement", t.loc,
                   hint="locals: x = expr;  state updates: s <- expr;  constraints: a <= b;")
        raise ParseError()

    def parse_term(self) -> A.TermStmt:
        t = self.next()
        name = self.expect_ident(f"{t.value} name").value
        self.expect_op("=")
        expr = self.parse_expr()
        unit = self._decl_unit()
        if isinstance(expr, A.Num) and expr.unit and unit == expr.unit:
            unit = None
        self.semi()
        return A.TermStmt(t.value, name, expr, unit=unit, loc=t.loc)

    def parse_constraint(self, t: Token) -> A.ConstraintStmt:
        soft = None
        if self.accept_ident("soft"):
            self.expect_op("(")
            soft = self.parse_expr()
            self.expect_op(")")
        name = None
        if self.tok.kind == "ident" and self.peek().is_op(":"):
            name = self.next().value
            self.next()
        expr = self.parse_expr()
        if not isinstance(expr, A.Compare):
            self.error("a constraint must be a comparison (<=, >=, ==)", expr.loc)
            raise ParseError()
        self.semi()
        return A.ConstraintStmt(name, expr, soft, loc=t.loc)

    def parse_connect(self) -> A.ConnectStmt:
        t = self.next()
        if self.accept_op("("):
            src = self.parse_expr()
            self.expect_op(",")
            dst = self.parse_expr()
            self.expect_op(")")
        else:
            src = self.parse_expr()
            self.expect_op("->")
            dst = self.parse_expr()
        via = None
        overrides = []
        name = None
        if self.accept_ident("via"):
            via = self.parse_name_template(type_position=True)
            if self.at_op("{"):
                overrides = self.parse_overrides()
        if self.accept_ident("as"):
            name = self.parse_name_template()
        self.semi()
        return A.ConnectStmt(src, dst, via, overrides, name, loc=t.loc)

    def parse_init(self) -> A.InitStmt:
        t = self.next()
        name = self.parse_name_template()
        if not self.accept_ident("as"):
            self.error("expected 'as TYPE' in init statement", hint="init NAME as TYPE { overrides };")
            raise ParseError()
        typ = self.parse_name_template(type_position=True)
        overrides = self.parse_overrides() if self.at_op("{") else []
        self.semi()
        return A.InitStmt(name, typ, overrides, loc=t.loc)

    def parse_overrides(self) -> list:
        self.expect_op("{")
        out = []
        while not self.at_op("}") and self.tok.kind != "eof":
            if self.accept_op(";"):
                continue
            t = self.tok
            key = self.expect_ident("property name").value
            self.expect_op("=")
            val = self.parse_expr()
            unit = self._decl_unit()
            if not self.accept_op(";") and not self.at_op("}"):
                self.accept_op(",")
            out.append(A.Override(key, val, unit, loc=t.loc))
        self.expect_op("}")
        return out

    def parse_for(self, ctx: str) -> A.ForStmt:
        t = self.next()
        var = self.expect_ident("loop variable").value
        if not self.accept_ident("in"):
            self.error("expected 'in' in for statement", hint="for i in 1..10 { ... }")
            raise ParseError()
        self.no_brace_template += 1
        try:
            it = self.parse_expr()
        finally:
            self.no_brace_template -= 1
        body = self.parse_block(ctx)
        self.accept_op(";")
        return A.ForStmt(var, it, body, loc=t.loc)

    def parse_if(self, ctx: str) -> A.IfStmt:
        t = self.next()
        self.no_brace_template += 1
        try:
            cond = self.parse_expr()
        finally:
            self.no_brace_template -= 1
        then = self.parse_block(ctx)
        orelse = []
        if self.accept_ident("else"):
            if self.at_ident("if"):
                orelse = [self.parse_if(ctx)]
            else:
                orelse = self.parse_block(ctx)
        self.accept_op(";")
        return A.IfStmt(cond, then, orelse, loc=t.loc)

    # ------------------------------------------------------------------ names & units
    def parse_name_template(self, type_position: bool = False) -> A.Template:
        start_tok = self.tok
        parts: list = []
        prev: Optional[Token] = None
        while True:
            t = self.tok
            if prev is not None and not self.adjacent(prev, t):
                break
            if t.kind == "ident" or (t.kind == "number" and prev is not None):
                parts.append(t.value)
                prev = self.next()
                continue
            if t.is_op("{"):
                if type_position and prev is not None and self._brace_is_block():
                    break
                self.next()
                e = self.parse_expr()
                close = self.expect_op("}")
                parts.append(e)
                prev = close
                continue
            break
        if not parts:
            self.error(f"expected a name but found {self.describe(self.tok)}")
            raise ParseError()
        return A.Template(parts, loc=start_tok.loc)

    def _brace_is_block(self) -> bool:
        # '{ name = ...' or '{ }' -> override block, not interpolation
        a, b = self.peek(1), self.peek(2)
        if a.is_op("}"):
            return True
        return a.kind == "ident" and b.is_op("=")

    def parse_name_template_text(self) -> str:
        tpl = self.parse_name_template()
        return tpl.plain() if tpl.is_plain() else "".join(
            p if isinstance(p, str) else "{?}" for p in tpl.parts)

    def _decl_unit(self) -> Optional[str]:
        """Unit of a declaration: explicit '[u]' after the expression, else the unit of a bare literal."""
        if self.at_op("["):
            return self.parse_unit()
        prev = self._last_expr
        if isinstance(prev, A.Num) and prev.unit:
            return prev.unit
        return None

    def parse_unit(self) -> str:
        open_tok = self.expect_op("[")
        depth = 0
        start = self.tok.loc.offset
        end = start
        while self.tok.kind != "eof":
            if self.at_op("["):
                depth += 1
            elif self.at_op("]"):
                if depth == 0:
                    break
                depth -= 1
            end = self.tok.loc.end_offset
            self.next()
        self.expect_op("]", "to close unit")
        unit = self.text[start:end].strip()
        if not unit:
            self.diags.warning("empty unit annotation", open_tok.loc, "W-UNIT")
        return unit

    # ------------------------------------------------------------------ expressions
    def parse_expr(self, min_prec: int = 0) -> A.Expr:
        e = self._parse_expr(min_prec)
        if min_prec == 0:
            self._last_expr = e
        return e

    def _parse_expr(self, min_prec: int = 0) -> A.Expr:
        left = self.parse_unary()
        while True:
            t = self.tok
            if t.kind == "op":
                op = t.value
            elif t.kind == "ident" and t.value in ("and", "or"):
                op = t.value
            else:
                break
            prec = BINARY_PREC.get(op)
            if prec is None or prec < min_prec:
                break
            if op in COMPARE_OPS:
                ops, operands = [], [left]
                while self.tok.kind == "op" and self.tok.value in COMPARE_OPS:
                    ops.append(self.next().value)
                    operands.append(self._parse_expr(prec + 1))
                left = A.Compare(ops, operands, loc=left.loc)
                continue
            self.next()
            if op in ("^", "**"):
                right = self._parse_expr(prec)            # right associative
                left = A.Binary("^", left, right, loc=left.loc)
            elif op == "..":
                right = self._parse_expr(prec + 1)
                left = A.Range(left, right, loc=left.loc)
            else:
                right = self._parse_expr(prec + 1)
                norm = {"&&": "and", "||": "or"}.get(op, op)
                left = A.Binary(norm, left, right, loc=left.loc)
        return left

    def parse_unary(self) -> A.Expr:
        t = self.tok
        if t.is_op("-") or t.is_op("+"):
            self.next()
            operand = self._parse_expr(8)
            if t.value == "-" and isinstance(operand, A.Num):
                return A.Num(-operand.value, operand.unit, loc=t.loc)
            return A.Unary(t.value, operand, loc=t.loc) if t.value == "-" else operand
        if t.is_op("!") or t.is_ident("not"):
            self.next()
            return A.Unary("not", self._parse_expr(3), loc=t.loc)
        return self.parse_postfix(self.parse_atom())

    def parse_atom(self) -> A.Expr:
        t = self.tok
        if t.kind == "number":
            self.next()
            if self.at_op("["):
                return A.Num(t.num, self.parse_unit(), loc=t.loc)
            return A.Num(t.num, loc=t.loc)
        if t.kind == "string":
            self.next()
            return A.Str(t.value, loc=t.loc)
        if t.kind == "ident":
            if t.value in ("true", "TRUE"):
                self.next()
                return A.Bool(True, loc=t.loc)
            if t.value in ("false", "FALSE"):
                self.next()
                return A.Bool(False, loc=t.loc)
            # function call: name(...)
            if self.peek().is_op("(") and (t.value not in RESERVED or t.value == "if"):
                self.next()
                return self.parse_call_args(t)
            tpl = self.parse_name_template_expr()
            return tpl
        if t.is_op("{") and not self.no_brace_template:
            return self.parse_name_template_expr()
        if t.is_op("("):
            self.next()
            first = self.parse_expr()
            if self.accept_op(","):
                items = [first]
                if not self.at_op(")"):
                    items.append(self.parse_expr())
                    while self.accept_op(","):
                        items.append(self.parse_expr())
                self.expect_op(")")
                return A.TupleExpr(items, loc=t.loc)
            self.expect_op(")")
            return first
        if t.is_op("["):
            self.next()
            items = []
            if not self.at_op("]"):
                items.append(self.parse_expr())
                while self.accept_op(","):
                    if self.at_op("]"):
                        break
                    items.append(self.parse_expr())
            self.expect_op("]")
            return A.ListExpr(items, loc=t.loc)
        if t.is_op("*"):
            self.next()
            return A.Wildcard(loc=t.loc)
        self.error(f"unexpected {self.describe(t)} in expression")
        raise ParseError()

    def parse_name_template_expr(self) -> A.Expr:
        if self.no_brace_template:
            t = self.expect_ident("name")
            return A.Name(t.value, loc=t.loc)
        tpl = self.parse_name_template()
        if tpl.is_plain():
            return A.Name(tpl.plain(), loc=tpl.loc)
        return tpl

    def parse_call_args(self, name_tok: Token) -> A.Call:
        self.expect_op("(")
        args, kwargs = [], {}
        if not self.at_op(")"):
            while True:
                if self.tok.kind == "ident" and self.peek().is_op("=") and not self.peek(2).is_op("="):
                    k = self.next().value
                    self.next()
                    kwargs[k] = self.parse_expr()
                else:
                    args.append(self.parse_expr())
                if not self.accept_op(","):
                    break
        self.expect_op(")")
        return A.Call(name_tok.value, args, kwargs, loc=name_tok.loc)

    def parse_postfix(self, e: A.Expr) -> A.Expr:
        while True:
            if self.at_op(".", "$"):
                dot = self.tok
                nxt = self.peek()
                if nxt.kind == "ident" or (nxt.is_op("{") and not self.no_brace_template):
                    self.next()                       # consume '.' / '$'
                    templ = (not self.no_brace_template) and (
                        self.tok.is_op("{") or (self.peek().is_op("{") and self.adjacent(self.tok, self.peek())))
                    if templ:
                        tpl = self.parse_name_template()
                        e = A.Member(e, tpl.plain() if tpl.is_plain() else tpl, loc=e.loc)
                    else:
                        e = A.Member(e, self.next().value, loc=e.loc)
                    continue
                if nxt.is_op("*"):
                    self.next()
                    self.next()
                    e = A.Member(e, "*", loc=e.loc)
                    continue
                if nxt.is_op("."):            # legacy  x$.$amount
                    self.error("legacy wildcard member access ('$.$') is not supported", dot.loc,
                               hint="reference port channels explicitly, e.g. feed.oil")
                    raise ParseError()
                break
            break
        return e


def parse(text: str, file: str = "<input>", diags: Optional[Diagnostics] = None) -> A.Program:
    p = Parser(text, file, diags)
    return p.parse_program()
