"""Type registry: packages, entity types, resources, groups, functions, sims.

Definitions inside packages are registered under qualified names
(``oilgas.esp_well``); ``use oilgas;`` makes the short names visible.
Inheritance (``define esp_well as well { ... }``) is resolved here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from . import ast as A
from .errors import Diagnostics, Loc, NOLOC, suggest
from .units import UnitError, parse_unit


@dataclass
class ResourceType:
    name: str
    unit: Optional[str] = None
    rate_unit: Optional[str] = None
    attrs: dict = field(default_factory=dict)      # attr -> default value (canonical float)
    doc: str = ""
    loc: Loc = NOLOC

    @property
    def display_rate_unit(self) -> Optional[str]:
        if self.rate_unit:
            return self.rate_unit
        if self.unit:
            return f"{self.unit}/d"
        return None


@dataclass
class EntityType:
    name: str                       # qualified
    short: str
    kind: str                       # ent | obj | env
    base: Optional[str] = None
    params: dict = field(default_factory=dict)      # name -> ParamDecl
    states: dict = field(default_factory=dict)      # name -> StateDecl
    decisions: dict = field(default_factory=dict)   # name -> DecisionDecl
    ports: dict = field(default_factory=dict)       # name -> PortDecl
    behavior: list = field(default_factory=list)
    contains: list = field(default_factory=list)
    geom: dict = field(default_factory=dict)
    telemetry: list = field(default_factory=list)
    rules: list = field(default_factory=list)
    constraints: list = field(default_factory=list)
    attrs: dict = field(default_factory=dict)
    doc: str = ""
    loc: Loc = NOLOC
    package: str = ""
    base_dir: str = "."
    ancestors: list = field(default_factory=list)    # qualified names, nearest first

    @property
    def is_composite(self) -> bool:
        return any(isinstance(s, (A.InitStmt, A.ForStmt, A.IfStmt)) or
                   (isinstance(s, A.EntityDef) and s.kind == "obj") for s in self.contains)

    def is_a(self, name: str) -> bool:
        return name == self.name or name == self.short or name in self.ancestors or \
            any(a.split(".")[-1] == name for a in self.ancestors)


JUNCTION = "__junction__"


def make_junction_type() -> EntityType:
    t = EntityType(name=JUNCTION, short=JUNCTION, kind="ent", doc="implicit junction (composite port)")
    t.ports["in"] = A.PortDecl("in", "in", ["*"])
    t.ports["out"] = A.PortDecl("out", "out", ["*"])
    t.behavior.append(A.Define(A.Name("out"), A.Name("in")))
    t.geom["kind"] = A.Str("junction")
    return t


class Library:
    def __init__(self, diags: Optional[Diagnostics] = None):
        self.diags = diags if diags is not None else Diagnostics()
        self.defs: dict[str, A.Definition] = {}          # qualified -> raw definition
        self.def_pkg: dict[str, str] = {}
        self.def_dir: dict[str, str] = {}
        self.aliases: dict[str, str] = {}                 # short -> qualified (via use)
        self.packages: dict[str, list[str]] = {}
        self.resources: dict[str, ResourceType] = {}
        self.groups: dict[str, A.GroupDef] = {}
        self.functions: dict[str, A.FunctionDef] = {}
        self.sims: dict[str, A.SimDef] = {}
        self.sets: list[A.SetStmt] = []
        self._types: dict[str, EntityType] = {JUNCTION: make_junction_type()}
        self.programs: list[A.Program] = []
        self.pending_uses: list = []

    # ------------------------------------------------------------------ loading
    def add_program(self, prog: A.Program, base_dir: str = ".") -> None:
        self.programs.append(prog)
        for item in prog.items:
            self._add_item(item, "", base_dir)
        for path, loc in self.pending_uses:
            self.use(path, loc)
        self.pending_uses.clear()
        # a file's own top-level 'use' statements
        for item in prog.items:
            if isinstance(item, A.UseDef):
                self.use(item.path, item.loc)

    def _add_item(self, item, pkg: str, base_dir: str) -> None:
        if isinstance(item, A.PackageDef):
            q = f"{pkg}.{item.name}" if pkg else item.name
            self.packages.setdefault(q, [])
            for sub in item.items:
                self._add_item(sub, q, base_dir)
            return
        if isinstance(item, A.UseDef):
            if pkg:
                self.pending_uses.append((item.path, item.loc))
            return
        if isinstance(item, A.SetStmt):
            self.sets.append(item)
            return
        if not isinstance(item, A.Definition):
            return
        q = f"{pkg}.{item.name}" if pkg else item.name
        if q in self.defs:
            self.diags.error(f"'{q}' is defined twice", item.loc, "E-DUP",
                             hint=f"first definition at {self.defs[q].loc}")
            return
        self.defs[q] = item
        self.def_pkg[q] = pkg
        self.def_dir[q] = base_dir
        if pkg:
            self.packages[pkg].append(q)
        if isinstance(item, A.ResourceDef):
            self.resources[q] = self._make_resource(q, item)
        elif isinstance(item, A.EntityDef) and item.kind in ("ent", "obj") and \
                self._attr_true(item, "part"):
            # legacy resource: define k as ent { part=TRUE; properties: quality = 1; }
            self.resources[q] = self._legacy_resource(q, item)
        elif isinstance(item, A.GroupDef):
            self.groups[q] = item
        elif isinstance(item, A.FunctionDef):
            self.functions[q] = item
        elif isinstance(item, A.SimDef):
            self.sims[q] = item

    @staticmethod
    def _attr_true(item: A.EntityDef, key: str) -> bool:
        v = item.attrs.get(key)
        return v is not None and isinstance(v[0], A.Bool) and v[0].value

    def _make_resource(self, q: str, item: A.ResourceDef) -> ResourceType:
        from .consteval import const_eval
        r = ResourceType(q, item.unit, item.rate_unit, doc=item.doc, loc=item.loc)
        for k, e in item.attrs.items():
            try:
                r.attrs[k] = float(const_eval(e, {}))
            except Exception as ex:  # noqa: BLE001
                self.diags.error(f"attribute '{k}' of resource '{q}' must be a constant ({ex})", item.loc)
        for u in (r.unit, r.rate_unit):
            if u:
                try:
                    parse_unit(u)
                except UnitError as ex:
                    self.diags.warning(f"resource '{q}': {ex}", item.loc, "W-UNIT")
        return r

    def _legacy_resource(self, q: str, item: A.EntityDef) -> ResourceType:
        from .consteval import const_eval
        r = ResourceType(q, None, None, doc=item.doc or "legacy divisible good (part=TRUE)", loc=item.loc)
        for p in item.params:
            try:
                r.attrs[p.name] = float(const_eval(p.default, {})) if p.default is not None else 0.0
            except Exception:  # noqa: BLE001
                pass
        if item.behavior:
            self.diags.warning(f"behavior of good '{q}' is ignored: attribute dynamics belong to the "
                               "entities that store or transform the good", item.loc, "W-LEGACY")
        return r

    def use(self, path: str, loc: Loc = NOLOC) -> None:
        if path.endswith(".*"):
            path = path[:-2]
        if path in self.packages:
            for q in self.packages[path]:
                short = q.split(".")[-1]
                prev = self.aliases.get(short)
                if prev and prev != q:
                    self.diags.warning(f"'{short}' from '{path}' shadows '{prev}'", loc, "W-SHADOW")
                self.aliases[short] = q
            # nested packages become visible by their last segment
            for sub in self.packages:
                if sub.startswith(path + "."):
                    self.aliases.setdefault(sub.split(".")[-1], sub)
        elif path in self.defs:
            self.aliases[path.split(".")[-1]] = path
        else:
            self.diags.error(f"unknown package '{path}'", loc, "E-USE",
                             hint=suggest(path, list(self.packages)))

    # ------------------------------------------------------------------ resolution
    def resolve(self, name: str, scope_pkg: str = "") -> Optional[str]:
        if name in self.defs or name in self._types:
            return name
        pkg = scope_pkg
        while pkg:
            q = f"{pkg}.{name}"
            if q in self.defs:
                return q
            pkg = pkg.rpartition(".")[0]
        if name in self.aliases:
            return self.aliases[name]
        return None

    def resource(self, name: str, scope_pkg: str = "") -> Optional[ResourceType]:
        q = self.resolve(name, scope_pkg)
        return self.resources.get(q) if q else None

    def expand_resources(self, names: list, scope_pkg: str, loc: Loc) -> list:
        """Expand a port's resource list (groups -> members). '*' stays a wildcard."""
        out: list = []
        for n in names:
            if n == "*":
                return ["*"]
            q = self.resolve(n, scope_pkg)
            if q in self.resources:
                out.append(q)
            elif q in self.groups:
                g = self.groups[q]
                out.extend(self.expand_resources(g.members, self.def_pkg.get(q, ""), loc))
            else:
                self.diags.error(f"unknown resource '{n}'", loc, "E-RES",
                                 hint=suggest(n, [r.split('.')[-1] for r in self.resources]))
        seen = []
        for r in out:
            if r not in seen:
                seen.append(r)
        return seen

    def function(self, name: str, scope_pkg: str = "") -> Optional[A.FunctionDef]:
        q = self.resolve(name, scope_pkg)
        return self.functions.get(q) if q else None

    def entity_names(self) -> list[str]:
        return [q for q, d in self.defs.items() if isinstance(d, A.EntityDef) and q not in self.resources]

    def get_type(self, name: str, scope_pkg: str = "", loc: Loc = NOLOC, _stack=None) -> Optional[EntityType]:
        q = self.resolve(name, scope_pkg)
        if q is None:
            self.diags.error(f"unknown entity type '{name}'", loc, "E-TYPE",
                             hint=suggest(name, [n.split('.')[-1] for n in self.entity_names()]))
            return None
        if q in self._types:
            return self._types[q]
        d = self.defs.get(q)
        if not isinstance(d, A.EntityDef):
            self.diags.error(f"'{name}' is not an entity type", loc, "E-TYPE")
            return None
        if q in self.resources:
            self.diags.error(f"'{name}' is a resource (good), not an entity", loc, "E-TYPE")
            return None
        stack = _stack or []
        if q in stack:
            self.diags.error(f"circular inheritance: {' -> '.join(stack + [q])}", d.loc, "E-TYPE")
            return None
        t = self._build_type(q, d, stack + [q])
        self._types[q] = t
        return t

    def register_inline(self, d: A.EntityDef, scope_pkg: str, base_dir: str, prefix: str) -> EntityType:
        q = f"{prefix}{d.name}"
        self.defs[q] = d
        self.def_pkg[q] = scope_pkg
        self.def_dir[q] = base_dir
        t = self._build_type(q, d, [q])
        self._types[q] = t
        return t

    def _build_type(self, q: str, d: A.EntityDef, stack: list) -> EntityType:
        pkg = self.def_pkg.get(q, "")
        t = EntityType(name=q, short=d.name, kind=d.kind, base=None, doc=d.doc, loc=d.loc,
                       package=pkg, base_dir=self.def_dir.get(q, "."))
        if d.base:
            parent = self.get_type(d.base, pkg, d.loc, stack)
            if parent is not None:
                t.base = parent.name
                t.ancestors = [parent.name] + parent.ancestors
                if parent.kind == "env":
                    t.kind = "env"
                    t.params = dict(parent.params)
                t.params = dict(parent.params)
                t.states = dict(parent.states)
                t.decisions = dict(parent.decisions)
                t.ports = dict(parent.ports)
                t.behavior = list(parent.behavior)
                t.contains = list(parent.contains)
                t.geom = dict(parent.geom)
                t.telemetry = list(parent.telemetry)
                t.attrs = dict(parent.attrs)
                t.constraints = list(parent.constraints)
                t.rules = list(parent.rules)
                if not t.doc:
                    t.doc = parent.doc
        for p in d.params:
            if p.name in t.params and t.base:
                old = t.params[p.name]
                if p.calibrate is None and old.calibrate is not None:
                    p = A.ParamDecl(p.name, p.default, p.unit or old.unit, old.calibrate, p.is_input or old.is_input,
                                    p.doc or old.doc, loc=p.loc)
                elif p.unit is None and old.unit is not None:
                    p = A.ParamDecl(p.name, p.default, old.unit, p.calibrate, p.is_input, p.doc or old.doc, loc=p.loc)
            elif p.name in t.params:
                self.diags.error(f"property '{p.name}' declared twice in '{d.name}'", p.loc, "E-DUP")
            t.params[p.name] = p
        for s in d.states:
            t.states[s.name] = s
        for dec in d.decisions:
            t.decisions[dec.name] = dec
        for port in d.ports:
            t.ports[port.name] = port
        t.behavior.extend(d.behavior)
        t.contains.extend(d.contains)
        t.geom.update(d.geom)
        tele = {x.signal: x for x in t.telemetry}
        for x in d.telemetry:
            tele[x.signal] = x
        t.telemetry = list(tele.values())
        t.rules = list(t.rules) + list(d.rules)
        t.constraints = list(t.constraints) + list(d.constraints)
        t.attrs.update(d.attrs)
        # name clash check among symbol kinds
        seen: dict[str, str] = {}
        for kind, names in (("property", t.params), ("state", t.states), ("decision", t.decisions),
                            ("port", t.ports)):
            for n in names:
                if n in seen:
                    self.diags.error(f"'{n}' is declared as both {seen[n]} and {kind} in '{d.name}'", d.loc, "E-DUP")
                seen[n] = kind
        return t


def load_library(files: list[str], diags: Optional[Diagnostics] = None, sources: Optional[dict] = None,
                 auto_use: tuple = ()) -> Library:
    """Parse files (paths or name->text via `sources`) into a Library."""
    from .parser import parse
    diags = diags if diags is not None else Diagnostics()
    lib = Library(diags)
    items = []
    for f in files:
        if sources and f in sources:
            text = sources[f]
            d = os.path.dirname(os.path.abspath(f))
            base_dir = d if os.path.isdir(d) else "."
        else:
            with open(f, encoding="utf-8") as fh:
                text = fh.read()
            base_dir = os.path.dirname(os.path.abspath(f))
        prog = parse(text, os.path.basename(f), diags)
        items.append((prog, base_dir))
    # register everything first, then process 'use' (packages may be in later files)
    for prog, base_dir in items:
        lib.programs.append(prog)
        for item in prog.items:
            lib._add_item(item, "", base_dir)
    for pkg in auto_use:
        if pkg in lib.packages:
            lib.use(pkg)
    for path, loc in lib.pending_uses:
        lib.use(path, loc)
    lib.pending_uses.clear()
    for prog, _ in items:
        for item in prog.items:
            if isinstance(item, A.UseDef):
                lib.use(item.path, item.loc)
    return lib
