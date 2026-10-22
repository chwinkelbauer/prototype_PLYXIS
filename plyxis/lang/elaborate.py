"""Elaboration: turn an environment definition into a flat instance netlist.

Like VHDL elaboration, this expands generate statements (``for``/``if``),
instantiates entities (also from data tables / GeoJSON), flattens composite
entities (their ports become implicit junctions), resolves connections
(including ``via`` link entities), propagates wildcard port resources and
evaluates all parameters into canonical units.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from . import ast as A
from .consteval import (Canon, EvalError, MISSING, Scope, const_eval, haversine_km, is_canon,
                        template_str)
from .errors import Diagnostics, Loc, NOLOC, suggest
from .library import JUNCTION, EntityType, Library
from .units import UnitError, parse_unit

RESERVED_PROPS = {"at", "coord", "tag", "label", "path", "polygon", "layer", "icon", "color", "name", "area"}


@dataclass
class Instance:
    id: int
    path: str
    type: EntityType
    params: dict = field(default_factory=dict)
    param_units: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    state0: dict = field(default_factory=dict)
    at: Optional[tuple] = None
    line: Optional[list] = None
    polygon: Optional[list] = None
    tag: str = ""
    label: str = ""
    parent: Optional[str] = None
    is_link: bool = False
    is_junction: bool = False
    loc: Loc = NOLOC
    spec_key: str = ""
    ports: dict = field(default_factory=dict)        # port name -> port id
    overrides: dict = field(default_factory=dict)    # name -> (value, unit, loc)
    scope: Any = None
    base_dir: str = "."

    @property
    def name(self) -> str:
        return self.path.rsplit(".", 1)[-1]


@dataclass
class Port:
    id: int
    inst: int
    name: str
    direction: str
    declared: list
    exclusive: bool = False
    resources: list = field(default_factory=list)
    edges: list = field(default_factory=list)


@dataclass
class Edge:
    id: int
    src: int
    dst: int
    resources: list = field(default_factory=list)
    loc: Loc = NOLOC
    label: str = ""


@dataclass
class SimSpec:
    name: str = "default"
    horizon: float = 24.0        # hours
    dt: float = 1.0              # hours
    objective: Optional[tuple] = None
    solver: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)   # name -> A.Expr
    input_units: dict = field(default_factory=dict)
    loc: Loc = NOLOC


@dataclass
class FlatModel:
    lib: Library
    env_type: EntityType
    instances: list = field(default_factory=list)
    ports: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    by_path: dict = field(default_factory=dict)
    env_params: dict = field(default_factory=dict)
    env_units: dict = field(default_factory=dict)
    env_inputs: set = field(default_factory=set)
    env_meta: dict = field(default_factory=dict)
    inst_groups: dict = field(default_factory=dict)
    rules: list = field(default_factory=list)
    sim: SimSpec = field(default_factory=SimSpec)
    origin: Optional[tuple] = None

    def instance(self, path: str) -> Optional[Instance]:
        i = self.by_path.get(path)
        return self.instances[i] if i is not None else None

    def port(self, inst: Instance, name: str) -> Optional[Port]:
        pid = inst.ports.get(name)
        return self.ports[pid] if pid is not None else None


class _InstRef:
    """Lets elaboration expressions write `ctb1.oil_tank` or `distance(a, b)`."""

    def __init__(self, el: "Elaborator", inst: Instance):
        self.el, self.inst = el, inst

    def lookup_attr(self, attr: str, loc=None):
        if attr in ("lat", "lon") and self.inst.at:
            return self.inst.at[0] if attr == "lat" else self.inst.at[1]
        if attr == "at":
            return self.inst.at
        if attr in self.inst.type.params:
            return self.el.param_value(self.inst, attr)
        if attr in self.inst.meta:
            return self.inst.meta[attr]
        raise EvalError(f"'{self.inst.path}' has no property '{attr}'", loc)


class _EnvRef:
    def __init__(self, el: "Elaborator"):
        self.el = el

    def lookup_attr(self, attr: str, loc=None):
        if attr in self.el.flat.env_params:
            return Canon(self.el.flat.env_params[attr])
        if attr in self.el.flat.env_meta:
            return self.el.flat.env_meta[attr]
        raise EvalError(f"environment has no property '{attr}'", loc)


class Elaborator:
    def __init__(self, lib: Library, diags: Optional[Diagnostics] = None):
        self.lib = lib
        self.diags = diags if diags is not None else lib.diags
        self._connects: list = []
        self._sets: list = []
        self._evaluating: set = set()
        self.flat: FlatModel = None  # type: ignore

    # ------------------------------------------------------------------ entry
    def elaborate(self, env_name: str, sim_name: Optional[str] = None) -> FlatModel:
        env_type = self.lib.get_type(env_name)
        if env_type is None:
            self.diags.raise_if_errors()
        if env_type.kind != "env":
            self.diags.error(f"'{env_name}' is not an environment (define it 'as env')", env_type.loc, "E-ENV")
        self.flat = FlatModel(self.lib, env_type)
        self._eval_env_params(env_type)
        self.flat.rules = list(env_type.rules)
        if env_type.ports:
            self.diags.warning("ports on an environment are ignored", env_type.loc, "W-ENV")
        origin = env_type.attrs.get("origin")
        if origin is not None:
            try:
                o = const_eval(origin[0], {})
                self.flat.origin = (float(o[0]), float(o[1]))
            except Exception:  # noqa: BLE001
                pass
        root_scope = Scope({"env": _EnvRef(self)}, resolver=self._global_resolver)
        ctx = _Ctx(prefix="", scope=root_scope, aliases={}, parent=None, base_dir=env_type.base_dir,
                   pkg=env_type.package, comp_inst=None)
        self._process(env_type.contains, ctx)
        for c in self._connects:
            self._connect(*c)
        for stmt, cctx in self._sets:
            self._apply_set(stmt, cctx)
        for stmt in self.lib.sets:
            self._apply_set(stmt, ctx, top_level=True)
        for inst in self.flat.instances:
            self._eval_params(inst)
        for inst in self.flat.instances:
            self._eval_states(inst)
        self._geometry()
        self._propagate_resources()
        self._spec_keys()
        self.flat.sim = self._sim(sim_name, env_type)
        return self.flat

    def _global_resolver(self, name: str):
        """Bare names in elaboration expressions may refer to already created instances."""
        idx = self.flat.by_path.get(name)
        if idx is not None:
            return _InstRef(self, self.flat.instances[idx])
        return MISSING

    # ------------------------------------------------------------------ env
    def _eval_env_params(self, env_type: EntityType) -> None:
        vals: dict = {}
        units: dict = {}

        def get(name, stack=()):
            if name in vals:
                return Canon(vals[name])
            p = env_type.params.get(name)
            if p is None:
                return MISSING
            if name in stack:
                raise EvalError(f"circular definition of environment property '{name}'", p.loc)
            if p.default is None:
                raise EvalError(f"environment property '{name}' has no value", p.loc)
            sc = Scope(resolver=lambda n: get(n, stack + (name,)))
            v = const_eval(p.default, sc, base_dir=env_type.base_dir)
            if isinstance(v, str):
                self.flat.env_meta[name] = v
                return v
            vals[name] = _to_canon(v, p.unit, self.diags, p.loc)
            return Canon(vals[name])

        for name, p in env_type.params.items():
            try:
                get(name)
            except EvalError as ex:
                self.diags.error(str(ex), ex.loc or p.loc, "E-ELAB")
            units[name] = p.unit
            if p.is_input:
                self.flat.env_inputs.add(name)
        for key, (expr, unit) in env_type.attrs.items():
            if key in ("origin", "lat_lon_grid", "crs"):
                continue
            try:
                v = const_eval(expr, {})
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    vals.setdefault(key, _to_canon(v, unit, self.diags, env_type.loc))
                    units.setdefault(key, unit)
                else:
                    self.flat.env_meta[key] = v
            except Exception:  # noqa: BLE001
                pass
        self.flat.env_params = vals
        self.flat.env_units = units

    # ------------------------------------------------------------------ statements
    def _process(self, stmts: list, ctx: "_Ctx") -> None:
        for st in stmts:
            try:
                self._stmt(st, ctx)
            except EvalError as ex:
                self.diags.error(str(ex), ex.loc or getattr(st, "loc", NOLOC), "E-ELAB")

    def _stmt(self, st, ctx: "_Ctx") -> None:
        if isinstance(st, A.InitStmt):
            name = template_str(st.name, ctx.scope, base_dir=ctx.base_dir)
            tname = template_str(st.type, ctx.scope, base_dir=ctx.base_dir)
            if tname in self.lib.groups or (self.lib.resolve(tname, ctx.pkg) in self.lib.groups):
                self.diags.info(f"'init {name} as {tname}' ignored (groups need no instantiation)", st.loc)
                return
            typ = self.lib.get_type(tname, ctx.pkg, st.type.loc)
            if typ is None:
                return
            self._instantiate(name, typ, st.overrides, ctx, st.loc)
        elif isinstance(st, A.ConnectStmt):
            self._connects.append((st, ctx))
        elif isinstance(st, A.ForStmt):
            seq = const_eval(st.iterable, ctx.scope, base_dir=ctx.base_dir)
            if isinstance(seq, (int, float)):
                seq = list(range(1, int(seq) + 1))
            if not isinstance(seq, (list, tuple)):
                raise EvalError("for-loops iterate over ranges (1..n), lists or table(...) rows", st.loc)
            for v in seq:
                self._process(st.body, ctx.child({st.var: v}))
        elif isinstance(st, A.IfStmt):
            if const_eval(st.cond, ctx.scope, base_dir=ctx.base_dir):
                self._process(st.then, ctx)
            else:
                self._process(st.orelse, ctx)
        elif isinstance(st, A.EntityDef):
            # inline object: define delivery as obj { coord = c(19,19) ... }
            typ = self.lib.register_inline(st, ctx.pkg, ctx.base_dir, prefix=f"{self.flat.env_type.name}::")
            overrides = []
            for key, (expr, unit) in st.attrs.items():
                overrides.append(A.Override(key, expr, unit, loc=st.loc))
            self._instantiate(st.name, typ, overrides, ctx, st.loc)
        elif isinstance(st, A.GroupDef):
            members = []
            for m in st.members:
                path = ctx.prefix + m
                if path in self.flat.by_path:
                    members.append(self.flat.by_path[path])
                elif path in self.flat.inst_groups:
                    members.extend(self.flat.inst_groups[path])
                elif self.lib.resolve(m) in self.lib.resources:
                    pass  # resource group inside env (legacy) - nothing to do
                else:
                    self.diags.error(f"group member '{m}' is not a known instance", st.loc, "E-GRP")
            if members:
                self.flat.inst_groups[ctx.prefix + st.name] = members
        elif isinstance(st, A.SetStmt):
            self._sets.append((st, ctx))
        elif isinstance(st, A.Define) and isinstance(st.target, A.Name):
            ctx.scope.values[st.target.id] = const_eval(st.value, ctx.scope, base_dir=ctx.base_dir)
        elif isinstance(st, (A.UseDef,)):
            pass
        elif isinstance(st, A.DecisionStmtLegacy):
            self.diags.warning("legacy 'decision' statement ignored here", st.loc, "W-LEGACY")
        else:
            self.diags.error(f"statement not allowed in 'contains': {type(st).__name__}", st.loc, "E-ELAB")

    def _instantiate(self, name: str, typ: EntityType, overrides: list, ctx: "_Ctx", loc: Loc) -> None:
        path = ctx.prefix + name
        if path in self.flat.by_path:
            self.diags.error(f"instance '{path}' already exists", loc, "E-DUP")
            return
        ov = {}
        for o in overrides:
            key = "at" if o.name == "coord" else o.name
            try:
                ov[key] = (const_eval(o.value, ctx.scope, base_dir=ctx.base_dir), o.unit, o.loc)
            except EvalError as ex:
                self.diags.error(str(ex), ex.loc or o.loc, "E-ELAB")
        if typ.is_composite:
            self._expand_composite(path, typ, ov, ctx, loc)
            return
        inst = Instance(id=len(self.flat.instances), path=path, type=typ, loc=loc, parent=ctx.comp_inst,
                        overrides=ov, scope=ctx.scope, base_dir=ctx.base_dir)
        self._apply_reserved(inst, ov, ctx)
        self.flat.instances.append(inst)
        self.flat.by_path[path] = inst.id
        for pname, pdecl in typ.ports.items():
            port = Port(id=len(self.flat.ports), inst=inst.id, name=pname, direction=pdecl.direction,
                        declared=self.lib.expand_resources(pdecl.resources, typ.package, pdecl.loc),
                        exclusive=pdecl.exclusive)
            self.flat.ports.append(port)
            inst.ports[pname] = port.id
        for key, (val, unit, oloc) in ov.items():
            if key in RESERVED_PROPS:
                continue
            if key not in typ.params and key not in typ.states:
                if isinstance(val, str):
                    inst.meta[key] = val
                else:
                    self.diags.warning(f"'{typ.short}' has no property '{key}' (stored as metadata)", oloc,
                                       "W-PROP", hint=suggest(key, list(typ.params)))
                    inst.meta[key] = val

    def _apply_reserved(self, inst: Instance, ov: dict, ctx: "_Ctx") -> None:
        for key, (val, unit, oloc) in ov.items():
            if key == "at":
                if isinstance(val, _InstRef):
                    inst.at = val.inst.at
                elif isinstance(val, (tuple, list)) and len(val) >= 2:
                    inst.at = (float(val[0]), float(val[1]))
                else:
                    self.diags.error("'at' must be a (lat, lon) pair", oloc, "E-GEO")
            elif key == "tag":
                inst.tag = str(val)
            elif key in ("label", "name"):
                inst.label = str(val)
            elif key == "path":
                inst.line = [(float(p[0]), float(p[1])) for p in val]
            elif key in ("polygon", "area"):
                inst.polygon = [(float(p[0]), float(p[1])) for p in val]
            elif key in ("layer", "icon", "color"):
                inst.meta[key] = str(val)
        if inst.at is None and ctx.comp_at is not None:
            inst.at = ctx.comp_at

    def _expand_composite(self, path: str, typ: EntityType, ov: dict, ctx: "_Ctx", loc: Loc) -> None:
        bad = [s for s in typ.behavior if not isinstance(s, A.DecisionStmtLegacy)]
        if bad or typ.states or typ.decisions:
            self.diags.error(f"composite entity '{typ.short}' may only contain properties, ports and "
                             "sub-instances; move behavior into a child entity", typ.loc, "E-COMP")
        # evaluate composite parameters (available to children as plain names)
        vals: dict = {}
        for key, (val, unit, oloc) in ov.items():
            if key in typ.params:
                p = typ.params[key]
                vals[key] = Canon(_to_canon(val, unit or p.unit, self.diags, oloc))
        sc = ctx.scope.child()

        def resolve(n):
            if n in vals:
                return vals[n]
            p = typ.params.get(n)
            if p is None or p.default is None:
                return MISSING
            v = const_eval(p.default, sc, base_dir=typ.base_dir)
            vals[n] = Canon(_to_canon(v, p.unit, self.diags, p.loc)) if not isinstance(v, str) else v
            return vals[n]

        sc.resolver = resolve
        comp_at = None
        if "at" in ov:
            v = ov["at"][0]
            if isinstance(v, (tuple, list)) and len(v) >= 2:
                comp_at = (float(v[0]), float(v[1]))
        aliases = {}
        for pname, pdecl in typ.ports.items():
            jpath = f"{path}.#{pname}"
            jt = self.lib.get_type(JUNCTION)
            j = Instance(id=len(self.flat.instances), path=jpath, type=jt, loc=loc, parent=path,
                         is_junction=True, at=comp_at or ctx.comp_at)
            self.flat.instances.append(j)
            self.flat.by_path[jpath] = j.id
            for jp in ("in", "out"):
                port = Port(id=len(self.flat.ports), inst=j.id, name=jp, direction=jp,
                            declared=self.lib.expand_resources(pdecl.resources, typ.package, pdecl.loc))
                self.flat.ports.append(port)
                j.ports[jp] = port.id
            aliases[pname] = jpath
        self.flat.inst_groups.setdefault(path, [])
        child = _Ctx(prefix=path + ".", scope=sc.child({"self": vals}), aliases=aliases, parent=ctx,
                     base_dir=typ.base_dir, pkg=typ.package, comp_inst=path, comp_at=comp_at or ctx.comp_at)
        before = len(self.flat.instances)
        self._process(typ.contains, child)
        self.flat.inst_groups[path] = list(range(before, len(self.flat.instances)))
        self.flat.env_meta.setdefault("__composites__", {})[path] = typ.name

    # ------------------------------------------------------------------ connections
    def _endpoint(self, e: A.Expr, ctx: "_Ctx", role: str, loc: Loc) -> list:
        segs = self._segments(e, ctx)
        if segs is None:
            self.diags.error("invalid connection endpoint", loc, "E-CONN",
                             hint="use instance.port, e.g. connect(w1.prod, pad1.feed)")
            return []
        # composite's own port (inside the composite)
        if len(segs) == 1 and segs[0] in ctx.aliases:
            j = self.flat.instance(ctx.aliases[segs[0]])
            return [j.ports["out" if role == "src" else "in"]]
        # group expansion: grp.port  or grp.*.port
        head = ctx.prefix + segs[0]
        if head in self.flat.inst_groups and (len(segs) == 2 or (len(segs) == 3 and segs[1] == "*")):
            pname = segs[-1]
            out = []
            for iid in self.flat.inst_groups[head]:
                inst = self.flat.instances[iid]
                if inst.is_junction:
                    continue
                out.extend(self._port_of(inst, pname, role, loc))
            return out
        for split in (len(segs) - 1, len(segs)):
            ipath = ctx.prefix + ".".join(segs[:split])
            pname = segs[split] if split < len(segs) else None
            inst = self.flat.instance(ipath)
            if inst is not None:
                return self._port_of(inst, pname, role, loc)
            # composite instance port -> junction
            if pname is not None and f"{ipath}.#{pname}" in self.flat.by_path:
                j = self.flat.instance(f"{ipath}.#{pname}")
                return [j.ports["out" if role == "src" else "in"]]
        cands = [p for p in self.flat.by_path if not p.count("#")]
        self.diags.error(f"unknown instance in connection: '{'.'.join(segs)}'", loc, "E-CONN",
                         hint=suggest(".".join(segs[:-1]) or segs[0], cands))
        return []

    def _port_of(self, inst: Instance, pname: Optional[str], role: str, loc: Loc) -> list:
        want = "out" if role == "src" else "in"
        if pname is None:
            cands = [n for n, pid in inst.ports.items() if self.flat.ports[pid].direction == want]
            if len(cands) == 1:
                return [inst.ports[cands[0]]]
            self.diags.error(f"'{inst.path}' has {len(cands)} {want}-ports; name one explicitly", loc, "E-CONN")
            return []
        pid = inst.ports.get(pname)
        if pid is None:
            self.diags.error(f"'{inst.path}' ({inst.type.short}) has no port '{pname}'", loc, "E-CONN",
                             hint=suggest(pname, list(inst.ports)))
            return []
        return [pid]

    def _segments(self, e: A.Expr, ctx: "_Ctx") -> Optional[list]:
        if isinstance(e, A.Name):
            return [e.id]
        if isinstance(e, A.Template):
            return template_str(e, ctx.scope, base_dir=ctx.base_dir).split(".")
        if isinstance(e, A.Member):
            head = self._segments(e.obj, ctx)
            if head is None:
                return None
            attr = e.attr if isinstance(e.attr, str) else template_str(e.attr, ctx.scope, base_dir=ctx.base_dir)
            return head + [attr]
        if isinstance(e, A.Str):
            return e.value.split(".")
        return None

    def _connect(self, st: A.ConnectStmt, ctx: "_Ctx") -> None:
        try:
            srcs = self._endpoint(st.src, ctx, "src", st.loc)
            dsts = self._endpoint(st.dst, ctx, "dst", st.loc)
        except EvalError as ex:
            self.diags.error(str(ex), ex.loc or st.loc, "E-CONN")
            return
        for s in srcs:
            for d in dsts:
                sp, dp = self.flat.ports[s], self.flat.ports[d]
                if sp.direction == "in" and dp.direction == "out":
                    self.diags.warning(f"connect({self._pname(sp)}, {self._pname(dp)}) is reversed; "
                                       "interpreting as out -> in", st.loc, "W-CONN")
                    sp, dp = dp, sp
                if sp.direction != "out" or dp.direction != "in":
                    self.diags.error(f"cannot connect {sp.direction}-port {self._pname(sp)} to "
                                     f"{dp.direction}-port {self._pname(dp)}", st.loc, "E-CONN",
                                     hint="connections go from an out-port to an in-port")
                    continue
                if st.via is not None:
                    self._via(sp, dp, st, ctx)
                else:
                    self._add_edge(sp, dp, st.loc)

    def _pname(self, p: Port) -> str:
        return f"{self.flat.instances[p.inst].path}.{p.name}"

    def _add_edge(self, sp: Port, dp: Port, loc: Loc, label: str = "") -> None:
        for eid in sp.edges:
            if self.flat.edges[eid].dst == dp.id:
                self.diags.warning(f"duplicate connection {self._pname(sp)} -> {self._pname(dp)}", loc, "W-CONN")
                return
        e = Edge(id=len(self.flat.edges), src=sp.id, dst=dp.id, loc=loc, label=label)
        self.flat.edges.append(e)
        sp.edges.append(e.id)
        dp.edges.append(e.id)

    def _via(self, sp: Port, dp: Port, st: A.ConnectStmt, ctx: "_Ctx") -> None:
        tname = template_str(st.via, ctx.scope, base_dir=ctx.base_dir)
        typ = self.lib.get_type(tname, ctx.pkg, st.via.loc)
        if typ is None:
            return
        ins = [n for n, p in typ.ports.items() if p.direction == "in"]
        outs = [n for n, p in typ.ports.items() if p.direction == "out"]
        if len(ins) < 1 or len(outs) != 1:
            self.diags.error(f"'via {tname}': link entities need an in-port and exactly one out-port",
                             st.loc, "E-VIA")
            return
        a, b = self.flat.instances[sp.inst], self.flat.instances[dp.inst]
        if st.name is not None:
            lname = template_str(st.name, ctx.scope, base_dir=ctx.base_dir)
        else:
            base = f"{_short(a)}__{_short(b)}"
            lname, k = base, 2
            while ctx.prefix + lname in self.flat.by_path:
                lname, k = f"{base}_{k}", k + 1
        self._instantiate(lname, typ, st.overrides, ctx, st.loc)
        link = self.flat.instance(ctx.prefix + lname)
        if link is None:
            return
        link.is_link = True
        if link.line is None and a.at and b.at:
            link.line = [a.at, b.at]
        if link.at is None and link.line:
            mid = link.line[len(link.line) // 2] if len(link.line) > 2 else \
                ((link.line[0][0] + link.line[-1][0]) / 2, (link.line[0][1] + link.line[-1][1]) / 2)
            link.at = mid
        self._add_edge(sp, self.flat.ports[link.ports[ins[0]]], st.loc)
        self._add_edge(self.flat.ports[link.ports[outs[0]]], dp, st.loc)

    # ------------------------------------------------------------------ sets / params
    def _apply_set(self, st: A.SetStmt, ctx: "_Ctx", top_level: bool = False) -> None:
        segs = self._segments(st.path, ctx)
        if not segs or len(segs) < 1:
            self.diags.error("invalid 'set' target", st.loc, "E-SET")
            return
        if top_level and segs[0] in (self.flat.env_type.short, self.flat.env_type.name):
            segs = segs[1:]
        elif top_level and segs[0] != self.flat.env_type.short:
            # 'set' for another environment - ignore silently
            if self.lib.resolve(segs[0]) and self.lib.resolve(segs[0]) != self.flat.env_type.name:
                return
        try:
            val = const_eval(st.value, ctx.scope, base_dir=ctx.base_dir)
        except EvalError as ex:
            self.diags.error(str(ex), ex.loc or st.loc, "E-SET")
            return
        if len(segs) == 1 or (len(segs) == 2 and segs[0] == "env"):
            name = segs[-1]
            p = self.flat.env_type.params.get(name)
            if p is None:
                self.diags.error(f"environment has no property '{name}'", st.loc, "E-SET")
                return
            self.flat.env_params[name] = _to_canon(val, st.unit or p.unit, self.diags, st.loc)
            return
        ipath = ctx.prefix + ".".join(segs[:-1]) if not top_level else ".".join(segs[:-1])
        inst = self.flat.instance(ipath)
        if inst is None:
            self.diags.error(f"'set': unknown instance '{ipath}'", st.loc, "E-SET",
                             hint=suggest(ipath, list(self.flat.by_path)))
            return
        name = segs[-1]
        if name not in inst.type.params and name not in inst.type.states and name not in RESERVED_PROPS:
            self.diags.error(f"'set': '{inst.type.short}' has no property '{name}'", st.loc, "E-SET",
                             hint=suggest(name, list(inst.type.params)))
            return
        inst.overrides[name] = (val, st.unit, st.loc)
        if name in RESERVED_PROPS:
            self._apply_reserved(inst, {name: (val, st.unit, st.loc)}, _Ctx.empty())

    def param_value(self, inst: Instance, name: str):
        if name in inst.params:
            return Canon(inst.params[name])
        if name in inst.meta and name not in inst.type.params:
            return inst.meta[name]
        p = inst.type.params.get(name)
        if p is None:
            raise EvalError(f"'{inst.type.short}' has no property '{name}'")
        key = (inst.id, name)
        if key in self._evaluating:
            raise EvalError(f"circular property definition '{name}' in '{inst.path}'", p.loc)
        self._evaluating.add(key)
        try:
            if name in inst.overrides:
                val, unit, oloc = inst.overrides[name]
                if isinstance(val, _InstRef):
                    raise EvalError(f"property '{name}' cannot be an instance reference", oloc)
                if isinstance(val, str):
                    inst.meta[name] = val
                    return val
                v = _to_canon(val, unit or p.unit, self.diags, oloc)
            else:
                if p.default is None:
                    raise EvalError(f"property '{name}' of '{inst.path}' has no value "
                                    f"(declare a default or set it in 'init')", inst.loc)
                if isinstance(p.default, A.Name) and p.default.id == "auto":
                    v = self._auto_value(inst, name, p)
                else:
                    sc = Scope({"env": _EnvRef(self)}, resolver=lambda n: self._param_or_missing(inst, n))
                    raw = const_eval(p.default, sc, base_dir=inst.type.base_dir,
                                     extra={"distance": self._distance})
                    if isinstance(raw, str):
                        inst.meta[name] = raw
                        return raw
                    v = _to_canon(raw, p.unit, self.diags, p.loc)
            inst.params[name] = float(v)
            inst.param_units[name] = p.unit
            return Canon(v)
        finally:
            self._evaluating.discard(key)

    def _param_or_missing(self, inst: Instance, n: str):
        if n in inst.type.params:
            return self.param_value(inst, n)
        return self._global_resolver(n)

    def _auto_value(self, inst: Instance, name: str, p: A.ParamDecl) -> float:
        if inst.line and len(inst.line) >= 2:
            km = sum(haversine_km(a[0], a[1], b[0], b[1]) for a, b in zip(inst.line, inst.line[1:]))
            return km * 1000.0            # canonical length: m
        raise EvalError(f"'{name} = auto' needs a line geometry (use the entity with 'via' or give 'path')", p.loc)

    def _distance(self, a, b):
        if isinstance(a, _InstRef) and isinstance(b, _InstRef) and a.inst.at and b.inst.at:
            return Canon(haversine_km(a.inst.at[0], a.inst.at[1], b.inst.at[0], b.inst.at[1]) * 1000.0)
        raise EvalError("distance(a, b) needs two instances with 'at' coordinates")

    def _eval_params(self, inst: Instance) -> None:
        for name in inst.type.params:
            try:
                self.param_value(inst, name)
            except EvalError as ex:
                self.diags.error(str(ex), ex.loc or inst.loc, "E-ELAB")

    def _eval_states(self, inst: Instance) -> None:
        for name, s in inst.type.states.items():
            try:
                if name in inst.overrides:
                    val, unit, oloc = inst.overrides[name]
                    v = _to_canon(val, unit or s.unit, self.diags, oloc)
                else:
                    sc = Scope({"env": _EnvRef(self)}, resolver=lambda n: self._param_or_missing(inst, n))
                    v = _to_canon(const_eval(s.init, sc, base_dir=inst.type.base_dir), s.unit, self.diags, s.loc)
                inst.state0[name] = float(v)
            except EvalError as ex:
                self.diags.error(str(ex), ex.loc or s.loc, "E-ELAB")

    # ------------------------------------------------------------------ geometry / resources
    def _geometry(self) -> None:
        for inst in self.flat.instances:
            if inst.is_junction or inst.line is not None or inst.at is not None:
                continue
            # place un-located instances at their composite parent / upstream neighbour
            for pid in inst.ports.values():
                for eid in self.flat.ports[pid].edges:
                    e = self.flat.edges[eid]
                    other = self.flat.ports[e.src if e.dst == pid else e.dst].inst
                    if self.flat.instances[other].at:
                        inst.at = self.flat.instances[other].at
                        break
                if inst.at:
                    break

    def _propagate_resources(self) -> None:
        ports = self.flat.ports
        for p in ports:
            p.resources = [] if p.declared == ["*"] else list(p.declared)
        wild = {p.id for p in ports if p.declared == ["*"]}
        changed = True
        it = 0
        while changed and it < 1000:
            changed = False
            it += 1
            for e in self.flat.edges:
                s, d = ports[e.src], ports[e.dst]
                if d.id in wild:
                    for r in s.resources:
                        if r not in d.resources:
                            d.resources.append(r); changed = True
                if s.id in wild and d.id not in wild:
                    for r in d.resources:
                        if r not in s.resources:
                            s.resources.append(r); changed = True
            for inst in self.flat.instances:
                ins = [ports[pid] for pid in inst.ports.values() if ports[pid].direction == "in"]
                outs = [ports[pid] for pid in inst.ports.values() if ports[pid].direction == "out"]
                wild_ins = [p for p in ins if p.id in wild]
                union = []
                for p in (wild_ins or ins):
                    for r in p.resources:
                        if r not in union:
                            union.append(r)
                for p in outs:
                    if p.id in wild:
                        for r in union:
                            if r not in p.resources:
                                p.resources.append(r); changed = True
                # backward: wildcard in-ports learn from wildcard out-ports (junction feeding typed ports)
                out_union = []
                for p in outs:
                    if p.id not in wild:
                        continue
                    for r in p.resources:
                        if r not in out_union:
                            out_union.append(r)
                for p in wild_ins:
                    if True:
                        for r in out_union:
                            if r not in p.resources:
                                p.resources.append(r); changed = True
        for e in self.flat.edges:
            s, d = ports[e.src], ports[e.dst]
            e.resources = [r for r in s.resources if r in d.resources]
            dropped = [r.split(".")[-1] for r in s.resources if r not in d.resources]
            if not e.resources:
                self.diags.error(f"incompatible ports: {self._pname(s)} carries "
                                 f"[{', '.join(r.split('.')[-1] for r in s.resources) or 'nothing'}] but "
                                 f"{self._pname(d)} accepts "
                                 f"[{', '.join(r.split('.')[-1] for r in d.resources) or 'nothing'}]",
                                 e.loc, "E-CONN")
            elif dropped and d.id not in wild:
                self.diags.warning(f"{self._pname(d)} does not accept {dropped} from {self._pname(s)}; "
                                   "those flows are dropped", e.loc, "W-CONN")
        for p in ports:
            inst = self.flat.instances[p.inst]
            if p.id in wild and not p.resources and not inst.is_junction:
                self.diags.warning(f"wildcard port {self._pname(p)} is unconnected", inst.loc, "W-CONN")
            if p.direction == "out" and not p.edges and not inst.is_junction and p.resources:
                self.diags.info(f"output {self._pname(p)} is not connected (flows leave the model)", inst.loc)

    def _spec_keys(self) -> None:
        for inst in self.flat.instances:
            wild_sig = []
            for pname, pid in sorted(inst.ports.items()):
                p = self.flat.ports[pid]
                if p.declared == ["*"]:
                    wild_sig.append(f"{pname}:{'+'.join(r.split('.')[-1] for r in p.resources)}")
            inst.spec_key = inst.type.name + (f"<{','.join(wild_sig)}>" if wild_sig else "")

    # ------------------------------------------------------------------ simulation
    def _sim(self, sim_name: Optional[str], env_type: EntityType) -> SimSpec:
        spec = SimSpec()
        sd = None
        if sim_name:
            q = self.lib.resolve(sim_name)
            sd = self.lib.sims.get(q) if q else None
            if sd is None:
                self.diags.error(f"unknown simulation '{sim_name}'", NOLOC, "E-SIM",
                                 hint=suggest(sim_name, list(self.lib.sims)))
                return spec
        else:
            for q, s in self.lib.sims.items():
                if s.env and self.lib.resolve(s.env) == env_type.name:
                    sd = s
                    break
        if sd is None:
            return spec
        spec.name, spec.loc = sd.name, sd.loc
        if sd.env and self.lib.resolve(sd.env) != env_type.name:
            self.diags.warning(f"simulation '{sd.name}' targets environment '{sd.env}'", sd.loc, "W-SIM",
                               hint=suggest(sd.env, [n for n in self.lib.entity_names()]))

        def num(key, default, unit_default="h"):
            if key not in sd.settings:
                return default
            expr, unit = sd.settings[key]
            v = const_eval(expr, Scope({"env": _EnvRef(self)}))
            return float(_to_canon(v, unit or unit_default, self.diags, sd.loc))

        try:
            if "t_end" in sd.settings or "t_step" in sd.settings:
                spec.horizon = num("t_end", 24.0) - num("t_start", 0.0)
                spec.dt = num("t_step", 1.0)
            spec.horizon = num("horizon", spec.horizon)
            spec.dt = num("step", num("dt", spec.dt))
        except EvalError as ex:
            self.diags.error(str(ex), ex.loc or sd.loc, "E-SIM")
        spec.objective = sd.objective
        for k, (expr, unit) in sd.solver.items():
            try:
                spec.solver[k] = const_eval(expr, {})
            except EvalError:
                spec.solver[k] = expr
        for d in sd.inputs:
            spec.inputs[d.target.id] = d.value
            spec.input_units[d.target.id] = d.unit
        return spec


@dataclass
class _Ctx:
    prefix: str
    scope: Any
    aliases: dict
    parent: Any
    base_dir: str
    pkg: str
    comp_inst: Optional[str]
    comp_at: Optional[tuple] = None

    def child(self, values: dict) -> "_Ctx":
        return _Ctx(self.prefix, self.scope.child(values), self.aliases, self.parent, self.base_dir, self.pkg,
                    self.comp_inst, self.comp_at)

    @staticmethod
    def empty() -> "_Ctx":
        return _Ctx("", Scope(), {}, None, ".", "", None)


def _short(inst: Instance) -> str:
    return inst.path.replace(".#", "_").replace(".", "_")


def _to_canon(v, unit: Optional[str], diags: Diagnostics, loc: Loc) -> float:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (tuple, list)):
        raise EvalError("expected a number but got a tuple/list", loc)
    if isinstance(v, str):
        try:
            v = float(v)
        except ValueError:
            raise EvalError(f"expected a number but got '{v}'", loc)
    if v is None:
        raise EvalError("missing value", loc)
    if is_canon(v) or not unit:
        return float(v)
    try:
        f, _ = parse_unit(unit)
    except UnitError as ex:
        diags.warning(str(ex), loc, "W-UNIT")
        return float(v)
    if math.isinf(v):
        return float(v)
    return float(v) * f


def elaborate(lib: Library, env: str, sim: Optional[str] = None) -> FlatModel:
    el = Elaborator(lib)
    flat = el.elaborate(env, sim)
    lib.diags.raise_if_errors()
    return flat
