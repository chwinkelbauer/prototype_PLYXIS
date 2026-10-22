"""PLX back end: scheduling + code generation.

The elaborated netlist is grouped by (specialised type, topological level) so
that every behavior statement is executed *once per group*, vectorised over
all instances of that group and over a batch of candidate schedules.

The global signal graph (nodes = group signals, edges = instant / delayed
dependencies) is decomposed into strongly connected components:

* trivial component           -> computed for all time steps at once
* state accumulator  x <- x+e -> exclusive cumulative sum along time
* any other feedback loop     -> sequential scan over time (only for that loop)

This "time vectorisation" is what makes the twin cheap enough to be evaluated
thousands of times per second by the optimizer, and it keeps everything
differentiable for gradient-based solvers and neural sub-solvers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from . import ast as A
from . import rt
from .behavior import (C, BehaviorCompiler, ConInfo, IR, Op, Ref, TypeIR, chan_key, instant_refs,
                       delayed_refs)
from .elaborate import FlatModel
from .errors import Diagnostics, NOLOC

# ----------------------------------------------------------------------------- structures


@dataclass
class Group:
    gid: str
    key: str
    level: int
    ir: TypeIR
    inst_ids: list
    is_env: bool = False

    @property
    def n(self) -> int:
        return len(self.inst_ids) if not self.is_env else 1


@dataclass
class GNode:
    nid: str
    gid: str
    key: str
    kind: str                 # calc | in | state | agg | delay
    expr: Any = None
    inst: set = field(default_factory=set)
    dly: set = field(default_factory=set)
    static: bool = False
    var: str = ""


@dataclass
class Block:
    kind: str                 # vec | acc | scan
    nodes: list               # nids (topological for scan)
    state: Optional[str] = None
    inc: Any = None


@dataclass
class RoutedPort:
    port: int                 # flat port id
    inst: int
    name: str
    ridx: list                # indices into R for each edge (in edge order)
    edges: list               # flat edge ids
    exclusive: bool


def _ident(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", s)


def split_accumulator(e: IR, x: str) -> Optional[IR]:
    """If e == state(x) + rest (rest free of x) return rest, else None."""
    def has_x(z) -> bool:
        return ("state", x) in instant_refs(z)

    def rec(z, sign):
        # returns (found, rest_terms)
        if isinstance(z, Ref) and z.kind == "state" and z.name == x:
            return (sign == 1, [])
        if isinstance(z, Op) and z.op in ("add", "sub"):
            a, b = z.args
            fa, ra = rec(a, sign) if has_x(a) else (False, [(sign, a)])
            if has_x(a) and not fa:
                return (False, None)
            fb, rb = rec(b, sign if z.op == "add" else -sign) if has_x(b) else \
                (False, [(sign if z.op == "add" else -sign, b)])
            if has_x(b) and not fb:
                return (False, None)
            if ra is None or rb is None:
                return (False, None)
            if fa and fb:
                return (False, None)
            return (fa or fb, ra + rb)
        if has_x(z):
            return (False, None)
        return (False, [(sign, z)])

    found, terms = rec(e, 1)
    if not found or terms is None:
        return None
    out: Optional[IR] = None
    for s, term in terms:
        t = term if s == 1 else Op("neg", [term])
        out = t if out is None else Op("add", [out, t])
    return out if out is not None else C(0.0)


# ----------------------------------------------------------------------------- compiler
class Compiler:
    def __init__(self, flat: FlatModel, diags: Optional[Diagnostics] = None, physical: bool = False):
        self.flat = flat
        self.diags = diags if diags is not None else flat.lib.diags
        self.physical = physical
        self.groups: list[Group] = []
        self.gmap: dict[str, Group] = {}
        self.inst_group: dict[int, tuple] = {}     # inst id -> (gid, local idx)
        self.nodes: dict[str, GNode] = {}
        self.blocks: list[Block] = []
        self.K: dict[str, torch.Tensor] = {}
        self.routed: list[RoutedPort] = []
        self.edge_ridx: dict[int, int] = {}
        self.type_irs: dict[str, TypeIR] = {}

    # ------------------------------------------------------------------ pipeline
    def compile(self) -> "Program":
        bc = BehaviorCompiler(self.flat.lib, self.diags)
        self._type_irs(bc)
        levels = self._levels()
        self._make_groups(levels)
        self._routing()
        self._env_group(bc)
        self.diags.raise_if_errors()
        self._graph()
        self._schedule()
        self.diags.raise_if_errors()
        src = self._codegen()
        bsrc = self._codegen_bounds()
        from .program import Program
        return Program(self, src, bsrc)

    def _type_irs(self, bc: BehaviorCompiler) -> None:
        seen: dict[str, Any] = {}
        for inst in self.flat.instances:
            if inst.spec_key in seen:
                continue
            seen[inst.spec_key] = inst
            port_res = {pname: self.flat.ports[pid].resources for pname, pid in inst.ports.items()}
            ir = bc.compile(inst.spec_key, inst.type, port_res, env_params=self.flat.env_params,
                            env_units=self.flat.env_units)
            self._lower_modes(ir)
            self.type_irs[inst.spec_key] = ir

    def _lower_modes(self, ir: TypeIR) -> None:
        """Mode-dependent operators: sat(a, b) is min(a, b) physically and a for the planner."""
        def rw(e):
            if isinstance(e, Op):
                args = [rw(a) for a in e.args]
                if e.op == "sat":
                    return Op("min", args) if self.physical else args[0]
                return Op(e.op, args, e.extra)
            return e
        for k in list(ir.nodes):
            ir.nodes[k] = rw(ir.nodes[k])
        for ci in ir.cons.values():
            ci.lhs, ci.rhs = rw(ci.lhs), rw(ci.rhs)
        bc_order = BehaviorCompiler.__dict__["_order_and_feed"]
        holder = type("H", (), {})()
        holder.ir, holder.typ, holder.diags = ir, ir.type, self.diags
        bc_order(holder)

    def _levels(self) -> dict:
        """Topological levels of instances along feed-through paths (breaks same-type chains)."""
        n = len(self.flat.instances)
        succ: list = [[] for _ in range(n)]
        indeg = [0] * n
        for e in self.flat.edges:
            sp, dp = self.flat.ports[e.src], self.flat.ports[e.dst]
            j = dp.inst
            ir = self.type_irs[self.flat.instances[j].spec_key]
            if any(dp.name in ins for ins in ir.feed.values()):
                succ[sp.inst].append(j)
                indeg[j] += 1
        level = [0] * n
        order = [i for i in range(n) if indeg[i] == 0]
        k = 0
        while k < len(order):
            i = order[k]
            k += 1
            for j in succ[i]:
                level[j] = max(level[j], level[i] + 1)
                indeg[j] -= 1
                if indeg[j] == 0:
                    order.append(j)
        if len(order) < n:
            # cycle among feed-through instances: keep them on one level; the node-level
            # analysis reports a genuine algebraic loop if there is one
            for i in range(n):
                if indeg[i] > 0:
                    level[i] = max(level[i], 0)
        return {i: level[i] for i in range(n)}

    def _make_groups(self, levels: dict) -> None:
        by_key: dict[tuple, list] = {}
        for inst in self.flat.instances:
            by_key.setdefault((inst.spec_key, levels[inst.id]), []).append(inst.id)
        for k, ((key, lvl), ids) in enumerate(sorted(by_key.items(), key=lambda kv: (kv[0][1], kv[0][0]))):
            g = Group(gid=f"g{k}", key=key, level=lvl, ir=self.type_irs[key], inst_ids=ids)
            self.groups.append(g)
            self.gmap[g.gid] = g
            for li, iid in enumerate(ids):
                self.inst_group[iid] = (g.gid, li)

    def _routing(self) -> None:
        for p in self.flat.ports:
            if p.direction != "out" or len(p.edges) < 2:
                continue
            rp = RoutedPort(port=p.id, inst=p.inst, name=p.name, ridx=[], edges=list(p.edges),
                            exclusive=p.exclusive)
            for eid in p.edges:
                r = len(self.edge_ridx)
                self.edge_ridx[eid] = r
                rp.ridx.append(r)
            self.routed.append(rp)
        self.n_routes = len(self.edge_ridx)

    def _env_group(self, bc: BehaviorCompiler) -> None:
        et = self.flat.env_type
        has = any(not isinstance(s, A.DecisionStmtLegacy) for s in et.behavior) or et.constraints
        self.env: Optional[Group] = None
        if not has:
            return
        ir = bc.compile("__env__", et, {}, is_env=True, env_params=self.flat.env_params,
                        env_units=self.flat.env_units, agg_resolver=self._agg_resolver)
        self._lower_modes(ir)
        g = Group(gid="env", key="__env__", level=10 ** 6, ir=ir, inst_ids=[], is_env=True)
        self.env = g
        self.groups.append(g)
        self.gmap["env"] = g

    def _select(self, sel: str):
        f = self.flat
        if sel in f.by_path:
            gid, li = self.inst_group[f.by_path[sel]]
            return [(self.gmap[gid], [li])], True
        if sel in f.inst_groups:
            by: dict = {}
            for iid in f.inst_groups[sel]:
                if f.instances[iid].is_junction:
                    continue
                gid, li = self.inst_group[iid]
                by.setdefault(gid, []).append(li)
            return [(self.gmap[gid], lis) for gid, lis in by.items()], False
        out = [(g, None) for g in self.groups if not g.is_env and g.ir.type.is_a(sel)]
        if out:
            return out, False
        return None, False

    def _signal_ref(self, ir: TypeIR, sig: list) -> Optional[tuple]:
        h = sig[0]
        if h in ir.ports and len(sig) >= 2:
            p = ir.ports[h]
            res = next((r for r in p.resources if r.split(".")[-1] == sig[1] or r == sig[1]), None)
            if res is None:
                return ("const", 0.0)
            k = chan_key(h, res, sig[2] if len(sig) > 2 else None)
            return ("in", k) if p.direction == "in" else ("node", f"out:{k}")
        if len(sig) == 1:
            if f"local:{h}" in ir.nodes:
                return ("node", f"local:{h}")
            if h in ir.states:
                return ("state", h)
            if h in ir.decisions:
                return ("dec", h)
            if h in ir.params:
                return ("param", h)
        if len(sig) == 2 and h in ("cost", "revenue", "emission", "metric"):
            k = f"term:{h}:{sig[1]}"
            if k in ir.nodes:
                return ("node", k)
        return None

    def _agg_resolver(self, bc, path: list, how: str, loc):
        for k in range(len(path) - 1, 0, -1):
            sel, sig = ".".join(path[:k]), path[k:]
            targets, single = self._select(sel)
            if targets is None:
                continue
            parts = []
            for g, lis in targets:
                ref = self._signal_ref(g.ir, sig)
                if ref is None:
                    self.diags.error(f"'{g.ir.type.short}' has no signal '{'.'.join(sig)}'", loc, "E-AGG")
                    return C(0.0)
                if ref[0] == "const":
                    continue
                parts.append((g.gid, ref, lis))
            if how == "value" and not single:
                self.diags.error(f"'{'.'.join(path)}' selects several instances; aggregate it with "
                                 f"sum(...), mean(...), max(...) or min(...)", loc, "E-AGG")
                return C(0.0)
            name = f"a{len(bc.ir.aggs)}"
            bc.ir.aggs[name] = (parts, "sum" if how == "value" else how)
            return Ref("agg", name)
        if len(path) == 1:
            targets, _ = self._select(path[0])
            if targets is not None and how == "count":
                name = f"a{len(bc.ir.aggs)}"
                bc.ir.aggs[name] = ([(g.gid, ("const", 1.0), lis) for g, lis in targets], "count")
                return Ref("agg", name)
        return None

    # ------------------------------------------------------------------ graph
    def _nid(self, gid: str, key: str) -> str:
        return f"{gid}|{key}"

    def _ref_nid(self, g: Group, kind: str, name: str) -> Optional[str]:
        if kind == "node":
            return self._nid(g.gid, name)
        if kind == "in":
            return self._nid(g.gid, f"in:{name}")
        if kind == "state":
            return self._nid(g.gid, f"state:{name}")
        if kind == "agg":
            return self._nid("env", f"agg:{name}")
        return None

    def _graph(self) -> None:
        for g in self.groups:
            ir = g.ir
            for k, e in ir.nodes.items():
                kind = "delay" if isinstance(e, Op) and e.op == "delay" else "calc"
                gn = GNode(self._nid(g.gid, k), g.gid, k, kind, e)
                for rk, rn in instant_refs(e):
                    nid = self._ref_nid(g, rk, rn)
                    if nid:
                        gn.inst.add(nid)
                if kind == "delay":
                    gn.dly.add(self._nid(g.gid, e.args[0].name))
                self.nodes[gn.nid] = gn
            for sname, s in ir.states.items():
                gn = GNode(self._nid(g.gid, f"state:{sname}"), g.gid, f"state:{sname}", "state")
                if s.has_update:
                    gn.dly.add(self._nid(g.gid, f"next:{sname}"))
                if self.physical and s.lo is not None:
                    for rk, rn in instant_refs(s.lo) | instant_refs(s.hi):
                        nid = self._ref_nid(g, rk, rn)
                        if nid:
                            gn.dly.add(nid)
                self.nodes[gn.nid] = gn
            if g.is_env:
                for a, (parts, how) in ir.aggs.items():
                    gn = GNode(self._nid("env", f"agg:{a}"), "env", f"agg:{a}", "agg", (parts, how))
                    for gid, ref, lis in parts:
                        nid = self._ref_nid(self.gmap[gid], ref[0], ref[1]) if ref[0] != "const" else None
                        if nid:
                            gn.inst.add(nid)
                    self.nodes[gn.nid] = gn
                continue
            # in-port channels: transfers from upstream out-ports
            for pname, r, a in ir.in_channels():
                ck = chan_key(pname, r, a)
                spec: dict = {}
                for li, iid in enumerate(g.inst_ids):
                    inst = self.flat.instances[iid]
                    port = self.flat.ports[inst.ports[pname]]
                    for eid in port.edges:
                        e = self.flat.edges[eid]
                        if r not in e.resources:
                            continue
                        sp = self.flat.ports[e.src]
                        sgid, sli = self.inst_group[sp.inst]
                        skey = f"out:{chan_key(sp.name, r, a)}"
                        if skey not in self.gmap[sgid].ir.nodes:
                            continue
                        spec.setdefault((sgid, skey), []).append((sli, li, self.edge_ridx.get(eid, -1)))
                gn = GNode(self._nid(g.gid, f"in:{ck}"), g.gid, f"in:{ck}", "in", spec)
                for (sgid, skey) in spec:
                    gn.inst.add(self._nid(sgid, skey))
                self.nodes[gn.nid] = gn
        # sanity: every referenced node exists
        for gn in self.nodes.values():
            for d in list(gn.inst) + list(gn.dly):
                if d not in self.nodes:
                    self.diags.error(f"internal: unresolved dependency {d} of {gn.nid}", NOLOC, "E-INTERNAL")

    def _schedule(self) -> None:
        nodes = self.nodes
        index: dict = {}
        low: dict = {}
        on: set = set()
        stack: list = []
        sccs: list = []
        counter = [0]
        import sys
        sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))

        def deps(v):
            gn = nodes[v]
            return list(gn.inst) + list(gn.dly)

        # iterative Tarjan
        for v0 in nodes:
            if v0 in index:
                continue
            work = [(v0, iter(deps(v0)))]
            index[v0] = low[v0] = counter[0]; counter[0] += 1
            stack.append(v0); on.add(v0)
            while work:
                v, it = work[-1]
                advanced = False
                for w in it:
                    if w not in index:
                        index[w] = low[w] = counter[0]; counter[0] += 1
                        stack.append(w); on.add(w)
                        work.append((w, iter(deps(w))))
                        advanced = True
                        break
                    elif w in on:
                        low[v] = min(low[v], index[w])
                if advanced:
                    continue
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[v])
                if low[v] == index[v]:
                    comp = []
                    while True:
                        w = stack.pop(); on.discard(w)
                        comp.append(w)
                        if w == v:
                            break
                    sccs.append(comp)
        for comp in sccs:
            cset = set(comp)
            if len(comp) == 1:
                v = comp[0]
                gn = nodes[v]
                if v not in gn.inst and v not in gn.dly:
                    self.blocks.append(Block("vec", [v]))
                    continue
            # algebraic loop check (instant edges only)
            if self._instant_cycle(cset):
                names = ", ".join(sorted(self._pretty(v) for v in cset if nodes[v].kind != "in"))
                loc = self.gmap[nodes[comp[0]].gid].ir.type.loc
                self.diags.error(f"algebraic loop (combinational feedback) through: {names}", loc, "E-LOOP",
                                 hint="every feedback path must pass through a state (s <- ...) or delay()")
                continue
            states = [v for v in comp if nodes[v].kind == "state"]
            if len(comp) == 2 and len(states) == 1:
                st = nodes[states[0]]
                sname = st.key.split(":", 1)[1]
                nxt = nodes.get(self._nid(st.gid, f"next:{sname}"))
                sinfo = self.gmap[st.gid].ir.states[sname]
                if nxt is not None and nxt.nid in cset and not (self.physical and sinfo.lo is not None):
                    inc = split_accumulator(nxt.expr, sname)
                    if inc is not None:
                        self.blocks.append(Block("acc", [states[0], nxt.nid], state=states[0], inc=inc))
                        continue
            self.blocks.append(Block("scan", self._topo(comp)))
        # static analysis in schedule order
        order = [v for b in self.blocks for v in b.nodes]
        for v in order:
            gn = nodes[v]
            gn.static = self._is_static(gn)
            gn.var = "v_" + _ident(gn.nid.replace("|", "_"))

    def _instant_cycle(self, cset: set) -> bool:
        color: dict = {}
        for s in cset:
            if s in color:
                continue
            work = [(s, iter([d for d in self.nodes[s].inst if d in cset]))]
            color[s] = 1
            while work:
                v, it = work[-1]
                adv = False
                for w in it:
                    c = color.get(w)
                    if c == 1:
                        return True
                    if c is None:
                        color[w] = 1
                        work.append((w, iter([d for d in self.nodes[w].inst if d in cset])))
                        adv = True
                        break
                if not adv:
                    color[v] = 2
                    work.pop()
        return False

    def _topo(self, comp: list) -> list:
        cset = set(comp)
        out, seen = [], set()
        states = [v for v in comp if self.nodes[v].kind == "state"]
        for s in states:
            seen.add(s)
            out.append(s)

        def visit(v):
            if v in seen:
                return
            seen.add(v)
            for d in self.nodes[v].inst:
                if d in cset:
                    visit(d)
            out.append(v)
        for v in comp:
            visit(v)
        return out

    def _pretty(self, nid: str) -> str:
        gid, key = nid.split("|", 1)
        g = self.gmap[gid]
        return f"{g.ir.type.short}.{key}"

    def _is_static(self, gn: GNode) -> bool:
        if gn.kind == "state":
            g = self.gmap[gn.gid]
            return not g.ir.states[gn.key.split(":", 1)[1]].has_update
        if gn.kind in ("in", "delay"):
            return False
        if gn.kind == "agg":
            parts, how = gn.expr
            if how == "count":
                return True
            for gid, ref, lis in parts:
                if ref[0] == "dec":
                    return False
                if ref[0] in ("node", "in", "state"):
                    if not self.nodes[self._ref_nid(self.gmap[gid], ref[0], ref[1])].static:
                        return False
            return True
        g = self.gmap[gn.gid]
        for kind, name in instant_refs(gn.expr):
            if kind == "dec":
                return False
            if kind == "b" and name != "dt":
                return False
            if kind == "env" and name in self.flat.env_inputs:
                return False
            if kind == "param" and g.is_env and name in self.flat.env_inputs:
                return False
            nid = self._ref_nid(g, kind, name)
            if nid and not self.nodes[nid].static:
                return False
        return True

    # ------------------------------------------------------------------ codegen
    def _kidx(self, vals: list, name_hint: str) -> str:
        k = f"k{len(self.K)}"
        self.K[k] = torch.tensor(vals, dtype=torch.long)
        return k

    def _emit(self, e: IR, R) -> str:
        if isinstance(e, C):
            return repr(float(e.value))
        if isinstance(e, Ref):
            return R(e.kind, e.name)
        a = [self._emit(x, R) for x in e.args] if e.op not in ("pwl",) else None
        op = e.op
        if op == "add": return f"({a[0]} + {a[1]})"
        if op == "sub": return f"({a[0]} - {a[1]})"
        if op == "mul": return f"({a[0]} * {a[1]})"
        if op == "div": return f"rt.div({a[0]}, {a[1]})"
        if op == "pow": return f"rt.pow({a[0]}, {a[1]})"
        if op == "mod": return f"({a[0]} % {a[1]})"
        if op == "neg": return f"(-{a[0]})"
        if op == "not": return f"(1.0 - {a[0]})"
        if op == "and": return f"({a[0]} * {a[1]})"
        if op == "or": return f"rt.or_({a[0]}, {a[1]})"
        if op in ("lt", "le", "gt", "ge", "eq", "ne"): return f"rt.{op}({a[0]}, {a[1]})"
        if op == "min": return f"rt.minn({', '.join(a)})"
        if op == "max": return f"rt.maxn({', '.join(a)})"
        if op == "sum": return "(" + " + ".join(a) + ")"
        if op == "mean": return "((" + " + ".join(a) + f") / {float(len(a))})"
        if op == "pos": return f"rt.relu({a[0]})"
        if op in ("abs", "sqrt", "exp", "log", "log10", "sin", "cos", "tanh", "sigmoid", "floor", "ceil",
                  "round", "sign", "step", "negpart"):
            return f"rt.{op}({a[0]})"
        if op == "clamp": return f"rt.clamp({a[0]}, {a[1]}, {a[2]})"
        if op == "where": return f"rt.where({a[0]}, {a[1]}, {a[2]})"
        if op in ("smin", "smax"): return f"rt.{op}({a[0]}, {a[1]}, {a[2]})"
        if op == "safediv": return f"rt.safediv({a[0]}, {a[1]})"
        if op == "pwl":
            x = self._emit(e.args[0], R)
            ys = [self._emit(y, R) for y in e.args[1:]]
            return f"rt.pwl({x}, {tuple(e.extra['xs'])!r}, {', '.join(ys)})"
        raise ValueError(f"cannot emit op {op}")

    def _renderer(self, g: Group, mode: str, scan: Optional[set] = None, final: bool = False):
        """Return a function (kind, name) -> code for references inside group g."""
        def R(kind: str, name: str) -> str:
            static = True
            if kind == "param":
                if g.is_env:
                    var = f"E_{_ident(name)}"
                    static = name not in self.flat.env_inputs
                else:
                    var = f"P_{g.gid}_{_ident(name)}"
            elif kind == "dec":
                var, static = f"D_{g.gid}_{_ident(name)}", False
            elif kind == "env":
                var = f"E_{_ident(name)}"
                static = name not in self.flat.env_inputs
            elif kind == "b":
                if name == "dt":
                    return "dt"
                var, static = {"t": "TT", "time": "TIME", "hour": "HOUR", "dow": "DOW"}[name], False
            else:
                nid = self._ref_nid(g, kind, name)
                gn = self.nodes[nid]
                if final and gn.kind == "state":
                    return f"{gn.var}_f"
                if mode == "scan" and scan is not None and nid in scan:
                    return f"c_{gn.var}" if gn.kind == "state" else f"{gn.var}_t"
                var, static = gn.var, gn.static
            if mode == "vec":
                return var
            if final:
                return f"{var}[..., 0]" if static else f"{var}[..., -1]"
            return f"{var}[..., 0]" if static else f"{var}[..., t]"
        return R

    def _gather_code(self, gn: GNode, mode: str, scan: Optional[set]) -> str:
        g = self.gmap[gn.gid]
        parts = []
        for (sgid, skey), triples in gn.expr.items():
            sg = self.gmap[sgid]
            src_nid = self._nid(sgid, skey)
            src = self.nodes[src_nid]
            if mode == "scan" and scan is not None and src_nid in scan:
                sv = f"{src.var}_t"
            elif mode == "scan":
                sv = f"{src.var}[..., 0]" if src.static else f"{src.var}[..., t]"
            else:
                sv = src.var
            si = self._kidx([t[0] for t in triples], "s")
            di = self._kidx([t[1] for t in triples], "d")
            if any(t[2] >= 0 for t in triples):
                if not all(t[2] >= 0 for t in triples):
                    # mix of routed and unrouted edges from the same source group: split into two
                    routed = [t for t in triples if t[2] >= 0]
                    plain = [t for t in triples if t[2] < 0]
                    si1, di1, ri1 = (self._kidx([t[0] for t in routed], "s"), self._kidx([t[1] for t in routed], "d"),
                                     self._kidx([t[2] for t in routed], "r"))
                    si2, di2 = self._kidx([t[0] for t in plain], "s"), self._kidx([t[1] for t in plain], "d")
                    parts.append(f"({sv}, K['{si1}'], K['{di1}'], K['{ri1}'])")
                    parts.append(f"({sv}, K['{si2}'], K['{di2}'], None)")
                    continue
                ri = self._kidx([t[2] for t in triples], "r")
                parts.append(f"({sv}, K['{si}'], K['{di}'], K['{ri}'])")
            else:
                parts.append(f"({sv}, K['{si}'], K['{di}'], None)")
        Rn = "R" if mode == "vec" else "R[..., t]"
        tl = "T_len" if mode == "vec" else "None"
        return f"rt.gather_sum(B, {g.n}, {tl}, [{', '.join(parts)}], {Rn})"

    def _agg_code(self, gn: GNode, mode: str, scan: Optional[set]) -> str:
        parts, how = gn.expr
        out = []
        for gid, ref, lis in parts:
            g = self.gmap[gid]
            if ref[0] == "const":
                out.append(f"(torch.full((1, {g.n}, 1), {float(ref[1])!r}), "
                           f"{('K[' + repr(self._kidx(lis, 'a')) + ']') if lis is not None else 'None'})")
                continue
            R = self._renderer(g, mode, scan)
            code = R(ref[0], ref[1])
            idx = f"K['{self._kidx(lis, 'a')}']" if lis is not None else "None"
            out.append(f"({code}, {idx})")
        tl = "T_len" if mode == "vec" else "None"
        return f"rt.agg([{', '.join(out)}], {how!r}, B, {tl})"

    def _codegen(self) -> str:
        L: list = []
        w = lambda s, i=1: L.append("    " * i + s)  # noqa: E731
        L.append("# generated by the PLYXIS PLX compiler - do not edit")
        L.append("import torch")
        L.append("from plyxis.lang import rt")
        L.append("")
        L.append("def simulate(P, D, S0, E, R, K, B, T_len, dt, t0, want_mag=False):")
        w("TT = torch.arange(T_len, dtype=rt._F).view(1, 1, -1)")
        w("TIME = t0 + TT * dt")
        w("HOUR = TIME % 24.0")
        w("DOW = torch.floor(TIME / 24.0) % 7.0")
        for g in self.groups:
            if g.is_env:
                continue
            for p in g.ir.params:
                w(f"P_{g.gid}_{_ident(p)} = P['{g.gid}'].get('{p}')")
            for d in g.ir.decisions:
                w(f"D_{g.gid}_{_ident(d)} = D['{g.gid}']['{d}']")
            for s in g.ir.states:
                w(f"S0_{g.gid}_{_ident(s)} = S0['{g.gid}']['{s}']")
        for name in self.flat.env_params:
            w(f"E_{_ident(name)} = E['{name}']")
        if self.env is not None:
            for s in self.env.ir.states:
                w(f"S0_env_{_ident(s)} = S0['env']['{s}']")
        out_names: list = []
        for b in self.blocks:
            if b.kind == "vec":
                gn = self.nodes[b.nodes[0]]
                g = self.gmap[gn.gid]
                w(f"# {self._pretty(gn.nid)}")
                w(f"{gn.var} = {self._node_code(gn, g, 'vec', None)}")
                if gn.kind == "state" and g.ir.states[gn.key.split(':', 1)[1]].has_update:
                    sname = gn.key.split(":", 1)[1]
                    nxt = self.nodes[self._nid(g.gid, f"next:{sname}")]
                    clamp = self._clamp_code(g, sname, nxt.var, "vec")
                    w(f"{gn.var}_f = rt.full({clamp}, B, {g.n}, T_len)[..., -1]")
                    if self.physical and g.ir.states[sname].lo is not None:
                        w(f"{gn.var}_clip = rt.full({nxt.var}, B, {g.n}, T_len) - rt.full({clamp}, B, {g.n}, T_len)")
                        out_names.append(f"{gn.var}_clip")
                    out_names.append(f"{gn.var}_f")
                elif gn.kind == "state":
                    w(f"{gn.var}_f = rt.full(S0_{g.gid}_{_ident(gn.key.split(':', 1)[1])}, B, {g.n})")
                    out_names.append(f"{gn.var}_f")
                out_names.append(gn.var)
            elif b.kind == "acc":
                st, nx = self.nodes[b.nodes[0]], self.nodes[b.nodes[1]]
                g = self.gmap[st.gid]
                sname = st.key.split(":", 1)[1]
                R = self._renderer(g, "vec")
                w(f"# accumulator {self._pretty(st.nid)} (cumsum)")
                w(f"_inc = rt.full({self._emit(b.inc, R)}, B, {g.n}, T_len)")
                w(f"{st.var} = S0_{g.gid}_{_ident(sname)}.unsqueeze(-1) + rt.excumsum(_inc)")
                w(f"{nx.var} = {st.var} + _inc")
                w(f"{st.var}_f = {nx.var}[..., -1]")
                out_names += [st.var, nx.var, f"{st.var}_f"]
            else:
                self._emit_scan(b, w, out_names)
        # final terms
        for g in self.groups:
            for fk, fe in g.ir.finals.items():
                R = self._renderer(g, "scan", None, final=True)
                var = f"F_{g.gid}_{_ident(fk)}"
                w(f"{var} = rt.full({self._emit(fe, R)}, B, {g.n})")
                out_names.append(var)
        w("_out = {" + ", ".join(f"'{n}': {n}" for n in out_names) + "}")
        w("if want_mag:")
        for g in self.groups:
            R = self._renderer(g, "vec")
            for ck, ci in g.ir.cons.items():
                w(f"_out['mag|{g.gid}|{ck}'] = rt.maxn(rt.abs({self._emit(ci.lhs, R)}), "
                  f"rt.abs({self._emit(ci.rhs, R)}))", 2)
        w("return _out")
        return "\n".join(L) + "\n"

    def _clamp_code(self, g: Group, sname: str, nxt_code: str, mode: str, scan=None) -> str:
        s = g.ir.states[sname]
        if not self.physical or s.lo is None:
            return nxt_code
        R = self._renderer(g, mode, scan)
        return f"rt.clamp({nxt_code}, {self._emit(s.lo, R)}, {self._emit(s.hi, R)})"

    def _node_code(self, gn: GNode, g: Group, mode: str, scan: Optional[set]) -> str:
        if gn.kind == "calc":
            return self._emit(gn.expr, self._renderer(g, mode, scan))
        if gn.kind == "in":
            return self._gather_code(gn, mode, scan)
        if gn.kind == "agg":
            return self._agg_code(gn, mode, scan)
        if gn.kind == "state":
            sname = gn.key.split(":", 1)[1]
            if not g.ir.states[sname].has_update:
                return f"S0_{g.gid}_{_ident(sname)}.unsqueeze(-1)"
            nxt = self.nodes[self._nid(g.gid, f"next:{sname}")]
            clamp = self._clamp_code(g, sname, f"rt.full({nxt.var}, B, {g.n}, T_len)", "vec")
            return f"rt.shift({clamp}, S0_{g.gid}_{_ident(sname)})"
        if gn.kind == "delay":
            R = self._renderer(g, "vec")
            src = self.nodes[self._nid(g.gid, gn.expr.args[0].name)]
            k = gn.expr.extra["k"]
            kc = repr(int(round(k.value))) if isinstance(k, C) else self._emit(k, R)
            init = self._emit(gn.expr.extra["init"], R)
            return f"rt.delay({src.var}, {kc}, {init}, T_len)"
        raise ValueError(gn.kind)

    def _emit_scan(self, b: Block, w, out_names: list) -> None:
        scan = set(b.nodes)
        w(f"# feedback loop scanned over time: {', '.join(self._pretty(v) for v in b.nodes)}")
        for v in b.nodes:
            gn = self.nodes[v]
            g = self.gmap[gn.gid]
            if gn.kind == "state":
                sname = gn.key.split(":", 1)[1]
                w(f"c_{gn.var} = rt.full(S0_{g.gid}_{_ident(sname)}, B, {g.n})")
            w(f"L_{gn.var} = []")
        w("for t in range(T_len):")
        for v in b.nodes:
            gn = self.nodes[v]
            g = self.gmap[gn.gid]
            if gn.kind == "state":
                w(f"L_{gn.var}.append(c_{gn.var})", 2)
                continue
            if gn.kind == "delay":
                src = self.nodes[self._nid(g.gid, gn.expr.args[0].name)]
                k = gn.expr.extra["k"]
                if not isinstance(k, C):
                    self.diags.error("per-instance delays inside feedback loops are not supported",
                                     NOLOC, "E-DELAY")
                    k = C(1.0)
                kk = int(round(k.value))
                R = self._renderer(g, "scan", scan)
                init = self._emit(gn.expr.extra["init"], R)
                w(f"{gn.var}_t = L_{src.var}[t - {kk}] if t >= {kk} else rt.full({init}, B, {g.n})", 2)
            else:
                code = self._node_code(gn, g, "scan", scan)
                w(f"{gn.var}_t = rt.full({code}, B, {g.n})", 2)
            w(f"L_{gn.var}.append({gn.var}_t)", 2)
        for v in b.nodes:
            gn = self.nodes[v]
            if gn.kind != "state":
                continue
            g = self.gmap[gn.gid]
            sname = gn.key.split(":", 1)[1]
            nxt = self.nodes[self._nid(g.gid, f"next:{sname}")]
            nref = f"{nxt.var}_t" if nxt.nid in scan else \
                (f"{nxt.var}[..., 0]" if nxt.static else f"{nxt.var}[..., t]")
            clamp = self._clamp_code(g, sname, nref, "scan", scan)
            w(f"c_{gn.var} = rt.full({clamp}, B, {g.n})", 2)
        for v in b.nodes:
            gn = self.nodes[v]
            w(f"{gn.var} = torch.stack(L_{gn.var}, -1)")
            out_names.append(gn.var)
        for v in b.nodes:
            gn = self.nodes[v]
            if gn.kind == "state":
                w(f"{gn.var}_f = c_{gn.var}")
                out_names.append(f"{gn.var}_f")
                g = self.gmap[gn.gid]
                sname = gn.key.split(":", 1)[1]
                if self.physical and g.ir.states[sname].lo is not None:
                    nxt = self.nodes[self._nid(g.gid, f"next:{sname}")]
                    nv = nxt.var
                    w(f"{gn.var}_clip = {nv} - torch.cat([{gn.var}[..., 1:], {gn.var}_f.unsqueeze(-1)], -1)")
                    out_names.append(f"{gn.var}_clip")

    def _codegen_bounds(self) -> str:
        L = ["# decision bounds, defaults and soft-constraint weights (property-only expressions)",
             "import torch", "from plyxis.lang import rt", "", "def bounds(P, E, dt):", "    out = {}"]
        for g in self.groups:
            R = self._renderer(g, "vec")
            L.append(f"    o = out.setdefault('{g.gid}', {{}})")
            for d, info in g.ir.decisions.items():
                lo = self._emit(info.lo, R)
                hi = self._emit(info.hi, R)
                df = self._emit(info.default, R) if info.default is not None else "None"
                L.append(f"    o['dec:{d}'] = ({lo}, {hi}, {df})")
            for ck, ci in g.ir.cons.items():
                if ci.soft is not None:
                    L.append(f"    o['soft:{ck}'] = {self._emit(ci.soft, R)}")
            for sname, s in g.ir.states.items():
                if s.lo is not None:
                    L.append(f"    o['bound:{sname}'] = ({self._emit(s.lo, R)}, {self._emit(s.hi, R)})")
        L.append("    return out")
        # bind parameter names
        head = []
        for g in self.groups:
            if g.is_env:
                continue
            for p in g.ir.params:
                head.append(f"    P_{g.gid}_{_ident(p)} = P['{g.gid}'].get('{p}')")
        for name in self.flat.env_params:
            head.append(f"    E_{_ident(name)} = E['{name}']")
        idx = L.index("    out = {}")
        return "\n".join(L[:idx] + head + L[idx:]) + "\n"


def compile_flat(flat: FlatModel, physical: bool = False):
    from .program import Program  # noqa: F401
    return Compiler(flat, physical=physical).compile()
