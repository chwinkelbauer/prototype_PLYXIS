"""The block structure of a twin's asset graph: which parts can be solved on their own and used by the rest as
input-output functions.

A twin is a graph: machines, stocks, markets and utilities as nodes, the flows between them as edges. Nothing here
knows what a plant, a site or a production line is. `find_structure` cuts the graph at the nodes that decouple their
neighbours, in order, and again inside every piece, which gives a tree:

1. **exchanges several parts share** - nodes that take flows from at least two others, pass nothing on and keep no
   stock (a market, a grid connection, a scrap market). Taken out, the rest falls apart; an exchange whose
   neighbours all lie in one piece belongs to it and goes back in. What a piece trades there is its interface;
2. **time scales** - nodes whose own durations (stated in time in the model: a cycle, a hold, a window) span no whole
   number of the part's steps need a finer step. They seed a piece at that step, which takes in every node that sends
   into it and every receiver whose stock holds less than `buffer_steps` of the part's steps of its flow; it stops at
   larger stocks and at stockless receivers, which take the flow step by step (`hierarchy.detect_plants`);
3. **stocks** - a node whose bounded stock holds at least `buffer_steps` of the part's steps of the flow through it
   decouples what feeds it from what draws from it: the stocks are taken out, the pieces between them are parts.

A piece no rule splits is a cell: its nodes are coupled within a step (a direct flow, a small buffer, a balance). Each
block records what makes it faster to solve than the whole: exact per-unit dynamic programs where its equations hold
a sequence rule (`optim/unitdp.py`), its structure class (LP, MILP, convex, ...: `optim/structure.py`), its own time
step, and how many flows cross its fence against the decisions inside. Which blocks the planner turns into
input-output functions is decided by what paid when measured (`used`, `note`): blocks at a finer step than the part
around them (bench/results/corporate.json); blocks at the same step coordinated by prices did not converge
(corporate_regions*.json) and are reported only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .hierarchy import _crossings, _in_deps, _steps_ok, _subflat, describe, detect_plants, static_value  # noqa: F401

RANK = {"none": 0, "lp": 1, "milp": 2, "slp_quality": 3, "slp": 4, "convex": 5, "nonlinear": 6}


@dataclass
class Block:
    name: str
    members: list                     # instance ids of every unit inside (children's included)
    step: float                       # the step its units need (hours)
    parent_step: float
    depth: int = 0
    own: list = field(default_factory=list)       # units at this level only: the cut points that coordinate children
    children: list = field(default_factory=list)
    rule: str = ""                    # how its children were cut out: exchanges | time scale | stocks | "" (a cell)
    why: list = field(default_factory=list)       # the cut points and what makes them cut points
    interface: list = field(default_factory=list)  # (direction, other path, port, resource, edges)
    channels: int = 0                 # decision channels inside (decisions per step, all its units)
    cls: str = "none"                 # structure class of its hardest unit
    exact: list = field(default_factory=list)     # exact per-unit methods inside
    used: bool = False                # an input-output function in the plan (compile_hierarchy, split="auto")
    note: str = ""

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


class AssetGraph:
    """The twin's graph with what the cut rules read: neighbours, exchanges, stocks and their hours of flow, the step
    each node needs, each type's structure class and sequence rules."""

    def __init__(self, model, buffer_steps: float = 24.0, samples: int = 4, seed: int = 0):
        import torch
        from ..optim.problem import Problem
        self.model, self.buffer_steps = model, buffer_steps
        prog = self.prog = model.program
        flat = self.flat = prog.flat
        self.master = float(prog.dt)
        self.by_id = {}
        for g in prog.groups:
            if g.is_env:
                continue
            for li, iid in enumerate(g.inst_ids):
                self.by_id[iid] = (g, li)
        self.nbrs = {i: set() for i in self.by_id}
        self.senders = {i: set() for i in self.by_id}
        self.passes = set()
        for e in flat.edges:
            u, v = flat.ports[e.src].inst, flat.ports[e.dst].inst
            if u in self.by_id and v in self.by_id and u != v:
                self.nbrs[u].add(v)
                self.nbrs[v].add(u)
                self.senders[v].add(u)
                self.passes.add(u)
        # the step every node needs (the plants' detector's rule, kept in one place)
        from .hierarchy import _duration_rules
        cands = [self.master / n for n in range(1, 721) if self.master / n >= 1.0 / 60 - 1e-9]
        self.need, self.need_why, self._rules_of = {}, {}, {}
        for iid, (g, li) in self.by_id.items():
            rules = self._rules_of.setdefault(g.gid, _duration_rules(g))
            self.need[iid] = self.master
            P = flat.instances[iid].params
            for s in cands:
                bad = [lab for lab, f in rules if not _steps_ok(f(P, s))]
                if not bad:
                    self.need[iid] = s
                    break
                self.need_why[iid] = bad[0]
        # flows through every in-port, measured on the twin at default and random decisions
        pb = Problem(prog)
        gen = torch.Generator().manual_seed(seed)
        u = torch.cat([pb.space.default_u(1), pb.space.random_u(samples, gen)])
        with torch.no_grad():
            self._raw = pb.evaluate(u, relax=True).raw
        self._stock_ports: dict = {}
        self._hours: dict = {}
        self._cls = None
        self._exact = None

    # ------------------------------------------------------------------ node properties
    def _in_flow(self, g, li, port_res) -> float:
        try:
            x = self._raw[self.prog.node_var(g.gid, f"in:{port_res}")]
        except KeyError:
            return 0.0
        x = x.reshape(x.shape[0], -1, x.shape[-1]) if x.dim() >= 2 else x.reshape(1, 1, -1)
        return float(x[:, li if x.shape[1] > 1 else 0].abs().mean())

    def stocks(self, g) -> dict:
        """Bounded states fed by in-ports: state -> the in-ports ("port:res" and "port") its update reads."""
        if g.gid not in self._stock_ports:
            out = {}
            for s, si in g.ir.states.items():
                if si.has_update and si.hi is not None and f"next:{s}" in g.ir.nodes:
                    from .behavior import walk_refs
                    acc, seen, full = set(), set(), set()

                    def rec(e):
                        for kind, name in walk_refs(e):
                            if kind == "in":
                                full.add(name)
                            elif kind == "node" and name not in seen and name in g.ir.nodes:
                                seen.add(name)
                                rec(g.ir.nodes[name])
                    rec(g.ir.nodes[f"next:{s}"])
                    if full:
                        out[s] = full
            self._stock_ports[g.gid] = out
        return self._stock_ports[g.gid]

    def hours(self, iid) -> Optional[tuple]:
        """(hours of its flow the node's smallest stock holds, state), or None if it keeps no stock of its inflows."""
        if iid in self._hours:
            return self._hours[iid]
        g, li = self.by_id[iid]
        P = self.flat.instances[iid].params
        best = None
        for s, ports in self.stocks(g).items():
            si = g.ir.states[s]
            lo = static_value(si.lo, P, self.master) if si.lo is not None else None
            hi = static_value(si.hi, P, self.master) if si.hi is not None else None
            rng = (hi - lo) if (hi is not None and lo is not None) else float("inf")
            thr = sum(self._in_flow(g, li, p) for p in ports)
            h = rng / thr if thr > 1e-9 else float("inf")
            if best is None or h < best[0]:
                best = (h, s)
        self._hours[iid] = best
        return best

    def is_exchange(self, iid) -> bool:
        return len(self.senders[iid]) >= 2 and iid not in self.passes and self.hours(iid) is None

    def decisions(self, iid) -> int:
        return len(self.by_id[iid][0].ir.decisions)

    def cls(self, iid) -> str:
        if self._cls is None:
            from ..optim.structure import classify_types
            try:
                types = classify_types(self.prog)
                self._cls = {gid: t.cls for gid, t in types.items()}
            except Exception:                                   # a class nobody can prove is no class
                self._cls = {}
        return self._cls.get(self.by_id[iid][0].gid, "nonlinear")

    def exact(self, iid) -> list:
        """Sequence rules of the node's type with an exact per-unit DP, found at the finest step any node needs."""
        if self._exact is None:
            from ..optim.problem import Problem
            from ..optim.unitdp import find_rules
            from .compiler import Compiler
            dt = min(self.need.values()) if self.need else self.master
            if dt < self.master - 1e-9:
                sub, _ = _subflat(self.flat, list(self.by_id), dt)
                prog = Compiler(sub, self.model.diagnostics, physical=False).compile()
            else:
                prog = self.prog
            names = {}
            for r in find_rules(prog, Problem(prog).space):
                t = prog.gmap[r.gid].ir.type.short
                kind = {"gap": "one event per window", "run": "longest run", "reset": "counter reset"}[r.kind]
                names.setdefault(t, []).append(f"{t}.{r.dec}: {kind} (exact DP)")
            self._exact = names
        return self._exact.get(self.by_id[iid][0].ir.type.short, [])

    # ------------------------------------------------------------------ pieces
    def pieces(self, members: set, out: set) -> list:
        """Connected pieces of `members` without `out`."""
        seen, res = set(), []
        for i in sorted(members):
            if i in out or i in seen:
                continue
            comp, stack = [i], [i]
            seen.add(i)
            while stack:
                a = stack.pop()
                for b in self.nbrs[a]:
                    if b in members and b not in out and b not in seen:
                        seen.add(b)
                        comp.append(b)
                        stack.append(b)
            res.append(sorted(comp))
        return res

    def shared_exchanges(self, members: set) -> set:
        shared = {i for i in members if self.is_exchange(i)}
        for _ in range(50):
            comp = {i: k for k, p in enumerate(self.pieces(members, shared)) for i in p}
            keep = {c for c in shared if len({comp[n] for n in self.nbrs[c] if n in comp}) >= 2}
            if keep == shared:
                break
            shared = keep
        return shared


def _name(G: AssetGraph, members: list, used: set) -> str:
    pre: dict = {}
    for i in members:
        inst = G.flat.instances[i]
        if getattr(inst, "line", None) is not None:
            continue
        p = inst.path.split("_")[0]
        pre[p] = pre.get(p, 0) + 1
    base = "+".join(sorted(p for p, n in pre.items() if n >= 3)) or (max(sorted(pre), key=lambda p: pre[p]) if pre else "block")
    # the unit type that carries most of its decisions names what it is
    types: dict = {}
    for i in members:
        t = G.by_id[i][0].ir.type.short
        types[t] = types.get(t, 0) + G.decisions(i)
    top = max(sorted(types), key=lambda t: types[t]) if types else ""
    name = f"{base}.{top}" if top and len(members) > 1 else (G.flat.instances[members[0]].path if len(members) == 1 else base)
    k, out = 2, name
    while out in used:
        out, k = f"{name}{k}", k + 1
    used.add(out)
    return out


def _describe_block(G: AssetGraph, b: Block) -> None:
    mem = set(b.members)
    iface: dict = {}
    for e in G.flat.edges:
        sp, dp = G.flat.ports[e.src], G.flat.ports[e.dst]
        a, c = sp.inst in mem, dp.inst in mem
        if a == c:
            continue
        other, port, d = (dp, dp.name, "out") if a else (sp, sp.name, "in")
        key = (d, G.flat.instances[other.inst].path, port, "/".join(r.split(".")[-1] for r in e.resources))
        iface[key] = iface.get(key, 0) + 1
    b.interface = [(*k, n) for k, n in sorted(iface.items())]
    b.channels = sum(G.decisions(i) for i in b.members if i in G.by_id)
    b.cls = max((G.cls(i) for i in b.members if G.decisions(i)), key=lambda c: RANK.get(c, 6), default="none")
    ex = []
    for i in b.members:
        for x in G.exact(i):
            if x not in ex:
                ex.append(x)
    b.exact = ex


def _split(G: AssetGraph, b: Block, used_names: set, min_units: int = 1) -> None:
    mem = set(b.members)
    _describe_block(G, b)
    if len(mem) < 2:
        b.why = ["one unit"]
        return

    def children_of(pieces, step, rule):
        out, own = [], set()
        for p in pieces:
            if len(p) >= min_units and any(G.decisions(i) for i in p):
                out.append(Block(_name(G, p, used_names), p, step, b.step, b.depth + 1))
            else:
                own |= set(p)
        return out, own

    # 1. exchanges several parts share
    shared = G.shared_exchanges(mem)
    if shared:
        kids, own = children_of(G.pieces(mem, shared), b.step, "exchanges")
        if len(kids) >= 2:
            b.rule, b.own, b.children = "exchanges", sorted(shared | own), kids
            b.why = [f"{G.flat.instances[i].path}: takes flows from {len(G.senders[i])} units, keeps no stock, shared by "
                     f"{len({k for k, kid in enumerate(kids) for n in G.nbrs[i] if n in set(kid.members)})} parts"
                     for i in sorted(shared)]
    # 2. time scales: nodes needing a finer step seed pieces at that step
    if not b.rule:
        fast = {i for i in mem if G.need[i] < b.step - 1e-9}
        if fast:
            found = _grow(G, mem, fast, b.step)
            kids = []
            for f in found:
                nm = f["name"] if f["name"] not in used_names else _name(G, f["members"], used_names)
                used_names.add(nm)
                kid = Block(nm, f["members"], f["dt"], b.step, b.depth + 1)
                kid.why = f["why"]
                kid.used, kid.note = True, f"used: its own step, {f['dt'] * 60:g} min inside {b.step * 60:g} min"
                kids.append(kid)
            rest = mem - {i for f in found for i in f["members"]}
            more, own = children_of(G.pieces(rest, _stock_cuts(G, rest, b.step)), b.step, "stocks") \
                if rest else ([], set())
            b.rule = "time scale"
            b.children = kids + (more if len(more) >= 2 else [])
            b.own = sorted(rest if len(more) < 2 else own | _stock_cuts(G, rest, b.step))
            b.why = [f"{G.flat.instances[i].path}: needs {G.need[i] * 60:g}-minute steps ({G.need_why.get(i, '')} spans "
                     f"no whole number of {b.step * 60:g}-minute steps)" for i in sorted(fast)]
    # 3. stocks that hold many of the part's steps of their flow
    if not b.rule:
        cuts = _stock_cuts(G, mem, b.step)
        if cuts:
            kids, own = children_of(G.pieces(mem, cuts), b.step, "stocks")
            if len(kids) >= 2:
                b.rule, b.own, b.children = "stocks", sorted(cuts | own), kids
                b.why = [f"{G.flat.instances[i].path}: its {G.hours(i)[1]} holds {G.hours(i)[0]:,.0f} h of its flow "
                         f"(>= {G.buffer_steps:g} steps of {b.step * 60:g} min)" for i in sorted(cuts)]
    if not b.rule:
        small = sorted((G.hours(i)[0], G.flat.instances[i].path) for i in mem if G.hours(i) is not None)
        b.why = ["a cell: coupled within its step - direct flows" +
                 (", stocks of " + ", ".join(f"{p} {h:.1f} h" for h, p in small[:4]) if small else "")]
        return
    for kid in b.children:
        if not kid.used and kid.step >= b.step - 1e-9:
            kid.note = "found, not used: same step as the part around it"
        _split(G, kid, used_names, min_units)


def _stock_cuts(G: AssetGraph, mem: set, step: float) -> set:
    out = set()
    for i in mem:
        h = G.hours(i)
        if h is not None and h[0] >= G.buffer_steps * step:
            out.add(i)
    return out


def _grow(G: AssetGraph, mem: set, fast: set, step: float) -> list:
    """Pieces at a finer step inside `mem`: seeds and what they take in (hierarchy.detect_plants' rule, restricted to
    the part and measured in its steps)."""
    flat = G.flat
    inside = set(fast)
    parent = {i: i for i in mem}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    joined = {}
    changed = True
    while changed:
        changed = False
        for e in flat.edges:
            sp, dp = flat.ports[e.src], flat.ports[e.dst]
            u, v = sp.inst, dp.inst
            if u not in mem or v not in mem:
                continue
            if u in inside and v in inside:
                union(u, v)
            elif v in inside:
                inside.add(u)
                union(u, v)
                joined[u] = f"sends into {flat.instances[v].path}.{dp.name}"
                changed = True
            elif u in inside:
                h = _port_hours(G, v, dp.name)
                if h is not None and h[0] < G.buffer_steps * step:
                    inside.add(v)
                    union(u, v)
                    joined[v] = f"its {h[1]} holds {h[0]:.1f} h of its flow"
                    changed = True
    comps: dict = {}
    for i in sorted(inside):
        comps.setdefault(find(i), []).append(i)
    out = []
    for members in sorted(comps.values(), key=lambda m: m[0]):
        for s in [step / n for n in range(1, 721) if step / n >= 1.0 / 60 - 1e-9]:
            if all(all(_steps_ok(f(flat.instances[i].params, s)) for _, f in G._rules_of.get(G.by_id[i][0].gid, []))
                   for i in members):
                break
        why = [f"{flat.instances[i].path}: needs {G.need[i] * 60:g}-minute steps" for i in members if i in fast] + \
              [f"{flat.instances[i].path}: joined - {w}" for i, w in sorted(joined.items()) if i in set(members)]
        seeds = sorted((i for i in members if i in fast), key=lambda i: flat.instances[i].path)
        pre = flat.instances[seeds[0]].path.split("_")[0]
        base = pre if all(flat.instances[i].path.split("_")[0] == pre for i in seeds) else "block"
        out.append({"members": members, "dt": s, "why": why,
                    "name": f"{base}.{flat.instances[seeds[0]].type.name.split('.')[-1]}"})
    return out


def _port_hours(G: AssetGraph, iid, port) -> Optional[tuple]:
    g, li = G.by_id[iid]
    P = G.flat.instances[iid].params
    best = None
    for s, ports in G.stocks(g).items():
        if not any(p.split(":")[0] == port for p in ports):
            continue
        si = g.ir.states[s]
        lo = static_value(si.lo, P, G.master) if si.lo is not None else None
        hi = static_value(si.hi, P, G.master) if si.hi is not None else None
        rng = (hi - lo) if (hi is not None and lo is not None) else float("inf")
        thr = sum(G._in_flow(g, li, p) for p in ports)
        h = rng / thr if thr > 1e-9 else float("inf")
        if best is None or h < best[0]:
            best = (h, s)
    return best


def find_structure(model, buffer_steps: float = 24.0) -> Block:
    """The block tree of a twin (see the module's docstring)."""
    G = AssetGraph(model, buffer_steps=buffer_steps)
    members = sorted(i for i in G.by_id if not G.flat.instances[i].is_junction)
    root = Block(G.flat.env_type.name.split(".")[-1] if getattr(G.flat, "env_type", None) is not None else "twin",
                 members, G.master, G.master, 0)
    _split(G, root, {root.name})
    root.note = "the whole twin"
    root._graph = G
    return root


def used_blocks(root: Block) -> list:
    return [b for b in root.walk() if b.used]


def render(root: Block, flat, detail: bool = True) -> str:
    """The tree in words."""
    L = []

    def line(b: Block):
        pad = "  " * b.depth
        tags = [f"{len(b.members)} units", f"step {b.step * 60:g} min", f"{b.channels} decisions per step",
                f"{len(b.interface)} flows at its fence", f"class {b.cls}"]
        L.append(f"{pad}{'*' if b.used else '-'} {b.name}: " + ", ".join(tags))
        if b.note and (detail or b.used or b.depth == 0):
            L.append(f"{pad}    {b.note}")
        if b.used and detail:
            ins = [f"{p}.{q} ({r})" for d, p, q, r, n in b.interface if d == "in"]
            outs = [f"{p}.{q} ({r}, {n} edges)" for d, p, q, r, n in b.interface if d == "out"]
            L.append(f"{pad}    its function: " + (f"in {', '.join(ins)}; " if ins else "") + f"out {', '.join(outs)}")
        if b.exact and (detail or b.used):
            L.append(f"{pad}    exact inside: " + "; ".join(b.exact))
        if b.rule:
            L.append(f"{pad}    cut by {b.rule}:" + ("" if detail else f" {len(b.why)} points"))
            if detail:
                for w in b.why[:8]:
                    L.append(f"{pad}      {w}")
                if len(b.why) > 8:
                    L.append(f"{pad}      ... {len(b.why) - 8} more")
        elif b.why and len(b.members) > 1:
            L.append(f"{pad}    {b.why[0]}")
        singles = [c for c in b.children if len(c.members) == 1 and not c.used and not c.children]
        for c in b.children:
            if c not in singles:
                line(c)
        if singles:
            L.append(f"{pad}  - {len(singles)} single units between the cut points: "
                     + ", ".join(c.name for c in singles[:12]) + (" ..." if len(singles) > 12 else ""))
    line(root)
    blocks = list(root.walk())[1:]
    used = [b for b in blocks if b.used]
    L.append(f"\n{len(blocks)} blocks found, {len(used)} used as input-output functions"
             + (f" ({', '.join(b.name for b in used)})" if used else "")
             + ". * = used; blocks at the same step as the part around them are found and reported: coordinating them"
               " by prices did not pay yet (bench/results/corporate_regions*.json).")
    return "\n".join(L)


__all__ = ["Block", "AssetGraph", "find_structure", "render", "used_blocks"]
