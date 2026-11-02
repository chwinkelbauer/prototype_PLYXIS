"""Structure recognition: which branches of a twin are problems with exact, fast methods.

The planner should not treat every part of a site alike. Many branches of a twin are, mathematically, problem
classes for which reliable deterministic methods exist - a linear flow network, a storage scheduled against
a price, an asset whose discrete states form a chain over time. This module proves which ones they are, from
the twin's own equations (the planning IR of the PLX compiler), with rules in the spirit of disciplined
convex programming (Grant & Boyd):

    curvature     every expression is classified as constant / affine / convex / concave / nonlinear in the
                  decisions, flows and states, with its sign (from the actual parameter values, decision
                  bounds and initial states) and two flags: piecewise affine (min, max, pos, abs, clamp, pwl of
                  affine expressions - LP-representable piece by piece) and discrete (depends on int / bool
                  decisions)
    propagation   flows are affine if their sources are; a state is affine if its update is affine in itself
                  and the rest (linear dynamics); blended qualities (flow-weighted attributes such as an ore
                  grade) are nonlinear - the pooling problem - unless they are frozen (successive LP)
    classes       per asset type and per connected branch of the twin:
                      affine, continuous          -> LP: exact global optimum (HiGHS dual simplex), exact prices
                      affine with int / bool      -> MILP on the branch (small) or the section DP
                      piecewise affine            -> sequential LP inside the current linear piece
                      affine with frozen qualities-> successive LP (pooling / blending)
                      convex with discrete choices, qualities fixed and prices at their sign
                                                  -> sequential LP on its tangent planes (or Newton-type methods)
                      nonlinear                   -> the cascade (relaxation + discrete search + polish)

`python -m plyxis.optim.structure scenarios/corvus` prints the report for a scenario.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from ..lang.behavior import C, Op, Ref

ORDER = {"const": 0, "affine": 1, "convex": 2, "concave": 2, "nonlinear": 3}


@dataclass(frozen=True)
class Form:
    curv: str = "const"          # const | affine | convex | concave | nonlinear
    sign: str = "?"              # + | - | ?   (of the value)
    pwa: bool = True             # piecewise affine in the variables
    disc: bool = False           # depends on integer / boolean decisions
    why: str = ""                # first reason for a non-affine form

    def scaled(self, s: str) -> "Form":
        if self.curv == "const":
            return replace(self, sign=_mul_sign(self.sign, s))
        if s == "+":
            return self
        if s == "-":
            return self.neg()
        if self.curv in ("convex", "concave"):
            return replace(self, curv="nonlinear", sign="?", why=self.why or "convex part times a coefficient of unknown sign")
        return replace(self, sign="?")

    def neg(self) -> "Form":
        c = {"convex": "concave", "concave": "convex"}.get(self.curv, self.curv)
        return replace(self, curv=c, sign={"+": "-", "-": "+"}.get(self.sign, "?"))


CONST_POS = Form("const", "+")
CONST_ANY = Form("const", "?")


def _mul_sign(a: str, b: str) -> str:
    if "?" in (a, b):
        return "?"
    return "+" if a == b else "-"


def _sign_of(values) -> str:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return "?"
    if np.all(v >= 0):
        return "+"
    if np.all(v <= 0):
        return "-"
    return "?"


def _join(a: Form, b: Form, op: str) -> Form:
    """Sum of two forms."""
    disc, pwa = a.disc or b.disc, a.pwa and b.pwa
    sign = a.sign if a.sign == b.sign else "?"
    why = a.why or b.why
    if a.curv == "const":
        return replace(b, sign=sign, disc=disc, pwa=pwa, why=why)
    if b.curv == "const":
        return replace(a, sign=sign, disc=disc, pwa=pwa, why=why)
    curv = {("affine", "affine"): "affine", ("affine", "convex"): "convex", ("convex", "affine"): "convex",
            ("convex", "convex"): "convex", ("affine", "concave"): "concave", ("concave", "affine"): "concave",
            ("concave", "concave"): "concave"}.get((a.curv, b.curv), "nonlinear")
    if curv == "nonlinear" and not why:
        why = f"{op} of a convex and a concave part"
    return Form(curv, sign, pwa, disc, why)


class Analyser:
    """Curvature of every node of every asset type of a planning program."""

    def __init__(self, program, frozen_quality: bool = False, fixed_discrete: bool = False):
        self.prog, self.comp = program, program.comp
        self.frozen = frozen_quality
        self.fixed = fixed_discrete
        # quality states: states that feed a blend-quality output (an average grade carried with the flow)
        self.quality_states: dict = {}
        if frozen_quality:
            from ..lang.behavior import walk_refs
            for g in self.comp.groups:
                if g.is_env:
                    continue
                qs, seen = set(), set()
                todo = [k for k in g.ir.nodes if k.startswith("out:") and k.count(":") >= 3]
                while todo:
                    k = todo.pop()
                    if k in seen or k not in g.ir.nodes:
                        continue
                    seen.add(k)
                    for kind, name in walk_refs(g.ir.nodes[k]):
                        if kind == "state":
                            qs.add(name)
                        elif kind == "node":
                            todo.append(name)
                self.quality_states[g.gid] = qs
        self.forms: dict = {}          # (gid, node key) -> Form
        self._busy: set = set()
        self._assume: dict = {}        # state update -> curvature assumed for the state in its own update
        self.assumed: set = set()      # prices whose current sign the verdict rests on (frozen analysis only)
        self.dec_form: dict = {}
        self.dec_range: dict = {}      # (gid, decision) -> (lowest, highest) value over the instances
        from .space import DecisionSpace
        bnd = DecisionSpace.build(program).bounds()
        for g in self.comp.groups:
            if g.is_env:
                continue
            for dn, di in g.ir.decisions.items():
                lo = bnd.get((g.gid, dn), (None,))[0]
                sgn = _sign_of(lo.numpy() if hasattr(lo, "numpy") else lo) if lo is not None else "?"
                self.dec_form[(g.gid, dn)] = Form("affine", "+" if sgn == "+" else "?", True,
                                                  di.domain in ("int", "bool"))
                if lo is not None:
                    hi = bnd[(g.gid, dn)][1]
                    self.dec_range[(g.gid, dn)] = (float(np.min(lo.numpy() if hasattr(lo, "numpy") else lo)),
                                                   float(np.max(hi.numpy() if hasattr(hi, "numpy") else hi)))

    # ------------------------------------------------------------------ references
    def _param(self, g, name) -> Form:
        if g.is_env:
            v = self.prog.E0.get(name)
            if name in self.prog.flat.env_inputs:            # exogenous series (prices): sign may change
                return Form("const", _sign_of(v) if v is not None else "?")
            return Form("const", _sign_of(v) if v is not None else "?")
        P = self.prog.P0.get(g.gid, {}).get(name)
        return Form("const", _sign_of(P.numpy() if hasattr(P, "numpy") else P) if P is not None else "?")

    def _env(self, name) -> Form:
        v = self.prog.E0.get(name)
        if v is None:
            return CONST_ANY
        if name in self.prog.flat.env_inputs and name.endswith("price"):
            if self.frozen and _sign_of(v.numpy() if hasattr(v, "numpy") else v) == "+":
                self.assumed.add(name)                        # prices are data of a solve: the verdict states this sign
                return Form("const", "+")
            return CONST_ANY                                  # prices can turn negative (gas, power)
        return Form("const", _sign_of(v.numpy() if hasattr(v, "numpy") else v))

    def _state(self, g, name) -> Form:
        if self.frozen and name in self.quality_states.get(g.gid, ()):
            return Form("const", "+")                         # a blend quality, frozen (successive LP)
        st = g.ir.states.get(name)
        if st is None or not st.has_update:                   # exogenous / constant state (e.g. up): known
            S = self.prog.S0.get(g.gid, {}).get(name)
            return Form("const", _sign_of(S.numpy()) if S is not None else "?")
        key = (g.gid, f"next:{name}")
        if key in self._busy:                                 # induction: assume affine, verified by the update
            return self._assume.get(key, Form("affine", "?"))
        f = self.node(g, f"next:{name}")
        if f.curv in ("convex", "concave") and key not in self._assume:
            # induction step: the update must reproduce this curvature from a state that already has it
            self._assume[key] = replace(f, why="")
            saved, self.forms = self.forms, {}
            self._busy.add(key)
            again = self.expr(g, g.ir.nodes[f"next:{name}"])
            self._busy.discard(key)
            self.forms = saved
            if again.curv != f.curv:
                f = replace(f, curv="nonlinear", why=f.why or f"state '{name}': {f.curv} update does not stay {f.curv}")
                self.forms[key] = f
        lo = st.lo
        sign = "+" if (lo is not None and isinstance(lo, C) and lo.value >= 0) else f.sign
        return replace(f, sign=sign)

    def _inflow(self, g, key) -> Form:
        if key.count(":") >= 2:                               # flow-weighted attribute (quality of a blend)
            if self.frozen:
                return Form("const", "+")
            return Form("nonlinear", "+", False, False, f"blended quality '{key}' (pooling)")
        return Form("affine", "+")

    def _agg(self, name) -> Form:
        env = self.comp.gmap.get("env")
        sel, how = env.ir.aggs[name]
        f = CONST_POS
        for gid, (kind, sig), _li in sel:
            g = self.comp.gmap[gid]
            h = self.node(g, sig) if kind == "node" else self.ref(g, Ref(kind, sig))
            f = _join(f, h, "sum")
        if how not in ("sum", "mean"):
            f = replace(f, curv="nonlinear" if f.curv != "const" else "const", pwa=f.pwa and how in ("min", "max"),
                        why=f.why or f"aggregation '{how}'")
        return f

    def ref(self, g, r: Ref) -> Form:
        k = r.kind
        if k == "param":
            return self._param(g, r.name)
        if k == "env":
            return self._env(r.name)
        if k == "b":
            return CONST_POS
        if k == "dec":
            f = self.dec_form.get((g.gid, r.name), Form("affine", "?"))
            if self.fixed and f.disc:                         # discrete choices fixed (continuous layer)
                return Form("const", f.sign)
            return f
        if k == "state":
            return self._state(g, r.name)
        if k == "in":
            return self._inflow(g, r.name)
        if k == "node":
            return self.node(g, r.name)
        if k == "agg":
            return self._agg(r.name)
        return Form("nonlinear", "?", False, False, f"reference kind {k}")

    def node(self, g, key: str) -> Form:
        k = (g.gid, key)
        if k in self.forms:
            return self.forms[k]
        e = g.ir.nodes.get(key)
        if e is None:
            return Form("affine", "?")
        self._busy.add(k)
        f = self.expr(g, e)
        self._busy.discard(k)
        self.forms[k] = f
        return f

    # ------------------------------------------------------------------ expressions
    def _value(self, g, e):
        """Value of a constant expression of parameters (per instance), None if it is more than that."""
        if isinstance(e, C):
            return np.asarray(float(e.value))
        if isinstance(e, Ref):
            P = self.prog.P0.get(g.gid, {}).get(e.name) if e.kind == "param" else None
            return None if P is None else np.asarray(P.numpy() if hasattr(P, "numpy") else P, dtype=np.float64)
        if e.op in ("add", "sub", "mul", "div") and len(e.args) == 2:
            x, y = self._value(g, e.args[0]), self._value(g, e.args[1])
            if x is None or y is None:
                return None
            return {"add": x + y, "sub": x - y, "mul": x * y}.get(e.op) if e.op != "div" else x / np.where(y == 0, np.nan, y)
        return None

    def _within_one_piece(self, g, e, xs) -> bool:
        """A piecewise-linear curve of a decision whose whole range lies between two breakpoints is affine there."""
        if not (isinstance(e, Ref) and e.kind == "dec") or (g.gid, e.name) not in self.dec_range or len(xs) < 2:
            return False
        lo, hi = self.dec_range[(g.gid, e.name)]
        return not any(lo + 1e-6 < b < hi - 1e-6 for b in xs) and xs[0] <= lo + 1e-6 and hi - 1e-6 <= xs[-1]

    def expr(self, g, e) -> Form:
        if isinstance(e, C):
            return Form("const", "+" if e.value >= 0 else "-")
        if isinstance(e, Ref):
            return self.ref(g, e)
        op, a = e.op, e.args
        if op in ("add", "sum", "mean"):
            f = self.expr(g, a[0])
            for x in a[1:]:
                f = _join(f, self.expr(g, x), op)
            return f
        if op == "sub":
            f = _join(self.expr(g, a[0]), self.expr(g, a[1]).neg(), op)
            if f.curv == "const":
                v = self._value(g, e)
                if v is not None:
                    return replace(f, sign=_sign_of(v))
            return f
        if op == "neg":
            return self.expr(g, a[0]).neg()
        if op in ("mul", "and"):
            x, y = self.expr(g, a[0]), self.expr(g, a[1])
            if x.curv == "const":
                return replace(y.scaled(x.sign), disc=y.disc)
            if y.curv == "const":
                return replace(x.scaled(y.sign), disc=x.disc)
            return Form("nonlinear", _mul_sign(x.sign, y.sign), False, x.disc or y.disc,
                        x.why or y.why or "product of two variables (bilinear)")
        if op in ("div", "safediv"):
            x, y = self.expr(g, a[0]), self.expr(g, a[1])
            if y.curv == "const":
                return x.scaled(y.sign)
            return Form("nonlinear", "?", False, x.disc or y.disc, x.why or y.why or "division by a variable")
        if op == "pow":
            x = self.expr(g, a[0])
            if x.curv == "const":
                return x
            if not isinstance(a[1], C):
                return Form("nonlinear", "?", False, x.disc, "power with a variable exponent")
            k = float(a[1].value)
            if k == 1:
                return x
            if x.curv == "affine" and (k >= 1 and (x.sign == "+" or k % 2 == 0)):
                return Form("convex", "+", False, x.disc, x.why or f"power {k:g} (convex, smooth)")
            if x.curv in ("affine", "concave") and 0 < k < 1 and x.sign == "+":
                return Form("concave", "+", False, x.disc, x.why or f"power {k:g} (concave, smooth)")
            return Form("nonlinear", "?", False, x.disc, x.why or f"power {k:g}")
        if op in ("min", "max", "smin", "smax"):
            fs = [self.expr(g, x) for x in (a if op in ("min", "max") else a[:2])]
            if all(f.curv == "const" for f in fs):
                return Form("const", "?")
            disc = any(f.disc for f in fs)
            pwa = all(f.pwa for f in fs) and op in ("min", "max")
            want = "concave" if op in ("min", "smin") else "convex"
            if all(f.curv in ("const", "affine", want) for f in fs):
                return Form(want, "?", pwa, disc, next((f.why for f in fs if f.why), "") or f"{op} (piecewise affine)")
            return Form("nonlinear", "?", pwa, disc, next((f.why for f in fs if f.why), f"{op} of mixed curvature"))
        if op in ("pos", "abs", "negpart"):
            x = self.expr(g, a[0])
            if x.curv == "const":
                return replace(x, sign="+")
            if (op in ("pos", "abs") and x.curv in ("affine", "convex")) or (op == "negpart" and x.curv in ("affine", "concave")):
                return Form("convex", "+", x.pwa, x.disc, x.why or f"{op} (piecewise affine)")
            return Form("nonlinear", "+", x.pwa, x.disc, x.why or f"{op} of a non-convex part")
        if op == "clamp":
            fs = [self.expr(g, x) for x in a]
            if all(f.curv == "const" for f in fs):
                return Form("const", "?")
            return Form("nonlinear", "?", all(f.pwa for f in fs), any(f.disc for f in fs),
                        next((f.why for f in fs if f.why), "clamp (piecewise affine, not convex)"))
        if op == "pwl":
            x = self.expr(g, a[0])
            ys = a[1:]
            if x.curv == "const":
                return Form("const", "?")
            xs = list(e.extra.get("xs", [])) if isinstance(e.extra, dict) else []
            if self._within_one_piece(g, a[0], xs) and all(isinstance(y, C) for y in ys):
                return replace(x, sign=_sign_of([y.value for y in ys]))
            if all(isinstance(y, C) for y in ys) and len(xs) == len(ys) and x.curv == "affine":
                s = np.diff([y.value for y in ys]) / np.maximum(np.diff(xs), 1e-12)
                if np.all(np.diff(s) >= -1e-12):
                    return Form("convex", "?", x.pwa, x.disc, "pwl with increasing slopes")
                if np.all(np.diff(s) <= 1e-12):
                    return Form("concave", "?", x.pwa, x.disc, "pwl with decreasing slopes")
            return Form("nonlinear", "?", x.pwa, x.disc, x.why or "piecewise-linear curve")
        if op in ("sqrt", "log", "log10"):
            x = self.expr(g, a[0])
            if x.curv == "const":
                return x
            if x.curv in ("affine", "concave"):
                return Form("concave", "?", False, x.disc, x.why or f"{op} (concave, smooth)")
            return Form("nonlinear", "?", False, x.disc, x.why or op)
        if op == "exp":
            x = self.expr(g, a[0])
            if x.curv == "const":
                return x
            if x.curv in ("affine", "convex"):
                return Form("convex", "+", False, x.disc, x.why or "exp (convex, smooth)")
            return Form("nonlinear", "+", False, x.disc, x.why or "exp")
        if op == "delay":
            return self.expr(g, a[0])
        fs = [self.expr(g, x) for x in a]
        if all(f.curv == "const" for f in fs):
            return Form("const", "?")
        return Form("nonlinear", "?", False, any(f.disc for f in fs), next((f.why for f in fs if f.why), f"operator {op}"))


# ----------------------------------------------------------------------------- classification
METHOD = {"lp": "LP - exact global optimum and prices (HiGHS dual simplex)",
          "milp": "affine with integer decisions - exact MILP on small branches, section DP on large ones",
          "slp": "piecewise affine once discrete choices and blend qualities are fixed - sequential LP",
          "slp_quality": "affine once discrete choices and blend qualities are fixed - successive LP (blending)",
          "convex": "convex with discrete choices and blend qualities fixed - sequential LP on its tangent planes",
          "nonlinear": "nonlinear - cascade (relaxation, discrete search, polish)",
          "none": "no decisions"}


@dataclass
class TypeClass:
    type: str
    gid: str
    instances: int
    decisions: int
    discrete: int
    cls: str
    reasons: list


def _relevant(ir) -> list:
    return [k for k in ir.nodes if k.startswith(("term:cost:", "term:revenue:", "term:emission:", "con:", "next:", "out:"))]


def _need(ir, k: str) -> str:
    """Curvature a node may have for the branch to stay a convex (here: maximisation) problem."""
    if k.startswith("term:revenue:"):
        return "concave"
    if k.startswith(("term:cost:", "term:emission:")):
        return "convex"
    if k.startswith("con:"):
        ci = ir.cons.get(k)
        return "affine" if ci is not None and ci.kind == "eq" else "convex"
    if k.startswith("next:"):
        return "state"                                # any verified curvature: each use of the state is checked itself
    return "affine"                                   # flows (out:): they leave the asset


def classify_types(program) -> dict:
    """Class of every asset type: strict (blends nonlinear) and with frozen blend qualities."""
    from .space import DecisionSpace
    sp = DecisionSpace.build(program)
    sp_dec: dict = {}
    for b in sp.blocks:
        d = sp_dec.setdefault(b.gid, [0, 0])
        d[0] += b.n * b.H
        if b.domain in ("int", "bool"):
            d[1] += b.n * b.H
    if sp.route is not None:                          # routing decisions count with the routed asset type
        for rp in program.routed:
            g = program.comp.gmap[program.inst_group[rp.inst][0]] if hasattr(program, "inst_group") else None
            if g is not None:
                d = sp_dec.setdefault(g.gid, [0, 0])
                d[0] += len(rp.ridx) * sp.route.H
    strict, frozen = Analyser(program), Analyser(program, frozen_quality=True, fixed_discrete=True)
    rank = {"affine": 0, "convex": 1, "nonlinear": 2}
    out = {}
    for g in program.comp.groups:
        if g.is_env:
            continue
        res = []
        for A in (strict, frozen):
            worst, pwa, reasons = "affine", True, []
            frozen_q = A.quality_states.get(g.gid, set()) if A.frozen else set()
            for k in _relevant(g.ir):
                if A.frozen and ((k.startswith("next:") and k[5:] in frozen_q) or (k.startswith("out:") and k.count(":") >= 3)):
                    continue                              # frozen qualities and their updates are data here
                f = A.node(g, k)
                need = _need(g.ir, k)
                lvl = "affine" if f.curv in ("const", "affine") else (
                    "convex" if f.curv == need or (need == "state" and f.curv in ("convex", "concave")) else "nonlinear")
                pwa = pwa and f.pwa
                if rank[lvl] > rank[worst]:
                    worst = lvl
                if lvl != "affine" and len(reasons) < 4:
                    reasons.append(f"{k}: {f.why or f.curv}")
            res.append((worst, pwa, reasons))
        (w_s, pwa_s, why_s), (w_f, pwa_f, why_f) = res
        n_dec, n_disc = sp_dec.get(g.gid, [0, 0])
        if n_dec == 0:
            cls = "none"
        elif w_s == "affine":
            cls = "milp" if n_disc else "lp"
        elif w_f == "affine":
            cls = "slp_quality"
        elif pwa_f:
            cls = "slp"
        elif w_s == "convex" or w_f == "convex":
            cls = "convex"
        else:
            cls = "nonlinear"
        if cls in ("slp", "slp_quality", "convex", "nonlinear") and why_f:
            why_s = why_s[:2] + [f"given fixed discrete choices and qualities - {why_f[0]}"]
        if cls == "convex" and w_s != "convex" and frozen.assumed:
            why_s = why_s + [f"at the current sign of {', '.join(sorted(frozen.assumed))}"]
        out[g.gid] = TypeClass(g.ir.type.short, g.gid, len(g.inst_ids), n_dec, n_disc, cls, why_s)
    return out


@dataclass
class Branch:
    instances: list               # instance paths
    cls: str
    decisions: int
    discrete: int


def branches(program, types: Optional[dict] = None) -> list:
    """Connected branches of the twin (flow edges) whose assets share a solvable class; assets with a
    nonlinear class or without decisions cut the branches, hubs couple them through prices."""
    types = types or classify_types(program)
    flat = program.flat
    cls_of = {}
    for g in program.comp.groups:
        if g.is_env:
            continue
        for iid in g.inst_ids:
            cls_of[iid] = types[g.gid].cls
    parent = {i: i for i in cls_of}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    solvable = {"lp", "milp", "slp", "slp_quality", "convex"}
    for e in flat.edges:
        a, b = flat.ports[e.src].inst, flat.ports[e.dst].inst
        if cls_of.get(a) in solvable and cls_of.get(a) == cls_of.get(b):
            parent[root(a)] = root(b)
    groups: dict = {}
    for i, c in cls_of.items():
        if c in solvable:
            groups.setdefault(root(i), []).append(i)
    per_inst = {}
    for g in program.comp.groups:
        if g.is_env:
            continue
        t = types[g.gid]
        for iid in g.inst_ids:
            per_inst[iid] = (t.decisions / max(1, t.instances), t.discrete / max(1, t.instances))
    out = []
    for members in groups.values():
        out.append(Branch([flat.instances[i].path for i in members], cls_of[members[0]],
                          int(round(sum(per_inst[i][0] for i in members))),
                          int(round(sum(per_inst[i][1] for i in members)))))
    return sorted(out, key=lambda b: -b.decisions)


def report(program) -> dict:
    types = classify_types(program)
    br = branches(program, types)
    total = sum(t.decisions for t in types.values())
    by_cls: dict = {}
    for t in types.values():
        by_cls[t.cls] = by_cls.get(t.cls, 0) + t.decisions
    return {"decisions": total,
            "share_by_class": {k: round(v / max(1, total), 4) for k, v in sorted(by_cls.items(), key=lambda kv: -kv[1])},
            "types": [vars(t) for t in sorted(types.values(), key=lambda t: -t.decisions)],
            "branches": [{"class": b.cls, "assets": len(b.instances), "decisions": b.decisions,
                          "example": b.instances[:3]} for b in br[:30]],
            "n_branches": len(br)}


def main() -> None:
    from ..lang import compile_model
    sd = sys.argv[1] if len(sys.argv) > 1 else "scenarios/corvus"
    sc = json.load(open(os.path.join(sd, "scenario.json")))
    m = compile_model([os.path.join(sd, f) for f in sc["twin"]["files"]], env=sc["twin"]["env"])
    r = report(m.program)
    print(f"{sd}: {r['decisions']:,} decisions")
    for c, s in r["share_by_class"].items():
        print(f"  {100 * s:5.1f} %  {c:12s} {METHOD[c]}")
    for t in r["types"]:
        print(f"  {t['type']:20s} x{t['instances']:4d}  decisions {t['decisions']:5d}  -> {t['cls']:11s}"
              + (f"   ({'; '.join(t['reasons'][:2])})" if t["reasons"] else ""))
    print(f"  {r['n_branches']} solvable branches; largest:")
    for b in r["branches"][:8]:
        print(f"    {b['class']:11s} {b['assets']:4d} assets {b['decisions']:5d} decisions  e.g. {', '.join(b['example'])}")


if __name__ == "__main__":
    main()
