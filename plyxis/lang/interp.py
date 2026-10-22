"""Reference interpreter for PLX (slow, simple, independent of the code generator).

Evaluates the elaborated model instance by instance and step by step with lazy
memoised evaluation. Used by the test-suite to prove that the time-vectorised
compiled code computes exactly the same trajectories.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from .behavior import C, IR, Op, Ref, chan_key


class Interpreter:
    def __init__(self, program, physical: bool = False):
        self.p = program
        self.flat = program.flat
        self.physical = physical

    def run(self, D: dict, R: Optional[torch.Tensor], T: int, dt: float, t0: float = 0.0,
            E: Optional[dict] = None, S0: Optional[dict] = None, P: Optional[dict] = None) -> dict:
        p = self.p
        P = P if P is not None else p.P0
        S0 = S0 if S0 is not None else p.S0
        Et = p.env_tensors(E, T)
        R = R if R is not None else p.default_routing(1, T)
        if R.shape[-1] == 1:
            R = R.expand(R.shape[0], R.shape[1], T)
        self.P, self.D, self.R, self.Et, self.dt, self.t0 = P, D, R, Et, dt, t0
        groups = p.groups
        # state init
        state: dict = {}
        for g in groups:
            for s in g.ir.states:
                x0 = S0[g.gid][s]
                for li in range(g.n):
                    state[(g.gid, li, s)] = float(x0.reshape(x0.shape[0], -1)[0, li if x0.numel() > 1 else 0])
        hist: dict = {}
        traj: dict = {}
        for t in range(T):
            self.t = t
            self.cache = {}
            self.state = state
            self.hist = hist
            for g in groups:
                for li in range(g.n):
                    for k in g.ir.nodes:
                        v = self.val(g, li, k)
                        traj.setdefault((g.gid, k), [[0.0] * T for _ in range(g.n)])[li][t] = v
                    for s in g.ir.states:
                        traj.setdefault((g.gid, f"state:{s}"), [[0.0] * T for _ in range(g.n)])[li][t] = \
                            state[(g.gid, li, s)]
            for (gid, li, k), v in list(self.cache.items()):
                if k.startswith("dsrc:"):
                    hist.setdefault((gid, li, k), []).append(v)
            new_state = dict(state)
            for g in groups:
                for s, info in g.ir.states.items():
                    if not info.has_update:
                        continue
                    for li in range(g.n):
                        v = self.val(g, li, f"next:{s}")
                        if self.physical and info.lo is not None:
                            lo = self.eval(g, li, info.lo)
                            hi = self.eval(g, li, info.hi)
                            v = min(max(v, lo), hi)
                        new_state[(g.gid, li, s)] = v
            state = new_state
        out = {}
        for (gid, k), rows in traj.items():
            out[(gid, k)] = torch.tensor(rows, dtype=torch.float64).unsqueeze(0)
        self.final_state = state
        return out

    # ------------------------------------------------------------------ evaluation
    def val(self, g, li: int, key: str) -> float:
        ck = (g.gid, li, key)
        if ck in self.cache:
            return self.cache[ck]
        if key.startswith("in:"):
            v = self._inflow(g, li, key[3:])
        elif key.startswith("state:"):
            v = self.state[(g.gid, li, key[6:])]
        else:
            e = g.ir.nodes[key]
            if isinstance(e, Op) and e.op == "delay":
                src = e.args[0].name
                k = e.extra["k"]
                kk = int(round(self.eval(g, li, k))) if not isinstance(k, C) else int(round(k.value))
                h = self.hist.get((g.gid, li, src), [])
                if self.t - kk >= 0:
                    v = h[self.t - kk]
                else:
                    v = self.eval(g, li, e.extra["init"])
            else:
                v = self.eval(g, li, e)
        self.cache[ck] = v
        return v

    def _inflow(self, g, li: int, ch: str) -> float:
        pname = ch.split(":")[0]
        rest = ch.split(":")[1:]
        res_short = rest[0]
        attr = rest[1] if len(rest) > 1 else None
        iid = g.inst_ids[li]
        inst = self.flat.instances[iid]
        port = self.flat.ports[inst.ports[pname]]
        total = 0.0
        for eid in port.edges:
            e = self.flat.edges[eid]
            res = next((r for r in e.resources if r.split(".")[-1] == res_short), None)
            if res is None:
                continue
            sp = self.flat.ports[e.src]
            sgid, sli = self.p.inst_group[sp.inst]
            sg = self.p.gmap[sgid]
            skey = f"out:{chan_key(sp.name, res, attr)}"
            if skey not in sg.ir.nodes:
                continue
            f = self.val(sg, sli, skey)
            r = self.p.edge_ridx.get(eid)
            if r is not None:
                f *= float(self.R[0, r, self.t])
            total += f
        return total

    def eval(self, g, li: int, e: IR) -> float:
        if isinstance(e, C):
            return e.value
        if isinstance(e, Ref):
            k, n = e.kind, e.name
            if k == "param":
                if g.is_env:
                    return self._env(n)
                return float(self.P[g.gid][n].reshape(-1)[li])
            if k == "dec":
                return float(self.D[g.gid][n][0, li, self.t])
            if k == "state":
                return self.state[(g.gid, li, n)]
            if k == "in":
                return self.val(g, li, f"in:{n}")
            if k == "node":
                return self.val(g, li, n)
            if k == "env":
                return self._env(n)
            if k == "b":
                tm = self.t0 + self.t * self.dt
                return {"dt": self.dt, "t": float(self.t), "time": tm, "hour": tm % 24.0,
                        "dow": math.floor(tm / 24.0) % 7.0}[n]
            if k == "agg":
                return self._agg(g, n)
            raise ValueError(k)
        a = [self.eval(g, li, x) for x in e.args] if e.op != "delay" else None
        op = e.op
        if op == "add": return a[0] + a[1]
        if op == "sub": return a[0] - a[1]
        if op == "mul": return a[0] * a[1]
        if op == "div": return a[0] / (a[1] if abs(a[1]) >= 1e-12 else math.copysign(1e-12, a[1] if a[1] else 1))
        if op == "pow":
            if float(a[1]).is_integer():
                return a[0] ** int(a[1])
            return max(a[0], 0.0) ** a[1]
        if op == "mod": return a[0] % a[1]
        if op == "neg": return -a[0]
        if op == "not": return 1.0 - a[0]
        if op == "and": return a[0] * a[1]
        if op == "or": return min(1.0, a[0] + a[1])
        if op == "lt": return float(a[0] < a[1])
        if op == "le": return float(a[0] <= a[1])
        if op == "gt": return float(a[0] > a[1])
        if op == "ge": return float(a[0] >= a[1])
        if op == "eq": return float(a[0] == a[1])
        if op == "ne": return float(a[0] != a[1])
        if op == "min": return min(a)
        if op == "max": return max(a)
        if op == "sum": return sum(a)
        if op == "mean": return sum(a) / len(a)
        if op == "pos": return max(a[0], 0.0)
        if op == "negpart": return min(a[0], 0.0)
        if op == "abs": return abs(a[0])
        if op == "sqrt": return math.sqrt(max(a[0], 0.0))
        if op == "exp": return math.exp(min(a[0], 80.0))
        if op == "log": return math.log(max(a[0], 1e-12))
        if op == "log10": return math.log10(max(a[0], 1e-12))
        if op == "sin": return math.sin(a[0])
        if op == "cos": return math.cos(a[0])
        if op == "tanh": return math.tanh(a[0])
        if op == "sigmoid": return 1.0 / (1.0 + math.exp(-a[0]))
        if op == "floor": return float(math.floor(a[0]))
        if op == "ceil": return float(math.ceil(a[0]))
        if op == "round": return float(torch.round(torch.tensor(a[0])).item())
        if op == "sign": return float((a[0] > 0) - (a[0] < 0))
        if op == "step": return 1.0 if a[0] > 0 else 0.0
        if op == "clamp": return min(max(a[0], a[1]), a[2])
        if op == "where": return a[1] if a[0] > 0.5 else a[2]
        if op == "smin":
            k = a[2]
            return -float(torch.logaddexp(torch.tensor(-k * a[0], dtype=torch.float64),
                                          torch.tensor(-k * a[1], dtype=torch.float64))) / k
        if op == "smax":
            k = a[2]
            return float(torch.logaddexp(torch.tensor(k * a[0], dtype=torch.float64),
                                         torch.tensor(k * a[1], dtype=torch.float64))) / k
        if op == "safediv": return a[0] / max(a[1], 1e-9)
        if op == "sat": return min(a[0], a[1]) if self.physical else a[0]
        if op == "pwl":
            xs = e.extra["xs"]
            x = min(max(a[0], xs[0]), xs[-1])
            ys = a[1:]
            for i in range(len(xs) - 1):
                if x <= xs[i + 1] or i == len(xs) - 2:
                    w = (x - xs[i]) / max(xs[i + 1] - xs[i], 1e-12)
                    w = min(max(w, 0.0), 1.0)
                    return (1 - w) * ys[i] + w * ys[i + 1]
        raise ValueError(f"interp: unsupported op {op}")

    def _env(self, n: str) -> float:
        t = self.Et[n]
        return float(t[0, 0, self.t if t.shape[-1] > 1 else 0])

    def _agg(self, g, name: str) -> float:
        parts, how = g.ir.aggs[name]
        vals = []
        cnt = 0
        for gid, ref, lis in parts:
            sg = self.p.gmap[gid]
            idxs = lis if lis is not None else list(range(sg.n))
            for li in idxs:
                cnt += 1
                if ref[0] == "const":
                    vals.append(ref[1])
                elif ref[0] == "dec":
                    vals.append(float(self.D[gid][ref[1]][0, li, self.t]))
                elif ref[0] == "param":
                    vals.append(float(self.P[gid][ref[1]].reshape(-1)[li]))
                elif ref[0] == "state":
                    vals.append(self.state[(gid, li, ref[1])])
                elif ref[0] == "in":
                    vals.append(self.val(sg, li, f"in:{ref[1]}"))
                else:
                    vals.append(self.val(sg, li, ref[1]))
        if how == "count":
            return float(cnt)
        if not vals:
            return 0.0
        if how in ("sum", "value"):
            return sum(vals)
        if how == "mean":
            return sum(vals) / len(vals)
        if how == "max":
            return max(vals)
        if how == "min":
            return min(vals)
        raise ValueError(how)
