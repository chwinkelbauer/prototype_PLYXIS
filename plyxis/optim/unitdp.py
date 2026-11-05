"""Exact per-unit dynamic programs for discrete decisions with a sequence rule (the plant's own subsolvers).

Gradient search with rounding cannot schedule heats: round a relaxed tap sequence and two taps land in one cycle,
two vessels tap into the ladle furnace at once. But the rules that make it hard are local to one unit and simple:

* gap      - at most one event in any `W` consecutive steps (a converter: `tap + delay(tap, 1) + delay(tap, 2) <= up`,
             or with a running count `taps <- taps + tap` and a window given as a duration,
             `taps + tap - delay(taps, cycle - dt) <= up`);
* run      - at most `W - 1` steps on in any `W` (a caster's tundish: a running total `ran <- ran + run` and
             `ran + run - delay(ran, k) <= k`, or `<= tundish / dt` with `delay(ran, tundish)` for a duration);
* reset    - a counter that grows with the unit's work and is reset by the decision, below a limit (a mill's work
             rolls: `wear <- (wear + rolled * dt) * (1 - change)`, `wear <= campaign`).

`find_rules` recognises them in the compiled model (any blueprint, no names). Given the value of each step's
decision - the derivative of the augmented Lagrangian, the same function the ALM descends, so limits shared with
other units (one heat at a time in the ladle furnace, the hot-metal pool) are priced in it - a dynamic program over
time finds the best sequence that keeps the rule exactly. Units are solved one after another with fresh gradients
(the second vessel sees the first one's taps), the continuous decisions are repaired by the ALM, and this repeats.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..lang.behavior import C, Op, Ref
from .gradient import _FlatResiduals, alm
from .problem import Problem

_DEBUG = bool(__import__('os').environ.get('PLX_UNITDP_DEBUG'))


@dataclass
class Rule:
    kind: str               # gap | run | reset
    gid: str
    dec: str                # the decision it schedules
    W: object = None            # gap/run: window (steps): a number, ('param', name) or ('expr', IR of params and dt)
    cap: Optional[tuple] = None  # gap: rhs atom ('state', 'up') / ('const', 1.0); run/reset: ('param', name), const, expr
    state: Optional[str] = None  # reset: the counter state
    wplus: int = 0              # gap from a running count: the window is W + 1 steps


def _linear(e, out: dict, s: float = 1.0) -> bool:
    """Atoms and coefficients of a linear IR expression; False if it is not linear in its atoms."""
    if isinstance(e, Ref):
        out[(e.kind, e.name)] = out.get((e.kind, e.name), 0.0) + s
        return True
    if isinstance(e, C):
        out[("const", "")] = out.get(("const", ""), 0.0) + s * float(e.value)
        return True
    if isinstance(e, Op):
        if e.op == "add":
            return _linear(e.args[0], out, s) and _linear(e.args[1], out, s)
        if e.op == "sub":
            return _linear(e.args[0], out, s) and _linear(e.args[1], out, -s)
        if e.op == "neg":
            return _linear(e.args[0], out, -s)
        if e.op == "mul":
            a, b = e.args
            if isinstance(a, C):
                return _linear(b, out, s * float(a.value))
            if isinstance(b, C):
                return _linear(a, out, s * float(b.value))
    return False


def _delay_of(ir, name: str):
    """(source atom, k) of a delay node, k a constant or ('param', name)."""
    d = ir.nodes.get(name)
    if not (isinstance(d, Op) and d.op == "delay"):
        return None
    src = ir.nodes.get(d.args[0].name) if isinstance(d.args[0], Ref) and d.args[0].kind == "node" else d.args[0]
    if not isinstance(src, Ref):
        return None
    k = d.extra["k"]
    if isinstance(k, C):
        k = float(k.value)
    elif isinstance(k, Ref) and k.kind == "param":
        k = ("param", k.name)
    elif _static(k):
        k = ("expr", k)
    else:
        return None
    return (src.kind, src.name), k


def _static(e) -> bool:
    """An expression of constants, properties and dt only (a duration in steps)."""
    if isinstance(e, C):
        return True
    if isinstance(e, Ref):
        return e.kind == "param" or (e.kind == "b" and e.name == "dt")
    if isinstance(e, Op):
        return e.op in ("add", "sub", "mul", "div", "neg", "min", "max") and all(_static(a) for a in e.args)
    return False


def _atom(rhs) -> Optional[tuple]:
    if isinstance(rhs, C):
        return ("const", float(rhs.value))
    if isinstance(rhs, Ref) and rhs.kind in ("param", "state"):
        return (rhs.kind, rhs.name)
    if isinstance(rhs, Op) and _static(rhs):
        return ("expr", rhs)
    return None


def _num(prog, P: dict, gid: str, item, li: int) -> float:
    """A window or cap of one unit: a number, ('const', v), ('param', name) or ('expr', IR)."""
    from ..lang.hierarchy import static_value
    if not isinstance(item, tuple):
        return float(item)
    kind, v = item
    if kind == "const":
        return float(v)
    pv = {k: float(t.reshape(-1)[li] if t.numel() > 1 else t.reshape(-1)[0]) for k, t in P[gid].items()}
    if kind == "param":
        return pv[v]
    out = static_value(v, pv, float(prog.dt))
    return float("nan") if out is None else out


def find_rules(prog, space) -> list:
    rules = []
    blocks = {(b.gid, b.name): b for b in space.blocks}
    for g in prog.groups:
        if g.is_env:
            continue
        ir = g.ir
        for ck, ci in ir.cons.items():
            if ci.soft is not None or ci.kind != "le":
                continue
            lin: dict = {}
            if not _linear(ci.lhs, lin):
                continue
            cap = _atom(ci.rhs)
            if cap is None:
                continue
            decs = [n for (k, n) in lin if k == "dec"]
            delays = {n: _delay_of(ir, n) for (k, n) in lin if k == "node"}
            # gap: d + delay(d, 1) + ... + delay(d, W-1) <= cap
            if len(decs) == 1 and (g.gid, decs[0]) in blocks and blocks[(g.gid, decs[0])].domain == "bool" \
                    and all(v is not None and v[0] == ("dec", decs[0]) and not isinstance(v[1], tuple) for v in delays.values()) \
                    and all(abs(c - 1.0) < 1e-9 for c in lin.values()) and len(lin) == len(delays) + 1:
                ks = sorted(int(round(v[1])) for v in delays.values())
                if ks == list(range(1, len(ks) + 1)):
                    rules.append(Rule("gap", g.gid, decs[0], W=len(ks) + 1, cap=cap))
                    continue
            # run / gap over a running total: s + d - delay(s, k) <= cap with s <- s + d
            states = [n for (k, n) in lin if k == "state"]
            if len(decs) == 1 and len(states) == 1 and len(delays) == 1 and (g.gid, decs[0]) in blocks \
                    and blocks[(g.gid, decs[0])].domain == "bool" and all(v is not None for v in delays.values()):
                s, d = states[0], decs[0]
                (src, k), = delays.values()
                nxt: dict = {}
                upd = ir.nodes.get(f"next:{s}")
                coef_ok = abs(lin[("state", s)] - 1) < 1e-9 and abs(lin[("dec", d)] - 1) < 1e-9 and \
                    abs(lin[("node", next(iter(delays)))] + 1) < 1e-9
                if src == ("state", s) and coef_ok and upd is not None and _linear(upd, nxt) and \
                        set(nxt) == {("state", s), ("dec", d)} and all(abs(v - 1) < 1e-9 for v in nxt.values()):
                    # the window is k + 1 steps, at most cap of them on: a longest-run limit when cap == k, a gap
                    # rule when cap is one (or the unit's availability)
                    kv = _num(prog, prog.P0, g.gid, k, 0)
                    cv = 1.0 if cap[0] == "state" else _num(prog, prog.P0, g.gid, cap, 0)
                    if abs(cv - kv) < 1e-6:
                        rules.append(Rule("run", g.gid, d, W=k, cap=cap))
                        continue
                    if abs(cv - 1.0) < 1e-6 and kv >= 1 - 1e-6:
                        rules.append(Rule("gap", g.gid, d, W=k, cap=cap, wplus=1))
                        continue
            # reset: s <= cap with s <- (s + ...) * (1 - d)
            if len(lin) == 1 and states and abs(lin[("state", states[0])] - 1) < 1e-9:
                s = states[0]
                upd = ir.nodes.get(f"next:{s}")
                if isinstance(upd, Op) and upd.op == "mul":
                    for a, b in (upd.args, upd.args[::-1]):
                        if isinstance(b, Op) and b.op == "sub" and isinstance(b.args[0], C) and \
                                float(b.args[0].value) == 1.0 and isinstance(b.args[1], Ref) and b.args[1].kind == "dec":
                            d = b.args[1].name
                            if (g.gid, d) in blocks and blocks[(g.gid, d)].domain == "bool" and blocks[(g.gid, d)].hold == 1 \
                                    and isinstance(a, Op) and ("state", s) in _atoms(a):
                                rules.append(Rule("reset", g.gid, d, cap=cap, state=s))
                                break
    return rules


def _atoms(e) -> set:
    out = set()
    if isinstance(e, Ref):
        out.add((e.kind, e.name))
    elif isinstance(e, Op):
        for a in e.args:
            out |= _atoms(a)
    return out


# ----------------------------------------------------------------------------- dynamic programs
def dp_gap(g: torch.Tensor, W: int, allowed: torch.Tensor) -> torch.Tensor:
    """max sum(g_t d_t), d binary, at most one 1 in any W consecutive steps, d_t = 0 where not allowed."""
    T = g.shape[0]
    NEG = -1e30
    # state s = steps since the last 1, capped at W (W = free)
    val = [NEG] * (W + 1)
    val[W] = 0.0
    back = []
    gl = g.tolist()
    al = allowed.tolist()
    for t in range(T):
        nv = [NEG] * (W + 1)
        bk = [None] * (W + 1)
        for s in range(1, W + 1):
            if val[s] <= NEG / 2:
                continue
            s2 = min(s + 1, W)                                  # no event now
            if val[s] > nv[s2]:
                nv[s2], bk[s2] = val[s], (s, 0)
            if s >= W and al[t] and val[s] + gl[t] > nv[1]:     # event now
                nv[1], bk[1] = val[s] + gl[t], (s, 1)
        val = nv
        back.append(bk)
    s = max(range(W + 1), key=lambda i: val[i])
    d = [0] * T
    for t in range(T - 1, -1, -1):
        ps, e = back[t][s]
        d[t] = e
        s = ps
    return torch.tensor(d, dtype=torch.float32)


def dp_run(g: torch.Tensor, R: int) -> torch.Tensor:
    """max sum(g_b d_b), d binary, at most R consecutive ones."""
    T = g.shape[0]
    NEG = -1e30
    val = [NEG] * (R + 1)
    val[0] = 0.0
    back = []
    gl = g.tolist()
    for t in range(T):
        nv = [NEG] * (R + 1)
        bk = [None] * (R + 1)
        for r in range(R + 1):
            if val[r] <= NEG / 2:
                continue
            if val[r] > nv[0]:
                nv[0], bk[0] = val[r], (r, 0)
            if r < R and val[r] + gl[t] > nv[r + 1]:
                nv[r + 1], bk[r + 1] = val[r] + gl[t], (r, 1)
        val = nv
        back.append(bk)
    r = max(range(R + 1), key=lambda i: val[i])
    d = [0] * T
    for t in range(T - 1, -1, -1):
        pr, e = back[t][r]
        d[t] = e
        r = pr
    return torch.tensor(d, dtype=torch.float32)


def dp_reset(g: torch.Tensor, inc: torch.Tensor, w0: float, L: float) -> torch.Tensor:
    """max sum(g_t d_t): a counter w_{t+1} = (w_t + inc_t)(1 - d_t), w_0 = w0, must stay <= L at every step."""
    T = g.shape[0]
    P = [0.0]
    for v in inc.tolist():
        P.append(P[-1] + max(v, 0.0))
    NEG = -1e30
    gl = g.tolist()
    best = [NEG] * T
    prev = [None] * T
    for t in range(T):
        cand, arg = NEG, None
        if w0 + P[t] <= L + 1e-9:
            cand, arg = 0.0, -1
        for r in range(t - 1, -1, -1):
            if P[t] - P[r + 1] > L + 1e-9:
                break
            if best[r] > cand:
                cand, arg = best[r], r
        if arg is not None:
            best[t], prev[t] = cand + gl[t], arg
    # the end: the last reset r must keep the counter below L up to the last step
    end, arg = (0.0, -1) if w0 + P[T - 1] <= L + 1e-9 else (NEG, None)
    for r in range(T - 1, -1, -1):
        if P[T - 1] - P[r + 1] > L + 1e-9:
            break
        if best[r] > end:
            end, arg = best[r], r
    d = [0] * T
    while arg is not None and arg >= 0:
        d[arg] = 1
        arg = prev[arg]
    return torch.tensor(d, dtype=torch.float32)


# ----------------------------------------------------------------------------- the stage
class UnitDP:
    def __init__(self, pb: Problem, rules: Optional[list] = None):
        self.pb = pb
        self.rules = rules if rules is not None else find_rules(pb.prog, pb.space)

    def _value(self, P: dict, gid: str, atom: tuple, n: int, T: int, raw: Optional[dict] = None) -> torch.Tensor:
        """An rhs atom as [n, T]."""
        kind, name = atom
        if kind == "const":
            return torch.full((n, T), float(name))
        if kind == "expr":
            v = torch.tensor([_num(self.pb.prog, P, gid, atom, li) for li in range(n)])
            return v.view(n, 1).expand(n, T)
        if kind == "param":
            v = P[gid][name].reshape(-1)
            return (v if v.numel() == n else v.expand(n)).view(n, 1).expand(n, T)
        x = raw[self.pb.prog.node_var(gid, f"state:{name}")]
        x = x.reshape(x.shape[0], -1, x.shape[-1])[0]
        return x.expand(n, T) if x.shape[-1] == T else x.expand(n, T)

    def _grad(self, x: torch.Tensor, lam: Optional[dict], rho: float, value: float = 1.0) -> tuple:
        """Gradient of the augmented Lagrangian (value = 0: of the penalty alone, what feasibility asks for)."""
        pb = self.pb
        xg = x.detach().clone().requires_grad_(True)
        e = pb.evaluate(xg, relax=True)
        fr = _FlatResiduals(e, 1)
        h = fr.flat(e, 1)
        lm = fr.from_dict(lam, 1)
        z = torch.relu(lm + rho * h)
        pen = torch.where(fr.eq, lm * h + 0.5 * rho * h * h, (z * z - lm * lm) / (2 * rho)).sum()
        L = value * e.objective.sum() / pb.J_scale - pen
        g, = torch.autograd.grad(L, xg)
        return g[0].detach(), e

    def feasibility(self, x: torch.Tensor, sweeps: int = 8, quantiles: tuple = (0.8, 0.95, 0.99, 0.998)) -> tuple:
        """Units one at a time on the gradient of the violations alone (a quadratic penalty, no multipliers): each
        unit's best sequence for it, kept only if the plan's total violation falls. A schedule whose units were rounded
        one by one breaks limits they share (a caster casting while the converters feeding its ladle furnace are
        idle); one sweep over all units on one linearisation over-reacts (every unit moves to fix the same shortfall)."""
        pb, sp = self.pb, self.pb.space
        x = sp.round_discrete(x[:1].detach().clone())
        with torch.no_grad():
            cur = float(pb.evaluate(x, relax=False).sum_violation()[0])
        kept = 0
        for _ in range(sweeps):
            better = False
            for r in self.rules:
                b = sp.block(r.gid, r.dec)
                for li in range(pb.prog.gmap[r.gid].n):
                    g, e = self._grad(x, None, 10.0, value=0.0)
                    idx = b.offset + li * b.H + torch.arange(b.H)
                    gb = g[idx]
                    if float(gb.abs().max()) <= 0.0:
                        continue
                    old = x[0, idx].clone().round()
                    last = None
                    # a change costs a high quantile of the gradient's size: only the few decisions that matter most
                    # may move (the mean lets a caster's whole campaign flip and the shortfall becomes an overflow)
                    for q in quantiles:
                        cost = float(torch.quantile(gb.abs(), q)) + 1e-12
                        gp = gb + cost * (2.0 * old - 1.0)
                        d = self._dp(r, li, b, gp, x, e.raw)
                        if bool((d == old).all()):
                            break
                        if last is not None and bool((d == last).all()):
                            continue
                        last = d
                        xt = x.clone()
                        xt[0, idx] = d
                        with torch.no_grad():
                            v = float(pb.evaluate(xt, relax=False).sum_violation()[0])
                        if _DEBUG:
                            print(f"feasibility {r.kind} {pb.prog.gmap[r.gid].ir.type.short}[{li}] q {q}: "
                                  f"{int(old.sum())} -> {int(d.sum())} on, violation {cur:.4g} -> {v:.4g}", flush=True)
                        if v < cur - 1e-9:
                            x, cur, better = xt, v, True
                            kept += 1
                            break
            if not better or cur <= 1e-9:
                break
        return x, kept

    def _dp(self, r: Rule, li: int, b, gb: torch.Tensor, x: torch.Tensor, raw: dict) -> torch.Tensor:
        """The best sequence of one unit for per-variable values gb (one per hold block), keeping its rule."""
        pb = self.pb
        n, T = pb.prog.gmap[r.gid].n, pb.T
        if r.kind == "gap":
            allowed = self._value(pb.P, r.gid, r.cap, n, T, raw)[li] >= 1.0 - 1e-6
            W = self._window(r, li)
            if b.hold == 1:
                return dp_gap(gb, W, allowed)
            ab = allowed.reshape(-1)[: b.H * b.hold].reshape(b.H, -1).all(-1)
            return dp_gap(gb, max(1, -(-W // b.hold)), ab)
        if r.kind == "run":
            return dp_run(gb, self._run_blocks(r, li, b))
        idx = b.offset + li * b.H + torch.arange(b.H)
        off = x.clone()
        off[0, idx] = 0.0
        with torch.no_grad():
            eo = pb.evaluate(off, relax=True)
        s = eo.raw[pb.prog.node_var(r.gid, f"state:{r.state}")]
        s = s.reshape(s.shape[0], -1, s.shape[-1])[0, li]
        inc = torch.cat([s[1:] - s[:-1], (s[-1:] - s[-2:-1]) if s.numel() > 1 else s[:1] * 0])
        L = float(self._value(pb.P, r.gid, r.cap, n, 1)[li, 0])
        return dp_reset(gb, inc, float(s[0]), L)

    def _run_blocks(self, r: Rule, li: int, b) -> int:
        pb = self.pb
        k = _num(pb.prog, pb.P, r.gid, r.W, li)
        cap = float(self._value(pb.P, r.gid, r.cap, pb.prog.gmap[r.gid].n, 1)[li, 0])
        return max(0, int(round(min(cap, k), 6) // b.hold))

    def _window(self, r: Rule, li: int) -> int:
        """A gap rule's window in steps for one unit."""
        return int(round(_num(self.pb.prog, self.pb.P, r.gid, r.W, li))) + r.wplus

    def start(self, x_rel: torch.Tensor, lam: Optional[dict]) -> torch.Tensor:
        """A first schedule from the relaxed plan: each unit's fractional decisions rounded by carrying the rounding
        error forward (its rate stays what the relaxation planned), within its rule. Units of one type start with
        staggered errors, so three vessels at a third of a heat per step do not all tap in the same step."""
        pb, sp = self.pb, self.pb.space
        x = sp.round_discrete(x_rel[:1].detach().clone())
        u_rel = x_rel[0].detach()
        with torch.no_grad():
            e = pb.evaluate(x, relax=True)
        for r in self.rules:
            b = sp.block(r.gid, r.dec)
            n = pb.prog.gmap[r.gid].n
            for li in range(n):
                idx = b.offset + li * b.H + torch.arange(b.H)
                u = u_rel[idx].clamp(0, 1).tolist()
                if r.kind == "gap":
                    allowed = self._value(pb.P, r.gid, r.cap, n, pb.T, e.raw)[li] >= 1.0 - 1e-6
                    W = self._window(r, li) if b.hold == 1 else max(1, -(-self._window(r, li) // b.hold))
                    al = allowed.tolist() if b.hold == 1 else \
                        allowed.reshape(-1)[: b.H * b.hold].reshape(b.H, -1).all(-1).tolist()
                    acc, last, d = li / max(n, 1), -10 ** 9, []
                    for t, ut in enumerate(u):
                        acc += ut
                        if acc >= 0.5 and al[t] and t - last >= W:
                            d.append(1.0)
                            acc -= 1.0
                            last = t
                        else:
                            d.append(0.0)
                    x[0, idx] = torch.tensor(d)
                elif r.kind == "run":
                    # follow the relaxation (on where it is mostly on), at most R blocks in a row
                    x[0, idx] = dp_run(torch.tensor(u) - 0.5, self._run_blocks(r, li, b))
                else:
                    # as few resets as the counter allows (a reset is not a way to stop the unit: its rate does that)
                    gb = torch.full((b.H,), -1.0)
                    x[0, idx] = self._dp(r, li, b, gb, x, e.raw)
        return self._declash(x)

    def _declash(self, x: torch.Tensor) -> torch.Tensor:
        """Identical units firing in the same step where the plan breaks a limit (two vessels tapping into one ladle
        furnace): the later one moves to the nearest step its own rule allows where no other unit of its type fires."""
        pb, sp = self.pb, self.pb.space
        with torch.no_grad():
            e = pb.evaluate(x, relax=False)
        bad = torch.zeros(pb.T, dtype=torch.bool)
        for v in e.viol.values():
            v = v.reshape(v.shape[0], -1, v.shape[-1])[0]
            if v.shape[-1] == pb.T:
                bad |= (v > 1e-3).any(0)
        for r in self.rules:
            b = sp.block(r.gid, r.dec)
            n = pb.prog.gmap[r.gid].n
            if r.kind != "gap" or b.hold != 1 or n < 2:
                continue
            W = self._window(r, 0)
            seq = [x[0, b.offset + li * b.H + torch.arange(b.H)].round().clone() for li in range(n)]
            for li in range(1, n):
                others = sum(seq[j] for j in range(n) if j != li)
                for t in torch.nonzero(seq[li] > 0.5).flatten().tolist():
                    if not (others[t] > 0.5 and bad[t]):
                        continue
                    for dlt in (1, -1, 2, -2, 3, -3, 4, -4):
                        t2 = t + dlt
                        if not 0 <= t2 < b.H or others[t2] > 0.5:
                            continue
                        trial = seq[li].clone()
                        trial[t] = 0.0
                        trial[t2] = 1.0
                        lo, hi = max(0, t2 - W + 1), min(b.H, t2 + W)
                        if all(float(trial[k:k + W].sum()) <= 1.0 for k in range(lo, max(lo, hi - W + 1))):
                            seq[li] = trial
                            others = sum(seq[j] for j in range(n) if j != li)
                            break
            for li in range(n):
                x[0, b.offset + li * b.H + torch.arange(b.H)] = seq[li]
        return x

    def sweep(self, x: torch.Tensor, lam: Optional[dict], mu: float, rho: float = 1.0) -> tuple:
        """One pass over every rule and instance, units one by one with fresh gradients: the best sequence for the
        linearised value with a cost for every decision it changes - a trust region, without which each unit swings
        between all and nothing. The cost is the `mu` quantile of the unit's gradient sizes if 0 < mu < 1 (only the
        decisions that matter most move), else `mu` times their mean."""
        pb, sp = self.pb, self.pb.space
        x = sp.round_discrete(x[:1].detach().clone())
        changed = 0
        for r in self.rules:
            b = sp.block(r.gid, r.dec)
            n = pb.prog.gmap[r.gid].n
            for li in range(n):
                g, e = self._grad(x, lam, rho)
                idx = b.offset + li * b.H + torch.arange(b.H)
                gb = g[idx]
                old = x[0, idx].clone().round()
                scale = (float(torch.quantile(gb.abs(), mu)) if 0.0 < mu < 1.0 else mu * float(gb.abs().mean())) + 1e-12
                gp = gb + scale * (2.0 * old - 1.0)               # keeping a decision is worth it, changing costs it
                d = self._dp(r, li, b, gp, x, e.raw)
                if _DEBUG:
                    print(f"unit dp {r.kind} {pb.prog.gmap[r.gid].ir.type.short}[{li}].{r.dec}: g mean {float(gb.mean()):.3g} "
                          f"pos {int((gb > 0).sum())}/{gb.numel()} mu {mu:.2g} -> {int(d.sum())} on (was {int(old.sum())})",
                          flush=True)
                x[0, idx] = d
                changed += int((old != d).sum())
        return x, changed

    def _repair(self, x, lam, repair_iters, callback, patience):
        pb = self.pb
        mask = pb.space.discrete_mask()
        rep = alm(pb, x, iters=repair_iters, lr=0.02, rho=0.5, outer=10, mask=~mask, relax=False,
                  stage="discrete", callback=callback, lam0=lam, patience=patience)
        with torch.no_grad():
            ev = pb.evaluate(rep.x[:1], relax=False)
        score = float(ev.objective[0]) / pb.J_scale - 50.0 / pb.T * float(ev.sum_violation()[0])
        return rep, ev, score

    def solve(self, x_rel: torch.Tensor, lam: Optional[dict], passes: int = 3, repair_iters: int = 60,
              callback=None, patience: Optional[int] = None, mu: float = 0.9) -> tuple:
        """Start from the relaxed plan, then trust-region sweeps; a sweep is kept only if the repaired plan scores
        better (penalised), else the trust region shrinks."""
        pb = self.pb
        x = self.start(x_rel, lam)
        rep, ev, score = self._repair(x, lam, repair_iters, callback, patience)
        best = (score, rep.x[:1].clone(), float(ev.objective[0]), float(ev.max_violation()[0]), rep.multipliers)
        if callback:
            callback({"stage": "discrete", "method": "unit dp", "pass": "start", "objective": best[2], "max_violation": best[3]})
        for k in range(4):
            if best[3] <= 2e-3:
                break
            # the rounded schedule breaks a shared limit: feasibility first, unit by unit, then the continuous
            # decisions repaired around it; again while the violation falls
            xf, kept = self.feasibility(best[1])
            if not kept:
                break
            rep, ev, score = self._repair(xf, best[4], repair_iters, callback, patience)
            mv = float(ev.max_violation()[0])
            ok = mv < best[3] - 1e-6
            if callback:
                callback({"stage": "discrete", "method": "unit dp", "pass": f"feasibility {k}", "changed": kept,
                          "accepted": ok, "objective": float(ev.objective[0]), "max_violation": mv})
            if not ok:
                break
            best = (score, rep.x[:1].clone(), float(ev.objective[0]), mv, rep.multipliers)
        for it in range(passes):
            xt, changed = self.sweep(best[1], best[4], mu)
            if changed == 0:
                break
            rep, ev, score = self._repair(xt, best[4], repair_iters, callback, patience)
            ok = score > best[0]
            if callback:
                callback({"stage": "discrete", "method": "unit dp", "pass": it, "changed": changed, "mu": mu,
                          "accepted": ok, "objective": float(ev.objective[0]), "max_violation": float(ev.max_violation()[0])})
            if ok:
                best = (score, rep.x[:1].clone(), float(ev.objective[0]), float(ev.max_violation()[0]), rep.multipliers)
            else:
                mu = 1.0 - (1.0 - mu) / 4.0                       # a tighter trust region: fewer decisions move
        _, xb, obj, mv, lam = best
        return xb, obj, mv, lam


__all__ = ["Rule", "find_rules", "dp_gap", "dp_run", "dp_reset", "UnitDP"]
