"""The twin's equations as a sparse algebraic model: the second backend of the PLX compiler.

The torch backend turns a twin into a simulator: decisions in, trajectories out. Exact methods (LP, MILP, dynamic
programs, decomposition with prices) need the same equations as algebra - rows and columns. `linearize(pb, x)` walks
the compiled program's node graph (the same nodes the simulator computes, in dependency order) and builds, for every
node, its value at the plan `x` and its sparse Jacobian, in the lifted form of optimal control:

    columns   u        the plan's canonical decisions (DecisionSpace: one per instance and hold block, routes)
              s[t]     every state with an update, per instance and step t = 1..T (s[0] is the start state)
              d[t]     delay nodes that sit on a recursion (a node that reads its own past through delay())
    rows      s[t+1] - next(t) = 0                       the state updates
              d[t] - src[t - k] = 0                      the lifted delays
              h(t) <= 0  /  h(t) = 0                     every hard constraint (signed residual, as the simulator)
              h(t) <= sigma, sigma >= 0, price w dt      every soft constraint
    objective the twin's own objective (terms, final terms, carbon, ...) as a gradient

With the states as columns every row is local in time, so the model is very sparse (a state's row touches the
state, its next value and the flows of that step). Where the equations are affine in the columns the model is
exact; elsewhere (products of two decisions, divisions by a flow, tables, min / max) it is the first-order model
at `x`, and every row carries its level: 1 affine, 2 piecewise affine (exact inside the current piece), 3
nonlinear. The torch simulator stays the judge: `check()` compares values and directional derivatives with it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch

from ..lang.behavior import C, Op, Ref

AFFINE, PWA, NONLIN = 1, 2, 3
AUXCAP = 1_000_000          # room for the columns of exact piecewise terms (sliced off at the end)
MU = 50.0                     # price of a hard violation per scaled unit and step, as in branchlp / Problem fitness


# ----------------------------------------------------------------------------- linear forms
class Lin:
    """Value V [n, T] and sparse Jacobian M [(n*T), nvar] (None = constant) of one node; lvl = exactness."""
    __slots__ = ("V", "M", "lvl")

    def __init__(self, V: np.ndarray, M=None, lvl: int = 0):
        self.V, self.M, self.lvl = V, M, lvl

    @property
    def moves(self) -> bool:
        return self.M is not None or self.lvl > 0


def _scale(M, c):
    """Rows of M scaled by c (flattened over (n, T)); None stays None."""
    if M is None:
        return None
    c = np.asarray(c, dtype=np.float64).ravel()
    if c.size == 1:
        return None if c[0] == 0 else M * float(c[0])
    nnz = np.diff(M.indptr)
    out = sp.csr_matrix((M.data * np.repeat(c, nnz), M.indices.copy(), M.indptr.copy()), shape=M.shape)
    if not np.all(c):
        out.eliminate_zeros()
        if out.nnz == 0:
            return None
    return out


def _rowmix(nrows: int, dst, src, w, M):
    """Rows dst[k] += w[k] * M[src[k]] (a gather of rows; costs the nonzeros, unlike a product with a selection
    matrix, which allocates a work array as wide as the model)."""
    if M is None:
        return None
    sub = _scale(M.tocsr()[np.asarray(src)], np.asarray(w, dtype=np.float64))
    if sub is None:
        return None
    rows = np.repeat(np.asarray(dst), np.diff(sub.indptr))
    return sp.csr_matrix((sub.data, (rows, sub.indices)), shape=(nrows, M.shape[1]))


def _add(A, B):
    if A is None:
        return B
    if B is None:
        return A
    return (A + B).tocsr()


def _lvl(*xs, at_least: int = 0) -> int:
    lv = max(x.lvl for x in xs)
    return max(lv, at_least) if any(x.moves for x in xs) else lv


def _mul(a: Lin, b: Lin) -> Lin:
    lv = NONLIN if (a.M is not None and b.M is not None) else _lvl(a, b)
    return Lin(a.V * b.V, _add(_scale(a.M, b.V), _scale(b.M, a.V)), lv)


def _select(m: np.ndarray, a: Lin, b: Lin, at_least: int) -> Lin:
    """a where m else b."""
    return Lin(np.where(m, a.V, b.V), _add(_scale(a.M, m), _scale(b.M, ~m)), _lvl(a, b, at_least=at_least))


def _unary(a: Lin, V: np.ndarray, d: np.ndarray, lvl: int) -> Lin:
    return Lin(V, _scale(a.M, d), _lvl(a, at_least=lvl))


def _np(x, n: int, T: int) -> np.ndarray:
    """Any tensor / float of the simulator broadcast to a float64 array [n, T]."""
    if isinstance(x, torch.Tensor):
        x = x.detach().to(torch.float64).numpy()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 3:
        x = x[0]
    elif x.ndim == 2:
        x = x[0][:, None] if x.shape[0] == 1 and x.shape[1] == n and n != 1 else x
        if x.shape == (1, n) and n == 1:
            x = x.reshape(1, 1)
    elif x.ndim == 1:
        x = x.reshape(-1, 1) if x.shape[0] == n and n != T else x.reshape(1, -1)
    return np.array(np.broadcast_to(x, (n, T)), dtype=np.float64)


# ----------------------------------------------------------------------------- the model
@dataclass
class LinearModel:
    """Rows J x (sense) J x_ref - r0 around the reference x_ref; the objective f0 + g (x - x_ref) - soft penalty."""
    nvar: int
    N: int                          # the first N columns are the plan's canonical decisions u
    x_ref: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    steps: np.ndarray               # integer steps of a discrete column (0 = continuous)
    var_inst: np.ndarray            # flat instance id of each column (-1 = environment / none)
    var_t: np.ndarray               # step of each column
    var_kind: np.ndarray            # 0 decision, 1 state, 2 lifted delay
    J: sp.csr_matrix
    r0: np.ndarray                  # row values at x_ref
    eq: np.ndarray                  # bool: equality row
    w: np.ndarray                   # price per unit and step of a soft row (USD, already x dt); 0 = hard
    mu: np.ndarray                  # price per raw unit of an elastic hard row's violation (scaled objective)
    row_inst: np.ndarray
    row_t: np.ndarray
    row_kind: np.ndarray            # 0 state update, 1 lifted delay, 2 hard, 3 soft
    row_con: np.ndarray             # index into `cons`
    row_lvl: np.ndarray
    cons: list                      # (gid, con key) per index; ("state", gid, s) / ("delay", nid) for the others
    g: np.ndarray                   # gradient of the objective (without soft penalties), USD
    f0: float                       # objective at x_ref without soft penalties, USD
    J_scale: float
    lifted: dict = field(default_factory=dict)       # name -> (offset, n, T, gid) of the lifted column blocks
    col_exact: Optional[np.ndarray] = None           # column only in affine rows and affine objective terms: the model is
                                                     # exact along it (no trust region needed)
    seconds: float = 0.0
    stats: dict = field(default_factory=dict)

    # -------------------------------------------------------------- evaluation of the model itself
    def rows_at(self, x: np.ndarray) -> np.ndarray:
        return self.r0 + self.J @ (x - self.x_ref)

    def objective_at(self, x: np.ndarray) -> float:
        """Linear model of the twin's objective including soft penalties (USD)."""
        r = self.rows_at(x)
        soft = self.w > 0
        v = np.where(self.eq[soft], np.abs(r[soft]), np.maximum(r[soft], 0.0))
        return float(self.f0 + self.g @ (x - self.x_ref) - (self.w[soft] * v).sum())

    def hard_violation(self, x: np.ndarray) -> float:
        r = self.rows_at(x)
        hard = self.row_kind < 3
        return float(np.where(self.eq[hard], np.abs(r[hard]), np.maximum(r[hard], 0.0)).max(initial=0.0))
    # row kinds: 0 state update, 1 lifted delay (equalities), 2 hard, 3 soft, 4 definition of an exact piecewise term

    def describe(self) -> dict:
        lv = np.bincount(self.row_lvl, minlength=4)
        return {"columns": self.nvar, "decisions": self.N, "discrete": int((self.steps > 0).sum()),
                "rows": int(self.J.shape[0]), "nonzeros": int(self.J.nnz),
                "rows_by_kind": {k: int((self.row_kind == i).sum()) for i, k in
                                 enumerate(("state", "delay", "hard", "soft", "piecewise_definition"))},
                "exact_piecewise_terms": int((self.var_kind == 5).sum()),
                "exact_epigraph_terms": int((self.var_kind == 6).sum()),
                "rows_by_level": {"constant": int(lv[0]), "affine": int(lv[1]), "piecewise_affine": int(lv[2]),
                                  "nonlinear": int(lv[3])},
                "seconds": round(self.seconds, 3)}


# ----------------------------------------------------------------------------- the walk
class Linearizer:
    def __init__(self, pb, exact_pwa: bool = True, epigraph: bool = False):
        self.pb, self.prog = pb, pb.prog
        self.exact_pwa = exact_pwa
        # epigraph terms (max() of states, exact where every use prefers the term small): correct, but on the steel
        # group they did not let the sequential LP take larger steps (HiGHS once failed on the larger model) - opt-in
        self.epigraph = epigraph
        self.epi_allowed = None if epigraph else set()   # sites whose epigraph is exact
        if getattr(self.prog.comp, "physical", False):
            raise ValueError("linearize the planning program, not the physical one")
        self.comp = self.prog.comp
        self.T, self.dt = pb.T, float(pb.dt)
        self._order()

    # -------------------------------------------------------------- schedule
    def _deps(self, gn, lifted: set) -> set:
        if gn.kind == "state" or gn.nid in lifted:
            return set()
        d = set(gn.inst)
        if gn.kind == "delay":
            d |= gn.dly
        return d

    def _order(self) -> None:
        """Topological order of the nodes with states (and delays on recursions) lifted to columns."""
        nodes = self.comp.nodes
        lifted: set = set()
        while True:
            order, state, cyc = [], {}, None
            for root in nodes:
                if root in state:
                    continue
                stack = [(root, iter(sorted(self._deps(nodes[root], lifted))))]
                state[root] = 1
                path = [root]
                while stack and cyc is None:
                    v, it = stack[-1]
                    nxt = next(it, None)
                    if nxt is None:
                        stack.pop()
                        path.pop()
                        state[v] = 2
                        order.append(v)
                        continue
                    s = state.get(nxt, 0)
                    if s == 1:
                        cyc = path[path.index(nxt):]
                    elif s == 0:
                        state[nxt] = 1
                        path.append(nxt)
                        stack.append((nxt, iter(sorted(self._deps(nodes[nxt], lifted)))))
                if cyc is not None:
                    break
            if cyc is None:
                break
            dl = [v for v in cyc if nodes[v].kind == "delay"]
            if not dl:
                raise ValueError(f"instant cycle without a delay: {' -> '.join(cyc)}")
            lifted.add(dl[0])
        self.order, self.lifted_delays = order, lifted

    # -------------------------------------------------------------- reference
    def _reference(self, x: torch.Tensor) -> dict:
        pb = self.pb
        D, R = pb.space.decode(pb.pinned(x.reshape(1, -1)), pb.P, relax=True)
        with torch.no_grad():
            raw = self.prog.simulate(D, R, pb.P, pb._states(pb.S0), pb.E, pb.T, pb.dt, pb.t0)
        return raw

    # -------------------------------------------------------------- columns
    def _layout(self, raw: dict, u: np.ndarray) -> None:
        pb, sp_, T = self.pb, self.pb.space, self.T
        self.N = sp_.N
        cols_ref = [u]
        inst, tt, kind = [np.full(self.N, -1)], [np.zeros(self.N, dtype=np.int64)], [np.zeros(self.N, dtype=np.int8)]
        for b in sp_.blocks:
            g = self.prog.gmap[b.gid]
            ids = np.array(g.inst_ids if not g.is_env else [-1])
            inst[0][b.offset:b.offset + b.size] = np.repeat(ids, b.H)
            tt[0][b.offset:b.offset + b.size] = np.tile(np.arange(b.H) * b.hold, b.n)
        if sp_.route is not None:
            r = sp_.route
            owner = np.full(r.E, -1)
            for rp in self.prog.routed:
                owner[rp.ridx] = rp.inst
            inst[0][r.offset:r.offset + r.size] = np.repeat(owner, r.H)
            tt[0][r.offset:r.offset + r.size] = np.tile(np.arange(r.H) * r.hold, r.E)
        off = self.N
        self.lifted = {}
        for nid, gn in self.comp.nodes.items():
            g = self.prog.gmap[gn.gid]
            if gn.kind == "state" and g.ir.states[gn.key.split(":", 1)[1]].has_update:
                traj = _np(raw[gn.var], g.n, T)
                fin = _np(raw[gn.var + "_f"], g.n, 1)
                vals = np.concatenate([traj[:, 1:], fin], 1)           # s[1..T]
            elif nid in self.lifted_delays:
                vals = _np(raw[gn.var], g.n, T)
            else:
                continue
            self.lifted[nid] = (off, g.n, T, gn.gid)
            cols_ref.append(vals.ravel())
            ids = np.array(g.inst_ids if not g.is_env else [-1])
            inst.append(np.repeat(ids, T))
            tt.append(np.tile(np.arange(1, T + 1) if gn.kind == "state" else np.arange(T), g.n))
            kind.append(np.full(g.n * T, 1 if gn.kind == "state" else 2, dtype=np.int8))
            off += g.n * T
        self.nvar = off
        self.x_ref = np.concatenate(cols_ref)
        self.var_inst, self.var_t, self.var_kind = np.concatenate(inst), np.concatenate(tt), np.concatenate(kind)

    def _ident(self, nid: str, n: int, shift_state: bool) -> sp.csr_matrix:
        """Jacobian of a lifted block's trajectory: identity on its columns (a state's step 0 is constant)."""
        off, _, T, _ = self.lifted[nid]
        if shift_state:
            r = np.arange(n * T).reshape(n, T)[:, 1:].ravel()
            c = off + np.arange(n * T).reshape(n, T)[:, :-1].ravel()
        else:
            r = np.arange(n * T)
            c = off + r
        return sp.csr_matrix((np.ones(r.size), (r, c)), shape=(n * T, self.nvar))

    # -------------------------------------------------------------- decisions and routes
    def _decisions(self, u: np.ndarray) -> None:
        sp_, T = self.pb.space, self.T
        bnd = sp_.bounds(self.pb.P)
        self.D = {}
        self.steps = np.zeros(self.nvar, dtype=np.int64)
        self.lo, self.hi = np.zeros(self.nvar), np.ones(self.nvar)
        for b in sp_.blocks:
            lo, hi, _ = bnd[(b.gid, b.name)]
            lo = lo.detach().double().numpy().reshape(-1)
            hi = hi.detach().double().numpy().reshape(-1)
            lo = np.broadcast_to(lo, (b.n,)) if lo.size == 1 else lo[:b.n]
            hi = np.broadcast_to(hi, (b.n,)) if hi.size == 1 else hi[:b.n]
            span = hi - lo
            h = np.minimum(np.arange(T) // b.hold, b.H - 1)
            col = b.offset + np.arange(b.n)[:, None] * b.H + h[None, :]          # [n, T]
            V = lo[:, None] + span[:, None] * u[col]
            rows = np.arange(b.n * T)
            M = sp.csr_matrix((np.repeat(span, T), (rows, col.ravel())), shape=(b.n * T, self.nvar))
            M.eliminate_zeros()
            self.D[(b.gid, b.name)] = Lin(V, M if M.nnz else None, AFFINE if M.nnz else 0)
            if b.domain in ("int", "bool"):
                self.steps[b.offset:b.offset + b.size] = np.repeat(np.round(span).astype(np.int64), b.H)
            fixed = np.repeat(span <= 1e-12, b.H)
            self.hi[b.offset:b.offset + b.size][fixed] = 0.0
        self.R = None
        r = sp_.route
        if r is None:
            return
        E = r.E
        h = np.minimum(np.arange(T) // r.hold, r.H - 1)
        col = r.offset + np.arange(E)[:, None] * r.H + h[None, :]                # [E, T]
        uu = u[col]
        w = np.clip(uu, 0.0, 1.0) + 1e-3
        dw = ((uu >= 0.0) & (uu <= 1.0)).astype(np.float64)
        V = np.zeros((E, T))
        rows, cols, vals = [], [], []
        for rp in self.prog.routed:
            idx = np.array(rp.ridx)
            tot = w[idx].sum(0)                                                   # [T]
            V[idx] = w[idx] / tot
            for a in idx:                       # d split_a / d u_b = (delta_ab tot - w_a) / tot^2 * dw_b
                for b2 in idx:
                    d = ((tot if a == b2 else 0.0) - w[a]) / tot ** 2 * dw[b2]
                    rows.append(a * T + np.arange(T))
                    cols.append(col[b2])
                    vals.append(d)
        M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(E * T, self.nvar))
        M.eliminate_zeros()
        self.R = Lin(V, M, NONLIN)

    # -------------------------------------------------------------- nodes
    def _node(self, nid: str) -> Lin:
        self._cur_key, self._call = nid, 0
        gn = self.comp.nodes[nid]
        g = self.prog.gmap[gn.gid]
        n, T = g.n, self.T
        if nid in self.lifted:
            off = self.lifted[nid][0]
            if gn.kind == "state":
                V = _np(self.raw[gn.var], n, T)
                return Lin(V, self._ident(nid, n, True), AFFINE)
            return Lin(_np(self.raw[gn.var], n, T), self._ident(nid, n, False), AFFINE)
        if gn.kind == "state":
            sname = gn.key.split(":", 1)[1]
            S = self.pb._states(self.pb.S0)
            return Lin(_np(S[gn.gid][sname], n, T))
        if gn.kind == "calc":
            return self._expr(gn.expr, g, False)
        if gn.kind == "in":
            return self._gather(gn, g)
        if gn.kind == "agg":
            return self._agg(gn)
        if gn.kind == "delay":
            return self._delay(gn, g)
        raise ValueError(gn.kind)

    def _gather(self, gn, g) -> Lin:
        n, T = g.n, self.T
        V = np.zeros((n, T))
        M = None
        lv = 0
        tt = np.arange(T)
        for (sgid, skey), triples in gn.expr.items():
            src = self.L[f"{sgid}|{skey}"]
            ns = self.prog.gmap[sgid].n
            sli = np.array([t[0] for t in triples])
            dli = np.array([t[1] for t in triples])
            ri = np.array([t[2] for t in triples])
            routed = ri >= 0
            fac = np.ones((len(triples), T))
            if routed.any():
                fac[routed] = self.R.V[ri[routed]]
            contrib = src.V[sli] * fac
            np.add.at(V, dli, contrib)
            lv = max(lv, src.lvl)
            r = (dli[:, None] * T + tt[None, :]).ravel()
            if src.M is not None:
                c = (sli[:, None] * T + tt[None, :]).ravel()
                M = _add(M, _rowmix(n * T, r, c, fac.ravel(), src.M))
            if routed.any():
                rr = (dli[routed][:, None] * T + tt[None, :]).ravel()
                rc = (ri[routed][:, None] * T + tt[None, :]).ravel()
                M = _add(M, _rowmix(n * T, rr, rc, src.V[sli[routed]].ravel(), self.R.M))
                lv = NONLIN if src.moves else max(lv, AFFINE)
        return Lin(V, M, lv)

    def _agg(self, gn) -> Lin:
        parts, how = gn.expr
        T = self.T
        if how == "count":
            return Lin(np.full((1, T), float(sum(len(lis) if lis is not None else self.prog.gmap[gid].n
                                                     for gid, ref, lis in parts))))
        vals = []
        for gid, ref, lis in parts:
            g = self.prog.gmap[gid]
            if ref[0] == "const":
                x = Lin(np.full((g.n, T), float(ref[1])))
            else:
                x = self._expr(Ref(ref[0], ref[1]), g, False)
            idx = np.arange(g.n) if lis is None else np.array(lis)
            if how in ("sum", "mean", "value"):
                V = x.V[idx].sum(0, keepdims=True)
                M = None
                if x.M is not None:
                    M = _rowmix(T, np.tile(np.arange(T), idx.size), (idx[:, None] * T + np.arange(T)[None, :]).ravel(),
                                np.ones(idx.size * T), x.M)
                vals.append(Lin(V, M, x.lvl))
            else:
                for i in idx:
                    vals.append(self._rows(x, i))
        if not vals:
            return Lin(np.zeros((1, T)))
        out = vals[0]
        for v in vals[1:]:
            if how in ("sum", "mean", "value"):
                out = Lin(out.V + v.V, _add(out.M, v.M), max(out.lvl, v.lvl))
            elif how == "max":
                out = _select(out.V >= v.V, out, v, PWA)
            else:
                out = _select(out.V <= v.V, out, v, PWA)
        if how == "mean":
            cnt = sum(len(lis) if lis is not None else self.prog.gmap[gid].n for gid, ref, lis in parts)
            out = Lin(out.V / max(cnt, 1), _scale(out.M, 1.0 / max(cnt, 1)), out.lvl)
        return out

    def _rows(self, x: Lin, i: int) -> Lin:
        """Instance i of a node as a [1, T] form."""
        T = self.T
        M = x.M[i * T:(i + 1) * T] if x.M is not None else None
        return Lin(x.V[i:i + 1].copy(), M, x.lvl)

    def _delay(self, gn, g) -> Lin:
        n, T = g.n, self.T
        src = self.L[f"{gn.gid}|{gn.expr.args[0].name}"]
        k = gn.expr.extra["k"]
        if isinstance(k, C):
            kk = np.full(n, int(round(k.value)))
        else:
            kv = self._expr(k, g, False).V[:, 0]
            kk = np.maximum(np.round(kv).astype(np.int64), 1)
        init = self._expr(gn.expr.extra["init"], g, False)
        t = np.arange(T)[None, :]
        take = t >= kk[:, None]                                       # [n, T]
        srcidx = np.where(take, np.arange(n)[:, None] * T + t - kk[:, None], 0)
        V = np.where(take, src.V.ravel()[srcidx], init.V)
        M = None
        if src.M is not None:
            r = np.nonzero(take.ravel())[0]
            M = _rowmix(n * T, r, srcidx.ravel()[r], np.ones(r.size), src.M)
        if init.M is not None:
            M = _add(M, _scale(init.M, ~take))
        return Lin(V, M, max(src.lvl, init.lvl))

    # -------------------------------------------------------------- expressions
    def _ref(self, e: Ref, g, final: bool) -> Lin:
        n, T = g.n, self.T
        k, name = e.kind, e.name
        if k == "param" and g.is_env or k == "env":
            return Lin(_np(self.Et[name], 1, T)[:, -1:] if final else _np(self.Et[name], n, T))
        if k == "param":
            v = self.pb.P[g.gid].get(name)
            if v is None:
                raise ValueError(f"parameter {g.gid}.{name} has no numeric value")
            return Lin(_np(v, n, 1 if final else T))
        if k == "dec":
            x = self.D[(g.gid, name)]
            return self._last(x, n) if final else x
        if k == "b":
            if name == "dt":
                return Lin(np.full((n, 1 if final else T), self.dt))
            tt = np.arange(T, dtype=np.float64)[None, :].repeat(n, 0)
            tm = self.pb.t0 + tt * self.dt
            v = {"t": tt, "time": tm, "hour": tm % 24.0, "dow": np.floor(tm / 24.0) % 7.0}[name]
            return Lin(v[:, -1:].copy() if final else v)
        nid = {"node": f"{g.gid}|{name}", "in": f"{g.gid}|in:{name}", "state": f"{g.gid}|state:{name}",
               "agg": f"env|agg:{name}"}[k]
        if final and k == "state":
            return self._final_state(nid, g)
        x = self.L[nid]
        if k == "agg" and n != 1:
            M = x.M[np.tile(np.arange(T), n)] if x.M is not None else None
            x = Lin(np.repeat(x.V, n, 0), M, x.lvl)
        return self._last(x, n) if final else x

    def _last(self, x: Lin, n: int) -> Lin:
        T = self.T
        M = x.M[np.arange(n) * T + T - 1] if x.M is not None else None
        return Lin(x.V[:, -1:].copy(), M, x.lvl)

    def _final_state(self, nid: str, g) -> Lin:
        n, T = g.n, self.T
        gn = self.comp.nodes[nid]
        if nid in self.lifted:
            off = self.lifted[nid][0]
            c = off + np.arange(n) * T + T - 1
            M = sp.csr_matrix((np.ones(n), (np.arange(n), c)), shape=(n, self.nvar))
            return Lin(_np(self.raw[gn.var + "_f"], n, 1), M, AFFINE)
        S = self.pb._states(self.pb.S0)[gn.gid][gn.key.split(":", 1)[1]]
        return Lin(_np(S, n, T)[:, -1:].copy())

    def _expr(self, e, g, final: bool) -> Lin:
        n, T = g.n, (1 if final else self.T)
        if isinstance(e, C):
            return Lin(np.full((n, T), float(e.value)))
        if isinstance(e, Ref):
            return self._ref(e, g, final)
        op = e.op
        if op == "pwl":
            return self._pwl(e, g, final)
        a = [self._expr(x, g, final) for x in e.args]
        if op == "add":
            return Lin(a[0].V + a[1].V, _add(a[0].M, a[1].M), _lvl(a[0], a[1]))
        if op == "sub":
            return Lin(a[0].V - a[1].V, _add(a[0].M, _scale(a[1].M, -1.0)), _lvl(a[0], a[1]))
        if op == "neg":
            return Lin(-a[0].V, _scale(a[0].M, -1.0), a[0].lvl)
        if op in ("mul", "and"):
            return _mul(a[0], a[1])
        if op == "not":
            return Lin(1.0 - a[0].V, _scale(a[0].M, -1.0), a[0].lvl)
        if op in ("sum", "mean"):
            out = a[0]
            for x in a[1:]:
                out = Lin(out.V + x.V, _add(out.M, x.M), _lvl(out, x))
            if op == "mean":
                out = Lin(out.V / len(a), _scale(out.M, 1.0 / len(a)), out.lvl)
            return out
        if op == "div":
            b = a[1].V
            bb = np.where(np.abs(b) < 1e-12, 1e-12 * np.where(b < 0, -1.0, 1.0), b)
            V = a[0].V / bb
            if not a[1].moves:
                return Lin(V, _scale(a[0].M, 1.0 / bb), _lvl(a[0], a[1]))
            return Lin(V, _add(_scale(a[0].M, 1.0 / bb), _scale(a[1].M, -V / bb)), NONLIN)
        if op == "safediv":
            b = a[1].V
            bb = np.maximum(b, 1e-9)
            V = a[0].V / bb
            if not a[1].moves:
                return Lin(V, _scale(a[0].M, 1.0 / bb), _lvl(a[0], a[1]))
            return Lin(V, _add(_scale(a[0].M, 1.0 / bb), _scale(a[1].M, np.where(b > 1e-9, -V / bb, 0.0))), NONLIN)
        if op == "pow":
            x, p = a
            if not p.moves:
                pv = p.V
                if np.all(pv == np.round(pv)):
                    V = x.V ** pv
                    d = pv * x.V ** (pv - 1) if np.any(pv != 0) else np.zeros_like(x.V)
                else:
                    xp = np.maximum(x.V, 0.0)
                    V = xp ** pv
                    with np.errstate(divide="ignore", invalid="ignore"):
                        d = np.where(x.V > 0, pv * xp ** (pv - 1), 0.0)
                lv = x.lvl if np.all(pv == 1) else (0 if np.all(pv == 0) else NONLIN)
                return Lin(V, _scale(x.M, d), lv if x.moves else 0)
            xp = np.maximum(x.V, 1e-300)
            V = np.maximum(x.V, 0.0) ** p.V
            return Lin(V, _add(_scale(x.M, p.V * xp ** (p.V - 1)), _scale(p.M, V * np.log(xp))), NONLIN)
        if op == "mod":
            return Lin(np.mod(a[0].V, a[1].V), a[0].M, _lvl(a[0], a[1], at_least=NONLIN))
        if op == "or":
            s = Lin(a[0].V + a[1].V, _add(a[0].M, a[1].M), _lvl(a[0], a[1]))
            return _select(s.V <= 1.0, s, Lin(np.ones_like(s.V)), PWA)
        if op in ("lt", "le", "gt", "ge", "eq", "ne"):
            f = {"lt": np.less, "le": np.less_equal, "gt": np.greater, "ge": np.greater_equal, "eq": np.equal,
                 "ne": np.not_equal}[op]
            return Lin(f(a[0].V, a[1].V).astype(np.float64), None, _lvl(a[0], a[1], at_least=NONLIN))
        if op in ("min", "max"):
            out = a[0]
            for x in a[1:]:
                out = self._max(out, x, g) if op == "max" else self._min(out, x, g)
            return out
        if op == "clamp":
            return self._min(self._max(a[0], a[1], g), a[2], g)
        if op == "where":
            c = a[0]
            out = _select(c.V > 0.5, a[1], a[2], 0)
            if c.moves:
                out.lvl = NONLIN
            return out
        x = a[0]
        V = x.V
        if op == "pos":
            return self._max(x, Lin(np.zeros_like(V)), g)
        if op == "negpart":
            return self._min(x, Lin(np.zeros_like(V)), g)
        if op == "abs":
            return self._max(x, Lin(-V, _scale(x.M, -1.0), x.lvl), g)
        if op == "sqrt":
            s = np.sqrt(np.maximum(V, 0.0))
            return _unary(x, s, np.where(V > 1e-12, 0.5 / np.maximum(s, 1e-300), 0.0), NONLIN)
        if op == "exp":
            ev = np.exp(np.minimum(V, 80.0))
            return _unary(x, ev, np.where(V < 80.0, ev, 0.0), NONLIN)
        if op in ("log", "log10"):
            c = 1.0 if op == "log" else 1.0 / np.log(10.0)
            vv = np.maximum(V, 1e-12)
            return _unary(x, np.log(vv) * c, np.where(V > 1e-12, c / vv, 0.0), NONLIN)
        if op == "sin":
            return _unary(x, np.sin(V), np.cos(V), NONLIN)
        if op == "cos":
            return _unary(x, np.cos(V), -np.sin(V), NONLIN)
        if op == "tanh":
            th = np.tanh(V)
            return _unary(x, th, 1.0 - th * th, NONLIN)
        if op == "sigmoid":
            s = 1.0 / (1.0 + np.exp(-V))
            return _unary(x, s, s * (1.0 - s), NONLIN)
        if op in ("floor", "ceil", "round"):        # straight-through, as the simulator's gradient
            f = {"floor": np.floor, "ceil": np.ceil, "round": np.round}[op]
            return _unary(x, f(V), np.ones_like(V), NONLIN)
        if op in ("sign", "step"):
            v = np.sign(V) if op == "sign" else (V > 0).astype(np.float64)
            return Lin(v, None, _lvl(x, at_least=NONLIN))
        if op in ("smin", "smax"):
            k = float(a[2].V.ravel()[0])
            s = -1.0 if op == "smin" else 1.0
            za, zb = s * k * a[0].V, s * k * a[1].V
            m = np.maximum(za, zb)
            lse = m + np.log(np.exp(za - m) + np.exp(zb - m))
            wa = np.exp(za - lse)
            return Lin(s * lse / k, _add(_scale(a[0].M, wa), _scale(a[1].M, 1.0 - wa)), _lvl(a[0], a[1], at_least=NONLIN))
        raise ValueError(f"algebra: unsupported op {op}")

    # -------------------------------------------------------------- exact piecewise-affine terms
    def _box(self):
        """Lower / upper bound, reference value and integer steps of every column created so far."""
        k = len(self.aux_ref)
        if self._box_cache is None or self._box_cache[0] != k:
            nb = self.nbase
            lo = np.concatenate([self.lo[:nb], np.asarray(self.aux_lo, dtype=np.float64)])
            hi = np.concatenate([self.hi[:nb], np.asarray(self.aux_hi, dtype=np.float64)])
            lo[self.N:nb], hi[self.N:nb] = -np.inf, np.inf            # states and lifted delays: no box
            xr = np.concatenate([self.x_ref, np.asarray(self.aux_ref, dtype=np.float64)])
            st = np.concatenate([self.steps[:nb], np.asarray(self.aux_steps, dtype=np.int64)])
            self._box_cache = (k, lo, hi, xr, st)
        return self._box_cache[1:]

    def _interval(self, d: Lin):
        """Range of an affine form over the full box of its columns, per element, and whether it reads a
        discrete column. Infinite where it reads a state (no box)."""
        V = d.V.ravel()
        if d.M is None:
            return V.copy(), V.copy(), np.zeros(V.size, dtype=bool)
        M = d.M.tocsr()
        lo, hi, xr, st = self._box()
        c, v = M.indices, M.data
        with np.errstate(invalid="ignore"):
            a1, a2 = v * (lo[c] - xr[c]), v * (hi[c] - xr[c])
        rows = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
        L = V + np.bincount(rows, weights=np.minimum(a1, a2), minlength=M.shape[0])
        U = V + np.bincount(rows, weights=np.maximum(a1, a2), minlength=M.shape[0])
        disc = np.bincount(rows, weights=(st[c] > 0).astype(np.float64), minlength=M.shape[0]) > 0
        return L, U, disc

    def _max(self, a: Lin, b: Lin, g) -> Lin:
        """max(a, b). Where the difference reads a discrete decision and only boxed columns, and the kink lies
        inside its range, the term is exact: a column y = max(a, b) with an indicator z (a >= b) and big-Ms
        from the range - y >= a, y >= b, y <= a + Mba (1 - z), y <= b + Mab z. Elsewhere first order."""
        out = _select(a.V >= b.V, a, b, PWA)
        self._call += 1
        if not self.exact_pwa or not (a.moves or b.moves):
            return out
        dM = _add(a.M, _scale(b.M, -1.0))
        if dM is None:
            return out
        d = Lin(a.V - b.V, dM, AFFINE)
        L, U, disc = self._interval(d)
        if a.lvl > AFFINE or b.lvl > AFFINE:            # an argument is itself first order: epigraph only (the kink exact)
            fixed = (L >= -1e-9) | (U <= 1e-9)
            return out if fixed.all() else self._epigraph(a, b, g, out, ~fixed)
        fixed = (L >= -1e-9) | (U <= 1e-9)
        need = disc & ~fixed & np.isfinite(L) & np.isfinite(U)
        if not need.any():
            if fixed.all():                                      # the same piece over the whole box: exact
                out.lvl = max(a.lvl, b.lvl)
                return out
            return self._epigraph(a, b, g, out, ~fixed)
        idx = np.nonzero(need)[0]
        ne = idx.size
        n, TT = a.V.shape
        La, Ua, _ = self._interval(a)
        Lb, Ub, _ = self._interval(b)
        av, bv = a.V.ravel()[idx], b.V.ravel()[idx]
        y0 = np.maximum(av, bv)
        z0 = (av >= bv).astype(np.float64)
        yc = self.ncols + np.arange(ne)
        zc = self.ncols + ne + np.arange(ne)
        if zc[-1] >= self.nvar:
            raise ValueError("algebra: too many exact piecewise terms")
        ids = np.array(g.inst_ids if not g.is_env else [-1])
        inst = ids[idx // TT]
        tt = idx % TT
        self.aux_ref += list(y0) + list(z0)
        self.aux_lo += list(np.maximum(La[idx], Lb[idx])) + [0.0] * ne
        self.aux_hi += list(np.maximum(Ua[idx], Ub[idx])) + [1.0] * ne
        self.aux_steps += [0] * ne + [1] * ne
        self.aux_inst += list(inst) * 2
        self.aux_t += list(tt) * 2
        self.aux_kind += [4] * ne + [5] * ne
        self.ncols += 2 * ne
        Mab, Mba = U[idx], -L[idx]
        r = np.arange(ne)
        unit = lambda cols, val: sp.csr_matrix((np.broadcast_to(val, (ne,)).astype(np.float64), (r, cols)),  # noqa: E731
                                               shape=(ne, self.nvar))
        Ar = a.M[idx] if a.M is not None else sp.csr_matrix((ne, self.nvar))
        Br = b.M[idx] if b.M is not None else sp.csr_matrix((ne, self.nvar))
        rows = [(Ar - unit(yc, 1.0), av - y0),                                   # a - y <= 0
                (Br - unit(yc, 1.0), bv - y0),                                   # b - y <= 0
                (unit(yc, 1.0) - Ar + unit(zc, Mba), y0 - av + Mba * z0 - Mba),  # y - a - Mba (1 - z) <= 0
                (unit(yc, 1.0) - Br - unit(zc, Mab), y0 - bv - Mab * z0)]        # y - b - Mab z <= 0
        for A_, r0 in rows:
            self.aux_rows.append((A_.tocsr(), r0, inst, tt, g.gid))
        keep = np.ones(n * TT)
        keep[idx] = 0.0
        M = _add(_scale(out.M, keep), sp.csr_matrix((np.ones(ne), (idx, yc)), shape=(n * TT, self.nvar)))
        V = out.V.copy().ravel()
        V[idx] = y0
        lv = max(a.lvl, b.lvl) if (fixed | need).all() else PWA
        return Lin(V.reshape(n, TT), M, lv)

    def _epigraph(self, a: Lin, b: Lin, g, out: Lin, where: np.ndarray) -> Lin:
        """max(a, b) as a column y with y >= a, y >= b only - exact where every use of the term prefers it small (a
        cost, a <= limit): the optimum then sits on the max. Which sites qualify is decided on the assembled model
        (`_favourable`, two passes); in the probing pass every candidate is lifted, elsewhere it stays first order."""
        site = (self._cur_key, self._call)
        if self.epi_mode == "off" or (self.epi_mode == "use" and site not in self.epi_allowed):
            return out
        idx = np.nonzero(where)[0]
        ne = idx.size
        n, TT = a.V.shape
        av, bv = a.V.ravel()[idx], b.V.ravel()[idx]
        y0 = np.maximum(av, bv)
        yc = self.ncols + np.arange(ne)
        if yc[-1] >= self.nvar:
            raise ValueError("algebra: too many exact piecewise terms")
        ids = np.array(g.inst_ids if not g.is_env else [-1])
        inst, tt = ids[idx // TT], idx % TT
        La, Ua, _ = self._interval(a)
        Lb, Ub, _ = self._interval(b)
        self.aux_ref += list(y0)
        self.aux_lo += list(np.maximum(La[idx], Lb[idx]))
        self.aux_hi += [np.inf] * ne
        self.aux_steps += [0] * ne
        self.aux_inst += list(inst)
        self.aux_t += list(tt)
        self.aux_kind += [6] * ne
        self.ncols += ne
        self.epi_cols[site] = yc
        r = np.arange(ne)
        unit = sp.csr_matrix((np.ones(ne), (r, yc)), shape=(ne, self.nvar))
        Ar = a.M[idx] if a.M is not None else sp.csr_matrix((ne, self.nvar))
        Br = b.M[idx] if b.M is not None else sp.csr_matrix((ne, self.nvar))
        for A_, r0 in ((Ar - unit, av - y0), (Br - unit, bv - y0)):
            self.aux_rows.append((A_.tocsr(), r0, inst, tt, g.gid))
            self.epi_rows.setdefault(site, []).append(len(self.aux_rows) - 1)
        keep = np.ones(n * TT)
        keep[idx] = 0.0
        M = _add(_scale(out.M, keep), sp.csr_matrix((np.ones(ne), (idx, yc)), shape=(n * TT, self.nvar)))
        V = out.V.copy().ravel()
        V[idx] = y0
        return Lin(V.reshape(n, TT), M, max(a.lvl, b.lvl) if where.all() or (~where).sum() == 0 else PWA)

    def _favourable(self, m: "LinearModel", aux_row_start: list) -> set:
        """Sites whose lifted column is only ever used where smaller is better and linearly: objective coefficient
        <= 0, coefficient >= 0 in every <= row (its own epigraph rows aside), absent from equalities and from rows that
        are not affine."""
        Jc = m.J.tocsc()
        ok = set()
        for site, cols in self.epi_cols.items():
            own = set()
            for k in self.epi_rows.get(site, []):
                st, ln = aux_row_start[k]
                own.update(range(st, st + ln))
            good = bool(np.all(m.g[cols] <= 1e-12))
            for c in cols:
                if not good:
                    break
                rows = Jc.indices[Jc.indptr[c]:Jc.indptr[c + 1]]
                vals = Jc.data[Jc.indptr[c]:Jc.indptr[c + 1]]
                for i, v in zip(rows, vals):
                    if i in own:
                        continue
                    if m.eq[i] or v < 0 or m.row_lvl[i] > AFFINE:
                        good = False
                        break
            if good:
                ok.add(site)
        return ok

    def _min(self, a: Lin, b: Lin, g) -> Lin:
        r = self._max(Lin(-a.V, _scale(a.M, -1.0), a.lvl), Lin(-b.V, _scale(b.M, -1.0), b.lvl), g)
        return Lin(-r.V, _scale(r.M, -1.0), r.lvl)

    def _pwl(self, e: Op, g, final: bool) -> Lin:
        xs = np.asarray(e.extra["xs"], dtype=np.float64)
        x = self._expr(e.args[0], g, final)
        ys = [self._expr(y, g, final) for y in e.args[1:]]
        Y = np.stack([y.V for y in ys])                                # [k, n, T]
        xc = np.clip(x.V, xs[0], xs[-1])
        i = np.clip(np.searchsorted(xs, xc, side="right") - 1, 0, len(xs) - 2)
        x0, x1 = xs[i], xs[i + 1]
        w = np.clip((xc - x0) / np.maximum(x1 - x0, 1e-12), 0.0, 1.0)
        y0 = np.take_along_axis(Y, i[None], 0)[0]
        y1 = np.take_along_axis(Y, (i + 1)[None], 0)[0]
        V = (1 - w) * y0 + w * y1
        tol = 1e-6 * max(abs(xs[0]), abs(xs[-1]), 1.0)
        inside = (x.V >= xs[0] - tol) & (x.V <= xs[-1] + tol)
        slope = np.where(inside, (y1 - y0) / np.maximum(x1 - x0, 1e-12), 0.0)
        M = _scale(x.M, slope)
        lv = _lvl(x, at_least=PWA) if x.moves else 0
        for j, y in enumerate(ys):
            if y.moves:
                cj = np.where(i == j, 1 - w, 0.0) + np.where(i + 1 == j, w, 0.0)
                M = _add(M, _scale(y.M, cj))
                lv = NONLIN if x.moves else max(lv, y.lvl)
        return Lin(V, M, lv)

    # -------------------------------------------------------------- assembly
    def run(self, x) -> LinearModel:
        t0 = time.perf_counter()
        if not self.exact_pwa or not self.epigraph:
            self.epi_mode = "off"
            return self._run(x, t0)
        if self.epi_allowed is None:                  # once per program: which max() sites have an exact epigraph
            self.epi_mode = "probe"
            mp = self._run(x, t0)
            self.epi_allowed = self._favourable(mp, self._aux_row_start)
            self.epi_report = {"candidates": len(self.epi_cols), "exact": len(self.epi_allowed)}
        self.epi_mode = "use"
        return self._run(x, t0)

    def _run(self, x, t0) -> LinearModel:
        pb, T = self.pb, self.T
        x = torch.as_tensor(x, dtype=pb.prog.dtype).reshape(1, -1)
        x = pb.pinned(x)
        self.raw = self._reference(x)
        self.Et = self.prog.env_tensors(pb.E, T)
        u = x.reshape(-1).double().numpy()
        self._layout(self.raw, u)
        self.nbase = self.nvar
        self.nvar = self.nbase + AUXCAP
        self.ncols = self.nbase
        self.aux_ref, self.aux_lo, self.aux_hi, self.aux_steps = [], [], [], []
        self.aux_inst, self.aux_t, self.aux_kind, self.aux_rows = [], [], [], []
        self._box_cache = None
        self._cur_key, self._call, self.epi_cols, self.epi_rows = "", 0, {}, {}
        self._decisions(u)
        self.L: dict = {}
        for nid in self.order:
            self.L[nid] = self._node(nid)
        # objective: d objective / d term total, then the terms' Jacobians
        gvec = np.zeros(self.nvar)
        f0 = 0.0
        coef = self._objective_coefficients()
        inexact_cols = []                                 # columns of objective terms that are not affine
        for tr in self.prog.terms:
            c = coef.get((tr.kind, tr.name), 0.0)
            if c == 0.0:
                continue
            gg = self.prog.gmap[tr.gid]
            if tr.final:
                self._cur_key, self._call = f"final|{tr.gid}|{tr.key}", 0
                lin = self._expr(gg.ir.finals[tr.key], gg, True)
                f0 += c * float(lin.V.sum())
                if lin.M is not None:
                    np.add.at(gvec, lin.M.indices, c * lin.M.data)
            else:
                lin = self.L[f"{tr.gid}|{tr.key}"]
                f0 += c * float(lin.V.sum()) * self.dt
                if lin.M is not None:
                    np.add.at(gvec, lin.M.indices, c * self.dt * lin.M.data)
            if lin.M is not None and lin.lvl > AFFINE:
                inexact_cols.append(lin.M.indices)
        f0 += self._objective_const
        if pb.move_ref is not None:          # move suppression, linearised (quadratic in the decisions)
            d = u - pb.move_ref.reshape(-1).double().numpy()
            wm = pb.move_w.reshape(-1).double().numpy()
            f0 -= pb.move_cost * float((wm * d * d).sum())
            gvec[:self.N] -= 2.0 * pb.move_cost * wm * d
        J, r0, eq, w, mu, rinst, rt_, rkind, rcon, rlvl = [], [], [], [], [], [], [], [], [], []
        cons: list = []

        def add_rows(lin: Lin, gid: str, kind: int, is_eq: bool, wsoft, mup, tag) -> None:
            g = self.prog.gmap[gid]
            n, TT = lin.V.shape
            M = lin.M if lin.M is not None else sp.csr_matrix((n * TT, self.nvar))
            J.append(M)
            r0.append(lin.V.ravel())
            eq.append(np.full(n * TT, is_eq))
            w.append(np.broadcast_to(np.asarray(wsoft, dtype=np.float64).reshape(-1, 1), (n, TT)).ravel()
                     if np.ndim(wsoft) else np.full(n * TT, float(wsoft)))
            mu.append(np.broadcast_to(np.asarray(mup, dtype=np.float64).reshape(-1, 1), (n, TT)).ravel()
                      if np.ndim(mup) else np.full(n * TT, float(mup)))
            ids = np.array(g.inst_ids if not g.is_env else [-1])
            rinst.append(np.repeat(ids, TT))
            rt_.append(np.tile(np.arange(TT), n))
            rkind.append(np.full(n * TT, kind, dtype=np.int8))
            rcon.append(np.full(n * TT, len(cons)))
            rlvl.append(np.full(n * TT, lin.lvl, dtype=np.int8))
            cons.append(tag)

        # state updates and lifted delays
        for nid, (off, n, TT, gid) in self.lifted.items():
            gn = self.comp.nodes[nid]
            ident = self._ident(nid, n, False)
            Vs = self.x_ref[off:off + n * TT].reshape(n, TT)
            if gn.kind == "state":
                nxt = self.L[f"{gid}|next:{gn.key.split(':', 1)[1]}"]
                lin = Lin(Vs - nxt.V, _add(ident, _scale(nxt.M, -1.0)), max(AFFINE, nxt.lvl))
                add_rows(lin, gid, 0, True, 0.0, 0.0, ("state", gid, gn.key))
            else:
                self._cur_key, self._call = f"delayrow|{nid}", 0
                d = self._delay(gn, self.prog.gmap[gid])
                lin = Lin(Vs - d.V, _add(ident, _scale(d.M, -1.0)), max(AFFINE, d.lvl))
                add_rows(lin, gid, 1, True, 0.0, 0.0, ("delay", nid))
        # constraints
        bnd = self.prog._bounds_cache()
        for cr in self.prog.cons:
            lin = self.L[f"{cr.gid}|{cr.key}"]
            g = self.prog.gmap[cr.gid]
            if pb.backoff is not None and (cr.gid, cr.key) in pb.backoff:
                s = pb.con_scale.get((cr.gid, cr.key))
                bo = _np(pb.backoff[(cr.gid, cr.key)], g.n, T) * (_np(s, g.n, T) if s is not None else 1.0)
                lin = Lin(lin.V + bo, lin.M, lin.lvl)
            if cr.soft:
                wv = bnd[cr.gid].get(f"soft:{cr.key}", 0.0)
                wv = _np(wv, g.n, 1)[:, 0] * self.dt
                add_rows(lin, cr.gid, 3, cr.kind == "eq", wv, 0.0, (cr.gid, cr.key))
            else:
                s = pb.con_scale.get((cr.gid, cr.key))
                sc = _np(s, g.n, 1)[:, 0] if s is not None else np.ones(g.n)
                add_rows(lin, cr.gid, 2, cr.kind == "eq", 0.0, MU / pb.T / np.maximum(sc, 1e-12), (cr.gid, cr.key))
        self._aux_row_start = []
        nrow = sum(len(x_) for x_ in r0)
        for A_, r0_, inst_, t_, gid_ in self.aux_rows:          # definitions of the exact piecewise terms
            k = len(r0_)
            self._aux_row_start.append((nrow, k))
            nrow += k
            J.append(A_)
            r0.append(np.asarray(r0_, dtype=np.float64))
            eq.append(np.zeros(k, dtype=bool))
            w.append(np.zeros(k))
            mu.append(np.zeros(k))
            rinst.append(np.asarray(inst_, dtype=np.int64))
            rt_.append(np.asarray(t_, dtype=np.int64))
            rkind.append(np.full(k, 4, dtype=np.int8))
            rcon.append(np.full(k, len(cons)))
            rlvl.append(np.full(k, AFFINE, dtype=np.int8))
            cons.append(("pwa", gid_))
        nb, nf = self.nbase, self.ncols
        Jm = sp.vstack(J).tocsr()[:, :nf] if J else sp.csr_matrix((0, nf))
        gvec = gvec[:nf]
        lo = np.concatenate([self.lo[:nb], np.asarray(self.aux_lo, dtype=np.float64)])
        hi = np.concatenate([self.hi[:nb], np.asarray(self.aux_hi, dtype=np.float64)])
        lo[self.N:nb], hi[self.N:nb] = -np.inf, np.inf
        steps = np.concatenate([self.steps[:nb], np.asarray(self.aux_steps, dtype=np.int64)])
        x_ref = np.concatenate([self.x_ref, np.asarray(self.aux_ref, dtype=np.float64)])
        vinst = np.concatenate([self.var_inst, np.asarray(self.aux_inst, dtype=np.int64)])
        vt = np.concatenate([self.var_t, np.asarray(self.aux_t, dtype=np.int64)])
        vk = np.concatenate([self.var_kind, np.asarray(self.aux_kind, dtype=np.int8)])
        if pb.pin_mask is not None:
            pm = pb.pin_mask.reshape(-1).numpy() > 0
            pv = pb.pin_val.reshape(-1).double().numpy()
            lo[:self.N][pm] = pv[pm]
            hi[:self.N][pm] = pv[pm]
        cat = lambda xs, dt=None: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dtype=dt)  # noqa: E731
        m = LinearModel(nf, self.N, x_ref, lo, hi, steps, vinst, vt, vk,
                        Jm, cat(r0, np.float64), cat(eq, bool), cat(w, np.float64), cat(mu, np.float64),
                        cat(rinst, np.int64), cat(rt_, np.int64), cat(rkind, np.int8), cat(rcon, np.int64),
                        cat(rlvl, np.int8), cons, gvec, f0, pb.J_scale, dict(self.lifted))
        lv = m.row_lvl > AFFINE
        bad = np.zeros(nf, dtype=bool)
        if lv.any():
            Jb = Jm[np.nonzero(lv)[0]]
            bad[Jb.indices] = True
        for ic in inexact_cols:
            bad[ic[ic < nf]] = True
        m.col_exact = ~bad
        m.seconds = time.perf_counter() - t0
        return m

    def _objective_coefficients(self) -> dict:
        """d objective / d (term total) for every (kind, name), from the objective expression (totals are floats)."""
        prog = self.prog
        keys = sorted({(tr.kind, tr.name) for tr in prog.terms})
        kinds = ("revenue", "cost", "emission", "metric")

        def obj(bump=None):
            totals = {k: 0.0 for k in kinds}
            named = {nm: 0.0 for _, nm in keys}
            if bump is not None:
                totals[bump[0]] += 1.0
                named[bump[1]] = named.get(bump[1], 0.0) + 1.0
            E = {k: v for k, v in (self.pb.E or {}).items() if not isinstance(v, (torch.Tensor, list, tuple))}
            return float(prog._objective(totals, named, 0.0, E))

        base = obj()
        self._objective_const = base
        return {k: obj(k) - base for k in keys}


def linearize(pb, x=None) -> LinearModel:
    """The twin's equations around plan x (canonical decisions; default: the default plan) as a LinearModel."""
    if x is None:
        x = pb.space.default_u(1)
    lz = getattr(pb, "_linearizer", None)
    if lz is None or lz.prog is not pb.prog:
        lz = Linearizer(pb)
        pb._linearizer = lz
    return lz.run(x)


def lift(pb, m: LinearModel, x) -> np.ndarray:
    """The full column vector of plan x: its decisions and the trajectories the simulator computes from them."""
    lz = pb._linearizer
    x = torch.as_tensor(x, dtype=pb.prog.dtype).reshape(1, -1)
    if m.nvar > m.N + sum(n * T for (_, n, T, _) in m.lifted.values()):
        return lz.run(x).x_ref
    raw = lz._reference(pb.pinned(x))
    out = np.zeros(m.nvar)
    out[:m.N] = pb.pinned(x).reshape(-1).double().numpy()
    for nid, (off, n, T, gid) in m.lifted.items():
        gn = lz.comp.nodes[nid]
        if gn.kind == "state":
            traj = _np(raw[gn.var], n, T)
            fin = _np(raw[gn.var + "_f"], n, 1)
            out[off:off + n * T] = np.concatenate([traj[:, 1:], fin], 1).ravel()
        else:
            out[off:off + n * T] = _np(raw[gn.var], n, T).ravel()
    return out


def twin_rows(pb, m: LinearModel, x) -> tuple:
    """(row values, objective incl. soft penalties) as the simulator computes them at plan x - the reference the
    model is checked against. State and delay rows are 0 by construction of the simulation."""
    xf = lift(pb, m, x)
    with torch.no_grad():
        e = pb.evaluate(torch.as_tensor(x, dtype=pb.prog.dtype).reshape(1, -1), relax=True)
    out = np.zeros(m.J.shape[0])
    for ci, tag in enumerate(m.cons):
        if tag[0] in ("state", "delay", "pwa"):
            continue
        sel = m.row_con == ci
        gid, ck = tag
        g = pb.prog.gmap[gid]
        h = _np(e.raw[pb.prog.node_var(gid, ck)], g.n, pb.T)
        if pb.backoff is not None and tag in pb.backoff:
            s = pb.con_scale.get(tag)
            h = h + _np(pb.backoff[tag], g.n, pb.T) * (_np(s, g.n, pb.T) if s is not None else 1.0)
        out[sel] = h.ravel()
    return out, float(e.objective.reshape(-1)[0]), xf


def check(pb, m: LinearModel, x=None, eps: float = 1e-3, seed: int = 0) -> dict:
    """The model against the simulator: values at the reference, and a step of size eps along a random direction
    of the continuous decisions (the model's prediction vs the simulation)."""
    xr = torch.as_tensor(m.x_ref[:m.N], dtype=pb.prog.dtype)
    r_tw, f_tw, xf = twin_rows(pb, m, xr)
    hard = (m.row_kind == 2) | (m.row_kind == 3)
    scale = np.maximum(np.abs(r_tw), 1.0)
    at_ref = {"objective_model": m.objective_at(m.x_ref), "objective_twin": f_tw,
              "rows_max_rel_err": float((np.abs(m.r0[hard] - r_tw[hard]) / scale[hard]).max(initial=0.0)),
              "state_rows_max_abs": float(np.abs(m.r0[m.row_kind <= 1]).max(initial=0.0)),
              "columns_max_abs_err": float(np.abs(xf - m.x_ref).max(initial=0.0))}
    gen = np.random.default_rng(seed)
    d = np.zeros(m.N)
    free = (m.steps[:m.N] == 0) & (m.hi[:m.N] > m.lo[:m.N])
    d[free] = gen.uniform(-1, 1, int(free.sum()))
    u1 = np.clip(m.x_ref[:m.N] + eps * d, 0.0, 1.0)
    r1, f1, xf1 = twin_rows(pb, m, torch.as_tensor(u1, dtype=pb.prog.dtype))
    pred = m.rows_at(xf1)
    dr_tw = r1 - r_tw
    dr_m = pred - m.r0
    lvl = m.row_lvl
    out = {"at_reference": at_ref, "step": eps}
    for name, sel in (("affine", lvl <= AFFINE), ("piecewise_affine", lvl == PWA), ("nonlinear", lvl == NONLIN)):
        sel = sel & hard
        if not sel.any():
            continue
        num = np.abs(dr_m[sel] - dr_tw[sel])
        den = np.abs(dr_tw[sel]).max(initial=0.0) + 1e-9
        out[f"rows_{name}"] = {"rows": int(sel.sum()), "max_err_rel_to_max_change": float(num.max() / den)}
    st = m.row_kind <= 1
    out["state_rows_after_step_max_abs"] = float(np.abs(pred[st]).max(initial=0.0))
    df_tw = f1 - f_tw
    df_m = m.objective_at(xf1) - m.objective_at(m.x_ref)
    out["objective_change"] = {"twin": df_tw, "model": df_m}
    return out
