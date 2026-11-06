"""The reduction engine: structure found in the twin's equations, solved piece by piece, coordinated by prices.

Input is the algebraic model of a twin (algebra.py) - rows and columns, each owned by an asset (an atom). The math
graph links two atoms when a row touches both. A bank of filters contracts that graph level by level, each
contraction turning a structure into one node with an exact algorithm:

    lp-cell     atoms without discrete decisions, connected: a linear program of any size (polynomial, exact)
    line/cell   discrete atoms coupled directly, merged along their strongest coupling (heavy-edge matching)
                while the merged block's integer count stays under `int_cap`: an exact MILP of bounded size
    recourse    an LP cell that touches exactly one discrete block joins it (its continuous recourse)
    site        an LP cell (a hub: a shared yard, a grid connection) whose discrete neighbours fit under the cap
                together joins them into one block - the hub becomes internal
    master      LP cells shared by more blocks than fit together (markets, group-wide limits) stay in the master,
                exact: a linear program over their columns and the blocks' convex weights

The filters run again on the contracted graph until nothing changes, so lines become sites become regions as far
as the exact algorithms stay fast. Coordination is Dantzig-Wolfe: the master LP prices every row that couples
blocks (its duals), each block answers with its exact best schedule at those prices (HiGHS MILP), the Lagrangian
bound (master value + every block's reduced cost) certifies how far any plan can be from the optimum of the
model. The commit chooses one schedule per block (restricted master MILP) and solves all continuous decisions
together exactly (one LP with the discrete ones fixed). Nothing in it knows what a plant, a line or a market is.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp

from .algebra import LinearModel
from .exact import Solution, model_F, solve


# ----------------------------------------------------------------------------- the math graph
@dataclass
class Block:
    name: str
    atoms: list
    cols: np.ndarray
    ints: int
    pattern: str
    level: int
    algo: str = "milp"
    rows: np.ndarray = None          # internal rows
    why: str = ""


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a
        return a


def math_graph(m: LinearModel):
    """atoms (instance ids, -1 = environment), column -> atom index, row -> atoms incidence, atom coupling W."""
    atoms = np.unique(np.concatenate([m.var_inst, m.row_inst]))
    aidx = {int(a): i for i, a in enumerate(atoms)}
    col_atom = np.searchsorted(atoms, m.var_inst)
    row_atom = np.searchsorted(atoms, m.row_inst)
    na = atoms.size
    P = sp.csr_matrix((np.ones(m.nvar), (np.arange(m.nvar), col_atom)), shape=(m.nvar, na))
    Jb = m.J.copy()
    Jb.data = np.ones_like(Jb.data)
    B = (Jb @ P).tocsr()
    B = (B + sp.csr_matrix((np.ones(m.J.shape[0]), (np.arange(m.J.shape[0]), row_atom)), shape=B.shape)).tocsr()
    B.data = np.ones_like(B.data)
    W = (B.T @ B).tocsr()
    W.setdiag(0)
    W.eliminate_zeros()
    return atoms, aidx, col_atom, row_atom, B, W


def contract(m: LinearModel, int_cap: int = 200, flat=None) -> dict:
    """Filters on the math graph, repeated until no structure changes. Returns blocks (discrete), the master's
    columns and a report of every level."""
    atoms, aidx, col_atom, row_atom, B, W = math_graph(m)
    na = atoms.size
    disc_col = m.steps > 0
    ints = np.bincount(col_atom[disc_col], minlength=na)
    ncols = np.bincount(col_atom, minlength=na)
    uf = _UF(na)
    report = []
    Wc = W.tocoo()
    # F1: continuous atoms connected through continuous atoms -> LP cells
    cont = ints == 0
    for a, b in zip(Wc.row, Wc.col):
        if cont[a] and cont[b]:
            uf.union(a, b)
    report.append({"level": 1, "filter": "lp-cell", "clusters": len({uf.find(i) for i in range(na)})})

    def clusters():
        root = np.array([uf.find(i) for i in range(na)])
        ids, inv = np.unique(root, return_inverse=True)
        k = ids.size
        Pm = sp.csr_matrix((np.ones(na), (np.arange(na), inv)), shape=(na, k))
        Wk = (Pm.T @ W @ Pm).tocsr()
        Wk.setdiag(0)
        Wk.eliminate_zeros()
        ci = np.bincount(inv, weights=ints, minlength=k).astype(np.int64)
        cc = np.bincount(inv, weights=ncols, minlength=k).astype(np.int64)
        return ids, inv, Wk, ci, cc

    level = 1
    while True:
        level += 1
        merged = {"discrete": 0, "recourse": 0, "site": 0}
        ids, inv, Wk, ci, cc = clusters()
        disc = ci > 0
        # F2: discrete clusters coupled directly, strongest coupling first, under the cap
        Wo = Wk.tocoo()
        order = np.argsort(-Wo.data)
        taken = set()
        for e in order:
            a, b = Wo.row[e], Wo.col[e]
            if a >= b or not (disc[a] and disc[b]) or a in taken or b in taken:
                continue
            if ci[a] + ci[b] <= int_cap:
                uf.union(ids[a], ids[b])
                taken |= {a, b}
                merged["discrete"] += 1
        ids, inv, Wk, ci, cc = clusters()
        disc = ci > 0
        # F3: an LP cell touching exactly one discrete block is its recourse; F5: a hub whose discrete neighbours
        # fit under the cap together joins them (smallest first)
        hubs = []
        for h in np.nonzero(~disc)[0]:
            nb = Wk.indices[Wk.indptr[h]:Wk.indptr[h + 1]]
            dn = nb[disc[nb]]
            if dn.size == 1:
                uf.union(ids[dn[0]], ids[h])
                merged["recourse"] += 1
            elif dn.size >= 2:
                hubs.append((int(ci[dn].sum()), h, dn))
        if not merged["recourse"]:
            used = set()
            for tot, h, dn in sorted(hubs, key=lambda z: z[0]):
                if tot > int_cap or used & set(dn.tolist()):
                    continue
                for d in dn:
                    uf.union(ids[h], ids[d])
                used |= set(dn.tolist())
                merged["site"] += 1
        n_after = len({uf.find(i) for i in range(na)})
        report.append({"level": level, "merges": merged, "clusters": n_after})
        if not any(merged.values()):
            break
    # closure: every definition row (state update, lifted delay) owned by a block's atom must lie inside it,
    # otherwise the block's states are free in its pricing problem (the Lagrangian relaxation is unbounded):
    # the atoms such a row reads move into the block (one by one: an LP cell is not swallowed whole), and two
    # blocks merge if they fit under the cap together
    ids, inv, Wk, ci, cc = clusters()
    assign = inv.copy()
    ci = ci.copy()
    Jc = m.J.tocsr()
    defrows = np.nonzero((m.row_kind <= 1) | (m.row_kind == 4))[0]
    moved = merged_b = 0
    changed = True
    while changed:
        changed = False
        for i in defrows:
            own = assign[row_atom[i]]
            if ci[own] == 0:
                continue
            cs = Jc.indices[Jc.indptr[i]:Jc.indptr[i + 1]]
            for a in np.unique(col_atom[cs]):
                k = assign[a]
                if k == own:
                    continue
                if ci[k] == 0:
                    assign[a] = own
                    moved += 1
                    changed = True
                elif ci[k] + ci[own] <= int_cap:
                    assign[assign == k] = own
                    ci[own] += ci[k]
                    ci[k] = 0
                    merged_b += 1
                    changed = True
    report.append({"level": level + 1, "filter": "closure", "atoms_moved": moved, "blocks_merged": merged_b})
    used = np.unique(assign)
    inv = np.searchsorted(used, assign)
    ids = used
    ci = np.bincount(inv, weights=ints, minlength=used.size).astype(np.int64)
    disc = ci > 0
    blocks = []
    master_cols = []
    for k in range(ids.size):
        mem = np.nonzero(inv == k)[0]
        cols = np.nonzero(np.isin(col_atom, mem))[0]
        if not disc[k]:
            master_cols.append(cols)
            continue
        inst = [int(atoms[i]) for i in mem]
        blocks.append(Block(_name(inst, flat), inst, cols, int(ci[k]), _pattern(mem, W, ints), level))
    master_cols = np.concatenate(master_cols) if master_cols else np.zeros(0, dtype=np.int64)
    # internal rows of every block: rows whose atoms all lie inside it
    blk_of_col = np.full(m.nvar, -1)
    for bi, b in enumerate(blocks):
        blk_of_col[b.cols] = bi
    Jc = m.J.tocsr()
    rb = np.full(m.J.shape[0], -2)              # -2: no block column; -1: master (several / master columns)
    for i in range(m.J.shape[0]):
        cs = Jc.indices[Jc.indptr[i]:Jc.indptr[i + 1]]
        if cs.size == 0:
            continue
        bs = np.unique(blk_of_col[cs])
        rb[i] = bs[0] if bs.size == 1 else -1
    for bi, b in enumerate(blocks):
        b.rows = np.nonzero(rb == bi)[0]
        b.algo = "milp" if b.ints else "lp"
    master_rows = np.nonzero(rb == -1)[0]
    return {"blocks": blocks, "master_cols": master_cols, "master_rows": master_rows, "row_block": rb,
            "levels": report, "atoms": int(na), "int_cap": int_cap}


def _pattern(mem, W, ints) -> str:
    if mem.size == 1:
        return "unit"
    sub = W[mem][:, mem]
    deg = np.diff(sub.tocsr().indptr)
    if (deg <= 2).all() and sub.nnz // 2 == mem.size - 1:
        return "line"
    return "cell"


def _name(inst, flat) -> str:
    if flat is None:
        return f"block[{','.join(str(i) for i in inst[:3])}{'...' if len(inst) > 3 else ''}]"
    names = [flat.instances[i].path if i >= 0 else "env" for i in inst]
    return names[0] + (f" +{len(names) - 1}" if len(names) > 1 else "")


# ----------------------------------------------------------------------------- coordination
@dataclass
class EngineResult:
    x: np.ndarray
    F: float                        # F_model of the committed plan
    bound: float                    # Lagrangian upper bound on F_model over the trust region
    lp_value: float                 # master LP value at the end
    rounds: int
    seconds: float
    pricing_seconds: float
    master_seconds: float
    commit_seconds: float
    columns: int
    blocks: int
    history: list = field(default_factory=list)

    @property
    def gap(self) -> float:
        return (self.bound - self.F) / max(1e-9, abs(self.bound))


class Engine:
    def __init__(self, m: LinearModel, int_cap: int = 200, flat=None):
        self.m = m
        self.st = contract(m, int_cap, flat)
        self.blocks: list = self.st["blocks"]

    def report(self) -> dict:
        b = self.blocks
        return {"atoms": self.st["atoms"], "levels": self.st["levels"], "blocks": len(b),
                "integer_columns": int(sum(x.ints for x in b)), "largest_block_ints": max((x.ints for x in b), default=0),
                "master_columns": int(self.st["master_cols"].size), "coupling_rows": int(self.st["master_rows"].size),
                "patterns": {p: sum(1 for x in b if x.pattern == p) for p in ("unit", "line", "cell")},
                "block_list": [{"name": x.name, "units": len(x.atoms), "ints": x.ints, "cols": int(x.cols.size),
                                "rows": int(x.rows.size), "pattern": x.pattern} for x in b]}

    # -------------------------------------------------------------- pieces
    def _cache(self) -> None:
        """Per block, once: its pricing sub-model (its columns; its own rows and the coupling rows it touches),
        its internal rows' matrix and its coupling matrix - so a round costs the blocks' sizes, not the model's."""
        if getattr(self, "_subs", None) is not None:
            return
        m, st = self.m, self.st
        mr = st["master_rows"]
        Jc = m.J.tocsc()
        self._Jr = m.J[mr].tocsr()
        self._subs = []
        for b in self.blocks:
            Jb = Jc[:, b.cols].tocsr()
            touch = np.nonzero(np.diff(Jb.indptr) > 0)[0]
            sub = _submodel(m, b.cols, touch, Jb[touch])
            internal = np.isin(touch, b.rows)
            self._subs.append((sub, touch, internal, Jb[b.rows], Jb[mr]))

    def _penalty(self, b: Block, x: np.ndarray) -> float:
        """Internal penalties of block b at x (USD): soft prices and the elastic price of its hard rows."""
        m = self.m
        self._cache()
        rows = b.rows
        Jint = self._subs[self.blocks.index(b)][3]
        r = m.r0[rows] + Jint @ (x[b.cols] - m.x_ref[b.cols])
        v = np.where(m.eq[rows], np.abs(r), np.maximum(r, 0.0))
        k = m.row_kind[rows]
        return float((m.w[rows] * v)[k == 3].sum() + (m.mu[rows] * v)[k == 2].sum() * m.J_scale)

    def _master_model(self, pool: list) -> tuple:
        """The restricted master as a LinearModel: master columns, then one weight per pool schedule."""
        m, st = self.m, self.st
        mc = st["master_cols"]
        mc = np.concatenate([mc[mc < m.N], mc[mc >= m.N]])
        mr = st["master_rows"]
        self._cache()
        Jr = self._Jr
        lam_cols, gl, owner = [], [], []
        for bi, b in enumerate(self.blocks):
            Jb = self._subs[bi][4]
            xr = m.x_ref[b.cols]
            for xk, val in pool[bi]:
                lam_cols.append(Jb @ (xk[b.cols] - xr))
                gl.append(val)
                owner.append(bi)
        nl = len(lam_cols)
        L = sp.csr_matrix(np.column_stack(lam_cols)) if nl else sp.csr_matrix((mr.size, 0))
        nb = len(self.blocks)
        conv = sp.csr_matrix((np.ones(nl), (np.array(owner), np.arange(nl))), shape=(nb, nl))
        Jm = sp.vstack([sp.hstack([Jr[:, mc], L]), sp.hstack([sp.csr_matrix((nb, mc.size)), conv])]).tocsr()
        ref = np.zeros(nl)
        first = {}
        for k, o in enumerate(owner):
            first.setdefault(o, k)
        ref[list(first.values())] = 1.0                       # every block's first pool entry is its reference
        nvar = mc.size + nl
        f0 = m.f0 - sum(self._penalty(b, m.x_ref) for b in self.blocks)
        z = lambda a, v=0: np.concatenate([a, np.full(nb, v, dtype=a.dtype)])  # noqa: E731
        mm = LinearModel(nvar, int((mc < m.N).sum()), np.concatenate([m.x_ref[mc], ref]),
                         np.concatenate([m.lo[mc], np.zeros(nl)]), np.concatenate([m.hi[mc], np.ones(nl)]),
                         np.concatenate([m.steps[mc], np.ones(nl, dtype=np.int64)]),
                         np.concatenate([m.var_inst[mc], np.full(nl, -1)]), np.concatenate([m.var_t[mc], np.zeros(nl, dtype=np.int64)]),
                         np.concatenate([m.var_kind[mc], np.full(nl, 3, dtype=np.int8)]),
                         Jm, z(m.r0[mr], 0.0), z(m.eq[mr], True), z(m.w[mr], 0.0), z(m.mu[mr], 0.0),
                         z(m.row_inst[mr], -1), z(m.row_t[mr], 0), z(m.row_kind[mr], 0), z(m.row_con[mr], -1),
                         z(m.row_lvl[mr], 1), m.cons, np.concatenate([m.g[mc], np.array(gl)]), f0, m.J_scale)
        return mm, mc, owner

    def _price(self, b: Block, duals_full: np.ndarray, radius, flips, time_limit) -> tuple:
        """Block b's exact best response at the master's prices: (schedule, its own value in USD, objective of
        the pricing problem in F units at the optimum and its dual bound)."""
        m = self.m
        self._cache()
        sub, touch, internal, _, _ = self._subs[self.blocks.index(b)]
        rad = radius if radius is None or np.ndim(radius) == 0 else np.asarray(radius)[b.cols]
        price = duals_full[touch]
        sol = solve(sub, rows=internal, row_price=price, radius=rad, flips=flips, integer=True, want_duals=False,
                    time_limit=time_limit)
        if not sol.status.startswith(("optimal", "time_limit")):
            # unbounded relaxation (a state the block does not define itself): solve inside a wide box and mark
            # the round's bound as not proven
            lif = np.arange(sub.N, sub.nvar)
            lo, hi = sub.lo[lif].copy(), sub.hi[lif].copy()
            wbox = 10.0 * (np.abs(sub.x_ref[lif]) + 1.0)
            sub.lo[lif], sub.hi[lif] = sub.x_ref[lif] - wbox, sub.x_ref[lif] + wbox
            try:
                sol = solve(sub, rows=internal, row_price=price, radius=rad, flips=flips, integer=True,
                            want_duals=False, time_limit=time_limit)
            finally:
                sub.lo[lif], sub.hi[lif] = lo, hi
            sol.mip_gap = np.inf
        x = m.x_ref.copy()
        x[b.cols] = sol.x
        val = float(m.g[b.cols] @ (x[b.cols] - m.x_ref[b.cols])) - (self._penalty(b, x) - self._penalty(b, m.x_ref))
        return x, val, sol

    # -------------------------------------------------------------- the loop
    def solve(self, radius=None, flips=None, rounds: int = 40, tol: float = 1e-7, time_limit: Optional[float] = None,
              block_time: Optional[float] = 20.0, commit_time: Optional[float] = 60.0, sweeps: int = 0,
              start: Optional[list] = None,
              log=None) -> EngineResult:
        m = self.m
        t0 = time.perf_counter()
        nb = len(self.blocks)
        total_ints = max(1, sum(b.ints for b in self.blocks))
        bflips = [None if flips is None else max(1.0, flips * b.ints / total_ints) for b in self.blocks]
        pool = [[(m.x_ref.copy(), 0.0)] for _ in self.blocks]     # reference schedule of every block first
        if start:
            for bi, xs in enumerate(start):
                for x in xs:
                    pool[bi].append(x)
        hist, tp, tm = [], 0.0, 0.0
        bound, lpv = np.inf, -np.inf
        best_bound = np.inf
        it = 0
        for it in range(1, rounds + 1):
            ts = time.perf_counter()
            mm, mc, owner = self._master_model(pool)
            ms = solve(mm, radius=radius, want_duals=True)
            last = (ms.x, mc.size, list(owner))
            tm += time.perf_counter() - ts
            if ms.duals is None:
                raise RuntimeError(f"master LP failed: {ms.status}")
            lpv = ms.F
            nr = self.st["master_rows"].size
            duals_full = np.zeros(m.J.shape[0])
            duals_full[self.st["master_rows"]] = ms.duals[:nr]
            conv = ms.duals[nr:nr + nb]
            # value of the pricing problem relative to the reference column = reduced cost + conv dual
            ts = time.perf_counter()
            add, rc_sum, proven = 0, 0.0, True
            for bi, b in enumerate(self.blocks):
                x, val, sol = self._price(b, duals_full, radius, bflips[bi], block_time)
                # objective of the new column in the master, minus the prices of its rows, minus conv dual
                a = self._subs[bi][4] @ (x[b.cols] - m.x_ref[b.cols])
                rc = val / m.J_scale - float(ms.duals[:nr] @ a) - conv[bi]
                if sol.mip_gap is not None and sol.mip_gap > 1e-4:     # inexact pricing: no proven bound this round
                    proven = False
                rc_sum += max(rc, 0.0)
                if rc > tol:
                    pool[bi].append((x, val))
                    add += 1
            tp += time.perf_counter() - ts
            bound = lpv + rc_sum if proven else np.inf
            best_bound = min(best_bound, bound)
            rec = {"round": it, "lp": lpv, "bound": best_bound, "added": add, "columns": sum(len(p) for p in pool),
                   "t": time.perf_counter() - t0}
            hist.append(rec)
            if log:
                log(rec)
            if add == 0 or (best_bound - lpv) <= tol * max(1.0, abs(lpv)) or \
                    (time_limit is not None and time.perf_counter() - t0 > time_limit):
                break
        # commit. (1) where every schedule the master LP weights agrees on a discrete column, it is fixed there; the
        # columns they disagree on (few: Shapley-Folkman) and every continuous decision are solved together, exactly
        # (one MILP). (2) one pool schedule per block (restricted master MILP), continuous decisions by one LP.
        # The better of the two is the plan.
        tc = time.perf_counter()
        disc = m.steps > 0
        lam = np.maximum(last[0][last[1]:], 0.0)
        xbar, agree = m.x_ref.copy(), np.ones(m.nvar, dtype=bool)
        for bi, b in enumerate(self.blocks):
            ks = [k for k, o in enumerate(last[2]) if o == bi]
            wv = lam[ks]
            wv = wv / wv.sum() if wv.sum() > 0 else np.eye(len(ks))[0]
            X = np.stack([pool[bi][j][0][b.cols] for j in range(len(ks))])
            xb = wv @ X
            xbar[b.cols] = xb
            used = wv > 1e-6
            agree[b.cols] = (np.abs(X[used] - xb) <= 1e-6).all(0)
        fix = disc & agree
        free_ints = int((disc & ~fix).sum())
        cands = []
        lo, hi = m.lo.copy(), m.hi.copy()
        try:
            m.lo, m.hi = np.where(fix, np.round(xbar * np.maximum(m.steps, 1)) / np.maximum(m.steps, 1), lo),                 np.where(fix, np.round(xbar * np.maximum(m.steps, 1)) / np.maximum(m.steps, 1), hi)
            r1 = solve(m, radius=radius, flips=flips, integer=True, want_duals=False, time_limit=commit_time)
            if r1.status in ("optimal", "time_limit"):
                cands.append(("fix-agreed", r1.x))
        finally:
            m.lo, m.hi = lo, hi
        mm, mc, owner = self._master_model(pool)
        cm = solve(mm, radius=radius, integer=True, want_duals=False, time_limit=commit_time)
        if cm.status in ("optimal", "time_limit"):
            lam2 = cm.x[mc.size:]
            x = m.x_ref.copy()
            for bi, b in enumerate(self.blocks):
                ks = [k for k, o in enumerate(owner) if o == bi]
                kbest = ks[int(np.argmax(lam2[ks]))]
                x[b.cols] = pool[bi][kbest - ks[0]][0][b.cols]
            try:
                m.lo, m.hi = np.where(disc, x, lo), np.where(disc, x, hi)
                fin = solve(m, x_fixed=x, radius=radius, want_duals=False)
                cands.append(("one-schedule-per-block", fin.x if fin.status == "optimal" else x))
            finally:
                m.lo, m.hi = lo, hi
        if not cands:
            cands.append(("reference", m.x_ref.copy()))
        scored = [(model_F(m, x), nm, x) for nm, x in cands]
        F, how, xfin = max(scored, key=lambda z: z[0])
        # block sweeps: one block's discrete columns free, the others fixed at the plan, solved exactly together
        # with every continuous decision (a MILP of one block's size); kept where F rises
        swept = 0
        for sweep in range(sweeps):
            better = False
            for b in self.blocks:
                if not (disc[b.cols] & ~fix[b.cols]).any():
                    continue
                free_b = np.zeros(m.nvar, dtype=bool)
                free_b[b.cols] = True
                fixd = disc & ~free_b
                try:
                    m.lo, m.hi = np.where(fixd, xfin, lo), np.where(fixd, xfin, hi)
                    rs = solve(m, radius=radius, flips=flips, integer=True, want_duals=False, time_limit=block_time)
                finally:
                    m.lo, m.hi = lo, hi
                if rs.status in ("optimal", "time_limit"):
                    Fs = model_F(m, rs.x)
                    if Fs > F + 1e-9 * max(1.0, abs(F)):
                        F, xfin, better = Fs, rs.x, True
                        swept += 1
            if not better:
                break
        tcommit = time.perf_counter() - tc
        res = EngineResult(xfin, F, best_bound, lpv, it, time.perf_counter() - t0, tp, tm, tcommit,
                           sum(len(p) for p in pool), nb, hist)
        res.commit = {"chosen": how, "free_integers": free_ints, "candidates": {nm: f for f, nm, _ in scored},
                      "sweep_improvements": swept, "F_after_sweeps": F}
        return res


def _submodel(m: LinearModel, cols: np.ndarray, rows: np.ndarray, J) -> LinearModel:
    """The model restricted to some columns (sorted: decisions first, as in the model) and rows; every other
    column stays at the reference, so the rows keep their reference values."""
    cols = np.asarray(cols)
    return LinearModel(cols.size, int((cols < m.N).sum()), m.x_ref[cols].copy(), m.lo[cols].copy(), m.hi[cols].copy(),
                       m.steps[cols], m.var_inst[cols], m.var_t[cols], m.var_kind[cols], J.tocsr(), m.r0[rows],
                       m.eq[rows], m.w[rows], m.mu[rows], m.row_inst[rows], m.row_t[rows], m.row_kind[rows],
                       m.row_con[rows], m.row_lvl[rows], m.cons, m.g[cols], 0.0, m.J_scale)


def _shift_radius(m: LinearModel, radius, x):
    return radius


def reduce_solver(int_cap: int = 200, flat=None, rounds: int = 40, log=None):
    """A solver(m, radius, flips) for exact.slp: the reduction engine in place of one whole-model HiGHS call."""
    def run(m, radius, flips):
        eng = Engine(m, int_cap, flat)
        r = eng.solve(radius=radius, flips=flips, rounds=rounds, log=log)
        return Solution(r.x, r.F, model_F(m, m.x_ref), "optimal", r.seconds, None, m.nvar, m.J.shape[0],
                        int(sum(b.ints for b in eng.blocks)), r.gap)
    return run
