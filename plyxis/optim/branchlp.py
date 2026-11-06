"""Exact solves of the twin's linear branches (classes from structure.py).

Where the structure analysis proves a branch of the twin affine in its decisions (ventilation regulators,
dewatering pumps against a power price, truck allocation, ...), the plan value restricted to that branch -
all other decisions fixed - is a linear program. Its data are the twin's own Jacobian on the branch, which
is exact everywhere for an affine branch, so it is extracted by batched finite differences in a few
simulations (sparse: only constraint rows that move are kept). The LP represents the penalised plan value
exactly:

    maximise   c . d  -  (mu / T) sum v  -  sum_soft w dt s / J_scale  -  sum move-suppression cuts
    subject to h + J d <= v,  v >= 0                hard constraints (elastic: a violation costs mu / T, as in F)
               h_s + J_s d <= s, s >= 0             soft constraints at their declared price
               m_j >= tangent cuts of the quadratic move-suppression penalty
               0 <= x + d <= 1,  |d| <= r           box, trust region (r = infinity for affine branches)

and is solved by HiGHS (dual simplex): the global optimum of the branch, with exact shadow prices. The
result is verified on the full twin. Branches that are only piecewise affine use the same LP with a trust
region and repeat it (sequential LP inside the current linear piece).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from scipy import sparse
from scipy.optimize import linprog

from .problem import Problem

MU = 50.0


@dataclass
class BranchResult:
    x: torch.Tensor
    F0: float
    F: float
    iterations: int
    seconds: float
    lp_seconds: float
    jac_seconds: float
    variables: int
    rows: int
    history: list = field(default_factory=list)


def class_variables(pb: Problem, classes=("lp",)) -> np.ndarray:
    """Indices of the continuous decisions of asset types whose structure class is in `classes`."""
    from .structure import classify_types
    key = (id(pb.prog), pb.space.N, tuple(classes))
    if key in _VARS and _VARS[key][0] is pb.prog:        # the program is kept alive: its id cannot be reused
        return _VARS[key][1]
    types = classify_types(pb.prog)
    idx = []
    for b in pb.space.blocks:
        t = types.get(b.gid)
        if t is not None and t.cls in classes and b.domain not in ("int", "bool"):
            idx.append(np.arange(b.offset, b.offset + b.size))
    _VARS[key] = (pb.prog, np.concatenate(idx) if idx else np.zeros(0, dtype=np.int64))
    return _VARS[key][1]


_VARS: dict = {}


def _F(pb: Problem, X: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        e = pb.evaluate(X, relax=False)
    return e.objective / pb.J_scale - MU / pb.T * e.sum_violation()


def _pieces(pb: Problem, e):
    """Smooth objective (without soft and move penalties), hard residuals (scaled), soft residuals (raw)."""
    obj = e.objective.clone()
    if pb.move_ref is not None:
        d = e.x - pb.move_ref
        obj = obj + pb.move_cost * (pb.move_w * d * d).sum(1)
    obj = obj + torch.as_tensor(e.totals.get("soft_penalty", 0.0)).reshape(-1).expand_as(obj)
    hard = {k: v for k, v in e.resid.items()}
    soft = {}
    for cr in pb.prog.cons:
        if cr.soft:
            soft[(cr.gid, cr.key)] = torch.as_tensor(e.raw[pb.prog.node_var(cr.gid, cr.key)])
    return obj / pb.J_scale, hard, soft


def _flat(t: torch.Tensor, B: int) -> torch.Tensor:
    t = torch.as_tensor(t)
    if t.dim() == 0:
        t = t.view(1, 1)
    t = t.reshape(t.shape[0], -1)
    return t.expand(B, -1) if t.shape[0] == 1 else t


def jacobian(pb: Problem, x: torch.Tensor, var: np.ndarray, step: float = 0.25, chunk: int = 128,
             tol: float = 1e-6) -> dict:
    """Sparse Jacobian of the smooth objective, hard and soft residual rows w.r.t. x[var] at x (finite
    differences; exact for affine branches up to float32 rounding). Vectorised: every residual tensor gets
    a block of global row numbers, non-zeros are gathered as coordinates, unused rows are dropped."""
    x = x.reshape(1, -1)
    with torch.no_grad():
        e0 = pb.evaluate(x, relax=False)
    o0, h0, s0 = _pieces(pb, e0)
    keys = [("h",) + k for k in h0] + [("s",) + k for k in s0]
    base = {("h",) + k: _flat(v, 1)[0].numpy() for k, v in h0.items()}
    base.update({("s",) + k: _flat(v, 1)[0].numpy() for k, v in s0.items()})
    offset, tot = {}, 0
    for k in keys:
        offset[k] = tot
        tot += len(base[k])
    grad = np.zeros(len(var))
    R, Cc, V = [], [], []
    for s in range(0, len(var), chunk):
        vs = var[s:s + chunk]
        X = x.repeat(len(vs), 1)
        h = np.where(x[0, vs].numpy() > 1 - step, -step, step)
        X[np.arange(len(vs)), vs] += torch.as_tensor(h, dtype=X.dtype)
        with torch.no_grad():
            e = pb.evaluate(X, relax=False)
        o, hd, sf = _pieces(pb, e)
        grad[s:s + len(vs)] = (o - o0[0]).numpy() / h
        for kind, dct in (("h", hd), ("s", sf)):
            for k, v in dct.items():
                kk = (kind,) + k
                diff = (_flat(v, len(vs)).numpy() - base[kk][None, :]) / h[:, None]
                jj, ii = np.nonzero(np.abs(diff) > tol)
                if len(jj):
                    R.append(offset[kk] + ii)
                    Cc.append(s + jj)
                    V.append(diff[jj, ii])
    R = np.concatenate(R) if R else np.zeros(0, dtype=np.int64)
    Cc = np.concatenate(Cc) if Cc else np.zeros(0, dtype=np.int64)
    V = np.concatenate(V) if V else np.zeros(0)
    used, inv = np.unique(R, return_inverse=True)
    J = sparse.csr_matrix((V, (inv, Cc)), shape=(len(used), len(var)))
    starts = np.array([offset[k] for k in keys] + [tot])
    which = np.searchsorted(starts, used, side="right") - 1
    meta = [keys[w] + (int(u - starts[w]),) for w, u in zip(which, used)]
    h0v = np.array([float(base[m[:3]][m[3]]) for m in meta]) if meta else np.zeros(0)
    return {"grad": grad, "J": J, "h0": h0v, "meta": meta, "F0": float(_F(pb, x)[0]),
            "len": {k[1:]: len(v) for k, v in base.items()}}


_PATTERN: dict = {}
STRUCTURED_JACOBIAN = os.environ.get("PLYXIS_STRUCTURED_JACOBIAN", "1") == "1"   # coloured / reverse-mode Jacobian for sequential LP


def jacobian_structured(pb: Problem, x: torch.Tensor, var: np.ndarray, step: float = 0.02,
                        dense_cols: int = 96, chunk: int = 128) -> dict:
    """The same Jacobian, cheaper, once its sparsity pattern is known (first call per twin structure: the full
    finite-difference Jacobian, whose pattern is cached):

        sparse rows   (an asset's own constraints): Curtis-Powell-Reid - columns that share no sparse row are
                      perturbed in the same simulation, one sample per colour of the column-intersection graph
        dense rows    (shared assets - mill, hoist, site power - and the objective): reverse mode on the
                      simulator's batch axis - B copies of the plan, copy b selects row b, one backward pass
                      gives B exact rows

    Rows outside the cached pattern (a kink crossed elsewhere) are missed; the verification step on the full
    twin guards that."""
    key = (id(pb.prog), pb.space.N, len(var), int(np.asarray(var).sum()))
    hit = _PATTERN.get(key)
    if hit is None or hit[0] is not pb.prog:
        jac = jacobian(pb, x, var, step=step)
        J = jac["J"].tocsr()
        nnz = np.diff(J.indptr)
        dense = np.nonzero(nnz > dense_cols)[0]
        sparse_rows = np.nonzero(nnz <= dense_cols)[0]
        Jc = J[sparse_rows].tocsc()
        colour = -np.ones(len(var), dtype=np.int64)
        row_cols: list = [set() for _ in range(len(sparse_rows))]
        order = np.argsort(-np.diff(Jc.indptr), kind="stable")
        for j in order:                                     # greedy colouring, high degree first
            rs = Jc.indices[Jc.indptr[j]:Jc.indptr[j + 1]]
            used = set()
            for r in rs:
                used |= row_cols[r]
            c = 0
            while c in used:
                c += 1
            colour[j] = c
            for r in rs:
                row_cols[r].add(c)
        _PATTERN[key] = (pb.prog, {"meta": jac["meta"], "J": J, "dense": dense, "sparse": sparse_rows,
                                   "Jc": Jc, "colour": colour, "len": jac["len"]})
        jac["colours"], jac["dense_rows"] = int(colour.max() + 1), int(len(dense))
        return jac
    pat = hit[1]
    meta, n = pat["meta"], len(var)
    x = x.reshape(1, -1)
    with torch.no_grad():
        e0 = pb.evaluate(x, relax=False)
    o0, h0, s0 = _pieces(pb, e0)
    base = {("h",) + k: _flat(v, 1)[0].numpy() for k, v in h0.items()}
    base.update({("s",) + k: _flat(v, 1)[0].numpy() for k, v in s0.items()})
    h0v = np.array([float(base[m[:3]][m[3]]) for m in meta]) if meta else np.zeros(0)
    rows, cols, vals = [], [], []
    # sparse rows: coloured forward differences
    colour, Jc, sp_rows = pat["colour"], pat["Jc"], pat["sparse"]
    K = int(colour.max() + 1) if len(colour) else 0
    hstep = np.where(x[0, var].numpy() > 1 - step, -step, step)
    col_of_row: list = []                                   # per sparse row: (columns, colours)
    Jr = Jc.tocsr()
    for i in range(len(sp_rows)):
        cs = Jr.indices[Jr.indptr[i]:Jr.indptr[i + 1]]
        col_of_row.append(cs)
    for s in range(0, K, chunk):
        ks = list(range(s, min(K, s + chunk)))
        X = x.repeat(len(ks), 1)
        for b, k in enumerate(ks):
            js = np.nonzero(colour == k)[0]
            X[b, var[js]] += torch.as_tensor(hstep[js], dtype=X.dtype)
        with torch.no_grad():
            e = pb.evaluate(X, relax=False)
        _, hd, sf = _pieces(pb, e)
        cur = {("h",) + k2: _flat(v, len(ks)).numpy() for k2, v in hd.items()}
        cur.update({("s",) + k2: _flat(v, len(ks)).numpy() for k2, v in sf.items()})
        for i, r in enumerate(sp_rows):
            m = meta[r]
            cs = col_of_row[i]
            sel = cs[(colour[cs] >= s) & (colour[cs] < s + len(ks))]
            if not len(sel):
                continue
            d = (cur[m[:3]][colour[sel] - s, m[3]] - base[m[:3]][m[3]]) / hstep[sel]
            rows += [r] * len(sel)
            cols += sel.tolist()
            vals += d.tolist()
    # dense rows and the objective: batched reverse mode (copy b differentiates row b)
    dense = list(pat["dense"]) + [-1]                       # -1 = the smooth objective
    grad = np.zeros(n)
    vt = torch.as_tensor(var, dtype=torch.long)
    vchunk = max(8, chunk // 2)                              # reverse mode keeps the graph: smaller batches
    for s in range(0, len(dense), vchunk):
        rs = dense[s:s + vchunk]
        X = x.repeat(len(rs), 1).requires_grad_(True)
        e = pb.evaluate(X, relax=False)
        o, hd, sf = _pieces(pb, e)
        flat = {("h",) + k2: _flat(v, len(rs)) for k2, v in hd.items()}
        flat.update({("s",) + k2: _flat(v, len(rs)) for k2, v in sf.items()})
        terms = []
        for b, r in enumerate(rs):
            if r < 0:
                terms.append(o[b])
            else:
                m = meta[r]
                terms.append(flat[m[:3]][b, m[3]])
        g = torch.autograd.grad(torch.stack(terms).sum(), X, allow_unused=True)[0]
        g = g[:, vt].detach().numpy() if g is not None else np.zeros((len(rs), n))
        for b, r in enumerate(rs):
            if r < 0:
                grad = g[b]
            else:
                nz = np.nonzero(np.abs(g[b]) > 1e-9)[0]
                rows += [r] * len(nz)
                cols += nz.tolist()
                vals += g[b][nz].tolist()
    J = sparse.csr_matrix((vals, (rows, cols)), shape=(len(meta), n))
    return {"grad": grad, "J": J, "h0": h0v, "meta": meta, "F0": float(_F(pb, x)[0]), "len": pat["len"],
            "colours": K, "dense_rows": len(pat["dense"])}


def solve_lp(pb: Problem, x: torch.Tensor, var: np.ndarray, jac: dict, trust: Optional[float] = None,
             margin: float = 0.0, warm: Optional[dict] = None):
    """The branch LP (see module doc) -> new x (only x[var] changes). `margin` keeps state limits (tanks, bins) that
    far inside, in units of the limit's scale: an LP optimum sits on the limits as the twin sees them, a real plant
    with an estimation error then runs into them (protection cuts production, a buffer is gone when it is needed)."""
    x = x.reshape(1, -1)
    n = len(var)
    meta, J, h0 = jac["meta"], jac["J"], jac["h0"]
    kinds = [m[0] for m in meta]
    bnd = pb.prog._bounds_cache()
    hard_i = [i for i, k in enumerate(kinds) if k == "h"]
    soft_i = [i for i, k in enumerate(kinds) if k == "s"]
    eq = np.array([pb.con_kind.get((meta[i][1], meta[i][2])) == "eq" for i in hard_i], dtype=bool)
    # soft weights per row: w * dt / J_scale (per instance if the weight is per instance)
    w_soft = []
    for i in soft_i:
        _, gid, key, flat = meta[i]
        w = torch.as_tensor(bnd[gid].get(f"soft:{key}", 0.0), dtype=torch.float32).reshape(-1)
        per_inst = max(1, jac["len"][(gid, key)] // max(1, pb.prog.gmap[gid].n))
        li = min(flat // per_inst, w.numel() - 1)
        w_soft.append(float(w[li]) * pb.dt / pb.J_scale)
    mv = []
    if pb.move_ref is not None:
        wv = pb.move_w[0, var].numpy() if pb.move_w.dim() == 2 else pb.move_w[var].numpy()
        mv = [j for j in range(n) if wv[j] > 0]
    nh, ns, nm = len(hard_i), len(soft_i), len(mv)
    nv_h = nh                                         # one violation slack per hard row
    N = n + nv_h + ns + nm
    c = np.zeros(N)
    c[:n] = -jac["grad"]
    c[n:n + nv_h] = MU / pb.T
    c[n + nv_h:n + nv_h + ns] = np.array(w_soft) if ns else 0.0
    c[n + nv_h + ns:] = 1.0
    A_rows, b = [], []
    Jh = J[hard_i] if nh else sparse.csr_matrix((0, n))
    I_h = sparse.identity(nh, format="csr") if nh else sparse.csr_matrix((0, 0))
    Z = lambda r, k: sparse.csr_matrix((r, k))  # noqa: E731
    if nh:
        A_rows.append(sparse.hstack([Jh, -I_h, Z(nh, ns + nm)]))
        back_off = np.zeros(nh)
        if margin > 0.0:
            bound = {(c.gid, c.key) for c in pb.prog.cons if c.is_bound}
            back_off = margin * np.array([(meta[i][1], meta[i][2]) in bound for i in hard_i], dtype=float)
        b.append(-h0[hard_i] - back_off)
        if eq.any():
            sel = np.nonzero(eq)[0]
            A_rows.append(sparse.hstack([-Jh[sel], -I_h[sel], Z(len(sel), ns + nm)]))
            b.append(h0[np.array(hard_i)[sel]])
    if ns:
        Js = J[soft_i]
        A_rows.append(sparse.hstack([Js, Z(ns, nv_h), -sparse.identity(ns, format="csr"), Z(ns, nm)]))
        b.append(-h0[soft_i])
    if nm:                                            # tangent cuts of a (x + d - ref)^2, a = cost w / J_scale
        cur = x[0, var].numpy()
        ref = pb.move_ref[0, var].numpy()
        wv = pb.move_w[0, var].numpy() if pb.move_w.dim() == 2 else pb.move_w[var].numpy()
        rr, cc, vv, bb = [], [], [], []
        r = 0
        for k, j in enumerate(mv):
            a = pb.move_cost * wv[j] / pb.J_scale
            d0s = (cur[j] - ref[j]) + np.array([-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0])
            for d0 in d0s:                            # m >= a d0^2 + 2 a d0 (cur - ref + d - d0)
                rr += [r, r]
                cc += [j, n + nv_h + ns + k]
                vv += [2 * a * d0, -1.0]
                bb.append(-(a * d0 * d0 + 2 * a * d0 * ((cur[j] - ref[j]) - d0)))
                r += 1
        A_rows.append(sparse.csr_matrix((vv, (rr, cc)), shape=(r, N)))
        b.append(np.array(bb))
    A = sparse.vstack(A_rows, format="csr") if A_rows else None
    bvec = np.concatenate(b) if b else None
    cur = x[0, var].numpy().astype(np.float64)
    lo, hi = -cur, 1.0 - cur
    if trust is not None:
        lo, hi = np.maximum(lo, -trust), np.minimum(hi, trust)
    t = time.perf_counter()
    if warm is not None and A is not None:            # HiGHS directly, from the last LP's basis (exact._highs_lp)
        from .exact import _highs_lp
        lo_all = np.concatenate([lo, np.zeros(nv_h + ns + nm)])
        hi_all = np.concatenate([hi, np.full(nv_h + ns + nm, np.inf)])
        # 10 s per LP: an LP without a limit once held a mine plan for over 15 minutes (a timed-out LP = no step)
        y, status, _ = _highs_lp(c, A, np.full(A.shape[0], -np.inf), bvec, lo_all, hi_all, warm, time_limit=10.0)
        lp_s = time.perf_counter() - t
        if y is None:
            return None, lp_s, None
        res = type("R", (), {"x": y, "status": 0})()
    else:
        bounds = list(zip(lo, hi)) + [(0, None)] * (nv_h + ns + nm)
        res = linprog(c, A_ub=A, b_ub=bvec, bounds=bounds, method="highs")
        lp_s = time.perf_counter() - t
        if res.status != 0 or res.x is None:
            return None, lp_s, res
    xn = x.clone()
    xn[0, var] = torch.as_tensor(np.clip(cur + res.x[:n], 0.0, 1.0), dtype=x.dtype)
    return xn, lp_s, res


def tangent_gap(pb: Problem, x: torch.Tensor, var: np.ndarray, jac: dict, margin: float = 0.0) -> Optional[float]:
    """How much the linearised plan value could still gain over x[var] in the whole box (units of F, the LP without a
    trust region). If the plan value is concave in these decisions, F(x) plus this is an upper bound of what they can
    reach. The LP's objective holds the penalties at d = 0 as well, they are added back here."""
    x = x.reshape(1, -1)
    _, _, res = solve_lp(pb, x, var, jac, None, margin)
    if res.x is None:
        return None
    return max(0.0, _penalties(pb, x, var, jac, margin) - float(res.fun))


def _penalties(pb: Problem, x: torch.Tensor, var: np.ndarray, jac: dict, margin: float) -> float:
    """The LP's penalty terms at d = 0: hard and soft violations and the move suppression, as `solve_lp` prices them."""
    meta, h0 = jac["meta"], jac["h0"]
    bnd = pb.prog._bounds_cache()
    bound = {(c.gid, c.key) for c in pb.prog.cons if c.is_bound} if margin > 0.0 else set()
    tot = 0.0
    for i, m in enumerate(meta):
        if m[0] == "h":
            v = float(h0[i]) + (margin if (m[1], m[2]) in bound else 0.0)
            is_eq = pb.con_kind.get((m[1], m[2])) == "eq"
            tot += MU / pb.T * (abs(v) if is_eq else max(0.0, v))
        else:
            w = torch.as_tensor(bnd[m[1]].get(f"soft:{m[2]}", 0.0), dtype=torch.float32).reshape(-1)
            per_inst = max(1, jac["len"][(m[1], m[2])] // max(1, pb.prog.gmap[m[1]].n))
            tot += float(w[min(m[3] // per_inst, w.numel() - 1)]) * pb.dt / pb.J_scale * max(0.0, float(h0[i]))
    if pb.move_ref is not None:
        cur, ref = x[0, var].numpy(), pb.move_ref[0, var].numpy()
        wv = pb.move_w[0, var].numpy() if pb.move_w.dim() == 2 else pb.move_w[var].numpy()
        tot += float((pb.move_cost * wv * (cur - ref) ** 2).sum() / pb.J_scale)
    return tot


def solve_branches(pb: Problem, x: torch.Tensor, var: np.ndarray, affine: bool = True, iters: int = 4,
                   trust: float = 0.25, tol: float = 1e-5, callback=None, margin: float = 0.0,
                   warm_lp: bool = False) -> BranchResult:
    """LP (affine) or sequential LP with a trust region (piecewise affine) over x[var], verified on the twin.
    Sequential LP reuses the Jacobian while its steps verify (slopes of a piecewise-affine function only
    change where a kink is crossed) and recomputes it after a rejected step."""
    t0 = time.perf_counter()
    x = x.reshape(1, -1).detach().clone()
    F0 = float(_F(pb, x)[0])
    f_cur = F0
    if len(var) == 0:
        return BranchResult(x, F0, F0, 0, time.perf_counter() - t0, 0.0, 0.0, 0, 0)
    r = None if affine else trust
    hist, jac_s, lp_s, rows = [], 0.0, 0.0, 0
    jac, x_jac = None, None
    lp_state = {} if warm_lp else None                 # HiGHS basis carried from one LP to the next
    it = 0
    for it in range(1, iters + 1):
        if jac is None:
            t = time.perf_counter()
            if affine:
                jac = jacobian(pb, x, var, step=0.25)
            elif STRUCTURED_JACOBIAN:                                # local slopes where there are kinks
                jac = jacobian_structured(pb, x, var, step=0.02)
            else:
                jac = jacobian(pb, x, var, step=0.02)
            x_jac = x.clone()
            jac_s += time.perf_counter() - t
        elif x_jac is not None and not torch.equal(x, x_jac):
            jac = _shift(pb, jac, x, var)                                  # same slopes, current offsets
            x_jac = x.clone()
        rows = len(jac["meta"])
        xn, s, _ = solve_lp(pb, x, var, jac, r, margin, warm=lp_state)
        lp_s += s
        accepted, step = False, 0.0
        if xn is not None:
            # verification (and a short line search towards the LP point when it does not verify)
            alphas = torch.tensor([1.0, 0.5, 0.25, 0.1]).view(-1, 1)
            X = x + alphas * (xn - x)
            Fn = _F(pb, X)
            k = int(torch.argmax(Fn))
            if float(Fn[k]) > f_cur + 1e-9:
                step = float(Fn[k]) - f_cur
                x, f_cur, accepted = X[k:k + 1].clone(), float(Fn[k]), True
        rec = {"stage": "branches", "iter": it, "F": f_cur, "accepted": accepted, "rows": rows,
               "trust": r, "t": time.perf_counter() - t0}
        hist.append(rec)
        if callback:
            callback(rec)
        if affine and accepted:
            break                                      # exact for an affine branch: one LP is the optimum
        if accepted and step < tol:
            break                                      # converged
        if not accepted:
            if jac is not None and x_jac is not None and torch.equal(x, x_jac) and r is not None:
                r *= 0.5                               # fresh slopes and still no improvement: smaller region
            jac = None                                 # rejected: recompute the slopes
            if r is None:
                r = trust
            if r < 0.01:
                break
    return BranchResult(x, F0, f_cur, it, time.perf_counter() - t0, lp_s, jac_s, len(var), rows, hist)


def _shift(pb: Problem, jac: dict, x: torch.Tensor, var: np.ndarray) -> dict:
    """Keep the slopes, refresh the offsets (objective gradient stays; residual values at the new x)."""
    with torch.no_grad():
        e = pb.evaluate(x.reshape(1, -1), relax=False)
    _, hd, sf = _pieces(pb, e)
    vals = {("h",) + k: _flat(v, 1)[0].numpy() for k, v in hd.items()}
    vals.update({("s",) + k: _flat(v, 1)[0].numpy() for k, v in sf.items()})
    out = dict(jac)
    out["h0"] = np.array([float(vals[m[:3]][m[3]]) for m in jac["meta"]]) if jac["meta"] else np.zeros(0)
    return out
