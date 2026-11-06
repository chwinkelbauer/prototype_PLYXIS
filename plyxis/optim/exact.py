"""Exact solves on the algebraic model of a twin (algebra.py): HiGHS LP / MILP, and sequential LP on the twin.

`solve(m, ...)` turns the LinearModel - or any part of it: a set of free columns, the rest fixed - into

    maximise   F_model = (f0 + g.(x - x_ref) - sum_soft w sigma) / J_scale  -  sum_hard mu v
    subject to r0 + J (x - x_ref) = 0                state updates, lifted delays (exact definitions)
               r0 + J (x - x_ref) <= v, v >= 0       hard limits, elastic at the planner's own price mu (MU / T per scaled
                                                     unit, as Problem fitness); eq rows with v+ - v-
               r0 + J (x - x_ref) <= sigma, s >= 0   soft limits at their declared price
               lo <= u <= hi, |u - u_ref| <= radius  decisions; discrete ones integer (steps of their grid)

F_model is the first-order model of the planner's penalised value F = objective / J_scale - MU / T * violation, with
the kinks of the penalties kept exact. An LP returns the shadow price of every row it contains (value of one unit
more slack in F units): the prices the coordination layer (reduce.py) puts on rows that couple parts of the twin.

`slp(pb, x)` repeats linearise -> solve -> simulate inside a trust region: every candidate is judged on the torch
twin and kept only if F improves. Where the twin is affine this is one exact LP; where it is piecewise affine or
nonlinear, a sequence of them.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
from scipy.optimize import LinearConstraint, linprog, milp

from .algebra import MU, LinearModel, linearize


@dataclass
class Solution:
    x: np.ndarray                   # all model columns (free ones solved, the others as fixed)
    F: float                        # F_model of the solution
    F_ref: float                    # F_model at the reference (same free set)
    status: str
    seconds: float
    duals: Optional[np.ndarray]     # shadow price per model row (F units per unit of slack), LP only
    columns: int = 0
    rows: int = 0
    integer: int = 0
    mip_gap: Optional[float] = None


def model_F(m: LinearModel, x: np.ndarray) -> float:
    """F_model at x (all columns)."""
    r = m.rows_at(x)
    hard = m.row_kind == 2
    v = np.where(m.eq[hard], np.abs(r[hard]), np.maximum(r[hard], 0.0))
    return m.objective_at(x) / m.J_scale - float((m.mu[hard] * v).sum())


def solve(m: LinearModel, free=None, x_fixed: Optional[np.ndarray] = None, rows=None, radius=None,
          flips: Optional[float] = None, integer: bool = False, elastic: bool = True,
          cost: Optional[np.ndarray] = None, row_price: Optional[np.ndarray] = None,
          time_limit: Optional[float] = None, mip_gap: float = 1e-4, want_duals: bool = True,
          warm: Optional[dict] = None) -> Solution:
    """HiGHS on the model restricted to the free columns (bool mask or indices; default all) and the given rows
    (default: every row that touches a free column).
      cost       extra objective per column in F units (added)
      row_price  Lagrangian price per row in F units: those rows leave the constraints and enter the objective as
                 -price * row value (the pricing problem of a decomposition)
      radius     trust region of the continuous decisions (|u - u_ref| <= radius)
      flips      trust region of the discrete ones: at most this many grid steps away from the reference in total
                 (local branching); None = free
    Rows are scaled to unit largest coefficient (tonnes, MW and USD in one model otherwise defeat the MILP).
      warm       a dict kept between calls on models of one layout (a sequential LP): the LP goes to HiGHS directly and
                 starts from the last call's basis; every row stays in, so the dimensions do not change between calls"""
    t0 = time.perf_counter()
    nvar = m.nvar
    xf = m.x_ref.copy() if x_fixed is None else np.asarray(x_fixed, dtype=np.float64).copy()
    fmask = np.ones(nvar, dtype=bool) if free is None else _mask(free, nvar)
    fidx = np.nonzero(fmask)[0]
    nf = fidx.size
    J = m.J
    Jf = J[:, fidx].tocsr()
    touch = np.diff(Jf.indptr) > 0
    rmask = touch.copy() if rows is None else _mask(rows, J.shape[0]) & touch
    if warm is not None and rows is None and not integer:
        rmask = np.ones(J.shape[0], dtype=bool)          # stable dimensions for the basis of the last call
    priced = np.zeros(J.shape[0], dtype=bool)
    if row_price is not None:
        priced = (np.asarray(row_price) != 0) & touch
        rmask &= ~priced
    xfix = xf.copy()
    xfix[fidx] = m.x_ref[fidx]
    c = m.r0 - Jf @ m.x_ref[fidx] + J @ (xfix - m.x_ref)          # r(x) = Jf x_f + c
    gobj = -m.g[fidx] / m.J_scale                                  # minimise -F
    if cost is not None:
        gobj = gobj - np.asarray(cost)[fidx]
    if priced.any():
        gobj = gobj + Jf[priced].T @ np.asarray(row_price)[priced]
    sel = np.nonzero(rmask)[0]
    k, is_eq = m.row_kind[sel], m.eq[sel]
    defs = sel[k <= 1]
    defs_le = sel[k == 4]                                          # definitions of exact piecewise terms: never elastic
    hard_le, hard_eq = sel[(k == 2) & ~is_eq], sel[(k == 2) & is_eq]
    soft_le, soft_eq = sel[(k == 3) & ~is_eq], sel[(k == 3) & is_eq]
    s = np.asarray(abs(Jf).max(axis=1).todense()).ravel()
    s = np.where(s > 0, s, 1.0)
    # integer columns
    steps = m.steps[fidx] if integer else np.zeros(nf, dtype=np.int64)
    disc = steps > 0
    scale = np.where(disc, steps, 1).astype(np.float64)            # integer column z = steps * u
    lo, hi = m.lo[fidx].copy(), m.hi[fidx].copy()
    if radius is not None:
        cont = (fidx < m.N) & ~disc
        r = np.broadcast_to(np.asarray(radius, dtype=np.float64), (nvar,))[fidx]
        lo[cont] = np.maximum(lo[cont], m.x_ref[fidx][cont] - r[cont])
        hi[cont] = np.minimum(hi[cont], m.x_ref[fidx][cont] + r[cont])
    lo = np.where(disc, np.ceil(lo * scale - 1e-9), lo)
    hi = np.where(disc, np.floor(hi * scale + 1e-9), hi)
    zref = np.round(m.x_ref[fidx] * scale)
    # variables: [x_f (nf) | slacks | flip distances]
    nh = hard_le.size if elastic else 0
    ne = hard_eq.size if elastic else 0
    ns, nse = soft_le.size, soft_eq.size
    ldisc = disc & (fidx < m.N)                                    # flips count decisions, not indicators
    nd = int(ldisc.sum()) if (flips is not None and ldisc.any()) else 0
    o_h, o_e, o_s, o_se = nf, nf + nh, nf + nh + 2 * ne, nf + nh + 2 * ne + ns
    o_d = o_se + nse
    ncol = o_d + nd
    Jc = (Jf @ sp.diags(1.0 / scale)).tocsr() if disc.any() else Jf

    def rows_of(idx, sign=1.0):
        return (sp.diags(sign / s[idx]) @ Jc[idx]).tocsr()

    def eye(n, off, val):
        return sp.csr_matrix((np.full(n, val), (np.arange(n), off + np.arange(n))), shape=(n, ncol))

    def pad(A):
        return sp.hstack([A, sp.csr_matrix((A.shape[0], ncol - nf))]).tocsr()

    blocks = []                    # (A, lb, ub, model rows | None, sign)
    if defs.size:
        blocks.append((pad(rows_of(defs)), -c[defs] / s[defs], -c[defs] / s[defs], defs, 1.0))
    if defs_le.size:
        blocks.append((pad(rows_of(defs_le)), np.full(defs_le.size, -np.inf), -c[defs_le] / s[defs_le], defs_le, 1.0))
    if hard_le.size:
        A = pad(rows_of(hard_le))
        if elastic:
            A = A + eye(nh, o_h, -1.0)
        blocks.append((sp.csr_matrix(A), np.full(hard_le.size, -np.inf), -c[hard_le] / s[hard_le], hard_le, 1.0))
    if hard_eq.size:
        A = pad(rows_of(hard_eq))
        if elastic:
            A = A + eye(ne, o_e, -1.0) + eye(ne, o_e + ne, 1.0)
        blocks.append((sp.csr_matrix(A), -c[hard_eq] / s[hard_eq], -c[hard_eq] / s[hard_eq], hard_eq, 1.0))
    if ns:
        A = pad(rows_of(soft_le)) + eye(ns, o_s, -1.0)
        blocks.append((sp.csr_matrix(A), np.full(ns, -np.inf), -c[soft_le] / s[soft_le], soft_le, 1.0))
    if nse:
        A = pad(rows_of(soft_eq)) + eye(nse, o_se, -1.0)
        blocks.append((sp.csr_matrix(A), np.full(nse, -np.inf), -c[soft_eq] / s[soft_eq], soft_eq, 1.0))
        A = pad(rows_of(soft_eq, -1.0)) + eye(nse, o_se, -1.0)
        blocks.append((sp.csr_matrix(A), np.full(nse, -np.inf), c[soft_eq] / s[soft_eq], soft_eq, -1.0))
    if nd:                                                         # local branching: sum |z - z_ref| <= flips
        di = np.nonzero(ldisc)[0]
        rr = np.concatenate([np.arange(nd)] * 2)
        cc = np.concatenate([di, o_d + np.arange(nd)])
        blocks.append((sp.csr_matrix((np.concatenate([np.ones(nd), -np.ones(nd)]), (rr, cc)), shape=(nd, ncol)),
                       np.full(nd, -np.inf), zref[di], None, 0.0))                 # z - d <= z_ref
        blocks.append((sp.csr_matrix((-np.ones(2 * nd), (rr, cc)), shape=(nd, ncol)),
                       np.full(nd, -np.inf), -zref[di], None, 0.0))                # -z - d <= -z_ref
        blocks.append((sp.csr_matrix((np.ones(nd), (np.zeros(nd, dtype=np.int64), o_d + np.arange(nd))),
                                     shape=(1, ncol)), np.array([-np.inf]), np.array([float(flips)]), None, 0.0))
    obj = np.zeros(ncol)
    obj[:nf] = gobj / scale
    if nh:
        obj[o_h:o_h + nh] = m.mu[hard_le] * s[hard_le]
    if ne:
        obj[o_e:o_e + 2 * ne] = np.concatenate([m.mu[hard_eq] * s[hard_eq]] * 2)
    if ns:
        obj[o_s:o_s + ns] = m.w[soft_le] * s[soft_le] / m.J_scale
    if nse:
        obj[o_se:o_se + nse] = m.w[soft_eq] * s[soft_eq] / m.J_scale
    lo_all = np.concatenate([lo, np.zeros(ncol - nf)])
    hi_all = np.concatenate([hi, np.full(ncol - nf, np.inf)])
    A = sp.vstack([b[0] for b in blocks]).tocsr() if blocks else sp.csr_matrix((0, ncol))
    lb = np.concatenate([b[1] for b in blocks]) if blocks else np.zeros(0)
    ub = np.concatenate([b[2] for b in blocks]) if blocks else np.zeros(0)
    duals, gap, y = None, None, None
    if disc.any():
        opts = {"mip_rel_gap": mip_gap}
        if time_limit:
            opts["time_limit"] = float(time_limit)
        integ = np.concatenate([disc.astype(np.int8), np.zeros(ncol - nf, dtype=np.int8)])
        res = milp(obj, constraints=[LinearConstraint(A, lb, ub)] if blocks else None, integrality=integ,
                   bounds=_bounds(lo_all, hi_all), options=opts)
        y = res.x
        status = "optimal" if res.status == 0 else ("time_limit" if y is not None else f"failed:{res.message}")
        gap = getattr(res, "mip_gap", None)
    elif warm is not None:
        y, status, row_dual = _highs_lp(obj, A, lb, ub, lo_all, hi_all, warm, time_limit)
        if y is not None and want_duals:
            duals = np.zeros(J.shape[0])
            i0 = 0
            for b in blocks:
                n = b[0].shape[0]
                if b[3] is not None:
                    np.add.at(duals, b[3], b[4] * (-row_dual[i0:i0 + n]) / s[b[3]])
                i0 += n
    else:
        isq = lb == ub
        Aub, Aeq = A[~isq], A[isq]
        res = linprog(obj, A_ub=Aub if Aub.shape[0] else None, b_ub=ub[~isq] if Aub.shape[0] else None,
                      A_eq=Aeq if Aeq.shape[0] else None, b_eq=ub[isq] if Aeq.shape[0] else None,
                      bounds=np.column_stack([lo_all, hi_all]), method="highs",
                      options={"time_limit": float(time_limit)} if time_limit else None)
        y = res.x
        status = "optimal" if res.status == 0 else f"failed:{res.message}"
        if y is not None and want_duals:
            duals = np.zeros(J.shape[0])
            mg_ub = -np.asarray(res.ineqlin.marginals) if Aub.shape[0] else np.zeros(0)
            mg_eq = -np.asarray(res.eqlin.marginals) if Aeq.shape[0] else np.zeros(0)
            iu = ie = 0
            for b in blocks:                                       # LP rows keep the blocks' order within ub / eq
                n = b[0].shape[0]
                if (b[1] == b[2]).all():
                    mg = mg_eq[ie:ie + n]
                    ie += n
                else:
                    mg = mg_ub[iu:iu + n]
                    iu += n
                if b[3] is not None:
                    np.add.at(duals, b[3], b[4] * mg / s[b[3]])    # F per raw unit of the row's slack
    xout = xf.copy()
    xout[fidx] = m.x_ref[fidx]
    if y is not None:
        xout[fidx] = y[:nf] / scale
    F_ref = model_F(m, _with(m.x_ref, fidx, m.x_ref[fidx], xf))
    F = model_F(m, xout) if y is not None else F_ref
    return Solution(xout, F, F_ref, status, time.perf_counter() - t0, duals, nf, int(sel.size), int(disc.sum()), gap)


def _highs_lp(c, A, lb, ub, lo, hi, warm: dict, time_limit=None):
    """min c.x, lb <= A x <= ub, lo <= x <= hi with HiGHS (scipy's bundled interface), from warm['basis'] if its
    dimensions match. Returns (x | None, status, row duals)."""
    from scipy.optimize._highspy import _core as _h
    inf = _h.kHighsInf
    fix = lambda v: np.clip(np.asarray(v, dtype=np.float64), -inf, inf)  # noqa: E731  (+-inf -> HiGHS infinity)
    Ac = A.tocsc()
    n, mrow = Ac.shape[1], Ac.shape[0]
    lp = _h.HighsLp()
    lp.num_col_, lp.num_row_ = n, mrow
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = n, mrow
    lp.a_matrix_.format_ = _h.MatrixFormat.kColwise
    lp.col_cost_ = np.asarray(c, dtype=np.float64)
    lp.col_lower_, lp.col_upper_ = fix(np.asarray(lo, dtype=np.float64)), fix(np.asarray(hi, dtype=np.float64))
    lp.row_lower_, lp.row_upper_ = fix(np.asarray(lb, dtype=np.float64)), fix(np.asarray(ub, dtype=np.float64))
    lp.a_matrix_.start_ = Ac.indptr.astype(np.int32)
    lp.a_matrix_.index_ = Ac.indices.astype(np.int32)
    lp.a_matrix_.value_ = Ac.data.astype(np.float64)
    h = _h._Highs()
    h.setOptionValue("output_flag", False)
    if time_limit:
        h.setOptionValue("time_limit", float(time_limit))
    for k, v in (warm.get("options") or {}).items():
        h.setOptionValue(k, v)
    h.passModel(lp)
    used = warm.get("basis") is not None and warm.get("dims") == (n, mrow)
    if used:
        for k, v in (warm.get("warm_options") or {}).items():
            h.setOptionValue(k, v)
        h.setBasis(warm["basis"])
        warm["warm_starts"] = warm.get("warm_starts", 0) + 1
    h.run()
    st = h.getModelStatus()
    ok = st == _h.HighsModelStatus.kOptimal
    if not ok:
        warm.pop("basis", None)
        if used and st != _h.HighsModelStatus.kTimeLimit:
            # a warm start that ends 'Unknown' (seen on the steel group after 20+ polish steps) is retried once
            # from scratch; it used to end the whole polish
            warm["retries"] = warm.get("retries", 0) + 1
            return _highs_lp(c, A, lb, ub, lo, hi, warm, time_limit)
        return None, f"failed:{h.modelStatusToString(st)}", None
    sol = h.getSolution()
    warm["basis"], warm["dims"] = h.getBasis(), (n, mrow)
    warm["iterations"] = warm.get("iterations", 0) + int(h.getInfo().simplex_iteration_count)
    return np.asarray(sol.col_value), "optimal", np.asarray(sol.row_dual)


def _bounds(lo, hi):
    from scipy.optimize import Bounds
    return Bounds(lo, hi)


def _with(base, idx, vals, fixed):
    out = fixed.copy()
    out[idx] = vals
    return out


def _mask(sel, n: int) -> np.ndarray:
    sel = np.asarray(sel)
    if sel.dtype == bool:
        return sel
    m = np.zeros(n, dtype=bool)
    m[sel] = True
    return m


# ----------------------------------------------------------------------------- sequential LP on the twin
def fitness(pb, u, relax: bool = False) -> float:
    with torch.no_grad():
        e = pb.evaluate(torch.as_tensor(u, dtype=torch.float32).reshape(1, -1), relax=relax)
    return float(e.objective[0] / pb.J_scale - MU / pb.T * e.sum_violation()[0]), float(e.max_violation()[0]), \
        float(e.objective[0])


def fitness_batch(pb, X, relax: bool = False):
    """F, max violation and objective of several plans in one batched simulation."""
    with torch.no_grad():
        e = pb.evaluate(torch.as_tensor(X, dtype=torch.float32), relax=relax)
    F = e.objective / pb.J_scale - MU / pb.T * e.sum_violation()
    return F.double().numpy(), e.max_violation().double().numpy(), e.objective.double().numpy()


@dataclass
class SLPResult:
    x: torch.Tensor
    F: float
    objective: float
    max_violation: float
    F0: float
    iterations: int
    seconds: float
    lp_seconds: float
    lin_seconds: float
    history: list = field(default_factory=list)


def slp(pb, x0=None, iters: int = 40, radius: float = 0.3, min_radius: float = 1e-3, integer: bool = False,
        flips: Optional[float] = None, time_budget: Optional[float] = None, free=None, tol: float = 1e-6,
        mip_time: Optional[float] = 30.0, solver=None, callback=None,
        line_search: tuple = (1.0,), fix_discrete: bool = False, exact_cols: bool = False,
        warm_lp: bool = False, decisions: Optional[np.ndarray] = None, lp_time: Optional[float] = None) -> SLPResult:
    """Trust-region sequential LP on the twin: linearise at x, solve the LP (or MILP) in the trust region, simulate
    the candidate, keep it if F rises; the radius grows when the model predicted well and shrinks when it did not.
    integer: the discrete decisions stay on their grid, within a budget of flips (local branching) that grows and
    shrinks with the radius. solver(m, radius, flips) -> Solution replaces the direct HiGHS solve (reduce.py).
    line_search: fractions of the LP step the twin scores in one batched simulation (continuous plans only; e.g.
    (1, 0.5, 0.25, 0.125) - on the steel group +0.52 M in 9 iterations against +0.70 M in 15 without, so off) - the
    LP's direction holds the coordinated move a first-order search misses, its full length may cross a kink.
    fix_discrete: every discrete column stays where x has it - integer and bool decisions, exclusive line-ups and the
    indicators of exact piecewise terms - and the LP moves all continuous decisions of the whole twin around them (the
    polish after a cascade whose discrete stage has decided).
    exact_cols: decisions along which the model is exact (only affine rows and objective terms) get no trust region -
    the exact LP of a linear branch inside the sequential LP of the rest."""
    t0 = time.perf_counter()
    x = (pb.space.default_u(1) if x0 is None else torch.as_tensor(x0, dtype=torch.float32)).reshape(-1).clone()
    relax = not integer
    if fix_discrete:                      # judged as executed: decisions on their grid, exclusive line-ups one-hot
        x = pb.space.round_discrete(x.view(1, -1)).reshape(-1)
        relax = False
    F, mv, obj = fitness(pb, x, relax)
    F0 = F
    hist = [{"it": 0, "F": F, "viol": mv, "radius": radius, "t": 0.0}]
    lp_s = lin_s = 0.0
    it = 0
    m = None
    lp_state = {} if (warm_lp and not integer) else None     # HiGHS basis carried from one LP to the next
    if integer and flips is None:
        nint = int((pb.space.discrete_mask()).sum())
        flips = max(4.0, 0.05 * nint)
    for it in range(1, iters + 1):
        if time_budget is not None and time.perf_counter() - t0 > time_budget:
            break
        if m is None:
            tl = time.perf_counter()
            m = linearize(pb, x)
            lin_s += time.perf_counter() - tl
            if fix_discrete or decisions is not None:
                free = (m.steps == 0) if fix_discrete else np.ones(m.nvar, dtype=bool)
                if fix_discrete:
                    free[:m.N] &= ~pb.space.discrete_mask().numpy()
                if decisions is not None:              # only these decisions move (states and terms follow)
                    keep = np.zeros(m.N, dtype=bool)
                    keep[np.asarray(decisions)] = True
                    free[:m.N] &= keep
        rad = radius
        if exact_cols and m.col_exact is not None:     # no trust region where the model is exact
            rad = np.where(m.col_exact, 1.0, radius)
        if solver is not None:
            sol = solver(m, rad, flips if integer else None)
        else:
            sol = solve(m, free=free, radius=rad, flips=flips if integer else None, integer=integer,
                        want_duals=False, time_limit=mip_time if integer else lp_time, warm=lp_state)
        lp_s += sol.seconds
        pred = sol.F - sol.F_ref
        if sol.status not in ("optimal", "time_limit") or pred <= tol:
            rec = {"it": it, "F": F, "viol": mv, "radius": radius, "pred": pred, "status": sol.status, "stop": True,
                   "t": time.perf_counter() - t0}
            hist.append(rec)
            if callback:
                callback(rec)
            if radius <= min_radius or pred <= tol:
                break
            radius *= 0.3
            flips = None if flips is None else max(1.0, flips * 0.3)
            continue
        cand = torch.as_tensor(np.clip(sol.x[:m.N], 0.0, 1.0), dtype=torch.float32)
        alphas = tuple(line_search) if (line_search and not integer) else (1.0,)
        X = torch.stack([x + a * (cand - x) for a in alphas])
        Fs, mvs, objs = fitness_batch(pb, X, relax)
        k = int(np.argmax(Fs))
        alpha = alphas[k]
        cand, Fc, mvc, objc = X[k], float(Fs[k]), float(mvs[k]), float(objs[k])
        rho = (Fc - F) / pred
        ok = Fc > F
        rec = {"it": it, "F": Fc, "viol": mvc, "radius": radius, "flips": flips, "pred": pred, "rho": rho,
               "alpha": alpha, "accepted": ok, "lp_s": sol.seconds, "t": time.perf_counter() - t0}
        hist.append(rec)
        if callback:
            callback(rec)
        if ok:
            x, F, mv, obj = cand, Fc, mvc, objc
            m = None
            if alpha < 1.0:                    # only part of the step paid: the next trust region is that part
                radius = max(min_radius, radius * alpha * 1.5)
            elif rho > 0.5:
                radius = min(1.0, radius * 2.0)
                flips = None if flips is None else flips * 2.0
        else:
            radius *= 0.3
            flips = None if flips is None else max(1.0, flips * 0.3)
        if radius < min_radius:
            break
    return SLPResult(x.reshape(1, -1), F, obj, mv, F0, it, time.perf_counter() - t0, lp_s, lin_s, hist)
