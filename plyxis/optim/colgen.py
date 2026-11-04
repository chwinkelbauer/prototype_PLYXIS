"""Column generation with exact DP sub-problems - the decomposition of docs/DECOMPOSITION.md, section 8.

A site is n blocks (assets) coupled only through shared limits per hour (a compressor, gas takeaway, a mill,
a fleet):

    maximise  sum_b v_b(z_b)     subject to  sum_b a_b(z_b) <= C   (m rows),   z_b in Z_b

Z_b is everything one asset can do over the horizon (its on/off pattern, hold times, restarts, set points);
v_b and a_b are its exact - possibly nonlinear - value and use of the shared limits. Dantzig-Wolfe:

    master LP    max sum_{b,j} lam_bj v_bj  s.t.  sum lam_bj a_bj <= C,  sum_j lam_bj = 1,  lam >= 0
                 over the schedules (columns) found so far; its row duals pi are prices of the shared limits
    pricing      every asset alone: max_z  v_b(z) - pi . a_b(z)   (exact DP over time, all assets in one pass)
    bound        L(pi) = pi . C + sum_b max_z [v_b(z) - pi . a_b(z)]  >=  the best plan, for every pi >= 0
    integer      the restricted master with one schedule per asset (HiGHS MILP over the columns found)

Cost per round: one LP plus one DP per asset - polynomial in assets x hours x (states per asset)^2;
exponential only in an asset's own state. The integer plan is within (rows + 1) x (largest single asset's
non-convexity) of the LP bound (Shapley-Folkman), and every solution reports its measured gap to the bound.
The number of rounds has no worst-case polynomial bound (it is measured, bench/colgen.py); dual smoothing
(Wentges) stabilises it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp


# ----------------------------------------------------------------------------- an asset family: unit commitment
@dataclass
class UnitFamily:
    """n assets of one kind over T hours, priced in one vectorised DP.

    on/off may change every `hold` hours (a restart costs `restart`); while on, a set point level is chosen per
    segment of `seg` hours from `levels`. value[b, t, l] and use[b, t, l, r] are the asset's exact value and use
    of shared resource r in hour t at level l (any nonlinearity is in these tables); off = 0 value and 0 use."""
    value: np.ndarray          # [n, T, L]
    use: np.ndarray            # [n, T, L, R]
    restart: np.ndarray        # [n]
    hold: int
    seg: int
    was_on: np.ndarray         # [n] bool: state before the horizon

    @property
    def n(self) -> int:
        return self.value.shape[0]

    def best_response(self, pi: np.ndarray) -> tuple:
        """pi [R, T] >= 0 -> for every asset its best schedule: (levels [n, S] with -1 = off, value [n], use [n, R, T],
        reduced value [n] = value - pi . use). Exact: DP over hold blocks, the best level per segment inside."""
        n, T, L = self.value.shape
        R = self.use.shape[3]
        S, K = T // self.seg, T // self.hold
        red = self.value - np.einsum("btlr,rt->btl", self.use, pi)          # [n, T, L]
        seg_red = red.reshape(n, S, self.seg, L).sum(2)                       # [n, S, L]
        lvl = seg_red.argmax(2)                                               # best level per segment if on
        best = np.take_along_axis(seg_red, lvl[..., None], 2)[..., 0]         # [n, S]
        per_k = self.hold // self.seg
        on_val = best.reshape(n, K, per_k).sum(2)                             # [n, K]
        del R
        # DP over hold blocks. cur_* = best value with the state before block k; it starts at the state before the horizon
        cur_off = np.where(self.was_on, -np.inf, 0.0)
        cur_on = np.where(self.was_on, 0.0, -np.inf)
        from_on = np.zeros((n, K, 2), dtype=np.int8)                         # predecessor of (block k, state) was on
        for k in range(K):
            new_off = np.maximum(cur_off, cur_on)                             # stopping is free
            from_on[:, k, 0] = cur_on > cur_off
            start = cur_off - self.restart                                    # a start costs the restart
            new_on = np.maximum(start, cur_on) + on_val[:, k]
            from_on[:, k, 1] = cur_on >= start
            cur_off, cur_on = new_off, new_on
        state = (cur_on > cur_off).astype(np.int8)
        on = np.zeros((n, K), dtype=bool)
        for k in range(K - 1, -1, -1):
            on[:, k] = state == 1
            state = from_on[np.arange(n), k, state]
        levels = np.where(np.repeat(on, per_k, 1), lvl, -1)                   # [n, S]
        return self.evaluate(levels, pi)

    def evaluate(self, levels: np.ndarray, pi: Optional[np.ndarray] = None) -> tuple:
        """Exact value and use of given schedules (levels [n, S], -1 = off)."""
        n, T, L = self.value.shape
        S = T // self.seg
        lv = np.repeat(levels, self.seg, 1)                                    # [n, T]
        on_t = lv >= 0
        li = np.clip(lv, 0, L - 1)
        val_t = np.take_along_axis(self.value, li[..., None], 2)[..., 0] * on_t
        use_t = np.take_along_axis(self.use, li[..., None, None].repeat(self.use.shape[3], 3), 2)[:, :, 0, :] * on_t[..., None]
        on_k = levels.reshape(n, T // self.hold, self.hold // self.seg)[:, :, 0] >= 0
        prev = np.concatenate([self.was_on[:, None], on_k[:, :-1]], 1)
        starts = (on_k & ~prev).sum(1)
        value = val_t.sum(1) - self.restart * starts
        use = use_t.transpose(0, 2, 1)                                           # [n, R, T]
        red = value - (np.einsum("brt,rt->b", use, pi) if pi is not None else 0.0)
        return levels, value, use, red


# ----------------------------------------------------------------------------- column generation
@dataclass
class CGResult:
    levels: np.ndarray            # chosen schedule per asset
    value: float                  # value of the integer plan
    violation: float              # largest excess over a shared limit (0 = feasible)
    bound: float                  # best Lagrangian upper bound
    lp: float                     # final master LP value
    gap: float                    # (bound - value) / |bound|
    rounds: int
    columns: int
    seconds: float
    history: list = field(default_factory=list)
    prices: Optional[np.ndarray] = None   # [R, T] prices of the shared limits at the best bound


def _dominant(lam: np.ndarray, cb: np.ndarray, n: int) -> np.ndarray:
    """Per asset the generated schedule with the largest LP weight."""
    pick = np.zeros(n, dtype=np.int64)
    best = np.full(n, -1.0)
    for j in range(len(lam)):
        if lam[j] > best[cb[j]]:
            best[cb[j]], pick[cb[j]] = lam[j], j
    return pick


def _repair_improve(pick: np.ndarray, V: np.ndarray, A: np.ndarray, cb: np.ndarray, cap: np.ndarray) -> np.ndarray:
    """Polynomial primal recovery. In a basic LP solution at most (rows) assets mix schedules, so the dominant
    schedules exceed the shared limits by little. Repair: switch the asset schedule that removes excess at the
    least value lost per unit of excess, until none is left (the all-off schedule always exists; about as many
    steps as assets with mixed schedules, <= rows). Improve: every asset's best swap that adds value and keeps
    every limit, found in one vectorised pass over the generated schedules (O(columns x rows)) and applied in
    order of gain while the limits still hold; a few passes."""
    pick = pick.copy()
    use = A[pick].sum(0)
    for _ in range(4 * len(pick) + 10):
        ex = np.maximum(use - cap, 0.0).sum()
        if ex <= 1e-9:
            break
        D = A - A[pick[cb]]                                   # change of use if each column replaced its asset's pick
        new_ex = np.maximum(use[None, :] + D - cap[None, :], 0.0).sum(1)
        red = ex - new_ex
        loss = V[pick[cb]] - V
        ok = red > 1e-12
        if not ok.any():
            break
        score = np.where(ok, loss / np.where(ok, red, 1.0), np.inf)
        j = int(np.argmin(score))
        use += D[j]
        pick[cb[j]] = j
    n = len(pick)
    for _ in range(20):                                       # improve: every asset's best swap at once, applied in order
        D = A - A[pick[cb]]
        gain = V - V[pick[cb]]
        feas = (use[None, :] + D <= cap[None, :] + 1e-9).all(1)
        gain = np.where(feas & (gain > 1e-9), gain, -np.inf)
        best = np.full(n, -np.inf)
        arg = np.full(n, -1)
        np.maximum.at(best, cb, gain)
        hit = np.nonzero(np.isfinite(gain) & (gain >= best[cb]))[0]
        arg[cb[hit]] = hit
        order = np.argsort(-best)
        applied = 0
        for b in order:
            j = arg[b]
            if j < 0 or not np.isfinite(best[b]):
                break
            d = A[j] - A[pick[b]]
            if (use + d <= cap + 1e-9).all():                # still within every limit after the swaps before it
                use += d
                pick[b] = j
                applied += 1
        if not applied:
            break
    return pick


def solve(fam: UnitFamily, cap: np.ndarray, rounds: int = 200, tol: float = 1e-4, alpha: float = 0.5,
          penalty: Optional[float] = None, integer: str = "auto", milp_seconds: float = 20.0,
          warm_pi: Optional[np.ndarray] = None) -> CGResult:
    """Column generation over one family and shared limits cap [R, T]. integer: "greedy" (polynomial), "milp" (the
    restricted master as a MILP) or "auto" (greedy, plus the MILP with 5 s while assets <= 4 x shared rows).
    warm_pi: prices to start from (the last plan's, or predicted - learnprice.py); they only seed the columns and
    the stability centre, so a poor guess costs rounds, never correctness."""
    t0 = time.perf_counter()
    n = fam.n
    R, T = cap.shape
    m = R * T
    big = penalty if penalty is not None else 10.0 * (np.abs(fam.value).max() + 1.0)
    cols_lv, cols_v, cols_a, cols_b = [], [], [], []
    seen = [set() for _ in range(n)]

    def add(bs, levels, value, use) -> int:
        """New schedules of assets bs (duplicates are skipped)."""
        k = 0
        for i, b in enumerate(bs):
            key = levels[i].tobytes()
            if key in seen[b]:
                continue
            seen[b].add(key)
            cols_lv.append(levels[i].copy())
            cols_v.append(float(value[i]))
            cols_a.append(use[i].reshape(-1).copy())
            cols_b.append(int(b))
            k += 1
        return k

    S = T // fam.seg
    off = np.full((n, S), -1, dtype=np.int64)
    every = np.arange(n)
    add(every, *fam.evaluate(off)[:3])                                # all off: always feasible
    add(every, *fam.best_response(np.zeros((R, T)))[:3])
    pi_stab, best_bound = np.zeros((R, T)), np.inf
    if warm_pi is not None:
        wp = np.maximum(np.asarray(warm_pi, dtype=np.float64).reshape(R, T), 0.0)
        lv, val, use, red = fam.best_response(wp)
        add(every, lv, val, use)
        pi_stab, best_bound = wp, float((wp * cap).sum() + red.sum())
    hist, lp_val = [], -np.inf
    capv = cap.reshape(-1)
    mu, lam_all, fresh = None, np.zeros(0), 0
    for it in range(1, rounds + 1):
        nc = len(cols_v)
        Vall, Aall, Ball = np.array(cols_v), np.array(cols_a), np.array(cols_b)
        # column management: the master holds the schedules in use, those near zero reduced cost at the last prices,
        # the newest and the all-off ones (the first n) - not every schedule ever generated, which made the LP from
        # scratch 94 % of the time on large sites. Pruned schedules come back when their reduced cost turns positive.
        if mu is not None and nc > 4 * n:
            keep = Vall - Aall @ pi_lp.reshape(-1) - mu[Ball] > -1e-3 * (1 + np.abs(Vall))
            keep[:n] = True
            keep[:len(lam_all)] |= lam_all > 1e-9
            keep[fresh:] = True
            idx = np.nonzero(keep)[0]
        else:
            idx = np.arange(nc)
        k = len(idx)
        A = sparse.csr_matrix(Aall[idx].T)                              # [m, k]
        A_ub = sparse.hstack([A, -sparse.identity(m)])
        E = sparse.csr_matrix((np.ones(k), (Ball[idx], np.arange(k))), shape=(n, k))
        A_eq = sparse.hstack([E, sparse.csr_matrix((n, m))])
        c = np.concatenate([-Vall[idx], np.full(m, big)])
        lp = linprog(c, A_ub=A_ub, b_ub=capv, A_eq=A_eq, b_eq=np.ones(n), bounds=(0, None), method="highs")
        lp_val = -lp.fun
        lam_all = np.zeros(nc)
        lam_all[idx] = lp.x[:k]
        pi_lp = np.maximum(-lp.ineqlin.marginals, 0.0).reshape(R, T)
        mu = -lp.eqlin.marginals
        # pruned schedules that are attractive at the new prices re-enter the master next round
        out = np.ones(nc, dtype=bool)
        out[idx] = False
        back = int((out & (Vall - Aall @ pi_lp.reshape(-1) - mu[Ball] > 1e-7 * (1 + np.abs(Vall)))).sum())
        fresh = nc
        # Wentges smoothing: price at a point between the stability centre and the LP duals
        found = 0
        for pi in ((alpha * pi_stab + (1 - alpha) * pi_lp) if it > 1 else pi_lp, pi_lp):
            levels, value, use, red = fam.best_response(pi)
            bound = float((pi * cap).sum() + red.sum())
            if bound < best_bound:
                best_bound, pi_stab = bound, pi
            red_lp = value - np.einsum("brt,rt->b", use, pi_lp) - mu         # reduced cost at the LP duals
            sel = np.nonzero(red_lp > 1e-7 * (1 + np.abs(value)))[0]
            if len(sel):
                found = add(sel, levels[sel], value[sel], use[sel])
            if found:
                break
        found += back
        hist.append({"round": it, "lp": lp_val, "bound": best_bound, "columns": len(cols_v), "master": k, "new": found,
                     "t": time.perf_counter() - t0})
        if not found or (best_bound - lp_val) <= tol * max(1.0, abs(best_bound)):
            break
    # integer plan: one generated schedule per asset
    nc = len(cols_v)
    V, Acol, cb = np.array(cols_v), np.array(cols_a), np.array(cols_b)
    picks = []
    if integer in ("greedy", "auto"):   # polynomial: the LP's dominant schedules, repaired and improved greedily
        picks.append(_repair_improve(_dominant(np.concatenate([lam_all, np.zeros(nc - len(lam_all))]), cb, n), V, Acol, cb, capv))
    if integer == "milp" or (integer == "auto" and n <= 4 * m):
        # few assets per shared row: Shapley-Folkman does not help (nearly every asset mixes schedules), but the
        # restricted master is small - solve it as a MILP (NP-hard in general, so with a time limit)
        A = sparse.csr_matrix(Acol.T)
        E = sparse.csr_matrix((np.ones(nc), (cb, np.arange(nc))), shape=(n, nc))
        cons = [LinearConstraint(sparse.hstack([A, -sparse.identity(m)]), -np.inf, capv),
                LinearConstraint(sparse.hstack([E, sparse.csr_matrix((n, m))]), 1, 1)]
        res = milp(np.concatenate([-V, np.full(m, big)]), constraints=cons, integrality=np.concatenate([np.ones(nc), np.zeros(m)]),
                   bounds=Bounds(0, np.concatenate([np.ones(nc), np.full(m, np.inf)])),
                   options={"time_limit": milp_seconds if integer == "milp" else min(milp_seconds, 5.0)})
        if res.x is not None:
            picks.append(_repair_improve(_dominant(res.x[:nc], cb, n), V, Acol, cb, capv))
    excess = lambda p: float(np.maximum(Acol[p].sum(0) - capv, 0.0).sum())  # noqa: E731
    pick = min(picks, key=lambda p: (excess(p) > 1e-9, -float(V[p].sum())))
    chosen = np.stack([cols_lv[j] for j in pick])
    _, value, use, _ = fam.evaluate(chosen)
    total = use.sum(0)
    viol = float(np.maximum(total - cap, 0.0).max())
    val = float(value.sum())
    return CGResult(chosen, val, viol, best_bound, lp_val, (best_bound - val) / max(1.0, abs(best_bound)),
                    len(hist), nc, time.perf_counter() - t0, hist, pi_stab.copy())


# ----------------------------------------------------------------------------- exact reference (small instances)
def exact_milp(fam: UnitFamily, cap: np.ndarray, seconds: float = 60.0) -> dict:
    """The same discrete model as one MILP (HiGHS): x[b,s,l] level choice, o[b,k] on, st[b,k] start."""
    t0 = time.perf_counter()
    n, T, L = fam.value.shape
    R = fam.use.shape[3]
    S, K = T // fam.seg, T // fam.hold
    per_k = fam.hold // fam.seg
    nx, no = n * S * L, n * K
    N = nx + 2 * no
    ix = lambda b, s, l: (b * S + s) * L + l          # noqa: E731
    io = lambda b, k: nx + b * K + k                  # noqa: E731
    ist = lambda b, k: nx + no + b * K + k            # noqa: E731
    seg_val = fam.value.reshape(n, S, fam.seg, L).sum(2)                    # [n, S, L]
    c = np.zeros(N)
    for b in range(n):
        for s in range(S):
            c[[ix(b, s, l) for l in range(L)]] = -seg_val[b, s]
        for k in range(K):
            c[ist(b, k)] = fam.restart[b]
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0
    for b in range(n):                                   # sum_l x[b,s,l] = o[b,k(s)]
        for s in range(S):
            for l in range(L):
                rows.append(r); cols.append(ix(b, s, l)); vals.append(1.0)
            rows.append(r); cols.append(io(b, s // per_k)); vals.append(-1.0)
            lo.append(0.0); hi.append(0.0); r += 1
        for k in range(K):                                # st >= o_k - o_{k-1}
            rows.append(r); cols.append(ist(b, k)); vals.append(1.0)
            rows.append(r); cols.append(io(b, k)); vals.append(-1.0)
            if k > 0:
                rows.append(r); cols.append(io(b, k - 1)); vals.append(1.0)
                lo.append(0.0)
            else:
                lo.append(-1.0 if fam.was_on[b] else 0.0)
            hi.append(np.inf); r += 1
    for rr in range(R):                                   # shared limits per hour
        for t in range(T):
            for b in range(n):
                for l in range(L):
                    u = fam.use[b, t, l, rr]
                    if u != 0.0:
                        rows.append(r); cols.append(ix(b, t // fam.seg, l)); vals.append(u)
            lo.append(-np.inf); hi.append(cap[rr, t]); r += 1
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(r, N))
    res = milp(c, constraints=[LinearConstraint(A, lo, hi)], integrality=np.ones(N), bounds=Bounds(0, 1),
               options={"time_limit": seconds, "mip_rel_gap": 1e-6})
    out = {"seconds": time.perf_counter() - t0, "status": int(res.status), "message": res.message}
    if res.x is not None:
        out["value"] = -float(res.fun)
        out["bound"] = -float(getattr(res, "mip_dual_bound", res.fun))
    return out


# ----------------------------------------------------------------------------- synthetic families (bench, tests)
def random_family(n: int, T: int = 24, L: int = 5, hold: int = 6, seg: int = 2, seed: int = 0,
                  cap_frac: float = 0.55, price_scale: float = 1.0) -> tuple:
    """Wells-like units: rate linear in speed, power ~ speed^3, wear ~ speed^4, gas ~ rate; one shared gas limit per
    hour at `cap_frac` of the gas all units could make. Returns (family, cap [1, T])."""
    rng = np.random.default_rng(seed)
    speed = np.linspace(0.35, 1.0, L)
    qmax = rng.uniform(300, 1500, n)
    gor = rng.lognormal(0.3, 0.5, n)
    price = 70 * price_scale * (1 + 0.1 * np.sin(np.arange(T) / T * 2 * np.pi + rng.uniform(0, 6.3)))[None, :, None]
    power = (40 + 30 * (np.arange(T) % 24 >= 14) * (np.arange(T) % 24 < 20))[None, :, None] / 1000
    rate = (qmax[:, None] * (0.22 + 0.78 * (speed - 0.35) / 0.65))[:, None, :] / 24       # bbl/h
    value = rate * price - 130 * speed[None, None, :] ** 3 * power * 24 / 24 - 3.0 * speed[None, None, :] ** 4
    use = np.broadcast_to((rate * gor[:, None, None])[..., None], (n, T, L, 1)).copy()      # Mscf/h of gas
    restart = rng.uniform(300, 900, n)
    cap = cap_frac * use[:, :, -1, 0].sum(0)[None, :] * rng.uniform(0.9, 1.1, T)[None, :]
    fam = UnitFamily(value=value.astype(np.float64), use=use.astype(np.float64), restart=restart, hold=hold, seg=seg,
                     was_on=rng.random(n) < 0.7)
    return fam, cap
