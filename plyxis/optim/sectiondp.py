"""Section dynamic programming on explicit table models - the mathematical core of the decomposition
(docs/DECOMPOSITION.md), independent of any twin.

A table model over blocks b = 1..n with finite option sets O_b:

    maximise   M(z) = sum_b u_b(z_b) + sum_{(a,b) in E} psi_ab(z_a, z_b)
    subject to sum_b r_hb(z_b) <= C_h        for every shared capacity h (crews, a mill, a fleet)

u_b are the blocks' exact input-output tables (a block = a section in a time window, solved exactly inside),
psi_ab the pairwise interactions between blocks, r_hb the use of a shared capacity.

    forest cover    E is covered by edge-disjoint forests F_1..F_K (greedy maximum-weight spanning forests,
                    weight = range of the interaction table): e.g. the time chains and the network trees.
    exact copies    copy k sees the forest F_k and a share theta_kb of every block's unary table; max-sum
                    dynamic programming solves it exactly in O(n q^2) (native kernel).
    prices          dual decomposition: the shares theta_kb (sum_k theta_kb = u_b - mu r_b) are prices that
                    make the copies agree, mu prices the shared capacities. For ANY theta, mu >= 0
                        D(theta, mu) = sum_k max_z [sum_b theta_kb(z_b) + sum_{e in F_k} psi_e] + mu C >= max M,
                    so the dual value is a certified upper bound; it is minimised by projected subgradient
                    steps (Polyak step sizes).
    primal          the copies' plans are repaired to the capacities and improved by block coordinate ascent
                    over all edges (exact per block, monotone; native kernel); the best plan is kept.
    certificate     gap = (best dual bound - best plan) / best dual bound, reported with every solution.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .. import native

NEG = -1e30


@dataclass
class TableModel:
    unary: list                              # per block: np.ndarray [n_options]
    edges: list                              # (a, b)
    pair: list                               # per edge: np.ndarray [n_a, n_b]
    usage: Optional[list] = None             # per hub: list per block of np.ndarray [n_options] (resource use)
    cap: Optional[np.ndarray] = None         # per hub capacity

    @property
    def n(self) -> int:
        return len(self.unary)

    def dense(self) -> "Dense":
        q = max(len(u) for u in self.unary)
        U = np.full((self.n, q), NEG)
        for b, u in enumerate(self.unary):
            U[b, :len(u)] = u
        m = len(self.edges)
        E = np.array(self.edges, dtype=np.int32).reshape(m, 2)
        P = np.zeros((max(m, 1), q, q))
        for k, Pk in enumerate(self.pair):
            P[k, :Pk.shape[0], :Pk.shape[1]] = Pk
        H = len(self.usage) if self.usage else 0
        R = np.zeros((max(H, 1), self.n, q))
        for h in range(H):
            for b in range(self.n):
                R[h, b, :len(self.usage[h][b])] = self.usage[h][b]
        cap = np.asarray(self.cap, dtype=np.float64) if H else np.zeros(1)
        return Dense(U, E, P, R, cap, H)


@dataclass
class Dense:
    U: np.ndarray            # [n, q] (NEG = no such option)
    E: np.ndarray            # [m, 2]
    P: np.ndarray            # [m, q, q]
    R: np.ndarray            # [H, n, q]
    cap: np.ndarray          # [H]
    H: int

    @property
    def n(self) -> int:
        return self.U.shape[0]

    def value(self, z: np.ndarray) -> float:
        v = float(self.U[np.arange(self.n), z].sum())
        if len(self.E):
            v += float(self.P[np.arange(len(self.E)), z[self.E[:, 0]], z[self.E[:, 1]]].sum())
        return v

    def use(self, z: np.ndarray) -> np.ndarray:
        if not self.H:
            return np.zeros(0)
        return self.R[:self.H, np.arange(self.n), z].sum(1)

    def feasible(self, z: np.ndarray, tol: float = 1e-9) -> bool:
        return not self.H or bool(np.all(self.use(z) <= self.cap[:self.H] + tol))


@dataclass
class Solution:
    z: np.ndarray
    value: float
    feasible: bool
    upper_bound: float
    seconds: float
    info: dict = field(default_factory=dict)

    @property
    def gap(self) -> float:
        return (self.upper_bound - self.value) / max(abs(self.upper_bound), 1e-12)


# ----------------------------------------------------------------------------- evaluation (TableModel API)
def value(m: TableModel, z: np.ndarray) -> float:
    v = sum(float(m.unary[b][z[b]]) for b in range(m.n))
    return v + sum(float(m.pair[k][z[a], z[b]]) for k, (a, b) in enumerate(m.edges))


def usage(m: TableModel, z: np.ndarray) -> np.ndarray:
    if not m.usage:
        return np.zeros(0)
    return np.array([sum(float(r[b][z[b]]) for b in range(m.n)) for r in m.usage])


def feasible(m: TableModel, z: np.ndarray, tol: float = 1e-9) -> bool:
    return not m.usage or bool(np.all(usage(m, z) <= np.asarray(m.cap) + tol))


# ----------------------------------------------------------------------------- forests
class Forest:
    """A forest of the interaction graph with its traversal precomputed for the DP kernel."""

    def __init__(self, n: int, E: np.ndarray, ids: np.ndarray):
        self.ids = np.asarray(ids, dtype=np.int64)
        adj = [[] for _ in range(n)]
        for i, k in enumerate(self.ids):
            a, b = int(E[k, 0]), int(E[k, 1])
            adj[a].append((b, i, 0))
            adj[b].append((a, i, 1))
        order, par, pe, fl, seen = [], [-1] * n, [0] * n, [0] * n, [False] * n
        for r in range(n):
            if seen[r]:
                continue
            seen[r] = True
            stack = [r]
            while stack:
                v = stack.pop()
                order.append(v)
                for w, i, f in adj[v]:
                    if not seen[w]:
                        seen[w] = True
                        par[w], pe[w], fl[w] = v, i, f
                        stack.append(w)
        self.post = np.array(order[::-1], dtype=np.int32)
        self.parent = np.array(par, dtype=np.int32)
        self.pedge = np.array(pe, dtype=np.int32)
        self.flip = np.array(fl, dtype=np.uint8)

    def argmax(self, U: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Exact argmax of sum_b U[b, z_b] + sum_{k in forest} P[k, z_a, z_b]."""
        Pf = np.ascontiguousarray(P[self.ids]) if len(self.ids) else np.zeros((1,) + P.shape[1:])
        return native.forest_dp(U, Pf, self.post, self.parent, self.pedge, self.flip).astype(np.int64)


def edge_weights(d: Dense) -> np.ndarray:
    if not len(d.E):
        return np.zeros(0)
    cross = d.P - d.P[:, :, :1] - d.P[:, :1, :] + d.P[:, :1, :1]
    return cross.max((1, 2)) - cross.min((1, 2))


def forest_cover(d: Dense, max_forests: int = 6) -> tuple:
    """Edge-disjoint maximum-weight spanning forests until every edge is covered (or max_forests); the
    edges left over are bounded by sum max psi_e in the certificate."""
    w = edge_weights(d)
    left = np.argsort(-w, kind="stable")
    forests = []
    while len(left) and len(forests) < max_forests:
        parent = np.arange(d.n)

        def root(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        take, rest = [], []
        for k in left:
            ra, rb = root(int(d.E[k, 0])), root(int(d.E[k, 1]))
            if ra == rb:
                rest.append(k)
            else:
                parent[ra] = rb
                take.append(k)
        forests.append(np.array(take, dtype=np.int64))
        left = np.array(rest, dtype=np.int64)
    return forests, left


# ----------------------------------------------------------------------------- local search and repair
def coordinate_ascent(d, z: np.ndarray, sweeps: int = 30) -> np.ndarray:
    """Block coordinate ascent over all edges (each block takes its exact best option given its neighbours,
    within the capacities left by the others). Monotone; stops at a block-wise optimum."""
    if isinstance(d, TableModel):
        d = d.dense()
    return native.coordinate_ascent(d.U, d.E, d.P, d.R[:d.H] if d.H else np.zeros((0,) + d.U.shape),
                                    d.cap[:d.H] if d.H else np.zeros(0), z, sweeps)


def repair(d, z: np.ndarray, rounds: int = 8) -> np.ndarray:
    """Greedy capacity repair: switch options in the order of value lost per unit of capacity freed (value
    change incl. neighbours, ranked once per round)."""
    if isinstance(d, TableModel):
        d = d.dense()
    z = z.copy()
    if not d.H:
        return z
    n, q = d.U.shape
    for _ in range(rounds):
        over = d.use(z) - d.cap[:d.H]
        h = int(np.argmax(over))
        if over[h] <= 1e-9:
            break
        s = d.U.copy()                                         # value of every option given the neighbours
        if len(d.E):
            a, b = d.E[:, 0], d.E[:, 1]
            np.add.at(s, a, d.P[np.arange(len(a)), :, z[b]])
            np.add.at(s, b, d.P[np.arange(len(a)), z[a], :])
        cur = s[np.arange(n), z]
        freed = d.R[h, np.arange(n), z][:, None] - d.R[h]
        ratio = np.where((freed > 1e-12) & (d.U > NEG / 2), (cur[:, None] - s) / np.maximum(freed, 1e-12), np.inf)
        j = ratio.argmin(1)
        best = ratio[np.arange(n), j]
        for blk in np.argsort(best, kind="stable"):
            if over[h] <= 1e-9 or not np.isfinite(best[blk]):
                break
            jj = int(j[blk])
            over += d.R[:d.H, blk, jj] - d.R[:d.H, blk, z[blk]]
            z[blk] = jj
    return z


# ----------------------------------------------------------------------------- exact large-neighbourhood moves
class _Graph:
    def __init__(self, d: Dense):
        n, m = d.n, len(d.E)
        deg = np.bincount(d.E.reshape(-1), minlength=n) if m else np.zeros(n, dtype=np.int64)
        self.start = np.concatenate([[0], np.cumsum(deg)]).astype(np.int64)
        ends = np.concatenate([d.E[:, 0], d.E[:, 1]]) if m else np.zeros(0, dtype=np.int64)
        other = np.concatenate([d.E[:, 1], d.E[:, 0]]) if m else np.zeros(0, dtype=np.int64)
        order = np.argsort(ends, kind="stable")
        self.nb = other[order].astype(np.int64)


def induced_forest(d: Dense, g: _Graph, rng) -> np.ndarray:
    """A random set S of blocks whose induced interaction graph is a forest (greedy, union-find): conditioned
    on the blocks outside S, the plan value over S is a forest problem - solved exactly by DP."""
    n = d.n
    parent = np.arange(n)
    inS = np.zeros(n, dtype=bool)

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for v in rng.permutation(n):
        roots = [root(int(w)) for w in g.nb[g.start[v]:g.start[v + 1]] if inS[w]]
        if len(roots) != len(set(roots)):
            continue                                   # two links into one tree would close a cycle
        inS[v] = True
        for r in roots:
            parent[r] = v
    return inS


def forest_move(d: Dense, z: np.ndarray, inS: np.ndarray, rng=None, price_steps: int = 14) -> np.ndarray:
    """Exact re-optimisation of the blocks in S (an induced forest) given the others; shared capacities by
    bisection on their price (one hub) - the best feasible plan found is returned."""
    n = d.n
    a, b = d.E[:, 0], d.E[:, 1]
    U = d.U.copy()
    if len(d.E):
        ia, ib = inS[a], inS[b]
        k1 = np.nonzero(ia & ~ib)[0]                       # links from S to fixed blocks: unary terms
        np.add.at(U, a[k1], d.P[k1, :, z[b[k1]]])
        k2 = np.nonzero(~ia & ib)[0]
        np.add.at(U, b[k2], d.P[k2, z[a[k2]], :])
        inner = np.nonzero(ia & ib)[0]
    else:
        inner = np.zeros(0, dtype=np.int64)
    fixed = np.nonzero(~inS)[0]
    U[fixed] = NEG
    U[fixed, z[fixed]] = 0.0                               # fixed blocks keep their option
    f = Forest(n, d.E, inner)

    def best_for(mu):
        Uk = U - (np.tensordot(mu, d.R[:d.H], 1) if d.H else 0.0)
        Uk = np.where(d.U > NEG / 2, Uk, NEG)
        Uk[fixed] = NEG
        Uk[fixed, z[fixed]] = 0.0
        return f.argmax(Uk, d.P)
    if not d.H:
        return best_for(np.zeros(0))
    zb = z.copy()
    v0 = d.value(z) if d.feasible(z) else -np.inf
    zt = best_for(np.zeros(d.H))
    if d.feasible(zt):
        return zt
    lo, hi = 0.0, 1.0
    for _ in range(40):                                    # bracket the price that meets the capacity
        zt = best_for(np.full(d.H, hi))
        if d.feasible(zt):
            break
        lo, hi = hi, 2 * hi
    for _ in range(price_steps):
        mid = 0.5 * (lo + hi)
        zt = best_for(np.full(d.H, mid))
        if d.feasible(zt):
            hi = mid
            v = d.value(zt)
            if v > v0:
                zb, v0 = zt, v
        else:
            lo = mid
    zt = best_for(np.full(d.H, hi))
    if d.feasible(zt) and d.value(zt) > v0:
        zb = zt
    return zb


# ----------------------------------------------------------------------------- solver
def dual_bound(d: Dense, forests: list, off: float, iters: int, target: float, primal=None,
               time_left: Optional[float] = None) -> tuple:
    """Dual decomposition into exactly solved forests + capacity prices; returns (best upper bound, prices,
    history). `primal(z)` is offered the copies' plans (majority vote) every few iterations."""
    t0 = time.perf_counter()
    n, q = d.U.shape
    K, H = len(forests), d.H
    valid = d.U > NEG / 2
    mu = np.zeros(H)
    theta = np.repeat((np.where(valid, d.U, 0.0) / K)[None], K, 0)   # shares of the unary tables
    best_ub, ubs = np.inf, []
    for it in range(iters):
        zs = []
        dual = float(mu @ d.cap[:H]) if H else 0.0
        for k, f in enumerate(forests):
            Uk = np.where(valid, theta[k], NEG)
            z = f.argmax(Uk, d.P)
            zs.append(z)
            dual += float(Uk[np.arange(n), z].sum())
            if len(f.ids):
                dual += float(d.P[f.ids, z[d.E[f.ids, 0]], z[d.E[f.ids, 1]]].sum())
        ub = dual + off
        ubs.append(ub)
        best_ub = min(best_ub, ub)
        if primal is not None and it % 10 == 0:
            Z = np.stack(zs)
            vote = np.array([np.bincount(Z[:, b], minlength=q).argmax() for b in range(n)]) if K > 1 else zs[0]
            target = max(target, primal(vote))
        if best_ub - target <= 1e-9 * max(1.0, abs(best_ub)):
            break
        if time_left is not None and time.perf_counter() - t0 > time_left:
            break
        ind = np.zeros((K, n, q))                                # subgradients: the copies' disagreement ...
        for k, z in enumerate(zs):
            ind[k, np.arange(n), z] = 1.0
        g_theta = ind - ind.mean(0, keepdims=True)
        g_mu = (d.cap[:H] - np.mean([d.use(z) for z in zs], 0)) if H else np.zeros(0)   # ... and the capacity
        norm2 = float((g_theta ** 2).sum() + (g_mu ** 2).sum())
        if norm2 <= 1e-12:
            break
        step = max(ub - target, 1e-6 * max(1.0, abs(ub))) / norm2           # Polyak step towards the best plan
        theta -= step * g_theta
        if H:
            new_mu = np.maximum(0.0, mu - step * g_mu)
            theta -= ((new_mu - mu) @ d.R[:H].reshape(H, -1)).reshape(n, q)[None] / K
            mu = new_mu
    return best_ub, mu, ubs


def smooth_dual_bound(d: Dense, forests: list, off: float, taus=(1.0, 0.3, 0.1, 0.03), iters: int = 80,
                      primal=None, time_left: Optional[float] = None) -> tuple:
    """Smoothed dual decomposition (Jojic, Gould & Koller 2010; Savchynskyy et al. 2011): every copy's max is
    replaced by tau * log-sum-exp - still an upper bound for every tau > 0, but smooth, with the copies'
    marginals as gradient - and minimised by L-BFGS-B over the shares (lambda) and the capacity prices (mu >= 0),
    with tau lowered in stages. Every value evaluated is a valid bound, and so is the plain dual (tau -> 0) at
    each stage's optimum; the best of them is returned."""
    from scipy.optimize import minimize
    t0 = time.perf_counter()
    n, q = d.U.shape
    K, H = len(forests), d.H
    valid = d.U > NEG / 2
    base = np.where(valid, d.U, 0.0)
    Uv = np.where(valid, d.U, np.nan)
    scale = float(np.median(np.nanmax(Uv, 1) - np.nanmin(Uv, 1))) or 1.0     # temperatures relative to this
    Pf = [np.ascontiguousarray(d.P[f.ids]) if len(f.ids) else np.zeros((1, q, q)) for f in forests]
    nl = (K - 1) * n * q
    best = {"ub": np.inf, "x": np.zeros(nl + H)}

    def shares(z):
        lam = z[:nl].reshape(K - 1, n, q) if K > 1 else np.zeros((0, n, q))
        mu = np.maximum(z[nl:], 0.0)
        th = np.empty((K, n, q))
        th[:K - 1] = base / K + lam
        th[K - 1] = base / K - lam.sum(0)
        if H:
            th -= (np.tensordot(mu, d.R[:H], 1) / K)[None]
        return th, mu

    def fun(z, tau):
        th, mu = shares(z)
        val, margs = float(mu @ d.cap[:H]) if H else 0.0, []
        for k, f in enumerate(forests):
            v, mg = native.forest_logz(np.where(valid, th[k], NEG), Pf[k], f.post, f.parent, f.pedge, f.flip, tau)
            val += v
            margs.append(mg)
        val += off
        if val < best["ub"]:
            best["ub"], best["x"] = val, z.copy()
        g = np.zeros_like(z)
        if K > 1:
            g[:nl] = (np.stack(margs[:K - 1]) - margs[K - 1][None]).reshape(-1)
        if H:
            g[nl:] = d.cap[:H] - sum(np.tensordot(d.R[:H], mg, axes=([1, 2], [0, 1])) for mg in margs) / K
        return val, g

    def plain(z):
        th, mu = shares(z)
        val = float(mu @ d.cap[:H]) if H else 0.0
        zs = []
        for k, f in enumerate(forests):
            Uk = np.where(valid, th[k], NEG)
            zz = f.argmax(Uk, d.P)
            zs.append(zz)
            val += float(Uk[np.arange(n), zz].sum())
            if len(f.ids):
                val += float(d.P[f.ids, zz[d.E[f.ids, 0]], zz[d.E[f.ids, 1]]].sum())
        return val + off, zs
    z = np.zeros(nl + H)
    hist = []
    bounds = [(None, None)] * nl + [(0.0, None)] * H
    for tau in taus:
        if time_left is not None and time.perf_counter() - t0 > time_left:
            break
        res = minimize(fun, z, args=(tau * scale,), jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": iters})
        z = res.x
        ub0, zs = plain(z)
        best["ub"] = min(best["ub"], ub0)
        hist.append({"tau": tau, "smooth": float(res.fun), "plain": ub0})
        if primal is not None:
            Z = np.stack(zs)
            vote = np.array([np.bincount(Z[:, b], minlength=q).argmax() for b in range(n)]) if K > 1 else zs[0]
            primal(vote)
    return best["ub"], hist


def solve(m, iters: int = 150, cd_sweeps: int = 30, max_forests: int = 6, lns: int = 30, seed: int = 0,
          time_budget: Optional[float] = None, dual: str = "subgradient") -> Solution:
    """Forest DP start -> exact induced-forest moves (LNS) -> dual decomposition for the certified bound.
    Accepts a TableModel or its Dense form."""
    t0 = time.perf_counter()
    d = m.dense() if isinstance(m, TableModel) else m
    n, q = d.U.shape
    forests_ids, left = forest_cover(d, max_forests)
    forests = [Forest(n, d.E, ids) for ids in forests_ids] or [Forest(n, d.E, np.zeros(0, dtype=np.int64))]
    off = float(d.P[left].max((1, 2)).sum()) if len(left) else 0.0
    best = {"z": None, "v": -np.inf}

    def offer(z):
        z = coordinate_ascent(d, repair(d, z), cd_sweeps)
        if d.feasible(z):
            v = d.value(z)
            if v > best["v"]:
                best["z"], best["v"] = z, v
        return best["v"]
    offer(forests[0].argmax(d.U, d.P))                           # the strongest forest, solved exactly
    rng = np.random.default_rng(seed)
    g = _Graph(d)
    lns_done = 0
    for it in range(lns):                                        # exact moves over random induced forests
        if best["z"] is None or (time_budget is not None and time.perf_counter() - t0 > 0.6 * time_budget):
            break
        zn = forest_move(d, best["z"], induced_forest(d, g, rng), rng)
        if (it + 1) % 5 == 0:
            zn = coordinate_ascent(d, zn, cd_sweeps)
        lns_done += 1
        if d.feasible(zn) and d.value(zn) > best["v"] + 1e-12:
            best["z"], best["v"] = zn, d.value(zn)
    left_t = None if time_budget is None else max(0.0, time_budget - (time.perf_counter() - t0))
    if dual == "smooth":
        ub, sh = smooth_dual_bound(d, forests, off, primal=offer, time_left=left_t)
        mu, ubs = np.zeros(d.H), [h["plain"] for h in sh]
    else:
        ub, mu, ubs = dual_bound(d, forests, off, iters, best["v"], primal=offer, time_left=left_t)
    if best["z"] is None:
        offer(np.zeros(n, dtype=np.int64))
    return Solution(best["z"], best["v"], d.feasible(best["z"]), max(ub, best["v"]), time.perf_counter() - t0,
                    {"forests": len(forests), "forest_edges": int(sum(len(f.ids) for f in forests)),
                     "edges_left": int(len(left)), "off_forest_bound": off, "dual_iterations": len(ubs),
                     "lns": lns_done, "prices": mu.tolist(),
                     "upper_bounds": [round(u, 6) for u in ubs[:: max(1, len(ubs) // 20)]]})
