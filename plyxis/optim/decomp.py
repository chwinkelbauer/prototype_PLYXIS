"""Section decomposition of a twin's discrete decisions - sections as exact input-output functions,
coordinated by dynamic programming and prices. Theory and proofs: docs/DECOMPOSITION.md.

The discrete decisions (which stopes fire in which shift, which wells run, compressor units, line-ups)
are grouped into *blocks* = (section, time window); a section is a set of assets whose decisions interact
strongly (found by the learned graph filter, sections.py). Around the incumbent plan the penalised plan
value F is modelled as

    M(z) = sum_b u_b(z_b) + sum_{(b,c) in T} psi_bc(z_b, z_c)

    u_b      unary table of block b: for every joint option of the block, the EXACT change of the penalised
             terms of the section's region (its assets plus the assets only it feeds - the "halo"), plus the
             change of the rest of the site through prices (the gradient of everything outside the region:
             shared capacities such as the mill, hoist, fleets, fans are coordinated by their marginal value).
             This table is the section's deterministic input-output function: given prices, its best
             response is an argmax over a table of <= 64 entries.
    psi_bc   exact pairwise interaction table of two coupled blocks (cross term of their joint options),
             for the edges of a maximum-weight spanning forest T of the block interaction graph (time edges:
             the same section in consecutive windows; space edges: neighbouring sections in one window).

Max-sum dynamic programming on the forest maximises M exactly in O(sum_edges q^2) - linear in the size of
the site, exponential only inside a block. The chosen plan is verified on the full twin (backtracking by
forest component), and the expansion is repeated around the new incumbent until no block improves.

Tables are evaluated in batched simulations with exact attribution: blocks whose sections are not
neighbours vary in the same sample, and each one's effect is read off its own region's terms
(`Problem.per_instance`). Error of the model: interactions on edges outside the forest, and the
curvature of the rest of the site (priced linearly); both are bounded in DECOMPOSITION.md and the
verification step makes every accepted move a real improvement.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

from .. import native
from . import sections as SX
from .problem import Problem

MU = SX.MU
NEG = -1e30


@dataclass
class DecompResult:
    x: torch.Tensor
    objective: float
    max_violation: float
    seconds: float
    evals: int
    sweeps: int
    history: list
    info: dict = field(default_factory=dict)


class Decomposer:
    """Blocks, regions, interaction forest and evaluation plans of one twin structure (built once, reused
    by every re-plan); `solve` runs the table / DP / verify sweeps for an operating context."""

    def __init__(self, pb: Problem, lay: SX.Layout, section_of: np.ndarray, g: SX.SiteGraph,
                 strength: dict, budget: int = 64, q: int = 6, hub_sections: int = 12):
        self.pb, self.lay, self.g, self.q = pb, lay, g, q
        n_cols = pb.n_inst
        sec = np.full(n_cols, -1, dtype=np.int32)
        sec[:len(section_of)] = section_of
        self.C = int(sec.max()) + 1 if (sec >= 0).any() else 0
        adj = [set() for _ in range(g.n)]
        for a, b in g.edges:
            adj[int(a)].add(int(b))
            adj[int(b)].add(int(a))
        # regions: a section's assets plus the assets that only it touches (halo); assets touched by several
        # sections are shared (their interaction is measured on the forest edges), hubs are priced
        region = sec.copy()
        self.shared: dict = {}
        hubs = set()
        for i in range(g.n):
            if sec[i] >= 0:
                continue
            touch = {int(sec[j]) for j in adj[i] if sec[j] >= 0}
            if len(touch) == 1:
                region[i] = next(iter(touch))
            elif len(touch) > hub_sections:
                hubs.add(i)
            elif len(touch) > 1:
                for c in touch:
                    for d in touch:
                        if c < d:
                            self.shared.setdefault((c, d), []).append(i)
        self.sec, self.region, self.hubs = sec, region, hubs
        # section adjacency: regions touching each other or sharing a non-hub asset
        nb = [set() for _ in range(self.C)]
        for a, b in g.edges:
            ra, rb = region[int(a)], region[int(b)]
            if ra >= 0 and rb >= 0 and ra != rb:
                nb[ra].add(rb)
                nb[rb].add(ra)
        for (c, d) in self.shared:
            nb[c].add(d)
            nb[d].add(c)
        self.sec_nb = nb
        self.region_cols = [np.nonzero(region[:g.n] == c)[0] for c in range(self.C)]
        # blocks = (section, window) with their joint options (option 0 = keep the incumbent)
        units_of: dict = {}
        for u in lay.units:
            c = int(sec[u.col])
            if c >= 0:
                units_of.setdefault((c, u.window), []).append(u)
        self.blocks: list = []                  # (section, window)
        self.opts: list = []                    # per block: list of option lists [(x index, value)]
        self.bidx: dict = {}
        import itertools
        for (c, w), us in sorted(units_of.items()):
            combos = list(itertools.product(*[u.options for u in us]))
            if len(combos) > budget:
                rng = np.random.default_rng(c * 131 + w)
                combos = [combos[i] for i in rng.choice(len(combos), budget, replace=False)]
            self.bidx[(c, w)] = len(self.blocks)
            self.blocks.append((c, w))
            self.opts.append([list(itertools.chain.from_iterable(cb)) for cb in combos])
        self.nb_blocks = len(self.blocks)
        self.dec_sec = np.where(lay.dec_col >= 0, sec[np.clip(lay.dec_col, 0, n_cols - 1)], -1).astype(np.int32)
        # block interaction graph: time edges (same section, next window) and space edges (neighbouring
        # sections, same window), weighted by the interaction strength (measured / predicted |cross term|)
        cand = []
        for (c, w), b in self.bidx.items():
            if (c, w + 1) in self.bidx:
                cand.append((strength.get(("t", c), 0.0), b, self.bidx[(c, w + 1)], "time"))
            for d in nb[c]:
                if d > c and (d, w) in self.bidx:
                    cand.append((strength.get((min(c, d), max(c, d)), 0.0), b, self.bidx[(d, w)], "space"))
        cand.sort(key=lambda e: -e[0])
        parent = list(range(self.nb_blocks))

        def root(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        self.forest: list = []                  # (block a, block b, weight, kind)
        self.off_forest_weight = 0.0
        for wgt, a, b, kind in cand:            # Kruskal: maximum-weight spanning forest
            ra, rb = root(a), root(b)
            if ra == rb:
                self.off_forest_weight += wgt
                continue
            parent[ra] = rb
            self.forest.append((a, b, wgt, kind))
        self.forest_weight = float(sum(e[2] for e in self.forest))
        self.unary_plan = self._unary_plan()
        self.pair_plan = self._pair_plan()

    # ------------------------------------------------------------------ evaluation plans
    def _closed(self, secs) -> set:
        out = set(secs)
        for c in secs:
            out |= self.sec_nb[c]
        return out

    def _unary_plan(self) -> list:
        """Greedy colouring of blocks: one batch may vary blocks of distinct, non-neighbouring sections."""
        classes: list = []
        for b in sorted(range(self.nb_blocks), key=lambda b: -len(self.sec_nb[self.blocks[b][0]])):
            c = self.blocks[b][0]
            for cl in classes:
                if c not in cl["blocked"]:
                    cl["blocks"].append(b)
                    cl["blocked"] |= self._closed([c])
                    break
            else:
                classes.append({"blocks": [b], "blocked": self._closed([c])})
        return [cl["blocks"] for cl in classes]

    def _pair_plan(self) -> list:
        """Forest edges evaluated in one batch when their sections and neighbourhoods are disjoint."""
        classes: list = []
        for e, (a, b, _, _) in enumerate(self.forest):
            secs = {self.blocks[a][0], self.blocks[b][0]}
            for cl in classes:
                if not (secs & cl["blocked"]):
                    cl["edges"].append(e)
                    cl["blocked"] |= self._closed(secs)
                    break
            else:
                classes.append({"edges": [e], "blocked": self._closed(secs)})
        return [cl["edges"] for cl in classes]

    # ------------------------------------------------------------------ objective pieces
    def _F(self, X: torch.Tensor):
        e = self.pb.evaluate(X, relax=False)
        return e, e.objective / self.pb.J_scale - MU / self.pb.T * e.sum_violation()

    def rest_gradient(self, x: torch.Tensor) -> np.ndarray:
        """Prices of the rest of the site for every decision: dF/dx minus d(own region terms)/dx, one
        backward pass per colour of the section graph (regions of one colour are disjoint and apart)."""
        pb = self.pb
        xg = x.detach().clone().requires_grad_(True)
        _, F = self._F(xg)
        g_tot = torch.autograd.grad(F.sum(), xg)[0][0].detach()
        own = torch.zeros_like(g_tot)
        colour = [-1] * self.C
        for c in sorted(range(self.C), key=lambda c: -len(self.sec_nb[c])):
            used = {colour[d] for d in self.sec_nb[c] if colour[d] >= 0}
            k = 0
            while k in used:
                k += 1
            colour[c] = k
        dec_sec = torch.as_tensor(self.dec_sec, dtype=torch.long)
        col_of = torch.as_tensor(colour + [-1], dtype=torch.long)
        for k in range(max(colour) + 1 if colour else 0):
            secs = [c for c in range(self.C) if colour[c] == k]
            cols = np.concatenate([self.region_cols[c] for c in secs]) if secs else np.zeros(0, dtype=np.int64)
            if len(cols) == 0:
                continue
            xg = x.detach().clone().requires_grad_(True)
            e = pb.evaluate(xg, relax=False)
            pi = pb.per_instance(e, MU)[0]
            gk = torch.autograd.grad(pi[torch.as_tensor(cols)].sum(), xg, allow_unused=True)[0]
            if gk is None:
                continue
            mine = (dec_sec >= 0) & (col_of[dec_sec.clamp(min=0)] == k)
            own = torch.where(mine, gk[0], own)
        return (g_tot - own).numpy()

    def tables(self, x: torch.Tensor, g_rest: np.ndarray, active: Optional[np.ndarray] = None) -> tuple:
        """Unary tables of all (active) blocks and pairwise tables of the forest edges around x."""
        pb = self.pb
        unary = [np.zeros(1 + len(o)) for o in self.opts]
        n_eval = 0
        with torch.no_grad():
            for cls in self.unary_plan:
                blocks = [b for b in cls if active is None or active[b]]
                if not blocks:
                    continue
                B = 1 + max(len(self.opts[b]) for b in blocks)
                X = x.repeat(B, 1)
                for b in blocks:
                    for j, opt in enumerate(self.opts[b], start=1):
                        for i, v in opt:
                            X[j, i] = v
                e = pb.evaluate(X, relax=False)
                n_eval += B
                dpi = (pb.per_instance(e, MU) - pb.per_instance(e, MU)[0:1]).numpy()
                dx = (X - X[0:1]).numpy()
                for b in blocks:
                    c = self.blocks[b][0]
                    n = len(self.opts[b])
                    own = dpi[:n + 1][:, self.region_cols[c]].sum(1)
                    idx = np.array(sorted({i for o in self.opts[b] for i, _ in o}), dtype=np.int64)
                    lin = dx[:n + 1][:, idx] @ g_rest[idx]
                    unary[b] = own + lin
            # top-q options of every block (always including "keep")
            top = []
            for b in range(self.nb_blocks):
                order = np.argsort(-unary[b][1:], kind="stable")[:self.q - 1] + 1
                top.append(np.concatenate([[0], order]).astype(np.int64))
            pair = [None] * len(self.forest)
            for cls in self.pair_plan:
                edges = [e for e in cls if active is None or (active[self.forest[e][0]] or active[self.forest[e][1]])]
                if not edges:
                    continue
                B = max(len(top[self.forest[e][0]]) * len(top[self.forest[e][1]]) for e in edges)
                X = x.repeat(B, 1)
                for e in edges:
                    a, b = self.forest[e][0], self.forest[e][1]
                    qa, qb = len(top[a]), len(top[b])
                    for r in range(qa * qb):
                        i, j = divmod(r, qb)
                        for blk, oi in ((a, int(top[a][i])), (b, int(top[b][j]))):
                            if oi > 0:
                                for k, v in self.opts[blk][oi - 1]:
                                    X[r, k] = v
                ev = pb.evaluate(X, relax=False)
                n_eval += B
                dpi = (pb.per_instance(ev, MU) - pb.per_instance(ev, MU)[0:1]).numpy()
                for e in edges:
                    a, b = self.forest[e][0], self.forest[e][1]
                    ca, cb = self.blocks[a][0], self.blocks[b][0]
                    cols = [self.region_cols[ca]] if ca == cb else \
                        [self.region_cols[ca], self.region_cols[cb],
                         np.asarray(self.shared.get((min(ca, cb), max(ca, cb)), []), dtype=np.int64)]
                    cols = np.concatenate(cols)
                    qa, qb = len(top[a]), len(top[b])
                    v = dpi[:qa * qb][:, cols].sum(1).reshape(qa, qb)
                    pair[e] = v - v[:, :1] - v[:1, :] + v[0, 0]
        return unary, top, pair, n_eval

    # ------------------------------------------------------------------ dynamic programme
    def dp(self, unary: list, top: list, pair: list, active: Optional[np.ndarray] = None) -> tuple:
        """Exact maximiser of M on the forest. Returns (option per block, group per block, predicted gain per
        group); a group is a connected forest component (or a single block without forest edges)."""
        use = [e for e in range(len(self.forest)) if pair[e] is not None]
        nodes = sorted({self.forest[e][k] for e in use for k in (0, 1)})
        in_forest = np.zeros(self.nb_blocks, dtype=bool)
        in_forest[nodes] = True
        choice = np.zeros(self.nb_blocks, dtype=np.int64)
        group = np.full(self.nb_blocks, -1, dtype=np.int64)
        gain: dict = {}
        for b in range(self.nb_blocks):                          # blocks without forest edges: plain argmax
            if (active is None or active[b]) and not in_forest[b]:
                j = int(np.argmax(unary[b]))
                choice[b], group[b] = j, b
                gain[b] = float(unary[b][j] - unary[b][0])
        if not use:
            return choice, group, gain
        loc = {b: k for k, b in enumerate(nodes)}
        q = self.q
        U = np.full((len(nodes), q), NEG, dtype=np.float64)
        for b in nodes:
            t = top[b]
            U[loc[b], :len(t)] = unary[b][t]
        edges = [(loc[self.forest[e][0]], loc[self.forest[e][1]]) for e in use]
        P = np.full((len(use), q, q), NEG, dtype=np.float64)
        for k, e in enumerate(use):
            qa, qb = pair[e].shape
            P[k, :qa, :qb] = pair[e]
        sel, comp = forest_dp(U, edges, P)
        for k, b in enumerate(nodes):
            choice[b] = int(top[b][sel[k]])
            group[b] = self.nb_blocks + int(comp[k])             # component ids after the block ids
            gain[group[b]] = gain.get(group[b], 0.0) + float(U[k, sel[k]] - U[k, 0])
        for k, (i, j) in enumerate(edges):
            gain[group[nodes[i]]] += float(P[k, sel[i], sel[j]] - P[k, 0, 0])
        return choice, group, gain

    # ------------------------------------------------------------------ sweeps
    def _apply(self, x: torch.Tensor, blocks) -> torch.Tensor:
        xn = x.clone()
        for b, j in blocks:
            if j > 0:
                for i, v in self.opts[b][j - 1]:
                    xn[0, i] = v
        return xn

    def solve(self, x0: torch.Tensor, sweeps: int = 3, focus: Optional[set] = None,
              callback: Optional[Callable] = None, time_budget: Optional[float] = None,
              tol: float = 1e-4) -> DecompResult:
        """Expand, solve the forest DP, verify; repeat around the new incumbent. `focus` = asset columns
        that changed (an event): only blocks of sections within one neighbourhood of them are re-solved."""
        t0 = time.perf_counter()
        pb = self.pb
        x = pb.space.round_discrete(x0.reshape(1, -1).detach().clone())
        evals0 = pb.n_evals
        active = None
        if focus is not None:
            secs = {int(self.region[c]) for c in focus if 0 <= c < len(self.region) and self.region[c] >= 0}
            secs = self._closed(secs) if secs else set()
            active = np.array([self.blocks[b][0] in secs for b in range(self.nb_blocks)], dtype=bool)
        with torch.no_grad():
            _, F = self._F(x)
        f_cur = float(F[0])
        hist = [{"stage": "decomposition", "iter": 0, "objective": f_cur, "t": 0.0}]
        done = 0
        for sw in range(sweeps):
            if time_budget is not None and time.perf_counter() - t0 > time_budget:
                break
            g_rest = self.rest_gradient(x)
            unary, top, pair, _ = self.tables(x, g_rest, active)
            choice, group, gain = self.dp(unary, top, pair, active)
            done = sw + 1
            changed = [b for b in range(self.nb_blocks) if choice[b] > 0]
            if not changed or sum(gain.get(int(group[b]), 0.0) for b in {int(group[b]): b for b in changed}.values()) <= 1e-9:
                break
            # verification as a batched line search over the DP move: prefixes of the changed blocks sorted by
            # their own predicted gain, every forest component alone, and the whole move - one simulation
            own = np.array([unary[b][choice[b]] - unary[b][0] for b in changed])
            order = [changed[i] for i in np.argsort(-own, kind="stable")]
            ks = sorted({max(1, int(round(k))) for k in np.geomspace(1, len(order), min(40, len(order)))})
            moves = [order[:k] for k in ks]
            comps: dict = {}
            for b in changed:
                comps.setdefault(int(group[b]), []).append(b)
            moves += sorted(comps.values(), key=lambda m: -gain.get(int(group[m[0]]), 0.0))[:24]
            X = torch.cat([self._apply(x, [(b, int(choice[b])) for b in mv]) for mv in moves], 0)
            with torch.no_grad():
                _, Fn = self._F(X)
            k = int(torch.argmax(Fn))
            step = float(Fn[k]) - f_cur
            improved = step > 1e-9
            if improved:
                x, f_cur = X[k:k + 1].clone(), float(Fn[k])
            rec = {"stage": "decomposition", "iter": sw + 1, "objective": f_cur, "changed": len(changed),
                   "accepted": len(moves[k]) if improved else 0, "predicted": round(float(own.sum()), 6),
                   "t": time.perf_counter() - t0}
            hist.append(rec)
            if callback:
                callback(rec)
            if step <= tol:            # converged: what is left is below the model's linearisation error
                break
        with torch.no_grad():
            e = pb.evaluate(x, relax=False)
        return DecompResult(x, float(e.objective[0]), float(e.max_violation()[0]), time.perf_counter() - t0,
                            pb.n_evals - evals0, done, hist)


# ----------------------------------------------------------------------------- forest max-sum DP
def forest_dp(U: np.ndarray, edges: list, P: np.ndarray) -> tuple:
    """Exact argmax of sum_v U[v, z_v] + sum_k P[k, z_a, z_b] over a forest (edges k = (a, b)).
    U [n, q] (NEG = invalid option), P [m, q, q]. Returns (z [n], component root [n]).
    Max-sum message passing: leaves to roots, then back-tracking - O(m q^2)."""
    n, q = U.shape
    adj = [[] for _ in range(n)]
    for k, (a, b) in enumerate(edges):
        adj[a].append((b, k, False))           # P[k] indexed [z_a, z_b]
        adj[b].append((a, k, True))
    order, par, par_edge, par_flip, comp = [], [-1] * n, [-1] * n, [False] * n, [-1] * n
    for r in range(n):
        if comp[r] >= 0:
            continue
        comp[r] = r
        stack = [r]
        while stack:
            v = stack.pop()
            order.append(v)
            for w, k, flip in adj[v]:
                if comp[w] < 0:
                    comp[w] = r
                    par[w], par_edge[w], par_flip[w] = v, k, flip
                    stack.append(w)
    parent = np.array(par, dtype=np.int32)
    pedge = np.array(par_edge, dtype=np.int32)
    flip = np.array(par_flip, dtype=np.uint8)
    post = np.array(order[::-1], dtype=np.int32)            # children before parents
    z = native.forest_dp(np.ascontiguousarray(U, dtype=np.float64), np.ascontiguousarray(P, dtype=np.float64),
                         post, parent, pedge, flip)
    return z, np.array(comp, dtype=np.int64)


# ----------------------------------------------------------------------------- build / cache
_CACHE: dict = {}


@torch.no_grad()
def time_couplings(pb: Problem, x: torch.Tensor, lay: SX.Layout, section_of: np.ndarray,
                   max_sections: int = 60, seed: int = 0) -> dict:
    """|cross term| of a section's decisions in consecutive windows, measured on the twin (sampled)."""
    sec = section_of
    first: dict = {}
    for u in lay.units:
        c = int(sec[u.col]) if u.col < len(sec) else -1
        if c >= 0:
            first.setdefault((c, u.window), []).append(u)
    secs = sorted({c for c, _ in first})
    rng = np.random.default_rng(seed)
    if len(secs) > max_sections:
        secs = sorted(rng.choice(secs, max_sections, replace=False).tolist())
    x = x.reshape(1, -1)

    def flip(xx, units):
        for u in units:
            cur = [float(x[0, i]) for i, _ in u.options[0]]
            best, d = None, -1.0
            for opt in u.options:
                dd = sum(abs(v - c) for (i, v), c in zip(opt, cur))
                if dd > d:
                    best, d = opt, dd
            for i, v in best:
                xx[i] = v
    rows, keys = [x[0].clone()], []
    for c in secs:
        ws = sorted(w for cc, w in first if cc == c)
        if len(ws) < 2:
            continue
        a, b = first[(c, ws[0])], first[(c, ws[1])]
        for units in (a, b, a + b):
            r = x[0].clone()
            flip(r, units)
            rows.append(r)
        keys.append(c)
    if not keys:
        return {}
    X = torch.stack(rows)
    e = pb.evaluate(X, relax=False)
    F = (e.objective / pb.J_scale - MU / pb.T * e.sum_violation()).numpy()
    vals = [abs(F[3 + 3 * k] - F[1 + 3 * k] - F[2 + 3 * k] + F[0]) for k in range(len(keys))]
    typical = float(np.median(vals)) if vals else 0.0
    out = {("t", c): typical for c in range(int(section_of.max()) + 1)}    # unmeasured sections: the median
    out.update({("t", c): float(v) for c, v in zip(keys, vals)})
    return out


def cached(pb: Problem, window_h: float = 12.0, budget: int = 64, q: int = 6) -> tuple:
    """Decomposer per twin structure: the sections (learned filter) are shared with sections.py; the
    forest weights combine the filter's predicted spatial couplings with measured time couplings."""
    key = (id(pb.prog), pb.space.N, float(window_h), int(budget), int(q))
    hit = _CACHE.get(key)
    if hit is None:
        t0 = time.perf_counter()
        solver, info, net, g = SX.cached_parts(pb, window_h=window_h, budget=budget)
        sec = solver.section_of
        with torch.no_grad():
            pred = 10 ** net(g).numpy()
        strength: dict = {}
        for (a, b), s in zip(g.pairs, pred):
            ca, cb = int(sec[int(a)]), int(sec[int(b)])
            if ca >= 0 and cb >= 0 and ca != cb:
                k = (min(ca, cb), max(ca, cb))
                strength[k] = max(strength.get(k, 0.0), float(s))
        x0 = pb.space.round_discrete(pb.space.default_u(1))
        strength.update(time_couplings(pb, x0, solver.lay, sec))
        dec = Decomposer(pb, solver.lay, sec, g, strength, budget=budget, q=q)
        info = dict(info, blocks=dec.nb_blocks, forest_edges=len(dec.forest),
                    forest_weight=round(dec.forest_weight, 6), off_forest_weight=round(dec.off_forest_weight, 6),
                    unary_batches=len(dec.unary_plan), pair_batches=len(dec.pair_plan),
                    halo_assets=int(((dec.region >= 0) & (dec.sec < 0)).sum()), hubs=len(dec.hubs),
                    decomp_setup_s=round(time.perf_counter() - t0, 2))
        hit = _CACHE[key] = (dec, info)
    dec, info = hit
    dec.pb = pb
    return dec, info
