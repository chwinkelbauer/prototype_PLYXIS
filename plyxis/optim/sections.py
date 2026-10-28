"""Deterministic sections: a learned graph filter decomposes the twin into small sections whose discrete
decisions are solved exactly, coordinated by a linearised model of everything outside the section.

Why: discrete operating decisions (which stopes to blast this shift, which wells run, how many compressor
units, which battery a pad is lined up to) form a scheduling problem whose exact solution grows
exponentially with the number of decisions. Most of these decisions interact only with a few neighbours in
the twin's graph. If the site is cut into sections along the *weak* interactions, every section can be
enumerated exactly - exponential only in the section size - and the sections are coordinated through prices
(the gradient of the rest of the site). The total work grows linearly with the number of sections.

    graph filter   a polynomial graph-filter network (sum_k A^k X W_k, i.e. learned spectral filters on the
                   twin's instance graph) predicts how strongly two assets' decisions interact. Labels come from
                   the twin itself: exact finite-difference cross terms of the penalised objective,
                   F(x+a+b) - F(x+a) - F(x+b) + F(x). At run time the filter replaces those probes
                   (hundreds of simulations) by one forward pass.
    sections       union-find clustering along the strongest predicted couplings, as long as a section's
                   joint discrete options per time window stay within the budget (native kernel).
    section solve  for each time window, every section's joint options are evaluated in ONE batched
                   simulation (all sections vary at once; candidate 0 = keep). A candidate's value is the exact
                   change of the section's own penalised objective plus the linearised effect on the rest of
                   the site (gradient of the other sections' terms, by graph-coloured backward passes).
                   The best candidate per section is chosen exactly (native kernel), applied jointly, and
                   verified on the full twin with backtracking.

No external model or service is involved; the network is trained locally from the twin.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from .. import native
from .problem import Problem

MU = 50.0           # violation weight of the penalised objective (same scale as the cascade's discrete score)


# ----------------------------------------------------------------------------- discrete units
@dataclass
class Unit:
    """One discrete choice of one asset in one time window: a list of options, each a set of (index, value)."""
    col: int                 # instance column in Problem.per_instance
    window: int
    options: list            # list of list[(x index, u value)]


@dataclass
class Layout:
    units: list
    windows: int
    by_col: dict = field(default_factory=dict)        # instance column -> [unit idx]
    dec_col: Optional[np.ndarray] = None               # decision index -> instance column (-1: none)


def discrete_layout(pb: Problem, window_h: float = 12.0) -> Layout:
    """All discrete decisions (int / bool blocks and exclusive line-ups), grouped per asset and time window."""
    sp, prog = pb.space, pb.prog
    bnd = sp.bounds()
    col_of = {iid: k for k, iid in enumerate(pb.inst_ids)}
    W = max(1, math.ceil(sp.T * sp.dt / window_h))
    per: dict = {}                       # (col, window) -> list of option lists per variable
    dec_col = np.full(sp.N, -1, dtype=np.int64)
    for b in sp.blocks:
        g = prog.gmap[b.gid]
        for li, iid in enumerate(g.inst_ids):
            c = col_of.get(iid, -1)
            dec_col[b.offset + li * b.H: b.offset + (li + 1) * b.H] = c
        if b.domain not in ("int", "bool"):
            continue
        lo, hi, _ = bnd[(b.gid, b.name)]
        lo = lo.reshape(-1).expand(b.n) if lo.numel() == 1 else lo.reshape(-1)
        hi = hi.reshape(-1).expand(b.n) if hi.numel() == 1 else hi.reshape(-1)
        for li, iid in enumerate(g.inst_ids):
            c = col_of.get(iid)
            if c is None:
                continue
            l, h = float(lo[li]), float(hi[li])
            if h - l < 0.5:
                continue
            vals = [(v - l) / (h - l) for v in range(int(round(l)), int(round(h)) + 1)]
            for hb in range(b.H):
                w = min(W - 1, int(hb * b.hold * sp.dt // window_h))
                idx = b.offset + li * b.H + hb
                per.setdefault((c, w), []).append([[(idx, v)] for v in vals])
    if sp.route is not None:
        r = sp.route
        for rp in prog.routed:
            c = col_of.get(rp.inst, -1)
            for ri in rp.ridx:
                dec_col[r.offset + ri * r.H: r.offset + (ri + 1) * r.H] = c
            if not rp.exclusive or len(rp.ridx) < 2 or c < 0:
                continue
            for hb in range(r.H):
                w = min(W - 1, int(hb * r.hold * sp.dt // window_h))
                idxs = [r.offset + ri * r.H + hb for ri in rp.ridx]
                per.setdefault((c, w), []).append([[(j, 1.0 if j == k else 0.0) for j in idxs] for k in idxs])
    units = []
    for (c, w), var_opts in sorted(per.items()):
        n = 1
        for o in var_opts:
            n *= len(o)
        if n > 64:                     # too many joint options for one asset-window: vary one variable at a time
            opts = [list(itertools.chain.from_iterable(o[0] for o in var_opts))]
            for vi, o in enumerate(var_opts):
                opts += [opt for opt in o]
        else:
            opts = [list(itertools.chain.from_iterable(combo)) for combo in itertools.product(*var_opts)]
        units.append(Unit(c, w, opts))
    lay = Layout(units, W, dec_col=dec_col)
    for k, u in enumerate(units):
        lay.by_col.setdefault(u.col, []).append(k)
    return lay


# ----------------------------------------------------------------------------- instance graph
@dataclass
class SiteGraph:
    n: int                        # instance columns
    edges: np.ndarray             # [E, 2] undirected
    feats: torch.Tensor           # [n, F]
    A: torch.Tensor               # normalised adjacency (sparse)
    pairs: np.ndarray             # [P, 2] candidate couplings between discrete assets (<= 2 hops via non-hubs)
    types: list


def site_graph(pb: Problem, lay: Layout, hub_degree: int = 14) -> SiteGraph:
    prog = pb.prog
    flat = prog.flat
    col_of = {iid: k for k, iid in enumerate(pb.inst_ids)}
    n = len(pb.inst_ids)
    adj = [set() for _ in range(n)]
    for e in flat.edges:
        a = col_of.get(flat.ports[e.src].inst)
        b = col_of.get(flat.ports[e.dst].inst)
        if a is None or b is None or a == b:
            continue
        adj[a].add(b)
        adj[b].add(a)
    edges = np.array(sorted({(min(a, b), max(a, b)) for a in range(n) for b in adj[a]}), dtype=np.int64).reshape(-1, 2)
    types = sorted({flat.instances[i].type.short for i in pb.inst_ids})
    tix = {t: k for k, t in enumerate(types)}
    sp = pb.space
    ndec = np.zeros(n)
    for b in sp.blocks:
        for li, iid in enumerate(prog.gmap[b.gid].inst_ids):
            c = col_of.get(iid)
            if c is not None:
                ndec[c] += b.H
    nopt = np.zeros(n)
    for u in lay.units:
        nopt[u.col] += math.log(len(u.options))
    deg = np.array([len(a) for a in adj], dtype=np.float64)
    X = np.zeros((n, len(types) + 4))
    for c, iid in enumerate(pb.inst_ids):
        X[c, tix[flat.instances[iid].type.short]] = 1.0
    X[:, -4] = np.log1p(ndec)
    X[:, -3] = nopt / max(1, lay.windows)
    X[:, -2] = np.log1p(deg)
    X[:, -1] = (nopt > 0).astype(float)
    # symmetric normalised adjacency with self loops
    rows = np.concatenate([edges[:, 0], edges[:, 1], np.arange(n)])
    cols = np.concatenate([edges[:, 1], edges[:, 0], np.arange(n)])
    d = np.bincount(rows, minlength=n).astype(np.float64)
    vals = 1.0 / np.sqrt(d[rows] * d[cols])
    A = torch.sparse_coo_tensor(torch.tensor(np.stack([rows, cols])), torch.tensor(vals, dtype=torch.float32), (n, n)).coalesce()
    # candidate couplings between discrete-bearing assets: direct neighbours or 2 hops through a non-hub
    disc = {u.col for u in lay.units}
    pairs = set()
    for a in disc:
        for m in adj[a]:
            if m in disc:
                pairs.add((min(a, m), max(a, m)))
            if len(adj[m]) <= hub_degree:
                for b in adj[m]:
                    if b != a and b in disc:
                        pairs.add((min(a, b), max(a, b)))
    return SiteGraph(n, edges, torch.tensor(X, dtype=torch.float32), A,
                     np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2), types)


# ----------------------------------------------------------------------------- graph filter network
class GraphFilterNet(nn.Module):
    """Polynomial graph filters H = sum_k A^k X W_k (K hops), two layers, and a pair head that predicts
    the log coupling strength of two assets from their filtered embeddings."""

    def __init__(self, n_feat: int, d: int = 32, K: int = 3):
        super().__init__()
        self.K = K
        self.l1 = nn.ModuleList([nn.Linear(n_feat, d, bias=(k == 0)) for k in range(K + 1)])
        self.l2 = nn.ModuleList([nn.Linear(d, d, bias=(k == 0)) for k in range(K + 1)])
        self.head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))

    def _filter(self, A, X, layers):
        out, Z = 0.0, X
        for k, lin in enumerate(layers):
            if k > 0:
                Z = torch.sparse.mm(A, Z)
            out = out + lin(Z)
        return torch.nn.functional.gelu(out)

    def forward(self, g: SiteGraph, pairs: Optional[np.ndarray] = None) -> torch.Tensor:
        H = self._filter(g.A, g.feats, self.l1)
        H = self._filter(g.A, H, self.l2)
        p = torch.as_tensor(g.pairs if pairs is None else pairs)
        a, b = H[p[:, 0]], H[p[:, 1]]
        return self.head(torch.cat([a + b, a * b, (a - b).abs()], 1)).squeeze(1)


@torch.no_grad()
def coupling_labels(pb: Problem, x: torch.Tensor, lay: Layout, g: SiteGraph, max_pairs: int = 900,
                    batch: int = 64, seed: int = 0) -> tuple:
    """Exact interaction strength of asset pairs on the twin: |F(x+a+b) - F(x+a) - F(x+b) + F(x)|, where a, b
    flip each asset's discrete choice in its first window to the option that changes it most."""
    rng = np.random.default_rng(seed)
    pairs = g.pairs if len(g.pairs) <= max_pairs else g.pairs[rng.choice(len(g.pairs), max_pairs, replace=False)]
    first = {}
    for u in lay.units:
        if u.col not in first or u.window < first[u.col].window:
            first[u.col] = u
    x = x.reshape(1, -1)

    def flip(xx, col):
        u = first[col]
        cur = [float(x[0, i]) for i, _ in u.options[0]]
        best, d = None, -1.0
        for opt in u.options:
            dd = sum(abs(v - c) for (i, v), c in zip(opt, cur))
            if dd > d:
                best, d = opt, dd
        for i, v in best:
            xx[i] = v

    cols = sorted({int(c) for p in pairs for c in p})
    variants = [("base", None)] + [("one", c) for c in cols] + [("two", tuple(p)) for p in pairs]
    F = []
    for s in range(0, len(variants), batch):
        chunk = variants[s:s + batch]
        X = x.repeat(len(chunk), 1)
        for k, (kind, arg) in enumerate(chunk):
            if kind == "one":
                flip(X[k], arg)
            elif kind == "two":
                flip(X[k], arg[0])
                flip(X[k], arg[1])
        e = pb.evaluate(X, relax=False)
        F.append(e.objective / pb.J_scale - MU / pb.T * e.sum_violation())
    F = torch.cat(F)
    f0 = float(F[0])
    one = {c: float(F[1 + k]) for k, c in enumerate(cols)}
    y = np.array([abs(float(F[1 + len(cols) + k]) - one[int(p[0])] - one[int(p[1])] + f0) for k, p in enumerate(pairs)])
    return pairs, y


def train_filter(g: SiteGraph, pairs: np.ndarray, y: np.ndarray, steps: int = 400, seed: int = 0,
                 net: Optional[GraphFilterNet] = None) -> tuple:
    torch.manual_seed(seed)
    net = net or GraphFilterNet(g.feats.shape[1])
    target = torch.tensor(np.log10(y + 1e-6), dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(steps):
        pred = net(g, pairs)
        loss = torch.mean((pred - target) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = net(g, pairs).numpy()
    rank = float(np.corrcoef(np.argsort(np.argsort(pred)), np.argsort(np.argsort(target.numpy())))[0, 1]) \
        if len(pairs) > 2 else 0.0
    return net, {"mse": float(loss), "spearman": rank, "pairs": int(len(pairs))}


def build_sections(lay: Layout, g: SiteGraph, strength: np.ndarray, budget: int = 64,
                   min_strength: float = 1e-5) -> np.ndarray:
    """Section id per instance column (-1: no discrete decisions). Merge along the strongest couplings while a
    section's joint options per window stay <= budget."""
    n = g.n
    logc = np.zeros(n)
    for u in lay.units:
        logc[u.col] = max(logc[u.col], math.log(len(u.options)))
    disc = np.array(sorted({u.col for u in lay.units}), dtype=np.int64)
    if len(disc) == 0:
        return np.full(n, -1, dtype=np.int32)
    pos = {c: k for k, c in enumerate(disc)}
    keep = [k for k, p in enumerate(g.pairs) if int(p[0]) in pos and int(p[1]) in pos]
    src = np.array([pos[int(g.pairs[k][0])] for k in keep], dtype=np.int32)
    dst = np.array([pos[int(g.pairs[k][1])] for k in keep], dtype=np.int32)
    w = np.asarray(strength, dtype=np.float32)[keep] if len(keep) else np.zeros(0, dtype=np.float32)
    lab = native.cluster(len(disc), src, dst, w, logc[disc], math.log(budget) + 1e-9, float(min_strength))
    out = np.full(n, -1, dtype=np.int32)
    out[disc] = lab
    return out


# ----------------------------------------------------------------------------- section solver
@dataclass
class SectionResult:
    x: torch.Tensor
    objective: float
    max_violation: float
    sections: int
    seconds: float
    evals: int
    history: list


class SectionSolver:
    """Exact enumeration of every section's joint discrete options per window, coordinated by prices."""

    def __init__(self, pb: Problem, lay: Layout, section_of: np.ndarray, g: SiteGraph, max_cand: int = 64):
        self.pb, self.lay, self.max_cand = pb, lay, max_cand
        self.section_of = section_of
        self.C = int(section_of.max()) + 1 if (section_of >= 0).any() else 0
        n_cols = pb.n_inst
        self.inst_sec = np.full(n_cols, -1, dtype=np.int32)
        self.inst_sec[:len(section_of)] = section_of
        self.dec_sec = np.where(lay.dec_col >= 0, self.inst_sec[np.clip(lay.dec_col, 0, n_cols - 1)], -1).astype(np.int32)
        # section x window -> joint options (exact product; sampled only if a section exceeds the budget)
        self.cands: dict = {}
        for w in range(lay.windows):
            by_sec: dict = {}
            for u in lay.units:
                if u.window == w and self.inst_sec[u.col] >= 0:
                    by_sec.setdefault(int(self.inst_sec[u.col]), []).append(u)
            for c, us in by_sec.items():
                combos = list(itertools.product(*[u.options for u in us]))
                if len(combos) > max_cand:
                    rng = np.random.default_rng(c * 131 + w)
                    combos = [combos[i] for i in rng.choice(len(combos), max_cand, replace=False)]
                self.cands[(c, w)] = [list(itertools.chain.from_iterable(cb)) for cb in combos]
        self.colours = self._colour(g)

    def _colour(self, g: SiteGraph, hub_sections: int = 12) -> list:
        """Greedy colouring of the section interaction graph: sections of one colour do not interact (no shared
        edge, no shared non-hub neighbour), so one backward pass gives each the gradient of its own terms."""
        nb = [set() for _ in range(self.C)]
        links = np.concatenate([g.pairs, g.edges]) if len(g.edges) and len(g.pairs) else (g.pairs if len(g.pairs) else g.edges)
        for a, b in links:
            sa, sb = self.inst_sec[int(a)], self.inst_sec[int(b)]
            if sa >= 0 and sb >= 0 and sa != sb:
                nb[sa].add(sb)
                nb[sb].add(sa)
        shared: dict = {}
        for a, b in g.edges:
            for u, v in ((int(a), int(b)), (int(b), int(a))):
                if self.inst_sec[u] >= 0 and self.inst_sec[v] < 0:
                    shared.setdefault(v, set()).add(int(self.inst_sec[u]))
        for ss in shared.values():
            if len(ss) > hub_sections:        # a hub (fleet pool, main fan) couples weakly per section: priced, not coloured
                continue
            for i in ss:
                nb[i].update(t for t in ss if t != i)
        col = [-1] * self.C
        for c in sorted(range(self.C), key=lambda c: -len(nb[c])):
            used = {col[d] for d in nb[c] if col[d] >= 0}
            k = 0
            while k in used:
                k += 1
            col[c] = k
        return col

    def _penalised(self, X: torch.Tensor):
        e = self.pb.evaluate(X, relax=False)
        F = e.objective / self.pb.J_scale - MU / self.pb.T * e.sum_violation()
        return e, F

    def _own_and_rest_gradient(self, x: torch.Tensor) -> np.ndarray:
        """g_rest = dF/dx - d(own section terms)/dx for every decision, via one backward per colour."""
        pb = self.pb
        xg = x.detach().clone().requires_grad_(True)
        e = pb.evaluate(xg, relax=False)
        F = e.objective / pb.J_scale - MU / pb.T * e.sum_violation()
        g_tot = torch.autograd.grad(F.sum(), xg)[0][0].detach()
        own = torch.zeros_like(g_tot)
        n_col = max(self.colours) + 1 if self.colours else 0
        if n_col == 0:
            return g_tot.numpy()
        dec_sec = torch.as_tensor(self.dec_sec, dtype=torch.long)
        sec_col = torch.as_tensor(self.colours, dtype=torch.long)
        for k in range(n_col):
            xg = x.detach().clone().requires_grad_(True)
            e = pb.evaluate(xg, relax=False)
            pi = pb.per_instance(e, MU)[0]                  # [n_inst]
            sel = torch.as_tensor((self.inst_sec >= 0) & (np.asarray(self.colours + [-1])[self.inst_sec] == k))
            if not bool(sel.any()):
                continue
            gk = torch.autograd.grad(pi[sel].sum(), xg, allow_unused=True)[0]
            if gk is None:
                continue
            gk = gk[0]
            mine = (dec_sec >= 0) & (sec_col[dec_sec.clamp(min=0)] == k)
            own = torch.where(mine, gk, own)
        return (g_tot - own).numpy()

    def solve(self, x0: torch.Tensor, sweeps: int = 2, callback: Optional[Callable] = None,
              time_budget: Optional[float] = None) -> SectionResult:
        t0 = time.perf_counter()
        pb = self.pb
        x = pb.space.round_discrete(x0.reshape(1, -1).detach().clone())
        evals0 = pb.n_evals
        with torch.no_grad():
            _, F = self._penalised(x)
        f_cur = float(F[0])
        hist = [{"stage": "sections", "iter": 0, "objective": f_cur, "t": 0.0}]
        it = 0
        for sw in range(sweeps):
            order = list(range(self.lay.windows)) if sw % 2 == 0 else list(range(self.lay.windows))[::-1]
            g_rest = self._own_and_rest_gradient(x)          # prices of the rest of the site, once per sweep
            for w in order:
                if time_budget is not None and time.perf_counter() - t0 > time_budget:
                    break
                secs = [c for c in range(self.C) if (c, w) in self.cands]
                if not secs:
                    continue
                n_c = np.ones(self.C, dtype=np.int32)
                B = 1
                for c in secs:
                    n_c[c] = len(self.cands[(c, w)]) + 1             # batch element 0 = keep the current choice
                    B = max(B, int(n_c[c]))
                X = x.repeat(B, 1)
                for c in secs:
                    opts = self.cands[(c, w)]
                    for b in range(1, min(B, len(opts) + 1)):
                        for i, v in opts[b - 1]:
                            X[b, i] = v
                with torch.no_grad():
                    e = pb.evaluate(X, relax=False)
                    pi = pb.per_instance(e, MU).numpy()           # [B, n_inst]
                dl = pi - pi[0:1]
                dx = (X - X[0:1]).numpy()
                scores = native.section_scores(dl, self.inst_sec, dx, g_rest, self.dec_sec, self.C)
                best, gain = native.argmax_sections(scores, n_c)
                better = [c for c in secs if best[c] > 0 and gain[c] > 1e-9]
                if not better:
                    continue
                # apply all improving sections jointly; verify on the full twin, halve on failure
                better.sort(key=lambda c: -gain[c])
                for frac in (1.0, 0.5, 0.25, 0.1):
                    take = better[:max(1, int(math.ceil(len(better) * frac)))]
                    xn = x.clone()
                    for c in take:
                        for i, v in self.cands[(c, w)][int(best[c]) - 1]:
                            xn[0, i] = v
                    with torch.no_grad():
                        _, Fn = self._penalised(xn)
                    if float(Fn[0]) > f_cur + 1e-9:
                        x, f_cur = xn, float(Fn[0])
                        break
                it += 1
                rec = {"stage": "sections", "iter": it, "window": w, "objective": f_cur, "improved": len(better),
                       "t": time.perf_counter() - t0}
                hist.append(rec)
                if callback:
                    callback(rec)
        with torch.no_grad():
            e = pb.evaluate(x, relax=False)
        return SectionResult(x, float(e.objective[0]), float(e.max_violation()[0]), self.C, time.perf_counter() - t0,
                             pb.n_evals - evals0, hist)


# ----------------------------------------------------------------------------- cache per twin structure
_CACHE: dict = {}


def cached_solver(pb: Problem, window_h: float = 12.0, budget: int = 64) -> tuple:
    """Layout, graph, trained filter and sections are built once per twin structure (the first solve trains the
    filter on the twin in a few seconds); later solves only swap in the current operating context."""
    key = (id(pb.prog), pb.space.N, float(window_h), int(budget))
    hit = _CACHE.get(key)
    if hit is None:
        t0 = time.perf_counter()
        lay = discrete_layout(pb, window_h=window_h)
        g = site_graph(pb, lay)
        x0 = pb.space.round_discrete(pb.space.default_u(1))
        pairs, y = coupling_labels(pb, x0, lay, g, max_pairs=400)
        net, fit = train_filter(g, pairs, y, steps=300)
        with torch.no_grad():
            strength = 10 ** net(g).numpy()
        thr = float(np.quantile(y, 0.5)) if len(y) else 0.0
        sec = build_sections(lay, g, strength, budget=budget, min_strength=thr)
        solver = SectionSolver(pb, lay, sec, g, max_cand=budget)
        sizes = np.bincount(sec[sec >= 0]) if (sec >= 0).any() else np.zeros(1, dtype=int)
        info = {"sections": int(solver.C), "largest_section": int(sizes.max()), "colours": int(max(solver.colours) + 1)
                if solver.colours else 0, "discrete_units": len(lay.units), "filter_spearman": round(fit["spearman"], 3),
                "filter_pairs": fit["pairs"], "setup_s": round(time.perf_counter() - t0, 2)}
        hit = _CACHE[key] = (solver, info, net, g)
    solver, info = hit[0], hit[1]
    solver.pb = pb
    return solver, info
