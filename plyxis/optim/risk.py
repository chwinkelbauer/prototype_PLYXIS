"""Plans under uncertainty: the distribution of a plan's outcomes (docs/OPTIMIZER.md, bench/risk.py).

Uncertain inputs are Gaussian in log space: the twin's calibrated properties with the posterior covariance of
the estimator's own fit (`Estimator.cov`), and prices as random walks of the log price (`Forecaster.log_sigma`).
The covariance is factorised into independent standard-normal factors (eigenvectors scaled by their standard
deviation; a price walk as one factor per time block), and every outcome f of a plan - its value and every
constraint residual of every asset and hour - is propagated with symmetric sigma points per factor k:

    d_k = (f(+k) - f(-k)) / 2           response to one standard deviation of factor k
    c_k = (f(+k) + f(-k)) / 2 - f(0)    curvature along k

    mean = f(0) + sum_k c_k             variance = sum_k d_k^2 + 2 sum_k c_k^2

Exact for outcomes affine in the inputs and for separable quadratics, deterministic, and it keeps every
correlation the twin creates (a well's rate error is the same in every hour, so it accumulates in a tank
instead of averaging out). Against Monte Carlo the residuals of every limit, asset and hour come out within a
few per cent on both twins - what chance constraints need. The plan value does not: where a total over many
assets meets a kink (the field's gas against compressor capacity: flaring starts only when many wells err the
same way) no single factor crosses the kink. A quadrature along the principal directions of the near-binding
limits did not fix it either. So the plan value is sampled at scrambled Sobol points, its mean with the
sigma-point surrogate as a regression control variate,

    S(xi) = f(0) + sum_k c_k + d . xi + sum_k c_k (xi_k^2 - 1)      known mean
    mean f ~ mean of the samples - beta (mean of S over them - its known mean)

and its spread and quantiles from the samples themselves (the surrogate's variance is wrong exactly where
kinks matter). Cost: 2K + 1 simulations for the sigma points plus N samples, once per plan.

Planning uses the result in two ways (`backoffs`, `RiskProblem`): hard limits are tightened by the mean shift
plus z_eps standard deviations of their residual (a chance constraint, P(violation) <= eps; by default only
state limits, `state_limits`), and the standard deviation of the plan value can be capped.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from .problem import Problem


# ----------------------------------------------------------------------------- uncertain inputs
@dataclass
class Uncertainty:
    params: list                          # (gid, pname, li) per entry of the parameter vector
    cov: torch.Tensor                     # [n, n] covariance of log(property)
    price_sigma: dict = field(default_factory=dict)   # env price -> std of log price per sqrt(hour)
    price_blocks: int = 6                 # a price walk as this many independent level steps
    min_var: float = 1e-10

    @staticmethod
    def prior(program, sigma: float = 0.15, price_sigma: Optional[dict] = None, only: Optional[set] = None) -> "Uncertainty":
        """Every calibratable property uncertain by `sigma` (log), independent: the engineering prior."""
        params = []
        for g in program.groups:
            if g.is_env:
                continue
            for pname, decl in g.ir.type.params.items():
                if decl.calibrate is None or pname not in program.P0[g.gid] or (only and pname not in only):
                    continue
                params += [(g.gid, pname, li) for li in range(g.n)]
        cov = torch.eye(len(params), dtype=torch.float64) * sigma ** 2
        return Uncertainty(params, cov, dict(price_sigma or {}))

    def factors(self, T: int, dt: float) -> "Factors":
        """Independent standard-normal factors: parameter eigen-directions, then price steps."""
        n = len(self.params)
        cols = torch.zeros(n, 0, dtype=torch.float64)
        if n:
            w, V = torch.linalg.eigh(0.5 * (self.cov + self.cov.T))
            keep = w > self.min_var * max(float(w.max()), 1e-30)
            cols = V[:, keep] * w[keep].clamp(min=0).sqrt()
        steps = []
        nb = max(1, min(self.price_blocks, T))
        edges = [round(i * T / nb) for i in range(nb + 1)]
        for name, s in self.price_sigma.items():
            if s <= 0:
                continue
            for j in range(nb):
                shape = torch.zeros(T, dtype=torch.float64)
                shape[edges[j]:] = s * math.sqrt(max(edges[j + 1] - edges[j], 1) * dt)
                steps.append((name, shape))
        return Factors(self.params, cols, steps)


@dataclass
class Factors:
    params: list
    cols: torch.Tensor                    # [n_params, K_p] log-space response to each parameter factor
    steps: list                           # (price, [T] log-space path) per price factor

    @property
    def K(self) -> int:
        return self.cols.shape[1] + len(self.steps)

    def _index(self, pb: Problem) -> dict:
        idx: dict = {}
        for i, (gid, pname, li) in enumerate(self.params):
            idx.setdefault((gid, pname), ([], []))
            idx[(gid, pname)][0].append(i)
            idx[(gid, pname)][1].append(li)
        return {k: (torch.tensor(a), torch.tensor(b)) for k, (a, b) in idx.items()}

    def contexts(self, pb: Problem, Z: torch.Tensor) -> tuple:
        """Parameters and prices for B draws of the factors: Z [B, K] (rows of 0 = the nominal inputs)."""
        B = Z.shape[0]
        Kp = self.cols.shape[1]
        P = {g: dict(d) for g, d in pb.P.items()}
        if Kp:
            dlog = (Z[:, :Kp].double() @ self.cols.T).float()            # [B, n_params]
            for (gid, pname), (rows, lis) in self._idx(pb).items():
                base = torch.as_tensor(pb.P[gid][pname], dtype=torch.float32)
                n = base.shape[1] if base.dim() == 3 else base.numel()
                fac = torch.zeros(B, n)
                fac[:, lis] = dlog[:, rows]
                P[gid][pname] = base.reshape(1, n, 1) * torch.exp(fac).unsqueeze(-1)
        E = dict(pb.E)
        if self.steps:
            path: dict = {}
            for k, (name, shape) in enumerate(self.steps):
                path[name] = path.get(name, 0.0) + Z[:, Kp + k:Kp + k + 1].double() * shape.view(1, -1)
            for name, lp in path.items():
                base = torch.as_tensor(pb.E.get(name, pb.prog.E0[name]), dtype=torch.float32).reshape(1, -1)
                base = base.expand(1, pb.T) if base.shape[1] == 1 else base[:, :pb.T]
                E[name] = base * torch.exp(lp.float())
        return P, E

    def _idx(self, pb):
        if getattr(self, "_cache", None) is None:
            self._cache = self._index(pb)
        return self._cache


# ----------------------------------------------------------------------------- propagation
@dataclass
class Outcome:
    value: dict            # f0, mean, std, d [K], c [K] of the plan value
    rows: dict             # (gid, key) -> {"f0", "mean", "std", "std_recourse"} each [n, T] (scaled residual units)
    K: int
    seconds: float
    simulations: int

    def prob_violation(self, key) -> torch.Tensor:
        r = self.rows[key]
        return _norm_sf(-r["mean"] / r["std"].clamp(min=1e-12))

    def quantile(self, q: float) -> float:
        v = self.value
        if "samples" in v:                       # the sampled plan values carry the shape (skew from kinks)
            return float(torch.quantile(v["samples"], q)) + (v["mean"] - float(v["samples"].mean()))
        return v["mean"] + _norm_ppf(q) * v["std"]


def _norm_sf(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.erfc(z / math.sqrt(2.0))


def _norm_ppf(q: float) -> float:
    return math.sqrt(2.0) * torch.erfinv(torch.tensor(2.0 * q - 1.0, dtype=torch.float64)).item()


def _shape3(h) -> torch.Tensor:
    h = torch.as_tensor(h)
    return h.unsqueeze(-1) if h.dim() == 2 else (h.view(1, 1, 1) if h.dim() == 0 else h)


def _rows(pb: Problem, e) -> dict:
    """Hard inequality residuals (scaled, <= 0 feasible)."""
    return {key: _shape3(h) for key, h in e.resid.items() if pb.con_kind.get(key) != "eq"}


def _sobol_normal(n: int, k: int, seed: int = 0) -> torch.Tensor:
    """n scrambled Sobol points mapped to k independent standard normals."""
    eng = torch.quasirandom.SobolEngine(k, scramble=True, seed=seed)
    u = eng.draw(n, dtype=torch.float64).clamp(1e-9, 1 - 1e-9)
    return math.sqrt(2.0) * torch.erfinv(2 * u - 1)


@torch.no_grad()
def propagate(pb: Problem, x: torch.Tensor, unc: Uncertainty, chunk: int = 96, recourse_h: Optional[float] = None,
              samples: int = 128, seed: int = 0) -> Outcome:
    """Mean and standard deviation of the plan value and of every hard-constraint residual of plan x.
    `recourse_h`: for state limits also the spread of the change over the last `recourse_h` hours - what a
    re-plan cannot correct any more (the twin's states are re-measured and the plan re-made meanwhile).
    `samples`: Sobol points for the nonlinear remainder of the plan value (0 = sigma points only)."""
    t0 = time.perf_counter()
    x = x.reshape(1, -1)
    fac = unc.factors(pb.T, pb.dt)
    K = fac.K
    e0 = pb.evaluate(x, relax=False)
    f0 = float(e0.objective[0])
    rows0 = {k: v[:1] for k, v in _rows(pb, e0).items()}
    bound = {(c.gid, c.key) for c in pb.prog.cons if getattr(c, "is_bound", False)}
    lag = int(round(recourse_h / pb.dt)) if recourse_h else 0
    acc = {k: {"c": torch.zeros_like(v[0]), "d2": torch.zeros_like(v[0]), "c2": torch.zeros_like(v[0]),
               "dr2": torch.zeros_like(v[0])} for k, v in rows0.items()}
    d_val = torch.zeros(K, dtype=torch.float64)
    c_val = torch.zeros(K, dtype=torch.float64)
    sims = 1
    for s in range(0, K, chunk):
        ks = list(range(s, min(K, s + chunk)))
        m = len(ks)
        Z = torch.zeros(2 * m, K)
        Z[torch.arange(m), ks] = 1.0
        Z[m + torch.arange(m), ks] = -1.0
        P, E = fac.contexts(pb, Z)
        e = pb.evaluate(x.expand(2 * m, -1), relax=False, P=P, E=E)
        sims += 2 * m
        ob = e.objective.double()
        d_val[s:s + m] = (ob[:m] - ob[m:]) / 2
        c_val[s:s + m] = (ob[:m] + ob[m:]) / 2 - f0
        hs = _rows(pb, e)
        for key, h in hs.items():
            h = h.expand(2 * m, *h.shape[1:]) if h.shape[0] == 1 else h
            d = (h[:m] - h[m:]) / 2                         # [m, n, T]
            c = (h[:m] + h[m:]) / 2 - rows0[key]
            a = acc[key]
            a["c"] += c.sum(0)
            a["d2"] += (d * d).sum(0)
            a["c2"] += (c * c).sum(0)
            if lag and key in bound and d.shape[-1] > 1:
                dr = d.clone()
                dr[..., lag:] = d[..., lag:] - d[..., :-lag]
                a["dr2"] += (dr * dr).sum(0)
            else:
                a["dr2"] += (d * d).sum(0)
    rows = {}
    for key, a in acc.items():
        f = rows0[key][0]
        rows[key] = {"f0": f, "mean": f + a["c"], "std": (a["d2"] + 2 * a["c2"]).sqrt(),
                     "std_recourse": (a["dr2"] + 2 * a["c2"]).sqrt()}
    lin_std = float((d_val.pow(2).sum() + 2 * c_val.pow(2).sum()).sqrt())
    value = {"f0": f0, "mean": f0 + float(c_val.sum()), "std": lin_std, "std_linear": lin_std,
             "d": d_val, "c": c_val}
    if samples > 0:
        # the remainder: sampled at scrambled Sobol points, the sigma-point surrogate as control variate
        Zs = _sobol_normal(samples, K, seed)
        f = []
        for s0 in range(0, samples, 2 * chunk):
            Z = Zs[s0:s0 + 2 * chunk]
            P, E = fac.contexts(pb, Z.float())
            f.append(pb.evaluate(x.expand(Z.shape[0], -1), relax=False, P=P, E=E).objective.double())
            sims += Z.shape[0]
        f = torch.cat(f)
        S_mean = f0 + float(c_val.sum())
        S = S_mean + Zs @ d_val + (Zs.pow(2) - 1) @ c_val
        # regression control variate: the surrogate counts only as far as it explains the samples (beta), so a poor
        # surrogate (strong kinks: the mine's grade window) cannot make the estimate worse than the samples alone
        vs = float(S.var())
        beta = float(((S - S.mean()) * (f - f.mean())).mean()) / vs if vs > 0 else 0.0
        R = f - beta * S
        # the mean from the control variate (its error falls by 1.7-2x on both twins); spread and quantiles from the
        # samples themselves: the surrogate's variance is wrong exactly where kinks matter, so it is not used for them
        value.update({"mean": float(f.mean()) - beta * (float(S.mean()) - S_mean), "std": float(f.std()),
                      "samples": f, "beta": beta, "remainder_std": float(R.std()), "n_samples": samples})
    return Outcome(value, rows, K, time.perf_counter() - t0, sims)


@torch.no_grad()
def monte_carlo(pb: Problem, x: torch.Tensor, unc: Uncertainty, n: int = 512, chunk: int = 128, seed: int = 0) -> dict:
    """Reference: the same inputs sampled. Plan value samples, and per residual row mean, std and P(> 0)."""
    t0 = time.perf_counter()
    x = x.reshape(1, -1)
    fac = unc.factors(pb.T, pb.dt)
    g = torch.Generator().manual_seed(seed)
    vals = []
    s1, s2, cnt = {}, {}, {}
    for s in range(0, n, chunk):
        m = min(chunk, n - s)
        Z = torch.randn(m, fac.K, generator=g)
        P, E = fac.contexts(pb, Z)
        e = pb.evaluate(x.expand(m, -1), relax=False, P=P, E=E)
        vals.append(e.objective.double())
        for key, h in _rows(pb, e).items():
            h = h.expand(m, *h.shape[1:]).double() if h.shape[0] == 1 else h.double()
            s1[key] = s1.get(key, 0.0) + h.sum(0)
            s2[key] = s2.get(key, 0.0) + (h * h).sum(0)
            cnt[key] = cnt.get(key, 0.0) + (h > 0).double().sum(0)
    v = torch.cat(vals)
    rows = {}
    for key in s1:
        mu = s1[key] / n
        rows[key] = {"mean": mu, "std": (s2[key] / n - mu * mu).clamp(min=0).sqrt(), "p_violation": cnt[key] / n}
    return {"value": v, "rows": rows, "seconds": time.perf_counter() - t0, "simulations": n}


# ----------------------------------------------------------------------------- planning
def backoffs(out: Outcome, eps: float = 0.05, recourse: bool = True, floor: float = 0.0,
             keys: Optional[set] = None) -> dict:
    """Tightening of the hard limits `keys` (all if None) so that P(violation) <= eps: mean shift + z_eps standard
    deviations (scaled residual units, added to the residual by `Problem.set_backoff`). Never loosens a limit.

    Which limits deserve it is a question of consequence, measured on the oil field: a flow above a capacity is cut
    by the plant's protection - it costs the cut flow, and the next plan reacts - while a state beyond its limit is
    a spilled tank, a flooded sump, a pump run dry. Back-offs on every limit at 5 % gave away USD 45 k in 16 h
    against the record (unobserved capacities keep their 15 % prior: the battery's liquid capacity lost 24 %).
    `state_limits` gives the keys of the second kind, the default scope of the planner."""
    z = _norm_ppf(1.0 - eps)
    b = {}
    for key, r in out.rows.items():
        if keys is not None and key not in keys:
            continue
        sd = r["std_recourse"] if recourse else r["std"]
        b[key] = ((r["mean"] - r["f0"]) + z * sd).clamp(min=floor).unsqueeze(0).float()
    return b


def state_limits(pb: Problem, upper_only: bool = False) -> set:
    """Hard limits on states (tank and sump levels, bins, stocks): the limits whose violation is a spill, a flood or
    a dry run rather than a cut flow. `upper_only`: only the upper ones (overflow) - a stock's lower bound of zero is
    kept by the plant itself (nothing is mucked that is not there), so its violation is a cut flow too."""
    return {(c.gid, c.key) for c in pb.prog.cons if getattr(c, "is_bound", False) and not c.soft
            and (not upper_only or c.kind == "le")}


class RiskProblem(Problem):
    """A problem whose evaluation also returns the standard deviation of the plan value, from sigma points
    along a few risk directions, and turns `sigma <= cap` into a hard constraint the ALM enforces. The
    directions are fixed at an anchor plan (the factor responses of the plan value, largest first, plus
    their sum - the direction of the value's own spread); the rest of the variance is kept constant."""

    def __init__(self, base: Problem, out: Outcome, unc: Uncertainty, cap: float, r: int = 4):
        self.__dict__.update(base.__dict__)
        fac = unc.factors(base.T, base.dt)
        d = out.value["d"].clone()
        order = torch.argsort(-d.abs())
        dirs = [d / d.norm().clamp(min=1e-12)]
        for k in order[:max(0, r - 1)].tolist():
            v = torch.zeros_like(d)
            v[k] = 1.0
            dirs.append(v)
        Q, _ = torch.linalg.qr(torch.stack(dirs, 1))          # orthonormal risk directions in factor space
        self._Q = Q.float()
        explained = float(((Q.T @ d) ** 2).sum())
        self._rest = max(out.value["std"] ** 2 - explained, 0.0)
        self._fac = fac
        self.cap = float(cap)
        self.risk_r = Q.shape[1]

    def evaluate(self, x, relax=True, P=None, S0=None, E=None):
        if P is not None or E is not None:
            return Problem.evaluate(self, x, relax, P, S0, E)
        B, r = x.shape[0], self.risk_r
        Z = torch.cat([torch.zeros(1, self._Q.shape[0]), self._Q.T, -self._Q.T], 0)    # [1 + 2r, K]
        Pz, Ez = self._fac.contexts(self, Z)
        tile = lambda v: v.repeat(B, *([1] * (v.dim() - 1))) if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == 1 + 2 * r else v  # noqa: E731
        Pb = {g: {k: tile(v) for k, v in d.items()} for g, d in Pz.items()}
        Eb = {k: tile(v) for k, v in Ez.items()}
        X = x.repeat_interleave(1 + 2 * r, 0)                 # plan b, sigma point j -> row b (1 + 2r) + j
        e = Problem.evaluate(self, X, relax, Pb, S0, Eb)
        ob = e.objective.view(B, 1 + 2 * r)
        d = (ob[:, 1:1 + r] - ob[:, 1 + r:]) / 2
        sigma = (d.pow(2).sum(1) + self._rest).sqrt()
        idx = torch.arange(B) * (1 + 2 * r)
        resid = {k: (v[idx] if v.shape[0] == X.shape[0] else v) for k, v in e.resid.items()}
        resid[("risk", "value_sigma")] = ((sigma - self.cap) / max(self.cap, 1.0)).view(B, 1, 1)
        totals = dict(e.totals)
        totals["value_sigma"] = sigma
        from .problem import Evaluation
        return Evaluation(ob[:, 0], None, totals, e.raw, x, resid, {**self.con_kind, ("risk", "value_sigma"): "le"})
