"""State & parameter estimation: keep the twin equal to reality.

* TagMap      - telemetry tags <-> twin signals, derived from the PLX `telemetry:` bindings
* LiveTwin    - the live parameter/state tensors of the twin plus a ring buffer of applied
                controls and measurements
* Estimator   - direct state assimilation (measured levels, run status, unit availability),
                well-test updates of parameters, and moving-horizon estimation (MHE):

                min_theta  sum_t sum_k ((y_k(t; theta) - m_k(t)) / sigma_k)^2 + prior(theta)

                where y = twin replay (differentiable, physical semantics) over the last
                window with the controls actually applied in the plant. theta = calibratable
                properties (log-multipliers) + unmeasured states at the window start
                (e.g. pump health seen only through vibration).

                Around the estimate an information filter keeps the posterior covariance of the
                properties (`cov`, log space): the Gauss-Newton information of each window's residuals
                (one batched replay, a column per property) plus the previous covariance with a small
                drift, scaled down when the residuals show the data noisier than the noise model says.
                Direct tests (well tests, assays) update it as measurements. `uncertainty_model` hands
                it to planning (plyxis/optim/risk.py).

The same code calibrates any PLX twin: what is estimated is declared in the model
(`calibrate in [lo, hi]`, `telemetry:` bindings), not hard-coded here.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch

from ..lang import units as U


@dataclass
class TagInfo:
    gid: str
    li: int
    path: str
    key: str            # signal key: node:local:x | state:x | in:.. | param:x | dec:x
    unit: Optional[str]
    opts: dict


class TagMap:
    def __init__(self, program, flat):
        self.by_tag: dict = {}
        self.by_path: dict = {}
        self.prefixes: set = set()
        from ..lang.consteval import const_eval
        for g in program.groups:
            if g.is_env:
                continue
            for (sk, suffix, unit, opts) in g.ir.telemetry:
                o = {}
                for k, e in opts.items():
                    try:
                        o[k] = const_eval(e, {})
                    except Exception:  # noqa: BLE001
                        pass
                for li, iid in enumerate(g.inst_ids):
                    inst = flat.instances[iid]
                    if not inst.tag:
                        continue
                    tag = f"{inst.tag}.{suffix}"
                    info = TagInfo(g.gid, li, inst.path, sk, unit, o)
                    self.by_tag[tag] = info
                    self.by_path.setdefault(inst.path, []).append(tag)
                    self.prefixes.add(inst.tag)

    def get(self, tag: str) -> Optional[TagInfo]:
        return self.by_tag.get(tag)


def _clone(d: dict) -> dict:
    return {g: {k: v.detach().clone() for k, v in dd.items()} for g, dd in d.items()}


class LiveTwin:
    """Live parameter and state estimate of the twin + data window."""

    def __init__(self, model, window: int = 18):
        self.model = model
        self.prog = model.program
        self.phys = model.physical
        self.P = _clone(self.prog.P0)
        self.S = _clone(self.prog.S0)
        self.prior_P = _clone(self.prog.P0)      # previous estimate (smoothness reference)
        self.anchor_P = _clone(self.prog.P0)     # engineering value or latest well test
        self.window = window
        self.buf: deque = deque(maxlen=window)    # per data step: dict(t, S_start, D, R, E, meas)
        self.param_history: dict = {}             # (path, param) -> list[(t, value)]
        self.state_history: dict = {}

    def controls_from_paths(self, values: dict, routes: dict) -> tuple:
        """Plant control read-back {(path, decision): value} -> twin tensors (D [1,n,1], R [1,E,1])."""
        prog = self.prog
        if not hasattr(self, "_dflt"):
            D0 = prog.default_decisions(B=1, T=1)
            self._dflt = D0
            self._route_key = {}
            flat = prog.flat
            for rp in prog.routed:
                src = flat.instances[rp.inst].path
                for r, eid in zip(rp.ridx, rp.edges):
                    dp = flat.ports[flat.edges[eid].dst]
                    self._route_key[(src, rp.name, flat.instances[dp.inst].path, dp.name)] = r
        D = {g: {k: v.clone() for k, v in d.items()} for g, d in self._dflt.items()}
        for (path, dname), v in values.items():
            iid = prog.flat.by_path.get(path)
            if iid is None:
                continue
            gid, li = prog.inst_group[iid]
            if dname in D.get(gid, {}):
                D[gid][dname][0, li, 0] = float(v)
        R = prog.default_routing(1, 1)
        for key, s in (routes or {}).items():
            r = self._route_key.get(key)
            if r is not None:
                R[0, r, 0] = float(s)
        return D, R

    def params_of(self, path: str) -> dict:
        gi = self.prog.inst_group.get(self.prog.flat.by_path.get(path, -1))
        if gi is None:
            return {}
        gid, li = gi
        return {k: float(v.reshape(-1)[li]) for k, v in self.P[gid].items()}


class Estimator:
    def __init__(self, live: LiveTwin, tagmap: TagMap, dt_data: float, test_sigma: Optional[float] = None,
                 prior_sigma: float = 0.15):
        self.live = live
        self.tags = tagmap
        self.dt = dt_data
        self.prog = live.prog
        self.test_sigma = test_sigma     # relative error of one direct sample; None = accurate (blend strongly)
        self.prior_sigma = prior_sigma   # uncertainty of an engineering value until samples show how wrong they are
        self.test_var: dict = {}         # (path, property) -> relative variance after the samples so far
        self.distrusted: set = set()     # (gid, state, instance) whose sensor is flagged: the model predicts them
        self.calib = []          # (gid, pname, lo, hi)
        # properties with their own measurement (e.g. well tests) are estimated from it directly,
        # not back-allocated from aggregates by the MHE
        direct = {(t.gid, t.key.split(":", 1)[1]) for t in tagmap.by_tag.values() if t.key.startswith("param:")}
        self.direct = direct
        for g in self.prog.groups:
            if g.is_env:
                continue
            for pname, decl in g.ir.type.params.items():
                if decl.calibrate is not None and pname in self.prog.P0[g.gid]:
                    lo, hi = self._calib_bounds(g, pname, decl)
                    self.calib.append((g.gid, pname, lo, hi))
        # constant physical lower limits of states (e.g. 'in [0, cap]'): measurements are clamped to them
        from ..lang.behavior import C as _Const
        self._const_lo = {(g.gid, s): float(info.lo.value) for g in self.live.phys.groups
                          for s, info in g.ir.states.items() if isinstance(info.lo, _Const)}
        # unmeasured states with an observation path (e.g. health -> vib)
        self.hidden_states = []
        measured_states = {(t.gid, t.key) for t in tagmap.by_tag.values() if t.key.startswith("state:")}
        self.measured_states = measured_states
        for g in self.prog.groups:
            if g.is_env:
                continue
            for s, info in g.ir.states.items():
                if (g.gid, f"state:{s}") in measured_states or info.lo is not None:
                    continue
                if s in ("health",):
                    self.hidden_states.append((g.gid, s))
        self.tested: dict = {}   # (gid, param) -> instance rows with a recent direct measurement (well test)
        self.released: set = set()  # asset paths with a persistent model mismatch (set by the anomaly detector)
        self.last_report: list = []
        # posterior covariance of the calibrated properties in log space (information filter around the MHE)
        self.cov_index = [(gid, pname, li) for gid, pname, _lo, _hi in self.calib for li in range(self.prog.gmap[gid].n)]
        self._cov_pos = {k: i for i, k in enumerate(self.cov_index)}
        self.cov = torch.eye(len(self.cov_index), dtype=torch.float64) * prior_sigma ** 2
        self.cov_drift = 0.01        # random-walk drift of a property per calibration (log); a released asset: 0.5
        self.cov_info: dict = {}
        self._cov_log: list = []     # direct tests (index, variance), re-applied over a covariance computed meanwhile
        self.cov_matching = False    # inflate what planning sees per property when estimates move further than predicted
        self._cm: dict = {}          # property -> EWMA of (estimate change)^2 / predicted variance of the change

    def _calib_bounds(self, g, pname, decl):
        from ..lang.consteval import const_eval
        lo, hi = decl.calibrate
        try:
            f = U.factor(decl.unit) if decl.unit else 1.0
            lo = float(const_eval(lo, {})) * f if lo is not None else 0.0
            hi = float(const_eval(hi, {})) * f if hi is not None else math.inf
        except Exception:  # noqa: BLE001
            lo, hi = 0.0, math.inf
        return lo, hi

    # ------------------------------------------------------------------ ingestion
    def ingest(self, t: float, telemetry: list, applied_D: dict, applied_R: torch.Tensor, E: dict) -> dict:
        """One data step: record controls & measurements, assimilate directly measured states."""
        meas = {}
        changes = {"states": [], "params": []}
        S = self.live.S
        for m in telemetry:
            info = self.tags.get(m.tag)
            if info is None:
                continue
            v = m.value
            if info.unit:
                try:
                    v = U.to_canonical(v, info.unit)
                except Exception:  # noqa: BLE001
                    pass
            kind, _, name = info.key.partition(":")
            if kind == "state" and name in S.get(info.gid, {}):
                old = float(S[info.gid][name].reshape(-1)[info.li])
                new = float(v)
                if name in ("up", "avail", "running"):
                    new = float(round(new))
                    if abs(new - old) > 0.5:
                        changes["states"].append((info.path, name, old, new))
                lo = self._const_lo.get((info.gid, name))
                if lo is not None and new < lo:        # a noisy level reading of an empty bin is not a negative stock
                    new = lo
                S[info.gid][name].reshape(-1)[info.li] = new
            elif kind == "param":
                # direct parameter measurement (well test): blend strongly towards it
                P = self.live.P[info.gid][name]
                old = float(P.reshape(-1)[info.li])
                if self.test_sigma:      # noisy samples (assays): a Kalman update, repeated samples average out
                    var = self.test_var.get((info.path, name), self.prior_sigma ** 2)
                    gain = var / (var + self.test_sigma ** 2)
                    newv = old + gain * (float(v) - old)
                    self.test_var[(info.path, name)] = (1.0 - gain) * var
                else:
                    newv = 0.15 * old + 0.85 * float(v)
                P.reshape(-1)[info.li] = newv
                self.live.prior_P[info.gid][name].reshape(-1)[info.li] = newv
                self.live.anchor_P[info.gid][name].reshape(-1)[info.li] = newv
                self.tested.setdefault((info.gid, name), set()).add(info.li)
                i = self._cov_pos.get((info.gid, name, info.li))
                if i is not None:             # the test is a measurement of the property: Kalman update of the covariance
                    s2 = (self.test_sigma or 0.04) ** 2
                    self._cov_test(i, s2)
                    self._cov_log.append((i, s2))
                changes["params"].append((info.path, name, old, newv, "well test"))
            else:
                meas[(info.gid, info.li, info.key)] = float(v)
        self.live.buf.append({"t": t, "S": _clone(S), "D": {g: {k: v.detach().clone() for k, v in d.items()}
                                                             for g, d in applied_D.items()},
                              "R": applied_R.detach().clone(), "E": {k: torch.as_tensor(v).reshape(-1)[:1].clone()
                                                                     for k, v in E.items()}, "meas": meas})
        return changes

    def _cov_test(self, i: int, s2: float) -> None:
        col = self.cov[:, i].clone()
        self.cov = self.cov - torch.outer(col, col) / (float(self.cov[i, i]) + s2)

    def uncertainty_model(self, price_sigma: Optional[dict] = None):
        """The twin's uncertainty for planning: calibrated properties with their posterior covariance, directly
        measured ones (water cut from well tests) independent - the sample's error after a test, the prior before."""
        from ..optim.risk import Uncertainty
        params, sds = list(self.cov_index), []
        for gid, pname in sorted(self.direct):
            if (gid, pname) in {(c[0], c[1]) for c in self.calib} or pname not in self.live.P.get(gid, {}):
                continue
            paths = [self.prog.flat.instances[iid].path for iid in self.prog.gmap[gid].inst_ids]
            for li, path in enumerate(paths):
                v = self.test_var.get((path, pname))
                s = math.sqrt(v) if v is not None else (0.04 if li in self.tested.get((gid, pname), ()) else self.prior_sigma)
                params.append((gid, pname, li))
                sds.append(s)
        n0 = len(self.cov_index)
        cov = torch.zeros(len(params), len(params), dtype=torch.float64)
        cov[:n0, :n0] = self.cov
        if self.cov_matching and self._cm:
            # model error the fit cannot see (a pump curve differing from the plant's: the rate absorbs it): what planning
            # sees is inflated per property by the matching statistic; the filter itself keeps its own covariance
            s = torch.tensor([math.sqrt(min(max(self._cm.get(p, 1.0), 1.0), 100.0)) for (_g, p, _l) in self.cov_index],
                             dtype=torch.float64)
            cov[:n0, :n0] = self.cov * torch.outer(s, s)
        if sds:
            cov[n0:, n0:] = torch.diag(torch.tensor(sds, dtype=torch.float64) ** 2)
        return Uncertainty(params, cov, dict(price_sigma or {}))

    def distrust(self, bad_tags: set) -> None:
        """Sensors the quality monitor has flagged: their states follow the model until the sensor recovers."""
        self.distrusted = set()
        for tag in bad_tags:
            info = self.tags.get(tag)
            if info is not None and info.key.startswith("state:"):
                self.distrusted.add((info.gid, info.key.split(":", 1)[1], info.li))

    def uncertainty(self, gid: str, pname: str) -> torch.Tensor:
        """Relative standard error of a property per instance. Sampled: the posterior of its samples. Not sampled:
        how far the samples so far put the engineering values off (their spread in log), `prior_sigma` until
        at least eight samples exist."""
        g = self.prog.gmap[gid]
        paths = [self.prog.flat.instances[iid].path for iid in g.inst_ids]
        base = self.prog.P0[gid][pname].reshape(-1)
        cur = self.live.anchor_P[gid][pname].reshape(-1)
        var = [self.test_var.get((p, pname)) for p in paths]
        errs = [math.log(float(cur[i]) / max(float(base[i]), 1e-12)) for i, v in enumerate(var) if v is not None]
        spread = math.sqrt(sum(e * e for e in errs) / len(errs)) if len(errs) >= 8 else self.prior_sigma
        return torch.tensor([math.sqrt(v) if v is not None else spread for v in var])

    def reset_states(self, gid: str, li: int, values: dict) -> None:
        """A known discontinuity (pump replaced, unit restarted - e.g. a closed work order): set the live
        state now. Hidden states are also rewritten in the data window, and the asset's older
        measurements are dropped, so estimation does not pull the state back to the old equipment."""
        hidden = {s for g, s in self.hidden_states if g == gid}
        for name, v in values.items():
            if name in self.live.S.get(gid, {}):
                self.live.S[gid][name].reshape(-1)[li] = v
            if name in hidden:
                for b in self.live.buf:
                    if name in b["S"].get(gid, {}):
                        b["S"][gid][name].reshape(-1)[li] = v
        for b in self.live.buf:
            for key in [k for k in b["meas"] if k[0] == gid and k[1] == li]:
                del b["meas"][key]

    # ------------------------------------------------------------------ window assembly
    def _window(self, buf: list) -> tuple:
        """Replay inputs for plant steps 1..W of the window, starting from the state measured at the end
        of step 0. Record k holds the controls applied *during* step k, the statuses after it and the
        measurements it produced, so replay step j uses record j+1 and is compared with its flows."""
        prog = self.live.phys
        rest = buf[1:]
        D = {g: {k: torch.cat([b["D"][g][k] for b in rest], -1) for k in rest[0]["D"][g]} for g in rest[0]["D"]}
        R = torch.cat([b["R"] for b in rest], -1)
        E = {k: torch.cat([b["E"][k].reshape(-1) for b in rest]).view(1, 1, -1) for k in rest[0]["E"]}
        S0 = dict((g, dict(d)) for g, d in buf[0]["S"].items())
        for g in prog.groups:
            for s, info in g.ir.states.items():
                if not info.has_update:     # statuses (run, units available): observed trajectory
                    S0[g.gid][s] = torch.stack([b["S"][g.gid][s].reshape(1, -1) for b in rest], -1)
        return D, R, E, S0, len(rest)

    # ------------------------------------------------------------------ prediction step
    @torch.no_grad()
    def propagate(self) -> None:
        """Advance the states that no sensor measures (last run status, units running, pump health)
        with the twin's own dynamics over the latest data step - the prediction step of the filter.
        Measured states are assimilated in `ingest`; hidden ones are corrected by the MHE."""
        if len(self.live.buf) < 2:
            return
        D, R, E, S0, W = self._window(list(self.live.buf)[-2:])
        phys = self.live.phys
        raw = phys.simulate(D, R, self.live.P, S0, E, T=1, dt=self.dt, t0=self.live.buf[-1]["t"] - self.dt)
        fin = phys.final_states(raw)
        rec = self.live.buf[-1]["S"]
        for g in phys.groups:
            for s, info in g.ir.states.items():
                if not info.has_update:
                    continue
                if (g.gid, f"state:{s}") in self.measured_states:
                    for gid, name, li in self.distrusted:       # no usable reading: the twin's own prediction instead
                        if gid == g.gid and name == s:
                            v = float(fin[g.gid][s].reshape(-1)[li])
                            self.live.S[g.gid][s].reshape(-1)[li] = v
                            rec[g.gid][s].reshape(-1)[li] = v
                    continue
                v = fin[g.gid][s].reshape(self.live.S[g.gid][s].shape).clone()
                self.live.S[g.gid][s] = v
                rec[g.gid][s] = v.clone()

    # ------------------------------------------------------------------ prediction (for anomaly detection)
    @torch.no_grad()
    def predict_last(self) -> dict:
        """One-step twin prediction of every measured signal for the latest data step."""
        if len(self.live.buf) < 2:
            return {}
        D, R, E, S0, W = self._window(list(self.live.buf)[-2:])
        rec = self.live.buf[-1]
        raw = self.live.phys.simulate(D, R, self.live.P, S0, E, T=1, dt=self.dt, t0=rec["t"] - self.dt)
        out = {}
        for (gid, li, key) in rec["meas"]:
            try:
                x = self.live.phys.signal(raw, gid, key)
            except KeyError:
                continue
            x = x.reshape(x.shape[0], -1, x.shape[-1]) if x.dim() == 3 else x
            out[(gid, li, key)] = float(x[0, li if x.shape[1] > 1 else 0, 0]) if x.dim() == 3 else float(x.reshape(-1)[0])
        return out

    # ------------------------------------------------------------------ moving horizon estimation
    def calibrate(self, iters: int = 30, **kw) -> list:
        """Estimate and commit (synchronous). See `calibrate_compute` for the method."""
        res = self.calibrate_compute(iters=iters, **kw)
        return self.calibrate_apply(res) if res is not None else []

    def calibrate_compute(self, iters: int = 30, lr: float = 1.0, min_steps: int = 6, sigma_anchor: float = 0.15,
                          sigma_step: float = 0.05, sigma_meas: float = 0.05, sigma_tested: float = 0.04,
                          huber: float = 3.0, huber_released: float = 10.0) -> Optional[dict]:
        """MAP estimate over the window (L-BFGS on the differentiable physical twin):

            min  sum huber((y - m) / sigma)  +  |theta / sigma_anchor|^2 / 2  +  |(theta - theta_prev) / sigma_step|^2 / 2

        theta = log(property / anchor), anchor = engineering value or latest well test. A property with
        a recent direct measurement is held tight (sigma_tested); an untested one may be corrected by
        aggregate measurements (station gas, battery levels) within the engineering uncertainty. Assets
        with a persistent mismatch (`released`, e.g. a plugged disposal well) lose their anchor and step
        priors until the mismatch clears - a step change is evidence, not an outlier."""
        buf = list(self.live.buf)
        if len(buf) < min_steps:
            return None
        live, prog = self.live, self.live.phys
        D, R, E, S0, W = self._window(buf)
        # snapshot of the live parameters: estimation may run on a worker while the plant keeps stepping
        anchor = {g: {k: v.clone() for k, v in d.items()} for g, d in live.anchor_P.items()}
        t0 = buf[1]["t"] - self.dt
        # parameters to estimate
        # assets with a persistent model mismatch (anomaly detector): their properties may jump
        released: dict = {}
        for path in self.released:
            iid = prog.flat.by_path.get(path)
            if iid is not None:
                g_, li_ = prog.inst_group[iid]
                released.setdefault(g_, set()).add(li_)
        theta, theta_prev, sig_anchor, sig_step, bounds = {}, {}, {}, {}, {}
        for gid, pname, lo, hi in self.calib:
            prev = torch.log(live.P[gid][pname].clamp(min=1e-12) / anchor[gid][pname].clamp(min=1e-12))
            theta_prev[(gid, pname)] = prev.detach()
            theta[(gid, pname)] = prev.detach().clone().requires_grad_(True)
            sa = torch.full_like(prev, sigma_anchor)
            ss = torch.full_like(prev, sigma_step)
            for li in self.tested.get((gid, pname), ()):
                sa.reshape(-1)[li] = sigma_tested
            for li in released.get(gid, ()):
                sa.reshape(-1)[li] = 1.0
                ss.reshape(-1)[li] = 0.5
            sig_anchor[(gid, pname)] = sa
            sig_step[(gid, pname)] = ss
            bounds[(gid, pname)] = (lo, hi)
        hstate = {}
        for gid, s in self.hidden_states:
            hstate[(gid, s)] = S0[gid][s].detach().clone().requires_grad_(True)
        params = list(theta.values()) + list(hstate.values())
        if not params:
            return None
        # measurements of replay step j come from record j+1; noise = relative + per-signal floor
        meas_index: dict = {}
        for w, b in enumerate(buf[1:]):
            for (gid, li, key), v in b["meas"].items():
                meas_index.setdefault((gid, key), []).append((w, li, v))
        meas_t = {}
        for k, rows in meas_index.items():
            m = torch.tensor([r[2] for r in rows])
            sig = sigma_meas * m.abs() + 0.02 * m.abs().mean() + 1e-6
            rel = released.get(k[0], ())
            c = torch.tensor([huber_released if r[1] in rel else huber for r in rows])
            meas_t[k] = (torch.tensor([r[0] for r in rows]), torch.tensor([r[1] for r in rows]), m, sig, c)
        P_base = {g: {k: v.clone() for k, v in d.items()} for g, d in live.P.items()}

        def objective():
            P = {g: dict(d) for g, d in P_base.items()}
            for (gid, pname), th in theta.items():
                lo, hi = bounds[(gid, pname)]
                P[gid][pname] = torch.clamp(anchor[gid][pname] * torch.exp(th), lo, hi)
            S = {g: dict(d) for g, d in S0.items()}
            for (gid, s), hv in hstate.items():
                S[gid][s] = hv.clamp(0.0, 1.0)
            raw = prog.simulate(D, R, P, S, E, T=W, dt=self.dt, t0=t0)
            loss = torch.zeros((), dtype=torch.float64)
            for (gid, key), (ws, ls, m, sig, c) in meas_t.items():
                try:
                    x = prog.signal(raw, gid, key)
                except KeyError:
                    continue
                if x.dim() != 3:
                    continue
                n = x.shape[1]
                pred = x[0, ls.clamp(max=n - 1), ws] if x.shape[-1] > 1 else x[0, ls.clamp(max=n - 1), 0]
                r = ((pred - m) / sig).abs()
                rho = torch.where(r <= c, 0.5 * r * r, c * r - 0.5 * c * c)
                loss = loss + rho.sum().double() / W
            prior = sum(((th / sig_anchor[k]) ** 2).sum() / 2 for k, th in theta.items())
            prior = prior + sum((((th - theta_prev[k]) / sig_step[k]) ** 2).sum() / 2 for k, th in theta.items())
            prior = prior + sum(((hv - S0[g][s]) ** 2).sum() for (g, s), hv in hstate.items()) * 2.0
            return loss + prior

        start = [p.detach().clone() for p in params]
        with torch.no_grad():
            f0 = float(objective())
        opt = torch.optim.LBFGS(params, lr=lr, max_iter=iters, history_size=20, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-6, tolerance_change=1e-9)

        def closure():
            opt.zero_grad()
            f = objective()
            f.backward()
            return f
        try:
            opt.step(closure)
            with torch.no_grad():
                f1 = float(objective())
            ok = math.isfinite(f1) and f1 <= f0 + 1e-9
        except (RuntimeError, ValueError):
            ok = False
        if not ok:                     # never accept a worse (or broken) estimate
            with torch.no_grad():
                for p, s in zip(params, start):
                    p.copy_(s)
        self.last_info = {"objective_before": f0, "objective_after": f1 if ok else f0, "accepted": ok, "window": W}
        with torch.no_grad():
            newP = {k: torch.clamp(anchor[k[0]][k[1]] * torch.exp(th), *bounds[k]).detach().clone()
                    for k, th in theta.items()}
            hidden_end, hidden_ref = {}, {}
            if hstate:
                P = {g: dict(d) for g, d in P_base.items()}
                for (gid, pname), v in newP.items():
                    P[gid][pname] = v
                S = {g: dict(d) for g, d in S0.items()}
                for (gid, s), hv in hstate.items():
                    S[gid][s] = hv.clamp(0.0, 1.0)
                raw = prog.simulate(D, R, P, S, E, T=W, dt=self.dt, t0=t0)
                fin = prog.final_states(raw)
                for (gid, s) in hstate:
                    hidden_end[(gid, s)] = fin[gid][s].detach().clone()
                    hidden_ref[(gid, s)] = buf[-1]["S"][gid][s].detach().clone()
            post = self._posterior(newP, P_base, D, R, E, S0, hstate, W, t0, meas_t, released)
        return {"newP": newP, "anchor": anchor, "released": released, "hidden_end": hidden_end,
                "hidden_ref": hidden_ref, "t": buf[-1]["t"], "posterior": post, "cov_mark": len(self._cov_log)}

    @torch.no_grad()
    def _posterior(self, newP, P_base, D, R, E, S0, hstate, W, t0, meas_t, released, h: float = 0.02) -> Optional[dict]:
        """Information-filter update of the property covariance at the new estimate: the Gauss-Newton
        information J^T w J of the window's standardised residuals (J by one batched replay, a column per
        property, Huber weights w), scaled down by the variance factor when the residuals are larger than the
        noise model says, plus the previous covariance with the drift."""
        n = len(self.cov_index)
        if n == 0 or not meas_t:
            return None
        prog = self.live.phys
        B = n + 1
        P = {g: dict(d) for g, d in P_base.items()}
        for (gid, pname), v in newP.items():
            P[gid][pname] = v
        rows: dict = {}
        for j, (gid, pname, li) in enumerate(self.cov_index):
            rows.setdefault((gid, pname), ([], []))
            rows[(gid, pname)][0].append(j + 1)
            rows[(gid, pname)][1].append(li)
        for (gid, pname), (bj, lis) in rows.items():
            base = torch.as_tensor(P[gid][pname], dtype=torch.float32)
            ng = base.numel()
            f = torch.ones(B, ng)
            f[torch.tensor(bj), torch.tensor(lis)] = math.exp(h)
            P[gid][pname] = base.reshape(1, ng, 1) * f.unsqueeze(-1)
        S = {g: dict(d) for g, d in S0.items()}
        for (gid, s), hv in hstate.items():
            S[gid][s] = hv.detach().clamp(0.0, 1.0)
        # the replay takes its batch from the controls: every input gets the batch of the perturbed properties
        bat = lambda v: v.expand(B, *v.shape[1:]) if torch.is_tensor(v) and v.dim() > 1 and v.shape[0] == 1 else v  # noqa: E731
        D = {g: {k: bat(v) for k, v in d.items()} for g, d in D.items()}
        S = {g: {k: bat(v) for k, v in d.items()} for g, d in S.items()}
        R = bat(R)
        try:
            raw = prog.simulate(D, R, P, S, E, T=W, dt=self.dt, t0=t0)
        except RuntimeError:
            return None
        rs, ws = [], []
        for (gid, key), (wi, ls, m, sig, c) in meas_t.items():
            try:
                x = prog.signal(raw, gid, key)
            except KeyError:
                continue
            if x.dim() != 3:
                continue
            nn = x.shape[1]
            pred = x[:, ls.clamp(max=nn - 1), wi] if x.shape[-1] > 1 else x[:, ls.clamp(max=nn - 1), 0]
            pred = pred.expand(B, -1) if pred.shape[0] == 1 else pred
            r = ((pred - m) / sig).double()
            a = r[0].abs()
            rs.append(r)
            ws.append(torch.where(a <= c, torch.ones_like(a), c.double() / a.clamp(min=1e-12)))
        if not rs:
            return None
        r, w = torch.cat(rs, 1), torch.cat(ws)
        J = (r[1:] - r[:1]) / h                                       # [n, k]
        k = r.shape[1]
        s2 = max(1.0, float((w * r[0] ** 2).sum()) / k)               # variance factor of the window
        H = (J * w) @ J.T / s2
        prior = self.cov + torch.eye(n, dtype=torch.float64) * self.cov_drift ** 2
        for i, (gid, pname, li) in enumerate(self.cov_index):
            if li in released.get(gid, ()):    # a suspected step change: what was known no longer holds - a wide prior,
                prior[i, :] = 0.0              # once (not growing with every calibration), uncorrelated with the rest
                prior[:, i] = 0.0
                prior[i, i] = 0.5 ** 2
        lam = H + torch.linalg.inv(prior)
        cov = torch.linalg.inv(0.5 * (lam + lam.T))
        return {"cov": 0.5 * (cov + cov.T), "variance_factor": s2, "measurements": k, "prior_diag": prior.diagonal().clone()}

    @torch.no_grad()
    def calibrate_apply(self, res: dict) -> list:
        """Commit an estimate. Safe after concurrent plant steps: a property re-anchored by a well test
        meanwhile keeps the test value, and hidden states are corrected by the estimated *change* at
        the end of the window (the live state kept propagating in between)."""
        live, prog = self.live, self.live.phys
        report = []
        released = res["released"]
        post = res.get("posterior")
        if post is not None and post["cov"].shape == self.cov.shape and "prior_diag" in post:
            # covariance matching (Mehra 1972): the filter predicts how far an estimate moves per window (prior minus
            # posterior variance); the ratio of the actual squared move to it, per property, averaged over time
            exp_var = (post["prior_diag"] - post["cov"].diagonal()).clamp(min=1e-12)
            ratios: dict = {}
            for i, (gid, pname, li) in enumerate(self.cov_index):
                if li in released.get(gid, ()) or (gid, pname) not in res["newP"]:
                    continue
                old = float(live.P[gid][pname].reshape(-1)[li])
                new = float(res["newP"][(gid, pname)].reshape(-1)[li])
                if old > 0 and new > 0:
                    ratios.setdefault(pname, []).append(math.log(new / old) ** 2 / float(exp_var[i]))
            for pname, rs in ratios.items():
                self._cm[pname] = 0.9 * self._cm.get(pname, 1.0) + 0.1 * sum(rs) / len(rs)
        for (gid, pname), newP in res["newP"].items():
            if gid not in live.P or pname not in live.P[gid]:
                continue
            old = live.P[gid][pname]
            if old.shape != newP.shape:
                continue
            fresh_test = (live.anchor_P[gid][pname] - res["anchor"][gid][pname]).abs() > 1e-12
            newP = torch.where(fresh_test, old, newP)
            rel = (newP - old).abs() / old.abs().clamp(min=1e-9)
            g = prog.gmap[gid]
            for li in torch.nonzero(rel.reshape(-1) > 0.01).reshape(-1).tolist():
                path = prog.flat.instances[g.inst_ids[li]].path
                report.append((path, pname, float(old.reshape(-1)[li]), float(newP.reshape(-1)[li]), "MHE"))
            live.P[gid][pname] = newP.clone()
            live.prior_P[gid][pname] = newP.clone()
            # a step change confirmed under a persistent mismatch becomes the new reference
            for li in released.get(gid, ()):
                if float(rel.reshape(-1)[li]) > 0.05:
                    live.anchor_P[gid][pname].reshape(-1)[li] = float(newP.reshape(-1)[li])
        for (gid, s), end in res["hidden_end"].items():
            cur = live.S[gid][s]
            ref = res["hidden_ref"][(gid, s)].reshape(cur.shape)
            delta = end.reshape(cur.shape) - ref
            jumped = (cur - ref).abs() > 0.2          # e.g. a pump replaced meanwhile: keep the known state
            new = torch.where(jumped, cur, (cur + delta).clamp(0.0, 1.0))
            d = (new - cur).abs()
            g = prog.gmap[gid]
            for li in torch.nonzero(d.reshape(-1) > 0.03).reshape(-1).tolist():
                path = prog.flat.instances[g.inst_ids[li]].path
                report.append((path, s, float(cur.reshape(-1)[li]), float(new.reshape(-1)[li]), "state estimate"))
            live.S[gid][s] = new.clone()
        post = res.get("posterior")
        if post is not None and post["cov"].shape == self.cov.shape:
            self.cov = post["cov"]
            for i, s2 in self._cov_log[res.get("cov_mark", len(self._cov_log)):]:
                self._cov_test(i, s2)          # tests that arrived while the window was being estimated
            self.cov_info = {"t": res["t"], "variance_factor": post["variance_factor"], "measurements": post["measurements"]}
        if len(self._cov_log) > 10000:
            self._cov_log = self._cov_log[-1000:]
        for path, name, old, new, how in report:
            live.param_history.setdefault((path, name), []).append((res["t"], new))
        self.last_report = report
        return report


def _as_seq(D: dict, T: int) -> dict:
    return {g: {k: (v if v.shape[-1] == T else v.expand(*v.shape[:-1], T)) for k, v in d.items()} for g, d in D.items()}
