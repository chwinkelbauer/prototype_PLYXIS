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
    def __init__(self, live: LiveTwin, tagmap: TagMap, dt_data: float):
        self.live = live
        self.tags = tagmap
        self.dt = dt_data
        self.prog = live.prog
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
                S[info.gid][name].reshape(-1)[info.li] = new
            elif kind == "param":
                # direct parameter measurement (well test): blend strongly towards it
                P = self.live.P[info.gid][name]
                old = float(P.reshape(-1)[info.li])
                newv = 0.15 * old + 0.85 * float(v)
                P.reshape(-1)[info.li] = newv
                self.live.prior_P[info.gid][name].reshape(-1)[info.li] = newv
                self.live.anchor_P[info.gid][name].reshape(-1)[info.li] = newv
                self.tested.setdefault((info.gid, name), set()).add(info.li)
                changes["params"].append((info.path, name, old, newv, "well test"))
            else:
                meas[(info.gid, info.li, info.key)] = float(v)
        self.live.buf.append({"t": t, "S": _clone(S), "D": {g: {k: v.detach().clone() for k, v in d.items()}
                                                             for g, d in applied_D.items()},
                              "R": applied_R.detach().clone(), "E": {k: torch.as_tensor(v).reshape(-1)[:1].clone()
                                                                     for k, v in E.items()}, "meas": meas})
        return changes

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
                if not info.has_update or (g.gid, f"state:{s}") in self.measured_states:
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
    def calibrate(self, iters: int = 30, lr: float = 1.0, min_steps: int = 6, sigma_anchor: float = 0.15,
                  sigma_step: float = 0.05, sigma_meas: float = 0.05, sigma_tested: float = 0.04,
                  huber: float = 3.0, huber_released: float = 10.0) -> list:
        """MAP estimate over the window (L-BFGS on the differentiable physical twin):

            min  sum huber((y - m) / sigma)  +  |theta / sigma_anchor|^2 / 2  +  |(theta - theta_prev) / sigma_step|^2 / 2

        theta = log(property / anchor), anchor = engineering value or latest well test. A property with
        a recent direct measurement is held tight (sigma_tested); an untested one may be corrected by
        aggregate measurements (station gas, battery levels) within the engineering uncertainty. Assets
        with a persistent mismatch (`released`, e.g. a plugged disposal well) lose their anchor and step
        priors until the mismatch clears - a step change is evidence, not an outlier."""
        buf = list(self.live.buf)
        if len(buf) < min_steps:
            return []
        live, prog = self.live, self.live.phys
        D, R, E, S0, W = self._window(buf)
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
            prev = torch.log(live.P[gid][pname].clamp(min=1e-12) / live.anchor_P[gid][pname].clamp(min=1e-12))
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
            return []
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
        P_base = {g: {k: v for k, v in d.items()} for g, d in live.P.items()}

        def objective():
            P = {g: dict(d) for g, d in P_base.items()}
            for (gid, pname), th in theta.items():
                lo, hi = bounds[(gid, pname)]
                P[gid][pname] = torch.clamp(live.anchor_P[gid][pname] * torch.exp(th), lo, hi)
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
        # commit estimates
        report = []
        with torch.no_grad():
            for (gid, pname), th in theta.items():
                lo, hi = bounds[(gid, pname)]
                newP = torch.clamp(live.anchor_P[gid][pname] * torch.exp(th), lo, hi)
                old = live.P[gid][pname]
                rel = (newP - old).abs() / old.abs().clamp(min=1e-9)
                g = prog.gmap[gid]
                for li in torch.nonzero(rel.reshape(-1) > 0.01).reshape(-1).tolist():
                    path = prog.flat.instances[g.inst_ids[li]].path
                    report.append((path, pname, float(old.reshape(-1)[li]), float(newP.reshape(-1)[li]), "MHE"))
                live.P[gid][pname] = newP.detach().clone()
                live.prior_P[gid][pname] = newP.detach().clone()
                # a step change confirmed under a persistent mismatch becomes the new reference
                for li in released.get(gid, ()):
                    if float(rel.reshape(-1)[li]) > 0.05:
                        live.anchor_P[gid][pname].reshape(-1)[li] = float(newP.reshape(-1)[li])
            # hidden states: replay to the end of the window with the estimated start
            if hstate:
                P = {g: dict(d) for g, d in live.P.items()}
                S = {g: dict(d) for g, d in S0.items()}
                for (gid, s), hv in hstate.items():
                    S[gid][s] = hv.clamp(0.0, 1.0)
                raw = prog.simulate(D, R, P, S, E, T=W, dt=self.dt, t0=t0)
                fin = prog.final_states(raw)
                for (gid, s) in hstate:
                    old = live.S[gid][s].clone()
                    live.S[gid][s] = fin[gid][s].detach().clone()
                    d = (live.S[gid][s] - old).abs()
                    g = prog.gmap[gid]
                    for li in torch.nonzero(d.reshape(-1) > 0.03).reshape(-1).tolist():
                        path = prog.flat.instances[g.inst_ids[li]].path
                        report.append((path, s, float(old.reshape(-1)[li]), float(live.S[gid][s].reshape(-1)[li]),
                                       "state estimate"))
        t = buf[-1]["t"]
        for path, name, old, new, how in report:
            live.param_history.setdefault((path, name), []).append((t, new))
        self.last_report = report
        return report


def _as_seq(D: dict, T: int) -> dict:
    return {g: {k: (v if v.shape[-1] == T else v.expand(*v.shape[:-1], T)) for k, v in d.items()} for g, d in D.items()}
