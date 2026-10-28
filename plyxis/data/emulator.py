"""Plant emulator - the 'real' site for demonstrations and tests.

The emulator runs a *hidden ground-truth* PLX model compiled in physical mode
(state limits saturate; sat() limits apply) at a fine time step. On top it adds
what real plants have and twins never capture perfectly:

* process noise and drift (well deliverability fluctuates, pumps wear out)
* failures when pump health is exhausted, scripted operational events
* an automatic protection layer: if a facility limit is exceeded, upstream
  production is throttled (backpressure / trips) - overloads cost production
* SCADA telemetry with measurement noise, published under real tag names

The twin never reads the emulator's internals - only its telemetry.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from ..lang import compile_model
from ..lang import units as U
from ..lang.behavior import walk_refs
from ..lang.program import _ident


@dataclass
class Telemetry:
    tag: str
    value: float
    unit: Optional[str]
    t: float
    quality: str = "good"


@dataclass
class Birth:
    """Sparkplug-style birth certificate of a newly connected device."""
    prefix: str
    t: float
    meta: dict = field(default_factory=dict)


class PlantEmulator:
    def __init__(self, files: list, env: str, dt: float = 1.0 / 6.0, seed: int = 0, name: str = "plant",
                 hidden: Optional[list] = None, noise: float = 1.0, defines_prices: Optional[Callable] = None,
                 kpi_metrics: Optional[list] = None):
        m = compile_model(files, env=env, physical=True)
        self.model = m
        self.prog = m.physical
        self.flat = m.flat
        self.name = name
        self.dt = dt
        self.t = 0.0
        self.noise = noise
        self.rng = random.Random(seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.P = {g: {k: v.clone() for k, v in d.items()} for g, d in self.prog.P0.items()}
        self.P_nominal = {g: {k: v.clone() for k, v in d.items()} for g, d in self.prog.P0.items()}
        # design wear rate of every pump (a defect raises the actual rate; a new pump restores it)
        self.wear_design = {g: d["wear_rate"].clone() for g, d in self.P_nominal.items() if "wear_rate" in d}
        self.S = {g: {k: v.clone() for k, v in d.items()} for g, d in self.prog.S0.items()}
        self.prices = defines_prices
        self.overrides: dict = {}          # env input overrides from events: name -> (value, until)
        self.hidden = set(hidden or [])    # instance paths not yet commissioned (no telemetry)
        self.path_index = {inst.path: (self.prog.inst_group[inst.id]) for inst in self.flat.instances}
        self._build_controls()
        self._build_routes()
        self._build_telemetry()
        self.kpi = _Kpi(kpi_metrics)
        self.events: list = []
        self.last_raw: Optional[dict] = None
        self.ar_noise = {gid: torch.zeros_like(v) for gid, d in self.P.items() for k, v in d.items() if k == "q_max"}
        self.test_cursor = 0
        self.throttle_log: list = []

    # ------------------------------------------------------------------ controls
    def _build_controls(self) -> None:
        bnd = self.prog.bounds()
        self.D: dict = {}
        self.dec_bounds: dict = {}
        for g in self.prog.groups:
            self.D[g.gid] = {}
            for d, info in g.ir.decisions.items():
                lo, hi, df = bnd[g.gid][f"dec:{d}"]
                lo = torch.as_tensor(lo, dtype=torch.float32).reshape(1, -1, 1).expand(1, g.n, 1)
                hi = torch.as_tensor(hi, dtype=torch.float32).reshape(1, -1, 1).expand(1, g.n, 1)
                df = torch.as_tensor(df if df is not None else hi, dtype=torch.float32).reshape(1, -1, 1).expand(1, g.n, 1)
                self.D[g.gid][d] = torch.minimum(torch.maximum(df, lo), hi).clone()
                self.dec_bounds[(g.gid, d)] = (lo.clone(), hi.clone())
        # controls the protection layer may throttle: continuous set points and run/stop (as duty cycle)
        self._real_dec = {(g.gid, d) for g in self.prog.groups for d, info in g.ir.decisions.items()
                          if info.domain in ("real", "bool")}

    def _build_routes(self) -> None:
        self.R = self.prog.default_routing(1, 1)
        self.route_key: dict = {}
        flat = self.flat
        for rp in self.prog.routed:
            src = flat.instances[rp.inst].path
            for r, eid in zip(rp.ridx, rp.edges):
                dp = flat.ports[flat.edges[eid].dst]
                self.route_key[(src, rp.name, flat.instances[dp.inst].path, dp.name)] = r

    def set_controls(self, values: dict, routes: Optional[dict] = None) -> None:
        """values: {(path, decision): canonical value}; routes: {(src, port, dst, dport): split}."""
        by: dict = {}
        pidx = self.path_index
        for (path, dname), v in values.items():
            gi = pidx.get(path)
            if gi is not None:
                e = by.get((gi[0], dname))
                if e is None:
                    e = by[(gi[0], dname)] = ([], [])
                e[0].append(gi[1])
                e[1].append(float(v))
        for (gid, dname), (lis, vs) in by.items():
            t = self.D.get(gid, {}).get(dname)
            if t is None:
                continue
            lo, hi = self.dec_bounds[(gid, dname)]
            idx = torch.tensor(lis, dtype=torch.long)
            val = torch.tensor(vs, dtype=t.dtype)
            t[0, idx, 0] = torch.minimum(torch.maximum(val, lo[0, idx, 0]), hi[0, idx, 0])
        if routes:
            touched = set()
            for key, s in routes.items():
                r = self.route_key.get(key)
                if r is not None:
                    self.R[0, r, 0] = float(s)
                    touched.add(r)
            # renormalise per port
            for rp in self.prog.routed:
                tot = float(self.R[0, rp.ridx, 0].sum())
                if tot > 1e-9:
                    self.R[0, rp.ridx, 0] = self.R[0, rp.ridx, 0] / tot

    def readback(self) -> tuple:
        """Actual applied controls as the DCS reports them: ({(path, decision): value}, {route key: split})."""
        vals = {}
        for g in self.prog.groups:
            paths = [self.flat.instances[iid].path for iid in g.inst_ids]
            for dname, t in self.D[g.gid].items():
                for path, v in zip(paths, t[0, :, 0].tolist()):
                    vals[(path, dname)] = v
        Rl = self.R[0, :, 0].tolist()
        routes = {k: Rl[r] for k, r in self.route_key.items()}
        return vals, routes

    def control_value(self, path: str, dname: str) -> Optional[float]:
        gi = self.path_index.get(path)
        if gi is None or dname not in self.D.get(gi[0], {}):
            return None
        return float(self.D[gi[0]][dname][0, gi[1], 0])

    # ------------------------------------------------------------------ telemetry
    def _build_telemetry(self) -> None:
        self.tele = []
        self._tele_groups = None
        for g in self.prog.groups:
            if g.is_env:
                continue
            for (sk, suffix, unit, opts) in g.ir.telemetry:
                opt = {}
                for k, e in opts.items():
                    try:
                        from ..lang.consteval import const_eval
                        opt[k] = const_eval(e, {})
                    except Exception:  # noqa: BLE001
                        pass
                for li, iid in enumerate(g.inst_ids):
                    inst = self.flat.instances[iid]
                    if not inst.tag:
                        continue
                    self.tele.append((g.gid, li, inst.path, f"{inst.tag}.{suffix}", sk, unit, opt))

    def _signal(self, raw: dict, gid: str, li: int, sk: str) -> float:
        kind, _, name = sk.partition(":")
        if kind == "param":
            return float(self.P[gid][name].reshape(-1)[li])
        if kind == "dec":
            return float(self.D[gid][name][0, li, 0])
        if kind == "state":
            v = self.S[gid][name]
            return float(v.reshape(-1)[li])
        x = self.prog.signal(raw, gid, sk)
        x = x.reshape(x.shape[0], -1, x.shape[-1]) if x.dim() == 3 else x
        if x.dim() == 3:
            return float(x[0, li if x.shape[1] > 1 else 0, 0])
        return float(x.reshape(-1)[0])

    def _signal_all(self, raw: dict, gid: str, sk: str) -> list:
        """One signal for every instance of a group (vectorised read)."""
        kind, _, name = sk.partition(":")
        if kind == "param":
            x = self.P[gid][name].reshape(-1)
        elif kind == "dec":
            x = self.D[gid][name][0, :, 0]
        elif kind == "state":
            x = self.S[gid][name].reshape(-1)
        else:
            x = self.prog.signal(raw, gid, sk)
            x = x.reshape(x.shape[0], -1, x.shape[-1])[0, :, 0] if x.dim() == 3 else x.reshape(-1)
        return x.tolist()

    def telemetry(self, raw: dict) -> list:
        out = []
        if getattr(self, "_tele_groups", None) is None:
            groups: dict = {}
            for gid, li, path, tag, sk, unit, opt in self.tele:
                if not opt.get("test"):
                    groups.setdefault((gid, sk, unit), []).append((li, path, tag))
            self._tele_groups = groups
        for (gid, sk, unit), items in self._tele_groups.items():
            vals = self._signal_all(raw, gid, sk)
            try:
                f = U.factor(unit) if unit else 1.0
            except Exception:  # noqa: BLE001
                f = 1.0
            is_status = sk.endswith(":up") or sk.endswith(":avail") or sk.endswith(":running")
            for li, path, tag in items:
                if path in self.hidden:
                    continue
                v = vals[li if len(vals) > 1 else 0] / f
                if not is_status and self.noise > 0:
                    sigma = 0.015 * abs(v) + 0.002 * (abs(v) + 1.0)
                    v = v + self.rng.gauss(0.0, sigma * self.noise)
                out.append(Telemetry(tag, round(v, 4), unit, self.t))
        # well tests: one well per hour on the test separator
        tests = [x for x in self.tele if x[6].get("test") and x[2] not in self.hidden]
        if tests and abs((self.t / 1.0) - round(self.t / 1.0)) < 1e-6:
            wells = sorted({x[2] for x in tests})
            w = wells[self.test_cursor % len(wells)]
            self.test_cursor += 1
            for gid, li, path, tag, sk, unit, opt in tests:
                if path != w:
                    continue
                v = self._signal(raw, gid, li, sk)
                if unit:
                    v = U.from_canonical(v, unit)
                v = v * (1 + self.rng.gauss(0, 0.01 * self.noise))
                out.append(Telemetry(tag, round(v, 4), unit, self.t, "test"))
        return out

    def births(self, path: str) -> Birth:
        inst = self.flat.instances[self.flat.by_path[path]]
        return Birth(inst.tag, self.t, {"lat": inst.at[0] if inst.at else None, "lon": inst.at[1] if inst.at else None,
                                        "type": inst.type.short, "label": inst.label})

    # ------------------------------------------------------------------ protection layer
    def _protect(self, raw: dict, max_iter: int = 4) -> dict:
        """Automatic protection: when a facility limit is exceeded (or a tank would overflow), the
        plant throttles exactly the controls that feed the overload - found by differentiating the
        overloaded quantity with respect to every control (backpressure / high-level trips)."""
        throttled = []
        for _ in range(max_iter):
            viol = self._violations(raw)
            if not viol:
                break
            Dg = {g: {k: v.detach().clone().requires_grad_(True) for k, v in d.items()} for g, d in self.D.items()}
            leaves = [(g, k, t) for g, d in Dg.items() for k, t in d.items() if (g, k) in self._real_dec]
            raw_g = self.prog.simulate(Dg, self.R, self.P, self.S, self.env_now(), T=1, dt=self.dt, t0=self.t,
                                       want_mag=True)
            cut: dict = {}           # control -> throttle fraction per element (strictest over all overloads)
            for gid, key, li, kind in viol:
                q = self._overload_quantity(raw_g, gid, key, li, kind)
                if q is None or float(q.detach()) <= 1e-9:
                    continue
                grads = torch.autograd.grad(q, [t for _, _, t in leaves], retain_graph=True, allow_unused=True)
                contrib, entries = 0.0, []
                for (g, k, t), gr in zip(leaves, grads):
                    if gr is None:
                        continue
                    pos = torch.relu(gr)
                    c = float((pos * t.detach()).sum())
                    if c > 1e-12:
                        contrib += c
                        entries.append((g, k, pos))
                if contrib <= 1e-12:
                    continue
                # cut exactly to the limit (a full tank pumping into a limited disposal well injects at
                # capacity, it does not drop below it); remaining nonlinearity is handled by iterating
                alpha = min(0.95, 1.002 * float(q.detach()) / contrib + 1e-4)
                for g, k, pos in entries:
                    m = (pos > 0).to(self.D[g][k].dtype) * alpha
                    cut[(g, k)] = torch.maximum(cut[(g, k)], m) if (g, k) in cut else m
                inst = self.flat.instances[self.prog.gmap[gid].inst_ids[li]]
                throttled.append({"asset": inst.label or inst.path, "constraint": key, "factor": round(1 - alpha, 3)})
            for (g, k), m in cut.items():
                lo, hi = self.dec_bounds[(g, k)]
                newv = self.D[g][k] * (1.0 - m)
                self.D[g][k] = torch.maximum(torch.minimum(newv, hi), torch.where(m > 0, lo * 0.0, lo))
            raw = self._simulate()
        if throttled:
            self.throttle_log.append((self.t, throttled))
            self.throttle_log = self.throttle_log[-500:]
        return raw

    def _violations(self, raw: dict) -> list:
        out = []
        ev = self.prog.evaluate(raw, self.dt)
        for (gid, ck), h in ev["resid"].items():
            info = self.prog.gmap[gid].ir.cons.get(ck)
            if info is None or info.is_bound or info.soft is not None or info.kind != "le":
                continue
            hh = h.reshape(h.shape[0], -1)
            for li in torch.nonzero(hh[0] > 1e-6).reshape(-1).tolist():
                out.append((gid, ck, li, "con"))
        for g in self.prog.groups:
            for sname, sinfo in g.ir.states.items():
                if sinfo.lo is None:
                    continue
                clip = raw.get(self.prog.node_var(g.gid, f"state:{sname}") + "_clip")
                if clip is None:
                    continue
                c = clip.reshape(clip.shape[0], -1, clip.shape[-1])[0, :, -1]
                for li in torch.nonzero(c > 1e-9).reshape(-1).tolist():
                    out.append((g.gid, sname, li, "overflow"))
        return out

    def _overload_quantity(self, raw: dict, gid: str, key: str, li: int, kind: str):
        if kind == "con":
            h = raw[self.prog.node_var(gid, key)]
            h = h.reshape(h.shape[0], -1, h.shape[-1])
            return torch.relu(h[0, li if h.shape[1] > 1 else 0, -1])
        nxt = raw.get(self.prog.node_var(gid, f"next:{key}"))
        if nxt is None:
            return None
        nxt = nxt.reshape(nxt.shape[0], -1, nxt.shape[-1])
        bnd = self.prog._bounds_cache()[gid].get(f"bound:{key}")
        hi = torch.as_tensor(bnd[1], dtype=nxt.dtype).reshape(-1)
        hi = hi[li] if hi.numel() > 1 else hi[0]
        return torch.relu(nxt[0, li if nxt.shape[1] > 1 else 0, -1] - hi)

    # ------------------------------------------------------------------ dynamics
    def env_now(self) -> dict:
        E = {}
        for k in self.prog.env_inputs:
            v = self.prices(k, self.t) if self.prices else float(self.prog.E0[k])
            o = self.overrides.get(k)
            if o is not None and self.t < o[1]:
                v = o[0]
            E[k] = torch.tensor([float(v)])
        return E

    def _simulate(self) -> dict:
        return self.prog.simulate(self.D, self.R, self.P, self.S, self.env_now(), T=1, dt=self.dt, t0=self.t,
                                  want_mag=True)

    def _process_noise(self) -> None:
        # deliverability fluctuates (AR(1), ~4 % std) around the true value
        for gid, d in self.P.items():
            if "q_max" in d:
                eps = self.ar_noise[gid]
                eps.mul_(0.97).add_(torch.randn(eps.shape, generator=self.gen) * 0.01 * self.noise)
                d["q_max"] = self.P_nominal[gid]["q_max"] * torch.exp(eps)

    def _fail_health(self, gid: int):
        """Health at which a pump fails (`fail_health` of the blueprint, 0.12 if it declares none)."""
        P = self.P.get(gid, {})
        return P["fail_health"].reshape(-1) if "fail_health" in P else 0.12

    def _failures(self) -> list:
        evs = []
        for g in self.prog.groups:
            S = self.S[g.gid]
            if "health" in S and "up" in S:
                h, up = S["health"], S["up"]
                failed = (h.reshape(-1) < self._fail_health(g.gid)) & (up.reshape(-1) > 0.5)
                for li in torch.nonzero(failed.reshape(-1)).reshape(-1).tolist():
                    up.reshape(-1)[li] = 0.0
                    if "fail_cost" in self.P[g.gid]:      # extra cost of an unplanned vs a planned replacement
                        self.kpi.charge("unplanned_failures", float(self.P[g.gid]["fail_cost"].reshape(-1)[li]))
                    inst = self.flat.instances[g.inst_ids[li]]
                    evs.append({"t": self.t, "kind": "failure", "asset": inst.path, "label": inst.label,
                                "detail": "ESP failure (pump health exhausted)"})
        return evs

    def step(self) -> dict:
        self._process_noise()
        raw = self._simulate()
        raw = self._protect(raw)
        ev = self.prog.evaluate(raw, self.dt, include_final=False)
        self.kpi.add(self, ev, raw)
        self.S = self.prog.final_states(raw)
        self.last_raw = raw
        tele = self.telemetry(raw)
        fails = self._failures()
        self.events.extend(fails)
        self.t = round(self.t + self.dt, 9)
        return {"telemetry": tele, "events": fails}

    # ------------------------------------------------------------------ events
    def apply_event(self, kind: str, asset: Optional[str] = None, **kw) -> dict:
        rec = {"t": self.t, "kind": kind, "asset": asset, **kw}
        gi = self.path_index.get(asset) if asset else None
        if kind == "trip" and gi:                         # compressor unit trip
            v = self.S[gi[0]]["avail"]
            v.reshape(-1)[gi[1]] = max(0.0, float(v.reshape(-1)[gi[1]]) - kw.get("units", 1))
        elif kind == "restore_units" and gi:
            n = float(self.P[gi[0]]["n_units"].reshape(-1)[gi[1]])
            self.S[gi[0]]["avail"].reshape(-1)[gi[1]] = n
        elif kind == "fail" and gi:
            self.S[gi[0]]["up"].reshape(-1)[gi[1]] = 0.0
        elif kind == "repair" and gi:
            self.S[gi[0]]["up"].reshape(-1)[gi[1]] = 1.0
            if "health" in self.S[gi[0]] and kw.get("replace", True):     # new pump installed
                if "pump_cost" in self.P[gi[0]]:
                    self.kpi.charge("pump_replacement", float(self.P[gi[0]]["pump_cost"].reshape(-1)[gi[1]]))
                self.S[gi[0]]["health"].reshape(-1)[gi[1]] = 1.0
                if gi[0] in self.wear_design:                               # the defect left with the old pump
                    w = float(self.wear_design[gi[0]].reshape(-1)[gi[1]])
                    self.P_nominal[gi[0]]["wear_rate"].reshape(-1)[gi[1]] = w
                    self.P[gi[0]]["wear_rate"].reshape(-1)[gi[1]] = w
        elif kind == "capacity" and gi:
            p = kw["param"]
            self.P_nominal[gi[0]][p].reshape(-1)[gi[1]] *= kw.get("factor", 1.0)
            self.P[gi[0]][p].reshape(-1)[gi[1]] *= kw.get("factor", 1.0)
        elif kind == "wear" and gi:
            self.P_nominal[gi[0]]["wear_rate"].reshape(-1)[gi[1]] = kw["rate"]
            self.P[gi[0]]["wear_rate"].reshape(-1)[gi[1]] = kw["rate"]
        elif kind == "price":
            self.overrides[kw["name"]] = (kw["value"], self.t + kw.get("hours", 24.0))
        elif kind == "commission" and asset:
            self.hidden.discard(asset)
            if gi and "up" in self.S[gi[0]]:
                self.S[gi[0]]["up"].reshape(-1)[gi[1]] = 1.0
        self.events.append(rec)
        return rec

    def snapshot(self) -> dict:
        """Ground-truth key signals (for the GUI's 'reality' view and A/B comparison)."""
        return self.kpi.now()


class _Kpi:
    """Accumulates realised economics and volumes of an emulated site.

    Realised operating result = sales - cash costs + change in asset value, identical for every world:

    * inventories are valued like the planner's terminal value - the model's `final` terms (oil in tanks
      at 90 % of the price, water in tanks as a disposal liability, ore in bins and on the pad at cost);
    * equipment: the remaining life of every pump (health x pump cost); a pump replacement is charged
      at the pump cost when it happens;
    * terms that exist only for planning (expected failure cost, flaring-risk margin, nominal pump wear,
      `value_to_go` terminal adjustments such as broken ore at half its cost) are excluded - their real
      consequences show up directly here;
    * the carbon cost at the model's internal carbon price is reported separately
      (`profit_after_carbon`), because it is a decision price, not a cash flow.
    """
    EXPECTATION_TERMS = {"risk", "flare_risk", "pump_wear", "value_to_go"}

    def __init__(self, metrics: Optional[list] = None):
        # scenario production metrics (model metric -> cumulative 'm_<key>' in `unit`, rate 'm_<key>' in `rate_unit`)
        self.metrics = list(metrics or [])
        self.cum = {"revenue": 0.0, "cost": 0.0, "emission_kg": 0.0, "oil_bbl": 0.0, "gas_mscf": 0.0,
                    "water_bbl": 0.0, "flared_mscf": 0.0, "spilled_bbl": 0.0, "hours": 0.0,
                    "inventory_value": 0.0, "equipment_value": 0.0, "profit": 0.0,
                    "carbon_cost": 0.0, "profit_after_carbon": 0.0, "flared_over_permit_mscf": 0.0}
        self.terms: dict = {}
        self.term_rate: dict = {}
        self.rate = {}
        self.smooth: dict = {}
        self.tau = 1.5                       # hours, smoothing of the displayed rates
        self.history: list = []
        self._inv_last = None
        self._eq_last = None
        self.inv_terms: dict = {}            # change of each inventory (final term) since the start, USD
        self._inv_terms_last: Optional[dict] = None

    def charge(self, name: str, usd: float) -> None:
        """A one-off cash cost outside the model's running terms (e.g. a replaced pump)."""
        self.terms[name] = self.terms.get(name, 0.0) - usd
        for k in ("cost",):
            self.cum[k] += usd
        self.cum["profit"] -= usd
        self.cum["profit_after_carbon"] -= usd

    @staticmethod
    def _inventory_terms(emu, raw) -> dict:
        """Value of every inventory (realised `final` term) by name, USD."""
        v: dict = {}
        for tr in emu.prog.terms:
            if tr.final and tr.kind in ("revenue", "cost") and tr.name not in _Kpi.EXPECTATION_TERMS:
                x = raw.get(f"F_{tr.gid}_{_ident(tr.key)}")
                if x is not None:
                    s = float(torch.as_tensor(x).sum())
                    v[tr.name] = v.get(tr.name, 0.0) + (s if tr.kind == "revenue" else -s)
        return v

    @staticmethod
    def _inventory(emu, raw) -> float:
        return sum(_Kpi._inventory_terms(emu, raw).values())

    @staticmethod
    def _equipment(emu, fin) -> float:
        """Remaining useful life of every pump x pump cost. Life runs from health 1 to the failure
        threshold (`fail_health`) at the design wear rate; a pump wearing faster (a defect) has proportionally
        less life left - a pump with a failing bearing is worth almost nothing, whatever its health."""
        v = 0.0
        for g in emu.prog.groups:
            S, P = fin.get(g.gid, {}), emu.P.get(g.gid, {})
            if "health" in S and "pump_cost" in P:
                fh = emu._fail_health(g.gid)
                life = ((S["health"].reshape(-1) - fh) / (1.0 - fh)).clamp(0.0, 1.0)
                if "wear_rate" in P and g.gid in emu.wear_design:
                    life = life * (emu.wear_design[g.gid].reshape(-1) / P["wear_rate"].reshape(-1).clamp(min=1e-12)).clamp(max=1.0)
                v += float((life * P["pump_cost"].reshape(-1)).sum())
        return v

    def add(self, emu: PlantEmulator, ev: dict, raw: dict) -> None:
        dt = emu.dt
        named = ev["named"]

        def nv(k):
            v = named.get(k)
            return float(torch.as_tensor(v).reshape(-1)[0]) if v is not None else 0.0
        rev, cost = 0.0, 0.0
        kinds = {}
        for tr in emu.prog.terms:
            if not tr.final:
                kinds.setdefault(tr.name, tr.kind)
        for name, kind in kinds.items():
            if kind not in ("revenue", "cost"):
                continue
            v = float(torch.as_tensor(named.get(name, 0.0)).reshape(-1)[0])
            self.terms[name] = self.terms.get(name, 0.0) + (v if kind == "revenue" else -v)
            self.term_rate[name] = (v if kind == "revenue" else -v) / dt
            if name in self.EXPECTATION_TERMS:
                continue
            if kind == "revenue":
                rev += v
            else:
                cost += v
        # water a sump could not hold (the plant saturates `level`; the twin plans it as a hard limit) is a
        # clean-up cost at the model's `overflow_cost`, in every world
        ovf_m3, ovf_usd = 0.0, 0.0
        for g in emu.prog.groups:
            oc = emu.P.get(g.gid, {}).get("overflow_cost")
            clip = raw.get(emu.prog.node_var(g.gid, "state:level") + "_clip") if oc is not None else None
            if clip is None:
                continue
            vol = torch.relu(torch.as_tensor(clip)).reshape(-1, g.n, torch.as_tensor(clip).shape[-1])[0].sum(-1)
            ovf_m3 += float(vol.sum())
            ovf_usd += float((vol * torch.as_tensor(oc).reshape(-1)).sum())
        if ovf_m3 > 0.0 or "overflow" in self.terms:
            self.terms["overflow"] = self.terms.get("overflow", 0.0) - ovf_usd
            self.term_rate["overflow"] = -ovf_usd / dt
            self.cum["overflow_m3"] = self.cum.get("overflow_m3", 0.0) + ovf_m3
            cost += ovf_usd
        em = float(torch.as_tensor(ev["emission"]).reshape(-1)[0])
        # asset value at the end of the step, valued like the planner's terminal value
        inv_by = self._inventory_terms(emu, raw)
        inv_value = sum(inv_by.values())
        if self._inv_terms_last is not None:
            for n, v in inv_by.items():
                self.inv_terms[n] = self.inv_terms.get(n, 0.0) + v - self._inv_terms_last.get(n, v)
        self._inv_terms_last = inv_by
        eq_value = self._equipment(emu, emu.prog.final_states(raw))
        if self._inv_last is None:
            self._inv_last, self._eq_last = inv_value, eq_value
        d_inv, d_eq = inv_value - self._inv_last, eq_value - self._eq_last
        self._inv_last, self._eq_last = inv_value, eq_value
        carbon = em * float(emu.prog.E0.get("carbon_price", 0.0) or 0.0)
        # named terms are integrated over the step already (rate * dt)
        oil = U.from_canonical(nv("field_oil"), "bbl")
        gas = U.from_canonical(nv("field_gas"), "Mscf")
        wat = U.from_canonical(nv("field_water"), "bbl")
        fl = U.from_canonical(nv("flared"), "Mscf")
        sold = U.from_canonical(nv("oil_sold"), "bbl")
        spill = 0.0
        for k, v in raw.items():
            if k.endswith("_clip") and "oil_lvl" in k:
                spill += float(torch.relu(torch.as_tensor(v)).sum())
        spill = U.from_canonical(spill, "bbl")
        c = self.cum
        c["revenue"] += rev
        c["cost"] += cost
        c["inventory_value"] += d_inv
        c["equipment_value"] += d_eq
        c["profit"] += rev - cost + d_inv + d_eq
        c["carbon_cost"] += carbon
        c["profit_after_carbon"] += rev - cost + d_inv + d_eq - carbon
        c["emission_kg"] += em
        c["oil_bbl"] += oil
        c["gas_mscf"] += gas
        c["water_bbl"] += wat
        c["flared_mscf"] += fl
        permit = emu.prog.E0.get("flare_permit")          # compliance: volume flared above the permit
        if permit is not None:
            c["flared_over_permit_mscf"] += max(0.0, fl - U.from_canonical(float(permit) * dt, "Mscf"))
        c["spilled_bbl"] += spill
        c["hours"] += dt
        mrate = {}
        for m in self.metrics:
            v = nv(m["metric"])
            c["m_" + m["key"]] = c.get("m_" + m["key"], 0.0) + U.from_canonical(v, m["unit"])
            mrate["m_" + m["key"]] = U.from_canonical(v / dt, m["rate_unit"])
        self.rate = {**mrate, "oil_bpd": oil / dt * 24, "gas_mscfd": gas / dt * 24, "water_bpd": wat / dt * 24,
                     "flare_mscfd": fl / dt * 24, "oil_sold_bpd": sold / dt * 24,
                     "profit_usd_h": (rev - cost + d_inv + d_eq) / dt, "revenue_usd_h": rev / dt, "cost_usd_h": cost / dt,
                     "inventory_usd_h": d_inv / dt, "equipment_usd_h": d_eq / dt, "emission_t_h": em / 1000 / dt}
        # display rates: exponentially smoothed (operators and executives read trends, not 10-min noise)
        a = 1.0 - math.exp(-dt / self.tau)
        if not self.smooth:
            self.smooth = dict(self.rate)
        else:
            for k, v in self.rate.items():
                self.smooth[k] = self.smooth.get(k, v) + a * (v - self.smooth.get(k, v))
        self.history.append({"t": emu.t, **{k: round(v, 3) for k, v in self.rate.items()},
                             "cum_profit": round(c["profit"], 1)})
        if len(self.history) > 20000:
            self.history = self.history[-20000:]

    def now(self) -> dict:
        return {"cum": dict(self.cum), "rate": dict(self.smooth or self.rate), "rate_raw": dict(self.rate),
                "terms": {k: round(v, 1) for k, v in self.terms.items()},
                "term_rate": {k: round(v, 1) for k, v in self.term_rate.items()}}
