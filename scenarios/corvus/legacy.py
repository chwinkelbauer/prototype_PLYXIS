"""Baseline: conventional mine control of a large underground mine (shift-based, rule-driven).

The comparison is only meaningful if the baseline is competent. These rules encode common practice of
short-interval control in a sublevel-stoping mine (documented in docs/SHOWCASE.md):

* loaders: every 4 h (and at shift change) the shift boss assigns each zone's available loaders to the
  stopes with the largest broken stocks (about 1.3 working places per loader), equal share per stope,
  mucking at the loaders' standard productivity (85 % utilisation) up to the drawpoint limit; grade
  control keeps the expected blend (block-model grades) within 0.05 % of the mine-plan grade; a place is
  only assigned where its level's design air still covers its loader's statutory air, and every step the
  loaders of a level are kept within that air (the ventilation officer's air budget per level); the shift's
  tonnes target is the mill rate (+15 % while the ore passes are below 40 % full, -10 % above 60 %)
* blasting: at shift change (06:00, 18:00), if a zone's broken stock is below `broken_days` (5) days of its
  share of the mill rate, up to two of its stopes whose broken stock is below 15 % of capacity are fired
* ventilation: main fans at the design speed (`fan_speed`); regulators are set by design - each fan's air
  is shared by the levels on its raise (full share where it is the primary raise, half where secondary);
  a level without enough air is throttled by the plant's gas-monitoring interlock
* main fan trip: diesel equipment is withdrawn from the districts that fan ventilates until it runs again;
  the remaining fans run at full speed
* ore passes: level controllers draw towards 50 % full; trucks are allocated in proportion to the draws;
  central ore passes tip to their home crusher
* hoist: full speed while the shaft bin is above 15 %, otherwise at the inflow
* dewatering: pumps start at 70 % sump level and stop at 30 % (no reaction to power prices)
* concentrator: runs at `mill_tph` (99 % of nameplate) from the ROM pad whenever the pad holds ore
* announced critical-peak pricing (> `dr_price`): hoisting and pumping are deferred during the window
  as long as the shaft bin and the sumps allow (the standard load-shifting practice of mines); the mill
  keeps running while the pad holds ore - its margin (~110 USD/t at the plan grade) is three times the
  peak power cost (36 USD/t), and a mill-bound mine cannot make up a lost mill hour
"""
from __future__ import annotations

import csv
import os

from plyxis.lang import units as U

HERE = os.path.dirname(os.path.abspath(__file__))


class LegacyOperators:
    def __init__(self, emu, cfg: dict):
        self.emu, self.cfg = emu, cfg
        flat = emu.flat
        self.typ = {i.path: i.type.short for i in flat.instances}
        self.stopes = [p for p, t in self.typ.items() if t == "stope"]
        self.zone = {s: s.split("_")[1][0] for s in self.stopes}
        self.rk = list(emu.route_key)
        self.fans = [p for p, t in self.typ.items() if t == "main_fan"]
        self.passes = [p for p, t in self.typ.items() if t == "orepass"]
        self.truck_routes = [k for k in self.rk if k[0] == "TRUCKS"]
        self.chute_routes = {}
        for k in self.rk:
            if self.typ.get(k[0]) == "orepass" and k[1] == "chute":
                self.chute_routes.setdefault(k[0], []).append(k)
        self.orepass_of, self.district_of = {}, {}
        for e in flat.edges:
            a = flat.instances[flat.ports[e.src].inst]
            b = flat.instances[flat.ports[e.dst].inst]
            if a.type.short == "stope" and b.type.short == "orepass":
                self.orepass_of[a.path] = b.path
            if a.type.short == "stope" and b.type.short == "vent_district":
                self.district_of[a.path] = b.path
        with open(os.path.join(HERE, "levels.csv"), encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.primary_fan = {r["name"]: r["fan"] for r in rows}
        self.second_fan = {r["name"]: r["fan2"] for r in rows}
        # design regulator settings: each fan's air over its districts, weight 1 as primary, 0.5 as secondary
        self.fan_weight = {f: 0.0 for f in self.fans}
        for r in rows:
            self.fan_weight[r["fan"]] += 1.0
            self.fan_weight[r["fan2"]] += 0.5
        with open(os.path.join(HERE, "stopes.csv"), encoding="utf-8") as fh:
            self.bm_grade = {r["name"]: float(r["grade"]) for r in csv.DictReader(fh)}   # block model, not the truth
        with open(os.path.join(HERE, "orepasses.csv"), encoding="utf-8") as fh:
            self.home_crusher = {r["name"]: r["crusher"] for r in csv.DictReader(fh)}
        self.active: dict = {z: [] for z in "NCS"}
        # each zone's share of production, by its loaders (the shift plan splits the mill rate this way)
        units = {z: self._p(f"LHD_{z}", "n_units") for z in "NCS"}
        self.zone_share = {z: units[z] / sum(units.values()) for z in "NCS"}
        self.last_alloc = -99.0
        self.blast_until: dict = {}
        self.pump_on: dict = {}
        self.last_shift = None
        self.dr_windows: list = []

    # ------------------------------------------------------------------ helpers
    def _p(self, path, name):
        g, li = self.emu.path_index[path]
        return float(self.emu.P[g][name].reshape(-1)[li])

    def _s(self, path, name):
        g, li = self.emu.path_index[path]
        return float(self.emu.S[g][name].reshape(-1)[li])

    def notice(self, ev: dict, t: float) -> None:
        if ev.get("kind") == "announce" and ev.get("name") == "power_price" \
                and float(ev["value"]) >= self.cfg.get("dr_price", 1.0):
            self.dr_windows.append((float(ev["start"]), float(ev["end"])))

    def _cycle(self, s):
        return 2 * self._p(s, "tram") / self._p(s, "lhd_speed") + self._p(s, "load_time")

    def _fan_speed(self):
        """Design speed; while a main fan is down the others run at full speed (the standard response)."""
        down = any(self._s(f, "up") < 0.5 for f in self.fans)
        return self.cfg.get("fan_trip_speed", 1.0) if down else self.cfg.get("fan_speed", 0.9)

    def _design_air(self, d):
        """Regulator settings by design: each fan's air shared by its districts (half share as secondary raise)."""
        q = lambda f: self._fan_speed() * self._p(f, "q_max") * (self._s(f, "up") > 0.5)  # noqa: E731
        fa, fb = self.primary_fan[d], self.second_fan[d]
        return q(fa) / self.fan_weight[fa], 0.5 * q(fb) / self.fan_weight[fb]

    def _district_air(self, d):
        """Air a district's loaders may use: design supply less leakage and the level's base need."""
        return sum(self._design_air(d)) * (1 - self._p(d, "leakage")) - self._p(d, "base_need")

    def _air_need(self, s, rate):
        """Statutory air of a working place mucking at `rate` (the model's air per kW of diesel)."""
        if rate <= 0.0:
            return 0.0
        use = rate / self._p(s, "bucket") * self._cycle(s)
        active = min(1.0, rate / (0.15 * self._p(s, "rate_max")))
        return self._p(s, "air_base") * active + self._p(s, "air_per_kw") * self._p(s, "lhd_kw") * use

    # ------------------------------------------------------------------ control step
    def control(self, t: float) -> None:
        emu, c = self.emu, self.cfg
        vals, routes = {}, {}
        dr = any(s <= t + 1e-9 < e for s, e in self.dr_windows)
        fans_down = {f for f in self.fans if self._s(f, "up") < 0.5}
        # loader allocation every 4 h, at shift change, or when an active stope runs dry
        dry = any(self._s(s, "broken") < 0.02 * self._p(s, "broken_cap") or self._s(s, "up") < 0.5
                  for z in self.active.values() for s in z)
        if t - self.last_alloc >= 4.0 - 1e-9 or dry:
            self.last_alloc = t
            for z in "NCS":
                fleet = f"LHD_{z}"
                units = self._s(fleet, "avail")
                cand = [s for s in self.stopes if self.zone[s] == z and s not in emu.hidden and self._s(s, "up") > 0.5
                        and self._s(s, "broken") > U.to_canonical(2000, "t")]
                cand.sort(key=lambda s: -self._s(s, "broken"))
                n = min(len(cand), max(1, round(units * c.get("places_per_loader", 1.3))))
                # grade control: take the biggest broken stocks, but only those that keep the expected blend
                # (block-model grades) within the tolerance of the mine-plan grade - or move it towards it
                target, tol = c.get("plan_grade", 1.40), c.get("grade_tol", 0.05)
                act: list = []
                for s in cand:
                    if len(act) >= n:
                        break
                    g = self.bm_grade.get(s, target)
                    cur = sum(self.bm_grade.get(a, target) for a in act) / len(act) if act else target
                    new = (cur * len(act) + g) / (len(act) + 1)
                    if abs(new - target) <= tol or abs(new - target) < abs(cur - target):
                        act.append(s)
                for s in cand:                        # fill remaining places if the blend rule was too strict
                    if len(act) >= n:
                        break
                    if s not in act:
                        act.append(s)
                # ventilation: a working place is only assigned where its district's design air still covers the
                # statutory air of its loader, at least at a reduced rate (the ventilation officer's air budget
                # per level); the per-step check below then sets the rates the air allows
                per_place = c.get("utilisation", 0.85) / c.get("places_per_loader", 1.3)
                budget: dict = {}
                order, act = act + [s for s in cand if s not in act], []
                for s in order:
                    if len(act) >= n:
                        break
                    d = self.district_of.get(s)
                    if d not in budget:
                        budget[d] = self._district_air(d) if d else float("inf")
                    r = min(self._p(s, "rate_max"), per_place * self._p(s, "bucket") / self._cycle(s))
                    if self._air_need(s, c.get("min_rate_share", 0.15) * r) <= budget[d] + 1e-9:
                        budget[d] -= self._air_need(s, r)
                        act.append(s)
                self.active[z] = act
        # blasting at shift change
        shift = int((t - 6.0) // 12)
        if shift != self.last_shift:
            self.last_shift = shift
            for z in "NCS":
                # keep a broken-stock buffer of `broken_days` of the zone's share of the mill rate; fire only below it
                zone_st = [s for s in self.stopes if self.zone[s] == z and s not in emu.hidden and self._s(s, "up") > 0.5]
                stock = sum(self._s(s, "broken") for s in zone_st)
                need = c.get("broken_days", 5) * 24 * self.zone_share[z] * U.to_canonical(c.get("mill_tph", 1450), "t/h")
                if stock >= need:
                    continue
                low = [s for s in zone_st if self._s(s, "broken") < c.get("blast_below", 0.15) * self._p(s, "broken_cap")
                       and self.blast_until.get(s, -1) <= t]
                low.sort(key=lambda s: (s not in self.active[z], self._s(s, "broken")))
                for s in low[:c.get("blasts_per_zone", 2)]:
                    self.blast_until[s] = t + 12.0
        # stopes: loaders, mucking, blasting
        for z in "NCS":
            act = self.active[z]
            fleet = f"LHD_{z}"
            units = self._s(fleet, "avail")
            for s in self.stopes:
                if self.zone[s] != z or s in emu.hidden:
                    continue
                on = s in act and self.primary_fan.get(self.district_of.get(s)) not in fans_down
                if on:
                    share = units / max(1, len(act))
                    cap = share * self._p(s, "bucket") / self._cycle(s) * c.get("utilisation", 0.85)
                    vals[(s, "muck")] = min(self._p(s, "rate_max"), cap)
                else:
                    vals[(s, "muck")] = 0.0
                vals[(s, "blast")] = 1.0 if self.blast_until.get(s, -1) > t else 0.0
        # each district's loaders stay within its air (checked every step, e.g. after a fan trip or when fewer
        # places share the fleet); the plant's gas-monitoring interlock is only the backstop
        by_d: dict = {}
        for s in self.stopes:
            if vals.get((s, "muck"), 0.0) > 0.0 and self.district_of.get(s):
                by_d.setdefault(self.district_of[s], []).append(s)
        for d, ss in by_d.items():
            avail, need = self._district_air(d), sum(self._air_need(s, vals[(s, "muck")]) for s in ss)
            if need > avail:
                base = sum(self._p(s, "air_base") for s in ss)
                f = max(0.0, (avail - base) / (need - base)) if need > base else 0.0
                for s in ss:
                    vals[(s, "muck")] *= min(1.0, f)
        # production target of the shift plan: the mill rate - more while the ore passes are below 40 % full
        # (catch up), less above 60 % (no mucking into full passes); rates only go down, so air stays within limits
        fill = sum(self._s(p, "stock") for p in self.passes) / max(1e-9, sum(self._p(p, "cap") for p in self.passes))
        goal = U.to_canonical(c.get("mill_tph", 1450), "t/h") * \
            (c.get("catch_up", 1.15) if fill < 0.4 else c.get("slow_down", 0.9) if fill > 0.6 else 1.0)
        planned = sum(vals.get((s, "muck"), 0.0) for s in self.stopes)
        if planned > goal:
            for s in self.stopes:
                if (s, "muck") in vals:
                    vals[(s, "muck")] *= goal / planned
        # ventilation: design speed (full speed while a fan is down), regulators at their design settings
        for f in self.fans:
            vals[(f, "speed")] = self._fan_speed()
        for d, fa in self.primary_fan.items():
            vals[(d, "air_a")], vals[(d, "air_b")] = self._design_air(d)
        # ore passes: level control towards 50 %, trucks in proportion to the draws
        inflow: dict = {}
        for s in self.stopes:
            op = self.orepass_of.get(s)
            if op:
                inflow[op] = inflow.get(op, 0.0) + vals.get((s, "muck"), 0.0)
        draws = {}
        for op in {p for p, t_ in self.typ.items() if t_ == "orepass"}:
            cap, stock = self._p(op, "cap"), self._s(op, "stock")
            d = inflow.get(op, 0.0) + 0.5 * (stock - 0.5 * cap) / 1.0
            draws[op] = max(0.0, min(self._p(op, "draw_max"), d))
            vals[(op, "draw")] = draws[op]
            for k in self.chute_routes.get(op, []):
                routes[k] = 1.0 if k[2].endswith(op) else 0.0          # home crusher route HR_<op>
        tot = sum(draws.values()) or 1.0
        for k in self.truck_routes:                       # HR_<orepass> = home route; HR_OP_C1_S etc. unused
            routes[k] = draws.get(k[2][3:], 0.0) / tot
        # hoist
        cap, b = self._p("SH1", "cap"), self._s("SH1", "bin")
        feed = sum(draws.values())
        hoist = cap if b > 0.15 * self._p("SH1", "bin_cap") else min(cap, feed)
        if dr and b < 0.85 * self._p("SH1", "bin_cap"):
            hoist = 0.0
        vals[("SH1", "hoist")] = hoist
        # dewatering: level control with hysteresis
        for p_, t_ in self.typ.items():
            if t_ != "pump_station":
                continue
            lvl = self._s(p_, "level") / self._p(p_, "sump")
            on = self.pump_on.get(p_, lvl > 0.5)
            if lvl > 0.7:
                on = True
            elif lvl < 0.3:
                on = False
            if dr and lvl < 0.6:
                on = False
            self.pump_on[p_] = on
            vals[(p_, "rate")] = self._p(p_, "cap") if on else 0.0
        # concentrator feed from the ROM pad
        target = U.to_canonical(c.get("mill_tph", 1450), "t/h")
        rom = self._s("ROM", "tonnes")
        vals[("ROM", "reclaim")] = target if rom > 0.05 * self._p("ROM", "cap") else min(target, hoist)
        emu.set_controls(vals, routes)
