"""Baseline: conventional manual operations of a mid-size onshore operator.

The comparison is only meaningful if the baseline is competent. These rules
encode common practice (documented in docs/SHOWCASE.md):

* wells run at the field-standard set point (95 % pump speed, `speed` in scenario.json) whenever available
* pads stay lined up to their home tank battery
* every available compressor unit runs
* tank level controllers (P-control) pump oil / water out around 50 % level
* CTB Central oil goes to the pipeline first; trucks lift the excess
* the dispatcher re-allocates the truck fleet every 2 h in proportion to the oil
  volume waiting on trucked batteries
* water goes to the home disposal well; the control room shifts part of it to the
  second outlet to keep the home well below its injection-pressure alarm, and
  curtails the two wettest pads of a battery by 20 % if its water tank still runs
  above 85 %
* sustained flaring (> 2 h) -> control room curtails the gassiest pads by 20 %
  until flaring stops; curtailment is lifted 2 h after
* gas takeaway at its limit (> 2 h) -> the same curtailment of the gassiest pads,
  kept until gas sales are clearly below the limit
* compressor trips: call-out crew restarts the unit after 6 h
* pump failures: workover rig scheduled first thing the next morning (07:00),
  first-come-first-served, 30 h job
* no real-time reaction to power or gas prices (monthly economics), except announced
  scarcity pricing: on a day-ahead notice above `dr_price` the control room drops every
  ESP to the pre-agreed demand-response set point (`dr_speed`) for the announced window
  (the bulk set-point change operators already use for ERCOT 4CP peaks)
"""
from __future__ import annotations

import math

import torch

from plyxis.lang import units as U


class LegacyOperators:
    def __init__(self, emu, cfg: dict):
        self.emu = emu
        self.cfg = cfg
        self.flat = emu.flat
        self.prog = emu.prog
        self.curtail: dict = {}             # pad path -> (factor, since)
        self.flare_since = None
        self.last_truck_alloc = -99.0
        self.repairs: list = []             # (t_done, kind, asset)
        self.handled = set()
        self.wells = [i.path for i in self.flat.instances if i.type.short == "esp_well"]
        self.pads = {i.path: i for i in self.flat.instances if i.type.short == "pad"}
        self.well_pad = {}
        for e in self.flat.edges:
            a = self.flat.instances[self.flat.ports[e.src].inst]
            b = self.flat.instances[self.flat.ports[e.dst].inst]
            if a.type.short == "esp_well" and b.type.short == "pad":
                self.well_pad[a.path] = b.path
        self.gor = {w: self._param(w, "gor") for w in self.wells}
        self.water_div: dict = {}           # battery -> share of its water sent to the home disposal well
        self.wcurtail: dict = {}            # pad -> since (water-constrained curtailment)
        self.tcurtail: dict = {}            # pad -> since (takeaway-constrained curtailment)
        self.takeaway_since = None
        self.gassy = sorted(self.pads, key=lambda p: -sum(self.gor.get(w, 0) for w, pp in self.well_pad.items() if pp == p))
        self.gas_plant = next((i.path for i in self.flat.instances if i.type.short == "gas_sales"), None)
        self.dr_windows: list = []          # (start, end) of announced scarcity pricing, plant hours

    def notice(self, ev: dict, t: float) -> None:
        """Notices reach the control room at the same time as PLYXIS:
        * a midstream curtailment -> curtail the gassiest pads right away instead of waiting for the
          takeaway limit to persist;
        * day-ahead scarcity pricing (`start`/`end` in plant hours) -> demand-response set point for
          the window."""
        if ev.get("kind") == "announce":
            if ev.get("name") == "power_price" and float(ev["value"]) >= self.cfg.get("dr_price", 1.0):
                self.dr_windows.append((float(ev["start"]), float(ev["end"])))
            return
        if ev.get("asset") == self.gas_plant and float(ev.get("factor", 1.0)) < 1.0:
            self.takeaway_since = t - self.cfg["flare_response_h"]

    def _param(self, path, name):
        gid, li = self.emu.path_index[path]
        return float(self.emu.P[gid][name].reshape(-1)[li])

    def _state(self, path, name):
        gid, li = self.emu.path_index[path]
        return float(self.emu.S[gid][name].reshape(-1)[li])

    def _sig(self, path, key):
        raw = self.emu.last_raw
        if raw is None:
            return 0.0
        gid, li = self.emu.path_index[path]
        return self.emu._signal(raw, gid, li, key)

    # ------------------------------------------------------------------ control step
    def control(self, t: float) -> None:
        emu, c = self.emu, self.cfg
        vals, routes = {}, {}
        # maintenance outcomes due
        for item in [r for r in self.repairs if r[0] <= t]:
            self.repairs.remove(item)
            _, kind, asset = item
            if kind == "well":
                emu.apply_event("repair", asset, replace=True)       # workover: new pump
            else:
                emu.apply_event("restore_units", asset)
        # react to failures / trips seen on SCADA
        for g in self.prog.groups:
            S = emu.S[g.gid]
            for li, iid in enumerate(g.inst_ids):
                inst = self.flat.instances[iid]
                if "up" in S and "q_max" in g.ir.params and float(S["up"].reshape(-1)[li]) < 0.5 \
                        and inst.path not in emu.hidden and (inst.path, "well") not in self.handled:
                    self.handled.add((inst.path, "well"))
                    day = math.floor(t / 24.0) * 24.0 + c["rig_start_hour"]
                    start = day if day > t else day + 24.0
                    busy = [r for r in self.repairs if r[1] == "well"]
                    start = max([start] + [r[0] for r in busy])
                    self.repairs.append((start + c["rig_job_h"], "well", inst.path))
                if "avail" in S and "n_units" in g.ir.params:
                    n = float(emu.P[g.gid]["n_units"].reshape(-1)[li])
                    a = float(S["avail"].reshape(-1)[li])
                    key = (inst.path, "units", round(t // 24))
                    if a < n - 0.5 and not any(r[2] == inst.path for r in self.repairs):
                        self.repairs.append((t + c["crew_callout_h"], "units", inst.path))
        for (p, kind) in list(self.handled):
            gid, li = emu.path_index[p]
            if float(emu.S[gid]["up"].reshape(-1)[li]) > 0.5:
                self.handled.discard((p, kind))
        # flaring response
        flare = sum(self._sig(f.path, "node:local:burned") for f in self.flat.instances if f.type.short == "flare")
        if flare > 1e-3:
            self.flare_since = self.flare_since if self.flare_since is not None else t
        else:
            self.flare_since = None
        if self.flare_since is not None and t - self.flare_since >= c["flare_response_h"]:
            for p in self.gassy[:4]:
                if p not in self.curtail:
                    self.curtail[p] = (c["curtail"], t)
        # midstream takeaway at its limit for > 2 h: curtail the gassiest pads the same way, and keep
        # them curtailed until gas sales are clearly below the limit
        if self.gas_plant is not None:
            deliv = self._sig(self.gas_plant, "node:local:delivered")
            cap = self._param(self.gas_plant, "cap")
            if deliv >= 0.98 * cap:
                self.takeaway_since = self.takeaway_since if self.takeaway_since is not None else t
            elif deliv < 0.90 * cap:
                self.takeaway_since = None
            if self.takeaway_since is not None and t - self.takeaway_since >= c["flare_response_h"]:
                for p in self.gassy[:4]:
                    self.tcurtail.setdefault(p, t)
            elif self.takeaway_since is None:
                for p in [p for p, t0 in self.tcurtail.items() if t - t0 > c["flare_response_h"] + 2.0]:
                    del self.tcurtail[p]
        if self.flare_since is None:
            for p in list(self.curtail):
                if t - self.curtail[p][1] > c["flare_response_h"] + 2.0:
                    del self.curtail[p]
        # water tank high despite re-balancing: curtail the wettest pads of that battery by 20 %
        for b in [i.path for i in self.flat.instances if i.type.short == "battery"]:
            lvl = self._state(b, "wat_lvl") / max(1.0, self._param(b, "wat_tank"))
            pads = [p for p in self.pads if self._battery_of(p) == b]
            if lvl > 0.85:
                wet = sorted(pads, key=lambda p: -self._pad_wc(p))
                for p in wet[:2]:
                    self.wcurtail.setdefault(p, t)
            elif lvl < 0.6:
                for p in pads:
                    if p in self.wcurtail and t - self.wcurtail[p] > 2.0:
                        del self.wcurtail[p]
        # wells
        dr = any(s <= t + 1e-9 < e for s, e in self.dr_windows)
        for w in self.wells:
            if w in emu.hidden:
                continue
            f = self.curtail.get(self.well_pad.get(w), (1.0, 0))[0]
            if self.well_pad.get(w) in self.wcurtail or self.well_pad.get(w) in self.tcurtail:
                f = min(f, c["curtail"])
            sp = max(0.35, c["speed"] * f)
            vals[(w, "speed")] = min(sp, c.get("dr_speed", 0.75)) if dr else sp
            vals[(w, "on")] = 1.0
        # compressors: all available units
        for inst in self.flat.instances:
            if inst.type.short == "compressor_station":
                vals[(inst.path, "units")] = self._state(inst.path, "avail")
        # tank level controllers
        for inst in self.flat.instances:
            if inst.type.short != "battery":
                continue
            gid, li = emu.path_index[inst.path]
            P = emu.P[gid]
            for lvl, cap, rate, pump in (("oil_lvl", "oil_tank", "oil_rate", "oil_pump"),
                                         ("wat_lvl", "wat_tank", "wat_rate", "wat_pump")):
                level = self._state(inst.path, lvl)
                capv = float(P[cap].reshape(-1)[li])
                inflow = self._sig(inst.path, "in:feed:oil" if lvl == "oil_lvl" else "in:feed:water")
                target = inflow + 0.6 * (level - 0.5 * capv) / 1.0
                vals[(inst.path, rate)] = max(0.0, min(float(P[pump].reshape(-1)[li]), target))
        # oil routing at CTB Central: pipeline first, trucks for the excess above ~95 % of line capacity
        pipe_cap = U.to_canonical(5200, "bbl/d") * 0.95
        c_oil = vals.get(("CTB_C", "oil_rate"), 0.0)
        to_pipe = min(1.0, pipe_cap / c_oil) if c_oil > 1e-9 else 1.0
        for key in self.emu.route_key:
            src, port, dst, dport = key
            if src == "CTB_C" and port == "oil_out":
                routes[key] = to_pipe if "truck" not in self._type(dst) else 1.0 - to_pipe
        # produced water: the control room keeps each home disposal well below its pressure alarm by
        # shifting part of the battery's water to the second outlet (a few % per 10 min), and shifts it
        # back once the pressure is comfortably low again
        alarm = U.to_canonical(2000, "psi")
        by_src: dict = {}
        for key in self.emu.route_key:
            if key[1] == "wat_out":
                by_src.setdefault(key[0], []).append(key)
        for src, keys in by_src.items():
            home = keys[0]
            swd = self._swd_of(home[2])
            p = self._sig(swd, "node:local:p_inj") if swd else 0.0
            h = self.water_div.get(src, 1.0)
            if p > 0.95 * alarm:
                h = max(0.2, h - 0.05)
            elif p < 0.85 * alarm:
                h = min(1.0, h + 0.02)
            self.water_div[src] = h
            for i, key in enumerate(keys):
                routes[key] = h if i == 0 else (1.0 - h) / (len(keys) - 1)
        # truck dispatch every 2 h: fleet split proportional to trucked oil volume waiting
        if t - self.last_truck_alloc >= 2.0:
            self.last_truck_alloc = t
            need = {}
            for key in self.emu.route_key:
                src, port, dst, dport = key
                if src == "FLEET":
                    lane = dst
                    lane_inst = self.flat.instances[self.flat.by_path[lane]]
                    origin = lane.split("__")[0] if "__" in lane else None
                    need[key] = 1.0
                    if "OIL" in lane:
                        b = "CTB_S" if "_S_" in lane else "CTB_C"
                        lvl = self._state(b, "oil_lvl")
                        need[key] = 0.2 + lvl / max(1.0, self._param(b, "oil_tank"))
                        if b == "CTB_S":
                            need[key] += 0.8
                    else:
                        need[key] = 0.35
            tot = sum(need.values()) or 1.0
            for key, v in need.items():
                routes[key] = v / tot
        emu.set_controls(vals, routes)

    def _battery_of(self, pad: str):
        """Battery a pad is currently lined up to (largest routing share)."""
        best, share = None, -1.0
        for key, r in self.emu.route_key.items():
            if key[0] == pad:
                v = float(self.emu.R[0, r, 0])
                if v > share:
                    best, share = key[2].split("__")[-1], v
        return best

    def _pad_wc(self, pad: str) -> float:
        ws = [w for w, p in self.well_pad.items() if p == pad]
        return sum(self._param(w, "wc") for w in ws) / max(1, len(ws))

    def _swd_of(self, link_path: str):
        """Disposal well fed by a water link (flowline 'CTB_N__SWD1' or a truck lane)."""
        iid = self.flat.by_path.get(link_path)
        if iid is None:
            return None
        inst = self.flat.instances[iid]
        for pid in inst.ports.values():
            p = self.flat.ports[pid]
            if p.direction == "out":
                for eid in p.edges:
                    dst = self.flat.instances[self.flat.ports[self.flat.edges[eid].dst].inst]
                    if dst.type.short == "swd_well":
                        return dst.path
        return None

    def _type(self, path: str) -> str:
        iid = self.flat.by_path.get(path)
        return self.flat.instances[iid].type.short if iid is not None else ""
