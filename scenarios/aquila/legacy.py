"""Baseline: conventional planning of a steel group - a corporate S&OP plan and the site control rooms.

The comparison is only meaningful if the baseline is competent. These rules encode common practice of a steel group
that plans with spreadsheets and standard costs instead of a corporate optimiser (documented in docs/SHOWCASE.md):

* S&OP (sales and operations planning): every 24 h, and at once after a notice or when a unit goes down or comes
  back, the planning department allocates each market's contract volume to the sales lanes in order of delivered
  margin (index price - freight - the site's standard variable cost of the product - its standard CO2e at the
  corporate's internal carbon price, the price the plan's objective uses too), within each site's capacity
  (blast furnaces, converters, arc furnaces, strip, galvanising and bar mills at 97 % utilisation) and each lane's
  capacity (a contract also at a loss smaller than its liquidated damages); spot volume next, wherever the margin
  after the spot discount is positive. A strip mill may roll a
  sister plant's slabs (slab transfers, at that plant's cost plus freight) where that is the better margin, Tubarao
  slabs go to Burns Harbor and Texas DRI to Osceola the same way. Blast furnaces stay at least at their 60 %
  turndown: what that iron makes beyond the allocation is offered spot wherever it covers freight and rolling;
* make to order: each mill rolls what its lanes ship plus a correction towards its yard's target stock (the start
  stock, over 24 h); an arc furnace is planned at what it can make on average - its capacity less the tariff's peak
  hours, within the site's grid connection - so the sales plan does not run its yards down; the melt shops make what the mills and slab lanes need, with the same correction for the slab
  and billet yards (blast furnaces by share of full wind, within 6 % of full wind per hour; converters at the
  standard 84 % hot metal, 78 % while a site's iron is short);
* arc furnaces follow the published time-of-use tariff: in the four peak hours of their zone they melt at 30 % and
  make it up in the other hours; during an announced price window above `dr_price` they stop (demand response);
  the site's grid connection limit is respected;
* metallics: the DRI/HBI share of a charge is at the furnace's maximum where DRI is cheaper than scrap (own plant,
  transfer or merchant HBI, including its extra power and its carbon), otherwise the least share that keeps the cast's copper 10 %
  inside the caster's limit, with the lab's copper of the bay (a running mean of the lab results - the same samples
  PLYXIS gets); DRI plants make what their furnaces and transfers draw, within the gas contract; scrap bays and
  HBI piles are bought back to their target stock;
* sales desk: lanes ship the allocation while the yard holds stock (spot only above 60 % of the target stock), sell
  what a yard holds above 115 % of its target, catch up a market's backlog, and never ship more than a market's
  call-offs plus its spot depth.
"""
from __future__ import annotations

import csv
import math
import os

from plyxis.lang import units as U

HERE = os.path.dirname(os.path.abspath(__file__))
K = 1000.0                      # canonical kg per t: rates kg/h, stocks kg
HM_SHARE, BOF_YIELD, CAST_YIELD, HSM_YIELD, CGL_YIELD, BAR_YIELD, EAF_YIELD = 0.84, 0.91, 0.98, 0.975, 0.95, 0.96, 0.9
# tonnes of each intermediate per tonne of product sold
USE = {"coated": {"cgl": 1.0, "hsm": 1 / CGL_YIELD, "slab": 1 / (CGL_YIELD * HSM_YIELD)},
       "hrc": {"hsm": 1.0, "slab": 1 / HSM_YIELD},
       "slab": {"slab": 1.0},
       "long": {"bar": 1.0, "billet": 1 / BAR_YIELD}}
UNIT = {"power": "USD/MWh", "gas": "USD/GJ", "eua": "USD/tCO2e"}


def _rows(name: str) -> list:
    with open(os.path.join(HERE, name), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class LegacyOperators:
    def __init__(self, emu, cfg: dict):
        self.emu, self.cfg = emu, cfg
        self.util = float(cfg.get("utilisation", 0.97))
        self.sites = {r["name"]: r for r in _rows("sites.csv")}
        self.markets = {r["name"]: r for r in _rows("markets.csv")}
        self.lanes = []
        for kind in ("hrc", "coated", "long", "slab"):
            for r in _rows(f"lanes_{kind}.csv"):
                self.lanes.append({"name": r["name"], "site": r["site"], "src": r["src"], "market": r["market"],
                                   "freight": float(r["freight"]), "cap": float(r["cap"]), "kind": kind})
        self.xfers = {r["name"]: {"src": r["src"], "dst": r["dst"], "freight": float(r["freight"]), "cap": float(r["cap"])}
                      for r in _rows("transfers_slab.csv") + _rows("transfers_dri.csv")}
        cp = float(cfg.get("carbon_price", 0.0)) / 1000.0           # USD per kg CO2e
        self.yards = {r["name"]: {"site": r["site"], "kind": r["kind"], "target": float(r["stock"]),
                                  "cost": float(r["unit_cost"]) + cp * float(r.get("co2") or 0.0)}
                      for r in _rows("product_yards.csv")}
        self.yard_of = {(y["site"], y["kind"]): n for n, y in self.yards.items()}
        self.bfs: dict = {}
        for r in _rows("blast_furnaces.csv"):
            self.bfs.setdefault(r["site"], []).append((r["name"], float(r["rate_max"])))
        shops = _rows("converter_shops.csv")
        self.bof = {r["site"]: r["name"] for r in shops}
        self.bof_cap = {r["site"]: float(r["cap"]) for r in shops}
        self.bof_bay = {r["site"]: r["yard"] for r in shops}
        billet = {r["name"]: r for r in _rows("billet_casters.csv")}
        self.casters = {**{r["name"]: r for r in _rows("casters.csv")}, **billet}
        self.eafs = {r["name"]: {"site": r["site"], "cap": float(r["cap"]), "kwh": float(r["kwh_t"]), "caster": r["caster"],
                                 "bay": r["yard"], "hbi": r["hbi"], "dri_max": float(r["dri_max"]),
                                 "kind": "billet" if r["caster"] in billet else "slab"} for r in _rows("arc_furnaces.csv")}
        self.dri = {r["name"]: {"site": r["site"], "cap": float(r["cap"]), "yard": r["yard"]} for r in _rows("dri_plants.csv")}
        self.dri_of = {v["yard"]: n for n, v in self.dri.items()}
        self.piles = {r["name"]: {"site": r["site"], "target": float(r["stock"]), "buy_max": float(r["buy_max"]),
                                  "cost": float(r["unit_cost"]) + cp * float(r.get("co2") or 0.0)}
                      for r in _rows("dri_yards.csv")}
        self.hsm = {r["site"]: (r["name"], float(r["cap"])) for r in _rows("strip_mills.csv")}
        self.cgl = {r["site"]: (r["name"], float(r["cap"])) for r in _rows("coating_lines.csv")}
        self.bar = {r["site"]: (r["name"], float(r["cap"])) for r in _rows("bar_mills.csv")}
        self.bays = {r["name"]: {"site": r["site"], "buy_max": float(r["buy_max"]), "target": float(r["stock"]),
                                 "market": r["market"], "cost": float(r["unit_cost"])} for r in _rows("scrap_yards.csv")}
        self.cu = {r["name"]: float(r["buy_cu"]) for r in _rows("scrap_yards.csv")}      # the buyers' spec until the lab reports
        self.cu_var: dict = {}
        self.assay_tag = {x[3]: x[2] for x in emu.tele if x[6].get("test")}
        self.units = [n for bf in self.bfs.values() for n, _r in bf] + list(self.eafs) + list(self.dri) + \
            [n for n, _c in self.hsm.values()] + [n for n, _c in self.cgl.values()] + [n for n, _c in self.bar.values()] + \
            list(self.bof.values()) + list(self.casters)
        self.dr_windows: list = []          # (zone, start, end)
        self.plan = None
        self.next_sop = 0.0
        self.sig = None
        self.bf_set: dict = {}
        self.short: set = set()

    # ------------------------------------------------------------------ helpers
    def _p(self, path, name):
        g, li = self.emu.path_index[path]
        return float(self.emu.P[g][name].reshape(-1)[li])

    def _s(self, path, name):
        g, li = self.emu.path_index[path]
        return float(self.emu.S[g][name].reshape(-1)[li])

    def _up(self, path) -> bool:
        return self._s(path, "up") > 0.5

    def _stock(self, yard) -> float:
        return self._s(yard, "stock") / K

    def _price(self, name: str) -> float:
        v = float(self.emu.env_now()[name].reshape(-1)[0])
        return U.from_canonical(v, UNIT.get(name.split("_")[0], "USD/t"))

    def _market_price(self, m: str) -> float:
        r = self.markets[m]
        return self._price(f"hrc_{r['region']}") * float(r["beta"]) + float(r["premium"])

    def notice(self, ev: dict, t: float) -> None:
        if ev.get("kind") == "announce" and str(ev.get("name", "")).startswith("power_") \
                and float(ev["value"]) >= self.cfg.get("dr_price", 300.0):
            self.dr_windows.append((ev["name"][6:], float(ev["start"]), float(ev["end"])))
        self.next_sop = t                   # every notice goes to the planning department at once

    def assays(self, telemetry: list) -> None:
        """The lab's copper of each scrap bay: a running mean of its results, weighted like the twin's update."""
        sigma = self.emu.assays.get("sigma", 0.08)
        for m in telemetry:
            b = self.assay_tag.get(m.tag)
            if b is None or m.quality != "test":
                continue
            var = self.cu_var.get(b, (0.15 * self.cu[b]) ** 2)
            meas = (sigma * m.value) ** 2
            gain = var / (var + meas)
            self.cu[b] += gain * (m.value - self.cu[b])
            self.cu_var[b] = (1.0 - gain) * var

    # ------------------------------------------------------------------ S&OP
    def _sop(self, t: float) -> None:
        u = self.util
        res: dict = {}                      # (resource, site or transfer) -> t/h left
        hm_max, hm_min = {}, {}
        for s, bf in self.bfs.items():
            hm_max[s] = sum(r for n, r in bf if self._up(n)) * u
            hm_min[s] = sum(0.6 * r for n, r in bf if self._up(n))
            bof = min(hm_max[s] / HM_SHARE * BOF_YIELD, self.bof_cap[s] * u) if self._up(self.bof[s]) else 0.0
            res[("slab", s)] = bof * CAST_YIELD
        for e, v in self.eafs.items():
            if self._up(e) and self._up(v["caster"]):
                key = (v["kind"], v["site"])
                res[key] = res.get(key, 0.0) + self._eaf_capacity(e) * CAST_YIELD
        for kind, table in (("hsm", self.hsm), ("cgl", self.cgl), ("bar", self.bar)):
            for s, (n, c) in table.items():
                res[(kind, s)] = c * u if self._up(n) else 0.0
        for x, v in self.xfers.items():
            res[("xfer", x)] = v["cap"]
        cost = {(y["site"], y["kind"]): y["cost"] for y in self.yards.values()}
        slab_in = {}                        # destination site -> [(transfer, source site)]
        for x, v in self.xfers.items():
            if x.startswith("T_") and v["src"].endswith("_SLABS"):
                slab_in.setdefault(v["dst"].split("_")[0], []).append((x, v["src"].split("_")[0]))
        # sales options: a lane, with the site's own slabs or a sister plant's (transfer)
        opts = []
        for ln in self.lanes:
            s, k, m = ln["site"], ln["kind"], ln["market"]
            base = self._market_price(m) - ln["freight"] - cost[(s, k)]
            opts.append((base, ln, s, None))
            if k in ("hrc", "coated"):
                for x, src in slab_in.get(s, []):
                    if (src, "slab") in cost and (s, "slab") in cost:
                        d = (cost[(s, "slab")] - cost[(src, "slab")] - self.xfers[x]["freight"]) * USE[k]["slab"]
                        opts.append((base + d, ln, src, x))
        left = {m: float(self._p(m, "contract")) / K for m in self.markets}
        spot = {m: float(self._p(m, "spot")) / K * 0.95 for m in self.markets}
        disc = {m: float(self._p(m, "spot_disc")) * K for m in self.markets}
        lane_left = {ln["name"]: ln["cap"] for ln in self.lanes}
        alloc = {ln["name"]: [0.0, 0.0] for ln in self.lanes}
        use_slab = {s: 0.0 for s in self.sites}
        xfer = {x: 0.0 for x in self.xfers}

        def take(margin, ln, src, x, room: dict, slot: int, cap_slab=None) -> float:
            k, s = ln["kind"], ln["site"]
            need = {}
            for r, q in USE[k].items():
                need[(r, src if r == "slab" else s)] = q
            if x is not None:
                need[("xfer", x)] = USE[k]["slab"]
            amt = min(lane_left[ln["name"]], room[ln["market"]])
            for key, q in need.items():
                amt = min(amt, res.get(key, 0.0) / q)
            if cap_slab is not None:
                amt = min(amt, cap_slab / USE[k]["slab"])
            if amt <= 1e-6:
                return 0.0
            for key, q in need.items():
                res[key] -= amt * q
            lane_left[ln["name"]] -= amt
            room[ln["market"]] -= amt
            alloc[ln["name"]][slot] += amt
            if "slab" in USE[k]:
                use_slab[src] += amt * USE[k]["slab"]
            if x is not None:
                xfer[x] += amt * USE[k]["slab"]
            return amt

        # contracts, best margin first - also at a loss, as long as it is smaller than the liquidated damages
        short = {m: float(self._p(m, "short_cost")) * K for m in self.markets}
        for margin, ln, src, x in sorted(opts, key=lambda o: -o[0]):
            if margin > -short[ln["market"]]:
                take(margin, ln, src, x, left, 0)
        for margin, ln, src, x in sorted(opts, key=lambda o: -(o[0] - disc[o[1]["market"]])):   # then spot
            if margin - disc[ln["market"]] > 0:
                take(margin, ln, src, x, spot, 1)
        # iron a furnace makes at its turndown anyway: offered wherever it covers freight and the rolling
        for s in self.bfs:
            surplus = hm_min[s] / HM_SHARE * BOF_YIELD * CAST_YIELD - use_slab[s]
            mine = [o for o in opts if o[2] == s and o[3] is None and "slab" in USE[o[1]["kind"]]]
            for margin, ln, src, x in sorted(mine, key=lambda o: -o[0]):
                if surplus <= 1e-6:
                    break
                k = ln["kind"]
                roll = margin + cost[(s, k)] - cost[(s, "slab")] * USE[k]["slab"]      # margin with the slab for free
                for room, slot in ((left, 0), (spot, 1)):
                    d = 0.0 if slot == 0 else disc[ln["market"]]
                    if roll - d > 0 and surplus > 1e-6:
                        surplus -= take(margin, ln, src, x, room, slot, cap_slab=surplus) * USE[k]["slab"]
        # DRI transfers wherever the source's DRI and freight cost less than the destination's
        dri_xfer = {}
        for x, v in self.xfers.items():
            if v["src"].endswith("_HBI"):
                src, dst = self.piles[v["src"]], self.piles[v["dst"]]
                dri_xfer[x] = v["cap"] if src["cost"] + v["freight"] < dst["cost"] else 0.0
        # production plan per site (t/h)
        plan = {"lane": alloc, "xfer": xfer, "dri_xfer": dri_xfer, "hm": {}, "eaf": {}, "cgl": {}, "hsm": {}, "bar": {}}
        sold = lambda s, k: sum(sum(alloc[ln["name"]]) for ln in self.lanes if ln["site"] == s and ln["kind"] == k)  # noqa: E731
        for s in self.sites:
            plan["cgl"][s] = sold(s, "coated")
            plan["hsm"][s] = sold(s, "hrc") + plan["cgl"][s] / CGL_YIELD
            plan["bar"][s] = sold(s, "long")
        for s in self.bfs:
            steel = use_slab[s] / CAST_YIELD
            plan["hm"][s] = min(max(steel * HM_SHARE / BOF_YIELD, hm_min[s]), hm_max[s])
        need = {}
        for s in self.sites:
            if s not in self.bfs:
                need[(s, "slab")] = use_slab[s] / CAST_YIELD
            need[(s, "billet")] = plan["bar"][s] / BAR_YIELD / CAST_YIELD
        for (s, kind), q in need.items():
            fs = [e for e, v in self.eafs.items() if v["site"] == s and v["kind"] == kind and self._up(e)]
            cap = sum(self.eafs[e]["cap"] for e in fs)
            for e in fs:
                plan["eaf"][e] = q * self.eafs[e]["cap"] / cap if cap > 0 else 0.0
        self.plan = plan
        self.next_sop = (math.floor(t / 24.0) + 1) * 24.0

    def _eaf_capacity(self, e: str) -> float:
        """What an arc furnace can make on average, t/h: its capacity at the planning utilisation, less what the
        tariff rule takes in the peak hours, and within its share of the site's grid connection (nominal loads)."""
        v, c = self.eafs[e], self.cfg
        s = v["site"]
        peaks = c.get("peak_hours", {}).get(self.sites[s]["zone"], [])
        cap = v["cap"] * self.util * (24.0 - len(peaks) * (1.0 - c.get("peak_factor", 0.3))) / 24.0
        fs = [f for f in self.eafs if self.eafs[f]["site"] == s]
        other = 105 * self.hsm.get(s, ("", 0.0))[1] + 140 * self.cgl.get(s, ("", 0.0))[1] + 110 * self.bar.get(s, ("", 0.0))[1] \
            + sum(110 * d["cap"] for d in self.dri.values() if d["site"] == s)
        room = 0.98 * self._p(f"{s}_GRID", "max_demand") / K - other / 1000.0
        need = sum(self.eafs[f]["cap"] * (self.eafs[f]["kwh"] + 150 * self.eafs[f]["dri_max"] + 20) / 1000.0 for f in fs)
        return cap * min(1.0, max(0.0, room) / need) if need > 0 else cap

    # ------------------------------------------------------------------ control step
    def control(self, t: float) -> None:
        sig = tuple(self._up(p) for p in self.units)
        if self.plan is None or t >= self.next_sop - 1e-9 or sig != self.sig:
            self._sop(t)
            self.sig = sig
        plan, c, dt = self.plan, self.cfg, self.emu.dt
        vals: dict = {}

        def rate(path, name, v):
            vals[(path, name)] = max(0.0, v) * K

        def yard_corr(yard):
            return (self.yards[yard]["target"] - self._stock(yard)) / 24.0 if yard in self.yards else 0.0

        def avail(yard, want):
            """A mill or lane takes no more than its yard can give: slows down below 30 % of the target stock."""
            y = self.yards.get(yard)
            if y is None:
                return want
            s = self._stock(yard)
            return want * min(1.0, s / (0.3 * y["target"])) if s < 0.3 * y["target"] else want

        # --- sales desk
        ship: dict = {}
        for ln in self.lanes:
            y, (con, sp) = ln["src"], plan["lane"][ln["name"]]
            s, tgt = self._stock(y), self.yards[y]["target"]
            v = con + (sp if s > 0.6 * tgt else 0.0)
            if s > 1.15 * tgt:
                v += (s - tgt) / 24.0
            ship[ln["name"]] = min(ln["cap"], avail(y, v))
        for m in self.markets:
            ls = [ln for ln in self.lanes if ln["market"] == m]
            backlog = max(0.0, self._s(m, "backlog") / K)
            contract, spot = self._p(m, "contract") / K, self._p(m, "spot") / K
            if backlog > 1.0:                   # catch up, from lanes whose yards hold stock
                ok = [ln for ln in ls if self._stock(ln["src"]) > 0.6 * self.yards[ln["src"]]["target"]]
                room = sum(ln["cap"] - ship[ln["name"]] for ln in ok)
                for ln in ok:
                    if room > 0:
                        ship[ln["name"]] += min(backlog / 12.0, room) * (ln["cap"] - ship[ln["name"]]) / room
            limit = contract + backlog / dt + 0.97 * spot
            tot = sum(ship[ln["name"]] for ln in ls)
            for ln in ls:
                if tot > limit:
                    ship[ln["name"]] *= limit / tot
                rate(ln["name"], "ship", ship[ln["name"]])
        for x, v in self.xfers.items():
            q = plan["xfer"].get(x, 0.0) if v["src"].endswith("_SLABS") else plan["dri_xfer"].get(x, 0.0)
            src = v["src"]
            stock = self._stock(src)
            tgt = self.yards[src]["target"] if src in self.yards else self.piles[src]["target"]
            rate(x, "ship", q if stock > 0.3 * tgt else q * stock / (0.3 * tgt))
        # --- mills: make to order + yard correction
        for s, (n, cap) in self.cgl.items():
            want = plan["cgl"][s] + yard_corr(self.yard_of[(s, "coated")])
            rate(n, "rate", min(cap, avail(self.yard_of[(s, "hrc")], want)))
        for s, (n, cap) in self.hsm.items():
            want = plan["hsm"][s] + yard_corr(self.yard_of[(s, "hrc")])
            rate(n, "rate", min(cap, avail(self.yard_of[(s, "slab")], want)))
        for s, (n, cap) in self.bar.items():
            want = plan["bar"][s] + yard_corr(self.yard_of[(s, "long")])
            rate(n, "rate", min(cap, avail(self.yard_of[(s, "billet")], want)))
        # --- blast furnaces and converters
        for s, bf in self.bfs.items():
            ups = [(n, r) for n, r in bf if self._up(n)]
            full = sum(r for _n, r in ups)
            want = plan["hm"][s] + yard_corr(self.yard_of[(s, "slab")]) / CAST_YIELD * HM_SHARE / BOF_YIELD
            hm = min(max(want, 0.6 * full), full)
            short = want > full + 1e-6
            for n, r in ups:
                v = hm * r / full if full > 0 else 0.0
                last = self.bf_set.get(n, self._s(n, "last") / K)
                step = 0.06 * r * dt
                v = min(max(v, last - step), last + step)
                self.bf_set[n] = v
                rate(n, "rate", v)
            vals[(self.bof[s], "hm_share")] = 0.78 if short else HM_SHARE
        # --- arc furnaces: tariff, demand response, connection limit
        hour = int(math.floor(t)) % 24
        pf = c.get("peak_factor", 0.3)
        eaf_rate, dri_share = {}, {}
        for e, v in self.eafs.items():
            s = v["site"]
            zone = self.sites[s]["zone"]
            yard = self.yard_of[(s, v["kind"])]
            n_kind = sum(1 for f, w in self.eafs.items() if w["site"] == s and w["kind"] == v["kind"])
            need = plan["eaf"].get(e, 0.0) + yard_corr(yard) / CAST_YIELD / n_kind
            peaks = c.get("peak_hours", {}).get(zone, [])
            r_off = min(v["cap"], max(0.0, need) * 24.0 / (24.0 - len(peaks) * (1.0 - pf)))
            r = pf * r_off if hour in peaks else r_off
            if any(z == zone and a <= t < b for z, a, b in self.dr_windows):
                r = 0.0
            eaf_rate[e] = r if self._up(e) else 0.0
            # metallic mix: DRI where it is cheaper than scrap, else the copper minimum
            pw = self._price(f"power_{zone}") / 1000.0
            dri_cost = self.piles[v["hbi"]]["cost"] + 150.0 * pw
            cu_max = float(self.casters[v["caster"]]["cu_max"])
            cu_min_share = max(0.0, 1.0 - c.get("cu_margin", 0.9) * cu_max * EAF_YIELD / max(self.cu[v["bay"]], 1e-6))
            share = v["dri_max"] if dri_cost < self.bays[v["bay"]]["cost"] else cu_min_share
            pile = self._stock(v["hbi"])
            if pile < 0.2 * self.piles[v["hbi"]]["target"]:
                share = min(share, cu_min_share)
            dri_share[e] = min(v["dri_max"], share)
        for s in self.sites:                    # the site's connection limit (nominal loads of the plan)
            fs = [e for e, v in self.eafs.items() if v["site"] == s]
            if not fs:
                continue
            g = f"{s}_GRID"
            other = 110 * sum(plan["hsm"].get(x, 0) for x in [s]) + 140 * plan["cgl"].get(s, 0) + 110 * plan["bar"].get(s, 0) \
                + sum(110 * d["cap"] for n, d in self.dri.items() if d["site"] == s) + 20 * sum(eaf_rate[e] for e in fs)
            room = 0.98 * self._p(g, "max_demand") / K - other / 1000.0
            mw = sum(eaf_rate[e] * (self.eafs[e]["kwh"] + 150 * dri_share[e]) / 1000.0 for e in fs)
            if mw > room > 0:
                for e in fs:
                    eaf_rate[e] *= room / mw
        for e in self.eafs:
            rate(e, "steel", eaf_rate[e])
            vals[(e, "dri_share")] = dri_share[e]
        # --- DRI plants: what their furnaces and transfers draw, within the gas contract
        for n, d in self.dri.items():
            s, pile = d["site"], d["yard"]
            draw = sum(eaf_rate[e] / EAF_YIELD * dri_share[e] for e, v in self.eafs.items() if v["hbi"] == pile)
            draw += sum(plan["dri_xfer"].get(x, 0.0) for x, v in self.xfers.items() if v["src"] == pile)
            want = draw + (self.piles[pile]["target"] - self._stock(pile)) / 24.0
            gas_other = 1.25 * plan["hsm"].get(s, 0) + 1.1 * plan["bar"].get(s, 0) + 0.9 * plan["cgl"].get(s, 0)
            gas_cap = U.from_canonical(self._p(f"{s}_GAS", "cap"), "GJ/h")
            rate(n, "rate", min(d["cap"], want, max(0.0, 0.97 * gas_cap - gas_other) / 10.4))
        # --- metallics purchasing: back to the target stock
        buys: dict = {}
        for b, v in self.bays.items():
            s = v["site"]
            if b == self.bof_bay.get(s):
                hm = sum(self.bf_set.get(n, 0.0) for n, _r in self.bfs[s])
                cons = hm / HM_SHARE * (1 - HM_SHARE)
            else:
                cons = sum(eaf_rate[e] / EAF_YIELD * (1 - dri_share[e]) for e, w in self.eafs.items() if w["bay"] == b)
            buys[b] = min(v["buy_max"], max(0.0, cons + (v["target"] - self._stock(b)) / 24.0))
        for mkt in {v["market"] for v in self.bays.values()}:
            bs = [b for b, v in self.bays.items() if v["market"] == mkt]
            tot, cap = sum(buys[b] for b in bs), 0.97 * self._p(mkt, "avail") / K
            for b in bs:
                rate(b, "buy", buys[b] * (cap / tot if tot > cap else 1.0))
        for p, v in self.piles.items():
            draw = sum(eaf_rate[e] / EAF_YIELD * dri_share[e] for e, w in self.eafs.items() if w["hbi"] == p)
            feed = sum(plan["dri_xfer"].get(x, 0.0) for x, w in self.xfers.items() if w["dst"] == p)
            if p in self.dri_of:
                feed += vals.get((self.dri_of[p], "rate"), 0.0) / K
                draw += sum(plan["dri_xfer"].get(x, 0.0) for x, w in self.xfers.items() if w["src"] == p)
            rate(p, "buy", min(v["buy_max"], max(0.0, draw - feed + (v["target"] - self._stock(p)) / 24.0)))
        self.emu.set_controls(vals, {})
