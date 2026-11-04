"""Aquila Steel - asset register, hidden truth and scenario of a synthetic global steel producer.

    python scenarios/aquila/generate.py

Thirteen sites on four continents (about 73 Mt/a of crude steel capacity): seven integrated sites (blast furnace ->
converter -> slab caster -> hot strip mill, most with a galvanising line), six electric sites (direct reduction
and/or merchant HBI + scrap -> electric arc furnace -> slab or billet caster -> strip or bar mill). Around them:
ore and coke yards with contracted vessel deliveries, regional scrap markets, eleven power zones (in local time),
five gas hubs, the EU ETS, regional markets for coil, coated coil, long products and export slabs, sales lanes
and transfers between sites. All data synthetic; magnitudes follow public industry figures.

Writes the CSV tables `field.plx` instantiates, `truth.plx` (the plant as it really is: furnace productivity,
fuel and power rates, scrap copper, mill yields that differ from the engineering values) and `scenario.json`.
"""
from __future__ import annotations

import csv
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 11

# ----------------------------------------------------------------------------- sites
# name, label, lat, lon, region (markets), power zone, gas hub, scrap market, EU ETS share (emissions not covered
# by free allocation), labour + overhead USD/h (crews, maintenance, G&A)
SITES = [
    ("NL", "Velsen (NL)", 52.477, 4.598, "eu", "nl", "ttf", "eu", 0.35, 62000, 20000),
    ("DE", "Duisburg (DE)", 51.488, 6.738, "eu", "de", "ttf", "eu", 0.35, 98000, 30000),
    ("FR", "Dunkirk (FR)", 51.030, 2.300, "eu", "fr", "ttf", "eu", 0.35, 64000, 21000),
    ("ES", "Aviles (ES)", 43.581, -5.920, "eu", "es", "ttf", "eu", 0.35, 41000, 14000),
    ("PL", "Warsaw (PL)", 52.296, 20.906, "eu", "pl", "ttf", "eu", 0.35, 11000, 4000),
    ("IN", "Burns Harbor (US-IN)", 41.630, -87.120, "na", "miso", "hh", "us", 0.0, 60000, 18000),
    ("AR", "Osceola (US-AR)", 35.705, -89.940, "na", "miso", "hh", "us", 0.0, 14000, 5000),
    ("TX", "Corpus Christi (US-TX)", 27.850, -97.330, "na", "ercot", "hh", "us", 0.0, 10000, 4000),
    ("MX", "Monterrey (MX)", 25.700, -100.250, "na", "mx", "hh", "us", 0.0, 12000, 4000),
    ("BR", "Tubarao (BR)", -20.240, -40.250, "sa", "br", "br", "br", 0.0, 42000, 14000),
    ("OD", "Kalinganagar (IN-OD)", 20.960, 86.020, "in", "in", "in", "in", 0.0, 30000, 10000),
    ("GJ", "Hazira (IN-GJ)", 21.120, 72.660, "in", "in", "in", "in", 0.0, 12000, 4000),
    ("SA", "Jubail (SA)", 27.010, 49.660, "me", "sa", "sa", "me", 0.0, 13000, 5000),
]
ZONES = ["nl", "de", "fr", "es", "pl", "miso", "ercot", "mx", "br", "in", "sa"]
HUBS = ["ttf", "hh", "br", "in", "sa"]
SCRAP = ["eu", "us", "br", "in", "me"]
REGIONS = ["eu", "na", "sa", "in", "me"]
UTC = {"nl": 1, "de": 1, "fr": 1, "es": 1, "pl": 1, "miso": -6, "ercot": -6, "mx": -6, "br": -3, "in": 5.5, "sa": 3}
GRID_CO2 = {"nl": 0.33, "de": 0.38, "fr": 0.06, "es": 0.16, "pl": 0.70, "miso": 0.48, "ercot": 0.39, "mx": 0.42,
            "br": 0.09, "in": 0.71, "sa": 0.57}

# integrated sites: blast furnaces (t HM/h at full wind), converter, caster, strip mill, galvanising line, yards
BFS = {"NL": [380, 330], "DE": [460, 420, 340], "FR": [420, 360], "ES": [320], "IN": [430, 380],
       "BR": [390, 340, 300], "OD": [470, 410]}
SHOP = {  # converter cap, caster cap, strip mill cap, coating line cap (0 = none), ore and coke unit cost USD/t
    "NL": (950, 950, 820, 220, 128, 330), "DE": (1600, 1600, 1350, 320, 130, 335), "FR": (1050, 1050, 900, 160, 128, 330),
    "ES": (430, 430, 380, 0, 132, 335), "IN": (1050, 1050, 900, 260, 118, 300), "BR": (1350, 1350, 520, 0, 70, 290),
    "OD": (1150, 1150, 950, 200, 62, 360)}
# electric sites: arc furnaces (cap t/h, kWh/t, caster kind), DRI plant cap (0 = none), merchant HBI t/h, mills
EAFS = {"ES": [(140, 420, "billet")], "PL": [(210, 400, "billet")], "AR": [(210, 395, "slab"), (210, 395, "slab")],
        "TX": [(170, 420, "billet")], "MX": [(290, 430, "slab")], "GJ": [(230, 440, "slab"), (230, 440, "slab")],
        "SA": [(190, 450, "billet"), (190, 450, "slab")]}
DRI = {"TX": 230, "MX": 240, "GJ": 300, "SA": 280}
HBI_BUY = {"ES": 60, "PL": 40, "AR": 120, "TX": 60, "MX": 60, "GJ": 60, "SA": 40}
EMILLS = {  # electric sites' rolling: strip mill cap, coating line cap, bar mill cap
    "ES": (0, 0, 140), "PL": (0, 0, 200), "AR": (400, 140, 0), "TX": (0, 0, 170), "MX": (290, 0, 0),
    "GJ": (450, 0, 0), "SA": (180, 0, 190)}
PELLET_COST = {"TX": 140, "MX": 135, "GJ": 120, "SA": 130}

# markets: name, region, product, contract t/h, spot t/h, beta, premium USD/t, label
MARKETS = [   # name, region, product, contract t/h, spot t/h, beta, premium USD/t, label - contracts at about 80 % of capacity
    ("EU_HRC", "eu", "hrc", 1750, 300, 1.0, 0, "Europe hot-rolled coil"),
    ("EU_COAT", "eu", "coated", 450, 90, 1.0, 170, "Europe coated coil (automotive, appliances)"),
    ("EU_LONG", "eu", "long", 230, 60, 0.95, 0, "Europe rebar and wire rod"),
    ("NA_HRC", "na", "hrc", 1100, 200, 1.0, 0, "North America hot-rolled coil"),
    ("NA_COAT", "na", "coated", 280, 60, 1.0, 190, "North America coated coil"),
    ("NA_LONG", "na", "long", 170, 40, 1.0, 20, "North America rebar"),
    ("SA_HRC", "sa", "hrc", 330, 80, 1.0, 0, "South America hot-rolled coil"),
    ("SLAB_EXP", "sa", "slab", 520, 150, 0.84, 0, "Export slabs (FOB Brazil)"),
    ("IN_HRC", "in", "hrc", 950, 200, 1.0, 0, "India hot-rolled coil"),
    ("IN_COAT", "in", "coated", 130, 40, 1.0, 150, "India coated coil"),
    ("ME_HRC", "me", "hrc", 240, 100, 1.0, 0, "Gulf hot-rolled coil"),
    ("ME_LONG", "me", "long", 150, 50, 1.0, -10, "Gulf rebar"),
]
MCOORD = {"EU": (50.1, 8.7), "NA": (39.0, -86.0), "SA": (-23.5, -46.6), "SLAB": (-20.3, -40.2), "IN": (19.1, 72.9),
          "ME": (25.2, 55.3)}
# sales lanes: site, product, market, freight USD/t, cap t/h
LANES = [
    ("NL", "hrc", "EU_HRC", 24, 700), ("NL", "coated", "EU_COAT", 28, 220),
    ("DE", "hrc", "EU_HRC", 20, 1100), ("DE", "coated", "EU_COAT", 24, 320),
    ("FR", "hrc", "EU_HRC", 24, 800), ("FR", "coated", "EU_COAT", 30, 160),
    ("ES", "hrc", "EU_HRC", 34, 380), ("ES", "long", "EU_LONG", 26, 140), ("PL", "long", "EU_LONG", 22, 200),
    ("IN", "hrc", "NA_HRC", 25, 800), ("IN", "coated", "NA_COAT", 25, 260),
    ("AR", "hrc", "NA_HRC", 30, 400), ("AR", "coated", "NA_COAT", 32, 140), ("TX", "long", "NA_LONG", 20, 170),
    ("MX", "hrc", "NA_HRC", 52, 290),
    ("BR", "hrc", "SA_HRC", 20, 520), ("BR", "slab", "SLAB_EXP", 32, 900),
    ("OD", "hrc", "IN_HRC", 20, 900), ("OD", "coated", "IN_COAT", 26, 200), ("OD", "hrc", "ME_HRC", 46, 300),
    ("GJ", "hrc", "IN_HRC", 18, 450), ("GJ", "hrc", "ME_HRC", 34, 300),
    ("SA", "hrc", "ME_HRC", 14, 180), ("SA", "long", "ME_LONG", 14, 190),
]
# transfers between sites: product, from, to, freight, cap
TRANSFERS = [("slab", "NL", "DE", 16, 250), ("slab", "DE", "NL", 16, 250), ("slab", "FR", "DE", 22, 200),
             ("slab", "BR", "IN", 48, 150), ("dri", "TX", "AR", 26, 120)]
SCRAP_AVAIL = {"eu": 900, "us": 950, "br": 260, "in": 500, "me": 420}
SCRAP_COST = {"eu": 395, "us": 405, "br": 360, "in": 425, "me": 395}
SCRAP_CU = {"eu": 0.17, "us": 0.18, "br": 0.15, "in": 0.20, "me": 0.19}       # what the buyers' specs assume


# nominal prices of the model (field.plx defaults): used to value every inventory at the site's own cost
HRC = {"eu": 680, "na": 820, "sa": 690, "in": 600, "me": 640}
POWER = {"nl": 85, "de": 90, "fr": 70, "es": 65, "pl": 100, "miso": 40, "ercot": 35, "mx": 60, "br": 55, "in": 70, "sa": 45}
GAS = {"ttf": 10.5, "hh": 2.6, "br": 9.0, "in": 11.5, "sa": 2.0}
EUA = 72


PRICE = {m[0]: HRC[m[1]] * m[5] + m[6] for m in MARKETS}


def unit_costs(site: str) -> dict:
    """Value of a tonne in each yard of a site: its variable production cost, from the same coefficients and nominal
    prices the model uses, capped a little below its net realisable value. Variable, not full cost: fixed costs are
    paid whatever the plan does, so making stock must create no value (a plan valued at full cost or NRV produced
    slabs into the yards because each one was "worth" more than it cost) and emptying a yard must not either
    (below variable cost a plan sold stock instead of melting). Only selling earns a margin."""
    s = next(x for x in SITES if x[0] == site)
    zone, hub, sm, ets = s[5], s[6], s[7], s[8]
    pw, gas, scr = POWER[zone] / 1000.0, GAS[hub], SCRAP_COST[sm]
    out = {}
    if site in BFS:
        conv, cast, hsm, cgl, ore_cost, coke_cost = SHOP[site]
        coke = 0.36 if site not in ("OD", "BR") else 0.38
        hm = 1.58 * ore_cost + coke * coke_cost + 0.15 * 140 + 14 + 85 * pw + 1450 / 1000 * ets * EUA
        steel_bof = (0.84 * hm + 0.16 * scr) / 0.91 + 32 + 25 * pw + 110 / 1000 * ets * EUA
        out["hm"], out["steel"] = hm, steel_bof
    if site in EAFS:
        dri = (1.42 * PELLET_COST[site] + 10.4 * gas + 12 + 110 * pw) if site in DRI else 360.0
        share = 0.5 if site in DRI else 0.25
        kwh = sum(k for _c, k, _t in EAFS[site]) / len(EAFS[site])
        steel_eaf = ((1 - share) * scr + share * dri) / 0.9 + 48 + (kwh + 150 * share) * pw
        out["dri"], out["steel_eaf"] = dri, steel_eaf
    base = out.get("steel", out.get("steel_eaf", 500.0))
    out["slab"] = base / 0.98 + 14 + 20 * pw
    long_base = out.get("steel_eaf", base)
    out["billet"] = long_base / 0.98 + 12 + 15 * pw
    out["hrc"] = out["slab"] / 0.975 + 22 + 105 * pw + 1.25 * gas
    out["coated"] = out["hrc"] / 0.95 + 55 + 140 * pw + 0.9 * gas
    out["long"] = out["billet"] / 0.96 + 20 + 110 * pw + 1.1 * gas
    # IAS 2: the lower of cost and net realisable value (selling price - freight, and for semi-finished steel also the
    # variable cost of finishing it), a little below NRV so that selling is never worth less than keeping
    nrv = {}
    for k in ("hrc", "coated", "long", "slab"):
        best = [PRICE[mk] - f for st, kk, mk, f, _c in LANES if st == site and kk == k]
        if best:
            nrv[k] = max(best)
    if "hrc" in nrv:
        nrv["slab"] = max(nrv.get("slab", 0.0), (nrv["hrc"] - (22 + 105 * pw + 1.25 * gas)) * 0.975)
    if "long" in nrv:
        nrv["billet"] = (nrv["long"] - (20 + 110 * pw + 1.1 * gas)) * 0.96
    for k, v in nrv.items():
        out[k] = min(out[k], 0.98 * v)
    return {k: round(v) for k, v in out.items()}


def co2_intensity(site: str) -> dict:
    """Standard CO2e of a tonne of each product of a site, kg (scope 1 + 2, the model's own coefficients): what the
    corporate's internal carbon price charges in the conventional S&OP's merit order, as in the plan's objective."""
    s = next(x for x in SITES if x[0] == site)
    gc = GRID_CO2[s[5]]
    out = {}
    if site in BFS:
        out["hm"] = 1450 + 85 * gc
        out["steel"] = 0.84 * out["hm"] / 0.91 + 110 + 25 * gc
    if site in EAFS:
        out["dri"] = (10.4 * 56 + 110 * gc) if site in DRI else 0.0          # merchant HBI: not the group's emission
        share = 0.5 if site in DRI else 0.25
        kwh = sum(k for _c, k, _t in EAFS[site]) / len(EAFS[site])
        out["steel_eaf"] = share * out["dri"] / 0.9 + 70 + (kwh + 150 * share) * gc
    base = out.get("steel", out.get("steel_eaf", 0.0))
    out["slab"] = base / 0.98 + 20 * gc
    out["billet"] = out.get("steel_eaf", base) / 0.98 + 15 * gc
    out["hrc"] = out["slab"] / 0.975 + 105 * gc + 1.25 * 56
    out["coated"] = out["hrc"] / 0.95 + 140 * gc + 0.9 * 56
    out["long"] = out["billet"] / 0.96 + 110 * gc + 1.1 * 56
    return {k: round(v) for k, v in out.items()}


def site_row(s):
    name, label, lat, lon, region, zone, hub, sm, ets, labour, overhead = s
    r = {"name": name, "label": label, "lat": lat, "lon": lon, "region": region, "zone": zone, "tag": f"AQ.{name}",
         "ets": ets, "labour": labour, "overhead": overhead, "grid_co2": GRID_CO2[zone]}
    for z in ZONES:
        r[f"w_{z}"] = 1 if z == zone else 0
    for h in HUBS:
        r[f"g_{h}"] = 1 if h == hub else 0
    # full load of every unit (MW = t/h x MWh/t): furnaces, converter, casters, mills, lines, direct reduction
    conv, cast, hsm, cgl = SHOP.get(name, (0, 0, 0, 0, 0, 0))[:4]
    ehsm, ecgl, ebar = EMILLS.get(name, (0, 0, 0))
    eaf = sum(c * k / 1000 + c * 0.02 for c, k, _t in EAFS.get(name, []))
    full = (sum(BFS.get(name, [])) * 0.085 + conv * 0.025 + cast * 0.02 + (hsm + ehsm) * 0.105 + (cgl + ecgl) * 0.14
            + ebar * 0.11 + DRI.get(name, 0) * 0.11 + eaf)
    # integrated sites have headroom; an electric site with two furnaces may not run both flat out with its mills
    r["max_demand"] = round(full * (0.85 if len(EAFS.get(name, [])) > 1 else 0.98) + 10)
    r["gas_cap"] = round(1.3 * (SHOP.get(name, (0, 0, 0, 0))[2] + EMILLS.get(name, (0, 0, 0))[0] + EMILLS.get(name, (0, 0, 0))[2])
                         + 11 * DRI.get(name, 0) + 200)
    return r


def off(lat, lon, d_lat, d_lon):
    return round(lat + d_lat, 4), round(lon + d_lon, 4)


def write(name, rows):
    with open(os.path.join(HERE, name), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    rng = random.Random(SEED)
    sites = [site_row(s) for s in SITES]
    by = {s["name"]: s for s in sites}
    write("sites.csv", sites)
    bfs, shops, eafs, dris, ores, cokes, scraps, dyards, pyards, casters, hsms, cgls, bars = ([] for _ in range(13))
    truth = []
    for site, caps in BFS.items():
        s = by[site]
        uc = unit_costs(site)
        conv, cast, hsm, cgl, ore_cost, coke_cost = SHOP[site]
        hm = sum(caps)
        ores.append({"name": f"{site}_ORE", "site": site, "label": f"{s['label']} ore and pellet yard", "tag": f"AQ.{site}.ORE",
                     "lat": off(s["lat"], s["lon"], 0.010, -0.012)[0], "lon": off(s["lat"], s["lon"], 0.010, -0.012)[1],
                     "cap": round(hm * 1.58 * 24 * 40), "delivery": round(hm * 1.58 * 0.92), "unit_cost": ore_cost,
                     "stock": round(hm * 1.58 * 24 * 18), "reserve": round(hm * 1.58 * 24 * 7)})
        cokes.append({"name": f"{site}_COKE", "site": site, "label": f"{s['label']} coke yard", "tag": f"AQ.{site}.COKE",
                      "lat": off(s["lat"], s["lon"], 0.008, -0.004)[0], "lon": off(s["lat"], s["lon"], 0.008, -0.004)[1],
                      "cap": round(hm * 0.36 * 24 * 30), "delivery": round(hm * 0.36 * 0.92), "unit_cost": coke_cost,
                      "stock": round(hm * 0.36 * 24 * 12), "reserve": round(hm * 0.36 * 24 * 5)})
        for i, c in enumerate(caps, 1):
            la, lo = off(s["lat"], s["lon"], 0.003 * i, 0.004 * i)
            bfs.append({"name": f"{site}_BF{i}", "site": site, "label": f"{s['label']} blast furnace {i}", "tag": f"AQ.{site}.BF{i}",
                        "lat": la, "lon": lo, "rate_max": c, "coke_rate": 0.36 if site not in ("OD", "BR") else 0.38,
                        "ets": s["ets"]})
            truth.append(f"set aquila_truth.{site}_BF{i}.rate_max = {round(c * rng.uniform(0.93, 1.03))} [t/h];")
            truth.append(f"set aquila_truth.{site}_BF{i}.coke_rate = {round(0.36 * rng.uniform(1.0, 1.07), 4)};")
        la, lo = off(s["lat"], s["lon"], -0.002, 0.010)
        shops.append({"name": f"{site}_BOF", "site": site, "label": f"{s['label']} converter shop", "tag": f"AQ.{site}.BOF",
                      "lat": la, "lon": lo, "cap": conv, "ets": s["ets"], "caster": f"{site}_CC", "yard": f"{site}_BOFSCRAP"})
        scraps.append({"name": f"{site}_BOFSCRAP", "site": site, "label": f"{s['label']} converter scrap bay", "tag": f"AQ.{site}.BOFSCRAP",
                       "lat": la + 0.001, "lon": lo - 0.002, "buy_max": round(conv * 0.3), "unit_cost": SCRAP_COST[s_scrap(site)],
                       "buy_cu": SCRAP_CU[s_scrap(site)], "stock": round(conv * 0.18 * 24 * 6), "cap": round(conv * 0.18 * 24 * 20),
                       "market": f"SCRAP_{s_scrap(site).upper()}"})
        truth.append(f"set aquila_truth.{site}_BOFSCRAP.buy_cu = {round(SCRAP_CU[s_scrap(site)] * rng.uniform(1.05, 1.3), 3)};")
        casters.append({"name": f"{site}_CC", "site": site, "kind": "slab", "label": f"{s['label']} slab casters", "tag": f"AQ.{site}.CC",
                        "lat": la - 0.002, "lon": lo + 0.003, "cap": cast, "yard": f"{site}_SLABS", "cu_max": 0.12})
        pyards.append({"name": f"{site}_SLABS", "site": site, "kind": "slab", "label": f"{s['label']} slab yard", "tag": f"AQ.{site}.SLABS",
                       "lat": la - 0.004, "lon": lo + 0.006, "cap": round(cast * 24 * 7), "stock": round(cast * 24 * 2.5), "unit_cost": uc["slab"]})
        if hsm:
            hsms.append({"name": f"{site}_HSM", "site": site, "label": f"{s['label']} hot strip mill", "tag": f"AQ.{site}.HSM",
                         "lat": la - 0.006, "lon": lo + 0.010, "cap": hsm, "slabs": f"{site}_SLABS", "coil": f"{site}_COIL"})
            pyards.append({"name": f"{site}_COIL", "site": site, "kind": "hrc", "label": f"{s['label']} coil yard", "tag": f"AQ.{site}.COIL",
                           "lat": la - 0.008, "lon": lo + 0.013, "cap": round(hsm * 24 * 6), "stock": round(hsm * 24 * 2), "unit_cost": uc["hrc"]})
            truth.append(f"set aquila_truth.{site}_HSM.yield = {round(0.975 * rng.uniform(0.985, 1.005), 4)};")
        if cgl:
            cgls.append({"name": f"{site}_CGL", "site": site, "label": f"{s['label']} galvanising line", "tag": f"AQ.{site}.CGL",
                         "lat": la - 0.010, "lon": lo + 0.016, "cap": cgl, "coil": f"{site}_COIL", "yard": f"{site}_COATED"})
            pyards.append({"name": f"{site}_COATED", "site": site, "kind": "coated", "label": f"{s['label']} coated coil warehouse",
                           "tag": f"AQ.{site}.COATED", "lat": la - 0.012, "lon": lo + 0.019, "cap": round(cgl * 24 * 8),
                           "stock": round(cgl * 24 * 2.5), "unit_cost": uc["coated"]})
    for site, units in EAFS.items():
        s = by[site]
        uc = unit_costs(site)
        if site in DRI:
            la, lo = off(s["lat"], s["lon"], 0.012, 0.006)
            dris.append({"name": f"{site}_DRI", "site": site, "label": f"{s['label']} direct reduction plant", "tag": f"AQ.{site}.DRI",
                         "lat": la, "lon": lo, "cap": DRI[site], "pellets": f"{site}_PELLETS", "yard": f"{site}_HBI"})
            truth.append(f"set aquila_truth.{site}_DRI.gj_t = {round(10.4 * rng.uniform(1.0, 1.08), 3)} [GJ/t];")
            ores.append({"name": f"{site}_PELLETS", "site": site, "label": f"{s['label']} pellet yard", "tag": f"AQ.{site}.PELLETS",
                         "lat": la + 0.004, "lon": lo - 0.004, "cap": round(DRI[site] * 1.42 * 24 * 40),
                         "delivery": round(DRI[site] * 1.42 * 0.9), "unit_cost": PELLET_COST[site],
                         "stock": round(DRI[site] * 1.42 * 24 * 15), "reserve": round(DRI[site] * 1.42 * 24 * 5)})
        dyards.append({"name": f"{site}_HBI", "site": site, "label": f"{s['label']} DRI / HBI pile", "tag": f"AQ.{site}.HBI",
                       "lat": off(s["lat"], s["lon"], 0.010, 0.010)[0], "lon": off(s["lat"], s["lon"], 0.010, 0.010)[1],
                       "cap": 150000, "stock": 25000 if site in DRI else 12000, "buy_max": HBI_BUY[site], "unit_cost": uc["dri"]})
        mills = EMILLS[site]
        for i, (cap, kwh, kind) in enumerate(units, 1):
            la, lo = off(s["lat"], s["lon"], -0.004 * i, -0.006 * i)
            cname = f"{site}_CC{i}" if site != "ES" else "ES_BC"
            eafs.append({"name": f"{site}_EAF{i}", "site": site, "label": f"{s['label']} arc furnace {i}", "tag": f"AQ.{site}.EAF{i}",
                         "lat": la, "lon": lo, "cap": cap, "kwh_t": kwh, "ets": s["ets"], "caster": cname,
                         "yard": f"{site}_EAF{i}SCRAP", "hbi": f"{site}_HBI", "dri_max": 0.6 if site in DRI else 0.35})
            truth.append(f"set aquila_truth.{site}_EAF{i}.kwh_t = {round(kwh * rng.uniform(1.02, 1.09))} [kWh/t];")
            truth.append(f"set aquila_truth.{site}_EAF{i}.cap = {round(cap * rng.uniform(0.92, 1.02))} [t/h];")
            scraps.append({"name": f"{site}_EAF{i}SCRAP", "site": site, "label": f"{s['label']} scrap bay {i}", "tag": f"AQ.{site}.EAF{i}SCRAP",
                           "lat": la + 0.001, "lon": lo - 0.002, "buy_max": round(cap * 1.3), "unit_cost": SCRAP_COST[s_scrap(site)],
                           "buy_cu": SCRAP_CU[s_scrap(site)], "stock": round(cap * 1.1 * 24 * 4), "cap": round(cap * 1.1 * 24 * 12),
                           "market": f"SCRAP_{s_scrap(site).upper()}"})
            truth.append(f"set aquila_truth.{site}_EAF{i}SCRAP.buy_cu = {round(SCRAP_CU[s_scrap(site)] * rng.uniform(1.05, 1.3), 3)};")
            yard = f"{site}_BILLETS" if kind == "billet" else (f"{site}_SLABS" if site != "SA" else "SA_SLABS")
            if not any(c["name"] == cname for c in casters):
                casters.append({"name": cname, "site": site, "kind": kind, "label": f"{s['label']} {kind} caster {i}",
                                "tag": f"AQ.{site}.{cname.split('_')[1]}", "lat": la - 0.002, "lon": lo + 0.003,
                                "cap": round(cap * 1.1), "yard": yard, "cu_max": 0.12 if kind == "slab" else 0.35})
            if not any(p["name"] == yard for p in pyards):
                pyards.append({"name": yard, "site": site, "kind": kind, "label": f"{s['label']} {kind} yard", "tag": f"AQ.{site}.{yard.split('_')[1]}",
                               "lat": la - 0.004, "lon": lo + 0.006, "cap": round(cap * 24 * 7), "stock": round(cap * 24 * 2.5),
                               "unit_cost": uc[kind]})
        hsm, cgl, bar = mills
        la, lo = off(s["lat"], s["lon"], -0.010, 0.004)
        if hsm:
            hsms.append({"name": f"{site}_HSM", "site": site, "label": f"{s['label']} hot strip mill", "tag": f"AQ.{site}.HSM",
                         "lat": la, "lon": lo, "cap": hsm, "slabs": f"{site}_SLABS", "coil": f"{site}_COIL"})
            pyards.append({"name": f"{site}_COIL", "site": site, "kind": "hrc", "label": f"{s['label']} coil yard", "tag": f"AQ.{site}.COIL",
                           "lat": la - 0.002, "lon": lo + 0.003, "cap": round(hsm * 24 * 6), "stock": round(hsm * 24 * 2), "unit_cost": uc["hrc"]})
        if cgl:
            cgls.append({"name": f"{site}_CGL", "site": site, "label": f"{s['label']} galvanising line", "tag": f"AQ.{site}.CGL",
                         "lat": la - 0.004, "lon": lo + 0.006, "cap": cgl, "coil": f"{site}_COIL", "yard": f"{site}_COATED"})
            pyards.append({"name": f"{site}_COATED", "site": site, "kind": "coated", "label": f"{s['label']} coated coil warehouse",
                           "tag": f"AQ.{site}.COATED", "lat": la - 0.006, "lon": lo + 0.009, "cap": round(cgl * 24 * 8),
                           "stock": round(cgl * 24 * 2.5), "unit_cost": uc["coated"]})
        if bar:
            bars.append({"name": f"{site}_BAR", "site": site, "label": f"{s['label']} bar and rod mill", "tag": f"AQ.{site}.BAR",
                         "lat": la + 0.004, "lon": lo - 0.006, "cap": bar, "billets": f"{site}_BILLETS", "yard": f"{site}_LONG"})
            pyards.append({"name": f"{site}_LONG", "site": site, "kind": "long", "label": f"{s['label']} rebar and rod store", "tag": f"AQ.{site}.LONG",
                           "lat": la + 0.006, "lon": lo - 0.009, "cap": round(bar * 24 * 8), "stock": round(bar * 24 * 2.5), "unit_cost": uc["long"]})
    write("blast_furnaces.csv", bfs)
    write("converter_shops.csv", shops)
    write("arc_furnaces.csv", eafs)
    write("dri_plants.csv", dris)
    write("ore_yards.csv", ores)
    write("coke_yards.csv", cokes)
    write("scrap_yards.csv", scraps)
    # stock targets of the corporate S&OP (finance sets the working capital, not the operations planner): the start
    # stock, a day and a half to two and a half days of melting, rolling and shipping. Both worlds hold them.
    for y in dyards + pyards:
        y["reserve"] = y["stock"]
    for y in pyards:
        y["co2"] = co2_intensity(y["site"])[y["kind"]]
        y["stock_max"] = round(y["stock"] * 1.1)
    for y in dyards:
        y["co2"] = co2_intensity(y["site"]).get("dri", 0)
    write("dri_yards.csv", dyards)
    write("product_yards.csv", pyards)
    write("casters.csv", [c for c in casters if c["kind"] == "slab"])
    write("billet_casters.csv", [c for c in casters if c["kind"] == "billet"])
    write("strip_mills.csv", hsms)
    write("coating_lines.csv", cgls)
    write("bar_mills.csv", bars)
    markets = []
    for name, region, product, contract, spot, beta, prem, label in MARKETS:
        la, lo = MCOORD[name.split("_")[0]]
        r = {"name": name, "label": label, "tag": f"AQ.MKT.{name}", "lat": la, "lon": lo, "region": region, "product": product,
             "contract": contract, "spot": spot, "beta": beta, "premium": prem}
        for g in REGIONS:
            r[f"r_{g}"] = 1 if g == region else 0
        markets.append(r)
    write("markets.csv", markets)
    sm = []
    for g in SCRAP:
        la, lo = {"eu": (51.2, 4.4), "us": (39.8, -86.2), "br": (-23.0, -43.2), "in": (19.0, 72.8), "me": (26.3, 50.2)}[g]
        r = {"name": f"SCRAP_{g.upper()}", "label": f"Scrap market {g.upper()}", "tag": f"AQ.SCRAP.{g.upper()}", "lat": la, "lon": lo,
             "avail": SCRAP_AVAIL[g]}
        for h in SCRAP:
            r[f"s_{h}"] = 1 if h == g else 0
        sm.append(r)
    write("scrap_markets.csv", sm)
    yard_of = {(p["site"], p["kind"]): p["name"] for p in pyards}
    mkt_at = {m["name"]: (m["lat"], m["lon"]) for m in markets}
    ends = lambda a, b: {"slat": a[0], "slon": a[1], "dlat": b[0], "dlon": b[1]}     # noqa: E731  map geometry of a lane
    at = lambda site: (by[site]["lat"], by[site]["lon"])                            # noqa: E731
    for kind in ("hrc", "coated", "long", "slab"):
        rows = [{"name": f"L_{site}_{m}", "site": site, "src": yard_of[(site, k)], "market": m, "freight": f, "cap": cap,
                 "label": f"{by[site]['label']} -> {m}", **ends(at(site), mkt_at[m])}
                for site, k, m, f, cap in LANES if k == kind]
        write(f"lanes_{kind}.csv", rows)
    slabs = [{"name": f"T_{a}_{b}", "src": yard_of[(a, "slab")], "dst": yard_of[(b, "slab")], "freight": f, "cap": c,
              "label": f"Slab transfer {a} -> {b}", **ends(at(a), at(b))} for k, a, b, f, c in TRANSFERS if k == "slab"]
    write("transfers_slab.csv", slabs)
    dri_t = [{"name": f"T_{a}_{b}", "src": f"{a}_HBI", "dst": f"{b}_HBI", "freight": f, "cap": c, "label": f"HBI transfer {a} -> {b}",
              **ends(at(a), at(b))} for k, a, b, f, c in TRANSFERS if k == "dri"]
    write("transfers_dri.csv", dri_t)
    with open(os.path.join(HERE, "truth.plx"), "w", encoding="utf-8") as fh:
        fh.write("## HIDDEN GROUND TRUTH - used only by the plant emulators, never by the twin.\n")
        fh.write(f"## Generated by generate.py (seed {SEED}). Synthetic data: furnace productivity and fuel rates, arc\n")
        fh.write("## furnace power and productivity, direct-reduction gas rates, scrap copper and strip yields as they\n")
        fh.write("## really are - the engineering values in the register are a few per cent off, as in every plant.\n")
        fh.write("use steel;\n\ndefine aquila_truth as aquila { };\n\n")
        fh.write("\n".join(truth) + "\n")
    print(f"{len(sites)} sites, {len(bfs)} blast furnaces, {len(eafs)} arc furnaces, {len(dris)} DRI plants, {len(hsms)} strip mills, "
          f"{len(cgls)} coating lines, {len(bars)} bar mills, {len(markets)} markets, {sum(1 for x in LANES)} sales lanes, "
          f"{len(TRANSFERS)} transfers")


def s_scrap(site):
    return next(s[7] for s in SITES if s[0] == site)


if __name__ == "__main__":
    main()
