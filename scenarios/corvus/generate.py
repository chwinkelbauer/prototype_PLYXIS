"""Generate the synthetic Corvus underground mine (asset register, 3D openings, hidden truth).

    python scenarios/corvus/generate.py

Writes stopes.csv, levels.csv, orepasses.csv, pumps.csv, equipment.csv, openings.geojson (3D lines,
GeoJSON positions lon, lat, elevation) and truth.plx. Deterministic (seed 7). All data is synthetic.

Layout (local metres, x east, y north, z = elevation, surface at +140 m):
  three ore lenses (North, Central, South) on a NNW-striking structure, 23 production levels between
  -380 and -920 m, ~160 stopes along the ore drives, an ore-pass system per zone (upper / lower),
  a haulage level at -1000 m with two crushers, conveyors to the production shaft, three exhaust raises.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
LAT0, LON0, SURFACE, HAULAGE = -30.9500, 136.8000, 140.0, -1000.0
rng = random.Random(7)


def ll(x: float, y: float) -> tuple:
    lat = LAT0 + y / 111320.0
    lon = LON0 + x / (111320.0 * math.cos(math.radians(LAT0)))
    return round(lat, 6), round(lon, 6)


ZONES = {   # centre (x, y), strike azimuth (deg), level elevations, base grade (Cu-eq %)
    "N": {"c": (-650.0, 760.0), "az": 330.0, "levels": list(range(-380, -741, -60)), "grade": 1.05, "label": "North"},
    "C": {"c": (0.0, 0.0), "az": 322.0, "levels": list(range(-440, -921, -60)), "grade": 1.32, "label": "Central"},
    "S": {"c": (600.0, -800.0), "az": 336.0, "levels": list(range(-560, -921, -60)), "grade": 1.12, "label": "South"},
}
FANS = {"FAN_N": (-980.0, 1050.0), "FAN_C": (-320.0, 260.0), "FAN_S": (980.0, -1150.0)}
SHAFT = (260.0, 160.0)
CRUSHERS = {"CR_N": (-120.0, 380.0), "CR_S": (420.0, -330.0)}
PORTAL = (900.0, 420.0)


def along(zone: str, s: float, off: float = 0.0) -> tuple:
    """Point on the zone's ore drive: s metres along strike from the centre, off metres across."""
    z = ZONES[zone]
    a = math.radians(z["az"])
    cx, cy = z["c"]
    return cx + s * math.sin(a) + off * math.cos(a), cy + s * math.cos(a) - off * math.sin(a)


def main() -> None:
    stopes, levels, orepasses, pumps, equipment, openings = [], [], [], [], [], []
    truth = []
    for zk, z in ZONES.items():
        lv = z["levels"]
        half = len(lv) // 2
        groups = {f"OP_{zk}1": lv[:half + 1], f"OP_{zk}2": lv[half + 1:]}
        for op, glv in groups.items():
            ox, oy = along(zk, 0.0, 40.0)
            if op.endswith("2"):
                ox, oy = along(zk, 30.0, 45.0)
            lat, lon = ll(ox, oy)
            crusher = "CR_N" if zk == "N" or (zk == "C" and op.endswith("1")) else "CR_S"
            alt = "CR_S" if crusher == "CR_N" else "CR_N"
            km = lambda c: round(1.25 * math.dist((ox, oy), CRUSHERS[c]) / 1000.0 + 0.3, 2)  # noqa: E731
            orepasses.append({"name": op, "zone": zk, "lat": lat, "lon": lon, "z": glv[0], "z_bottom": HAULAGE + 15,
                              "cap": 5000 if zk != "C" else 6500, "draw_max": 900, "crusher": crusher,
                              "haul_km": km(crusher), "alt_crusher": alt, "alt_km": km(alt),
                              "label": f"Ore pass {z['label']} {'upper' if op.endswith('1') else 'lower'}",
                              "tag": f"COR.{op.replace('_', '')}"})
            openings.append({"name": f"RAISE_{op}", "kind": "orepass", "label": f"{op} raise",
                             "path": [(ox, oy, glv[0]), (ox, oy, HAULAGE + 15)]})
        for li, elev in enumerate(lv):
            lname = f"{zk}{-elev}"
            op = f"OP_{zk}1" if elev in groups[f"OP_{zk}1"] else f"OP_{zk}2"
            fan = {"N": "FAN_N", "C": "FAN_C", "S": "FAN_S"}[zk]
            fan2 = ("FAN_N" if elev >= -620 else "FAN_S") if zk == "C" else "FAN_C"
            # lens shape: the middle levels are longest and richest
            shape = 1.0 - abs(li - (len(lv) - 1) / 2) / len(lv)
            n_st = max(5, round(5 + 4 * shape + rng.uniform(-0.6, 0.6)))
            length = 45.0 * (n_st - 1)
            cx, cy = along(zk, 0.0, 0.0)
            lat, lon = ll(cx, cy)
            levels.append({"name": f"VD_{lname}", "level": lname, "zone": zk, "z": elev, "lat": lat, "lon": lon,
                           "fan": fan, "fan2": fan2, "orepass": op, "base_need": round(rng.uniform(16, 24), 1),
                           "leakage": round(rng.uniform(0.14, 0.22), 3),
                           "label": f"{z['label']} {-elev} level", "tag": f"COR.{zk}{-elev}.REG"})
            a0, a1 = along(zk, -length / 2 - 25, 0.0), along(zk, length / 2 + 25, 0.0)
            openings.append({"name": f"DRIVE_{lname}", "kind": "drive", "label": f"{z['label']} {-elev} ore drive",
                             "path": [(a0[0], a0[1], elev), (a1[0], a1[1], elev)]})
            acc = along(zk, 0.0, 120.0)
            openings.append({"name": f"ACCESS_{lname}", "kind": "access", "label": f"{z['label']} {-elev} access",
                             "path": [(acc[0], acc[1], elev + 8), (cx, cy, elev)]})
            for k in range(n_st):
                s = -length / 2 + 45.0 * k
                x, y = along(zk, s, rng.uniform(-6, 6))
                lat, lon = ll(x, y)
                g_bm = z["grade"] * (0.8 + 0.5 * shape) * math.exp(rng.gauss(0, 0.22))
                rate = rng.uniform(150, 320)
                up = 1 if rng.random() < 0.72 else 0
                name = f"ST_{lname}_{k + 1:02d}"
                tram = round(max(0.12, abs(s) / 1000.0 + 0.12 + rng.uniform(0, 0.08)), 3)
                stopes.append({"name": name, "zone": zk, "level": lname, "district": f"VD_{lname}", "orepass": op,
                               "lat": lat, "lon": lon, "z": elev, "grade": round(g_bm, 3), "rate_max": round(rate),
                               "ring_t": round(rng.uniform(7000, 11000), -2), "broken_cap": 60000,
                               "broken": round(rng.uniform(3000, 25000), -2) if up and rng.random() < 0.35 else 0, "up": up,
                               "tram": tram, "label": f"{z['label']} {-elev} stope {k + 1}",
                               "tag": f"COR.{zk}{-elev}.ST{k + 1:02d}"})
                # hidden truth: block-model grade error, productivity (fragmentation) error
                g_true = g_bm * math.exp(rng.gauss(0, 0.12))
                if zk == "S" and elev >= -680:
                    g_true *= 0.78            # the reconciliation surprise: upper South is diluted
                truth.append(f"set corvus_truth.{name}.grade = {g_true:.3f};")
                truth.append(f"set corvus_truth.{name}.rate_max = {rate * math.exp(rng.gauss(0, 0.12)):.0f} [t/h];")
        # dewatering: upper / middle / bottom station per zone
        for j, elev in enumerate([lv[0] - 10, lv[len(lv) // 2] - 10, lv[-1] - 10]):
            x, y = along(zk, -60.0 + 40 * j, 70.0)
            lat, lon = ll(x, y)
            inflow = rng.uniform(18, 55)
            pumps.append({"name": f"PS_{zk}{j + 1}", "zone": zk, "lat": lat, "lon": lon, "z": elev,
                          "inflow": round(inflow, 1), "cap": round(inflow * 2.1), "sump": round(rng.uniform(1800, 3200), -2),
                          "head": round(SURFACE - elev), "label": f"Pump station {z['label']} {-elev}",
                          "tag": f"COR.PS{zk}{j + 1}"})
            truth.append(f"set corvus_truth.PS_{zk}{j + 1}.inflow = {inflow * math.exp(rng.gauss(0, 0.2)):.1f} [L/s];")
    lat, lon = ll(SHAFT[0] - 60, SHAFT[1] + 40)
    inflow = 70.0
    pumps.append({"name": "PS_MAIN", "zone": "H", "lat": lat, "lon": lon, "z": HAULAGE - 20, "inflow": inflow,
                  "cap": 160, "sump": 6000, "head": round(SURFACE - HAULAGE + 20), "label": "Main pump station (haulage)",
                  "tag": "COR.PSMAIN"})
    truth.append(f"set corvus_truth.PS_MAIN.inflow = {inflow * 1.15:.1f} [L/s];")

    # mobile equipment (individual units, for the 3D twin and maintenance)
    kinds = [("LHD", "Loader (LHD)", {"N": 8, "C": 11, "S": 8}, 300), ("JMB", "Development jumbo", {"N": 3, "C": 3, "S": 2}, 110),
             ("LHR", "Longhole drill", {"N": 3, "C": 4, "S": 3}, 90), ("CHG", "Charge-up unit", {"N": 2, "C": 2, "S": 2}, 75),
             ("UTL", "Utility vehicle", {"N": 4, "C": 4, "S": 4}, 120)]
    for code, label, per_zone, kw in kinds:
        n = 0
        for zk, cnt in per_zone.items():
            st = [s for s in stopes if s["zone"] == zk and s["up"]]
            for _ in range(cnt):
                n += 1
                s = rng.choice(st)
                lat, lon = s["lat"], s["lon"]
                equipment.append({"name": f"{code}_{n:02d}", "zone": zk, "lat": lat, "lon": lon, "z": s["z"],
                                  "rated_kw": kw, "label": f"{label} {n:02d}", "tag": f"COR.{code}{n:02d}"})
    for n in range(1, 17):
        x, y = rng.uniform(-200, 500), rng.uniform(-400, 450)
        lat, lon = ll(x, y)
        equipment.append({"name": f"TRK_{n:02d}", "zone": "H", "lat": lat, "lon": lon, "z": HAULAGE, "rated_kw": 520,
                          "label": f"Underground truck {n:02d}", "tag": f"COR.TRK{n:02d}"})

    # main openings: shaft, decline, exhaust raises, haulage drives
    openings.append({"name": "SHAFT_PS", "kind": "shaft", "label": "Production shaft",
                     "path": [(SHAFT[0], SHAFT[1], SURFACE), (SHAFT[0], SHAFT[1], HAULAGE - 40)]})
    openings.append({"name": "SHAFT_SV", "kind": "shaft", "label": "Service shaft (fresh air)",
                     "path": [(SHAFT[0] + 90, SHAFT[1] - 70, SURFACE), (SHAFT[0] + 90, SHAFT[1] - 70, HAULAGE)]})
    for f, (x, y) in FANS.items():
        bottom = {"FAN_N": -760, "FAN_C": -940, "FAN_S": -940}[f]
        openings.append({"name": f"RAISE_{f}", "kind": "raise", "label": f"Exhaust raise {f[-1]}",
                         "path": [(x, y, SURFACE), (x, y, bottom)]})
    pts, zc, ang = [], SURFACE, 0.0
    cx, cy = PORTAL
    while zc > HAULAGE + 20:
        r = 140.0
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang), zc))
        ang += math.radians(30)
        zc -= 2 * math.pi * r / 12 / 7.0          # 30-degree arc (73 m) at a 1:7 gradient
    openings.append({"name": "DECLINE", "kind": "decline", "label": "Main decline", "path": pts})
    for op in orepasses:
        cx, cy = CRUSHERS[op["crusher"]]
        x0, y0 = [(o["path"][0][0], o["path"][0][1]) for o in openings if o["name"] == f"RAISE_{op['name']}"][0]
        openings.append({"name": f"HAUL_{op['name']}", "kind": "haulage", "label": f"Haulage {op['name']} - {op['crusher']}",
                         "path": [(x0, y0, HAULAGE), (cx, cy, HAULAGE)]})

    def write(name, rows):
        with open(os.path.join(HERE, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    write("stopes.csv", stopes)
    write("levels.csv", levels)
    write("orepasses.csv", orepasses)
    write("pumps.csv", pumps)
    write("equipment.csv", equipment)
    feats = []
    for o in openings:
        coords = []
        for x, y, z in o["path"]:
            lat, lon = ll(x, y)
            coords.append([lon, lat, round(z, 1)])
        feats.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                      "properties": {"name": o["name"], "kind": o["kind"], "label": o["label"]}})
    with open(os.path.join(HERE, "openings.geojson"), "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": feats}, fh, indent=0)
    with open(os.path.join(HERE, "truth.plx"), "w", encoding="utf-8") as fh:
        fh.write("## HIDDEN GROUND TRUTH - used only by the plant emulator, never by the twin.\n")
        fh.write("## Generated by generate.py (seed 7). Synthetic data.\nuse mining;\n\n")
        fh.write("## The real mine = the registered mine + a stope whose development has just been completed.\n")
        fh.write("define corvus_truth as corvus {\n  contains:\n")
        x, y = along("C", 170.0, 4.0)
        lat, lon = ll(x, y)
        fh.write(f"    init ST_C560_10 as stope {{ at = ({lat}, {lon}, -560); grade = 1.95; rate_max = 280 [t/h];\n"
                 f"      tram = 0.31 [km]; up = 0; broken = 42000 [t]; tag = \"COR.C560.ST10\"; label = \"Central 560 stope 10\"; }};\n")
        fh.write("    connect(ST_C560_10.ore_out, OP_C1.feed);\n    connect(ST_C560_10.lhd_req, LHD_C.requests);\n"
                 "    connect(ST_C560_10.air_req, VD_C560.demand);\n};\n\n")
        fh.write("\n".join(truth) + "\n")
    print(f"stopes {len(stopes)} (open {sum(s['up'] for s in stopes)}), levels {len(levels)}, orepasses {len(orepasses)}, "
          f"pumps {len(pumps)}, equipment {len(equipment)}, openings {len(openings)}")


if __name__ == "__main__":
    main()
