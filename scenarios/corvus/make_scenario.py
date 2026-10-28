"""Write scenario.json for the Corvus mine (run after generate.py)."""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "stopes.csv"), encoding="utf-8") as fh:
    stopes = list(csv.DictReader(fh))
closed = [s["name"] for s in stopes if s["level"] == "N560"]

scenario = {
    "name": "Corvus Mine - autonomous underground operations showcase",
    "title": "Corvus Mine",
    "region": "Gawler Craton",
    "subtitle": "synthetic underground copper mine · 186 stopes · 23 levels · 12 Mt/a · OPEX ~USD 500 M/a",
    "description": "Synthetic sublevel-stoping copper mine (3 ore lenses, 23 production levels 520-1060 m below "
                   "surface, 186 stopes, 6 ore-pass systems, 2 crushers, skip hoist, 3 main fans, 10 pump stations, "
                   "27 loaders, 16 trucks, concentrator). Two identical copies of the mine run side by side from the "
                   "same state with the same events: one operated by PLYXIS, one by conventional mine control.",
    "twin": {"files": ["field.plx"], "env": "corvus", "sim": "shift48"},
    "truth": {"files": ["field.plx", "truth.plx"], "env": "corvus_truth", "hidden": ["ST_C560_10"]},
    "time": {"dt_emulator_h": 1.0 / 6.0, "replan_every_h": 3.0, "calibrate_every_h": 1.0,
             "start_hour": 6.0, "end_hour": 48.0},
    "prices": {
        "metal_price": {"base": 9.4, "unit": "USD/kg", "walk": 0.03},
        "power_price": {"base": 0.11, "unit": "USD/kWh",
                        "daily": [0.95, 0.9, 0.85, 0.85, 0.9, 1.05, 1.35, 1.45, 1.2, 0.75, 0.5, 0.4, 0.35, 0.35,
                                  0.42, 0.6, 1.1, 2.0, 2.6, 2.3, 1.6, 1.25, 1.1, 1.0], "spike_prob": 0.01},
    },
    "market_tags": {"metal_price": "MKT.LME.CU", "power_price": "MKT.NEM.SA1"},
    "market_desc": "LME copper, NEM South Australia (synthetic)",
    "kpi": {
        "metrics": [
            {"key": "ore", "metric": "ore_mucked", "unit": "t", "rate_unit": "t/h", "label": "Ore mined"},
            {"key": "hoisted", "metric": "ore_hoisted", "unit": "t", "rate_unit": "t/h", "label": "Ore hoisted"},
            {"key": "milled", "metric": "milled", "unit": "t", "rate_unit": "t/h", "label": "Ore milled"},
            {"key": "copper", "metric": "metal_sold", "unit": "t", "rate_unit": "t/d", "label": "Copper in concentrate"},
            {"key": "power", "metric": "site_power", "unit": "MWh", "rate_unit": "MW", "label": "Site power",
             "better": "lower"},
        ],
        "primary": "milled", "product": "copper", "secondary": "power", "plan_series": "milled",
        "unit_cost_label": "Cost per tonne milled",
    },
    "events": [
        {"t": 4.0, "kind": "fail", "asset": "FAN_C", "hours": 6.0, "title": "Main fan VR2 trips",
         "restore_title": "Main fan VR2 back in service",
         "narrative": "Motor protection trips the central exhaust fan. A third of the mine's air is gone for six hours."},
        {"t": 9.0, "kind": "fail", "assets": closed, "hours": 20.0, "title": "Seismic event closes North 560 level",
         "restore_title": "North 560 level re-opened after inspection",
         "narrative": "A magnitude 1.6 event on North 560. The level is barricaded until the geotechnical inspection."},
        {"t": 15.0, "kind": "fail", "asset": "SH1", "hours": 5.0, "title": "Skip hoist stopped: rope inspection finding",
         "restore_title": "Skip hoist back in service",
         "narrative": "The rope inspection finds broken wires. Five hours without hoisting - every bin and ore pass now matters."},
        {"t": 21.0, "kind": "announce", "name": "power_price", "start": 34.0, "end": 37.0, "value": 1.5,
         "title": "NEM forecast: critical peak tomorrow 16:00-19:00",
         "narrative": "Heat wave in South Australia: the market operator forecasts 1,500 USD/MWh for three hours tomorrow evening."},
        {"t": 26.0, "kind": "trip", "asset": "LHD_C", "units": 3, "hours": 10.0, "title": "Three Central loaders down",
         "restore_title": "Central loaders back from the workshop",
         "narrative": "Two transmission failures and a tyre change: the Central zone loses three of eleven loaders for ten hours."},
        {"t": 30.0, "kind": "commission", "asset": "ST_C560_10", "title": "New stope handed over on Central 560",
         "narrative": "Development completed a new stope. It is not in the mine plan database yet - only its tags appear."},
        {"t": 34.0, "kind": "price", "name": "power_price", "value": 1.5, "hours": 3.0,
         "title": "Power at 1,500 USD/MWh", "narrative": "Every fan, pump and the hoist now costs twelve times more per hour."},
        {"t": 40.0, "kind": "fail", "asset": "PS_S3", "hours": 8.0, "title": "Pump station South 930 fails",
         "restore_title": "Pump station South 930 repaired",
         "narrative": "A pump seal fails. The sump keeps filling - nobody announced it, the twin sees the level rise."},
    ],
    "planner": {"relax_iters": 120, "multistart": 4, "population": 8, "repair_iters": 60, "polish_iters": 80,
                "discrete": "ga+sections"},
    "legacy": {"places_per_loader": 1.3, "utilisation": 0.85, "blast_below": 0.15, "blasts_per_zone": 2, "broken_days": 5, "plan_grade": 1.40, "grade_tol": 0.05,
               "fan_speed": 0.9, "mill_tph": 1580, "dr_price": 1.0},
    "demo": [
        {"id": "intro", "t": 0.0, "title": "One operating system for the whole mine", "view": "overview",
         "text": "Corvus: 186 stopes on 23 levels, 1 km deep, 12 Mt per year, USD 500 M operating cost. The same mine "
                 "also runs under conventional mine control in parallel - same state, same events, same physics."},
        {"id": "mine3d", "t": 1.0, "title": "The twin is three-dimensional", "view": "mine3d",
         "text": "Every stope, ore pass, fan and pump sits at its true position and depth. The openings come straight "
                 "from a 3D GeoJSON export of the mine design; the twin is compiled from 135 lines of PLX code plus the asset register."},
        {"id": "fan", "t": 4.2, "title": "Main fan trip: ventilation on demand", "view": "mine3d",
         "text": "The central fan trips. Air is the binding constraint for every diesel loader. PLYXIS re-plans in "
                 "seconds: the remaining fans speed up, regulators send air to the stopes that are worth most, loaders "
                 "move. Conventional control withdraws equipment from the whole affected area."},
        {"id": "explain", "t": 6.0, "title": "Why this stope and not that one?", "view": "schedule",
         "text": "Every decision carries its reasons: grade, tramming distance, loader hours, the shadow price of air and "
                 "hoisting capacity - computed from the model's gradients."},
        {"id": "seismic", "t": 9.2, "title": "Seismic event closes a level", "view": "mine3d",
         "text": "North 560 is barricaded. Its loaders and air go to other levels within one plan; blasting is "
                 "re-sequenced so broken stocks do not run out elsewhere."},
        {"id": "hoist", "t": 15.3, "title": "Hoist stopped: every bin matters", "view": "overview",
         "text": "Five hours without hoisting. PLYXIS had kept ore passes and the shaft bin low enough to keep mucking "
                 "into them, and the mill keeps running from the ROM pad. Conventional control fills the passes and stops."},
        {"id": "grade", "t": 20.0, "title": "Reality drifts - the twin follows", "view": "data",
         "text": "Grab-sample assays and the mill's on-stream analyser show upper South grading below the block model. "
                 "The estimator re-calibrates the stope grades and the planner blends to the head-grade window again."},
        {"id": "power", "t": 21.2, "title": "Price signal a day ahead", "view": "overview",
         "text": "The market operator forecasts 1,500 USD/MWh tomorrow evening. PLYXIS pre-pumps the sumps at midday solar "
                 "prices, draws the shaft bin down and slows the fans where few loaders work."},
        {"id": "loaders", "t": 26.2, "title": "Three loaders down", "view": "schedule",
         "text": "Central loses three loaders. PLYXIS re-assigns the rest to the stopes with the best value per loader "
                 "hour, not the biggest broken stock."},
        {"id": "tiein", "t": 30.5, "title": "The twin rewrites itself", "view": "studio",
         "text": "Tags of an unknown stope appeared. A discovery rule turned them into PLX code - reviewed, committed and "
                 "compiled into the live twin in seconds."},
        {"id": "peak", "t": 34.3, "title": "Critical peak: 1,500 USD/MWh", "view": "overview",
         "text": "Pumps, hoist and fans run at their minimum for three hours; the mill keeps running. Conventional control "
                 "defers hoisting and pumping too - the difference is what was prepared yesterday."},
        {"id": "value", "t": 47.5, "title": "Value", "view": "overview",
         "text": "Same mine, same events: the difference in tonnes, metal, cost per tonne and energy is the value of "
                 "autonomous operations."},
    ],
}
with open(os.path.join(HERE, "scenario.json"), "w", encoding="utf-8") as fh:
    json.dump(scenario, fh, indent=2)
print("events", len(scenario["events"]), "closed stopes", len(closed))
