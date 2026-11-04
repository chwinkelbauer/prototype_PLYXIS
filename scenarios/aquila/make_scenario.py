"""Write scenario.json for Aquila Steel (run after generate.py).

Times are UTC (start_hour 0). Every power zone has its own daily price shape in its local time, shifted here to
UTC, so a European evening peak and a Texan afternoon peak fall on different plant hours."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

UTC = {"nl": 1, "de": 1, "fr": 1, "es": 1, "pl": 1, "miso": -6, "ercot": -6, "mx": -6, "br": -3, "in": 5, "sa": 3}
POWER = {"nl": 85, "de": 90, "fr": 70, "es": 65, "pl": 100, "miso": 40, "ercot": 35, "mx": 60, "br": 55, "in": 70, "sa": 45}
# daily shapes in local time (hour 0-23), normalised to a mean of 1 below
SHAPE = {
    "eu": [0.82, 0.78, 0.75, 0.74, 0.76, 0.85, 1.05, 1.30, 1.32, 1.10, 0.85, 0.70, 0.62, 0.60, 0.65, 0.80, 1.05, 1.45,
           1.65, 1.55, 1.30, 1.10, 0.98, 0.90],                      # morning and evening peaks, solar trough
    "us": [0.70, 0.66, 0.64, 0.63, 0.65, 0.72, 0.85, 0.95, 0.95, 0.92, 0.90, 0.92, 0.98, 1.08, 1.25, 1.50, 1.85, 2.10,
           1.90, 1.45, 1.15, 0.95, 0.82, 0.75],                      # afternoon air-conditioning peak
    "mx": [0.80, 0.76, 0.74, 0.73, 0.75, 0.80, 0.90, 0.98, 1.00, 1.00, 1.00, 1.02, 1.05, 1.08, 1.10, 1.12, 1.18, 1.30,
           1.55, 1.70, 1.55, 1.25, 1.00, 0.88],
    "br": [0.85, 0.80, 0.78, 0.78, 0.80, 0.85, 0.92, 1.00, 1.05, 1.05, 1.02, 0.95, 0.90, 0.90, 0.95, 1.02, 1.15, 1.40,
           1.55, 1.45, 1.25, 1.08, 0.95, 0.90],
    "in": [0.90, 0.86, 0.84, 0.84, 0.86, 0.92, 1.00, 1.05, 1.00, 0.85, 0.70, 0.62, 0.60, 0.62, 0.70, 0.85, 1.05, 1.35,
           1.60, 1.65, 1.50, 1.30, 1.10, 0.98],                      # solar trough, evening peak
    "sa": [0.80, 0.76, 0.74, 0.74, 0.76, 0.80, 0.86, 0.92, 1.00, 1.10, 1.22, 1.35, 1.48, 1.58, 1.62, 1.55, 1.40, 1.25,
           1.15, 1.05, 0.98, 0.92, 0.88, 0.84],
}
SHAPE_OF = {"nl": "eu", "de": "eu", "fr": "eu", "es": "eu", "pl": "eu", "miso": "us", "ercot": "us", "mx": "mx",
            "br": "br", "in": "in", "sa": "sa"}


def utc_profile(zone: str) -> list:
    s = SHAPE[SHAPE_OF[zone]]
    m = sum(s) / 24.0
    return [round(s[(h + UTC[zone]) % 24] / m, 3) for h in range(24)]


prices = {}
for z, base in POWER.items():
    prices[f"power_{z}"] = {"base": base, "unit": "USD/MWh", "daily": utc_profile(z), "walk": 0.12,
                            "spike_prob": 0.01 if z == "ercot" else 0.004}
for name, base, walk in (("hrc_eu", 680, 2.0), ("hrc_na", 820, 2.5), ("hrc_sa", 690, 2.0), ("hrc_in", 600, 2.0),
                         ("hrc_me", 640, 2.0), ("scrap_eu", 395, 1.5), ("scrap_us", 405, 1.5), ("scrap_br", 360, 1.2),
                         ("scrap_in", 425, 1.5), ("scrap_me", 395, 1.5), ("hbi_price", 360, 1.2)):
    prices[name] = {"base": base, "unit": "USD/t", "walk": walk}
for name, base, walk in (("gas_ttf", 10.5, 0.06), ("gas_hh", 2.6, 0.02), ("gas_br", 9.0, 0.04), ("gas_in", 11.5, 0.05),
                         ("gas_sa", 2.0, 0.0)):
    prices[name] = {"base": base, "unit": "USD/GJ", "walk": walk}
prices["eua_price"] = {"base": 72, "unit": "USD/tCO2e", "walk": 0.3}

tags = {k: "MKT." + k.upper().replace("_", ".") for k in prices}

# published time-of-use peak windows of each zone (UTC hours): the four dearest hours of its daily shape. A
# conventional melt shop knows its tariff and keeps its furnaces out of these hours where it can.
peak_hours = {z: sorted(sorted(range(24), key=lambda h: -prices[f"power_{z}"]["daily"][h])[:4]) for z in POWER}

scenario = {
    "name": "Aquila Steel - autonomous planning for a global steel producer",
    "title": "Aquila Steel",
    "region": "13 sites, 4 continents",
    "subtitle": "synthetic global steel producer · 13 sites · 15 blast furnaces · 10 arc furnaces · 73 Mt/a capacity",
    "description": "Synthetic global steel group: seven integrated sites (blast furnace -> converter -> caster -> strip "
                   "mill and galvanising), six electric sites (direct reduction and scrap -> arc furnace -> caster -> "
                   "strip or bar mill), ore, coke, scrap and HBI supply, eleven power zones in their local time, five "
                   "gas hubs, the EU ETS, twelve regional markets and 29 sales and transfer lanes. Two identical copies "
                   "of the group run side by side from the same state with the same events: one planned by PLYXIS, "
                   "one by conventional corporate S&OP with site control rooms.",
    "twin": {"files": ["field.plx"], "env": "aquila", "sim": "corp96"},
    "truth": {"files": ["field.plx", "truth.plx"], "env": "aquila_truth", "hidden": []},
    "time": {"dt_emulator_h": 1.0 / 6.0, "replan_every_h": 4.0, "calibrate_every_h": 1.0,
             "start_hour": 0.0, "end_hour": 96.0},
    "prices": prices,
    "market_tags": tags,
    # the historian's last week of market prices: the first plan knows each zone's daily shape (both control rooms do -
    # the conventional one through its published time-of-use windows)
    "price_history_days": 7,
    "market_desc": "regional coil indices, scrap and HBI, day-ahead power per zone, gas hubs, EUA (synthetic)",
    "kpi": {
        "metrics": [
            {"key": "shipped", "metric": "shipments", "unit": "t", "rate_unit": "t/h", "label": "Steel shipped"},
            {"key": "crude", "metric": "crude_steel", "unit": "t", "rate_unit": "t/h", "label": "Crude steel"},
            {"key": "iron", "metric": "hot_metal_made", "unit": "t", "rate_unit": "t/h", "label": "Hot metal"},
            {"key": "eaf", "metric": "eaf_steel", "unit": "t", "rate_unit": "t/h", "label": "Arc furnace steel"},
            {"key": "power", "metric": "group_power", "unit": "MWh", "rate_unit": "MW", "label": "Group power",
             "better": "lower"},
        ],
        "primary": "shipped", "product": "crude", "secondary": "power", "plan_series": "shipments",
        "unit_cost_label": "Cost per tonne shipped",
    },
    "events": [
        {"t": 5.0, "kind": "fail", "asset": "DE_BF1", "hours": 30.0, "title": "Duisburg blast furnace 1: cooling stave leak",
         "restore_title": "Duisburg blast furnace 1 back on wind",
         "narrative": "Water in the furnace: the largest furnace in Europe goes off wind for about thirty hours."},
        {"t": 18.0, "kind": "capacity", "asset": "NL_ORE", "param": "delivery", "factor": 0.2, "hours": 36.0, "notice": True,
         "title": "Storm closes IJmuiden: no vessels for 36 hours",
         "restore_title": "IJmuiden open again, vessels discharging",
         "narrative": "A North Sea storm closes the port. Ore and pellet deliveries to Velsen fall to a fifth."},
        {"t": 20.0, "kind": "announce", "name": "power_ercot", "start": 45.0, "end": 49.0, "value": 1500.0,
         "title": "ERCOT: scarcity prices forecast tomorrow 15:00-19:00 CST",
         "narrative": "A Texas heat wave: the grid operator expects 1,500 USD/MWh for four hours tomorrow afternoon."},
        {"t": 26.0, "kind": "capacity", "asset": "GJ_GAS", "param": "cap", "factor": 0.45, "hours": 30.0, "notice": True,
         "title": "Hazira: gas curtailed to 45 % for pipeline maintenance",
         "restore_title": "Hazira gas supply restored",
         "narrative": "The pipeline operator curtails the site's gas for thirty hours. Direct reduction needs most of it."},
        {"t": 34.0, "kind": "fail", "asset": "BR_HSM", "hours": 12.0, "title": "Tubarao hot strip mill: main drive fault",
         "restore_title": "Tubarao hot strip mill rolling again",
         "narrative": "A drive motor fails. Twelve hours without rolling; the slabs keep coming from the casters."},
        {"t": 45.0, "kind": "price", "name": "power_ercot", "value": 1500.0, "hours": 4.0,
         "title": "ERCOT at 1,500 USD/MWh",
         "narrative": "An arc furnace hour in Corpus Christi now costs more than the steel it melts is worth."},
        {"t": 50.0, "kind": "capacity", "asset": "NA_COAT", "param": "contract", "factor": 1.25, "hours": 40.0, "notice": True,
         "title": "North American automotive customers pull orders forward",
         "restore_title": "Coated call-offs back to plan",
         "narrative": "Two car makers call off a quarter more coated coil for the next forty hours."},
        {"t": 60.0, "kind": "price", "name": "scrap_eu", "value": 470.0, "hours": 36.0,
         "title": "European scrap jumps to 470 USD/t",
         "narrative": "Export demand empties the dealers' yards. Shredded scrap rises by a fifth overnight."},
        {"t": 72.0, "kind": "fail", "asset": "MX_EAF1", "hours": 8.0, "title": "Monterrey arc furnace: transformer trip",
         "restore_title": "Monterrey arc furnace melting again",
         "narrative": "The furnace transformer trips on a fault. Eight hours without melting."},
    ],
    # the lab analyses composite scrap samples of every bay about every six hours (8 % relative error), the
    # same sampling in both worlds
    "assays": {"policy": "working", "signal": "YARD.STOCK", "per_hour": 3, "sigma": 0.08},
    # a relaxation that ended at violation 0.1 lost its way (one seed's first plan shipped 5,983 t/h against 6,250 t/h of
    # contracts); counting iterates up to 0.01 as feasible, as on the mine, keeps the better one. The first plan starts
    # cold and has nothing to be compared with: it is solved from two random streams and the better is kept
    "planner": {"relax_tol": 0.01, "cold_starts": 2},
    # no exact-branch stage: on the allocation layer (mills, lanes, DRI plants, furnaces, scrap; 1,228 variables) the
    # sequential LP found nothing to improve after the polish (bench.policy_in_twin), and it costs ~36 s per plan
    # the corporate's internal carbon price is in the S&OP's standard costs as in the plan's objective (field.plx)
    "legacy": {"utilisation": 0.97, "peak_hours": peak_hours, "peak_factor": 0.3, "dr_price": 300.0,
               "cu_margin": 0.9, "carbon_price": 30.0},
    "demo": [
        {"id": "intro", "t": 0.0, "title": "One plan for a whole steel group", "view": "overview",
         "text": "Aquila: 13 sites on four continents, 15 blast furnaces, 10 arc furnaces, 73 Mt of crude steel capacity a year. "
                 "The same group also runs under conventional S&OP and site control rooms in parallel - same state, "
                 "same events, same physics."},
        {"id": "map", "t": 1.0, "title": "The whole value chain in one twin", "view": "map",
         "text": "Ore and coke yards, scrap bays, furnaces, casters, mills, the lanes to twelve regional markets and "
                 "the transfers between sites - every asset of the group, compiled from PLX code and the asset register."},
        {"id": "bf", "t": 5.2, "title": "Blast furnace off wind", "view": "overview",
         "text": "Duisburg loses its largest furnace for about 30 hours; both planning departments know when it is due "
                 "back. PLYXIS re-plans the whole group at once - iron, slabs (Velsen and Dunkirk can ship slabs to "
                 "Duisburg's strip mill) and markets. The Schedule shows what moved and why."},
        {"id": "explain", "t": 8.0, "title": "Why this site and not that one?", "view": "schedule",
         "text": "Every decision carries its reasons: delivered margin per market, the shadow price of iron, of rolling "
                 "capacity and of each lane - computed from the model's gradients."},
        {"id": "storm", "t": 18.2, "title": "A port closes", "view": "map",
         "text": "No vessels at IJmuiden for 36 hours. The ore yard holds weeks of burden: nothing to do but watch it - "
                 "the twin knows that and does not over-react."},
        {"id": "ercot", "t": 20.2, "title": "A price signal a day ahead", "view": "overview",
         "text": "ERCOT forecasts 1,500 USD/MWh tomorrow afternoon, and both worlds know it a day ahead. The conventional "
                 "control room will stop the Texas arc furnace for the window (demand response); PLYXIS plans the "
                 "furnace's heats and the billet yard around it."},
        {"id": "gas", "t": 26.2, "title": "Gas curtailed in India", "view": "schedule",
         "text": "Hazira's direct reduction loses half its gas. Its arc furnaces need more scrap - within the copper limit "
                 "of their coil grades, measured by the lab - and merchant HBI where it pays."},
        {"id": "drift", "t": 30.0, "title": "Reality drifts - the twin follows", "view": "data",
         "text": "Furnaces use more coke and more power than the engineering data say, scrap carries more copper than "
                 "the specs. The estimator re-calibrates every hour from the plant's own measurements."},
        {"id": "demand", "t": 50.2, "title": "Customers pull orders forward", "view": "overview",
         "text": "North American car makers want a quarter more coated coil for 40 hours. Which site supplies it, and "
                 "what does it displace? PLYXIS answers that for the whole group in the next plan."},
        {"id": "scrap", "t": 60.2, "title": "Scrap jumps 19 %", "view": "overview",
         "text": "European scrap costs 470 USD/t. Each site's metallic mix - hot metal, scrap, HBI - is re-planned "
                 "against the new price, within its heat and copper balances."},
        {"id": "value", "t": 95.5, "title": "Value", "view": "overview",
         "text": "Same group, same events. PLYXIS plans to the group's objective, profit after its internal carbon "
                 "price, and runs Europe on more scrap and less hot metal: less CO2e and a better result after the "
                 "carbon price, at an operating profit within a few per cent of a competent S&OP's."},
    ],
}
with open(os.path.join(HERE, "scenario.json"), "w", encoding="utf-8") as fh:
    json.dump(scenario, fh, indent=2)
print("prices", len(prices), "events", len(scenario["events"]), "peak hours", peak_hours)
