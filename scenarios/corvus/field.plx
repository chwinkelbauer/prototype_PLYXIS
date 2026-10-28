## =====================================================================================
##  CORVUS - underground copper mine (synthetic demonstration asset, Gawler Craton style)
##  Sublevel open stoping with paste fill, ~12 Mt/a. Three ore lenses (North, Central, South),
##  23 production levels from 520 to 1060 m below surface, 186 stopes, 6 ore-pass systems,
##  haulage level with two crushers, conveyors, a skip-hoist shaft, three exhaust fans,
##  23 ventilation districts, 10 dewatering stations, loader and truck fleets, 79 mobile units,
##  surface ROM pad, concentrator and rail load-out. Operating cost ~ USD 500 M per year.
##
##  Everything underground is 3D: points are (lat, lon, elevation m) and the mine openings come
##  from a 3D GeoJSON export of the mine design (openings.geojson).
## =====================================================================================
use mining;

define corvus as env {
  origin = (-30.950, 136.800);
  properties:
    metal_price  = 9.40 [USD/kg]  input;     ## LME copper, paid metal
    power_price  = 0.12 [USD/kWh] input;     ## NEM South Australia, time-of-use
    grid_co2     = 0.35 [kgCO2e/kWh];
    carbon_price = 45   [USD/tCO2e];          ## internal carbon price
    max_demand   = 72   [MW];                 ## grid connection agreement

  contains:
    # ------------------------------------------------------------- surface
    init SITE as site_services { at = (-30.9470, 136.8050, 140); tag = "COR.SITE"; label = "Site services";
                                 labour = 13500 [USD/h]; overhead = 6000 [USD/h]; };   ## ~1,250 FTE incl. contractors; G&A, camp, FIFO
    init ROM as stockpile { at = (-30.9482, 136.8040, 140); tag = "COR.ROM"; label = "ROM pad";
                            cap = 60000 [t]; tonnes = 18000 [t]; reclaim_max = 2200 [t/h]; grade_avg = 1.40;
                            reserve_t = 12000 [t]; };   ## ~7.5 h of mill feed: bridges a hoist stop
    init MILL as concentrator { at = (-30.9460, 136.8075, 140); tag = "COR.MILL"; label = "Concentrator";
                                cap = 1600 [t/h]; kwh_t = 24 [kWh/t]; cost_t = 8.0 [USD/t];
                                grade_min = 1.35; grade_max = 1.45; conc_grade = 26;
                                grade_penalty = 250 [USD/t]; };   ## mine-plan head grade 1.40 % +- 0.05, no high-grading
    init RAIL as conc_sales { at = (-30.9440, 136.8110, 140); tag = "COR.RAIL"; label = "Rail load-out";
                              conc_grade = 26; };
    connect(ROM.out, MILL.feed);
    connect(MILL.conc, RAIL.rec);

    # ------------------------------------------------------------- shaft, crushers, conveyors
    init SH1 as skip_hoist { at = (-30.9486, 136.8027, -1040); tag = "COR.SH1"; label = "Production shaft hoist";
                             cap = 1750 [t/h]; depth = 1180 [m]; grade_avg = 1.40; };
    connect(SH1.skip, ROM.feed);
    init CR_N as ug_crusher { at = (-30.9466, 136.7987, -1000); tag = "COR.CRN"; label = "Crusher North"; };
    init CR_S as ug_crusher { at = (-30.9530, 136.8044, -1000); tag = "COR.CRS"; label = "Crusher South"; };
    connect(CR_N.product, SH1.feed) via conveyor { cap = 1300 [t/h]; } as CV_N;
    connect(CR_S.product, SH1.feed) via conveyor { cap = 1300 [t/h]; } as CV_S;

    # ------------------------------------------------------------- ore passes and truck haulage
    init TRUCKS as haul_fleet { at = (-30.9490, 136.8010, -1000); tag = "COR.TRUCKS"; label = "Underground trucks";
                                n_trucks = 16; ownership = 110 [USD/h]; };
    for o in table("orepasses.csv") {
      init {o.name} as orepass { at = (o.lat, o.lon, o.z_bottom); tag = o.tag; label = o.label;
                                 cap = o.cap [t]; draw_max = o.draw_max [t/h]; grade_avg = 1.40; };
      connect({o.name}.chute, {o.crusher}.feed) via haul_route { distance = o.haul_km [km];
          payload = 60 [t]; speed_loaded = 14 [km/h]; speed_empty = 20 [km/h]; spot_load = 0.06 [h];
          fuel_cost = 175 [USD/h]; } as HR_{o.name};
      connect(TRUCKS.pool, HR_{o.name}.trucks);
    }
    ## the central ore passes can tip to either crusher
    connect(OP_C1.chute, CR_S.feed) via haul_route { distance = 0.96 [km];
        payload = 60 [t]; speed_loaded = 14 [km/h]; speed_empty = 20 [km/h]; spot_load = 0.06 [h];
        fuel_cost = 175 [USD/h]; } as HR_OP_C1_S;
    connect(OP_C2.chute, CR_N.feed) via haul_route { distance = 0.75 [km];
        payload = 60 [t]; speed_loaded = 14 [km/h]; speed_empty = 20 [km/h]; spot_load = 0.06 [h];
        fuel_cost = 175 [USD/h]; } as HR_OP_C2_N;
    connect(TRUCKS.pool, HR_OP_C1_S.trucks);
    connect(TRUCKS.pool, HR_OP_C2_N.trucks);

    # ------------------------------------------------------------- ventilation
    init FAN_N as main_fan { at = (-30.9406, 136.7898, 140); tag = "COR.FANN"; label = "Main fan North (VR1)"; };
    init FAN_C as main_fan { at = (-30.9477, 136.7967, 140); tag = "COR.FANC"; label = "Main fan Central (VR2)";
                             q_max = 480 [m3/s]; kw_max = 3900 [kW]; };
    init FAN_S as main_fan { at = (-30.9603, 136.8102, 140); tag = "COR.FANS"; label = "Main fan South (VR3)"; };
    for l in table("levels.csv") {
      init {l.name} as vent_district { at = (l.lat, l.lon, l.z); tag = l.tag; label = l.label;
                                      base_need = l.base_need [m3/s]; leakage = l.leakage; };
      connect({l.name}.from_a, {l.fan}.requests);
      connect({l.name}.from_b, {l.fan2}.requests);
    }

    # ------------------------------------------------------------- loaders and stopes
    init LHD_N as lhd_fleet { at = (-30.9425, 136.7930, -560); tag = "COR.LHDN"; label = "Loader fleet North"; n_units = 8; };
    init LHD_C as lhd_fleet { at = (-30.9500, 136.8000, -680); tag = "COR.LHDC"; label = "Loader fleet Central"; n_units = 11; };
    init LHD_S as lhd_fleet { at = (-30.9572, 136.8065, -740); tag = "COR.LHDS"; label = "Loader fleet South"; n_units = 8; };
    for s in table("stopes.csv") {
      init {s.name} as stope { at = (s.lat, s.lon, s.z); tag = s.tag; label = s.label; grade = s.grade;
                               rate_max = s.rate_max [t/h]; ring_t = s.ring_t [t]; broken = s.broken [t]; cost_t = 10.0 [USD/t];
                               tram = s.tram [km]; up = s.up; };
      connect({s.name}.ore_out, {s.orepass}.feed);
      connect({s.name}.lhd_req, LHD_{s.zone}.requests);
      connect({s.name}.air_req, {s.district}.demand);
    }

    # ------------------------------------------------------------- dewatering
    for p in table("pumps.csv") {
      init {p.name} as pump_station { at = (p.lat, p.lon, p.z); tag = p.tag; label = p.label;
                                      inflow = p.inflow [L/s]; cap = p.cap [L/s]; sump = p.sump [m3]; head = p.head [m];
                                      reserve_h = 8 [h]; };   ## a shift of storage for a pump failure
    }

    # ------------------------------------------------------------- mobile equipment and openings (3D)
    for e in table("equipment.csv") {
      init {e.name} as mobile_unit { at = (e.lat, e.lon, e.z); tag = e.tag; label = e.label; rated_kw = e.rated_kw [kW]; };
    }
    for g in geojson("openings.geojson") {
      init {g.name} as opening { path = g.path; label = g.label; layer = g.kind; };
    }

  behavior:
    site_kw = sum(main_fan.kw) + sum(skip_hoist.kw) + sum(pump_station.kw) + sum(concentrator.kw)
              + sum(ug_crusher.kw) + sum(conveyor.kw) + sum(site_services.kw);
    ## grid connection limit: above it the network operator trips the mine (priced as a penalty for planning)
    constraint soft(2 [USD/kWh]) grid_limit: site_kw <= max_demand;
    metric site_power = site_kw [MW];
    metric ore_mined = sum(stope.mucked) [t/h];

  discover:
    ## telemetry of a stope that is not in the register = development completed, stope handed to production
    rule new_stope: on "COR.{zone}{lvl}.ST{num}.LHD.TPH" {
      init ST_{zone}{lvl}_{num} as stope { tag = "COR.{zone}{lvl}.ST{num}"; label = "Stope {zone}{lvl}-{num}"; };
      connect(ST_{zone}{lvl}_{num}.ore_out, OP_{zone}1.feed);
      connect(ST_{zone}{lvl}_{num}.lhd_req, LHD_{zone}.requests);
      connect(ST_{zone}{lvl}_{num}.air_req, VD_{zone}{lvl}.demand);
    };
};

## Short-interval control: 48 h rolling horizon at hourly resolution.
define shift48 as sim {
  env: corvus;
  horizon = 48 [h];
  step = 1 [h];
  objective:
    maximize sum(revenue) - sum(cost) - carbon_price * sum(emission);
  solver:
    partition = "level";
    route_hold = 4;
};
