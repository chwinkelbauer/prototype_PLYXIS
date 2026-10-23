## =====================================================================================
##  KESTREL FIELD - digital twin (synthetic demonstration asset, Delaware Basin style)
##  96 ESP wells on 12 pads, 3 tank batteries, 2 compressor stations, gas plant,
##  LACT terminal, 3 salt-water disposal wells, trucking fleet, rigs and crews.
##
##  The whole site is ~120 lines: blueprints come from 'oilgas', the asset register
##  (wells.csv / pads.csv) instantiates them. Adding a well = adding a CSV row.
## =====================================================================================
use oilgas;

define kestrel as env {
  origin = (31.61, -103.68);
  properties:
    oil_price    = 72.0  [USD/bbl]  input;     ## WTI + quality / transport netback
    gas_price    = 1.20  [USD/Mscf] input;     ## Waha hub (can turn negative)
    power_price  = 0.055 [USD/kWh]  input;     ## ERCOT West load zone
    grid_co2     = 0.38  [kgCO2e/kWh];
    carbon_price = 60    [USD/tCO2e];          ## internal carbon price (EU-ETS level)
    flare_permit = 1500  [Mscf/d];             ## field-wide flaring allowance

  contains:
    # ---------------------------------------------------------- tank batteries
    init CTB_N as battery { at = (31.6750, -103.7000); tag = "KST.CTBN"; label = "CTB North";
                            liq_cap = 24000 [bbl/d]; gas_cap = 17000 [Mscf/d]; };
    init CTB_C as battery { at = (31.6100, -103.6650); tag = "KST.CTBC"; label = "CTB Central";
                            liq_cap = 30000 [bbl/d]; gas_cap = 19000 [Mscf/d]; };
    init CTB_S as battery { at = (31.5450, -103.7050); tag = "KST.CTBS"; label = "CTB South";
                            liq_cap = 20000 [bbl/d]; gas_cap = 16000 [Mscf/d]; oil_tank = 6000 [bbl]; };

    # ---------------------------------------------------------- pads and wells (asset register)
    for p in table("pads.csv") {
      init {p.name} as pad { at = (p.lat, p.lon); tag = p.tag; label = p.label; };
      connect({p.name}.header, {p.home}.feed) via flowline { cap_liquid = 14000 [bbl/d]; };
      connect({p.name}.header, {p.alt}.feed)  via flowline { cap_liquid = 9000 [bbl/d]; };
    }
    for w in table("wells.csv") {
      init {w.name} as esp_well { q_max = w.q_max; wc = w.wc; gor = w.gor; kw_rated = w.kw_rated;
                                  health = w.health; at = (w.lat, w.lon); tag = w.tag; label = w.label; };
      connect({w.name}.prod, {w.pad}.wells);
    }

    # ---------------------------------------------------------- gas: compression, flares, plant
    init CS1 as compressor_station { at = (31.6500, -103.6250); tag = "KST.CS1"; label = "CS-1";
                                     n_units = 3; unit_cap = 9000 [Mscf/d]; };
    init CS2 as compressor_station { at = (31.5700, -103.6300); tag = "KST.CS2"; label = "CS-2";
                                     n_units = 2; unit_cap = 8500 [Mscf/d]; unit_kw = 1000 [kW]; };
    init FL1 as flare { at = (31.6510, -103.6230); tag = "KST.FL1"; label = "Flare CS-1"; };
    init FL2 as flare { at = (31.5710, -103.6280); tag = "KST.FL2"; label = "Flare CS-2"; };
    init GASPLANT as gas_sales { at = (31.6000, -103.5600); tag = "KST.GP"; label = "Gas plant";
                                 cap = 42000 [Mscf/d]; };
    connect(CTB_N.gas_out, CS1.suction) via flowline;
    connect(CTB_C.gas_out, CS1.suction) via flowline;
    connect(CTB_C.gas_out, CS2.suction) via flowline;
    connect(CTB_S.gas_out, CS2.suction) via flowline;
    connect(CS1.discharge, GASPLANT.rec) via flowline;
    connect(CS2.discharge, GASPLANT.rec) via flowline;
    connect(CS1.flare, FL1.burn);
    connect(CS2.flare, FL2.burn);

    # ---------------------------------------------------------- oil: pipelines and trucking
    init LACT as oil_sales { at = (31.6250, -103.5750); tag = "KST.LACT"; label = "LACT terminal"; };
    init FLEET as truck_fleet { at = (31.6000, -103.6000); tag = "KST.FLEET"; label = "Truck yard";
                                n_trucks = 8; };
    connect(CTB_N.oil_out, LACT.rec) via flowline { cap_liquid = 9000 [bbl/d]; };
    connect(CTB_C.oil_out, LACT.rec) via flowline { cap_liquid = 5200 [bbl/d]; };
    connect(CTB_C.oil_out, LACT.rec) via truck_lane { distance = 16 [km]; } as LANE_C_OIL;
    connect(CTB_S.oil_out, LACT.rec) via truck_lane { distance = 24 [km]; } as LANE_S_OIL;

    # ---------------------------------------------------------- produced water disposal
    init SWD1 as swd_well { at = (31.6900, -103.7600); tag = "KST.SWD1"; label = "SWD-1"; inj_cap = 22000 [bbl/d]; };
    init SWD2 as swd_well { at = (31.6050, -103.7700); tag = "KST.SWD2"; label = "SWD-2"; inj_cap = 24000 [bbl/d]; };
    init SWD3 as swd_well { at = (31.5300, -103.6400); tag = "KST.SWD3"; label = "SWD-3"; inj_cap = 16000 [bbl/d]; };
    connect(CTB_N.wat_out, SWD1.inj) via flowline;
    connect(CTB_N.wat_out, SWD2.inj) via flowline;
    connect(CTB_C.wat_out, SWD2.inj) via flowline;
    connect(CTB_C.wat_out, SWD3.inj) via flowline;
    connect(CTB_S.wat_out, SWD3.inj) via flowline;
    connect(CTB_S.wat_out, SWD1.inj) via truck_lane { distance = 21 [km]; cost_h = 85 [USD/h]; } as LANE_S_WTR;

    # every lane draws trucks from the shared fleet (allocation is a decision)
    connect(FLEET.pool, LANE_C_OIL.fleet);
    connect(FLEET.pool, LANE_S_OIL.fleet);
    connect(FLEET.pool, LANE_S_WTR.fleet);

    # ---------------------------------------------------------- field resources
    init RIG1 as workover_rig { at = (31.6400, -103.7300); tag = "KST.RIG1"; label = "Workover rig 1"; };
    init RIG2 as workover_rig { at = (31.5600, -103.6600); tag = "KST.RIG2"; label = "Workover rig 2"; };
    for c in 1..4 {
      init CREW{c} as field_crew { at = (31.6000 + 0.002 * c, -103.6000); tag = concat("KST.CREW", c); label = concat("Lease crew ", c); };
    }

  behavior:
    ## the permit is a license-to-operate limit, not a price: 500 USD/Mscf makes it binding for planning
    constraint soft(500 [USD/Mscf]) flare_permit: sum(flare.burned) <= flare_permit;
    ## zero-routine-flaring policy: flaring is for safety and upsets, never an economic choice -
    ## 150 USD/Mscf makes the planner start a compressor or choke the gassiest wells before it flares
    constraint soft(150 [USD/Mscf]) routine_flaring: sum(flare.burned) <= 0 [Mscf/d];
    metric field_oil = sum(esp_well.prod.oil) [bbl/d];
    metric field_gas = sum(esp_well.prod.gas) [Mscf/d];
    metric field_water = sum(esp_well.prod.water) [bbl/d];

  discover:
    ## a flowmeter tag from an unknown well on a known pad = a new well being tied in
    rule new_well: on "KST.P{pad}.W{well}.FLOW.LIQ" {
      init P{pad}_W{well} as esp_well { tag = "KST.P{pad}.W{well}"; label = "PAD-{pad} W-{well}"; };
      connect(P{pad}_W{well}.prod, PAD{pad}.wells);
    };
};

## Operational planning problem: 72 h rolling horizon at hourly resolution.
define ops72 as sim {
  env: kestrel;
  horizon = 72 [h];
  step = 1 [h];
  objective:
    maximize sum(revenue) - sum(cost) - carbon_price * sum(emission);
  solver:
    partition = "pad";
    route_hold = 24;
};
