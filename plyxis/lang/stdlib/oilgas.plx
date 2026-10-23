## PLYXIS standard library - upstream & midstream oil and gas.
##
## Nothing in here is special to the engine: every asset is ordinary PLX code.
## Copy an entity, change a line, and the digital twin follows.
##
## Conventions
##   * flows on ports are rates (internally per hour); declare literals with units
##   * cost / revenue / emission terms are rates per hour - the objective integrates them
##   * 'calibrate in [lo, hi]' marks properties that the data-integration layer
##     estimates online from telemetry (moving-horizon estimation)
##   * sat(rate, available) saturates in the plant emulator; planners keep it feasible

define oilgas as pkg {

  # ------------------------------------------------------------------ resources
  define oil   as res [bbl]  { rate_unit = "bbl/d"; attr: api = 40; };
  define gas   as res [Mscf] { rate_unit = "Mscf/d"; };
  define water as res [bbl]  { rate_unit = "bbl/d"; };
  define truck_hours as res [h] { rate_unit = "trucks"; };
  define fluids as grp { oil; gas; water; };
  define liquids as grp { oil; water; };

  # ------------------------------------------------------------------ wells
  ## Oil well lifted by an electric submersible pump (ESP) on a variable speed drive.
  ## Rate follows the pump affinity law (rate ~ speed, power ~ speed^3); running fast
  ## accelerates pump wear, which raises the expected cost of a failure.
  define esp_well as ent {
    geom: point; layer = "wells"; icon = "well";
    properties:
      q_max    = 900  [bbl/d]    calibrate in [30, 5000];   ## liquid rate at 100 % pump speed
      wc       = 0.70;                                      ## water cut (updated by well tests)
      gor      = 1.6  [Mscf/bbl] calibrate in [0.05, 30];   ## producing gas-oil ratio
      kw_rated = 130  [kW];                                 ## motor power at full speed
      opex     = 14   [USD/h];                              ## fixed lease operating cost
      restart_cost = 650 [USD];                             ## crew visit + ESP restart wear
      wear_rate = 0.0001 [1/h];                             ## health loss per hour at full speed (~14 months life)
      fail_cost = 45000 [USD];                              ## extra cost of an unplanned (vs planned) failure
      pump_cost = 140000 [USD];                             ## replacement cost of the pump (life consumed)
      vib0     = 1.8  [mm/s];
      vib_gain = 9.0  [mm/s];
    state:
      up = 1;            ## 1 running, 0 tripped/failed (from SCADA)
      was_on = 1;
      health = 1;        ## 1 = new pump, 0 = failure (estimated from vibration)
    decision:
      speed in [0.35, 1.0] default 0.85 hold 2;    ## VSD speed set point
      on : bool default 1 hold 6;                   ## stop / start
    nodes:
      out prod <- oil, gas, water;
    telemetry:
      liquid   <- "FLOW.LIQ" [bbl/d];
      motor_kw <- "ESP.KW" [kW];
      vib      <- "ESP.VIB" [mm/s] { warn = 5.5; alarm = 8.0; };
      up       <- "ESP.RUN";
      wc       <- "TEST.WC" { test = true; };              ## well test (test separator)
      gor      <- "TEST.GOR" [Mscf/bbl] { test = true; };
    behavior:
      run = on * up;
      liquid = q_max * pwl(speed, [0.35, 1.0], [0.22, 1.0]) * run;
      prod.oil   = liquid * (1 - wc);
      prod.water = liquid * wc;
      prod.gas   = prod.oil * gor;
      motor_kw = kw_rated * speed^3 * run;
      vib = vib0 + vib_gain * (1 - health);
      health <- health - wear_rate * speed^4 * run * dt;
      was_on <- run;
      hazard = 0.002 * exp(5 * (1 - health));          ## failures per day
      cost power    = motor_kw * env.power_price;
      cost lease    = opex * up;
      cost restarts = restart_cost * pos(run - was_on) / dt;
      cost risk     = fail_cost * hazard / 24 [h] * run;
      cost pump_wear = pump_cost * wear_rate * speed^4 * run;
      emission grid = motor_kw * env.grid_co2;
      metric oil_rate = prod.oil [bbl/d];
  };

  ## Rod-pumped (beam) well with pump-off control: runs a duty cycle.
  define rod_pump_well as ent {
    geom: point; layer = "wells"; icon = "rodpump";
    properties:
      q_max = 120 [bbl/d] calibrate in [5, 600];
      wc    = 0.8          calibrate in [0, 0.99];
      gor   = 0.8 [Mscf/bbl] calibrate in [0.05, 20];
      kw    = 22  [kW];
      opex  = 6   [USD/h];
    state: up = 1;
    decision: duty in [0.2, 1.0] default 0.75 hold 4;
    nodes: out prod <- oil, gas, water;
    telemetry:
      liquid <- "FLOW.LIQ" [bbl/d];
      up     <- "RPC.RUN";
    behavior:
      liquid = q_max * duty * up;
      prod.oil = liquid * (1 - wc);
      prod.water = liquid * wc;
      prod.gas = prod.oil * gor;
      cost power = kw * duty * up * env.power_price;
      cost lease = opex * up;
      emission grid = kw * duty * up * env.grid_co2;
      metric oil_rate = prod.oil [bbl/d];
  };

  # ------------------------------------------------------------------ gathering
  ## Multi-well pad header. The valve line-up sends the whole pad to one destination.
  define pad as ent {
    geom: point; layer = "pads"; icon = "pad";
    nodes:
      in  wells  <- oil, gas, water;
      out header <- oil, gas, water exclusive;
    behavior:
      header = wells;
  };

  ## Generic flowline / pipeline segment (used with 'connect(a, b) via flowline').
  define flowline as ent {
    geom: line; layer = "pipelines";
    properties:
      cap_liquid = 60000 [bbl/d];
      cap_gas    = 80000 [Mscf/d];
      length     = auto [km];
    nodes: in a <- *; out b <- *;
    behavior:
      b = a;
      constraint liquid_capacity: a.oil + a.water <= cap_liquid;
      constraint gas_capacity: a.gas <= cap_gas;
  };

  # ------------------------------------------------------------------ facilities
  ## Central tank battery: three-phase separation, oil and water storage, pump-out.
  define battery as ent {
    geom: point; layer = "facilities"; icon = "battery";
    properties:
      liq_cap    = 30000 [bbl/d]  calibrate in [5000, 90000];
      gas_cap    = 22000 [Mscf/d];
      oil_tank   = 9000  [bbl];
      wat_tank   = 12000 [bbl];
      oil_pump   = 1600  [bbl/h];
      wat_pump   = 2600  [bbl/h];
      treat_cost = 0.22  [USD/bbl];
    state:
      oil_lvl = 0.45 * oil_tank [bbl] in [0, oil_tank];
      wat_lvl = 0.45 * wat_tank [bbl] in [0, wat_tank];
    decision:
      oil_rate in [0, oil_pump] [bbl/h] default 0.17 * oil_pump hold 2;
      wat_rate in [0, wat_pump] [bbl/h] default 0.32 * wat_pump hold 2;
    nodes:
      in  feed    <- oil, gas, water;
      out gas_out <- gas;
      out oil_out <- oil;
      out wat_out <- water;
    telemetry:
      oil_lvl <- "TK.OIL.LVL" [bbl];
      wat_lvl <- "TK.WTR.LVL" [bbl];
      liquid  <- "SEP.LIQ"    [bbl/d];
      gas_in  <- "SEP.GAS"    [Mscf/d];
    behavior:
      liquid = feed.oil + feed.water;
      gas_in = feed.gas;
      constraint liquid_capacity: liquid <= liq_cap;
      constraint gas_capacity: gas_in <= gas_cap;
      oil_sent = sat(oil_rate, oil_lvl / dt + feed.oil);
      wat_sent = sat(wat_rate, wat_lvl / dt + feed.water);
      gas_out.gas = feed.gas;
      oil_out.oil = oil_sent;
      wat_out.water = wat_sent;
      oil_lvl <- oil_lvl + (feed.oil - oil_sent) * dt;
      wat_lvl <- wat_lvl + (feed.water - wat_sent) * dt;
      cost treating = treat_cost * liquid;
      final revenue oil_inventory = 0.9 * oil_lvl * env.oil_price;
      final cost water_inventory = 0.3 [USD/bbl] * wat_lvl;
  };

  ## Compressor station with identical units (unit commitment is a decision).
  ## Gas above the running capacity is routed to the station flare.
  define compressor_station as ent {
    geom: point; layer = "facilities"; icon = "compressor";
    properties:
      n_units    = 3;
      unit_cap   = 9000 [Mscf/d];
      unit_kw    = 1100 [kW];
      idle_frac  = 0.35;             ## share of full-load power drawn by a running unit at zero load
      fuel_frac  = 0.012;            ## fuel gas share of compressed gas
      start_cost = 1800 [USD];
      co2e_fuel  = 54 [kgCO2e/Mscf];
      headroom_cost = 4 [USD/Mscf];   ## expected flaring cost per Mscf/d of throughput in the last 5 % of capacity
    state:
      avail = n_units;               ## units available (trips / maintenance reduce it)
      running_prev = n_units;
    decision:
      units : int in [0, n_units] default n_units hold 3;
    nodes:
      in  suction   <- gas;
      out discharge <- gas;
      out flare     <- gas;
    telemetry:
      suction_gas <- "SUC.FLOW" [Mscf/d];
      running     <- "UNITS.RUN";
      avail       <- "UNITS.AVAIL";
    behavior:
      suction_gas = suction.gas;
      running  = min(units, avail);
      capacity = running * unit_cap;
      compressed = min(suction_gas, capacity);
      discharge.gas = compressed * (1 - fuel_frac);
      flare.gas = suction_gas - compressed;
      load = compressed / max(capacity, 1 [Mscf/d]);
      kw = unit_kw * (idle_frac * running + (1 - idle_frac) * compressed / unit_cap);
      cost power  = kw * env.power_price;
      cost starts = start_cost * pos(running - running_prev) / dt;
      ## robust planning: within 5 % of capacity, forecast error of the gas rate makes flaring likely
      cost flare_risk = headroom_cost * pos(suction_gas - 0.95 * capacity);
      running_prev <- running;
      emission fuel = compressed * fuel_frac * co2e_fuel;
      emission grid = kw * env.grid_co2;
  };

  ## Flare stack: burns whatever gas cannot be compressed.
  define flare as ent {
    geom: point; layer = "facilities"; icon = "flare";
    properties: co2e = 110 [kgCO2e/Mscf];      ## CO2 + methane slip at real-world combustion efficiency
    nodes: in burn <- gas;
    telemetry: burned <- "FLARE.FLOW" [Mscf/d];
    behavior:
      burned = burn.gas;
      emission flaring = burned * co2e;
      metric flared = burned [Mscf/d];
  };

  ## Salt-water disposal well.
  define swd_well as ent {
    geom: point; layer = "water"; icon = "swd";
    properties:
      inj_cap    = 25000 [bbl/d] calibrate in [500, 90000];
      fee        = 0.08  [USD/bbl];
      kwh_per_bbl = 0.9  [kWh/bbl];
    state: up = 1;
    nodes: in inj <- water;
    telemetry:
      injected <- "INJ.FLOW" [bbl/d];
      p_inj    <- "INJ.PRESS" [psi];
      up       <- "PUMP.RUN";
    behavior:
      injected = inj.water;
      p_inj = 650 [psi] + 1500 [psi] * (injected / max(inj_cap, 1 [bbl/d]))^2;   ## injectivity curve
      constraint injectivity: injected <= inj_cap * up;
      cost disposal = fee * injected;
      cost pumping  = kwh_per_bbl * injected * env.power_price;
  };

  # ------------------------------------------------------------------ logistics
  ## Truck fleet: supplies 'truck hours per hour' (= trucks in service) to lanes.
  define truck_fleet as ent {
    geom: point; layer = "logistics"; icon = "truckyard";
    properties: n_trucks = 8; standby = 22 [USD/h];
    state: avail = n_trucks;
    nodes: out pool <- truck_hours;
    telemetry: avail <- "FLEET.AVAIL";
    behavior:
      pool.truck_hours = avail;
      cost standby = standby * avail;
  };

  ## Trucking lane (fluid model): hauling a rate needs rate / load * cycle trucks.
  ## The dispatch layer turns the planned rate into individual truck trips.
  define truck_lane as ent {
    geom: line; layer = "roads";
    properties:
      distance  = 30 [km];
      speed     = 48 [km/h];
      load      = 180 [bbl];
      load_time = 0.75 [h];
      cost_h    = 92 [USD/h];
    nodes:
      in  cargo <- *;
      in  fleet <- truck_hours;
      out drop  <- *;
    behavior:
      cycle  = 2 * distance / speed + load_time;
      volume = cargo.oil + cargo.water;
      trucks = volume / load * cycle;
      constraint fleet_capacity: trucks <= fleet.truck_hours;
      drop = cargo;
      cost hauling = trucks * cost_h;
      emission diesel = trucks * 38 [kgCO2e/h];
      metric trucks_busy = trucks [trucks];
  };

  # ------------------------------------------------------------------ markets
  ## Crude oil sales point (LACT unit / terminal).
  define oil_sales as ent {
    geom: point; layer = "markets"; icon = "terminal";
    properties: diff = -1.8 [USD/bbl]; cap = 1000000 [bbl/d];
    nodes: in rec <- oil;
    telemetry: delivered <- "LACT.FLOW" [bbl/d];
    behavior:
      delivered = rec.oil;
      constraint takeaway: delivered <= cap;
      revenue crude = delivered * (env.oil_price + diff);
      metric oil_sold = delivered [bbl/d];
  };

  ## Gas processing plant / sales meter with firm takeaway capacity.
  define gas_sales as ent {
    geom: point; layer = "markets"; icon = "gasplant";
    properties:
      cap  = 60000 [Mscf/d];
      fee  = 0.42  [USD/Mscf];      ## gathering & processing fee
      ngl  = 0.55  [USD/Mscf];      ## NGL uplift captured by processing
    nodes: in rec <- gas;
    telemetry: delivered <- "METER.FLOW" [Mscf/d];
    behavior:
      delivered = rec.gas;
      constraint takeaway: delivered <= cap;
      revenue gas = delivered * (env.gas_price + ngl - fee);
      metric gas_sold = delivered [Mscf/d];
  };

  # ------------------------------------------------------------------ field resources
  ## Workover rig / maintenance crew (scheduled by the PLYXIS maintenance dispatcher).
  define workover_rig as ent {
    geom: point; layer = "crews"; icon = "rig";
    properties: day_rate = 16000 [USD/d]; move_speed = 25 [km/h]; job_hours = 30 [h];
  };
  define field_crew as ent {
    geom: point; layer = "crews"; icon = "crew";
    properties: hourly = 140 [USD/h]; move_speed = 55 [km/h]; job_hours = 3 [h];
  };
};
