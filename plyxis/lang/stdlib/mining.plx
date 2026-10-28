## PLYXIS standard library - mining: open pit, underground (sublevel stoping, shaft hoisting,
## ventilation on demand, dewatering) and mineral processing.
## Same engine as oil & gas: only the entity definitions differ.

define mining as pkg {

  define ore     as res [t] { rate_unit = "t/h"; attr: grade = 1.0; };   ## grade in % (e.g. Cu)
  define waste   as res [t] { rate_unit = "t/h"; };
  define concentrate as res [t] { rate_unit = "t/h"; };
  define haul_hours as res [h] { rate_unit = "trucks"; };
  define lhd_hours as res [h] { rate_unit = "units"; };           ## loaders in service
  define air as res [m3] { rate_unit = "m3/s"; };                  ## ventilation air

  ## Excavator working a dig face with known (block-model) grade.
  define dig_face as ent {
    geom: point; layer = "faces"; icon = "excavator";
    properties:
      rate_max  = 2200 [t/h];
      grade     = 0.9;
      strip     = 1.5;                ## t waste per t ore
      reserve   = 400000 [t];
      cost_t    = 0.65 [USD/t];
    state: up = 1; remaining = reserve [t] in [0, reserve];
    decision: rate in [0, rate_max] default 0.75 * rate_max hold 2;
    nodes: out ore_out <- ore; out waste_out <- waste;
    telemetry: dug <- "EXC.RATE" [t/h]; up <- "EXC.RUN";
    behavior:
      dug = sat(rate * up, remaining / dt);
      ore_out.ore = dug;
      ore_out.ore.grade = grade;
      waste_out.waste = dug * strip;
      remaining <- remaining - dug * dt;
      cost digging = cost_t * dug * (1 + strip);
  };

  ## Haul truck fleet (trucks in service flow to the routes that need them).
  define haul_fleet as ent {
    geom: point; layer = "logistics"; icon = "truckyard";
    properties: n_trucks = 24; ownership = 0 [USD/h];        ## lease + fixed maintenance per truck
    state: avail = n_trucks;
    nodes: out pool <- haul_hours;
    telemetry: avail <- "FLEET.AVAIL";
    behavior:
      pool.haul_hours = avail;
      cost ownership = ownership * n_trucks;
  };

  ## Haul route: payload, cycle time and fuel; needs trucks from the fleet.
  define haul_route as ent {
    geom: line; layer = "roads";
    properties:
      distance = 3.2 [km]; speed_loaded = 22 [km/h]; speed_empty = 35 [km/h];
      payload = 220 [t]; spot_load = 0.08 [h]; fuel_cost = 210 [USD/h];
    nodes: in load <- *; in trucks <- haul_hours; out dump <- *;
    behavior:
      cycle = distance / speed_loaded + distance / speed_empty + spot_load;
      tonnes = sum(load);
      needed = tonnes / payload * cycle;
      constraint fleet: needed <= trucks.haul_hours;
      dump = load;
      cost hauling = needed * fuel_cost;
      metric trucks_busy = needed;
  };

  ## Blending stockpile tracking tonnage and average grade.
  define stockpile as ent {
    geom: point; layer = "stockpiles"; icon = "stockpile";
    properties: cap = 150000 [t]; reclaim_max = 3000 [t/h];
      unit_cost = 12.7 [USD/t];   ## inventory valued at cost (IAS 2: lower of cost and net realisable value)
      reserve_t = 0 [t];          ## working stock kept for what is not announced (hoist stop, loaders down)
      reserve_price = 1 [USD/t/h];   ## planning price of drawing on it - far below the mill margin, so an
                                     ## outage is bridged from the reserve instead of stopping the mill
    state: tonnes = 40000 [t] in [0, cap]; grade_avg = 0.85;
    decision: reclaim in [0, reclaim_max] default 0.6 * reclaim_max hold 2;
    nodes: in feed <- ore; out out <- ore;
    behavior:
      taken = sat(reclaim, tonnes / dt + feed.ore);
      out.ore = taken;
      out.ore.grade = grade_avg;
      tonnes <- tonnes + (feed.ore - taken) * dt;
      grade_avg <- (grade_avg * pos(tonnes) + feed.ore.grade * feed.ore * dt) / max(pos(tonnes) + feed.ore * dt, 1 [t]);
      final revenue inventory = unit_cost * tonnes;
      ## a buffer, not a store of value: valued at cost like every inventory, with a reserve for surprises.
      ## (A terminal value at the full milling margin assumes spare mill capacity after the horizon; in a
      ## mill-bound mine it made the planner idle the mill for five hours of a price peak.)
      constraint soft(reserve_price) working_stock: tonnes >= reserve_t;
  };

  ## Crusher + concentrator: grade window, recovery curve, energy.
  define concentrator as ent {
    geom: point; layer = "plants"; icon = "plant";
    properties:
      cap = 2600 [t/h]; grade_min = 0.6; grade_max = 1.4;
      kwh_t = 14 [kWh/t]; conc_grade = 28;          ## % metal in concentrate
      cost_t = 0 [USD/t];                           ## grinding media, liners, reagents, tailings
      grade_penalty = 250 [USD/t];                  ## planning price per tonne milled and %-point outside the
                                                    ## head-grade window (~3x the metal in 1 %-point of a tonne)
    nodes: in feed <- ore; out conc <- concentrate;
    telemetry: throughput <- "MILL.TPH" [t/h]; g <- "OSA.HEAD";       ## on-stream analyser: head grade
    behavior:
      throughput = feed.ore;
      g = feed.ore.grade;
      recovery = pwl(g, [0.3, 0.8, 1.5], [0.70, 0.88, 0.92]);
      constraint capacity: throughput <= cap;
      ## off-spec feed is priced per tonne milled: an idle mill has no head grade (the flow-weighted grade
      ## of no flow is 0, which a plain `g >= grade_min` would charge every hour the mill stands)
      constraint soft(grade_penalty) grade_low: throughput * g >= throughput * grade_min;
      constraint soft(grade_penalty) grade_high: throughput * g <= throughput * grade_max;
      conc.concentrate = throughput * g / 100 * recovery * 100 / conc_grade;
      kw = kwh_t * throughput;
      cost energy = kw * env.power_price;
      cost consumables = cost_t * throughput;
      metric milled = throughput [t/h];
  };

  ## Waste dump.
  define dump as ent {
    geom: point; layer = "dumps"; icon = "dump";
    properties: cost_t = 0.15 [USD/t];
    nodes: in rec <- waste;
    behavior: cost dumping = cost_t * rec.waste;
  };

  ## Concentrate sale (paid metal).
  define conc_sales as ent {
    geom: point; layer = "markets"; icon = "terminal";
    properties: payable = 0.965; conc_grade = 28;
    nodes: in rec <- concentrate;
    behavior:
      revenue metal = rec.concentrate * conc_grade / 100 * payable * env.metal_price;
      metric metal_sold = conc_grade / 100 * rec.concentrate [t/h];
  };

  # ================================================================== underground
  ## Production stope (sublevel open stope with its drawpoints). Ring blasts break ore into the
  ## broken stock; loaders (LHDs) muck it to the ore pass. Diesel loaders need ventilation air by
  ## statute (m3/s per kW), so every tonne mucked competes for loaders and for air.
  define stope as ent {
    geom: point; layer = "stopes"; icon = "stope";
    properties:
      rate_max   = 260 [t/h] calibrate in [20, 900];   ## drawpoint productivity (fragmentation, hang-ups)
      grade      = 1.2 calibrate in [0.05, 6];         ## Cu-equivalent % (block model, reconciled by the mill)
      ring_t     = 9000 [t];                           ## tonnes broken by one ring firing
      broken_cap = 60000 [t];
      tram       = 0.35 [km];                          ## one-way tram drawpoint -> ore pass tip
      bucket     = 17 [t];
      lhd_speed  = 10 [km/h];
      load_time  = 0.025 [h];
      lhd_kw     = 300 [kW];                           ## diesel engine of a loader
      lhd_cost   = 185 [USD/h];                        ## fuel, tyres, parts per operating hour
      air_per_kw = 0.06 [m3/s/kW];                     ## statutory air per kW of diesel
      air_base   = 10 [m3/s];                          ## stope ventilation while it is worked
      cost_t     = 8.5 [USD/t];                        ## drilling, ground support, paste fill
      ring_cost  = 30000 [USD];                        ## explosives and charging per ring
      reentry    = 0.5;                                ## share of a blasting shift lost to re-entry
    state:
      up = 1;                                          ## 0 = closed (seismicity, fill, rehab)
      broken = 0.5 * broken_cap [t] in [0, broken_cap];
    decision:
      muck in [0, rate_max] default 0.4 * rate_max hold 4;
      blast : bool default 0 hold 12;
    nodes:
      ## the stope requests loader hours from its zone's fleet and air from its ventilation district;
      ## the capacity limits sit with the fleet and the district (sum of requests <= what they have)
      out lhd_req <- lhd_hours;
      out air_req <- air;
      out ore_out <- ore;
    telemetry:
      mucked <- "LHD.TPH" [t/h];
      broken <- "CMS.BROKEN" [t];
      air_q  <- "VENT.AIR" [m3/s];
      up     <- "STOPE.OPEN";
      grade  <- "ASSAY.CU" { test = true; };           ## muck-pile grab samples (assayed, rotating)
    behavior:
      ## re-entry after a blast caps the drawpoint rate; it is never a way to reduce mucking (that is the set point)
      mucked = sat(min(muck, rate_max * (1 - reentry * blast)) * up, broken / dt);
      cycle = 2 * tram / lhd_speed + load_time;
      lhd_use = mucked / bucket * cycle;
      lhd_req.lhd_hours = lhd_use;
      active = clamp(mucked / (0.15 * rate_max), 0, 1);
      air_need = air_base * active + air_per_kw * lhd_kw * lhd_use;
      air_q = air_need;
      air_req.air = air_need;
      ore_out.ore = mucked;
      ore_out.ore.grade = grade;
      ## a ring fired into a stope without void chokes: it breaks only what fits (the cost is paid anyway)
      broken <- broken - mucked * dt + blast * up * min(ring_t, pos(broken_cap - broken)) / 12 [h] * dt;
      cost mining = cost_t * mucked;
      cost loading = lhd_cost * lhd_use;
      cost blasting = blast * up * ring_cost / 12 [h];
      emission diesel = lhd_use * lhd_kw * 0.27 [kgCO2e/kWh];
      metric ore_mucked = mucked [t/h];
      ## broken ore is an inventory at cost (IAS 2: its blasting cost), like ore in passes, bins and on the pad
      ## - not its metal: the in-situ ore is mined sooner or later anyway, so blasting creates no value
      final revenue broken_value = broken * ring_cost / ring_t;
      ## planning only: broken ore left at the horizon end counts at half its cost, so rings are fired just in time
      final cost value_to_go = 0.5 * broken * ring_cost / ring_t;
  };

  ## Ore pass: gravity raise from the production levels to the haulage level (surge bin).
  define orepass as ent {
    geom: point; layer = "orepasses"; icon = "orepass";
    properties: cap = 5000 [t]; draw_max = 900 [t/h];
      unit_cost = 11.2 [USD/t];   ## cost to break and muck a tonne into the pass (IAS 2 inventory at cost)
    state: stock = 0.4 * cap [t] in [0, cap]; grade_avg = 1.0;
    decision: draw in [0, draw_max] default 0.3 * draw_max hold 2;
    nodes: in feed <- ore; out chute <- ore;
    telemetry: stock <- "OP.LEVEL" [t];
    behavior:
      drawn = sat(draw, stock / dt + feed.ore);
      chute.ore = drawn;
      chute.ore.grade = grade_avg;
      stock <- stock + (feed.ore - drawn) * dt;
      grade_avg <- (grade_avg * pos(stock) + feed.ore.grade * feed.ore * dt) / max(pos(stock) + feed.ore * dt, 1 [t]);
      final revenue stock_value = unit_cost * stock;
  };

  ## Loader (LHD) fleet of a mining zone: the loader hours its stopes request cannot exceed the loaders
  ## available (a loader can serve two stopes in a shift, so the requests are fractional).
  define lhd_fleet as ent {
    geom: point; layer = "equipment"; icon = "lhd";
    properties: n_units = 9; ownership = 95 [USD/h];     ## lease + fixed maintenance per unit
    state: avail = n_units;
    nodes: in requests <- lhd_hours;
    telemetry: avail <- "FLEET.AVAIL"; in_use <- "FLEET.BUSY";
    behavior:
      in_use = requests.lhd_hours;
      constraint fleet_size: in_use <= avail;
      cost ownership = ownership * n_units;
      metric loaders_busy = in_use [units];
  };

  ## Main ventilation fan (surface exhaust): air quantity ~ speed, power ~ speed^3 (fan laws).
  define main_fan as ent {
    geom: point; layer = "ventilation"; icon = "fan";
    properties: q_max = 420 [m3/s]; kw_max = 3400 [kW];
    state: up = 1;
    decision: speed in [0.35, 1.0] default 0.9 hold 2;
    nodes: in requests <- air;
    telemetry: airflow <- "FAN.Q" [m3/s]; kw <- "FAN.KW" [kW]; up <- "FAN.RUN";
    behavior:
      airflow = q_max * speed * up;
      ## the regulators of the districts on this raise cannot draw more than the fan moves
      constraint capacity: requests.air <= airflow;
      kw = kw_max * speed^3 * up;
      cost power = kw * env.power_price;
      emission grid = kw * env.grid_co2;
  };

  ## Ventilation district (a production level behind its regulator): receives air from the main fans,
  ## loses its leakage, covers fixed needs (development, workshops) and the air its working stopes request.
  ## The regulator setting is the air drawn from each of the two raises that ventilate the level.
  define vent_district as ent {
    geom: point; layer = "ventilation"; icon = "regulator";
    properties: leakage = 0.18; base_need = 20 [m3/s]; q_max = 160 [m3/s];
    decision:
      air_a in [0, q_max] default 0.3 * q_max hold 2;
      air_b in [0, q_max] default 0.1 * q_max hold 2;
    nodes: out from_a <- air; out from_b <- air; in demand <- air;
    telemetry: supply_q <- "REG.Q" [m3/s]; demand_q <- "REG.DEMAND" [m3/s];
    behavior:
      from_a.air = air_a;
      from_b.air = air_b;
      supply_q = air_a + air_b;
      demand_q = demand.air;
      usable = supply_q * (1 - leakage);
      constraint ventilation: base_need + demand.air <= usable;
  };

  ## Underground jaw crusher on the haulage level.
  define ug_crusher as ent {
    geom: point; layer = "processing"; icon = "crusher";
    properties: cap = 1100 [t/h]; kwh_t = 0.6 [kWh/t]; cost_t = 0.35 [USD/t];
    state: up = 1;
    nodes: in feed <- ore; out product <- ore;
    telemetry: tph <- "CR.TPH" [t/h]; up <- "CR.RUN";
    behavior:
      tph = feed.ore;
      constraint capacity: tph <= cap * up;
      product = feed;
      kw = kwh_t * tph;
      cost power = kw * env.power_price;
      cost wear = cost_t * tph;
  };

  ## Conveyor (link entity): capacity and power per tonne-kilometre.
  define conveyor as ent {
    geom: line; layer = "conveyors";
    properties: cap = 2000 [t/h]; kwh_t_km = 0.45 [kWh/t/km]; length = auto [km];
    nodes: in a <- *; out b <- *;
    behavior:
      b = a;
      constraint capacity: a.ore <= cap;
      kw = kwh_t_km * length * a.ore;
      cost power = kw * env.power_price;
  };

  ## Production shaft: loading-pocket bin and skip hoist to the surface.
  define skip_hoist as ent {
    geom: point; layer = "shafts"; icon = "headframe";
    properties:
      cap = 1750 [t/h]; bin_cap = 6000 [t]; depth = 1050 [m]; unit_cost = 12.2 [USD/t];
      kwh_t_km = 3.2 [kWh/t/km]; fixed_kw = 450 [kW];
    state: up = 1; bin = 0.4 * bin_cap [t] in [0, bin_cap]; grade_avg = 1.0;
    decision: hoist in [0, cap] default 0.75 * cap hold 1;
    nodes: in feed <- ore; out skip <- ore;
    telemetry: hoisted <- "HOIST.TPH" [t/h]; bin <- "BIN.LEVEL" [t]; kw <- "HOIST.KW" [kW]; up <- "HOIST.RUN";
    behavior:
      hoisted = sat(hoist * up, bin / dt + feed.ore);
      skip.ore = hoisted;
      skip.ore.grade = grade_avg;
      bin <- bin + (feed.ore - hoisted) * dt;
      grade_avg <- (grade_avg * pos(bin) + feed.ore.grade * feed.ore * dt) / max(pos(bin) + feed.ore * dt, 1 [t]);
      kw = fixed_kw * up + kwh_t_km * depth * hoisted;
      cost power = kw * env.power_price;
      emission grid = kw * env.grid_co2;
      metric ore_hoisted = hoisted [t/h];
      final revenue bin_value = unit_cost * bin;   ## at cost (IAS 2): mining ahead of the mill creates no value
  };

  ## Dewatering pump station: sump as buffer, pumps the inflow to surface; power ~ flow x head.
  define pump_station as ent {
    geom: point; layer = "dewatering"; icon = "pump";
    properties:
      inflow = 40 [L/s] calibrate in [2, 400]; cap = 90 [L/s]; sump = 2500 [m3];
      head = 900 [m]; eff = 0.72;
      reserve_h = 0 [h];                  ## storage kept free for this many hours of inflow (pump failure)
      reserve_price = 5 [USD/m3/h];       ## planning price of using that reserve while the pump runs
      overflow_cost = 60 [USD/m3];        ## water the sump cannot hold: pumping back, cleaning drives, wet muck
    state: up = 1; level = 0.5 * sump [m3] in [0, sump];
    decision: rate in [0, cap] default 0.5 * cap hold 1;
    telemetry: level <- "SUMP.LEVEL" [m3]; pumped <- "PUMP.FLOW" [L/s]; up <- "PUMP.RUN";
    behavior:
      pumped = sat(rate * up, level / dt + inflow);
      level <- level + (inflow - pumped) * dt;
      ## load shifting may use the sump, but not the room an unannounced pump failure needs
      constraint soft(reserve_price) storage: level <= sump - inflow * reserve_h * up;
      kw = 0.00981 [kW*s/L/m] * pumped * head / eff;
      cost power = kw * env.power_price;
      emission grid = kw * env.grid_co2;
  };

  ## Site services: labour, G&A, camp, fixed maintenance and base power. Not decision-dependent,
  ## but part of the operating cost every plan carries.
  define site_services as ent {
    geom: point; layer = "surface"; icon = "office";
    properties: labour = 14500 [USD/h]; overhead = 4400 [USD/h]; base_kw = 3500 [kW];
    behavior:
      kw = base_kw;
      cost labour = labour;
      cost overhead = overhead;
      cost power = kw * env.power_price;
      emission grid = kw * env.grid_co2;
  };

  ## Mobile equipment unit and mine openings (geometry for the 3D twin, maintenance resources).
  define mobile_unit as ent {
    geom: point; layer = "equipment"; icon = "vehicle";
    properties: rated_kw = 300 [kW];
  };
  define opening as ent {
    geom: line; layer = "development";
    properties: length = auto [km];
  };
};
