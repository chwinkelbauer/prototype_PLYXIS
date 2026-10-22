## PLYXIS standard library - open-pit mining and aggregates.
## Same engine as oil & gas: only the entity definitions differ.

define mining as pkg {

  define ore     as res [t] { rate_unit = "t/h"; attr: grade = 1.0; };   ## grade in % (e.g. Cu)
  define waste   as res [t] { rate_unit = "t/h"; };
  define concentrate as res [t] { rate_unit = "t/h"; };
  define haul_hours as res [h] { rate_unit = "trucks"; };

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
    properties: n_trucks = 24;
    state: avail = n_trucks;
    nodes: out pool <- haul_hours;
    behavior: pool.haul_hours = avail;
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
    state: tonnes = 40000 [t] in [0, cap]; grade_avg = 0.85;
    decision: reclaim in [0, reclaim_max] default 0.6 * reclaim_max hold 2;
    nodes: in feed <- ore; out out <- ore;
    behavior:
      taken = sat(reclaim, tonnes / dt + feed.ore);
      out.ore = taken;
      out.ore.grade = grade_avg;
      tonnes <- tonnes + (feed.ore - taken) * dt;
      grade_avg <- (grade_avg * tonnes + feed.ore.grade * feed.ore * dt) / max(tonnes + feed.ore * dt, 1 [t]);
  };

  ## Crusher + concentrator: grade window, recovery curve, energy.
  define concentrator as ent {
    geom: point; layer = "plants"; icon = "plant";
    properties:
      cap = 2600 [t/h]; grade_min = 0.6; grade_max = 1.4;
      kwh_t = 14 [kWh/t]; conc_grade = 28;          ## % metal in concentrate
    nodes: in feed <- ore; out conc <- concentrate;
    telemetry: throughput <- "MILL.TPH" [t/h];
    behavior:
      throughput = feed.ore;
      g = feed.ore.grade;
      recovery = pwl(g, [0.3, 0.8, 1.5], [0.70, 0.88, 0.92]);
      constraint capacity: throughput <= cap;
      constraint soft(5000) grade_low: g >= grade_min;
      constraint soft(5000) grade_high: g <= grade_max;
      conc.concentrate = throughput * g / 100 * recovery * 100 / conc_grade;
      cost energy = kwh_t * throughput * env.power_price;
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
  };
};
