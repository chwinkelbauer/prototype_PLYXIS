# PLX — the PLYXIS twin description language

PLX describes physical operations the way VHDL describes circuits: **blueprints** (entities with ports,
properties, state, decisions and behavior) are **instantiated** and **connected** into a site; the compiler
**elaborates** the site into a netlist and **synthesises** an executable, differentiable simulator from it.
Nothing about trucks, wells or excavators is built into the engine — every asset type is ordinary PLX code
(see `plyxis/lang/stdlib/*.plx`).

```
PLX source ──lexer/parser──► AST ──library/types──► elaboration (generate loops, data tables,
composites, connections, units) ──► flat netlist ──behavior compiler──► dataflow IR per type
──scheduler (SCC: vectorised │ cumsum │ scan)──► generated Python/torch simulator + bounds
```

## 1. Program structure

```plx
define oilgas as pkg {                 # package (namespace); 'use oilgas;' imports it
  define oil   as res [bbl] { rate_unit = "bbl/d"; attr: api = 40; };   # flowing resource (+ attributes)
  define fluids as grp { oil; gas; water; };                             # group of resources (or instances)
  define esp_well as ent { ... };                                        # entity blueprint
  define esp_well_hp as esp_well { properties: kw_rated = 220 [kW]; };   # inheritance: 'as <base type>'
  define margin(p, c) = p - c;                                           # pure function (inlined)
};
use oilgas;
define kestrel as env { ... };          # site / environment (root of elaboration)
define ops72 as sim { env: kestrel; horizon = 72 [h]; step = 1 [h]; objective: maximize ...; };
set kestrel.P01_W03.q_max = 720 [bbl/d];   # overlay (used by calibration and what-if studies)
```

Kinds: `pkg`, `ent` (entity), `obj` (one-off object, instantiated where defined), `env`, `sim`, `grp`, `res`.
`define X as Y` where `Y` is an existing type means inheritance; an environment can extend another
environment (`define kestrel_live as kestrel { contains: ... }`) — this is how the data-integration layer adds
discovered assets without touching the engineered model.

## 2. Entities

```plx
define battery as ent {
  geom: point; layer = "facilities"; icon = "battery";        # GIS rendering hints
  properties:                                                  # parameters (runtime-updatable tensors)
    liq_cap  = 30000 [bbl/d] calibrate in [5000, 90000];       # 'calibrate' -> estimated online from data
    oil_tank = 9000  [bbl];
  state:                                                       # state with initial value and physical limits
    oil_lvl = 0.45 * oil_tank [bbl] in [0, oil_tank];
  decision:                                                    # degrees of freedom for the optimizer
    oil_rate in [0, oil_pump] default 0.17 * oil_pump hold 2;  # bounds may use properties; hold = block length
    on : bool default 1 hold 6;                                # bool / int decisions
  nodes:                                                       # ports carry resource flows (rates)
    in  feed    <- oil, gas, water;
    out oil_out <- oil;                                        # fan-out to several destinations => routing decision
    out header  <- oil, gas, water exclusive;                  # exclusive => one destination at a time
  telemetry:                                                   # binding to SCADA tags (instance tag + suffix)
    oil_lvl <- "TK.OIL.LVL" [bbl];
    vib     <- "ESP.VIB" [mm/s] { warn = 5.5; alarm = 8; };
    wc      <- "TEST.WC" { test = true; };                     # direct measurement of a property (well test)
  behavior:
    liquid = feed.oil + feed.water;                            # local signal (same time step)
    constraint liquid_capacity: liquid <= liq_cap;             # hard constraint
    constraint soft(40 [USD/Mscf]) permit: flare <= limit;     # soft constraint with penalty
    oil_sent = sat(oil_rate, oil_lvl / dt + feed.oil);         # saturates in the emulator, constraint for planner
    oil_out.oil = oil_sent;                                    # out-port flow
    oil_out.oil.api = feed.oil.api;                            # attribute (flow-weighted quality) - optional
    oil_lvl <- oil_lvl + (feed.oil - oil_sent) * dt;           # next-step state update
    cost treating = treat_cost * liquid;                       # objective terms are rates per hour
    revenue sales = oil_sent * env.oil_price;
    emission flaring = burned * co2e;                          # kgCO2e/h, priced in the sim objective
    metric field_oil = sum(esp_well.prod.oil) [bbl/d];         # KPI (display unit optional)
    final revenue inventory = 0.9 * oil_lvl * env.oil_price;   # terminal value at the end of the horizon
};
```

### Semantics (synchronous dataflow)

| statement | meaning |
|---|---|
| `x = e;` | signal `x` at the same time step (single assignment; algebraic loops are compile errors) |
| `port.res = e;` | flow on an out-port channel (rate) |
| `out = in;` | copy all channels of an in-port to an out-port (pass-through, e.g. pipelines) |
| `s <- e;` | state update: `s[t+1] = e[t]` |
| `a <= b;` / `a >= b;` / `a == b;` | constraint (named with `constraint name:`; `soft(w)` makes it a penalty) |
| `cost / revenue / emission / metric name = e;` | objective / KPI terms (per hour); `final` = terminal term |

In-port flows are the sum of the incoming edges; an out-port with several edges is split by **routing
decisions** generated automatically by the compiler. State limits (`in [lo, hi]`) are *constraints* for the
planner and *saturations* in the plant emulator (physical mode). Every feedback path must pass through a state
or `delay()` — exactly like combinational loops in hardware design.

### Builtins

`min max abs sqrt exp log pos clamp sigmoid tanh floor ceil round sign step`, `if(c, a, b)`,
`pwl(x, [xs], [ys])` (piecewise-linear curves, e.g. pump curves), `smin/smax(a, b, k)` (smooth min/max),
`delay(x, k)`, `prev(x, init)`, `sum(port)` (all resources of a port), `sat(rate, available)`,
`dt t time hour dow`, `env.<property>`. Environment-level behavior may aggregate over types, groups or single
instances: `sum(flare.burned)`, `max(battery.oil_lvl)`, `CTB_N.oil_lvl`.

### Units

Literals carry units (`1200 [bbl/d]`, `1.5 [USD/t] * dig`). Values are converted to a coherent internal system
(m, kg, h, USD, kWh, kPa, K) so that equations stay dimensionally consistent; the compiler warns about
dimension mismatches (`bbl/d` vs `bbl/h` are compatible, `bbl` vs `kW` are not). Displays convert back.

## 3. Environments (sites)

```plx
define kestrel as env {
  origin = (31.61, -103.68);
  properties:
    oil_price = 72 [USD/bbl] input;          # 'input' => time series supplied by forecasts
    flare_permit = 1500 [Mscf/d];
  contains:
    init CTB_N as battery { at = (31.675, -103.700); tag = "KST.CTBN"; liq_cap = 24000 [bbl/d]; };
    for w in table("wells.csv") {            # data-driven instantiation (asset register / GIS export)
      init {w.name} as esp_well { q_max = w.q_max; at = (w.lat, w.lon); tag = w.tag; };
      connect({w.name}.prod, {w.pad}.wells);
    }
    for i in 1..4 { init CREW{i} as field_crew; }            # generate loops (VHDL 'generate')
    connect(PAD01.header, CTB_N.feed) via flowline { cap_liquid = 14000 [bbl/d]; };   # link entity
    connect(CTB_S.oil_out, LACT.rec) via truck_lane { distance = 24 [km]; } as LANE_S_OIL;
  behavior:
    constraint soft(40 [USD/Mscf]) flare_permit: sum(flare.burned) <= flare_permit;
  discover:                                   # how the twin rewrites itself from live data
    rule new_well: on "KST.P{pad}.W{well}.FLOW.LIQ" {
      init P{pad}_W{well} as esp_well { tag = "KST.P{pad}.W{well}"; };
      connect(P{pad}_W{well}.prod, PAD{pad}.wells);
    };
};
```

* `via T` inserts a link entity (pipeline, truck lane, conveyor) between two ports; `length = auto` takes the
  GIS length of the link.
* Composite entities (an `ent` with `contains:`) are flattened; their ports become implicit junctions.
* Ports declared `<- *` adapt to whatever is connected (the type is specialised per signature).
* `{expr}` inside names interpolates loop variables or table fields (`w{i}`, `{r.name}`). Inside string
  literals, `{name}` interpolates a loop variable or discovery-rule binding (`tag = "KST.P{pad}.W{well}"`);
  unknown names are left as written.
* The kind keywords (`env`, `site`, `ent`, `pkg`, `sim`, `scenario`, ...) cannot be extended by name:
  `define x as site` declares a new environment. Name environments something else (the compiler warns).

## 4. Simulations

```plx
define ops72 as sim {
  env: kestrel; horizon = 72 [h]; step = 1 [h];
  objective: maximize sum(revenue) - sum(cost) - carbon_price * sum(emission);
  solver: partition = "pad"; route_hold = 24;    # hints for the decomposition and routing block length
};
```

## 5. What the compiler produces

* **Structural output** — instances, ports, edges, GIS geometry, hierarchy, subsystems (`/api/structure`).
* **Constraint output** — decision space with bounds, hard/soft constraints with scales, objective terms.
* **Executable twin** — generated, time-vectorised Python/torch code (viewable in Twin Studio → *Generated
  code*). One behavior statement is executed once per asset group, over all instances, all time steps and a
  whole batch of candidate plans; feedback loops are scheduled as cumulative sums or sequential scans.
  The generated code is verified against an independent reference interpreter (`plyxis/lang/interp.py`) to
  machine precision in the test-suite.

## 6. Migration from the 2025 DSL draft (`str3` / `test`)

| draft | PLX |
|---|---|
| `define k as ent { part=TRUE; ... }` | still accepted (becomes a resource); prefer `define k as res { attr: quality = 1; }` |
| `inA <- .;` / `outX <- grp_good;` | still accepted (direction from the name); prefer `in feed <- *;` / `out prod <- grp_good;` |
| `decision outX$.$amount(t);` | `decision: amount in [0, max_out];` with `out.k = amount;` |
| `outX$m$amount(t+t_p) <- m;` | `out.m = delay(m, t_p);` |
| `.in$.$amount <- ...` | an explicit state per stored resource: `stock <- stock + (in.k - out.k) * dt;` |
| `costs <- ...` | `cost storage = ...;` |
| `max .all$revenues - .all$costs;` | `objective: maximize sum(revenue) - sum(cost);` (legacy form still parsed) |
| `x$y` | `x.y` (`$` still accepted) |

The compiler reports precise diagnostics (file:line:col, hints) for the unsupported draft constructs instead of
silently ignoring them.
