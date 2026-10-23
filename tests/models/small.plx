## Small test field: 3 wells -> manifold -> 2 batteries (routing) -> sales. Used by the test-suite.
define og as pkg {
  define oil as res [bbl] { rate_unit = "bbl/d"; };
  define gas as res [Mscf];
  define water as res [bbl];
  define fluids as grp { oil; gas; water; };

  define well as ent {
    properties:
      q_max = 1000 [bbl/d] calibrate in [100, 3000];
      wc = 0.6;
      gor = 2.0 [Mscf/bbl];
    state: up = 1;
    decision:
      speed in [0.2, 1.0] default 0.8;
      on : bool default 1 hold 4;
    nodes: out prod <- fluids;
    telemetry:
      liquid <- "FLOW.LIQ" [bbl/d];
      up <- "RUN";
    behavior:
      liquid = q_max * speed * on * up;
      prod.oil = liquid * (1 - wc);
      prod.water = liquid * wc;
      prod.gas = prod.oil * gor;
      cost power = 0.4 [USD/bbl] * liquid;
  };

  define manifold as ent {
    nodes:
      in feed <- fluids;
      out out <- fluids;
    behavior:
      out = feed;
  };

  define battery as ent {
    properties:
      gas_cap = 1500 [Mscf/d];
      tank = 3000 [bbl];
      pump = 150 [bbl/h];
    state:
      lvl = 1000 [bbl] in [0, tank];
    decision:
      draw in [0, pump] [bbl/h] default 30 [bbl/h];
    nodes:
      in feed <- fluids;
      out oil_out <- oil;
      out gas_out <- gas;
    telemetry:
      lvl <- "TK.LVL" [bbl];
      gas_in <- "GAS" [Mscf/d];
    behavior:
      gas_in = feed.gas;
      constraint gas_cap: gas_in <= gas_cap;
      sent = sat(draw, lvl / dt + feed.oil);
      oil_out.oil = sent;
      gas_out.gas = feed.gas;
      lvl <- lvl + (feed.oil - sent) * dt;
      cost treat = 0.5 [USD/bbl] * feed.water;
      final revenue inventory = 0.9 * lvl * env.oil_price;
  };

  define sales as ent {
    properties: diff = -2 [USD/bbl];
    nodes: in rec <- oil, gas;
    behavior:
      revenue oil = rec.oil * (env.oil_price + diff);
      revenue gas = rec.gas * env.gas_price;
      metric oil_sold = rec.oil [bbl/d];
  };
};
use og;
define testfield as env {
  properties:
    oil_price = 70 [USD/bbl] input;
    gas_price = 2 [USD/Mscf] input;
  contains:
    for i in 1..3 {
      init w{i} as well { q_max = 800 + 100 * i; at = (31.4 + 0.01 * i, -103.5); tag = concat("T.W", i); };
      connect(w{i}.prod, m.feed);
    }
    init m as manifold;
    init b1 as battery { tag = "T.B1"; };
    init b2 as battery { gas_cap = 900 [Mscf/d]; tag = "T.B2"; };
    connect(m.out, b1.feed);
    connect(m.out, b2.feed);
    init s as sales;
    connect(b1.oil_out, s.rec); connect(b2.oil_out, s.rec);
    connect(b1.gas_out, s.rec); connect(b2.gas_out, s.rec);
  discover:
    rule new_well: on "T.W{n}.FLOW.LIQ" {
      init w{n} as well { tag = "T.W{n}"; };
      connect(w{n}.prod, m.feed);
    };
};
define plan as sim { env: testfield; horizon = 24 [h]; step = 1 [h]; objective: maximize sum(revenue) - sum(cost); };
