## A small discrete twin for the exact layer (tests/test_reduce.py): kilns that are on or off, with a fixed cost
## while on and a dearer tariff in the day, a minimum run of three hours once started, a shared store and a market
## that takes at most `demand`. The whole MILP is small enough for HiGHS to prove its optimum; the reduction engine
## must reach it (or certify its distance) from the kilns' own exact MILPs and the prices of the shared rows.
define kilns as pkg {
  define lime as res [t] { rate_unit = "t/h"; };
  define kiln as ent {
    properties: cap = 30 [t/h]; fixed = 900 [USD/h]; unit_cost = 35 [USD/t]; day_cost = 20 [USD/t];
      min_run = 3;
    state: starts = 0;
    decision: on : bool default 1 hold 1;
    nodes: out o <- lime;
    behavior:
      day = (hour >= 7) * (hour < 19);
      made = on * cap;
      o.lime = made;
      starts <- starts + pos(on - delay(on, 1, 1));
      ## once started, stay on for min_run hours: on(t) >= start(t - k) for k < min_run
      constraint min_run1: pos(delay(on, 1, 1) - delay(on, 2, 1)) <= on;
      constraint min_run2: pos(delay(on, 2, 1) - delay(on, 3, 1)) <= on;
      cost running = on * fixed + made * (unit_cost + day_cost * day);
  };
  define store as ent {
    properties: cap = 150 [t]; unit_cost = 30 [USD/t];
    state: stock = 40 [t] in [0, cap];
    nodes: in feed <- lime; in draws <- lime;
    behavior:
      stock <- stock + (sum(feed) - sum(draws)) * dt;
      final revenue inventory = stock * unit_cost;
  };
  define lane as ent {
    properties: cap = 90 [t/h];
    decision: ship in [0, cap] default 0.3 * cap hold 1;
    nodes: out draw <- lime; out deliver <- lime;
    behavior:
      draw.lime = ship;
      deliver.lime = ship;
  };
  define market as ent {
    properties: price = 110 [USD/t]; demand = 55 [t/h];
    nodes: in rec <- lime;
    behavior:
      sold = sum(rec);
      constraint demand: sold <= demand;
      revenue sales = sold * price;
  };
};
use kilns;
define kiln_site as env {
  contains:
    init K1 as kiln { cap = 30 [t/h]; fixed = 900 [USD/h]; };
    init K2 as kiln { cap = 25 [t/h]; fixed = 600 [USD/h]; };
    init K3 as kiln { cap = 20 [t/h]; fixed = 700 [USD/h]; unit_cost = 30 [USD/t]; };
    init ST as store;
    init L as lane;
    init MKT as market;
    connect(K1.o, ST.feed);
    connect(K2.o, ST.feed);
    connect(K3.o, ST.feed);
    connect(L.draw, ST.draws);
    connect(L.deliver, MKT.rec);
};
define kilns_day as sim {
  env: kiln_site; horizon = 24 [h]; step = 1 [h];
  objective: maximize sum(revenue) - sum(cost);
};
