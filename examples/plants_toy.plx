## A toy for plants at their own step (tests/test_hierarchy.py, bench/corporate.py): a plant that makes widgets
## (a maker whose cost is high in peak hours and who can change its rate every half hour, a small store, a lane)
## and a group with a market that takes at most 60 t/h and a dearer supplier. Nothing declares the plant: the
## maker's half-hour hold makes it a unit of 30-minute steps, the store is too small to decouple it, the lane draws
## from the store. Hand-worked optimum: USD 37,100.
define widgets as pkg {
  define widget as res [t] { rate_unit = "t/h"; };
  define maker as ent {
    properties: cap = 50 [t/h]; unit_cost = 40 [USD/t]; peak_cost = 90 [USD/t];
    decision: rate in [0, cap] default 0.5 * cap hold 30 [min];
    nodes: out o <- widget;
    behavior:
      o.widget = rate;
      cost making = rate * (unit_cost + (peak_cost - unit_cost) * env.peak);
  };
  define store as ent {
    properties: cap = 200 [t]; unit_cost = 40 [USD/t];
    state: stock = 50 [t] in [0, cap];
    nodes: in feed <- widget; in draws <- widget;
    behavior:
      stock <- stock + (sum(feed) - sum(draws)) * dt;
      final revenue inventory = stock * unit_cost;
  };
  define lane as ent {
    properties: cap = 80 [t/h];
    decision: ship in [0, cap] default 0.5 * cap hold 1;
    nodes: out draw <- widget; out deliver <- widget;
    behavior:
      draw.widget = ship;
      deliver.widget = ship;
  };
  define market as ent {
    properties: price = 100 [USD/t]; demand = 60 [t/h];
    nodes: in rec <- widget;
    behavior:
      sold = sum(rec);
      constraint demand: sold <= demand;
      revenue sales = sold * price;
  };
  define supplier as ent {
    properties: cap = 50 [t/h]; unit_cost = 70 [USD/t];
    decision: rate in [0, cap] default 0.5 * cap hold 1;
    nodes: out o <- widget;
    behavior:
      o.widget = rate;
      cost buying = rate * unit_cost;
  };
};
use widgets;
define toy as env {
  properties: peak = 0 input;
  contains:
    init P_MAKER as maker;
    init P_STORE as store;
    init L_P as lane;
    init MKT as market;
    init SUP as supplier;
    connect(P_MAKER.o, P_STORE.feed);
    connect(L_P.draw, P_STORE.draws);
    connect(L_P.deliver, MKT.rec);
    connect(SUP.o, MKT.rec);
  behavior:
    metric made = sum(maker.rate) [t/h];
};
define toy_sim as sim {
  env: toy; horizon = 12 [h]; step = 1 [h];
  objective: maximize sum(revenue) - sum(cost);
};
