## Stress model exercising every compiler path.
define core as pkg {
  define ore as res [t] { attr: grade = 0.5; };
  define power as res [kWh];
  define rock as grp { ore; };

  define pit as ent {
    properties: rate_max = 100 [t/h]; grade0 = 0.6;
    decision: dig in [0, rate_max] default 0.5 * rate_max;
    nodes: out out <- ore;
    behavior:
      out.ore = dig;
      out.ore.grade = grade0;
      cost dig = 1.5 [USD/t] * dig;
  };
  ## pass-through conveyor (wildcard ports, via-link)
  define conveyor as ent {
    properties: cap = 500 [t/h]; length = auto [km];
    nodes: in a <- *; out b <- *;
    behavior:
      b = a;
      constraint cap: sum(a) <= cap;
      cost run = 0.01 [USD/t/km] * sum(a) * length;
  };
  ## stockpile with feedback: reclaim depends on level (scan) and a delayed signal
  define stockpile as ent {
    properties: cap = 2000 [t]; k = 0.05;
    state: lvl = 500 [t] in [0, cap]; g = 0.5;
    decision: reclaim in [0, 300] default 80;
    nodes: in feed <- ore; out out <- ore;
    behavior:
      avail = lvl / dt;
      taken = min(reclaim, avail);
      out.ore = taken;
      out.ore.grade = g;
      lvl <- lvl + (feed.ore - taken) * dt;
      g <- (g * lvl + feed.ore.grade * feed.ore * dt) / max(lvl + feed.ore * dt, 1);
      lag = delay(taken, 2);
      metric lagged = lag;
  };
  ## same-type chain (level splitting) and smooth ops
  define splitter as ent {
    nodes: in i <- ore; out o <- ore;
    behavior:
      o.ore = 0.98 * i.ore;
  };
  define plant as ent {
    properties: cap = 260 [t/h]; price = 40 [USD/t];
    state: wear = 0;
    nodes: in feed <- ore;
    behavior:
      fed = smin(feed.ore, cap, 0.2);
      wear <- wear + 0.001 [1/t] * fed * dt;
      constraint grade_lo: feed.ore.grade >= 0.45;
      constraint soft(5) cap: feed.ore <= cap;
      revenue sales = fed * price * pwl(feed.ore.grade, [0.3, 0.5, 0.8], [0.5, 1.0, 1.2]) * if(hour < 22, 1, 0.9);
      final revenue salvage = -10 * wear;
  };
  ## composite: two pits behind one port (junction flattening)
  define pit_pair as ent {
    properties: r = 80;
    nodes: out out <- ore;
    contains:
      init p1 as pit { rate_max = r; grade0 = 0.55; };
      init p2 as pit { rate_max = r * 1.5; grade0 = 0.7; };
      connect(p1.out, out);
      connect(p2.out, out);
  };
  ## recycle loop through storage (cross-instance feedback -> scan)
  define recycler as ent {
    properties: frac = 0.1;
    nodes: in i <- ore; out back <- ore; out fwd <- ore;
    behavior:
      back.ore = frac * i.ore;
      fwd.ore = (1 - frac) * i.ore;
  };
};
use core;
define mine as env {
  properties: diesel = 1.2 [USD/L] input; cap_total = 600 [t/h];
  contains:
    init pp as pit_pair { r = 90; at = (-23.0, 133.0); };
    init p3 as pit { at = (-23.1, 133.1); };
    init sp as stockpile { at = (-23.05, 133.05); };
    init s1 as splitter; init s2 as splitter;
    init pl as plant { at = (-23.06, 133.2); };
    init rc as recycler;
    connect(pp.out, sp.feed) via conveyor;
    connect(p3.out, sp.feed) via conveyor { cap = 300; };
    connect(sp.out, s1.i);
    connect(s1.o, s2.i);
    connect(s2.o, rc.i);
    connect(rc.back, sp.feed);
    connect(rc.fwd, pl.feed);
  behavior:
    constraint total: sum(pit.out.ore) <= cap_total;
    metric total_dig = sum(pit.out.ore);
};
define run as sim { env: mine; horizon = 30 [h]; step = 1 [h]; objective: maximize sum(revenue) - sum(cost); };
