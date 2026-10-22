## PLYXIS standard library - resource-agnostic building blocks.
## Ports declared with '*' adapt to whatever is connected (specialised at elaboration).

define core as pkg {

  ## Capacity-limited transport link.
  define transport as ent {
    geom: line; layer = "links";
    properties: cap = 1000000; length = auto [km];
    nodes: in a <- *; out b <- *;
    behavior:
      constraint capacity: sum(a) <= cap;
      b = a;
  };

  ## Pass-through capacity check (e.g. a valve, a meter, a conveyor transfer point).
  define limiter as ent {
    geom: point; layer = "links";
    properties: cap = 1000000;
    nodes: in a <- *; out b <- *;
    behavior:
      constraint capacity: sum(a) <= cap;
      b = a;
  };

  ## Market sink: everything delivered is sold at 'price' (per canonical unit).
  define market as ent {
    geom: point; layer = "markets"; icon = "market";
    properties: price = 0; cap = 1000000000;
    nodes: in rec <- *;
    behavior:
      delivered = sum(rec);
      constraint demand: delivered <= cap;
      revenue sales = delivered * price;
  };

  ## Disposal sink with a unit cost.
  define disposal as ent {
    geom: point; layer = "markets"; icon = "sink";
    properties: cost_unit = 0; cap = 1000000000;
    nodes: in rec <- *;
    behavior:
      received = sum(rec);
      constraint capacity: received <= cap;
      cost disposal = received * cost_unit;
  };
};
