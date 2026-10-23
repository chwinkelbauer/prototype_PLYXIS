## =====================================================================================
##  RIDGEBACK PIT - open-pit copper mine (synthetic). Same engine as the oil field:
##  only the blueprints differ (stdlib 'mining'). Dig faces with block-model grades,
##  a shared haul fleet, ROM stockpile blending, concentrator grade window, waste dumps.
## =====================================================================================
use mining;

define ridgeback as env {
  origin = (-23.36, 119.73);
  properties:
    metal_price = 9.1 [USD/kg] input;        ## copper, paid metal
    power_price = 0.11 [USD/kWh] input;
  contains:
    for f in table("faces.csv") {
      init {f.name} as dig_face { at = (f.lat, f.lon); grade = f.grade; strip = f.strip;
                                  rate_max = f.rate [t/h]; reserve = f.reserve [t]; label = f.label; };
    }
    init FLEET as haul_fleet { at = (-23.352, 119.745); n_trucks = 26; label = "Haul fleet"; };
    init ROM as stockpile { at = (-23.358, 119.752); label = "ROM pad"; };
    init MILL as concentrator { at = (-23.361, 119.760); label = "Concentrator"; };
    init DUMP_N as dump { at = (-23.344, 119.722); label = "North waste dump"; };
    init DUMP_S as dump { at = (-23.372, 119.728); label = "South waste dump"; };
    init PORT as conc_sales { at = (-23.365, 119.772); label = "Rail load-out"; };

    for f in table("faces.csv") {
      connect({f.name}.ore_out, ROM.feed) via haul_route { distance = f.d_rom [km]; } as R_{f.name}_ROM;
      connect({f.name}.waste_out, DUMP_N.rec) via haul_route { distance = f.d_dn [km]; } as R_{f.name}_DN;
      connect({f.name}.waste_out, DUMP_S.rec) via haul_route { distance = f.d_ds [km]; } as R_{f.name}_DS;
      connect(FLEET.pool, R_{f.name}_ROM.trucks);
      connect(FLEET.pool, R_{f.name}_DN.trucks);
      connect(FLEET.pool, R_{f.name}_DS.trucks);
    }
    connect(ROM.out, MILL.feed);
    connect(MILL.conc, PORT.rec);
};

define shift_plan as sim {
  env: ridgeback; horizon = 48 [h]; step = 1 [h];
  objective: maximize sum(revenue) - sum(cost);
  solver: route_hold = 12;
};
