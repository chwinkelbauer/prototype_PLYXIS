## =====================================================================================
##  AQUILA STEEL with the Duisburg works in detail - one twin, two time scales
##  The group of scenarios/aquila (13 sites, the same register), but Duisburg (DE) is modelled machine by machine:
##  three blast furnaces, the torpedo pool and pig caster, three converter vessels heat by heat, the ladle furnace,
##  two slab casters with their tundish sequences, a slab yard with hot slabs, two reheating furnaces, the hot strip
##  mill with its roll changes, the galvanising line, the works' process-gas network and power plant.
##
##  Nothing declares a plant. The works' blueprints state their durations in time (tap to tap 45 minutes, a tundish
##  ten hours, set points held for 15 or 30 minutes); plyxis/lang/hierarchy.py finds the units that need a finer step
##  than the simulation's hour, grows a plant around them through the small buffers between them and cuts it at the
##  large ones; plyxis/optim/corporate.py coordinates the plant and the group by prices (column generation).
## =====================================================================================
use steel;

define aquila_works as env {
  origin = (30.0, 10.0);
  properties:
    hrc_eu = 680 [USD/t] input;  hrc_na = 820 [USD/t] input;  hrc_sa = 690 [USD/t] input;
    hrc_in = 600 [USD/t] input;  hrc_me = 640 [USD/t] input;
    scrap_eu = 395 [USD/t] input;  scrap_us = 405 [USD/t] input;  scrap_br = 360 [USD/t] input;
    scrap_in = 425 [USD/t] input;  scrap_me = 395 [USD/t] input;
    hbi_price = 360 [USD/t] input;
    power_nl = 85 [USD/MWh] input;  power_de = 90 [USD/MWh] input;  power_fr = 70 [USD/MWh] input;
    power_es = 65 [USD/MWh] input;  power_pl = 100 [USD/MWh] input;  power_miso = 40 [USD/MWh] input;
    power_ercot = 35 [USD/MWh] input;  power_mx = 60 [USD/MWh] input;  power_br = 55 [USD/MWh] input;
    power_in = 70 [USD/MWh] input;  power_sa = 45 [USD/MWh] input;
    gas_ttf = 10.5 [USD/GJ] input;  gas_hh = 2.6 [USD/GJ] input;  gas_br = 9 [USD/GJ] input;
    gas_in = 11.5 [USD/GJ] input;  gas_sa = 2.0 [USD/GJ] input;
    eua_price = 72 [USD/tCO2e] input;
    carbon_price = 30 [USD/tCO2e];

  contains:
    # ------------------------------------------------------------- the group (scenarios/aquila), Duisburg's
    # furnaces, converter shop, caster, strip mill and slab yard left out: they are modelled below
    for s in table("../aquila/sites.csv") {
      init {s.name}_GRID as grid { at = (s.lat, s.lon); tag = concat(s.tag, ".GRID"); label = concat(s.label, " grid connection");
          max_demand = s.max_demand [MW]; grid_co2 = s.grid_co2 [kgCO2e/kWh];
          w_nl = s.w_nl; w_de = s.w_de; w_fr = s.w_fr; w_es = s.w_es; w_pl = s.w_pl; w_miso = s.w_miso;
          w_ercot = s.w_ercot; w_mx = s.w_mx; w_br = s.w_br; w_in = s.w_in; w_sa = s.w_sa; };
      init {s.name}_GAS as gas_supply { at = (s.lat, s.lon); tag = concat(s.tag, ".GAS"); label = concat(s.label, " gas supply");
          cap = s.gas_cap [GJ/h]; ets_share = s.ets;
          g_ttf = s.g_ttf; g_hh = s.g_hh; g_br = s.g_br; g_in = s.g_in; g_sa = s.g_sa; };
      init {s.name}_SVC as site_services { at = (s.lat, s.lon); tag = concat(s.tag, ".SVC"); label = s.label;
          labour = s.labour [USD/h]; overhead = s.overhead [USD/h]; };
    }
    for m in table("../aquila/scrap_markets.csv") {
      init {m.name} as scrap_market { at = (m.lat, m.lon); tag = m.tag; label = m.label; avail = m.avail [t/h];
          s_eu = m.s_eu; s_us = m.s_us; s_br = m.s_br; s_in = m.s_in; s_me = m.s_me; };
    }
    for m in table("../aquila/markets.csv") {
      init {m.name} as steel_market { at = (m.lat, m.lon); tag = m.tag; label = m.label;
          contract = m.contract [t/h]; spot = m.spot [t/h]; beta = m.beta; premium = m.premium [USD/t];
          r_eu = m.r_eu; r_na = m.r_na; r_sa = m.r_sa; r_in = m.r_in; r_me = m.r_me; };
    }
    for y in table("../aquila/ore_yards.csv") {
      init {y.name} as ore_yard { at = (y.lat, y.lon); tag = y.tag; label = y.label; cap = y.cap [t];
          delivery = y.delivery [t/h]; unit_cost = y.unit_cost [USD/t]; stock = y.stock [t]; reserve_t = y.reserve [t]; };
    }
    for y in table("../aquila/coke_yards.csv") {
      init {y.name} as coke_yard { at = (y.lat, y.lon); tag = y.tag; label = y.label; cap = y.cap [t];
          delivery = y.delivery [t/h]; unit_cost = y.unit_cost [USD/t]; stock = y.stock [t]; reserve_t = y.reserve [t]; };
    }
    for y in table("../aquila/scrap_yards.csv") {
      init {y.name} as scrap_yard { at = (y.lat, y.lon); tag = y.tag; label = y.label; cap = y.cap [t]; stock = y.stock [t];
          buy_max = y.buy_max [t/h]; unit_cost = y.unit_cost [USD/t]; buy_cu = y.buy_cu; cu_avg = y.buy_cu;
          reserve_t = y.stock [t]; };
      connect({y.name}.purchase, {y.market}.requests);
    }
    for y in table("../aquila/dri_yards.csv") {
      init {y.name} as dri_yard { at = (y.lat, y.lon); tag = y.tag; label = y.label; cap = y.cap [t]; stock = y.stock [t];
          buy_max = y.buy_max [t/h]; unit_cost = y.unit_cost [USD/t]; reserve_t = y.reserve [t]; };
    }
    for y in table("../aquila/product_yards.csv") {
      if y.name != "DE_SLABS" {
        init {y.name} as product_yard { at = (y.lat, y.lon); tag = y.tag; label = y.label; cap = y.cap [t]; stock = y.stock [t];
            unit_cost = y.unit_cost [USD/t]; reserve_t = y.reserve [t]; stock_max = y.stock_max [t]; };
      }
    }
    for b in table("../aquila/blast_furnaces.csv") {
      if b.site != "DE" {
        init {b.name} as blast_furnace { at = (b.lat, b.lon); tag = b.tag; label = b.label; rate_max = b.rate_max [t/h];
            coke_rate = b.coke_rate; ets_share = b.ets; };
        connect({b.name}.hm, {b.site}_BOF.hm_in);
        connect({b.name}.ore_req, {b.site}_ORE.draws);
        connect({b.name}.coke_req, {b.site}_COKE.draws);
        connect({b.name}.pw, {b.site}_GRID.requests);
      }
    }
    for d in table("../aquila/dri_plants.csv") {
      init {d.name} as dri_plant { at = (d.lat, d.lon); tag = d.tag; label = d.label; cap = d.cap [t/h]; };
      connect({d.name}.dri_out, {d.yard}.feed);
      connect({d.name}.ore_req, {d.pellets}.draws);
      connect({d.name}.gas_req, {d.site}_GAS.requests);
      connect({d.name}.pw, {d.site}_GRID.requests);
    }
    for c in table("../aquila/casters.csv") {
      if c.site != "DE" {
        init {c.name} as slab_caster { at = (c.lat, c.lon); tag = c.tag; label = c.label; cap = c.cap [t/h]; cu_max = c.cu_max; };
        connect({c.name}.cast, {c.yard}.feed);
        connect({c.name}.pw, {c.site}_GRID.requests);
      }
    }
    for c in table("../aquila/billet_casters.csv") {
      init {c.name} as billet_caster { at = (c.lat, c.lon); tag = c.tag; label = c.label; cap = c.cap [t/h]; cu_max = c.cu_max; };
      connect({c.name}.cast, {c.yard}.feed);
      connect({c.name}.pw, {c.site}_GRID.requests);
    }
    for b in table("../aquila/converter_shops.csv") {
      if b.site != "DE" {
        init {b.name} as bof_shop { at = (b.lat, b.lon); tag = b.tag; label = b.label; cap = b.cap [t/h]; ets_share = b.ets; };
        connect({b.name}.scrap_req, {b.yard}.draws);
        connect({b.yard}.charge, {b.name}.scrap_in);
        connect({b.name}.liq, {b.caster}.liq);
        connect({b.name}.pw, {b.site}_GRID.requests);
      }
    }
    for e in table("../aquila/arc_furnaces.csv") {
      init {e.name} as eaf { at = (e.lat, e.lon); tag = e.tag; label = e.label; cap = e.cap [t/h]; kwh_t = e.kwh_t [kWh/t];
          dri_max = e.dri_max; ets_share = e.ets; };
      connect({e.name}.scrap_req, {e.yard}.draws);
      connect({e.yard}.charge, {e.name}.scrap_in);
      connect({e.name}.dri_req, {e.hbi}.draws);
      connect({e.name}.liq, {e.caster}.liq);
      connect({e.name}.pw, {e.site}_GRID.requests);
    }
    for h in table("../aquila/strip_mills.csv") {
      if h.site != "DE" {
        init {h.name} as hot_strip_mill { at = (h.lat, h.lon); tag = h.tag; label = h.label; cap = h.cap [t/h]; };
        connect({h.name}.slab_req, {h.slabs}.draws);
        connect({h.name}.coil, {h.coil}.feed);
        connect({h.name}.pw, {h.site}_GRID.requests);
        connect({h.name}.gas_req, {h.site}_GAS.requests);
      }
    }
    for g in table("../aquila/coating_lines.csv") {
      init {g.name} as coating_line { at = (g.lat, g.lon); tag = g.tag; label = g.label; cap = g.cap [t/h]; };
      connect({g.name}.coil_req, {g.coil}.draws);
      connect({g.name}.product, {g.yard}.feed);
      connect({g.name}.pw, {g.site}_GRID.requests);
      connect({g.name}.gas_req, {g.site}_GAS.requests);
    }
    for r in table("../aquila/bar_mills.csv") {
      init {r.name} as long_mill { at = (r.lat, r.lon); tag = r.tag; label = r.label; cap = r.cap [t/h]; };
      connect({r.name}.billet_req, {r.billets}.draws);
      connect({r.name}.product, {r.yard}.feed);
      connect({r.name}.pw, {r.site}_GRID.requests);
      connect({r.name}.gas_req, {r.site}_GAS.requests);
    }

    # ------------------------------------------------------------- Duisburg (DE), machine by machine
    init DE_BF1 as bf_works { at = (51.491, 6.742); tag = "AQ.DE.BF1"; label = "Duisburg (DE) blast furnace 1"; rate_max = 460 [t/h]; ets_share = 0.35; };
    init DE_BF2 as bf_works { at = (51.494, 6.746); tag = "AQ.DE.BF2"; label = "Duisburg (DE) blast furnace 2"; rate_max = 420 [t/h]; ets_share = 0.35; };
    init DE_BF3 as bf_works { at = (51.497, 6.750); tag = "AQ.DE.BF3"; label = "Duisburg (DE) blast furnace 3"; rate_max = 340 [t/h]; ets_share = 0.35; };
    init DE_HM as hot_metal_pool { at = (51.490, 6.746); tag = "AQ.DE.HM"; label = "Duisburg (DE) torpedo cars and mixer"; };
    init DE_PIG as pig_caster { at = (51.489, 6.744); tag = "AQ.DE.PIG"; label = "Duisburg (DE) pig casting machine"; };
    init DE_BOF1 as converter { at = (51.4865, 6.7470); tag = "AQ.DE.BOF1"; label = "Duisburg (DE) converter 1"; ets_share = 0.35; };
    init DE_BOF2 as converter { at = (51.4862, 6.7480); tag = "AQ.DE.BOF2"; label = "Duisburg (DE) converter 2"; ets_share = 0.35; };
    init DE_BOF3 as converter { at = (51.4859, 6.7490); tag = "AQ.DE.BOF3"; label = "Duisburg (DE) converter 3"; ets_share = 0.35; };
    init DE_LADLE as ladle_pool { at = (51.4855, 6.7495); tag = "AQ.DE.LF"; label = "Duisburg (DE) ladle furnace and degasser"; };
    init DE_CC1 as caster_works { at = (51.4845, 6.7505); tag = "AQ.DE.CC1"; label = "Duisburg (DE) slab caster 1"; cap = 850 [t/h]; };
    init DE_CC2 as caster_works { at = (51.4838, 6.7515); tag = "AQ.DE.CC2"; label = "Duisburg (DE) slab caster 2"; cap = 750 [t/h]; };
    init DE_SLABS as slab_store { at = (51.482, 6.754); tag = "AQ.DE.SLABS"; label = "Duisburg (DE) slab yard"; cap = 268800 [t];
        stock = 96000 [t]; unit_cost = 506 [USD/t]; reserve_t = 96000 [t]; stock_max = 105600 [t]; };
    init DE_RF1 as reheat_furnace { at = (51.4810, 6.7560); tag = "AQ.DE.RF1"; label = "Duisburg (DE) reheating furnace 1"; };
    init DE_RF2 as reheat_furnace { at = (51.4806, 6.7568); tag = "AQ.DE.RF2"; label = "Duisburg (DE) reheating furnace 2"; };
    init DE_HSM as hsm_works { at = (51.480, 6.758); tag = "AQ.DE.HSM"; label = "Duisburg (DE) hot strip mill"; cap = 1350 [t/h]; };
    init DE_GASNET as works_gas { at = (51.493, 6.735); tag = "AQ.DE.PGAS"; label = "Duisburg (DE) process gas network and holder"; };
    init DE_PP as power_plant { at = (51.495, 6.730); tag = "AQ.DE.PP"; label = "Duisburg (DE) power plant"; };

    connect(DE_BF1.hm, DE_HM.feed);  connect(DE_BF2.hm, DE_HM.feed);  connect(DE_BF3.hm, DE_HM.feed);
    connect(DE_BF1.ore_req, DE_ORE.draws);  connect(DE_BF2.ore_req, DE_ORE.draws);  connect(DE_BF3.ore_req, DE_ORE.draws);
    connect(DE_BF1.coke_req, DE_COKE.draws);  connect(DE_BF2.coke_req, DE_COKE.draws);  connect(DE_BF3.coke_req, DE_COKE.draws);
    connect(DE_BF1.pw, DE_GRID.requests);  connect(DE_BF2.pw, DE_GRID.requests);  connect(DE_BF3.pw, DE_GRID.requests);
    connect(DE_BF1.bfg, DE_GASNET.supply);  connect(DE_BF2.bfg, DE_GASNET.supply);  connect(DE_BF3.bfg, DE_GASNET.supply);
    connect(DE_PIG.draw, DE_HM.draws);
    connect(DE_BOF1.hm_req, DE_HM.draws);  connect(DE_BOF2.hm_req, DE_HM.draws);  connect(DE_BOF3.hm_req, DE_HM.draws);
    connect(DE_BOF1.scrap_req, DE_BOFSCRAP.draws);  connect(DE_BOF2.scrap_req, DE_BOFSCRAP.draws);  connect(DE_BOF3.scrap_req, DE_BOFSCRAP.draws);
    connect(DE_BOF1.liq, DE_LADLE.feed);  connect(DE_BOF2.liq, DE_LADLE.feed);  connect(DE_BOF3.liq, DE_LADLE.feed);
    connect(DE_BOF1.pw, DE_GRID.requests);  connect(DE_BOF2.pw, DE_GRID.requests);  connect(DE_BOF3.pw, DE_GRID.requests);
    connect(DE_BOF1.bofg, DE_GASNET.supply);  connect(DE_BOF2.bofg, DE_GASNET.supply);  connect(DE_BOF3.bofg, DE_GASNET.supply);
    connect(DE_LADLE.pw, DE_GRID.requests);
    connect(DE_CC1.liq_req, DE_LADLE.draws);  connect(DE_CC2.liq_req, DE_LADLE.draws);
    connect(DE_CC1.cast, DE_SLABS.feed);  connect(DE_CC2.cast, DE_SLABS.feed);
    connect(DE_CC1.pw, DE_GRID.requests);  connect(DE_CC2.pw, DE_GRID.requests);
    connect(DE_RF1.hot_req, DE_SLABS.draws_hot);  connect(DE_RF2.hot_req, DE_SLABS.draws_hot);
    connect(DE_RF1.cold_req, DE_SLABS.draws);  connect(DE_RF2.cold_req, DE_SLABS.draws);
    connect(DE_RF1.heated, DE_HSM.slabs);  connect(DE_RF2.heated, DE_HSM.slabs);
    connect(DE_RF1.pg_req, DE_GASNET.requests);  connect(DE_RF2.pg_req, DE_GASNET.requests);
    connect(DE_RF1.ng_req, DE_GAS.requests);  connect(DE_RF2.ng_req, DE_GAS.requests);
    connect(DE_RF1.pw, DE_GRID.requests);  connect(DE_RF2.pw, DE_GRID.requests);
    connect(DE_HSM.coil, DE_COIL.feed);
    connect(DE_HSM.pw, DE_GRID.requests);
    connect(DE_GASNET.surplus, DE_PP.fuel);
    connect(DE_PP.pw, DE_GRID.requests);

    # ------------------------------------------------------------- sales and transfers
    for l in table("../aquila/lanes_hrc.csv") {
      init {l.name} as lane_hrc { label = l.label; path = [(l.slat, l.slon), (l.dlat, l.dlon)]; cap = l.cap [t/h]; freight = l.freight [USD/t]; };
      connect({l.name}.draw, {l.src}.draws);
      connect({l.name}.deliver, {l.market}.rec);
    }
    for l in table("../aquila/lanes_coated.csv") {
      init {l.name} as lane_coated { label = l.label; path = [(l.slat, l.slon), (l.dlat, l.dlon)]; cap = l.cap [t/h]; freight = l.freight [USD/t]; };
      connect({l.name}.draw, {l.src}.draws);
      connect({l.name}.deliver, {l.market}.rec);
    }
    for l in table("../aquila/lanes_long.csv") {
      init {l.name} as lane_long { label = l.label; path = [(l.slat, l.slon), (l.dlat, l.dlon)]; cap = l.cap [t/h]; freight = l.freight [USD/t]; };
      connect({l.name}.draw, {l.src}.draws);
      connect({l.name}.deliver, {l.market}.rec);
    }
    for l in table("../aquila/lanes_slab.csv") {
      init {l.name} as lane_slab { label = l.label; path = [(l.slat, l.slon), (l.dlat, l.dlon)]; cap = l.cap [t/h]; freight = l.freight [USD/t]; };
      connect({l.name}.draw, {l.src}.draws);
      connect({l.name}.deliver, {l.market}.rec);
    }
    for t in table("../aquila/transfers_slab.csv") {
      init {t.name} as lane_slab { label = t.label; path = [(t.slat, t.slon), (t.dlat, t.dlon)]; cap = t.cap [t/h]; freight = t.freight [USD/t]; };
      connect({t.name}.draw, {t.src}.draws);
      if t.dst == "DE_SLABS" {
        connect({t.name}.deliver, DE_SLABS.feed_cold);
      } else {
        connect({t.name}.deliver, {t.dst}.feed);
      }
    }
    for t in table("../aquila/transfers_dri.csv") {
      init {t.name} as lane_dri { label = t.label; path = [(t.slat, t.slon), (t.dlat, t.dlon)]; cap = t.cap [t/h]; freight = t.freight [USD/t]; };
      connect({t.name}.draw, {t.src}.draws);
      connect({t.name}.deliver, {t.dst}.feed);
    }

  behavior:
    metric crude_steel = sum(bof_shop.made) + sum(eaf.made) + sum(converter.made) [t/h];
    metric shipments = sum(steel_market.delivered) [t/h];
    metric group_power = sum(grid.load) [MW];
    metric hot_metal_made = sum(blast_furnace.made) + sum(bf_works.made) [t/h];
    metric eaf_steel = sum(eaf.made) [t/h];
};

## Corporate operations plan: four days, hourly (the works' plant at the step its units need).
define corp96 as sim {
  env: aquila_works;
  horizon = 96 [h];
  step = 1 [h];
  objective:
    maximize sum(revenue) - sum(cost) - carbon_price * sum(emission);
  solver:
    route_hold = 8;
};

## The same twin as one program at 15-minute steps everywhere (the reference the decomposition is measured against).
define corp96_fine as sim {
  env: aquila_works;
  horizon = 96 [h];
  step = 15 [min];
  objective:
    maximize sum(revenue) - sum(cost) - carbon_price * sum(emission);
  solver:
    route_hold = 32;
};
