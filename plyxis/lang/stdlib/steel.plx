## PLYXIS standard library - steel: integrated (blast furnace -> converter) and electric (direct reduction ->
## electric arc furnace) steelmaking, casting, hot rolling, coating, long products, raw-material and product
## yards, merchant metallics, power and gas supply, regional scrap and steel markets, sales and transfer lanes.
## Same engine as oil & gas and mining: only the entity definitions differ.
##
## Corporate modelling conventions:
## * flows are t/h, stocks t, prices USD/t (energy USD/MWh, USD/GJ); a material a unit consumes from a yard is a
##   request into the yard (`draws`), so a yard serves many consumers without routing decisions;
## * a site is connected to the regional price of its market by weights (`w_*`, `g_*`, `s_*`, `r_*` = 1 for its
##   zone): one blueprint for every grid, gas hub, scrap market and steel market, vectorised over all of them;
## * every inventory is valued at cost as a terminal value, never at its sales price: bought material at its
##   price, made material at its variable cost, capped a little below its net realisable value (IAS 2). Fixed
##   costs are paid whatever the plan does, so making stock creates no value and only selling earns a margin;
## * yards hold their stock target (`reserve_t`): coil waits for its vessel, slabs for their rolling slot, and the
##   working capital is set by finance, not by the plan. Without it a plan sells the yards down instead of producing
##   (it saves the production's carbon and whatever the true cost is above the book value) - a gain that lasts only
##   until the stock is rebuilt. The penalty grows with the square of the shortfall (per t of target and hour): with a
##   kink at the target, where every yard starts, any extra shipment paid the full penalty until mill, caster and
##   furnace moved together, and the search left the first day's lanes near their start. The plan's end value of a
##   shortfall (`value_to_go`, planning only) is smooth for the same reason: quadratic up to 10 % of the target, then
##   `restock` per t (as a hinge, the first tonne short at the end already cost the full `restock`);
## * EU sites pay for their emissions (`ets_share` = share not covered by free allocation, at the EUA price) - a
##   cash cost; the internal carbon price of the objective is a decision price, reported separately.

define steel as pkg {

  define iron_ore  as res [t] { rate_unit = "t/h"; };            ## pellets and sinter feed, as charged
  define coke      as res [t] { rate_unit = "t/h"; };
  define hot_metal as res [t] { rate_unit = "t/h"; };            ## liquid iron from the blast furnace
  define scrap     as res [t] { rate_unit = "t/h"; attr: cu = 0.15; };   ## residual copper, %
  define dri       as res [t] { rate_unit = "t/h"; };            ## direct reduced iron / hot briquetted iron
  define steel_liq as res [t] { rate_unit = "t/h"; attr: cu = 0.05; };
  define slab      as res [t] { rate_unit = "t/h"; };
  define billet    as res [t] { rate_unit = "t/h"; };
  define hrc       as res [t] { rate_unit = "t/h"; };            ## hot-rolled coil
  define coated    as res [t] { rate_unit = "t/h"; };            ## cold-rolled and galvanised coil
  define long_prod as res [t] { rate_unit = "t/h"; };            ## rebar, wire rod, sections
  define power     as res [kWh] { rate_unit = "MW"; };
  define fuel_gas  as res [GJ] { rate_unit = "GJ/h"; };

  # ================================================================== site utilities and markets
  ## Grid connection of a site: what its units draw, at the price of its power zone (one weight = 1).
  define grid as ent {
    geom: point; layer = "utilities"; icon = "power";
    properties:
      max_demand = 400 [MW];                     ## connection agreement
      overrun = 1500 [USD/MWh];                  ## what the grid operator charges for every MWh above it
      grid_co2 = 0.35 [kgCO2e/kWh];              ## scope 2, reported
      w_nl = 0; w_de = 0; w_fr = 0; w_es = 0; w_pl = 0; w_miso = 0; w_ercot = 0; w_mx = 0; w_br = 0; w_in = 0; w_sa = 0;
    nodes: in requests <- power;
    telemetry: load <- "GRID.MW" [MW];
    behavior:
      load = requests.power;
      price = w_nl * env.power_nl + w_de * env.power_de + w_fr * env.power_fr + w_es * env.power_es
            + w_pl * env.power_pl + w_miso * env.power_miso + w_ercot * env.power_ercot + w_mx * env.power_mx
            + w_br * env.power_br + w_in * env.power_in + w_sa * env.power_sa;
      cost power = load * price;
      cost overrun_charge = pos(load - max_demand) * overrun;
      emission grid = load * grid_co2;
      metric site_power = load [MW];
  };

  ## Natural gas supply of a site (reheating furnaces, direct reduction), at its hub's price.
  define gas_supply as ent {
    geom: point; layer = "utilities"; icon = "gas";
    properties:
      cap = 4000 [GJ/h];                         ## contracted capacity (a curtailment lowers it)
      ets_share = 0;
      g_ttf = 0; g_hh = 0; g_br = 0; g_in = 0; g_sa = 0;
    nodes: in requests <- fuel_gas;
    behavior:
      use = requests.fuel_gas;
      price = g_ttf * env.gas_ttf + g_hh * env.gas_hh + g_br * env.gas_br + g_in * env.gas_in + g_sa * env.gas_sa;
      constraint contract: use <= cap;
      cost gas = use * price;
      emission combustion = use * 56 [kgCO2e/GJ];
      cost ets = use * 56 [kgCO2e/GJ] * ets_share * env.eua_price;
  };

  ## Site overheads: crews, maintenance, administration - paid whatever the site produces.
  define site_services as ent {
    geom: point; layer = "sites"; icon = "site";
    properties: labour = 20000 [USD/h]; overhead = 8000 [USD/h];
    behavior:
      cost labour = labour;
      cost overhead = overhead;
  };

  ## Regional scrap market: what the region's yards buy, within what the region's dealers can supply.
  define scrap_market as ent {
    geom: point; layer = "markets"; icon = "scrap";
    properties:
      avail = 1500 [t/h];                        ## what the region's dealers deliver at the index price
      s_eu = 0; s_us = 0; s_br = 0; s_in = 0; s_me = 0;
    nodes: in requests <- scrap;
    behavior:
      bought = requests.scrap;
      price = s_eu * env.scrap_eu + s_us * env.scrap_us + s_br * env.scrap_br + s_in * env.scrap_in + s_me * env.scrap_me;
      constraint supply: bought <= avail;
      cost scrap = bought * price;
      metric scrap_bought = bought [t/h];
  };

  ## Regional steel market for one product. Contract customers call off `contract` per hour: what falls behind is
  ## a backlog, delivered later at the contract price, and a backlog left at the end costs liquidated damages
  ## (`short_cost` per t). They take nothing ahead of their call-offs; what goes beyond is sold spot, up to `spot`
  ## per hour, `spot_disc` below the index. Price = the region's HRC index x `beta` + `premium`.
  ## `late_delivery` prices every tonne-hour of backlog for the plan only (customers waiting; not in the accounts):
  ## without it shipping now or tomorrow scores the same, and every rolling plan put deliveries off to "later"
  ## (at 0.5 USD/t/h one seed's first plan still left European coil 20 % short).
  define steel_market as ent {
    geom: point; layer = "markets"; icon = "market";
    properties:
      contract = 200 [t/h]; spot = 50 [t/h]; spot_disc = 40 [USD/t]; beta = 1.0; premium = 0 [USD/t];
      short_cost = 80 [USD/t]; late_cost = 2 [USD/t/h];
      r_eu = 0; r_na = 0; r_sa = 0; r_in = 0; r_me = 0;
    state: backlog = 0 [t];
    nodes: in rec <- *;
    telemetry: delivered <- "MKT.TPH" [t/h];
    behavior:
      delivered = sum(rec);
      price = (r_eu * env.hrc_eu + r_na * env.hrc_na + r_sa * env.hrc_sa + r_in * env.hrc_in + r_me * env.hrc_me) * beta + premium;
      spot_sold = pos(delivered - contract - pos(backlog) / dt);
      constraint depth: spot_sold <= spot;
      backlog <- pos(backlog + (contract - delivered) * dt);
      revenue sales = delivered * price - spot_sold * spot_disc;
      final cost shortfall = pos(backlog) * short_cost;
      cost late_delivery = pos(backlog) * late_cost;
      metric shipped = delivered [t/h];
  };

  # ================================================================== yards
  ## Port stockyard of contracted raw material (ore and pellets, or coke): vessels discharge `delivery` on average,
  ## the furnaces' consumption is drawn. Purchases are paid on delivery; what is left is valued at cost.
  define ore_yard as ent {
    geom: point; layer = "yards"; icon = "stockpile";
    properties: cap = 2000000 [t]; delivery = 1000 [t/h]; unit_cost = 120 [USD/t];
      reserve_t = 0 [t]; reserve_price = 0.5 [USD/t/h];    ## working stock for a vessel delay
    state: stock = 800000 [t] in [0, cap];
    nodes: in draws <- iron_ore;
    telemetry: stock <- "YARD.STOCK" [t];
    behavior:
      stock <- stock + (delivery - draws.iron_ore) * dt;
      cost purchases = delivery * unit_cost;
      final revenue inventory = unit_cost * stock;
      constraint soft(reserve_price) working_stock: stock >= reserve_t;
  };
  define coke_yard as ent {
    geom: point; layer = "yards"; icon = "stockpile";
    properties: cap = 400000 [t]; delivery = 250 [t/h]; unit_cost = 330 [USD/t];
      reserve_t = 0 [t]; reserve_price = 0.5 [USD/t/h];
    state: stock = 150000 [t] in [0, cap];
    nodes: in draws <- coke;
    telemetry: stock <- "YARD.STOCK" [t];
    behavior:
      stock <- stock + (delivery - draws.coke) * dt;
      cost purchases = delivery * unit_cost;
      final revenue inventory = unit_cost * stock;
      constraint soft(reserve_price) working_stock: stock >= reserve_t;
  };

  ## Scrap bay of a melt shop: buys from the regional market, charges the furnace. The copper of the bay is the
  ## blend of what was bought (`buy_cu`, measured on samples by the lab) and what was there.
  define scrap_yard as ent {
    geom: point; layer = "yards"; icon = "scrap";
    properties: cap = 120000 [t]; buy_max = 600 [t/h]; unit_cost = 390 [USD/t];
      buy_cu = 0.16 calibrate in [0.05, 0.45];
      reserve_t = 0 [t]; reserve_price = 30 [USD/t/h]; restock = 200 [USD/t];   ## stock target, as the product yards
    state: stock = 40000 [t] in [0, cap]; cu_avg = 0.16;
    decision: buy in [0, buy_max] default 0.3 * buy_max hold 8;
    nodes: out purchase <- scrap; in draws <- scrap; out charge <- scrap;
    telemetry: stock <- "YARD.STOCK" [t]; buy_cu <- "LAB.CU" { test = true; };
    behavior:
      purchase.scrap = buy;
      taken = sat(draws.scrap, stock / dt + buy);
      charge.scrap = taken;
      charge.scrap.cu = cu_avg;
      stock <- stock + (buy - taken) * dt;
      cu_avg <- (cu_avg * pos(stock) + buy_cu * buy * dt) / max(pos(stock) + buy * dt, 1 [t]);
      final revenue inventory = unit_cost * stock;
      ## without a target a plan melted the bay down instead of buying (USD 11 M less scrap bought in 96 h)
      constraint soft(reserve_price) working_stock: (reserve_t - stock) * pos(reserve_t - stock) / max(reserve_t, 1 [t]) <= 0 [t];
      final cost value_to_go = restock * (min(pos(reserve_t - stock), max(0.1 * reserve_t, 1 [t])) * min(pos(reserve_t - stock), max(0.1 * reserve_t, 1 [t]))
                                          / (2 * max(0.1 * reserve_t, 1 [t])) + pos(reserve_t - stock - max(0.1 * reserve_t, 1 [t])));
  };

  ## Pile of direct reduced iron / HBI: fed by the site's DRI plant or transfers, drawn by the furnaces; merchant
  ## HBI can be bought on top.
  define dri_yard as ent {
    geom: point; layer = "yards"; icon = "stockpile";
    properties: cap = 150000 [t]; buy_max = 0 [t/h]; unit_cost = 330 [USD/t];
      reserve_t = 0 [t]; reserve_price = 30 [USD/t/h]; restock = 200 [USD/t];
    state: stock = 30000 [t] in [0, cap];
    decision: buy in [0, buy_max] default 0 hold 8;
    nodes: in feed <- dri; in draws <- dri;
    telemetry: stock <- "YARD.STOCK" [t];
    behavior:
      stock <- stock + (feed.dri + buy - draws.dri) * dt;
      cost merchant_hbi = buy * env.hbi_price;
      final revenue inventory = unit_cost * stock;
      constraint soft(reserve_price) working_stock: (reserve_t - stock) * pos(reserve_t - stock) / max(reserve_t, 1 [t]) <= 0 [t];
      final cost value_to_go = restock * (min(pos(reserve_t - stock), max(0.1 * reserve_t, 1 [t])) * min(pos(reserve_t - stock), max(0.1 * reserve_t, 1 [t]))
                                          / (2 * max(0.1 * reserve_t, 1 [t])) + pos(reserve_t - stock - max(0.1 * reserve_t, 1 [t])));
  };

  ## Yard of semi-finished or finished steel (slabs, billets, coil, coated coil, long products), valued at cost.
  define product_yard as ent {
    geom: point; layer = "yards"; icon = "warehouse";
    properties: cap = 300000 [t]; unit_cost = 500 [USD/t];
      reserve_t = 0 [t]; reserve_price = 30 [USD/t/h];     ## stock target: order mix, vessel and rolling slots
      restock = 200 [USD/t];                              ## what a tonne short of the target at the horizon end costs the plan
      stock_max = 1000000000 [t]; hold_price = 0.5 [USD/t/h]; ## top of the S&OP stock band (planning only)
    state: stock = 80000 [t] in [0, cap];
    nodes: in feed <- *; in draws <- *;
    telemetry: stock <- "YARD.STOCK" [t];
    behavior:
      stock <- stock + (sum(feed) - sum(draws)) * dt;
      final revenue inventory = unit_cost * stock;
      constraint soft(reserve_price) working_stock: (reserve_t - stock) * pos(reserve_t - stock) / max(reserve_t, 1 [t]) <= 0 [t];
      ## the plan ends at the target (planning only, not in the accounts): an hourly penalty costs little in the last
      ## hours, so a plan ran the yards down in its last day - and over-produced at its front to have the stock, the
      ## only part a rolling plan executes
      final cost value_to_go = restock * (min(pos(reserve_t - stock), max(0.1 * reserve_t, 1 [t])) * min(pos(reserve_t - stock), max(0.1 * reserve_t, 1 [t]))
                                          / (2 * max(0.1 * reserve_t, 1 [t])) + pos(reserve_t - stock - max(0.1 * reserve_t, 1 [t])));
      ## stock beyond the band is never used, but at standard cost making it in a cheap hour books a paper gain
      constraint soft(hold_price) stock_band: stock <= stock_max;
  };

  # ================================================================== iron and steelmaking
  ## Blast furnace: hot metal from burden (ore, pellets, sinter), coke and pulverised coal. A furnace cannot stop
  ## at will (it would have to be banked): its rate stays between `rate_min` and its full wind, and changes slowly.
  define blast_furnace as ent {
    geom: point; layer = "ironmaking"; icon = "furnace";
    properties:
      rate_max  = 400 [t/h] calibrate in [200, 650];       ## hot metal at full wind
      rate_min  = 0.6;                                     ## share of full wind below which it would be banked
      ore_rate  = 1.58;                                    ## t burden per t hot metal
      coke_rate = 0.36 calibrate in [0.28, 0.48];          ## t coke per t hot metal (coke + coal = fuel rate)
      pci_rate  = 0.15; pci_cost = 140 [USD/t];
      kwh_t = 85 [kWh/t]; cost_t = 14 [USD/t]; fixed = 6000 [USD/h];
      co2_t = 1450 [kgCO2e/t];                             ## per t hot metal, net of the gas credit
      ets_share = 0;
      ramp = 0.06;                                         ## largest change of rate per hour, share of full wind
    state: up = 1; last = 0.9 * rate_max [t/h];
    decision: rate in [rate_min * rate_max, rate_max] default 0.85 * rate_max hold 8;
    nodes: out hm <- hot_metal; out ore_req <- iron_ore; out coke_req <- coke; out pw <- power;
    telemetry: made <- "BF.HM.TPH" [t/h]; coke_use <- "BF.COKE.TPH" [t/h]; up <- "BF.RUN";
    behavior:
      made = rate * up;
      hm.hot_metal = made;
      ore_req.iron_ore = made * ore_rate;
      coke_use = made * coke_rate;
      coke_req.coke = coke_use;
      pw.power = made * kwh_t;
      last <- rate;
      ## a furnace changes its wind slowly: thermal state and burden descent (priced per t/h beyond the ramp)
      constraint soft(40 [USD/t]) ramp_limit: abs(rate - last) <= ramp * rate_max;
      cost pci = made * pci_rate * pci_cost;
      cost ironmaking = made * cost_t;
      cost furnace_fixed = fixed;
      emission furnace_co2 = made * co2_t;
      cost ets = made * co2_t * ets_share * env.eua_price;
      metric hot_metal = made [t/h];
  };

  ## Basic oxygen converter shop: hot metal + scrap -> liquid steel. The hot-metal share of the charge is a
  ## decision within the heat balance (more scrap = more steel per tonne of hot metal, up to `hm_min`).
  define bof_shop as ent {
    geom: point; layer = "steelmaking"; icon = "converter";
    properties:
      cap = 900 [t/h];
      hm_min = 0.74; hm_max = 0.92; yield = 0.91;
      kwh_t = 25 [kWh/t]; cost_t = 32 [USD/t]; fixed = 4000 [USD/h];
      co2_t = 110 [kgCO2e/t]; ets_share = 0;
      pig_value = 300 [USD/t];                   ## hot metal the converters cannot take, cast to pig iron and sold
    state: up = 1;
    decision: hm_share in [hm_min, hm_max] default 0.84 hold 8;
    nodes: in hm_in <- hot_metal; in scrap_in <- scrap; out scrap_req <- scrap; out liq <- steel_liq; out pw <- power;
    telemetry: made <- "BOF.TPH" [t/h]; pig <- "BOF.PIG.TPH" [t/h]; up <- "BOF.RUN";
    behavior:
      ## the shop blows all the hot metal its furnaces send (it cannot wait): the furnaces' wind is the throughput
      ## decision, the scrap ratio the shop's own one
      blown = min(hm_in.hot_metal, cap * up / yield * hm_share);
      pig = hm_in.hot_metal - blown;
      charge = blown / hm_share;
      made = charge * yield;
      scrap_req.scrap = charge * (1 - hm_share);
      liq.steel_liq = made;
      liq.steel_liq.cu = scrap_in.scrap.cu * (1 - hm_share) / yield;
      pw.power = made * kwh_t;
      cost steelmaking = made * cost_t;
      cost shop_fixed = fixed;
      revenue pig_iron = pig * pig_value;
      emission melt_co2 = made * co2_t;
      cost ets = made * co2_t * ets_share * env.eua_price;
  };


  ## Electric arc furnace: scrap + DRI/HBI -> liquid steel with electric power. Heats can be paused hour by hour
  ## (load shifting); DRI dilutes the scrap's copper but costs power (it melts colder).
  define eaf as ent {
    geom: point; layer = "steelmaking"; icon = "eaf";
    properties:
      cap = 200 [t/h] calibrate in [80, 320];             ## tap-to-tap productivity
      kwh_t = 410 [kWh/t] calibrate in [320, 560];        ## on an all-scrap charge
      dri_kwh = 150 [kWh/t];                              ## extra power per t of DRI share
      dri_max = 0.6; yield = 0.9;
      cost_t = 48 [USD/t];                                ## electrodes, refractories, oxygen, alloys, fluxes
      fixed = 3000 [USD/h];
      co2_t = 70 [kgCO2e/t]; ets_share = 0;               ## process: electrodes, charge carbon, lime
    state: up = 1;
    decision: steel in [0, cap] default 0.8 * cap hold 1; dri_share in [0, dri_max] default 0.3 hold 8;
    nodes: in scrap_in <- scrap; out scrap_req <- scrap; out dri_req <- dri; out liq <- steel_liq; out pw <- power;
    telemetry: made <- "EAF.TPH" [t/h]; mw <- "EAF.MW" [MW]; up <- "EAF.RUN";
    behavior:
      made = steel * up;
      charge = made / yield;
      scrap_req.scrap = charge * (1 - dri_share);
      dri_req.dri = charge * dri_share;
      liq.steel_liq = made;
      liq.steel_liq.cu = scrap_in.scrap.cu * (1 - dri_share) / yield;
      mw = made * (kwh_t + dri_kwh * dri_share);
      pw.power = mw;
      cost steelmaking = made * cost_t;
      cost shop_fixed = fixed;
      emission melt_co2 = made * co2_t;
      cost ets = made * co2_t * ets_share * env.eua_price;

  };

  ## Gas-based direct reduction plant (shaft furnace): pellets + natural gas -> DRI. Runs continuously between
  ## its turndown and full rate.
  define dri_plant as ent {
    geom: point; layer = "ironmaking"; icon = "reactor";
    properties:
      cap = 250 [t/h]; rate_min = 0.5;
      pellet_rate = 1.42; gj_t = 10.4 [GJ/t] calibrate in [8.5, 13]; kwh_t = 110 [kWh/t];
      cost_t = 12 [USD/t]; fixed = 2500 [USD/h];
    state: up = 1;
    decision: rate in [rate_min * cap, cap] default 0.9 * cap hold 6;
    nodes: out dri_out <- dri; out ore_req <- iron_ore; out gas_req <- fuel_gas; out pw <- power;
    telemetry: made <- "DRI.TPH" [t/h]; gas <- "DRI.GAS" [GJ/h]; up <- "DRI.RUN";
    behavior:
      made = rate * up;
      dri_out.dri = made;
      ore_req.iron_ore = made * pellet_rate;
      gas = made * gj_t;
      gas_req.fuel_gas = gas;
      pw.power = made * kwh_t;
      cost reduction = made * cost_t;
      cost plant_fixed = fixed;
      metric dri_made = made [t/h];
  };

  # ================================================================== casting and rolling
  ## Continuous caster: liquid steel cannot wait, so the caster takes what the melt shop taps, up to its capacity.
  ## Copper above the product's limit downgrades the cast (sold as a lower grade: a real discount per t and %-point).
  define slab_caster as ent {
    geom: point; layer = "casting"; icon = "caster";
    properties: cap = 900 [t/h]; yield = 0.98; cost_t = 14 [USD/t]; kwh_t = 20 [kWh/t];
      cu_max = 0.12; downgrade = 3000 [USD/t];             ## per t cast and %-point of copper above cu_max
    state: up = 1;
    nodes: in liq <- steel_liq; out cast <- slab; out pw <- power;
    telemetry: cast_rate <- "CC.TPH" [t/h]; up <- "CC.RUN";
    behavior:
      steel = liq.steel_liq;
      cast_rate = steel * yield;
      constraint capacity: steel <= cap * up;
      cast.slab = cast_rate;
      pw.power = steel * kwh_t;
      cost casting = steel * cost_t;
      cost downgrade = pos(steel * liq.steel_liq.cu - steel * cu_max) * downgrade;
  };
  define billet_caster as ent {
    geom: point; layer = "casting"; icon = "caster";
    properties: cap = 250 [t/h]; yield = 0.98; cost_t = 12 [USD/t]; kwh_t = 15 [kWh/t];
      cu_max = 0.35; downgrade = 1500 [USD/t];
    state: up = 1;
    nodes: in liq <- steel_liq; out cast <- billet; out pw <- power;
    telemetry: cast_rate <- "CC.TPH" [t/h]; up <- "CC.RUN";
    behavior:
      steel = liq.steel_liq;
      cast_rate = steel * yield;
      constraint capacity: steel <= cap * up;
      cast.billet = cast_rate;
      pw.power = steel * kwh_t;
      cost casting = steel * cost_t;
      cost downgrade = pos(steel * liq.steel_liq.cu - steel * cu_max) * downgrade;
  };

  ## Hot strip mill: slabs reheated (gas) and rolled to coil.
  define hot_strip_mill as ent {
    geom: point; layer = "rolling"; icon = "mill";
    properties: cap = 700 [t/h]; yield = 0.975 calibrate in [0.93, 0.995];
      kwh_t = 105 [kWh/t]; gj_t = 1.25 [GJ/t]; cost_t = 22 [USD/t]; fixed = 3500 [USD/h];
    state: up = 1;
    decision: rate in [0, cap] default 0.8 * cap hold 4;
    nodes: out slab_req <- slab; out coil <- hrc; out pw <- power; out gas_req <- fuel_gas;
    telemetry: rolled <- "HSM.TPH" [t/h]; up <- "HSM.RUN";
    behavior:
      rolled = rate * up;
      slab_req.slab = rolled / yield;
      coil.hrc = rolled;
      pw.power = rolled * kwh_t;
      gas_req.fuel_gas = rolled * gj_t;
      cost rolling = rolled * cost_t;
      cost mill_fixed = fixed;
      metric hot_rolled = rolled [t/h];
  };

  ## Cold mill + continuous galvanising line: coil -> coated coil (automotive, appliances, construction).
  define coating_line as ent {
    geom: point; layer = "rolling"; icon = "galvanising";
    properties: cap = 200 [t/h]; yield = 0.95; kwh_t = 140 [kWh/t]; gj_t = 0.9 [GJ/t];
      cost_t = 55 [USD/t];                                 ## zinc, rolls, chemicals
      fixed = 2000 [USD/h];
    state: up = 1;
    decision: rate in [0, cap] default 0.8 * cap hold 8;
    nodes: out coil_req <- hrc; out product <- coated; out pw <- power; out gas_req <- fuel_gas;
    telemetry: made <- "CGL.TPH" [t/h]; up <- "CGL.RUN";
    behavior:
      made = rate * up;
      coil_req.hrc = made / yield;
      product.coated = made;
      pw.power = made * kwh_t;
      gas_req.fuel_gas = made * gj_t;
      cost coating = made * cost_t;
      cost line_fixed = fixed;
  };

  ## Bar and rod mill: billets -> rebar and wire rod.
  define long_mill as ent {
    geom: point; layer = "rolling"; icon = "mill";
    properties: cap = 180 [t/h]; yield = 0.96; kwh_t = 110 [kWh/t]; gj_t = 1.1 [GJ/t]; cost_t = 20 [USD/t];
      fixed = 1500 [USD/h];
    state: up = 1;
    decision: rate in [0, cap] default 0.8 * cap hold 4;
    nodes: out billet_req <- billet; out product <- long_prod; out pw <- power; out gas_req <- fuel_gas;
    telemetry: rolled <- "BAR.TPH" [t/h]; up <- "BAR.RUN";
    behavior:
      rolled = rate * up;
      billet_req.billet = rolled / yield;
      product.long_prod = rolled;
      pw.power = rolled * kwh_t;
      gas_req.fuel_gas = rolled * gj_t;
      cost rolling = rolled * cost_t;
      cost mill_fixed = fixed;
  };

  # ================================================================== lanes
  ## A lane ships one product from a yard to a market (sales: freight to the customer) or to another site's yard
  ## (transfer: rail or barge between sites of a region). Its rate is a decision; it draws from the source yard.
  define lane_hrc as ent {
    geom: line; layer = "logistics";
    properties: cap = 400 [t/h]; freight = 25 [USD/t];
    decision: ship in [0, cap] default 0.5 * cap hold 8;
    nodes: out draw <- hrc; out deliver <- hrc;
    behavior:
      draw.hrc = ship;
      deliver.hrc = ship;
      cost freight = ship * freight;
  };
  define lane_coated as ent {
    geom: line; layer = "logistics";
    properties: cap = 200 [t/h]; freight = 28 [USD/t];
    decision: ship in [0, cap] default 0.5 * cap hold 8;
    nodes: out draw <- coated; out deliver <- coated;
    behavior:
      draw.coated = ship;
      deliver.coated = ship;
      cost freight = ship * freight;
  };
  define lane_long as ent {
    geom: line; layer = "logistics";
    properties: cap = 200 [t/h]; freight = 22 [USD/t];
    decision: ship in [0, cap] default 0.5 * cap hold 8;
    nodes: out draw <- long_prod; out deliver <- long_prod;
    behavior:
      draw.long_prod = ship;
      deliver.long_prod = ship;
      cost freight = ship * freight;
  };
  define lane_slab as ent {
    geom: line; layer = "logistics";
    properties: cap = 300 [t/h]; freight = 30 [USD/t];
    decision: ship in [0, cap] default 0.5 * cap hold 8;
    nodes: out draw <- slab; out deliver <- slab;
    behavior:
      draw.slab = ship;
      deliver.slab = ship;
      cost freight = ship * freight;
  };
  define lane_dri as ent {
    geom: line; layer = "logistics";
    properties: cap = 150 [t/h]; freight = 18 [USD/t];
    decision: ship in [0, cap] default 0.5 * cap hold 8;
    nodes: out draw <- dri; out deliver <- dri;
    behavior:
      draw.dri = ship;
      deliver.dri = ship;
      cost freight = ship * freight;
  };
};
