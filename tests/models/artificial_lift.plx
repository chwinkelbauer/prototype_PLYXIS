## Small test field for the extended oil & gas library: gas lift, plunger lift, separator, pipeline hydraulics,
## chemical skid, vapour recovery, LACT. No twin of a real site; used by the test-suite.
use oilgas;

define lift_field as env {
  origin = (31.60, -103.70);
  properties:
    oil_price = 70 [USD/bbl] input;
    gas_price = 2.0 [USD/Mscf] input;
    power_price = 0.08 [USD/kWh] input;
    grid_co2 = 0.4 [kgCO2e/kWh];
    carbon_price = 45 [USD/tCO2e];
  contains:
    init LC as lift_compressor { at = (31.60, -103.70); tag = "LF.LC"; cap = 900 [Mscf/d]; };
    init W1 as gas_lift_well { at = (31.61, -103.71); tag = "LF.W1"; q_max = 1500 [bbl/d]; lift_opt = 700 [Mscf/d]; };
    init W2 as gas_lift_well { at = (31.62, -103.71); tag = "LF.W2"; q_max = 900 [bbl/d]; lift_opt = 700 [Mscf/d]; };
    init PL as plunger_lift_well { at = (31.63, -103.72); tag = "LF.PL"; };
    connect(W1.lift_req, LC.requests);
    connect(W2.lift_req, LC.requests);
    init SEP as separator { at = (31.60, -103.72); tag = "LF.SEP"; };
    connect(W1.prod, SEP.feed);
    connect(W2.prod, SEP.feed);
    connect(PL.prod, TK.feed);
    init TK as battery { at = (31.59, -103.72); tag = "LF.TK"; };
    connect(SEP.liquids_out, TK.feed) via pipeline { path = [(31.60, -103.72), (31.59, -103.72)]; q_design = 2500 [bbl/d];
                                                     dp_design = 200 [psi]; dp_max = 400 [psi]; } as LINE;
    init VR as vru { at = (31.59, -103.73); tag = "LF.VRU"; };
    connect(TK.gas_out, VR.vapour);
    init GS as gas_sales { at = (31.58, -103.73); tag = "LF.GS"; };
    connect(SEP.gas_out, GS.rec);
    connect(VR.sales, GS.rec);
    init SWD as swd_well { at = (31.58, -103.74); tag = "LF.SWD"; inj_cap = 8000 [bbl/d]; };
    connect(TK.wat_out, SWD.inj);
    init SK as chemical_skid { at = (31.58, -103.75); };
    init LACT as lact_unit { at = (31.57, -103.75); tag = "LF.LACT"; };
    connect(TK.oil_out, LACT.rec) via chemical_skid { at = (31.58, -103.75); dose = 0.01 [gal/bbl]; } as SKID;
};

define lift_day as sim {
  env: lift_field;
  horizon = 12 [h];
  step = 1 [h];
  objective:
    maximize sum(revenue) - sum(cost) - carbon_price * sum(emission);
};
