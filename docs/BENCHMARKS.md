# Benchmarks

Every number here is produced by the scripts in `bench/` and stored in `bench/results/`
(`python -m bench.report` prints these tables). Hardware: one laptop, AMD Ryzen 5 PRO 4650U
(6 cores, 15 W), 16 GB RAM, CPU only, PyTorch 2.14, g++ 14 (MinGW-w64) for the native kernels.
All data is synthetic.

```bash
python -m bench.showcase_run --hours 72                      # Kestrel A/B run of the full showcase
python -m bench.showcase_run --scenario corvus               # Corvus mine A/B run (48 h)
python -m bench.showcase_run --scenario aquila --seed 1      # Aquila steel group A/B run (96 h)
python -m bench.policy_in_twin --scenario aquila             # plan vs. the baseline's rules in the twin's own world
python -m bench.corporate --rounds 3 --reference --workers 6 # steel group hourly + the works' plants found at 15 min; whole twin at 15 min
python -m bench.corporate --regions --rounds 5 --workers 6   # the hourly group: whole vs its regional networks
python -m bench.corporate --plant-ablation                   # each plant alone: GA vs per-unit DPs
python -m bench.structure                                    # the block tree of every twin's asset graph
python -m bench.twin_accuracy --hours 30 --full              # twin vs. hidden truth over time
python -m bench.solver_comparison --budget 60                # solvers at equal wall time
python -m bench.solver_budget --cases 5                      # what the live solver budget buys
python -m bench.neural_ablation --experts 60 --test 8        # what the neural sub-solvers add
python -m bench.decomposition --seeds 3                      # deterministic sections, native kernels
python -m bench.section_theory                               # section DP vs. exact MILP, known structure
python -m bench.exact_branches                               # structure recognition, exact LP branches
python -m bench.scaling --fields 1 2 4 8 --ga                # enterprise twins up to 768 wells
python -m bench.backtest --scenario kestrel                  # pilot gate G1: historian export, frozen twin, unseen history
python -m bench.risk                                         # outcome distributions: sigma points vs Monte Carlo; --frontier
python -m bench.uncertainty_coverage --hours 48              # the twin's posterior intervals vs the hidden truth
python -m bench.colgen                                       # column generation vs exact MILP by size; --learn
python -m bench.planner_speed                                # plan time and value per twin and planner (alone!)
```

## Kestrel showcase run (72 h)

`bench/results/showcase_run.json` (seed 0) and `showcase_run_s1.json`, `showcase_run_s2.json` - the full
Kestrel scenario: two compressor trips, a 10-hour gas plant upset that cuts takeaway by a third, a
fast-wearing ESP, an unannounced 55 % loss of disposal injectivity, 24 h of negative gas prices, two hours of
power at ERCOT's 5,000 USD/MWh cap, and a new well that appears only as SCADA tags. Two worlds start from
the same state and see the same hidden physics, events, notices, prices and sensor noise: one operated by
PLYXIS (re-planning every 3 h and after every event), one by a rule-based model of competent manual
operations ([SHOWCASE.md](SHOWCASE.md)). The seed changes the realised prices and the sensor and process
noise - for both worlds alike.

| seed | operating profit | after carbon cost | oil | cash operating cost | flaring | CO2e |
|---|---|---|---|---|---|---|
| 0 | +382,137 (+13.8 %) | +14.5 % | +11.1 % | -8.7 % | -85 % | -31 % |
| 1 | +398,257 (+14.7 %) | +15.4 % | +11.4 % | -10.4 % | -82 % | -29 % |
| 2 | +398,127 (+14.7 %) | +15.3 % | +11.2 % | -9.9 % | -85 % | -31 % |
| **mean** | **+392,840 (+14.4 %)** | **+15.1 %** | +11.3 % | -9.7 % | -84 % | -30 % |

Seed 0 in detail:

| KPI (72 h) | PLYXIS | manual operations | difference |
|---|---|---|---|
| **Operating profit** (USD) | 3,146,031 | 2,763,894 | **+382,137 (+13.8 %)** |
| **Profit after carbon cost** (USD) | 3,119,211 | 2,724,958 | **+394,253 (+14.5 %)** |
| Oil produced (bbl) | 54,169 | 48,756 | +5,412 (+11.1 %) |
| Gas produced (Mscf) | 103,423 | 102,688 | +734 (+0.7 %) |
| **Cash operating cost** (USD) | 402,855 | 441,080 | **-38,224 (-8.7 %)** |
| **Gas flared** (Mscf) | 390 | 2,630 | **-2,240 (-85 %)** |
| Flared above the permit (Mscf) | 51 | 1,880 | -1,829 (-97 %) |
| Emissions (t CO2e) | 447 | 649 | -202 (-31 %) |
| Unplanned pump failures | 0 | 1 | |

How the comparison is valued - identically for both worlds: sales minus cash costs, plus the change in
inventories (oil in tanks at 90 % of the price, water in tanks as a disposal liability - the planner's
own terminal valuation), plus the change in remaining pump life (a pump's value runs from new to the
failure threshold; a replacement is charged at the pump cost, an unplanned failure additionally at the
model's failure premium). Planning-only terms (expected failure cost, flaring-risk margin, nominal wear)
are not counted. The carbon cost uses the field's internal price (60 USD/t), which is also what the
planner optimises.

Where the difference comes from (seed 0, realised terms, `showcase_run.json` -> `bridge`, adds up to the
operating-profit difference):

| Term | USD |
|---|---|
| Oil sales | +617,678 |
| Oil and water in tanks | -213,123 |
| Pump life consumed | -58,648 |
| Avoided unplanned pump failure | +45,000 |
| Power | -15,647 |
| Well restarts | +12,182 |
| Water injection power | +3,872 |
| Disposal fees, treating, gas sales, trucking, lease | -9,177 |
| **total** | **+382,137** |

PLYXIS produces 11 % more oil (choosing which wells to run and how fast when gas takeaway, compression
and disposal bind) and also sells down tank inventory that manual operations keep at about half full.
The tank draw-down is netted out by the inventory line; what remains of it in the result is the 10 % gap
between selling a barrel now and valuing it in a tank at the end.

By period (seed 0; difference in operating profit; flaring in Mscf):

| Period | delta profit (USD) | PLYXIS flare | manual flare |
|---|---|---|---|
| normal operation (first 7 h) | +56,180 | 0 | 0 |
| CS-1 trip | +21,514 | 196 | 1,055 |
| gas plant upset (takeaway -1/3, 10 h), ESP wear | +70,537 | 10 | 198 |
| SWD-2 injectivity loss, new well | +45,661 | 0 | 0 |
| negative gas price | +108,106 | 0 | 0 |
| CS-2 trip during negative price | +42,709 | 155 | 1,160 |
| evening incl. the ERCOT price cap | +20,392 | 7 | 209 |
| recovery to end | +17,038 | 0 | 2 |

Twin and planner during the run (seed 0): well deliverability error 11.9 % → 2.3 %, GOR 12.9 % → 4.7 %,
water cut 4.7 % → 2.1 %; 32 plans of 4,983 decisions each, all feasible. These runs shared the machine with other benchmarks, so
their solve times mean nothing; alone, a plan takes about 10 s (Solver budget below). Over the three seeds 93 of 96 plans
were feasible. No seed had an unplanned pump failure in the PLYXIS world; the manual-operations world had one on each.

**A wrong slope, fixed: no measurable effect on this result.** `rt.pwl` returned slope 0 at the top of its table: in float32 the ratio inside it can lie one rounding step above 1, where `clamp` has derivative 0. A pump at full speed therefore had no slope of its rate, only of its power and wear, and about half of a plan's well speeds sit there. It surfaced when a sequential LP built on reverse-mode slopes lost money where one built on finite differences gained (slopes correlated 0.47 at that plan, 0.999 after the fix). The fix changes derivatives only, not one simulated value. Every number above is with the fix.

| seed | before the fix (`arms/showcase_run_beforeslope*`) | with the fix | difference |
|---|---|---|---|
| 0 | +397,124 | +382,137 | -14,987 |
| 1 | +378,360 | +398,257 | +19,897 |
| 2 | +401,056 | +398,127 | -2,929 |
| **mean** | **+392,180** | **+392,840** | **+660** |

The first-order stages evidently found their way around the wrong slope: the difference is within the noise of three seeds, feasible plans went from 94 of 95 to 93 of 96, and the manual-operations world is identical to the cent. So the fix is right, and it is not a gain. The mine is not affected: its values and gradients were identical at all four points tested (its only table is the recovery curve, and the head grade stays inside it), so its results stand. The comparison arms in this file, the `core` run below and the Kestrel sections further down (solver comparison, budget, neural ablation, scaling, twin accuracy, decomposition) were not re-run after the fix.

Cold plans are less clear (no neural start, four seeds, equal evaluations, `bench/checks/cold_plans_old_vs_new.py`, `arms/cold_plans_old_vs_new.log`): three got better with the fix (+8,449, +53,060, +9,113) and one far worse (-326,482), so the mean falls from 4,330,118 to 4,266,153. Four seeds cannot say whether that is a wider spread or a real loss. The A/B above starts its later plans warm and does not show it.

**The convex stage has no A/B result yet.** Sequential LP over the wells (`--exact convex --exact-margin 0.10`) gains in the plan (Structure section) but was stopped for time at hour 24 of 72, before the negative-price period: +162,505 against +163,053 without it at that hour, three seeds (`arms/showcase_run_slope_convex_partial_s*.log`). That is no decision either way, so the stage stays off on the oil field.

**Without the two newest events** (`showcase_run_core.json`, before the slope fix: no gas-plant upset, no ERCOT price cap; seed 0):
+294,368 (+9.5 %) operating profit, +10.1 % after carbon, oil +7.3 %, flaring -83 %. About four fifths of
the result does not depend on those two events; they add the rest because they are situations where a
plan that sees the whole field is worth more than fixed rules.

**Reproducibility.** Re-running seed 0 reproduces the manual-operations world exactly. The PLYXIS world is not bit-reproducible: multi-threaded floating point in the solver differs between runs. With unchanged code and thread count a rerun differed by USD 136; with 2 threads instead of 6 the same seed gave +355,215 instead of about +386,000. Compare arms only at equal thread counts, and over seeds.

**The exact-branch stage stays off on the oil field** (measured before the slope fix, on both sides of each comparison). Without margins it lowered the result on every seed (mean -47,832: the LP drove the tank batteries to their limits and the buffer was gone when the gas-plant upset and the compressor trips came).
It was re-measured with a margin: `exact_margin` keeps state limits inside by a share of their scale. Measured with 10 % over the same three seeds (`showcase_run_branches*.json`, `python -m bench.showcase_run --exact lp,slp,slp_quality --exact-margin 0.10`):

| seed | without the stage (configured) | with the stage and 10 % margins | effect of the stage |
|---|---|---|---|
| 0 | +397,124 | +390,174 | -6,950 |
| 1 | +378,360 | +373,791 | -4,569 |
| 2 | +401,056 | +390,446 | -10,611 |

The margin removes most of the harm (mean -7,377, -10,611 to -4,569) but the stage still does not pay here, so it stays off. A 5 % margin was worse than 10 % in a separate comparison at 2 threads each (-12,570 against +6,840, both against the same no-stage arm, `arms/showcase_run_guard*`, `_exact*`, `_exact10*`); the difference between those two is within the noise of a three-seed mean. On the mine the stage stays on (Corvus section).

**How this number was arrived at.** An earlier version of this benchmark reported +16 % operating
profit. Investigation showed that most of it was not PLYXIS: the emulator's protection layer
over-throttled overloaded flows, which hurt the rule-based world more, and the baseline's water handling
was weaker than a real control room. After those fixes, and with pump condition and carbon in the
accounting, the result was +2.0 %. It grew to +13.7 % (one run) for reasons that are documented and
testable: (1) the solver - more diverse starts, a longer polish and a feasibility restoration; (2) planner
corrections - announced price transients are no longer extrapolated into the forecast, pump wear is priced
the way the accounting values it; (3) a more realistic scenario - firm gas takeaway below field
deliverability (the Permian situation before new pipelines), a midstream upset with a curtailment notice,
disposal fees, and a scarcity price window. Every new event came with the standard response a competent
control room would have (immediate curtailment on the midstream notice, the 75 % demand-response set point
for the price cap), so the baseline was not weakened. A bug found on the way - price events had silently
lasted 24 h instead of their stated duration - was fixed before these numbers were produced. Then the price
process was made independent of the code's call order (one random stream per price) and the neural
sub-solvers were retrained; that gave +392,180 over three seeds. The slope fix described above then gave the result at the top of this section.

## Twin accuracy over time (Kestrel)

`bench/results/twin_accuracy.json` - every hour of plant time the live twin is compared with the
emulator's hidden truth (production planner running, all scenario events active). The twin sees
only SCADA telemetry and one well test per hour.

| plant time (h) | well deliverability q_max | GOR | water cut | pad gas, one step ahead (mean / max) | compressor stations |
|---|---|---|---|---|---|
| 6 (start, engineering model) | 10.9 % | 12.7 % | 4.7 % | 8.0 % / 32 % | +4.3 %, +2.2 % |
| 10 | 3.0 % | 12.9 % | 4.5 % | 5.7 % / 10 % | +0.4 %, +0.5 % |
| 18 | 2.4 % | 11.8 % | 4.2 % | 5.0 % / 17 % | -0.4 %, +0.0 % |
| 26 | 2.5 % | 11.2 % | 3.8 % | 6.3 % / 15 % | +0.1 %, +2.7 % |
| 36 | 2.3 % | 10.3 % | 3.5 % | 4.3 % / 14 % | +0.7 %, +0.2 % |

Well deliverability is pinned down within four hours; GOR and water cut improve as well tests rotate
through the 96 wells (after 72 h: GOR 4.6 %, water cut 2.1 %, see the showcase run). With the plant's
true parameters, the twin reproduces all ~1,000 plant signals exactly - the remaining error is
estimation error, not model error.

## Solver comparison (equal wall time, Kestrel)

`bench/results/solver_comparison.json` - the Kestrel planning problem (72 h, hourly): 4,983 decision
variables of which 1,272 discrete. Every method starts from the same default plan (which violates
constraints); every final solution is evaluated identically (discrete decisions rounded, exact
simulation). Objective = 72 h operating profit minus carbon cost. Budget 60 s; the gradient methods stop
by themselves.

| Method | objective (USD) | max violation | feasible | seconds |
|---|---|---|---|---|
| PSO | 3,145,609 | 0.5963 | no | 60.0 |
| gravitational search | 1,935,002 | 0.0617 | no | 60.1 |
| GA | 4,182,931 | 0.0004 | yes | 60.0 |
| DE | 3,786,100 | 0.0000 | yes | 60.0 |
| PSO, cooperative (per subsystem) | 1,711,091 | 1.6285 | no | 60.0 |
| GA, cooperative (per subsystem) | 1,112,336 | 0.0000 | yes | 60.0 |
| augmented Lagrangian, single start | 3,236,756 | 0.0000 | yes | 7.0 |
| **PLYXIS cascade** | **4,354,262** | 0.0016 | yes | **10.3** |
| **PLYXIS cascade + neural sub-solvers** | **4,387,048** | 0.0017 | yes | **10.6** |

The vectorised GA is the strongest classic method, but it needs the whole minute and ends 4.7 % below
the cascade with neural sub-solvers, which finishes in 10.6 s. PSO and gravitational search - the
original concept's front end - do not reach a feasible plan at this dimension.

## Solver budget (Kestrel)

`bench/results/solver_budget.json` - 5 sampled event situations re-planned from the nominal plan, as in
live operation:

| Configuration | relax / starts / discrete / repair / polish | mean objective (USD) | vs previous live | feasible | mean seconds |
|---|---|---|---|---|---|
| previous live | 160 / 3 / 12x1 / 80 / 50 | 4,161,414 | - | 4/5 | 7.0 |
| **live** | 200 / 8 / 12x1 / 80 / 120 | 4,217,939 | **+1.4 %** | 5/5 | 10.0 |
| large | 480 / 8 / 24x3 / 150 / 150 | 4,199,357 | +0.9 % | 5/5 | 31.6 |

## Neural sub-solvers (ablation, Kestrel)

`bench/results/neural_ablation.json` - the cascade solves 60 realistic event situations warm-started from
the current plan; the graph network is trained on them (137 s). Then 8 held-out situations are re-planned
with equal budgets:

| Start / method | mean objective (USD) | feasible | mean seconds |
|---|---|---|---|
| previous plan, as is | 1,369,012 | 0/8 | 0.02 |
| neural sub-solver plan, as is | 1,740,764 | 0/8 | 0.02 |
| previous plan + gradient repair | 2,870,377 | 3/8 | 1.54 |
| neural plan + gradient repair | 2,924,572 | 5/8 | 1.50 |
| full cascade | 3,123,836 | 8/8 | 6.16 |
| full cascade + neural start | 3,352,147 | 8/8 | 6.46 |

* The network's raw plan (20 ms for the whole field) is a start, never the final word.
* With the same short repair, the neural start reaches a feasible plan in 5/8 cases (previous plan: 3/8).
* Inside the full cascade the neural start wins 7 of 8 cases and loses 1 (median +4.6 %, mean +8.4 %,
  range -8.3 % to +24.9 %).

## Scaling: from one field to a company

`bench/results/scaling.json` - generated enterprise twins: N copies of the Kestrel field sharing one gas
plant, one oil terminal and a company-wide flare permit. The GA gets exactly the cascade's wall time.

| Fields | wells | decisions (discrete) | compile | simulate 1 / 16 plans | gradient | generated code | cascade solve | cascade objective | GA, same time |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 96 | 4,983 (1,272) | 0.16 s | 7 / 12 ms | 15 ms | 432 lines | 12 s | USD 4.38 M, feasible | infeasible (max violation 0.56) |
| 2 | 192 | 9,966 (2,544) | 0.20 s | 8 / 19 ms | 20 ms | 432 lines | 13 s | USD 8.19 M, feasible | infeasible (max violation 1.24) |
| 4 | 384 | 19,932 (5,088) | 0.24 s | 10 / 34 ms | 21 ms | 432 lines | 19 s | USD 16.21 M, feasible | infeasible (max violation 2.28) |
| 8 | 768 | 39,864 (10,176) | 0.40 s | 12 / 105 ms | 34 ms | 432 lines | 38 s | USD 30.54 M, feasible | infeasible (max violation 2.44) |

* Adding assets adds tensor rows, not code: 768 wells compile to the same 432 lines as 96.
* Solve time grows sub-linearly: 8x the decisions take 3.2x the time (38 s vs. 12 s).
* With the same wall time the GA ends infeasible at every size; its higher 'objective' at 8 fields comes
  from violating shared capacity limits.

## Corvus mine showcase run (48 h)

`bench/results/showcase_run_corvus.json` (seed 0), `showcase_run_corvus_s1.json`, `_s2.json` - the underground copper
mine (186 stopes on 23 levels, ~5,000 planning decisions, see
[SHOWCASE.md](SHOWCASE.md#case-study-2-corvus-underground-mine)): main fan trip, seismic closure of a level,
a 5-hour hoist stop, three loaders down, a critical-peak price day, a pump failure and a new stope. Same
set-up as Kestrel: two copies of the mine, same truth, events, prices and noise. The baseline is competent
conventional mine control: loader allocation within each level's air budget with grade control against the
block model reconciled with its own assays, a tonnes target at the mill rate, a broken-stock buffer rule for blasting, design ventilation
(full fan speed after a fan trip), level-controlled pumps, the mill at 99 % of nameplate, deferral of
hoisting and pumping during announced price peaks. The planner runs the exact-branch stage (sequential LP
over the mine's linear and piecewise-linear layer, [DECOMPOSITION.md](DECOMPOSITION.md)).

Re-run on 4 Nov 2026 after the forecaster fix: until then the engine looked up market tags through the oil field's names, so the
mine's forecaster never received a price and every plan used flat nominal copper and power prices (announced windows only). The
lead did not move (mean +583,368 against +586,825 before; the old runs are `arms/showcase_run_corvus_beforefcfix*.json`). The
variants further down (capped window, stage off, re-plan budget) still ran before the fix.

| seed | operating profit | copper | head grade (PLYXIS / conventional) | production cost per t of copper | cash cost | plans feasible |
|---|---|---|---|---|---|---|
| 0 | +554,899 (+9.8 %) | +6.1 % | 1.430 % / 1.370 % | USD 2,952 / 3,161 (-6.6 %) | +1.4 % | 19/19 |
| 1 | +595,227 (+10.8 %) | +6.6 % | 1.431 % / 1.365 % | USD 2,949 / 3,176 (-7.2 %) | -1.4 % | 19/19 |
| 2 | +599,977 (+10.3 %) | +6.4 % | 1.436 % / 1.372 % | USD 2,886 / 3,110 (-7.2 %) | -1.1 % | 18/19 |
| **mean** | **+583,368 (+10.3 %)** | +6.4 % | | -7.0 % | -0.4 % | |

The baseline's annualised cash cost is USD 493-504 M over the seeds: the unit costs are calibrated to the USD 500 M/a the
founder specified (see below). The mill runs at about 14 Mt/a in this window (1,580 t/h baseline, 1,600 t/h PLYXIS, no planned downtime).

Seed 0 in detail:

| KPI (48 h) | PLYXIS | conventional control | difference |
|---|---|---|---|
| Operating profit (USD) | 6,216,692 | 5,661,794 | +554,899 (+9.8 %) |
| Ore milled (t) | 76,777 | 75,840 | +937 (+1.2 %) |
| Copper in concentrate (t) | 1,006 | 948 | +58 (+6.1 %) |
| Ore mined (t) | 72,877 | 71,670 | +1,207 (+1.7 %) |
| Ore hoisted (t) | 71,804 | 66,858 | +4,946 (+7.4 %) |
| Cash operating cost (USD) | 2,776,289 | 2,739,237 | +37,052 (+1.4 %) |
| Head grade (from copper, milled tonnes and the recovery curve) | 1.430 % | 1.370 % | mine-plan window 1.35-1.45 % |
| Production cost per t of copper (cash cost + ore inventories drawn down, at cost) | USD 2,952 | USD 3,161 | -6.6 % |
| Cash cost per t milled | USD 36.16 | USD 36.12 | |
| Annualised cash operating cost | USD 507 M | USD 500 M | |
| Site power (MWh) | 2,942 | 2,917 | +0.9 % |
| Sump overflow | none | none | |

| Where the operating-profit difference comes from | USD |
|---|---|
| Copper sales | +526,154 |
| Ore inventories (at cost) | +65,797 |
| Explosives and charging | -44,417 |
| Power (fans, pumps, hoist) | +30,294 |
| Mining consumables | -9,901 |
| Mill consumables | -6,089 |
| Mill power | -4,880 |
| Loader operating cost | -1,314 |
| Crusher wear | -863 |
| Trucking | +118 |
| **total** | **+554,899** |

Plans: 19 per run, 56 of 57 feasible, none far below the last one (plan values USD 7.1-7.9 M). These runs shared the machine with other
benchmarks, so their solve times mean nothing. Plant protection trimmed the level ventilation limits in both worlds about equally
often (seed 0: 1,095 against 1,034 steps); no stock was drawn below empty in either world.
Stope grade error of the twin: 11.8 % (engineering model) to 9.6-10.0 % after 48 h; drawpoint rates 8.9 % to 7.8-8.0 %.

**How to read this result.** Almost all of it is copper, and about four fifths of the copper is head grade: PLYXIS blends
near the top of the plan's window (1.43 %, limit 1.45 %), conventional control aims at the middle and delivers 1.37 %,
both inside the window. The rest is 1.2 % more milled tonnes. Power adds a little; cash cost is about equal (-1.4 % to +1.4 % over the
seeds; seed 0 works the mine harder for the same mill rate). This is a grade-control result, not a cost reduction. Anyone who reads the window as "1.40 % please, 1.45 % at most"
will take about half of it away; the sensitivity below measures that.

Cost per tonne milled is the same over the 48 h (seed 0: USD 36.16 against 36.12), so each tonne earns more at equal cost: revenue per tonne milled is USD 119.65 against 114.19. Early in a run the cumulative figure is higher for PLYXIS (seed 0 after 24 h: USD 33.9 against 33.2), which is what the console's tile shows; it evens out over the 48 h. With a 65 % margin in the baseline, +6.4 % revenue is about +10 % profit even with no change in cost.

**What this lead does not settle.** The orebody's true grade averages 1.416 % (`truth.plx`, weighted by ring tonnes), so PLYXIS's 1.43 % is little above it; most of the gap is the baseline running below it. The baseline lands 0.03 below its own 1.40 % target and never reads the mill's on-stream analyser, while PLYXIS corrects from it (`head_factor`). A real control room would see that offset, so this part of the lead may be flattering. If the baseline reached 1.40 %, about USD 206,000 of seed 0's USD 554,899 would go, leaving roughly +6.2 %. That figure is arithmetic on the recovery curve, not a benchmark. The twin also has no in-situ depletion, so 48 h cannot show what a richer blend does to later grade.

**The head grade now stays inside the window.** The twin plans it one standard error of the blend inside the limits
(`concentrator.grade_margin`, 0.009-0.036 %-points during the runs, from `Estimator.uncertainty`), and grade control samples
the working places instead of rotating through all stopes (three composite samples per hour at 5 % error, in both worlds; the
baseline reconciles its block model with its own assays). An earlier version ended 0.010-0.017 %-points above the
window. Staying inside costs copper: this is one reason the lead fell from +16.3 % (see below).

**What the exact-branch stage adds on the mine** (same seeds, stage off: `showcase_run_corvus_nobranches*.json`; the runs without
the stage are from before the forecaster fix and have not been re-run, the "with" column before the fix was +583,760, +584,584 and +592,133):

| seed | with the stage | without | effect | head grade with / without | head-grade factor with / without |
|---|---|---|---|---|---|
| 0 | +554,899 | +535,961 | +18,937 | 1.430 % / 1.432 % | 0.988 / 1.003 |
| 1 | +595,227 | +535,005 | +60,222 | 1.431 % / 1.429 % | 1.003 / 0.999 |
| 2 | +599,977 | +547,639 | +52,338 | 1.436 % / 1.435 % | 1.006 / 1.005 |

The stage adds USD 43,832 on average (positive on all three seeds, +18,937 to +60,222; before the fix, at equal code, USD 47,290). The USD 205,000 reported before included
grade above the window and plans that had lost their way; the head-grade factor no longer drifts to 1.05-1.07, so stopes whose grade the twin
underestimates are no longer selected systematically. The stage now uses the structured Jacobian (coloured finite differences plus batched
reverse mode): under equal load the Jacobian took 17-21 s per plan instead of 33-36 s, and the three-seed A/B was no worse
(+580,176 against +557,707; `bench/results/arms/showcase_run_corvus_sj*.json` and `_margin*.json`).

**Sensitivity: window capped at the plan grade** (`showcase_run_corvus_gradecap*.json`, `grade_max` 1.40 instead of 1.45 in both
worlds, 2 threads):

| seed | operating profit | copper | head grade (PLYXIS / conventional) | plans feasible |
|---|---|---|---|---|
| 0 | +264,526 (+4.7 %) | +2.6 % | 1.388 % / 1.370 % | 19/19 |
| 1 | +287,183 (+5.2 %) | +2.9 % | 1.385 % / 1.365 % | 18/19 |
| 2 | +281,198 (+4.8 %) | +2.8 % | 1.391 % / 1.372 % | 19/19 |
| **mean** | **+277,636 (+4.9 %)** | +2.8 % | | |

Without the top of the window the lead is about half: this is the number to quote to anyone who wants 1.40 %. Before the fix below this configuration
lost to the baseline (-3.1 to -5.5 %, `arms/showcase_run_corvus_gradecap_tol002*.json`), because the first plan ran the mill below 1,500 t/h for 7-8 hours.

**How this number was arrived at.** An earlier version of this section reported +17.4 %. Re-examining it
hour by hour found planner errors and a baseline that was not competent, all fixed before these runs (and
recorded in CLAUDE.md):

* *A phantom penalty.* The head grade is a flow-weighted average, which is 0 when the mill stands, so the
  soft grade window charged about USD 2.7 M for every idle mill hour. The planner therefore milled through
  the 1,500 USD/MWh peak and, during the hoist stop, cut a full ROM pad's feed to 350 t/h for the night. The
  window is now priced per tonne milled (250 USD per t and %-point outside it).
* *No repair estimates.* The twin assumed every failed asset stays down for the whole 48 h horizon. Crew
  estimates now reach the twin as a known return time.
* *Buffers without reserves.* Shifting pumping into cheap hours, PLYXIS kept sumps at 75-100 % full; one
  sump overflowed for hours - free in the accounting - and the unannounced pump failure flooded another.
  Sumps now keep room for 8 h of inflow, the ROM pad a 12,000 t working stock, and overflow is charged
  (USD 60/m3, both worlds).
* *A terminal value that was tried and removed.* Valuing pad ore at its full milling margin (planning only)
  made the planner idle the mill for 5-8 hours of the price peak in a mill-bound mine. Inventories are now
  valued at cost in plan and KPI alike; broken ore is valued at its blasting cost in the KPI (IAS 2, as for
  the other ore inventories; it was half) and at half of it in the plan, so rings are fired just in time.
* *The baseline.* Conventional loader allocation gave every working place an equal loader share regardless
  of its level's air; the plant's ventilation interlock then cut ~5,200 t/h of set points to ~1,000 t/h, and
  the baseline mill starved. A competent ventilation officer allocates within each level's air, and a shift
  boss works to a tonnes target: with those rules the baseline mines 40 % more ore, and the same planner's
  lead fell from +34 % to +12 %. Its annualised cash cost rose from USD 491 M to about USD 585 M with the
  extra tonnes.
* *Head grade.* The earlier "1.52 % vs 1.33 %" was not derived from the results; `bench.report` now inverts
  the concentrator's recovery curve on copper and milled tonnes (for that run: 1.455 % vs 1.287 %).
* *Exact plans need margins.* With the exact-branch stage the LP planned a sump exactly on its storage
  reserve, computed from an inflow the twin underestimated by 17 %; the unannounced pump failure then
  flooded it (92 m3, almost exactly 8 h of the inflow error). The reserve now carries a 25 % design margin
  on the inflow, and the head grade is planned offset-free (the analysed / computed head-grade factor is
  estimated online). Then the market prices were made independent of the code's call order (one random
  stream per price), which gave +16.3 % (three seeds, baseline USD 585 M/a).
* *Unit costs recalibrated.* The founder's figure for this mine is USD 500 M/a; the baseline measured USD 585 M/a. All non-energy unit
  costs were lowered by about 18 % (labour 13,500 to 11,000 USD/h, G&A 6,000 to 5,000, stope 10.0 to 8.2 USD/t, mill 8.0 to 6.5, rings 30,000
  to 24,600, fleets and fuel alike; inventories at the new cost). Power stays at market. Baseline profit rises by USD 0.5 M, which
  alone lowers the percentage; PLYXIS's mining savings shrink with the costs. With the old planner: +9.1 % (three seeds, `arms/showcase_run_corvus_recal*.json`).
* *Inside the window, working-place assays, charge-up crews.* See above. Ring firings per shift are now limited by the crews, two per zone,
  which only restricts PLYXIS: the baseline already fires at most two per zone and shift. One run had fired 17 rings in 48 h.
* *A plan can lose its way.* About one plan in 50 came out far below the last one (USD 5.5-6.8 M instead of about 7.5 M), not reproducibly: with the same seed and
  code, one plan before the price peak idled the mill for three hours in one run and cost about USD 360,000 against the next run (+557,288, `arms/diag_s0.log`; the poor run's file
  was overwritten). The cause: the relaxation picks its best iterate among those with violation up to 0.002, so a poor but strictly feasible iterate beat a far better one
  the discrete stage and polish could still have repaired. With a tolerance of 0.01 (`relax_tol`, mine only so far) a cold plan goes from USD 6.1 M to 7.5 M, no plan in 57 falls far below
  the last, the result is the same within noise (+586,825 against +594,876, `arms/showcase_run_corvus_tol002*.json`) and the seeds sit closer together (10.1-10.6 % against 9.9-11.0 %).
  On the oil field the same tolerance is neutral (+389,642 against +392,180, both before the slope fix, `arms/showcase_run_tol*.json`), so it stays at the default there. Two safety nets stay on: the running plan
  is a candidate after the polish (`kept_running_plan`), and a plan far below the last gets a second solve from another random stream (`second_opinion`). Neither fired in the final runs.

**A smaller re-plan budget does not hold up** (`arms/showcase_run_corvus_fast*.json`, `--discrete ga --set warm_relax=0.5 --set population=4`). It saves a quarter of the twin evaluations (3,232 per plan against 4,292) but ends at +569,294 (+10.0 %) against +586,825, with seeds much further apart (+545,826 to +588,513 against +583,760 to +592,133) and 56 of 57 plans feasible. The configured budget stays.

## Aquila steel group showcase run (96 h)

**Current code** (5 Nov 2026 night: the ALM's best-iterate fix, the exact polish, a quarter of the cascade budget,
`showcase_run_aquila_quarter*.json`, 6 seeds): operating profit USD +457,470 on the mean (+1.9 %; seeds -0.3 % to +4.9 %),
after carbon +1,641,307 (6 of 6 seeds ahead), CO2e -4.1 %; plans 2.3 times faster ("Planner speed" below). The tables
below are the runs of 4 Nov, before both; why PLYXIS trails and what changed: "Why PLYXIS trails on operating profit" below.

`bench/results/showcase_run_aquila.json` (seed 0), `showcase_run_aquila_s1.json`, `_s2.json` - the global steel group
(13 sites on four continents, 237 instances, 2,476 decisions per plan, see
[SHOWCASE.md](SHOWCASE.md#case-study-3-aquila-steel---planning-a-whole-corporate)): a blast furnace off wind for 30 h, a port
closed by a storm, an announced and then realised ERCOT scarcity window, a gas curtailment in India, a strip mill down, customers
pulling orders forward, a European scrap price jump and an arc-furnace transformer trip. Same set-up as the other case studies: two
copies of the group, same truth, events, prices and noise. The baseline is a competent corporate S&OP with site control rooms:
merit-order allocation of every market to the lanes by delivered margin at standard cost and standard CO2e (at the same internal
carbon price), sister-plant slab and DRI transfers, make-to-order mills with a stock correction, time-of-use and demand response at
the arc furnaces, the metallic mix by cost and the copper limit. Both worlds plan with the group's internal carbon price
(30 USD/t CO2e); operating profit does not contain it, profit after carbon does.

| seed | operating profit | after carbon | CO2e | shipped | crude steel | hot metal | arc-furnace steel |
|---|---|---|---|---|---|---|---|
| 0 | +177,459 (+0.8 %) | +1,947,706 | -6.1 % | -1.7 % | -1.2 % | -8.8 % | +11.9 % |
| 1 | -1,263,766 (-4.9 %) | +973,816 | -7.7 % | -2.7 % | -1.7 % | -10.6 % | +12.3 % |
| 2 | -174,762 (-0.7 %) | +1,249,706 | -4.9 % | -1.1 % | +0.3 % | -7.4 % | +12.4 % |
| **mean** | **-420,356 (-1.6 %)** | **+1,390,409** | **-6.2 %** | -1.8 % | -0.9 % | -8.9 % | +12.2 % |

The baseline earns USD 22-26 M of operating profit in the 96 h (about USD 2.0 bn a year) and USD -3 to -7 M after carbon: at
30 USD/t the group's emissions cost more than its margin. Percentages after carbon are therefore not shown.

Seed 0 in detail:

| KPI (96 h) | PLYXIS | conventional planning | difference |
|---|---|---|---|
| Operating profit (USD) | 22,337,437 | 22,159,978 | +177,459 (+0.8 %) |
| Profit after carbon cost (USD) | -4,801,658 | -6,749,364 | +1,947,706 |
| Revenue (USD) | 447,955,906 | 454,491,084 | -6,535,178 (-1.4 %) |
| Cash operating cost (USD) | 455,980,294 | 434,530,750 | +21,449,544 (+4.9 %) |
| Inventory value change (USD, at cost) | +30,361,825 | +2,199,645 | +28,162,180 |
| Steel shipped (t) | 629,715 | 640,313 | -10,598 (-1.7 %) |
| Crude steel (t) | 664,618 | 672,597 | -7,979 (-1.2 %) |
| Hot metal (t) | 450,763 | 494,391 | -43,628 (-8.8 %) |
| Arc-furnace steel (t) | 148,629 | 132,821 | +15,808 (+11.9 %) |
| Group power (MWh) | 222,794 | 219,888 | +1.3 % |
| Emissions (t CO2e) | 904,636 | 963,645 | -59,008 (-6.1 %) |

| Where the operating-profit difference comes from | USD |
|---|---|
| Inventories (ore and coke not charged, scrap bought into the bays, at cost) | +28,162,180 |
| Scrap bought | -23,800,364 |
| Sales | -6,535,178 |
| EU emission allowances | +2,002,732 |
| Merchant HBI | -1,774,119 |
| Coal injection, ironmaking | +1,526,992 |
| Power | +553,448 |
| Copper downgrades | -308,279 |
| Rolling, freight, casting | +525,015 |
| Grid connection overrun | -112,946 |
| Other | -62,023 |
| **total** | **+177,459** |

Plans: 31 per run (two random streams for the first), 88 of 93 feasible. The runs shared the machine, so their solve times
(mean 144 s, the first plan 270-350 s) mean little. Plant protection had nothing to cut in PLYXIS's seed 0; in seed 1 it trimmed
European scrap purchases to the market's supply (72 steps), and in seeds 1 and 2 Hazira's gas to the curtailed contract (59-96
steps; the baseline's 180). No stock was drawn below empty in either world.

**How to read this result.** PLYXIS does not beat a competent S&OP on operating profit for this group: -1.6 % on the mean,
one seed slightly ahead. It plans to the group's objective, profit after the internal carbon price, and there it is ahead on all
three seeds (USD +1.0 to +1.9 M per 96 h) with 6.2 % less CO2e. Almost all of it is the metallic mix. In Europe the plan runs
the blast furnaces near their 60 % turndown and the converters at their lowest hot-metal share: with the EU's emission allowances
and the internal carbon price, European hot metal costs the plan more than scrap at 395 USD/t (in a start plan the gradient of every
European furnace's wind is about zero - which was read here as an optimum; it is not, see "Why PLYXIS trails on operating profit"
below: the search stalls there, and the gap is not the carbon price). The bridge shows that trade: USD 26 M more scrap and HBI against the ore and coke left in the yards (valued at
cost) and USD 2 M of allowances, less steel made and sold. The merit-order allocation of markets to sites, which the baseline
does with standard costs, leaves little to gain: on the allocation layer the exact LP stage found nothing to improve after the
cascade.

**What this result does not settle.**
* *Furnace turndown is free in the model.* A blast furnace at 62 % of its wind uses the same coke and coal per tonne as at full
  wind, in the twin and in the truth. Real furnaces lose efficiency (fixed heat losses over fewer tonnes); the plan's European
  strategy is optimistic by that much. A fuel-rate penalty at low wind would be the next model change.
* *The strategy has no buffer.* With the European furnaces near turndown, the European crunch (Duisburg BF1 out from hour 5,
  IJmuiden closed from hour 18) cost seed 1 about 1,000 t/h of shipments from hour 26 and USD 1.4 M by hour 34 (the same happened
  at hour 31 in another run of that seed). The European scrap jump (470 USD/t from hour 60) meets the plan with its furnaces down.
* *Scrap bays have no top of the stock band.* Product yards do (110 % of target); scrap bays and DRI piles do not, so a plan buys
  whenever the market is a few dollars below the bay's book value - bays at up to 144 % of target, worth nothing at book but working
  capital. To be added with an A/B.
* *Runs of the same seed differ.* The plans depend on thread scheduling: a replay of seed 1 with identical code was USD 0.7 M
  ahead at hour 42 where the benchmark run was 0.15 M behind. Differences between the variants below are within that noise.

**The twin's own world** (`bench/results/policy_in_twin_aquila.json`, `python -m bench.policy_in_twin --scenario aquila`: the
engineering model as the plant, nominal prices, no events, one 96 h plan against the rules):

| | operating profit (USD) | after carbon (USD) | CO2e (t) |
|---|---|---|---|
| conventional rules | 28,926,111 | -618,983 | 984,836 |
| one plan | 33,046,209 | 6,251,482 | 893,158 |
| difference | +4,120,098 | +6,870,465 | -91,679 |

| Yards, change over the 96 h | rules | plan | plan - rules at unit cost |
|---|---|---|---|
| finished and semi-finished steel | +7,991 t | -28,727 t | USD -18.1 M |
| scrap bays | -383 t | +77,731 t | USD +31.2 M |
| ore | +31,319 t | +121,062 t | USD +12.0 M |
| coke | -3,601 t | +17,875 t | USD +7.3 M |

The plan's operating lead here is the stock it sells at its horizon end: 36,700 t more steel out of the yards than the rules, worth
about USD 4-5 M at margin over book. A rolling plan never reaches its horizon end, so this does not carry over to the A/B. What
does carry over is the carbon: 92 kt less CO2e, USD 2.8 M at the internal price.

**How this number was arrived at.** Three variants of the planning model, each over the same three seeds (USD M, operating / after
carbon; files in `bench/results/arms/`):

| variant | seed 0 | seed 1 | seed 2 | mean | CO2e | shipped |
|---|---|---|---|---|---|---|
| end-of-horizon stock value as a hinge at the target (`showcase_run_aquila_hinge*`) | -1.42 / +0.12 | -1.93 / -0.08 | -0.58 / +0.53 | -1.31 / +0.19 | -5.2 % | -2.3 % |
| smooth end value (`showcase_run_aquila_smooth*`) | -2.99 / -0.47 | +0.34 / +2.05 | -0.80 / +0.67 | -1.15 / +0.75 | -6.6 % | -2.5 % |
| smooth end value + two cold starts (current) | +0.18 / +1.95 | -1.26 / +0.97 | -0.17 / +1.25 | -0.42 / +1.39 | -6.2 % | -1.8 % |

* *Smooth end value.* Every yard ends a plan at its stock target (`value_to_go`, 200 USD per t short, planning only). As a hinge at
  the target - where every yard starts - the first tonne short already cost the full 200 USD, the same trap the running stock floor
  had been taken out of: no single link of furnace, mill, coating line and lane could move alone. Quadratic up to 10 % of the target,
  a start plan is worth USD 3.1 M more in the twin's own world (much of it stock sold at the horizon end, as above) and plans
  6,620-6,870 t/h of shipments instead of about 6,490. In the A/B the gain is within the noise.
* *Two cold starts.* Seed 0's first plan with the smooth end value came out USD 2-6 M below two other random streams' and shipped
  700 t/h less European coil than its contracts (USD -0.24 M in the first hour); every later plan started from it and the seed ended
  at -3.0 M. The second opinion compares a plan with the last one and so never fires for a first plan. The first plan is now solved
  from two streams (`cold_starts`, steel only so far); seed 0 then ends at +0.18 M.
* *Earlier, before any of these runs*, the economics were fixed in both worlds (SHOWCASE.md, case study 3): inventories at variable
  cost capped below NRV, contracts at about 80 % of capacity with a discounted spot, stock targets with a smooth floor and a top,
  2 USD per t and hour of backlog in the plan, the baseline's S&OP with the carbon price and the arc furnaces' real capacity, the
  forecaster's market tags.

### Why PLYXIS trails on operating profit (5 Nov 2026)

Three explanations were tested, each with its own run; the third is what the data supports.

**Not the console's lag.** `bench/results/console_run_aquila.json` (`python -m bench.console_run --scenario aquila`): the A/B
as the console runs it, the plant clock in wall time at 600x and plans computed in the background. A steel-group plan takes 10-12
plant hours there, so the plant ran on 8 plans instead of 31 - and seed 0 still ended where the benchmark, which waits for every
plan, did: operating profit USD +101,798 (benchmark +177,459), after carbon +1,990,418 (+1,947,706).

**Not the carbon price.** `showcase_run_aquila_nocarbon*.json`: the same A/B with the internal carbon price at 0 in the twin's
objective and in the baseline's standard costs (a copy of the scenario, `--scenario-dir`; EU allowances stay a real cost in both
worlds). If the operating gap were the price of the plan's lower CO2e, it would close:

| seed | operating profit, carbon priced | operating profit, carbon at 0 | shipped, carbon at 0 | hot metal, carbon at 0 |
|---|---|---|---|---|
| 0 | +177,459 (+0.8 %) | +57,002 (+0.3 %) | -13,929 t | -8,022 t |
| 1 | -1,263,766 (-4.9 %) | +1,110,464 (+4.3 %) | -3,824 t | -12,472 t |
| 2 | -174,762 (-0.7 %) | -2,535,952 (-9.9 %) | -11,107 t | -4,746 t |
| **mean** | **-420,356** | **-456,162** | -9,620 t | -8,413 t |

It does not: without the carbon price PLYXIS is as far behind, still ships less, and still buys USD 17-30 M more scrap than the
baseline (into the bays, at cost).

**The plans stall in a low-throughput local optimum.** `bench/results/plan_quality_aquila.json` (`python -m bench.plan_quality`,
the default planning problem, values in the planner's own objective after carbon):

| plan | value (USD) | shipped | of it spot | hot metal |
|---|---|---|---|---|
| the cascade (live budget), stored | 1,392,003,584 | 642,746 t | 43,336 t | 446,937 t |
| a better plan, stored | 1,394,723,456 | 667,803 t | 63,010 t | 468,411 t |
| cascade, 8 restarts, seed 0 / 1 (current code) | 1,391,994,240 / 1,391,100,288 | 642,527 / 634,612 t | 43,155 / 32,406 t | |
| cascade, 16 restarts | 1,392,544,000 | 640,947 t | 38,416 t | |
| sequential LP from the cascade's plan, 15 iterations (trust region 0.02, floor 2e-4) | 1,393,043,712 | 651,598 t | 49,460 t | 448,274 t |

The better plan was found by accident (a short relaxation of an earlier code version). In the same objective it is worth USD 2.7 M
more: it ships 3.9 % more, almost all of it spot (+45 %: European hot-rolled and coated coil, slab exports, South American coil),
and runs the French, Dutch and Spanish blast furnaces higher. So the European "scrap strategy" is not the optimum it was taken for.
Why a first-order search stops short of it: at a yard's stock target the smooth floor has zero slope - a lane that ships more
empties the yard, a furnace that blows more fills it, and neither gains alone; only the joint move earns the spot margin. More
random restarts move a cold plan by about USD 0.5 M, not into the better basin.

The exact layer sees the joint move: the LP on the twin's algebraic model links furnace, yard and lane through the yard's balance
row. From the cascade's plan the sequential LP gains USD 1.04 M in 15 iterations (70 s) and ships 8,852 t more, 6,124 t of it
spot - from a trust region of 0.005 with a floor of 1e-3 it stopped after 3 iterations with nothing. Its steps stay small because
the first-order model gets a few kinks wrong - measured on one LP step of 0.02 from the cascade's plan ("step"): the grid
connections' overrun charge (model -0.21 M, twin +1.04 M: the step pushes sites over their contracted load, where the slope below
the limit is 0), the spot discount (sales +5.27 M against +4.49 M), the backlog's shortfall and late-delivery costs (model -0.30 M,
twin +0.30 M) and the yards' end value; the only limit the step breaks is the markets' depth (0.61), through the backlog's
`pos()`. The twin's value of that step: USD 1,392.0 M -> 1,391.1 M, where the model expected a gain.

Where the polish ends depends on where it starts ("polish_starts", up to 40 iterations; it stops by itself when the model
promises nothing more):

| start plan | its value (USD) | after the polish | gain | shipped |
|---|---|---|---|---|
| the cascade's (stored) | 1,392,003,584 | 1,393,112,704 | +1.11 M | 651,813 t |
| the better one (stored) | 1,394,723,456 | 1,394,884,352 | +0.16 M | 669,379 t |
| a cold plan of the current code, seed 1 | 1,391,086,336 | 1,394,632,576 | +3.55 M | 655,106 t |
| a cold plan of the earlier code, from a top-of-range start | 1,393,333,632 | 1,395,947,136 | +2.61 M | 664,389 t |

Polishing several candidates and keeping the best would have found USD 3.9 M more than the cascade's plan in this problem - the
next thing to try in the planner (the first plan already comes from two cold starts; polish both).

**A bug on the way.** The ALM stored its best iterate one optimiser step after scoring it (it copied the plan tensor after
`opt.step()` had moved it in place), so a feasible warm start was never kept: the relaxation turned a plan worth USD 1,394.7 M into
one of 1,387.2 M at violation 0.29. Fixed (`gradient.py`, `test_alm_returns_the_iterate_it_scored`). It did not cause the stall:
the current code's cold plans land where the old ones did (table above). Every A/B result above predates the fix. On the other
twins the fix is neutral to slightly positive (`showcase_run_almfix*.json`, `showcase_run_corvus_almfix*.json`, 3 seeds each):
the oil field USD +403,607 per 72 h on the mean (before +392,840; every seed higher, by 5-14 k), the mine +576,462 per 48 h
(+583,368; seeds closer together, 574-579 k against 555-600 k) - the headline numbers of both stay as measured before it.

**The A/B with the fix** (`showcase_run_aquila_almfix*.json`, current code, otherwise as the benchmark):

| seed | operating profit | after carbon | CO2e | shipped | hot metal |
|---|---|---|---|---|---|
| 0 | +83,107 (+0.4%) | +1,956,414 | -62.4 kt | -11.7 kt | -45.2 kt |
| 1 | -2,679,432 (-10.4%) | -400,420 | -76.0 kt | -28.8 kt | -53.4 kt |
| 2 | -293,317 (-1.2%) | +1,371,569 | -55.5 kt | -11.8 kt | -41.2 kt |
| **mean** | **-963,214** | **+975,854** | -64.6 kt | -17.4 kt | -46.6 kt |

No better: seeds 0 and 2 end within run-to-run noise of the benchmark, seed 1 is USD 1.4 M worse. Every seed was ahead by hour
40-48 (USD +0.33 to +1.44 M) and lost it in the second half, steadily, by shipping less: in seed 2's last twelve hours PLYXIS
shipped 6,093-6,710 t/h against the baseline's 6,807-6,853 t/h. In seed 1 the shipments fell from 6,883 t/h (hour 38) to
4,964 t/h (hour 49) while crude steel stayed at the baseline's level; the plan of hour 38.17 was worth USD 9 M less than the
one before it and did not keep the running plan. The same failure the benchmark's seed 1 showed in the European crunch, larger. It is not deterministic: a replay of seed 1 with
the same code to hour 38.5 (`bench/results/arms/replay_aquila_s1_h38.json`) was USD 1.02 M ahead there, shipping 6,437 t/h, and
its plan of hour 38.17 ships 6,478 t/h over the next twelve hours against 6,300 t/h of contracts with the yards at their targets -
runs of one seed diverge with thread scheduling.

**The A/B with the exact polish** (`showcase_run_aquila_almfix_slp*.json`; the fix and `slp_iters` = 15, the sequential LP on the
twin's algebraic model after every cascade, trust region from 0.02 down to 2e-4, kept only where the plan's F rises):

| seed | operating profit | after carbon | CO2e | shipped | hot metal | polish kept |
|---|---|---|---|---|---|---|
| 0 | -333,243 (-1.5 %) | +1,526,046 | -6.4 % | -13.7 kt | -45.3 kt | 19 of 30 |
| 1 | +344,281 (+1.3 %) | +2,167,944 | -6.3 % | -9.9 kt | -44.4 kt | 21 of 30 |
| 2 | -274,077 (-1.1 %) | +1,314,634 | -5.5 % | -9.2 kt | -39.8 kt | 24 of 30 |
| 3 | +926,476 (+4.7 %) | +2,027,450 | -3.9 % | -3.6 kt | -31.1 kt | 21 of 30 |
| 4 | +195,366 (+0.7 %) | +1,818,719 | -5.6 % | -7.3 kt | -41.2 kt | 22 of 30 |
| 5 | +676,431 (+3.1 %) | +2,448,895 | -6.1 % | -7.7 kt | -43.7 kt | 23 of 30 |
| **mean, seeds 0-2** | **-87,680 (-0.4 %)** | **+1,669,541** | -6.1 % | -10.9 kt | -43.2 kt | |
| **mean, all 6** | **+255,872 (+1.1 %)** | **+1,883,948** | -5.6 % | -8.5 kt | -40.9 kt | |

Seeds 3-5 were run afterwards with the same code to firm up the current result (no benchmark or fix-only run exists for them;
each run is its own A/B). The best of the three arms on seeds 0-2: operating profit USD -0.09 M on the mean (benchmark -0.42 M, fix alone -0.96 M),
after carbon +1.67 M (+1.39 M, +0.98 M). Seed 1's collapse did not happen (+0.34 M where the fix alone ended at -2.68 M); seeds 0
and 2 end within the run-to-run noise of the other arms. Over all six seeds the mean is +1.1 % (USD +255,872; seeds -1.5 % to
+4.7 %), after carbon +1.88 M on every seed - a lead, but within the spread of the seeds (about USD 0.7 M of noise per seed), and
seeds 3-5 have no old-code run to compare with. It is the arm that attacks the cause, and it is now on for the steel group (`scenario.json`
planner `slp_iters` 15; each plan takes 23-32 s longer, 5-8 iterations). The polish sees the joint move the first-order search
misses; what it cannot do yet is take large steps (the kinks above), so the plans still ship 9-14 kt less than the baseline's.

## The block tree of every twin

`bench/results/structure.json` (`python -m bench.structure`; `python -m plyxis structure <file.plx> --detail` for one
twin) - `plyxis/lang/blocks.py` cuts each twin's asset graph at its shared exchanges, time scales and large stocks,
level by level; no rule knows what a plant, a site or a line is. Found in under 3 s per twin:

| Twin | blocks | levels | what the cuts find | used by the plan |
|---|---|---|---|---|
| Kestrel (oil field) | 15 | 2 | the three central batteries (33-60 h of their flow) separate the twelve well pads - the partition the solver was given by hand so far (`partition = "pad"`); the compressor station splits at its exchange | none |
| Corvus (mine) | 11 | 1 | the ROM pad (65 h) separates the mine's ore network (238 units, one cell: ore passes of 1.8-2.5 h couple stopes, loaders and fans) from the ten dewatering pump stations | none |
| Aquila (steel group, hourly) | 94 | 3 | ten regional networks at the markets they share; sites at their slab and coil yards; lines at a site's grid connection (Brazil: its blast-furnace chain and its strip mill) | none |
| Aquila with the Duisburg works | 93 | 3 | the same, and inside the European network two blocks at 15 minutes (`DE.converter`, `DE.hsm_works`, with the converters' tap to tap, the casters' tundish life and the mill's roll campaign as exact per-unit DPs); Velsen splits at its grid and gas connection into its iron and steel chain, its strip mill and its coating line | the two 15-minute blocks |

The plan uses the blocks at a finer step than the part around them - the ones measured to pay (below). Blocks at the
same step as their part are found and reported: coordinated by prices they did not converge on the steel group
(regional networks, below), so they stay in their part's program for now.

## One twin, plants found in it: the steel group hourly, the Duisburg works' plants at their own step

`bench/results/corporate.json` (`python -m bench.corporate --rounds 3 --workers 6 --reference-from
bench/results/arms/corporate_detected_before_commit_retry.json`; the reference, the whole twin solved by the cascade,
is unchanged by the coordinator's last fix and comes from that run of `--reference`) -
`scenarios/aquila_works`: the Aquila group with the Duisburg works modelled machine by machine (three blast furnaces
that burn more coke at low wind, the torpedo pool and pig caster, three converter vessels heat by heat, the ladle
furnace, two slab casters with their tundish sequences, a slab yard with hot slabs, two reheating furnaces on process or
natural gas, the hot strip mill with its work-roll changes, the galvanising line, the works' gas network and power
plant). One twin, planned hourly; nothing declares a plant. The blueprints state their durations in time (tap to tap
45 minutes, a tundish ten hours, set points held 15 or 30 minutes) and `plyxis/lang/hierarchy.py` finds:

* **DE.converter** (11 units at 15 minutes): the three converters need 15-minute steps (their tap-to-tap window spans no
  whole number of coarser steps), the casters 30 (speed held 30 minutes); the blast furnaces and the pig caster join
  because they send into the torpedo pool, which holds 2.1 h of its flow, the ladle furnace 1.2 h. Its fence: the
  slab yard (144 h of flow), the ore, coke and scrap bays (845-1,780 h), the grid connection and the gas network
  (no stock: they take the flow hour by hour);
* **DE.hsm_works** (3 units at 15 minutes): the strip mill (roll changes held 15 minutes) and the two reheating
  furnaces (30 minutes). Its fence: the slab yard, the coil yard (131 h), the gas network, the gas supply, the grid.

The slab yard cuts the works in two, as works are run. Each plant's sequence rules are found in its equations and
solved by exact per-unit dynamic programs (`discrete = "auto"`): the converters' tap to tap (one heat per window, from
a running count), the casters' tundish life (longest run), the mill's roll campaign (reset).

| | group (master) | DE.converter | DE.hsm_works | the whole twin as one program at 15 min |
|---|---|---|---|---|
| instances | 249 (incl. 2 x 7 column slots, 2 simplexes) | 17 (incl. 6 boundary sinks) | 9 (incl. 6) | |
| steps | 96 x 1 h | 384 x 15 min | 384 x 15 min | 384 x 15 min |
| decisions (discrete) | 2,572 | 2,040 (1,344) | 1,440 (384) | 13,096 (1,728) |
| plan value (USD) | **1,395,638,144**, feasible (schedules' own limits: 0.0002, 0.0000) | | | 1,375,778,944, violation 0.022 |
| seconds | 363 (all rounds, blocks on 6 worker processes) | | | 1,140 |

Both with the same budget per program (the works' larger one: relaxation 800, repair 300, polish 800, restoration 200).
Round 0 solved each plant at 0, 0.7 and 1.0 times the group's prices as it stands, side by side; the steel side broke
its torpedo and ladle limits at 0 and 0.7 (0.014, 0.015: at part load the furnaces' minimum wind forces taps and casts
to match around a ladle furnace of four heats), so it was also solved at 1.3 and 1.6 and the two infeasible schedules
were kept out of the master. Round 1's new schedules were worth USD 0.2 M and 1.2 M more than the master's mix at its
prices (below 0.1 % of the plan): converged. Both plants' schedules near the master's mix kept their limits, and the
group re-planned around them was feasible at the first commit.

Duisburg's committed schedule: 82, 86 and 82 heats on the three vessels in 96 hours, 10 and 9 tundish sequences on the
two casters, 18 work-roll changes, no pig iron, no gas flared, blast furnaces at 881 t/h, the mill at 525 t/h.

What this shows and what it does not:

* **Found, not declared, and better than one program.** The decomposition the detector found beats the whole twin
  solved at once by USD 19.9 M (1.4 %) in 96 hours at 0.32 of the time, and it is feasible where the reference is not.
  Runs of the same code vary: the run before (`arms/corporate_detected_before_commit_retry.json`, identical but for
  the commit's retry) committed USD 1,393.3 M in 446 s - its first commit's mill schedule missed its limits by 0.0026,
  the commit fell back to a pool schedule rolling 397 t/h, and the group was re-planned around less coil (the retry
  now solves such a schedule once more from itself first; in the run above it was not needed). The reference ended at
  USD 1,376.0 M and violation 0.0021 in an earlier run (2,544 s, before the faster mill counter below;
  `arms/corporate_detected_first_run.json`).
* **Side by side halves the time.** The same decomposition with the blocks solved one after another, each on six
  threads (`corporate_sequential.json`, `--workers 0`): USD 1,391.1 M, feasible (own limits 0.0010, 0.0000), in 730 s
  - each plant solve takes 20-48 s on six threads against 46-94 s for a worker on one, but eight start solves side by
  side beat eight in a row. Its start columns are solved warm from each other (the steel side kept its limits at 0.7x
  this way) and it ran all three rounds (reduced costs USD 3-4 M in round 1); the USD 4.5 M between the two runs is
  within the spread of runs of the same code.
* **The same as declaring it.** On the toy (`examples/plants_toy.plx`: a maker that changes its rate every half hour, a
  small store, a lane; a market with a demand limit, a dearer supplier) the detector finds the plant a hand-written
  group named before, and the committed plan is the same: USD 37,069 against 36,052 for the whole model at once
  (hand-worked optimum 37,100). The declared Duisburg plant of the previous version (one plant of 31 units including
  yards, lanes, grid and power plant; `arms/corporate_declared_plant.json`): USD 1,387.1 M in 1,817 s against a
  reference of 1,369.8 M at a smaller budget.
* **What made it work** (each found on this run's predecessors): only schedules that keep a block's own limits are
  points of its input-output function (the coordinator keeps the others out of the master and adds start columns at
  higher prices where fewer than two keep them); a committed plan is feasible only if every block's schedule is; a
  feasibility phase in the per-unit DPs (units one at a time on the violations alone) - without it the steel side's
  rounded schedule broke the ladle furnace's limits by 0.11 and the value sweeps made it worse; the mill's roll-wear
  counter vectorised as an associative scan (49 to 15 ms per evaluation, the same values).
* **A heat schedule needs its own subsolver.** Each plant alone at the group's prices as it stands
  (`bench/results/corporate_plant.json`, `--plant-ablation`): on the steel side the cascade's GA ends at violation 2.0
  (two taps inside one cycle, two vessels in the ladle furnace at once), the per-unit dynamic programs at 0.0057 with
  the group's live budget (10 s) and 0.0003 with a larger one (relaxation 400, repair 150, polish 400: 21 s, 241
  heats); on the hot-rolling side the GA at 0.37 (furnace content and mill capacity), the DPs at 0.030 and 0.007 -
  feasible only with the benchmark's own budget above. (The declared plant of the previous version, with the old
  stage: GA 2.0, DPs 0.009 / 0.0013 in 197 / 333 s; `arms/corporate_plant_declared_plant.json`.)
* **Duisburg still runs below its capacity**: the mill at 525 t/h of its 1,350 (the master's mix asked for 612 t/h of
  coil), the reference at 498. Whether the group's optimum is a fuller mill is not checked.
* **Not measured: an A/B.** No hidden truth and no conventional baseline at works level yet: the planning layer only.


### Regional networks: found, but coordinating them by prices does not pay yet

`bench/results/corporate_regions.json` (`python -m bench.corporate --regions --rounds 5 --workers 6`) - the Aquila group
hourly (`scenarios/aquila`, no works detail). `detect_regions` finds ten regional networks at the exchanges they
share (the steel markets several networks sell to, the scrap markets they buy from, Spain's grid connection and gas
supply shared by its two networks): DE+FR+NL (53 units, 372 decisions, trading at the European coil and scrap
markets), AR+TX (508 decisions), SA (340), GJ (316), BR+IN (228), MX (184), PL (168), ES2 (168), OD (120), ES1 (72).
The master keeps no decisions of its own: 840 column weights.

| | the whole twin as one program | ten regional networks, coordinated by prices |
|---|---|---|
| plan value (USD) | **1,392,000,512**, feasible | 1,390,109,056 committed, violation 0.63 (market depth) |
| seconds | 47 | 642 (5 rounds, 6 worker processes) |

* **Phase I was needed and is not enough.** The first master mixing the networks' start schedules broke a scrap
  market's supply limit (violation 0.39, recorded before Phase I stepped in): every network buys scrap as if it were
  plentiful, and an infeasible master's multipliers are not used as prices, so nobody learns the scarcity. Without
  Phase I (`arms/corporate_regions_no_phase1.json`) every master stayed at violation 0.31-0.35, and the commit ended at
  USD 1,394.9 M with violation 1.0 in 1,102 s. With an artificial schedule per block the masters come within 0.008-0.039, but they still lean on it
  (Poland 97 %, ES2 89 %, GJ 36 % of the weight) and the networks' new schedules are still worth USD 340-480 M more at
  the master's prices after five rounds: the shared markets' depth limits bind. More rounds do not fix it
  (`corporate_regions_r12.json`, twelve rounds): the masters stall at USD 0.80-0.82 B with the artificial schedules,
  the reduced costs fall to a plateau of USD 400-430 M from round 8, the commit ends at USD 1,392.5 M with violation
  0.97 (19 minutes). Price coordination alone does not converge here; a stabilised dual or a
  resource-directive split of the shared limits (each network its share of a market's depth) is the next thing to try.
* **Small blocks are not cheap.** A regional solve took 35-95 s on one thread - the cascade's number of evaluations is
  fixed, and a small program's evaluation costs nearly as much as a large one's - against 47-67 s for the whole group
  on six threads. At 2,476 decisions the whole program wins; decomposition at the same step is for twins too large to
  solve at once.
* So `split = "auto"` decomposes into plants (time scales) only; regional networks are detected and available with
  `split = "regions"` (or `"both"`: plants and networks side by side), not used by default.

## The exact layer: the twin as algebra, and the reduction engine

`bench/results/reduction.json` (`python -m bench.reduction`; method in
[OPTIMIZER.md](OPTIMIZER.md#the-exact-layer-the-twin-as-algebra-algebrapy-exactpy-reducepy)). Timing-sensitive, run
alone, 12 threads.

**The algebraic model is the simulator's own.** On every case study `linearize` reproduces the simulator at a
random plan (objective and every constraint row within float32 precision, state updates within 1e-7 of the state's
size) and predicts a 2 % step of all continuous decisions on its affine rows to 1e-4 of the change; nonlinear rows
are first order (`tests/test_reduce.py`). Built in 0.10-0.24 s per twin:

| twin | linearise s | rows | nonzeros | filters s | blocks found | largest block (integers) | master columns |
|---|---|---|---|---|---|---|---|
| Kestrel (oil field) | 0.11 | 22,320 | 454,998 | 0.35 | 8 (16 wells each) | 192 | 759 |
| Corvus (mine) | 0.10 | 68,304 | 226,627 | 0.85 | 120 (stopes, mostly in pairs) | 156 | 3,264 |
| Aquila (steel group, hourly) | 0.12 | 43,584 | 127,974 | 0.27 | 0 (no discrete decisions: one LP) | 0 | 14,188 |
| Aquila with the Duisburg works | 0.24 | 47,416 | 144,851 | 0.36 | 5 (converters, casters, reheating furnaces with the mill) | 192 | 14,740 |

The mine's model holds 8,928 piecewise terms of discrete decisions encoded exactly (indicators), the works' 190.

**On an affine twin one LP is the optimum.** The widget toy (`examples/plants_toy.plx`, no peak hours): one HiGHS LP,
USD 43,100, feasible on the twin - the hand-worked value. On the small discrete twin `tests/models/kilns.plx` the
whole MILP's plan is worth on the twin exactly what the model says (F 0.232344, no violation): with minimum runs
and start counts encoded exactly, the MILP is the twin's global optimum. Before that encoding the same MILP's plan
broke the minimum run on the twin (violation 1.0): linearised at "all on", the rule sat on its kink and vanished.

**Sequential LP against the cascade** (steel group, both from the default plan, judged on the twin; the cascade with
the live budget and no neural start):

| | F | plan value (USD) | max violation | seconds |
|---|---|---|---|---|
| start (default plan) | -16.10 | 1,283,211,904 | 0.33 | |
| sequential LP, 25 iterations | 0.61405 | 1,392,019,200 | 0 | 56 (LP 53) |
| cascade | 0.60956 | 1,392,005,504 | 0.00053 | 52 |

The same plan value, without the cascade's small violation - with no tuning and no learned start. Not yet on twins
with on/off decisions: on the oil field the sequential MILP from the default plan stalled at F -37.5 (violation
1.39) - a well's pump curve times its on/off switch changes the model more than first order can say. It needs the
cascade's relaxation as its start (open work).

**Reduction engine against the whole-model MILP as a twin grows.** N kiln sites (three on/off kilns each, costs
varied by site, minimum runs), all selling into one market that takes 55 t/h per site: the shared limit binds,
so the sites must be coordinated. Both solve the same exact model; every plan is simulated on the twin (all
feasible, violation below 1e-7):

| sites | binaries | whole MILP s | whole F | engine s | engine F | engine bound | gap to bound | rounds | blocks |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 270 | 1.1 | 0.21275 | 0.8 | 0.21247 | 0.21370 | 0.57 % | 4 | 3 |
| 2 | 540 | 0.1 | 0.22745 | 1.9 | 0.22745 | 0.22745 | 0.00 % | 8 | 6 |
| 4 | 1,080 | 0.6 | 0.22532 | 2.5 | 0.22464 | 0.22538 | 0.33 % | 6 | 12 |
| 8 | 2,160 | 1.6 | 0.23303 | 4.9 | 0.23294 | 0.23303 | 0.04 % | 6 | 24 |
| 16 | 4,320 | 1.9 | 0.22992 | 15.8 | 0.22985 | 0.22994 | 0.04 % | 8 | 48 |
| 32 | 8,640 | 88.8 | 0.22962 | 39.2 | 0.22960 | 0.22963 | 0.01 % | 10 | 96 |
| 64 | 17,280 | 27.0 | 0.23141 | 110.1 | 0.23137 | 0.23141 | 0.02 % | 15 | 192 |

What this says, and what not:

- The filters find the kilns as units without being told; the engine's bound is valid at every size (never below the
  whole MILP's optimum) and its plan is within 0.04 % of it from eight sites on - the duality gap of coordinating
  many small units shrinks with their number, as in the column-generation benchmark.
- On this family HiGHS solves the whole MILP faster at most sizes: its LP relaxation is tight (the engine's bound
  equals the optimum to five digits). Its time is erratic - 1.9 s at 16 sites, 88.8 s at 32, 27.0 s at 64 - the
  engine's grows steadily, but faster than linearly: 15.8 -> 39.2 -> 110.1 s per doubling, because the rounds grow
  (8 -> 15) and the commit still solves one whole-model MILP over the undecided integers (62 s of the 110 s at 64
  sites; pricing 41 s - 2,880 block MILPs at about 14 ms each, one process).
- So there is no speed-up to claim yet. What the engine adds today is the structure (blocks, master, prices) and a
  certificate; the time it needs is in its commit and in per-call overhead, both removable (open work in
  CLAUDE.md: block MILPs in parallel processes, a warm-started master, a commit that stays inside the blocks).
- On the works twin at the default plan (trust region 0.1) the whole MILP stopped at its 300 s limit with a 1.9 %
  gap, where the engine's bound and master agreed within 2e-6 - but its committed plan was far below both, and that
  run predates the exact piecewise terms and the current commit. Not in the results file; to be re-run.

## Planner speed and the exact polish on every twin

The exact polish (`CascadeConfig.slp_iters`, [OPTIMIZER.md](OPTIMIZER.md#the-exact-layer-the-twin-as-algebra-algebrapy-exactpy-reducepy))
is now the cascade's last stage for any twin: the whole twin's continuous decisions by sequential LP on its algebraic model,
discrete decisions fixed where the discrete stage put them, HiGHS LPs warm-started from the last basis. It runs by default
where a twin has no discrete decisions (`slp_iters = None`); where it has them, a discrete stage decides and the polish was
measured neutral. The mine's branch stage (structure classes found in the twin's equations) stays where it is and now
warm-starts its LPs too.

**Plan time** (`bench/results/planner_speed.json`, `python -m bench.planner_speed`, alone on the machine, one plan per row
on each twin's default problem; "warm" starts from the cold plan shifted by the re-plan interval):

| twin | planner | cold plan | plan value (USD) | warm plan | plan value (USD) |
|---|---|---|---|---|---|
| oil field | now (GA, no polish - the same as on 5 Nov) | 10.7 s | 4,409,946 | 10.7 s | 4,417,753 |
| oil field | + the section search after the GA | 15.7 s | 4,409,946 | 13.8 s | 4,417,753 |
| oil field | a quarter of the budget + the exact polish | 6.5 s | 3,809,069 | 7.9 s | 4,262,134 |
| mine | now (GA alone, branch stage with warm LPs) | 22.2 s | 7,638,470 | 21.7 s | 7,645,828 |
| mine | + the section search (its planner until 6 Nov) | 26.9 s | 7,646,505 | 26.0 s | 7,625,270 |
| mine | a quarter of the budget + the exact polish | 23.2 s | 7,640,077 | 15.9 s | 7,557,121 |
| steel group | 5 Nov morning (full budget, no polish) | 46.6 s | 1,392,005,760 | 46.5 s | 1,393,234,944 |
| steel group | now (a quarter of the budget + the exact polish) | 19.3 s | 1,394,019,968 | 21.1 s | 1,395,093,760 |

6 threads, one plan per row. The steel group plans 2.3x faster and better (USD +2.0 M cold, +1.9 M warm in its own
objective); the stages now take relaxation 6 s, ALM polish 6 s, exact polish 7-9 s (were 25 s and 22 s). On the oil
field and the mine a quarter budget is faster but loses plan value (USD -0.6 M cold on the oil field, -0.09 M warm on the
mine), so they keep their budgets; the section search costs 4-5 s per plan on both (29-47 % of the oil field's plan time,
20-21 % of the mine's) for no consistent plan value. Single plans are not bit-for-bit repeatable (thread scheduling moves
the mine's by a few thousand dollars); the steel group's gap is far outside that.

**A/B, the steel group at a quarter of the cascade budget** (`showcase_run_aquila_quarter*.json`: relaxation 50, repair 20,
ALM polish 30 iterations instead of 200 / 80 / 120, and the exact polish), six seeds, against the full budget with the polish
(`showcase_run_aquila_almfix_slp*.json`, the same six seeds):

| seed | operating, quarter | operating, full | after carbon, quarter | after carbon, full | CO2e, quarter | shipped, quarter |
|---|---|---|---|---|---|---|
| 0 | -65,619 (-0.3 %) | -333,243 | +1,325,404 | +1,526,046 | -4.8 % | -4.7 kt |
| 1 | +143,808 (+0.6 %) | +344,281 | +1,570,059 | +2,167,944 | -4.9 % | -5.5 kt |
| 2 | +396,983 (+1.6 %) | -274,077 | +1,455,668 | +1,314,634 | -3.7 % | -0.2 kt |
| 3 | +962,260 (+4.9 %) | +926,476 | +1,752,100 | +2,027,450 | -2.8 % | -0.8 kt |
| 4 | +1,099,762 (+4.1 %) | +195,366 | +2,179,452 | +1,818,719 | -3.7 % | +4.2 kt |
| 5 | +207,626 (+1.0 %) | +676,431 | +1,565,160 | +2,448,895 | -4.7 % | -6.3 kt |
| **mean** | **+457,470 (+1.9 %)** | +255,872 | **+1,641,307** | +1,883,948 | -4.1 % | -2.2 kt |

Faster and as good: operating profit USD +0.20 M better on the mean (better on four of six seeds), after carbon 0.24 M lower
(lower on four of six), both within the noise of about USD 0.7 M per seed; it ships more (2.2 kt less than the baseline
instead of 8.5 kt) and so emits more (CO2e -4.1 % instead of -5.6 %). Plans take a median 44-63 s instead of 145-147 s
in these runs (several runs shared the machine; alone: the table above). This is now the steel group's planner.

**A/B, the oil field with the exact polish** (`showcase_run_polish*.json`, three seeds): USD +402,831 per 72 h on the mean
against +403,607 without (`showcase_run_almfix*.json`) - neutral, although one plan gains USD 40-60 k in the twin's own
objective (single-plan gains do not predict the A/B, again). The polish stays off there (the default for twins with
discrete decisions).

**A/B, the mine** (48 h, seeds 0-2, every arm after the ALM's best-iterate fix):

| arm | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| ALM fix (`showcase_run_corvus_almfix*.json`) | +575,860 | +574,417 | +579,110 | +576,462 |
| + branch-stage LPs warm-started (`_final*`) | +595,477 | +610,604 | +562,950 | +589,677 |
| + the GA alone, no section search (`_ga*`, now the mine's planner) | +598,310 | +606,239 | +581,996 | **+595,515 (+10.5 %)** |
| ALM fix + the exact polish after the branch stage (`_polish*`) | +558,409 | - | +586,671 | +572,540 (2 seeds) |

The warm start changes which optimal vertex HiGHS returns, not the LP; the three upper rows are within the seeds' noise of each
other and of the headline (+583,368, before the fix). The section search had no measurable gain ("Deterministic sections") and
costs 5-10 s per plan, so the mine now runs the GA alone. The polish on top of the branch stage gained nothing on two seeds; on
the third one of its LPs ran for over 15 minutes and the run was stopped - since then every LP of the polish and of the branch
stage has a 10 s limit (a timed-out LP is a rejected step) and the polish 60 s in all. The plan times in these A/B runs mean
nothing (several runs shared the machine).

## Deterministic sections (learned graph filter + exact section search)

`bench/results/decomposition.json` - the cascade with the GA discrete stage alone, with the exact section search
on the GA's best plan (`ga+sections`), and with the section search on the polished plan, accepted after its own polish (`ga+late`).
3 seeds each, same budgets, cold start, default prices:

| Case study | decisions (discrete) | method | mean objective (USD) | vs GA | feasible | mean seconds |
|---|---|---|---|---|---|---|
| Kestrel | 4,983 (1,272) | GA | 4,208,529 | - | 3/3 | 5.7 |
| Kestrel | 4,983 (1,272) | GA + sections | 4,249,347 | +1.0 % | 3/3 | 8.5 |
| Kestrel | 4,983 (1,272) | GA + sections after the polish | 4,222,577 | +0.3 % | 2/3 | 8.1 |
| Corvus | 4,992 (744) | GA | 7,478,371 | - | 3/3 | 40.0 |
| Corvus | 4,992 (744) | GA + sections | 7,476,820 | -0.0 % | 3/3 | 48.8 |
| Corvus | 4,992 (744) | GA + sections after the polish | 7,513,248 | +0.5 % | 3/3 | 50.0 |

| Case study | sections (largest) | colours | filter: rank correlation with exact couplings | filter forward | probing all couplings exactly | one sweep native / NumPy |
|---|---|---|---|---|---|---|
| Kestrel | 76 (3 assets) | 8 | 0.73 | 1.0 ms | 0.3 s (442 pairs) | 0.99 s / 1.04 s |
| Corvus | 46 (6 assets) | 10 | 0.07 | 1.8 ms | 1.1 s (665 pairs) | 2.15 s / 2.17 s |

Scaling - enterprise twins of N Kestrel fields:

| Fields | decisions | discrete units | sections (largest) | colours | one exact section sweep | one simulation |
|---|---|---|---|---|---|---|
| 1 | 4,983 | 624 | 76 (3) | 8 | 1.2 s | 10 ms |
| 2 | 9,966 | 1,248 | 152 (3) | 8 | 1.9 s | 10 ms |
| 4 | 19,932 | 2,496 | 304 (3) | 8 | 2.7 s | 15 ms |
| 8 | 39,864 | 4,992 | 688 (3) | 9 | 3.1 s | 21 ms |

Reading:
* **The complexity claim holds.** Sections stay small (3 assets, ≤ 64 joint options per window) while
  their number grows linearly with the site (76 → 688), the colouring stays at 8-9 colours, and one exact
  sweep over all sections grows sub-linearly in wall time (1.0 s → 3.2 s for 8x the decisions): the work is
  exponential only inside a section.
* **No measurable plan gain yet, also with the two changes that were meant to give one.** Charge-up crews make the mine's ring firings
  really compete (sections 186 → 46, largest 6 stopes instead of 1), and the section search can now run after the polish. Plan value against
  the GA alone: mine -0.0 % (before the polish) and +0.5 % (after it); oil field +1.0 % and +0.3 %. A previous run showed -0.0 % on the unchanged
  oil-field model and the one before +1.0 %, so that number is noise. It costs 3-10 s per plan.
  Over the full oil-field A/B the section search adds nothing either: +394,889 against +392,180 (three seeds, both before the slope fix, `arms/showcase_run_sections*.json`), within the noise. The oil field stays on the GA alone;
  the mine kept its setting (`ga+sections`) until 6 Nov, when its A/B without the search came out no worse (+595,515 against +589,677,
  three seeds, `showcase_run_corvus_ga*.json` / `_final*.json`, "Planner speed" above): the mine now runs the GA alone too. The search
  after the polish is an option (`late_sections`) that has not been A/B tested on the mine.
* **The graph filter does learn real structure where there is some** (Kestrel: 0.73 rank correlation
  with the exact cross terms) and replaces the exact probing at ~1-2 ms per call - on these sites the
  probing itself is still cheap (0.4-1.1 s); the filter matters for larger sites and repeated re-planning.
  On the mine the couplings between discrete decisions stay weak (rank correlation 0.07) even with the shared crews, because a crew limit
  only couples firings when it binds, and in these plans it rarely does.
* **Native kernels.** The C++/OpenMP kernels give identical results to the NumPy reference; inside a
  sweep they make no measurable difference (0.99 s vs. 1.03 s on Kestrel) because the batched twin
  simulation dominates the sweep. They are kept for the combinatorial core as sites grow.

## Section DP vs. exact MILP (scheduling models with known structure)

`bench/results/section_theory.json` - the decomposition core (`sectiondp.py`, theory in
[DECOMPOSITION.md](DECOMPOSITION.md)) on table models built like a site plan: S sections x 6 shifts, every
block choosing one of 8 joint options (3 machines on/off); strong interactions over time (a section in
consecutive shifts), moderate ones along a random network tree (one shift), weak ones on random extra edges
that close cycles (10 % of the blocks), and one shared crew capacity (at most 30 % of the machine-shifts).
Methods: **section DP** (the strongest forest solved exactly, exact moves over random induced forests,
dual decomposition for the certified bound), block coordinate ascent from the empty plan, multi-start
coordinate ascent (8 random feasible starts), and the exact MILP of the same model (HiGHS; one-hot options,
pairwise marginals, the crew knapsack) with a 120 s limit (not run above 9,000 blocks).

| blocks | section DP | certified gap | DP seconds | coordinate ascent | multi-start | MILP best | MILP bound | MILP seconds | MILP optimal |
|---|---|---|---|---|---|---|---|---|---|
| 120 | 304.8 | 2.5 % | 0.14 | 188.9 | 238.0 | 307.0 | 307.0 | 8 | yes |
| 360 | 952.7 | 3.9 % | 0.23 | 588.3 | 736.1 | 956.4 | 959.1 | 120 | no |
| 1,200 | 3,239.4 | 9.3 % | 0.64 | 1,888.3 | 2,519.0 | 3,006.2 | 3,254.3 | 121 | no |
| 3,600 | 9,611.8 | 13.2 % | 2.95 | 5,931.7 | 7,371.2 | 8,393.7 | 132,776.0 | 124 | no |
| 9,000 | 24,001.7 | 14.4 % | 7.10 | 14,793.1 | 18,333.2 | 20,933.8 | 331,975.5 | 220 | no |
| 30,000 | 79,787.7 | 18.5 % | 27.38 | 49,257.6 | 60,622.8 | - | - | - | - |
| 90,000 | 239,360.3 | 19.3 % | 95.18 | 146,512.5 | 181,789.0 | - | - | - | - |

Stronger cycles (1,200 blocks, extra edges for 30 % of the blocks, weak-edge scale varied):

| weak-edge scale | section DP | certified gap | MILP best (120 s) | MILP bound |
|---|---|---|---|---|
| 0.0 | 3,178.5 | 6.6 % | 2,887.7 | 3,207.4 |
| 0.1 | 3,181.4 | 9.1 % | 2,875.6 | 3,213.2 |
| 0.3 | 3,191.9 | 7.0 % | 2,636.7 | 3,237.3 |
| 0.6 | 3,241.9 | 8.2 % | 2,195.1 | 3,304.7 |
| 1.0 | 3,361.2 | 10.0 % | 678.8 | 52,645.6 |

Reading:
* **Near-optimal, fast.** On the smallest model the section DP is 0.7 % below the proven optimum, in 0.14 s
  against 8 s. Where HiGHS's bound is informative, the DP plan is within 0.5-1.9 % of the optimum
  (360 blocks: 0.7 %; 1,200 blocks: 0.5 %; the coupling series: 0.9-1.9 %). From 1,200 blocks on, the exact
  MILP's best plan after two to four minutes is 7-13 % worse than the DP's, and with strong cycles it
  collapses (679 against 3,361).
* **Linear time.** 0.14 s for 120 blocks to 95 s for 90,000 blocks - about 1 ms per block; the exponential
  part stays inside the 8-option blocks.
* **Exact moves matter.** Local search over single blocks (coordinate ascent, multi-start) ends 22-39 %
  below: with interactions, one-block moves are myopic; moves that re-solve a whole induced forest exactly
  are not.
* **The fast certificate is conservative.** The bound from subgradient steps says 2.5-19 %; where HiGHS
  proves more, the true gap is far smaller (1,200 blocks: certificate 9.3 %, true gap at most 0.5 %). The
  smoothed dual below tightens it.

**A tighter certificate** (`bench/results/section_theory_smooth.json`, `--dual smooth`: smoothed dual
decomposition, L-BFGS-B, temperature lowered in stages; same plans, only the proof differs):

| blocks | certified gap, subgradient | certified gap, smoothed | seconds, subgradient / smoothed |
|---|---|---|---|
| 120 | 2.5 % | 2.5 % | 0.14 / 0.77 |
| 360 | 3.9 % | 1.9 % | 0.23 / 1.76 |
| 1,200 | 9.3 % | 1.7 % | 0.64 / 6.22 |
| 3,600 | 13.2 % | 2.0 % | 2.95 / 20.39 |
| 9,000 | 14.4 % | 2.3 % | 7.10 / 57.67 |
| 30,000 | 18.5 % | 2.2 % | 27.38 / 182.93 |
| 90,000 | 19.3 % | 2.7 % | 95.18 / 563.28 |

With strong cycles (the coupling series) the smoothed certificate is 2.3-3.7 % (subgradient: 6.6-10.0 %).
So every plan of up to 90,000 blocks is *proven* to be within 1.7-3.7 % of the optimum, without an exact
solver; the price is about 6-9x the time (about 6 ms per block, still linear). Where HiGHS can prove more,
the true gap is smaller still (at most 0.5 % at 1,200 blocks).

## Column generation with exact DP sub-problems (polynomial decomposition)

`bench/results/colgen.json` (`python -m bench.colgen`; theory in [DECOMPOSITION.md, section 11](DECOMPOSITION.md#11-column-generation-with-exact-dp-sub-problems-colgenpy-learnpricepy)).
Wells-like units over 24 h: on/off every 6 h with restart costs, a speed level per 2 h out of 5, rate linear in speed, power
~ speed^3, wear ~ speed^4; one shared gas limit per hour at 55 % of what all units could make. Column generation (each unit's
exact DP at the master LP's prices, HiGHS master, Wentges smoothing, polynomial integer step) against the same discrete model
as one MILP (HiGHS, 60 s limit). "Gap to proven bound" is what every plan reports about itself; "below MILP" compares with the
MILP's best plan (its proven optimum where it finished).

| assets | decisions | CG seconds | rounds | gap to proven bound | MILP | CG below MILP |
|---|---|---|---|---|---|---|
| 10 | 160 | 2.24 / 0.94 | 25 / 24 | 3.798 % / 3.880 % | 18 s optimal / 3 s optimal | 3.32 % / 3.29 % |
| 20 | 320 | 1.84 / 1.98 | 26 / 30 | 2.515 % / 2.101 % | 13 s optimal / 25 s optimal | 2.18 % / 1.71 % |
| 40 | 640 | 5.13 / 5.10 | 22 / 17 | 0.891 % / 0.814 % | 60 s time limit / 60 s time limit | 0.80 % / 0.66 % |
| 80 | 1,280 | 5.13 / 5.17 | 16 / 20 | 0.395 % / 0.289 % | 60 s time limit / 60 s time limit | 0.36 % / 0.23 % |
| 160 | 2,560 | 0.25 / 0.22 | 21 / 20 | 0.187 % / 0.202 % | - | - |
| 320 | 5,120 | 0.56 / 0.45 | 22 / 22 | 0.060 % / 0.089 % | - | - |
| 640 | 10,240 | 0.77 / 0.87 | 18 / 18 | 0.056 % / 0.051 % | - | - |
| 1,280 | 20,480 | 2.52 / 2.32 | 17 / 18 | 0.024 % / 0.023 % | - | - |
| 2,560 | 40,960 | 10.90 / 5.66 | 22 / 17 | 0.013 % / 0.009 % | - | - |

Reading:
* **The rounds do not grow with the site** (16-30 from 10 to 2,560 units, two seeds each), and from 160 to 1,280 units the time
  grows about linearly (0.22 s to 2.5 s). Below 4 units per shared row (96 here) the integer step also solves the restricted
  master as a MILP (5 s limit), which is why 40-80 units take longer than 160. The exact MILP proves its optimum at 10 and 20 units
  (3-25 s) and stops at its 60 s limit from 40 units on; its best plan after those 60 s is still 0.2-0.8 % better than column
  generation's after 5 s, but it cannot say how far from the optimum it is, and from a few hundred units it is not an option.
* **The gap falls as the site grows**, as Shapley-Folkman says (Proposition 9): 2-4 % with 10-20 units on 24 shared rows (1.7-3.3 %
  below the proven optimum), under 1 % from 40, 0.01-0.02 % at 1,280-2,560. With as many shared rows as units the integer step is the
  weak part (a test case of 8 units over 12 hours ended several per cent below the optimum while the bound stayed tight): the
  generated schedules are those the relaxation needs, not always those an integer plan needs.
* **This is the model-agnostic core, not yet the twins.** Deriving the asset families and shared rows from a PLX twin (tanks
  and sumps as their own blocks with priced flow balances) is the next step.

**Learned prices** (`colgen.json -> learned_prices`, `--learn`; 60 units, trained on 48 situations with the limit at 35-80 % and
prices +-30 %, tested on 16 others):

| start | rounds | seconds | gap to proven bound |
|---|---|---|---|
| cold | 21.9 | 5.16 | 0.595 % |
| predicted prices | 20.7 | 5.21 | 0.614 % |
| the exact prices (best possible start) | 14.9 | 4.95 | 0.638 % |

A start from the exact prices saves a third of the rounds; the network's prediction saves 1.2 of those 7 at equal plan quality,
and at 60 units the time is the small-site MILP's, not the rounds'. Learning the prices is in place but does not pay yet; it needs
better features or a larger training set. The plans themselves do not vary between runs: column generation and its DPs are
deterministic.

## Structure recognition and exact branches (twins)

`bench/results/exact_branches.json` - which branches of the twins the structure analysis
(`structure.py`, theory in [DECOMPOSITION.md](DECOMPOSITION.md#7-recognising-branches-that-classical-methods-solve-exactly))
proves to be linear, piecewise-linear or convex programmes, and what solving them with LP methods adds to a plan of
the live cascade (scenario planner budget, no neural start; the polish comparison continues the cascade's
own augmented Lagrangian with its multipliers for at least the same wall time). The Kestrel rows are from after the
slope fix, the Corvus rows from before it (the fix does not change the mine's gradients). The previous Kestrel rows: `arms/exact_branches_beforeslope.json`.

| twin | decisions | LP | sequential LP (given the discrete choices, blend qualities frozen) | convex (same, power price at its current sign) | nonlinear |
|---|---|---|---|---|---|
| Corvus mine | 4,992 | 33.7 % (regulators, pumps, trucks) | 64.9 % (stopes, ore passes, hoist, ROM pad) | 1.4 % (main fans) | 0 % |
| Kestrel field | 4,983 | 0.2 % (trucks) | 7.3 % (batteries, pads, compressor stations) | 92.5 % (wells) | 0 % |

| twin | cascade plan (USD, F) | + LP branches | + sequential LP over the recognised layer | + polish, at least the same time |
|---|---|---|---|---|
| Corvus | 7,638,002 (F 0.3739) | +0 in 2.1 s | +14,460 (+0.19 %, F +0.0007) in 11.7 s, 4,032 decisions, 16,885 rows, feasible | -67,184 and F -26.1 (violation 0.105) |
| Kestrel | 4,219,444 (F 0.8486) | - (no LP decisions in the blocks) | +58,042 (+1.4 %, F +0.0117) in 0.6 s, 216 decisions | +26,762 but F -1.6 (violation 0.056) |

| twin | full cascade | half the cascade's iterations + LP branches |
|---|---|---|
| Corvus | 7,638,002 in 69.2 s | 6,365,872 in 34.3 s |
| Kestrel | 4,219,444 in 23.0 s | 4,294,715 in 8.7 s |

| twin | sequential LP including the convex class | tangent gap left |
|---|---|---|
| Kestrel | +136,368 (+3.2 %, F +0.0317) in 24.4 s, 3,672 decisions, 4,464 rows, violation 0.0001 | USD 36,650 |

Reading:
* **Most of the mine is (piecewise) linear.** Once the discrete choices (ring firings) and the blend grades
  are fixed - the situation of the cascade's continuous polish - 98.6 % of the mine's decisions are proven
  linear or piecewise linear; the fans (power ~ speed^3) are convex.
* **The oil field's wells are convex, not nonlinear.** With on/off fixed a well's rate is linear in its speed (the pump
  curve has one segment on the speed range) and its power, wear and failure risk are convex. This class is 92.5 % of the
  field's decisions. The verdict rests on a non-negative power price, and it is a per-asset proof: the plan value as a whole
  also passes through kinks downstream, so the tangent gap is a measured estimate, not a theorem.
* **The LP branches are already solved by the cascade.** On the regulators, pumps and trucks the exact LP
  adds nothing to the cascade's plan; it certifies the plan's optimum on those branches (global optimum,
  exact shadow prices) in 2.1 s.
* **A gain in one plan does not predict the realised result.** Sequential LP over the recognised layer
  improves this oil-field plan by 1.4 % in the twin's objective - and over 72 h on three seeds (before the slope fix) the same stage
  *lowers* the realised result (by USD 47,800 without margins on the tank limits, by USD 7,400 with 10 % margins); it barely changes this mine plan
  (+0.19 %) - and over 48 h on three seeds it *adds* USD 47,300 ([Kestrel](#kestrel-showcase-run-72-h) and
  [Corvus](#corvus-mine-showcase-run-48-h) sections). Plans are executed for three hours and re-made; where
  the twin is not exact, a plan that sits on a limit costs in reality what it gained in the model. Only the
  A/B over seeds decides - so the stage runs on the mine and is off on the oil field. Continuing the polish
  instead of the LP does not help at all: it buys objective with constraint violations. On the mine most of the time
  goes into the Jacobian (7.3 s of 11.7 s, with the structured Jacobian now on by default: column colouring for
  the sparse rows, batched reverse mode for the dense ones); the LPs take 3.5 s.
* **It finishes the cascade, it does not replace it.** Half the cascade's iterations plus exact branches is
  far worse than the full cascade: the relaxation and the discrete search find the region; LP methods then
  solve the part of the problem that is linear in that region exactly.

## Planning under uncertainty

Theory and design in [OPTIMIZER.md](OPTIMIZER.md#planning-under-uncertainty-riskpy-estimatorcov).

**Distributions of a plan's outcomes** (`bench/results/risk.json`, `python -m bench.risk`). A plan of the live cascade, every
calibratable property uncertain by the engineering prior (0.15 in log), prices as log random walks (0.02 per sqrt h); reference:
Monte Carlo with 8,192 samples. "Sigma points" = one batched simulation at +-1 standard deviation per factor (2K + 1 simulations);
the errors of the two plan-value estimators are root-mean-square over five sampling seeds, at equal total simulations.

| twin | inputs | factors | limits: spread ratio median (p10-p90) | P(violation) error | plan value: reference std | sigma points only | + 128 Sobol: error of mean / std | Monte Carlo, same total: error of mean / std |
|---|---|---|---|---|---|---|---|---|
| Kestrel | properties | 198 | 0.986 (0.971-0.999) | 0.008 | 72,272 | 66,842 | 1,796 / 6,533 | 1,897 / 5,775 |
| Kestrel | properties and prices | 216 | 0.988 (0.973-1.002) | 0.007 | 602,003 | 587,953 | 9,168 / 22,159 | 29,183 / 19,893 |
| Corvus | properties | 383 | 0.993 (0.984-1.002) | 0.015 | 2,285,223 | 4,472,094 | 37,100 / 63,381 | 62,274 / 72,733 |
| Corvus | properties and prices | 395 | 0.992 (0.982-1.000) | 0.014 | 2,442,869 | 4,549,110 | 61,353 / 102,734 | 81,519 / 38,150 |

Reading:
* **The limits are right without sampling.** Over every hard limit, asset and hour with a spread (350 rows on the oil field, 1,057
  on the mine), the propagated standard deviation is within a few per cent of Monte Carlo and the probability of a violation within
  0.007-0.015. This is what chance constraints need, and it is deterministic: the same plan gets the same margins every time.
* **The plan value is not.** Where a total over many assets meets a kink, no single factor crosses it. On another oil-field plan,
  one sitting exactly on its gas capacity, the sigma points saw 66 k of a 215 k spread (`arms/risk_plan_on_kink.log`): the
  flaring penalty, 0 in the nominal plan, is large when many wells' gas-oil ratios err together. On the mine a 15 % prior on every stope grade and on the analyser
  correction pushes the head grade far out of its window, and the sigma points overstate the spread twice. A quadrature along the
  principal directions of the near-binding limits did not fix it (they explain 60-72 % of the penalty; the remaining factors still
  move the total). The value's spread and quantiles therefore come from samples; its mean from the samples with the sigma-point
  surrogate as a regression control variate, 1.7-3.2 times more accurate than Monte Carlo at equal cost in three of four cases,
  not in the fourth. For the spread, Monte Carlo spending the same total is as good or better: the honest gain is that the planner
  needs the sigma points for the limits anyway, so the value's distribution costs 128 further simulations.
* The mine's 15 % prior is far wider than its twin's real uncertainty: grades are assayed, the analyser correction is estimated
  online. It is a stress test of the method, not of the mine.

**Is the twin's uncertainty honest?** (`bench/results/uncertainty_coverage.json`, `python -m bench.uncertainty_coverage`; oil
field, 48 h with planning, every hour after a 6 h warm-up: z = log(truth / estimate) / posterior sd for every property of
every asset, 12,678 values.)

| | within 1 sd (honest: 0.68) | within 2 sd (0.95) | z rms (1.0) |
|---|---|---|---|
| posterior covariance (`Estimator.cov`) | 0.73 | 0.87 | 1.49 |
| fixed engineering sigma 0.15 | 0.90 | 0.98 | 0.61 |

| property | posterior: within 1 sd / 2 sd / z rms | mean posterior sd | rms log error |
|---|---|---|---|
| well rate (q_max) | 0.38 / 0.65 / 2.39 | 0.021 | 0.032 |
| gas-oil ratio | 0.80 / 0.96 / 0.81 | 0.242 | 0.148 |
| disposal injectivity | 0.85 / 0.98 / 3.05 | 0.012 | 0.065 |
| battery liquid capacity | 1.00 / 1.00 / 0.32 | 0.156 | 0.050 |
| water cut (well tests) | 1.00 / 1.00 / 0.36 | 0.131 | 0.051 |

The fixed prior is too wide for most properties; the posterior is right for the gas-oil ratios and too wide where nothing
measures a property (battery capacity) - but it is overconfident for one: each well's rate. Its flow is measured every few
minutes, so the fit pins the rate to about 2 %, while the twin's pump curve differs from the plant's and the rate absorbs the
difference (3 % rms): a model error no residual shows (the variance factor stayed 1.0). The injectivity outliers are the
unannounced 55 % step at SWD-2, until the anomaly detector releases it.

Covariance matching (`Estimator.cov_matching`, `uncertainty_coverage_matching.json`, `--matching`): what planning sees is inflated
per property by how much further the estimates move between windows than the filter predicts. It singled out the wells' rates
(factor at its cap of 100, the others below 1) and moved their coverage from 0.38 / 0.65 / 2.39 to 0.85 / 0.92 / 1.17 (all
properties: 0.88 / 0.96 / 0.89) - better tails, but a core far too wide: the rate errors are mostly small with a few large ones,
which one Gaussian per property cannot describe. Part of the cause is an inconsistency it exposed: the estimator lets an
estimate move 5 % per window (its step prior), the filter assumes 1 % drift. It stays off.

**Chance constraints in the A/B** (`--set risk_eps=0.05`, 5 % probability of breaking a limit, the estimator's covariance and
the observed price uncertainty, three seeds, against the record above):

| arm | oil field (72 h) | mine (48 h) |
|---|---|---|
| every hard limit (`risk_scope=all`) | stopped at hour 10-13: +22,123 / +21,479 / +19,621 at Day 1 16:00, record +67,313 / +65,827 / +68,663 (`arms/partial/kestrel_risk_s*.log`) | stopped at hour 18-21: +149,059 / +137,464 at Day 2 00:00, record +152,807 / +163,435 / +156,020 (`arms/partial/corvus_risk_s*.log`) |
| upper state limits only (`risk_scope=upper`) | not run | **+575,634 (+10.2 %)**: -23,362 / +4,317 / -14,529 by seed; 57 of 57 plans feasible (`arms/showcase_run_corvus_risk_upper*.json`) |
| state limits only (`risk_scope=states`, default) | **+399,910 (+14.7 %)** against +392,840: +13,419 / -697 / +8,487 by seed; 95 of 96 plans feasible against 93 of 96 (`arms/showcase_run_risk_states*.json`) | **+550,028 (+9.7 %)** against +586,825: -61,566 / -2,434 / -46,394 by seed; 56 of 57 plans feasible (`arms/showcase_run_corvus_risk_states*.json`) |

On every limit the planner gave about two thirds of the oil field's early lead away: capacities nothing measures keep their 15 %
prior, so the back-offs on the batteries' liquid capacity and the gas takeaway were large - for limits whose violation the plant's
protection turns into a cut flow the next plan reacts to. On state limits only (tanks, sumps, bins: spills, floods, dry runs) the oil
field gains USD 7,070 on average, within the seed noise, with no spill in either world (none of the scenario's events strains a
tank far enough). On the mine it costs USD 36,797: less copper sold (head grade 1.424-1.431 % against 1.430-1.435 %) and more
blasting. The back-offs keep the stopes' broken stocks above their lower bound and the sumps away from theirs. But a stock's
lower bound of zero is kept by the plant itself - nothing is mucked that is not there - so its violation is a cut flow, like a
capacity, and by the rule above deserves no chance constraint; for the sumps the back-offs add to the hand-set reserve (25 % more
inflow). Taken to its end - chance constraints on the upper state limits only (overflow: `risk_scope=upper`,
`arms/showcase_run_corvus_risk_upper*.json`) - the rule cuts the mine's cost to USD 11,191 (+575,634, +10.2 %: -23,362 / +4,317 /
-14,529 by seed, 57 of 57 plans feasible), close to the seed noise but still no gain. The stage stays off on both twins; `risk_eps` and `risk_scope` are planner settings per scenario.

**Expected value against spread** (`risk.json -> frontier`, `--frontier`; one cold oil-field plan, polished under a cap on the plan
value's standard deviation; same inputs as above):

| cap on the value's std | expected value | std | 5 % quantile | max violation |
|---|---|---|---|---|
| none | 4,292,563 | 586,367 | 3,266,455 | 0.0009 |
| 0.9 x uncapped | 4,178,414 | 521,146 | 3,318,486 | 0.0000 |
| 0.8 x uncapped | 3,723,288 | 448,752 | 2,987,454 | 0.0000 |
| 0.7 x uncapped | 3,495,190 | 398,909 | 2,805,867 | 0.0000 |
| 0.6 x uncapped | 3,031,120 | 329,210 | 2,465,775 | 0.0214 (no feasible plan) |

Most of the spread is price: a plan reduces it by producing less. The first tenth costs 2.7 % of expected value and even raises the
5 % quantile; beyond that, expected value falls steeply and the bad case gets worse as well. A variance cap is an expensive hedge
for price risk (a financial hedge would not cost production); the cap is a founder's choice and stays off until one is set.

## Backtest, pilot gate G1

`bench/results/backtest_*.json`, `python -m bench.backtest`. The plant runs under manual operations and writes a raw historian export. A fresh twin sees only that file: it is calibrated on the first half (oil field 36 h, mine 24 h), then frozen, and predicts the second half: every signal one step ahead, every level 3 h ahead from the state 3 h earlier. The limits are the pilot plan's for the oil field; for the mine the same limits are applied to the corresponding signals, because no pilot plan exists for a mine.

| Case study | signal | error (% , levels: of capacity) | limit | result | worst tags |
|---|---|---|---|---|---|
| Kestrel | Well liquid rates | 4.88 | 8 | pass | KST.P07.W11.FLOW.LIQ 38.7 %, KST.P02.W06.FLOW.LIQ 10.5 % |
| Kestrel | Compression suction gas | 3.90 | 5 | pass | KST.CS2.SUC.FLOW 4.2 %, KST.CS1.SUC.FLOW 3.6 % |
| Kestrel | Battery separator gas | 3.72 | 5 | pass | KST.CTBC.SEP.GAS 5.7 %, KST.CTBS.SEP.GAS 3.8 % |
| Kestrel | Disposal injection pressure | 16.89 | 10 | **fail** | KST.SWD2.INJ.PRESS 47.9 %, KST.SWD3.INJ.PRESS 1.4 % |
| Kestrel | Oil tank level, 3 h ahead | 1.27 | 10 | pass |  |
| Kestrel | Water tank level, 3 h ahead | 1.65 | 10 | pass |  |
| Corvus | Stope mucking rates | 1.38 | 8 | pass | COR.S920.ST02.LHD.TPH 1.6 %, COR.C680.ST06.LHD.TPH 1.6 % |
| Corvus | Crusher throughput | 1.31 | 5 | pass | COR.CRN.CR.TPH 1.3 %, COR.CRS.CR.TPH 1.3 % |
| Corvus | Hoist throughput | 1.26 | 5 | pass | COR.SH1.HOIST.TPH 1.3 % |
| Corvus | Head grade (on-stream analyser) | 1.62 | 5 | pass | COR.MILL.OSA.HEAD 1.6 % |
| Corvus | Sump level, 3 h ahead | 1.87 | 10 | pass |  |
| Corvus | Shaft bin level, 3 h ahead | 1.63 | 10 | pass |  |
| Corvus | ROM pad level, 3 h ahead | 0.23 | 10 | pass |  |
| Corvus | Ore pass level, 3 h ahead | 1.04 | 10 | pass |  |

**The oil field does not pass as it stands.** The disposal-pressure failure is SWD-2 alone (47.9 %; the other wells 1.4 %): an injectivity loss nobody announces, in the frozen half. The live twin re-estimates it within 3 h; a frozen twin cannot. A pilot has to decide whether gate G1 is meant frozen or with live estimation. The mine passes. Two limits on what this shows: the window is days, not the months a pilot would use, and the data are synthetic; one step ahead with the set points known, most of what remains is sensor noise (about 2 %), so the 3 h level predictions are the harder test.

## What these numbers do not show

* **Real-world uplift.** The A/B gain depends on the baseline and on how constrained the site is. The
  baselines are rule-based models of competent operations (documented in [SHOWCASE.md](SHOWCASE.md));
  real operators have more context and also more delays. Pilot targets are deliberately lower
  ([PILOT_PLAN_OIL_GAS.md](PILOT_PLAN_OIL_GAS.md)). The mine result is a grade-control result at equal
  tonnes and equal cash cost (see above); it moved from +17 % to +34 % to +12 % as planner errors and the
  baseline were corrected - expect the mine number to keep moving as the models are refined. The steel group
  shows what happens against a good S&OP in a market-limited business: an operating-profit lead of +1.1 % at most
  (six seeds, within their spread; -1.6 % before the 5 Nov planner fixes), a clear lead only after the carbon price.
* **Model-form error.** The emulated plants are written in PLX, so the twin's structure matches reality;
  only parameters, drifts, noise and events differ. Real sites add structural mismatch, which a pilot's
  backtest gate measures.
* **Statistical spread.** The Kestrel showcase is reported over 3 seeds (13.8-14.7 %), the mine over 3
  (9.8-10.8 %), the steel group over 3 (-4.9 to +0.8 %); a re-run of the same seed reproduces the manual world
  exactly and PLYXIS within about 1 % on the oil field (multi-threaded floating point on a non-convex problem) -
  on the steel group a replay of one seed was USD 0.85 M apart after 42 h, about 3 % of the period's profit. The solver and section benchmarks use 3-8
  situations or seeds. Single-plan benchmarks are a poor predictor of the realised result (see the exact
  branches) - planner changes are decided on the A/B over seeds.
* **Reproducibility of the scenario.** Until 28 October the market prices of both case studies were drawn
  from one shared random stream in the order the code first asked for them, so a code change that altered
  that order changed the realised prices (within a run both worlds always saw the same prices). Each price
  now has its own stream; all results in this document were re-run after that fix.
