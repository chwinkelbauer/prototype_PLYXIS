# Benchmarks

Every number here is produced by the scripts in `bench/` and stored in `bench/results/`
(`python -m bench.report` prints these tables). Hardware: one laptop, AMD Ryzen 5 PRO 4650U
(6 cores, 15 W), 16 GB RAM, CPU only, PyTorch 2.14, g++ 14 (MinGW-w64) for the native kernels.
All data is synthetic.

```bash
python -m bench.showcase_run --hours 72                      # Kestrel A/B run of the full showcase
python -m bench.showcase_run --scenario corvus               # Corvus mine A/B run (48 h)
python -m bench.twin_accuracy --hours 30 --full              # twin vs. hidden truth over time
python -m bench.solver_comparison --budget 60                # solvers at equal wall time
python -m bench.solver_budget --cases 5                      # what the live solver budget buys
python -m bench.neural_ablation --experts 60 --test 8        # what the neural sub-solvers add
python -m bench.decomposition --seeds 3                      # deterministic sections, native kernels
python -m bench.scaling --fields 1 2 4 8 --ga                # enterprise twins up to 768 wells
```

## Kestrel showcase run (72 h)

`bench/results/showcase_run.json` - the full Kestrel scenario: two compressor trips, a 10-hour gas
plant upset that cuts takeaway by a third, a fast-wearing ESP, an unannounced 55 % loss of disposal
injectivity, 24 h of negative gas prices, two hours of power at ERCOT's 5,000 USD/MWh cap, and a new well
that appears only as SCADA tags. Two worlds start from the same state and see the same hidden physics,
events, notices and sensor noise: one operated by PLYXIS (re-planning every 3 h and after every event),
one by a rule-based model of competent manual operations ([SHOWCASE.md](SHOWCASE.md)).

| KPI (72 h) | PLYXIS | manual operations | difference |
|---|---|---|---|
| **Operating profit** (USD) | 3,190,744 | 2,806,032 | **+384,713 (+13.7 %)** |
| **Profit after carbon cost** (USD) | 3,163,710 | 2,767,096 | **+396,614 (+14.3 %)** |
| Oil produced (bbl) | 53,613 | 48,756 | +4,857 (+10.0 %) |
| Gas produced (Mscf) | 102,542 | 102,688 | -146 (-0.1 %) |
| **Cash operating cost** (USD) | 398,957 | 440,043 | **-41,086 (-9.3 %)** |
| **Gas flared** (Mscf) | 463 | 2,630 | **-2,168 (-82 %)** |
| Flared above the permit (Mscf) | 58 | 1,880 | -1,822 (-97 %) |
| Emissions (t CO2e) | 451 | 649 | -198 (-31 %) |
| Unplanned pump failures | 0 | 1 | |

How the comparison is valued - identically for both worlds: sales minus cash costs, plus the change in
inventories (oil in tanks at 90 % of the price, water in tanks as a disposal liability - the planner's
own terminal valuation), plus the change in remaining pump life (a pump's value runs from new to the
failure threshold; a replacement is charged at the pump cost, an unplanned failure additionally at the
model's failure premium). Planning-only terms (expected failure cost, flaring-risk margin, nominal wear)
are not counted. The carbon cost uses the field's internal price (60 USD/t), which is also what the
planner optimises.

Where the difference comes from (realised terms, `bench/results/showcase_run.json` -> `bridge`, adds up to
the operating-profit difference):

| Term | USD |
|---|---|
| Oil sales | +706,954 |
| Oil and water in tanks | -318,635 |
| Avoided unplanned pump failure | +45,000 |
| Pump life consumed | -40,937 |
| Power | -9,293 |
| Well restarts | +8,604 |
| Water injection power | +3,947 |
| Disposal fees, gas sales, treating, trucking, lease | -10,929 |
| **total** | **+384,711** |

PLYXIS produces 10 % more oil (choosing which wells to run and how fast when gas takeaway, compression
and disposal bind) and also sells down tank inventory that manual operations keep at about half full.
The tank draw-down is netted out by the inventory line; what remains of it in the result is the 10 % gap
between selling a barrel now and valuing it in a tank at the end (about USD 35,000 of the total).

By period (difference in operating profit; flaring in Mscf):

| Period | delta profit (USD) | PLYXIS flare | manual flare |
|---|---|---|---|
| normal operation (first 7 h) | +59,883 | 0 | 0 |
| CS-1 trip | +9,934 | 174 | 1,055 |
| gas plant upset (takeaway -1/3, 10 h), ESP wear | +68,836 | 8 | 198 |
| SWD-2 injectivity loss, new well | +49,256 | 0 | 0 |
| negative gas price | +108,731 | 0 | 0 |
| CS-2 trip during negative price | +57,507 | 227 | 1,160 |
| evening incl. the ERCOT price cap | +24,451 | 24 | 209 |
| recovery to end | +6,115 | 9 | 2 |

Twin and planner during the run: well deliverability error 11.9 % → 2.4 %, GOR 12.9 % → 4.0 %,
water cut 4.7 % → 2.1 %; 32 plans of 4,983 decisions each, all feasible, mean solve time 10.5 s
(max 15.8 s), each preceded by a first response within 1-2 s after urgent triggers.

**How this number was arrived at.** An earlier version of this benchmark reported +16 % operating
profit. Investigation showed that most of it was not PLYXIS: the emulator's protection layer
over-throttled overloaded flows, which hurt the rule-based world more, and the baseline's water handling
was weaker than a real control room. After those fixes, and with pump condition and carbon in the
accounting, the result was +2.0 %. It has since grown to +13.7 % for reasons that are documented and
testable: (1) the solver - more diverse starts, a longer polish and a feasibility restoration (+2.6 % per
plan, `solver_budget`); (2) planner corrections - announced price transients are no longer extrapolated
into the forecast, pump wear is priced the way the accounting values it; (3) a more realistic scenario -
firm gas takeaway below field deliverability (the Permian situation before new pipelines), a midstream
upset with a curtailment notice, disposal fees, and a scarcity price window. Every new event came with the
standard response a competent control room would have (immediate curtailment on the midstream notice,
the 75 % demand-response set point for the price cap), so the baseline was not weakened. A bug found on
the way - price events had silently lasted 24 h instead of their stated duration - was fixed before these
numbers were produced. The period table above shows what each event phase contributes; a run without the
two newest events (`showcase_run --skip-events GASPLANT,power_price --tag core`) has not been made yet.

## Twin accuracy over time (Kestrel)

`bench/results/twin_accuracy.json` - every hour of plant time the live twin is compared with the
emulator's hidden truth (production planner running, all scenario events active). The twin sees
only SCADA telemetry and one well test per hour.

| plant time (h) | well deliverability q_max | GOR | water cut | pad gas, one step ahead (mean / max) | compressor stations |
|---|---|---|---|---|---|
| 6 (start, engineering model) | 10.9 % | 12.7 % | 4.7 % | 7.7 % / 32 % | +3.6 %, +2.9 % |
| 10 | 3.0 % | 12.1 % | 4.5 % | 5.8 % / 19 % | +0.4 %, +1.3 % |
| 18 | 2.4 % | 11.2 % | 4.2 % | 4.4 % / 15 % | -0.7 %, +0.4 % |
| 26 | 2.3 % | 10.7 % | 3.8 % | 5.5 % / 15 % | +0.4 %, +2.2 % |
| 36 | 2.2 % | 9.8 % | 3.5 % | 3.8 % / 12 % | +0.9 %, +0.8 % |

Well deliverability is pinned down within four hours; GOR and water cut improve as well tests rotate
through the 96 wells (after 72 h: GOR 4.0 %, water cut 2.1 %, see the showcase run). With the plant's
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
| PSO | 3,144,324 | 0.5935 | no | 60.0 |
| gravitational search | 1,935,002 | 0.0617 | no | 60.0 |
| GA | 4,183,679 | 0.0001 | yes | 60.0 |
| DE | 3,810,866 | 0.0000 | yes | 60.0 |
| PSO, cooperative (per subsystem) | 1,711,091 | 1.6285 | no | 60.0 |
| GA, cooperative (per subsystem) | 1,112,336 | 0.0000 | yes | 60.0 |
| augmented Lagrangian, single start | 3,259,812 | 0.0002 | yes | 7.0 |
| **PLYXIS cascade** | **4,383,413** | 0.0008 | yes | **10.3** |
| **PLYXIS cascade + neural sub-solvers** | **4,431,554** | 0.0012 | yes | **10.5** |

The vectorised GA is the strongest classic method, but it needs the whole minute and ends 5.6 % below
the cascade with neural sub-solvers, which finishes in 10.5 s. PSO and gravitational search - the
original concept's front end - do not reach a feasible plan at this dimension.

## Solver budget (Kestrel)

`bench/results/solver_budget.json` - 5 sampled event situations re-planned from the nominal plan, as in
live operation:

| Configuration | relax / starts / discrete / repair / polish | mean objective (USD) | vs previous live | feasible | mean seconds |
|---|---|---|---|---|---|
| previous live | 160 / 3 / 12x1 / 80 / 50 | 4,254,832 | - | 5/5 | 7.1 |
| **live** | 200 / 8 / 12x1 / 80 / 120 | 4,366,757 | **+2.6 %** | 5/5 | 10.2 |
| large | 480 / 8 / 24x3 / 150 / 150 | 4,376,936 | +2.9 % | 5/5 | 34.3 |

## Neural sub-solvers (ablation, Kestrel)

`bench/results/neural_ablation.json` - the cascade solves 60 realistic event situations warm-started from
the current plan; the graph network is trained on them (132 s). Then 8 held-out situations are re-planned
with equal budgets:

| Start / method | mean objective (USD) | feasible | mean seconds |
|---|---|---|---|
| previous plan, as is | -1,268,311 | 0/8 | - |
| neural sub-solver plan, as is | 266,094 | 0/8 | 0.02 |
| previous plan + gradient repair | 3,240,462 | 5/8 | 1.36 |
| neural plan + gradient repair | 3,234,521 | 8/8 | 1.36 |
| full cascade | 3,454,010 | 8/8 | 5.64 |
| full cascade + neural start | 3,567,896 | 8/8 | 5.15 |

* The network's raw plan (20 ms for the whole field) is a start, never the final word.
* With the same short repair, the neural start reaches a feasible plan in 8/8 cases (previous plan: 5/8).
* Inside the full cascade the neural start wins 5 of 8 cases, ties 2, loses 1 (median +3.0 %, mean +3.8 %).

## Scaling: from one field to a company

`bench/results/scaling.json` - generated enterprise twins: N copies of the Kestrel field sharing one gas
plant, one oil terminal and a company-wide flare permit. The GA gets exactly the cascade's wall time.

| Fields | wells | decisions (discrete) | compile | simulate 1 / 16 plans | gradient | generated code | cascade solve | cascade objective | GA, same time |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 96 | 4,983 (1,272) | 0.15 s | 10 / 13 ms | 15 ms | 432 lines | 12 s | USD 4.40 M, feasible | infeasible (max violation 0.56) |
| 2 | 192 | 9,966 (2,544) | 0.17 s | 9 / 20 ms | 18 ms | 432 lines | 13 s | USD 8.31 M, feasible | infeasible (max violation 1.24) |
| 4 | 384 | 19,932 (5,088) | 0.24 s | 10 / 34 ms | 19 ms | 432 lines | 19 s | USD 16.45 M, feasible | infeasible (max violation 2.28) |
| 8 | 768 | 39,864 (10,176) | 0.32 s | 16 / 87 ms | 31 ms | 432 lines | 38 s | USD 30.28 M, feasible | infeasible (max violation 2.44) |

* Adding assets adds tensor rows, not code: 768 wells compile to the same 432 lines as 96.
* Solve time grows sub-linearly: 8x the decisions take 3.2x the time (38 s vs. 12 s).
* With the same wall time the GA ends infeasible at every size; its higher 'objective' at 8 fields comes
  from violating shared capacity limits.

## Corvus mine showcase run (48 h)

`bench/results/showcase_run_corvus.json` - the underground copper mine (186 stopes on 23 levels, ~5,000
planning decisions, see [SHOWCASE.md](SHOWCASE.md#case-study-2-corvus-underground-mine)): main fan trip,
seismic closure of a level, a 5-hour hoist stop, three loaders down, a critical-peak price day, a pump
failure and a new stope. Same set-up as Kestrel: two copies of the mine, same truth, events and noise. The
baseline is competent conventional mine control: loader allocation within each level's air budget with
grade control against the block model, a tonnes target at the mill rate, a broken-stock buffer rule for
blasting, design ventilation (full fan speed after a fan trip), level-controlled pumps, the mill at 99 % of
nameplate, deferral of hoisting and pumping during announced price peaks.

| KPI (48 h) | PLYXIS | conventional control | difference |
|---|---|---|---|
| Operating profit (USD) | 5,894,144 | 5,247,053 | +647,091 (+12.3 %) |
| Ore milled (t) | 76,292 | 75,840 | +452 (+0.6 %) |
| Head grade (from copper, milled tonnes and the recovery curve) | 1.449 % | 1.368 % | mine-plan window 1.35-1.45 % |
| Copper in concentrate (t) | 1,014 | 947 | +67 (+7.1 %) |
| Ore mined (t) | 64,997 | 71,685 | -6,689 (-9.3 %) |
| Ore hoisted (t) | 71,188 | 66,874 | +4,314 (+6.5 %) |
| Cash operating cost (USD) | 3,158,574 | 3,184,662 | -26,088 (-0.8 %) |
| Production cost per t of copper (cash cost + ore inventories drawn down, at cost) | USD 3,414 | USD 3,685 | -7.4 % |
| Annualised cash operating cost | USD 576 M | USD 581 M | |
| Site power (MWh) | 2,928 | 2,917 | +0.4 % |
| Sump overflow | none | none | |

| Where the operating-profit difference comes from | USD |
|---|---|
| Copper sales | +619,581 |
| Mining consumables | +65,223 |
| Explosives and charging | -50,000 |
| Power (fans, pumps, hoist) | +13,955 |
| Loader operating cost | +6,001 |
| other (mill power and consumables, inventories at cost, crushers, trucks) | -7,669 |
| **total** | **+647,091** |

Plans: 19, of which 17 feasible (the other two within 0.25 % of a limit); mean solve time 48.9 s, maximum
62.6 s. Stope grade error of the twin: 11.8 % (engineering model) → 9.9 % after 48 h.

**How to read this result.** Both mills run at capacity; the gain is grade within the mine plan. The twin
learns the stope grades from grab-sample assays and the mill's on-stream analyser and blends to the top of
the plan's head-grade window (1.449 % against a limit of 1.45 %); conventional grade control aims at 1.40 %
with block-model grades that overstate the diluted upper South zone and delivers 1.368 %. Same tonnes, same
cash cost, 7.1 % more copper: the production cost per tonne of copper - the unit cost miners report - is
7.4 % lower. PLYXIS mined 9 % less ore and drew the ore passes down instead (inventories at cost are equal
in total: -303 k vs -305 k USD), so part of its lower mining cost is buffer it will have to rebuild. It is
not yet a cost-reduction story at the site level: cash cost is the same.

**How this number was arrived at.** The previous version of this section reported +17.4 %. Re-examining it
hour by hour found planner errors and a baseline that was not competent, all fixed before this run (and
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
  lead fell from +34 % to the +12.3 % above. Its annualised cash cost rose from USD 491 M to USD 581 M with
  the extra tonnes.
* *Head grade.* The earlier "1.52 % vs 1.33 %" was not derived from the results; `bench.report` now inverts
  the concentrator's recovery curve on copper and milled tonnes (for that run: 1.455 % vs 1.287 %).

## Deterministic sections (learned graph filter + exact section search)

`bench/results/decomposition.json` - the cascade with the GA discrete stage alone vs. with the exact
section search on the GA's best plan (`ga+sections`), 3 seeds each, same budgets:

| Case study | decisions (discrete) | method | mean objective (USD) | vs GA | feasible | mean seconds |
|---|---|---|---|---|---|---|
| Kestrel | 4,983 (1,272) | GA | 4,207,025 | - | 3/3 | 6.3 |
| Kestrel | 4,983 (1,272) | GA + sections | 4,205,566 | -0.0 % | 3/3 | 9.2 |
| Corvus | 4,992 (744) | GA | 7,275,111 | - | 3/3 | 38.8 |
| Corvus | 4,992 (744) | GA + sections | 7,247,980 | -0.4 % | 3/3 | 48.1 |

| Case study | sections (largest) | colours | filter: rank correlation with exact couplings | filter forward | probing all couplings exactly | one sweep native / NumPy |
|---|---|---|---|---|---|---|
| Kestrel | 76 (3 assets) | 8 | 0.73 | 1.4 ms | 0.4 s (442 pairs) | 1.09 s / 1.15 s |
| Corvus | 46 (6 assets) | 10 | 0.03 | 2.1 ms | 1.1 s (665 pairs) | 2.34 s / 2.38 s |

Scaling - enterprise twins of N Kestrel fields:

| Fields | decisions | discrete units | sections (largest) | colours | one exact section sweep | one simulation |
|---|---|---|---|---|---|---|
| 1 | 4,983 | 624 | 76 (3) | 8 | 0.9 s | 9 ms |
| 2 | 9,966 | 1,248 | 152 (3) | 8 | 1.5 s | 12 ms |
| 4 | 19,932 | 2,496 | 304 (3) | 8 | 2.9 s | 16 ms |
| 8 | 39,864 | 4,992 | 688 (3) | 9 | 3.2 s | 28 ms |

Reading:
* **The complexity claim holds.** Sections stay small (3 assets, ≤ 64 joint options per window) while
  their number grows linearly with the site (76 → 688), the colouring stays at 8-9 colours, and one exact
  sweep over all sections grows sub-linearly in wall time (0.9 s → 3.2 s for 8x the decisions): the work is
  exponential only inside a section.
* **No measurable plan gain yet.** On this run the section search changes the plan value by -0.0 % (oil
  field) and -0.4 % (mine) against the GA alone - both within the run-to-run spread of these non-convex
  solves. The previous run of the same benchmark showed +1.0 % on the unchanged oil-field model, so that
  number was noise too. (The search only accepts a plan that improves the GA's best; the polish that follows
  is not deterministic across runs.) It costs 3-9 s per plan.
* **The graph filter does learn real structure where there is some** (Kestrel: 0.73 rank correlation
  with the exact cross terms) and replaces the exact probing at ~1-2 ms per call - on these sites the
  probing itself is still cheap (0.4-1.1 s); the filter matters for larger sites and repeated re-planning.
  On the mine the couplings between discrete decisions are weak (rank correlation 0.03).
* **Native kernels.** The C++/OpenMP kernels give identical results to the NumPy reference; inside a
  sweep they make no measurable difference (1.09 s vs. 1.15 s on Kestrel) because the batched twin
  simulation dominates the sweep. They are kept for the combinatorial core as sites grow.

## What these numbers do not show

* **Real-world uplift.** The A/B gain depends on the baseline and on how constrained the site is. The
  baselines are rule-based models of competent operations (documented in [SHOWCASE.md](SHOWCASE.md));
  real operators have more context and also more delays. Pilot targets are deliberately lower
  ([PILOT_PLAN_OIL_GAS.md](PILOT_PLAN_OIL_GAS.md)). The mine result is a grade-control result at equal
  tonnes and equal cash cost (see above); it moved from +17 % to +34 % to +12 % as planner errors and the
  baseline were corrected - expect the mine number to keep moving as the models are refined.
* **Model-form error.** The emulated plants are written in PLX, so the twin's structure matches reality;
  only parameters, drifts, noise and events differ. Real sites add structural mismatch, which a pilot's
  backtest gate measures.
* **Statistical spread.** The showcase runs are single runs; the solver and section benchmarks use 3-8
  situations or seeds, and the mine solver results vary noticeably between runs (multi-threaded floating
  point on a non-convex problem). The direction of the Kestrel results has been stable; exact sizes are not.
