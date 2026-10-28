# Showcase: demo scripts and assumptions

Two case studies run on the same platform; pick one with the title in the top bar (the console stops the
running twin, compiles the other one and starts it - about 20 s for the mine):

* **Kestrel Field** - 96-well Permian oil field, 72 h scenario (this page, first part).
* **Corvus Mine** - large underground copper mine in 3D, 48 h scenario (second part, below).

# Case study 1: Kestrel field

## Setup

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start.ps1          # http://127.0.0.1:8000
```

Open the console full-screen (1920×1080 works best). The guided demo panel sits bottom right; it
can be minimised. Each step has a **Go to** button that fast-forwards the plant clock to the
moment of the event (at 1 h of plant time per second), switches to the right view and pauses.
Between steps, run at 10 min/s. After a disruption the plant clock is held only until PLYXIS's
*first response* is live (neural sub-solvers plus a short gradient repair, typically 1-2 s); the
full optimisation then continues in the background and replaces it when it is done (about 10 s).
Compressed demo time would otherwise penalise the planner for its solve time. Displayed rates are
smoothed over about 1.5 h of plant time so the numbers stay readable.

**What the audience sees:** the same synthetic field run twice from the same state, with the same
hidden physics, the same events and the same sensor noise: *PLYXIS* (brand pink) and *manual
operations* (neutral slate, a rule-based model of competent current practice, see below).

## The scenario (plant time; Day 1 06:00 = start)

| Time | Event | What it tests |
|---|---|---|
| Day 1 13:00 | one of three compressor units at CS-1 trips | disruption response: re-plan, re-route gas, avoid flaring |
| Day 1 19:00 | gas plant upset: midstream cuts Kestrel's takeaway by a third (36,000 → 24,000 Mscf/d) for 10 h; both control rooms get the notice | constrained takeaway: which gas to keep, which wells to slow |
| Day 1 22:00 | an ESP on PAD-11 starts wearing fast (only vibration shows it) | hidden-state estimation, predictive maintenance |
| Day 2 04:00 | SWD-2 silently loses 55 % of its injectivity | the twin must notice reality drifting and recalibrate |
| Day 2 08:00 | pipeline notice: Waha gas will be negative (-2.50 USD/Mscf) for 24 h from 12:00 | market-aware planning |
| Day 2 15:00 | ERCOT day-ahead notice: power at the 5,000 USD/MWh cap tomorrow 17:00-19:00 | demand response with a day of warning |
| Day 2 18:00 | a new well comes online; it exists only as SCADA tags | discovery: the twin rewrites its own code |
| Day 3 06:00 | a compressor unit at CS-2 trips during the negative-price window | combined disruption |
| Day 3 17:00 | power at the 5,000 USD/MWh cap for 2 h | load shedding vs. lost oil, tank buffering |
| Day 4 06:00 | end of scenario (auto-pause) | value |

The gas plant upset and the price cap are realistic for the Permian: gas takeaway is the basin's
binding constraint (Waha has traded negative repeatedly since 2019 because pipelines are full), and
5,000 USD/MWh is ERCOT's system-wide offer cap, reached during heat waves.

## Demo script (about 14 minutes)

1. **One operating system for the whole field** (Overview). Production, profit, flaring and
   emissions of both worlds; the live feed on the right. *"Everything you see is computed from a
   model of the field that PLYXIS keeps equal to reality - no dashboards were drawn by hand."*
2. **The twin is the map** (Map). Click a well, a battery, a compressor. Every asset is an instance
   of a PLX blueprint, instantiated from the asset register (`wells.csv`, `pads.csv`) by about 120
   lines of field code. Open the drawer: live telemetry, twin parameters, the plan, *Why*.
3. **Disruption: compressor trip** (Optimizer). Watch the cascade stream its stages: compile,
   neural sub-solvers (milliseconds), coordinated relaxation, discrete search, polish - ~5,000
   decisions in about 10 seconds. The new plan re-routes gas to CS-2, starts spare units and
   chokes back the gassiest wells; manual operations keep flaring until the call-out crew arrives.
4. **Why did it do that?** (Schedule). Hover any bar: the reason is the marginal value of the
   decision split into objective terms and the shadow prices of the binding bottlenecks -
   computed from the model's gradients, not generated text.
5. **Midstream cuts takeaway by a third** (Overview). The notice reaches both control rooms. PLYXIS
   re-selects which wells produce: oil-rich wells keep running, gas-rich ones slow down, so the
   field loses a third of its gas but far less of its oil. Manual operations curtail their four
   gassiest pads by 20 % (the standard response); the rest of the cut happens through the plant's
   back-pressure on every well, so they lose about as much oil as gas.
6. **Predictive maintenance** (Data, then Schedule). The vibration trend on PAD-11 W-01 is
   extrapolated to the alarm level ("alarm expected in ~7 h"). Pump health is not measured by any
   sensor; the twin estimates it from vibration, and the maintenance planner books a workover rig
   for a planned replacement before the pump fails (visible as a rig job in the schedule). Pumps
   that are worn but stable keep running - replacing them early would waste remaining pump life.
   In the other world the pump fails and waits for the rig.
7. **Reality drifts - the twin follows** (Data). SWD-2 lost more than half of its injectivity and
   nobody told the system. The injection pressure no longer matches the twin; the estimator
   releases the disposal well's priors and re-estimates its capacity within about three hours
   (19,970 → 9,531 bbl/d, true 9,450), and the planner re-routes water and eases the wells with the
   most water per barrel of oil.
8. **Markets move - operations follow** (Overview). Waha gas is negative for 24 hours. PLYXIS
   re-prices every barrel: oil still pays for the gas it brings, so production continues, and it
   does not flare to dodge the price (flaring costs carbon and breaks the zero-routine-flaring
   policy). The point to make: the planner reasons about the whole economics, not about one price.
9. **The twin rewrites itself** (Twin Studio). Unknown tags from a new well matched the field's
   `discover:` rule, which generated PLX code (with coordinates from the device's birth
   certificate). The code was compiled, committed (see the commit history and diff) and
   hot-swapped into the live twin; the planner immediately includes the well.
10. **Power at the price cap** (Overview, Day 3 17:00). The day-ahead notice came yesterday, so the
   plan already contains the response. Manual operations drop every pump to the pre-agreed 75 %
   demand-response set point; PLYXIS decides per well (power grows with the cube of pump speed,
   oil roughly linearly), buffers produced water in the tanks instead of injecting it at the cap
   price, and brings everything back afterwards.
11. **Value** (Overview, after Day 4 06:00). Same field, same events - the difference is the value of
   autonomous operations (numbers below).

### Live interactions (ask the audience to pick)

* **Break something:** select a compressor station, a well or a disposal well on the map; the
  drawer's footer offers *Trip one unit*, *Fail the ESP* or *Lose 50 % injectivity*. The event hits
  both worlds; PLYXIS re-plans immediately.
* **Change the model:** in Twin Studio, change a property in `field.plx` (e.g. a compressor's
  `unit_cap` or the `carbon_price`), press *Check* (diagnostics with line numbers) and *Apply*: the
  twin recompiles, is committed and the planner re-plans with the new reality.
* **Ask why:** open any asset's *Why* tab.
* **Look under the hood:** Optimizer view (stage timings, convergence), Benchmarks view (solver
  comparison, neural ablation, scaling).

## Results of the reference run

`python -m bench.showcase_run --hours 72` (headless, deterministic plans) writes
`bench/results/showcase_run.json`. Results of the reference run are in
[BENCHMARKS.md](BENCHMARKS.md#kestrel-showcase-run-72-h).

## Assumptions behind "manual operations"

The comparison is only meaningful if the baseline is competent. `scenarios/kestrel/legacy.py`
encodes common practice of a mid-size onshore operator:

| Area | Rule |
|---|---|
| Wells | all available wells run at the field-standard set point (95 % pump speed); no load shifting against power prices |
| Gathering | pads stay lined up to their home tank battery |
| Compression | every available unit runs |
| Tanks | level controllers pump oil and water out around 50 % level |
| Oil sales | CTB Central to the pipeline first, trucks lift the excess; the dispatcher re-allocates the truck fleet every 2 h in proportion to oil waiting |
| Water | home disposal well; the control room keeps it below its injection-pressure alarm (2,000 psi) by shifting part of the battery's water to the second outlet, and curtails the two wettest pads of a battery by 20 % while its water tank runs above 85 % |
| Flaring | sustained flaring (> 2 h) leads the control room to curtail the four gassiest pads by 20 % until flaring stops, lifted 2 h later |
| Gas takeaway | gas sales at the plant limit for > 2 h: the same curtailment of the four gassiest pads, kept until sales are clearly below the limit; a midstream curtailment notice triggers it immediately |
| Compressor trips | the call-out crew restarts the unit after 6 h |
| Pump failures | workover rig scheduled first thing next morning (07:00), first come first served, 30 h job |
| Economics | no reaction to hourly power or gas prices (monthly economics) |
| Scarcity pricing | on a day-ahead notice above 1,000 USD/MWh, every ESP drops to a 75 % demand-response set point for the announced window (the bulk set-point change operators use for ERCOT 4CP peaks) |

The baseline was hardened during development. An earlier version diverted a fixed 60 % of the
water on a pressure alarm; combined with an over-cautious protection layer in the emulator, its
tanks filled after the SWD-2 event and its production collapsed, which inflated PLYXIS's advantage.
Both were fixed (pressure-controlled split, curtailment of wet pads, protection that limits flow
exactly to the binding capacity), and all results reported here use the stronger baseline. When
the gas plant upset and the price cap were added to the scenario, the baseline received the
corresponding standard responses at the same time (immediate curtailment on the notice, the
demand-response set point), so the new events test decision quality, not missing rules.

What manual operations are **not** penalised for: they see perfect SCADA data, react without
fatigue or shift handovers, and are never slower than the rules above. What PLYXIS is **not**
given: it never sees the hidden truth (engineering estimates are off by realistic margins and
deliverability drifts), it pays for every restart, and its plans are executed by the same plant
with the same protection layer.

## Caveats (and how to answer the hard questions)

* **Synthetic field.** Physics, events and the baseline are synthetic. The plant emulator is
  written in PLX, so model-form error is limited to parameters, noise and events; a real field
  adds structural mismatch, which is why the pilot plan starts with a backtest gate
  ([PILOT_PLAN_OIL_GAS.md](PILOT_PLAN_OIL_GAS.md)).
* **The size of the uplift.** The relative gain depends on how good the baseline is and how
  constrained the field is. Pilot targets are set far lower (1-3 % production at equal
  constraints, larger effects on flaring and failures) and are verified by M&V.
* **Why not just better rules?** Rules encode one situation each; the twin-based planner
  re-optimises the whole field for whatever combination of events, prices and states occurs,
  and explains each decision.
* **Is this "AI"?** No external AI model or service is involved. The neural sub-solvers are
  small graph networks trained by the platform on the twin itself; every plan they propose is
  checked and refined by exact optimisation.

# Case study 2: Corvus underground mine

A synthetic sublevel-stoping copper mine in the Gawler Craton style, built to stress the platform: far more
assets than the oil field, real 3D geometry, and an operating cost of about USD 580 M per year.

| | |
|---|---|
| Ore body | three lenses (North, Central, South), 23 production levels 520-1,060 m below surface |
| Assets in the twin | 392 instances (393 in the plant, incl. a stope still in development): 186 stopes, 23 ventilation districts, 6 ore-pass systems, 8 haul routes, 2 crushers, 2 conveyors, skip hoist, 3 main fans, 10 pump stations, 3 loader fleets (27 LHDs), 16 trucks, ROM pad, concentrator, rail load-out, 79 individual mobile units, 64 mine openings |
| Geometry | every point has an elevation; shafts, raises, the decline spiral, ore and haulage drives come from a 3D GeoJSON export (`openings.geojson`) |
| Planning problem | 48 h at hourly resolution: ~5,000 decisions, 744 of them discrete (ring firing per stope and shift) |
| Throughput | ~12 Mt/a (1,450-1,600 t/h milled), Cu-equivalent head grade ~1.2 % |
| Operating cost | labour, G&A, mining consumables, explosives, loader/truck operation and ownership, power (fans, hoist, pumps, mill), mill consumables - about USD 580 M/a with the mill at nameplate (measured for the baseline, see BENCHMARKS.md) |

Everything is in `scenarios/corvus`: `generate.py` writes the asset register (CSV), the 3D openings and the
hidden truth (stope grades ±12 % vs. the block model, upper South zone 22 % lower - the reconciliation
surprise, productivity and water inflows off by realistic margins); `make_scenario.py` writes
`scenario.json`; `field.plx` is the twin (135 lines of PLX plus the tables).

## Events (plant time; Day 1 06:00 = start)

| Time | Event | What it tests |
|---|---|---|
| Day 1 10:00 | central main fan trips for 6 h | ventilation on demand: air is the binding constraint for every diesel loader |
| Day 1 15:00 | seismic event closes the North 560 level for 20 h | re-allocation of loaders and air, blast re-sequencing |
| Day 1 21:00 | skip hoist stopped 5 h (rope inspection finding) | buffer management: ore passes, shaft bin, ROM pad |
| Day 2 03:00 | market forecast: 1,500 USD/MWh tomorrow 16:00-19:00 | load shifting with a day of warning (pumps, hoist, fans) |
| Day 2 08:00 | three Central loaders down for 10 h | loader dispatch by value, not by broken stock |
| Day 2 12:00 | a new stope is handed over; only its tags appear | discovery: the twin rewrites its own code |
| Day 2 16:00 | power at 1,500 USD/MWh for 3 h | demand response |
| Day 2 22:00 | South 930 pump station fails for 8 h | unannounced: the twin sees the sump rise |
| throughout | grab-sample assays and the mill's on-stream analyser reveal the diluted upper South zone | estimation of stope grades |

## Demo script (about 12 minutes)

1. **One operating system for the whole mine** (Overview): tonnes, metal, cost per tonne, site power - both worlds.
2. **The twin is three-dimensional** (3D twin): rotate, zoom, click a stope; colour by activity, grade or broken stock.
3. **Main fan trip** (3D twin): PLYXIS re-balances the remaining fans and regulators within one plan;
   conventional control withdraws equipment from the affected districts.
4. **Why this stope?** (Schedule): decisions with their reasons (grade, tram distance, loader hours, the
   shadow prices of air and hoisting).
5. **Seismic closure, hoist stop, loaders down, the price peak** - the Overview's value bridge shows what
   each event is worth.
6. **The twin rewrites itself** (Twin Studio): the discovered stope becomes PLX code, committed and compiled.

## Assumptions behind "conventional mine control"

`scenarios/corvus/legacy.py` encodes competent short-interval control:

| Area | Rule |
|---|---|
| Loaders | every 4 h and at shift change the shift boss assigns each zone's available loaders to the stopes with the largest broken stocks (1.3 working places per loader), mucking at standard productivity (85 % utilisation) up to the drawpoint limit; grade control keeps the expected blend within 0.05 % of the plan grade using the block model; a place is only assigned where its level's design air still covers its loader's statutory air, and each level's loaders are kept within that air every step (the ventilation officer's air budget); the shift's tonnes target is the mill rate (+15 % while the ore passes are below 40 % full, -10 % above 60 %) |
| Blasting | at shift change, if a zone's broken stock is below 5 days of its share of the mill rate, up to two of its stopes below 15 % broken stock are fired (a broken-stock buffer, not firing every empty stope) |
| Ventilation | main fans at design speed (90 %); regulators set by design (each fan's air shared by its levels, half share where it is the secondary raise) |
| Fan trip | diesel equipment withdrawn from the districts that fan ventilates until it runs again; the other fans run at full speed |
| Ore passes, trucks | level control towards 50 % full; trucks in proportion to the draws |
| Hoist | full speed while the shaft bin is above 15 % |
| Dewatering | pumps start at 70 % sump level, stop at 30 % |
| Concentrator | runs at 1,580 t/h (99 % of nameplate) from the ROM pad whenever the pad holds ore |
| Announced price peak | hoisting and pumping deferred during the window when bins and sumps allow; the mill keeps running while the pad holds ore (its margin of ~110 USD/t at plan grade is three times the peak power cost of 36 USD/t) |
| Outages | both control rooms get the crew's estimate of when a failed hoist, fan, pump, loader or closed level is back; the rules above need no forecast, PLYXIS plans the asset back at that time |

Results of the reference run (48 h) are in [BENCHMARKS.md](BENCHMARKS.md#corvus-mine-showcase-run-48-h):
+12.3 % operating profit at equal mill throughput and equal cash cost, from head grade inside the mine
plan's window (1.449 % vs. 1.368 %): 7.1 % more copper, 7.4 % lower production cost per tonne of copper.
Present it as grade control with an assay-calibrated twin; at site level cash cost is the same.

What PLYXIS adds: it plans all of this jointly over 48 h - which stopes to muck and fire, how many loader
hours and how much air each gets, fan speeds, regulator settings, ore-pass draws, hoisting and pumping
against the time-of-use price - and re-plans after every event.
