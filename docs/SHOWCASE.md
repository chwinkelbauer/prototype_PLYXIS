# Showcase: demo scripts and assumptions

Three case studies run on the same platform; pick one with the title in the top bar (the console stops the
running twin, compiles the other one and starts it - about 20 s for the mine):

* **Kestrel Field** - 96-well Permian oil field, 72 h scenario (this page, first part).
* **Corvus Mine** - large underground copper mine in 3D, 48 h scenario (second part, below).
* **Aquila Steel** - a global steel group, 13 sites on four continents, 96 h scenario (third part, below).

# In every case study: people in the loop

The Shift page (left bar) is worth two minutes after the autonomous story, for anyone who asks "who is in control?":

1. Switch to **Advisory**. The plant now holds what it runs; the badge counts the recommendations waiting.
2. Press **Why** on a change: the reasons come from the model's gradients and shadow prices.
3. **Reject** a change with a reason ("pad is being serviced"). The decision is held, the planner re-optimises around it, the
   audit trail has it with name, time and reason. Then **Accept all**.
4. **Shift report**: the handover, built from the accounts by fixed templates.
5. Optimizer page, **What if**: take an asset out of service for 6 h. What the event costs, and what reacting is worth.
6. `POST /api/event {"kind": "sensor", "tag": ..., "fault": "freeze"}`, then the Data page: the stuck sensor is flagged within
   2 h and the twin predicts the signal instead.

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
assets than the oil field, real 3D geometry, and an operating cost of about USD 500 M per year (unit costs
calibrated on the baseline's measured cash cost, about USD 36 per tonne milled).

| | |
|---|---|
| Ore body | three lenses (North, Central, South), 23 production levels 520-1,060 m below surface |
| Assets in the twin | 395 instances (396 in the plant, incl. a stope still in development): 186 stopes, 23 ventilation districts, 6 ore-pass systems, 8 haul routes, 2 crushers, 2 conveyors, skip hoist, 3 main fans, 10 pump stations, 3 loader fleets (27 LHDs), 3 charge-up crews, 16 trucks, ROM pad, concentrator, rail load-out, 79 individual mobile units, 64 mine openings |
| Geometry | every point has an elevation; shafts, raises, the decline spiral, ore and haulage drives come from a 3D GeoJSON export (`openings.geojson`) |
| Planning problem | 48 h at hourly resolution: ~5,000 decisions, 744 of them discrete (ring firing per stope and shift) |
| Throughput | ~14 Mt/a (1,580-1,600 t/h milled), mine-plan head grade 1.40 % Cu (window 1.35-1.45 %) |
| Operating cost | labour, G&A, mining consumables, explosives, loader/truck operation and ownership, power (fans, hoist, pumps, mill), mill consumables - about USD 500 M/a with the mill at nameplate (measured for the baseline: USD 493-504 M over the seeds, see BENCHMARKS.md) |

Everything is in `scenarios/corvus`: `generate.py` writes the asset register (CSV), the 3D openings and the
hidden truth (stope grades ±12 % vs. the block model, upper South zone 22 % lower - the reconciliation
surprise, productivity and water inflows off by realistic margins); `make_scenario.py` writes
`scenario.json`; `field.plx` is the twin (146 lines of PLX plus the tables).

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
| throughout | composite muck samples at the working places (both worlds) and the mill's on-stream analyser reveal the diluted upper South zone | estimation of stope grades |

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
| Loaders | every 4 h and at shift change the shift boss assigns each zone's available loaders to the stopes with the largest broken stocks (1.3 working places per loader), mucking at standard productivity (85 % utilisation) up to the drawpoint limit; grade control keeps the expected blend within 0.05 % of the plan grade using the block model, reconciled with the mine's own muck-sample assays (same sampling and update as the twin); a place is only assigned where its level's design air still covers its loader's statutory air, and each level's loaders are kept within that air every step (the ventilation officer's air budget); the shift's tonnes target is the mill rate (+15 % while the ore passes are below 40 % full, -10 % above 60 %) |
| Blasting | at shift change, if a zone's broken stock is below 5 days of its share of the mill rate, up to two of its stopes below 15 % broken stock are fired (a broken-stock buffer, not firing every empty stope); two charge-up crews per zone limit both worlds |
| Ventilation | main fans at design speed (90 %); regulators set by design (each fan's air shared by its levels, half share where it is the secondary raise) |
| Fan trip | diesel equipment withdrawn from the districts that fan ventilates until it runs again; the other fans run at full speed |
| Ore passes, trucks | level control towards 50 % full; trucks in proportion to the draws |
| Hoist | full speed while the shaft bin is above 15 % |
| Dewatering | pumps start at 70 % sump level, stop at 30 % |
| Concentrator | runs at 1,580 t/h (99 % of nameplate) from the ROM pad whenever the pad holds ore |
| Announced price peak | hoisting and pumping deferred during the window when bins and sumps allow; the mill keeps running while the pad holds ore (its margin of ~110 USD/t at plan grade is three times the peak power cost of 36 USD/t) |
| Outages | both control rooms get the crew's estimate of when a failed hoist, fan, pump, loader or closed level is back; the rules above need no forecast, PLYXIS plans the asset back at that time |

Results of the reference run (48 h) are in [BENCHMARKS.md](BENCHMARKS.md#corvus-mine-showcase-run-48-h):
+10.3 % operating profit (mean of three seeds, +9.8 to +10.8 %) at about equal cash cost: 6.4 % more copper, 7 % lower
production cost per tonne of copper. About four fifths of the copper is head grade: PLYXIS blends near the top of the plan's
window (1.43 %, limit 1.45 %), conventional control delivers 1.37 %, both inside the window. Present it as grade
control and unit cost, not as a site-level cost reduction, and expect the question "what if we want 1.40 %?": with the window capped
there the lead is +4.9 % (+4.7 to +5.2 %; see BENCHMARKS.md). Also expect "why does conventional control miss its own 1.40 %?": the baseline does not use the mill's analyser yet, which may flatter this lead (BENCHMARKS.md, "What this lead does not settle").

What PLYXIS adds: it plans all of this jointly over 48 h - which stopes to muck and fire, how many loader
hours and how much air each gets, fan speeds, regulator settings, ore-pass draws, hoisting and pumping
against the time-of-use price - and re-plans after every event.

# Case study 3: Aquila Steel - planning a whole corporate

The goal of the platform is to plan the operations of whole corporates, not single sites. Aquila is a synthetic global steel
group with the structure of the large integrated producers: one twin, one plan, every site, market and lane of the group.

| | |
|---|---|
| Sites | 13 on four continents. Integrated (blast furnace -> converter -> slab caster -> hot strip mill, most with a galvanising line): Velsen (NL), Duisburg (DE), Dunkirk (FR), Aviles (ES), Burns Harbor (US-IN), Tubarao (BR), Kalinganagar (IN-OD). Electric (scrap and DRI -> arc furnace -> slab or billet caster -> strip or bar mill): Warsaw (PL), Osceola (US-AR), Corpus Christi (US-TX), Monterrey (MX), Hazira (IN-GJ), Jubail (SA); Aviles also melts in an arc furnace |
| Assets in the twin | 237 instances: 15 blast furnaces, 7 converter shops, 10 arc furnaces, 4 direct-reduction plants, 17 casters, 11 strip mills, 6 galvanising lines, 4 bar mills, 78 yards (ore, coke, pellets, scrap bays, DRI/HBI piles, slabs, billets, coil, coated coil, long products), 13 grid connections in 11 power zones, 13 gas supplies at 5 hubs, 5 scrap markets, 12 product markets, 24 sales lanes, 5 transfers between sites |
| Twin | 2D GIS on the world map; the library `stdlib/steel.plx` (405 lines of PLX) and the group's `field.plx` (173 lines plus the asset register, one CSV per asset class, written by `generate.py`) |
| Planning problem | 96 h at hourly resolution, 2,476 decisions: blast-furnace wind, converter scrap ratio, arc-furnace heats (hourly, for load shifting) and DRI share, DRI plant rates, strip, galvanising and bar mill rates, scrap and HBI purchases, every sales lane and transfer |
| Markets | contract customers call off about 80 % of the group's capacity (a backlog is delivered later, what is left at the end costs liquidated damages); spot buyers take a limited extra volume 40 USD/t below the index - a market-limited group, as most of the industry is |
| Prices | regional coil indices, scrap, HBI, gas hubs and EUAs walk; day-ahead power per zone follows each zone's daily shape in its local time (the European evening peak and the Texan afternoon peak fall on different hours); the EU sites pay for the emissions not covered by free allocation; the corporate's internal carbon price (30 USD/t) is a decision price, reported separately; both worlds plan with it |
| Hidden truth | blast-furnace capacity and coke rate, arc-furnace power and productivity, DRI gas, strip-mill yield and the copper of each region's scrap differ from the engineering data by realistic margins (`truth.plx`); the lab analyses composite scrap samples, about every six hours per bay (both worlds) |

How the economics are kept honest (both worlds, `steel.plx`): inventories are valued at their variable production cost,
capped a little below their net realisable value (IAS 2) - valued higher, a plan makes slabs for the yard; valued lower, it
sells the yard instead of melting. Yards hold the stock band of the corporate S&OP (the start stock to 110 % of it; in the plan a
shortfall costs 30 USD per hour times its square over the target, an excess 0.5 USD/t per hour): working capital is set by
finance, not by the operations plan. Without the floor a plan sells stock down instead of producing - it saves the production's carbon and the
difference between the true cost and the book value, a gain that lasts only until the stock is rebuilt; without the top
it makes steel for the yard in cheap hours, a paper gain at standard cost that nobody uses. Scrap bays and DRI piles
hold their targets the same way, and a plan ends at the targets (planning only: 200 USD per t short at the horizon end) -
otherwise it runs the yards down in its last day and over-produces at its front, the only part a rolling plan executes. In the plan only, every tonne-hour of contract backlog costs
2 USD (customers waiting; not in the accounts): without it shipping now or tomorrow scores the same, and every rolling plan
put deliveries off. A grid connection overrun is charged (1,500 USD/MWh above the agreement), copper above a caster's limit downgrades the cast (a cost per tonne and %-point), and a converter that
cannot take its hot metal sends it to the pig caster.

## Events (UTC; Day 1 00:00 = start)

| Time | Event | What it tests |
|---|---|---|
| Day 1 05:00 | Duisburg blast furnace 1 off wind for 30 h (stave leak, crew estimate given) | re-allocation of iron, slabs and markets across sites |
| Day 1 18:00 | storm closes IJmuiden for 36 h: Velsen ore deliveries at a fifth (notice) | a disruption the yard absorbs: no over-reaction |
| Day 1 20:00 | ERCOT forecasts 1,500 USD/MWh for Day 2 21:00-01:00 (15:00-19:00 CST) | arc-furnace load shifting with a day of warning |
| Day 2 02:00 | Hazira gas curtailed to 45 % for 30 h (notice) | DRI plant, metallic mix and copper limits under a gas constraint |
| Day 2 10:00 | Tubarao strip mill down 12 h | slabs to the yard or to the export market |
| Day 2 21:00 | ERCOT at 1,500 USD/MWh for 4 h | demand response |
| Day 3 02:00 | North American car makers call off 25 % more coated coil for 40 h (notice) | which site supplies, what it displaces |
| Day 3 12:00 | European scrap jumps to 470 USD/t for 36 h | metallic mix: hot metal vs. scrap vs. HBI per site |
| Day 4 00:00 | Monterrey arc furnace transformer trip, 8 h | an electric site down |

## Demo script (about 10 minutes)

1. **One plan for a whole steel group** (Overview): shipments, crude steel, cost per tonne shipped, group power - both worlds.
2. **The value chain on the world map** (Live map): sites, markets, the sales and transfer lanes; one layer per process step.
3. **Blast furnace off wind** (Overview, then Schedule): the group is re-planned; the schedule shows each furnace, mill and lane
   with its reasons (delivered margin, the shadow prices of iron and of rolling capacity).
4. **A price signal a day ahead** (Overview): both worlds know the scarcity window a day ahead; the conventional control room
   stops the Texan arc furnace for it, PLYXIS plans its heats and the billet yard around it.
5. **Gas curtailed, scrap jumps, customers pull orders forward** - the value bridge shows what each part is worth.

## Result (96 h, three seeds)

[BENCHMARKS.md](BENCHMARKS.md#aquila-steel-group-showcase-run-96-h) has the numbers. PLYXIS has **no operating-profit lead** over
the competent S&OP: -1.6 % on the mean of three seeds (-4.9 to +0.8 %). After the group's internal carbon price - the objective
both worlds plan with - it is ahead on every seed (USD +1.0 to +1.9 M per 96 h) with 6.2 % less CO2e. The difference is the
metallic mix: in Europe the plan runs the blast furnaces near turndown and the converters on more scrap, because with the EU's
allowances and the carbon price hot metal costs it more than scrap; outside Europe every furnace runs at full wind. Present it as
that trade, not as a margin gain, and expect two questions: furnace turndown costs no extra fuel in the model (optimistic for this
strategy), and the European crunch in the first two days hits a plan with its furnaces down (one seed lost USD 1.4 M in eight hours).
The allocation of markets to sites, which a good S&OP does by merit order, leaves little to improve in a group that is limited by
its markets.

## Assumptions behind "conventional corporate planning"

`scenarios/aquila/legacy.py` encodes a competent group that plans with spreadsheets and standard costs instead of a corporate
optimiser:

| Area | Rule |
|---|---|
| S&OP | every 24 h, and at once after a notice or when a unit goes down or comes back, the planning department allocates each market's contract volume to the sales lanes in order of delivered margin (index price - freight - the site's standard variable cost - its standard CO2e at the internal carbon price, the price of the plan's objective), within each site's capacity (furnaces, converters, casters, mills at 97 % utilisation) and each lane's capacity - a contract also at a loss smaller than its liquidated damages; spot volume next where the margin after the spot discount is positive; a strip mill may roll a sister plant's slabs where that is the better margin; Tubarao slabs go to Burns Harbor and Texan DRI to Osceola wherever that is cheaper than the destination's own cost; blast furnaces stay at least at their 60 % turndown, and what that iron makes beyond the allocation is offered spot wherever it covers freight and rolling |
| Make to order | each mill rolls what its lanes ship plus a correction towards its yard's stock target (over 24 h); an arc furnace is planned at what it can make on average (its capacity less the tariff's peak hours, within the site's grid connection), so the sales plan does not run its yards down; the melt shops make what the mills and slab lanes need, with the same correction for the slab and billet yards; blast furnaces by share of full wind, within 6 % of full wind per hour; converters at the standard 84 % hot metal, 78 % while a site's iron is short |
| Arc furnaces | follow the published time-of-use tariff: 30 % in the four peak hours of their zone, made up in the other hours; stop during an announced price window above 300 USD/MWh (demand response); the site's grid connection is respected |
| Metallics | DRI/HBI share at the furnace's maximum where DRI is cheaper than scrap (own plant, transfer or merchant HBI, including its extra power and its carbon), otherwise the least share that keeps the cast's copper 10 % inside the caster's limit, with the lab's running mean of the bay's copper (the same samples PLYXIS gets); DRI plants make what their furnaces and transfers draw, within the gas contract; scrap bays and HBI piles are bought back to their stock target |
| Sales desk | lanes ship the allocation while the yard holds stock (spot only above 60 % of the target), sell what a yard holds above 115 % of its target, catch up a market's backlog, and never ship more than a market's call-offs plus its spot depth |
| Notices and outages | both planning departments get every notice at the same time, and the crew's estimate of when a failed unit is back |

The baseline needs no help from the plant: over 12 h in `tests/test_steel.py` its plant's protection layer cut nothing and no
yard was drawn below empty; every A/B result file records the same for both worlds (`protection`).
