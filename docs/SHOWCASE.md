# Showcase: Kestrel field - demo script and assumptions

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
[BENCHMARKS.md](BENCHMARKS.md#showcase-run-72-h).

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
