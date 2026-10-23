# Benchmarks

Every number here is produced by the scripts in `bench/` and stored in `bench/results/`
(`python -m bench.report` prints these tables). Hardware: one laptop, AMD Ryzen 5 PRO 4650U
(6 cores, 15 W), 16 GB RAM, CPU only, PyTorch 2.14. All data is synthetic.

```bash
python -m bench.showcase_run --hours 72                      # A/B run of the full showcase
python -m bench.twin_accuracy --hours 30 --full              # twin vs. hidden truth over time
python -m bench.solver_comparison --budget 60                # solvers at equal wall time
python -m bench.neural_ablation --experts 60 --test 8        # what the neural sub-solvers add
python -m bench.scaling --fields 1 2 4 8 --ga                # enterprise twins up to 768 wells
```

## Showcase run (72 h)

`bench/results/showcase_run.json` - the full Kestrel scenario (two compressor trips, a fast-wearing
ESP, an unannounced 55 % loss of disposal injectivity, 24 h of negative gas prices, a new well that
appears only as SCADA tags). Two worlds start from the same state, see the same hidden physics,
events and sensor noise: one operated by PLYXIS (re-planning every 3 h and after every event), one by
a rule-based model of competent manual operations ([SHOWCASE.md](SHOWCASE.md)).

| KPI (72 h) | PLYXIS | manual operations | difference |
|---|---|---|---|
| **Profit after carbon cost** (USD) | 3,504,762 | 3,421,457 | **+83,305 (+2.4 %)** |
| Operating profit (USD) | 3,532,677 | 3,464,328 | +68,349 (+2.0 %) |
| Oil produced (bbl) | 57,071 | 55,997 | +1,074 (+1.9 %) |
| **Gas flared** (Mscf) | 185 | 2,600 | **-2,415 (-93 %)** |
| **Emissions** (t CO2e) | 465 | 715 | **-249 (-35 %)** |
| Unplanned pump failures | 0 | 1 | |
| Pump replacements (planned) | 1 | 0 (the failed pump waits for the rig) | |

How the comparison is valued - identically for both worlds: sales minus cash costs, plus the change
in inventories (oil in tanks at 90 % of the price, water in tanks as a disposal liability - the
planner's own terminal valuation), plus the change in remaining pump life (health × pump cost; a
replacement is charged at the pump cost, an unplanned failure additionally at the model's failure
premium). Planning-only terms (expected failure cost, flaring-risk margin, nominal wear) are not
counted. The carbon cost uses the field's internal price (60 USD/t), which is also what the planner
optimises.

Where the difference comes from (USD over 72 h, + = in favour of PLYXIS): crude sales +100,341;
avoided unplanned failure +45,000; pump life +74,366 net of one planned replacement (-140,000 cash);
flaring-risk exposure and gas sales +12,806; more well restarts -5,383; power -3,130; carbon +14,956.

Twin and planner during the run: well deliverability error 11.9 % → 2.4 %, GOR 12.9 % → 5.4 %,
water cut 4.7 % → 2.1 %; 32 plans of 4,983 decisions each, mean solve time 7.9 s (max 9.9 s), 29 of
them within the feasibility tolerance (the other three exceed a limit by at most 0.22 % of its scale).

**How this number was arrived at.** An earlier version of this benchmark reported +16 % operating
profit. Investigation showed that most of it was not PLYXIS: the emulator's protection layer
over-throttled overloaded flows (5 % below the limit, compounded across limits), which hurt the
rule-based world more, and the baseline's water handling was weaker than a real control room. The
accounting also ignored pump condition and carbon. All of this was fixed before the numbers above
were produced; they are the conservative, defensible figure.

## Twin accuracy over time

`bench/results/twin_accuracy.json` - every hour of plant time the live twin is compared with the
emulator's hidden truth (production planner running, all scenario events active). The twin sees
only SCADA telemetry and one well test per hour.

| plant time (h) | well deliverability q_max | GOR | water cut | pad gas, one step ahead (mean / max) | compressor stations |
|---|---|---|---|---|---|
| 6 (start, engineering model) | 10.9 % | 12.7 % | 4.7 % | 9.2 % / 33 % | +1.0 %, +3.1 % |
| 10 | 3.5 % | 12.2 % | 4.5 % | 7.7 % / 19 % | +0.8 %, +0.7 % |
| 18 | 2.6 % | 11.2 % | 4.2 % | 6.4 % / 20 % | -1.0 %, +3.2 % |
| 26 | 2.3 % | 10.7 % | 3.8 % | 6.7 % / 19 % | -0.4 %, +3.2 % |
| 36 | 2.4 % | 9.8 % | 3.5 % | 5.7 % / 19 % | +0.3 %, +2.9 % |

Well deliverability is pinned down within four hours; GOR and water cut improve as well tests
rotate through the 96 wells (after 72 h: GOR 5.4 %, water cut 2.1 %, see the showcase run). The
unannounced loss of SWD-2 injectivity is re-estimated within three hours (19,970 → 9,531 bbl/d,
true 9,450). With the plant's true parameters, the twin reproduces all ~1,000 plant signals exactly -
the remaining error is estimation error, not model error.

## Solver comparison (equal wall time)

`bench/results/solver_comparison.json` - the Kestrel planning problem (72 h, hourly):
4,983 decision variables of which 1,272 discrete, ~7,300 constraint signals. Every method starts
from the same default plan (which violates constraints); every final solution is evaluated
identically (discrete decisions rounded, exact simulation). Objective = 72 h operating profit minus
carbon cost. Budget 60 s; the gradient methods stop by themselves.

| Method | objective (USD) | max violation | feasible | seconds |
|---|---|---|---|---|
| PSO | 3,024,280 | 1.2453 | no | 60.0 |
| gravitational search | 1,878,639 | 0.1045 | no | 60.1 |
| GA | 4,263,637 | 0.0000 | yes | 60.0 |
| DE | 3,772,366 | 0.0000 | yes | 60.0 |
| PSO, cooperative (per subsystem) | 1,825,838 | 45.2643 | no | 60.0 |
| GA, cooperative (per subsystem) | 1,612,341 | 0.0000 | yes | 60.0 |
| augmented Lagrangian, single start | 3,281,208 | 0.0022 | yes | 8.4 |
| **PLYXIS cascade** | **4,376,285** | 0.0015 | yes | **13.5** |
| **PLYXIS cascade + neural sub-solvers** | **4,636,228** | 0.0018 | yes | **13.7** |

Reading: the vectorised GA is the strongest classic method on this problem, but it needs the
whole minute to recover feasibility and ends 8.7 % below the cascade with neural sub-solvers,
which finishes in under 14 s. PSO and gravitational search - the original concept's front end -
do not reach a feasible plan at this dimension. Cooperative decomposition alone does not fix
population methods: the shared constraints (compression, takeaway, permits) couple the
subsystems. A single gradient start is fast but stuck in a local optimum; the cascade's
multi-start, discrete search and polish recover 33 % more value. How the GA degrades as the
problem grows is measured in the scaling benchmark below.

## Neural sub-solvers (ablation)

`bench/results/neural_ablation.json` - expert iteration: the cascade solves 60 realistic event
situations (compressor trips, ESP failures, injectivity losses, price shocks, tank levels,
deliverability drift) warm-started from the current plan, as in live operation; the graph network
is trained on them (173 s). Then 8 held-out situations are re-planned with equal budgets:

| Start / method | mean objective (USD) | feasible | mean seconds |
|---|---|---|---|
| previous plan, as is | -838,701 | 1/8 | - |
| neural sub-solver plan, as is | 529,681 | 0/8 | 0.02 |
| previous plan + gradient repair | 3,455,581 | 8/8 | 1.91 |
| neural plan + gradient repair | 3,578,726 | 8/8 | 1.82 |
| full cascade | 3,831,070 | 8/8 | 6.33 |
| full cascade + neural start | 3,909,037 | 8/8 | 6.48 |

* The network's raw plan (20 ms for the whole field) is not feasible on its own - it is a start,
  and is always repaired and verified.
* With the same 1.8 s repair, the neural start is worth 3.6 % more than repairing the previous plan.
* Inside the full cascade, the neural start wins 7 of 8 cases (median +2.5 %, mean +2.0 %; per case
  +4.0, -0.5, +0.1, +0.9, +3.5, +2.0, +3.0, +3.0 %) at the same solve time.

## Scaling: from one field to a company

`bench/results/scaling.json` - generated enterprise twins: N copies of the Kestrel field (each with its own
pads, batteries, compression, disposal and trucking) sharing one gas plant, one oil terminal and a company-wide
flare permit. The same cascade configuration solves every size; the GA gets exactly the cascade's wall time.

| Fields | wells | decisions (discrete) | compile | simulate 1 / 16 plans | gradient | generated code | cascade solve | cascade objective | GA, same time |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 96 | 4,983 (1,272) | 0.22 s | 10 / 16 ms | 20 ms | 429 lines | 17 s | USD 4.27 M, feasible | infeasible (max violation 0.08) |
| 2 | 192 | 9,966 (2,544) | 0.19 s | 9 / 21 ms | 20 ms | 429 lines | 18 s | USD 8.28 M, feasible | infeasible (max violation 1.09) |
| 4 | 384 | 19,932 (5,088) | 0.28 s | 11 / 58 ms | 23 ms | 429 lines | 26 s | USD 15.02 M, feasible | infeasible (max violation 2.25) |
| 8 | 768 | 39,864 (10,176) | 0.50 s | 16 / 76 ms | 26 ms | 429 lines | 55 s | USD 29.98 M, feasible | infeasible (max violation 2.37) |

* **Adding assets adds tensor rows, not code.** 768 wells compile to the same 17 entity groups and the same
  429 lines of generated simulator code as 96 wells; simulating a 72 h plan for the whole company takes 16 ms.
* **Solve time grows sub-linearly with size**: 8× the decisions (39,864) take 3.3× the time (55 s vs. 17 s),
  because the decomposition keeps subsystems local and gradients cost a small multiple of a simulation.
* **Population methods fall further behind as the problem grows**: with the same wall time, the GA ends
  infeasible at every size, and its constraint violation grows from 0.08 (1 field) to 2.4 (8 fields). Its
  higher 'objective' at 4 and 8 fields comes from violating shared capacity limits.
* Per-field value declines slightly with size (USD 4.27 M for one field, 3.75 M per field at 8) because the
  shared gas plant, terminal and company flare permit bind - exactly the coupling a company-level optimiser
  has to resolve.

## What these numbers do not show

* **Real-world uplift.** The A/B gain depends on the baseline and on how constrained the field
  is. The baseline is a rule-based model of competent manual operations (documented in
  [SHOWCASE.md](SHOWCASE.md)); real operators have more context and also more delays. Pilot
  targets are deliberately lower ([PILOT_PLAN_OIL_GAS.md](PILOT_PLAN_OIL_GAS.md)).
* **Model-form error.** The emulated plant is written in PLX, so the twin's structure matches
  reality; only parameters, drifts, noise and events differ. Real fields add structural mismatch,
  which the pilot's backtest gate measures.
* **Statistical spread.** Each benchmark is a single seed (the neural ablation uses 8 held-out
  situations). The direction of the results has been stable across runs; their exact size is not.
