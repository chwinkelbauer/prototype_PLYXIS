# CLAUDE.md - PLYXIS prototype 2

PLYXIS ("The Operating System for Industries", https://plyxis.at) builds digital twins of industrial
sites as code and runs them autonomously. Founder: Christian Winkelbauer. This repo is the investor /
pilot-customer prototype: PLX language + compiler, data integration, layered optimizer with neural
sub-solvers, live console. All data is synthetic.

## Non-negotiables

- **No external AI.** Never call an LLM or any external AI service from the product. Learned parts
  (neural sub-solvers) are trained locally from the twin. Explanations come from gradients and shadow
  prices, rendered by deterministic templates.
- **Honest benchmarks.** The A/B showcase compares PLYXIS with a *competent* rule-based model of manual
  operations (`scenarios/kestrel/legacy.py`). Never weaken the baseline, tweak the emulator or the
  accounting in ways that only favour PLYXIS, or hand-enter numbers. Both worlds share the same hidden
  truth, events, noise and valuation (`plyxis/data/emulator.py::_Kpi`). Gains must come from better
  decisions; scenario changes must be realistic and documented in `docs/SHOWCASE.md` /
  `docs/BENCHMARKS.md`. An earlier +16 % claim turned out to be an emulator/baseline artifact - see the
  "How this number was arrived at" note in BENCHMARKS.md. Keep that standard.
- **Docs follow the code.** Every number in README / docs must come from `bench/results/*.json`
  (`python -m bench.report` prints the tables). Re-run the benchmark that produced a number after
  changing anything that affects it.

## Commands (Windows, Git Bash or PowerShell)

```
.venv/Scripts/python.exe -m pytest tests -q            # about 155 tests, ~10-12 min
.venv/Scripts/python.exe -m plyxis serve --port 8000   # console at http://127.0.0.1:8000 (needs web/dist)
cd web && npx vite build                               # rebuild console; npx tsc -b --noEmit to type-check
.venv/Scripts/python.exe -m bench.showcase_run --hours 72      # Kestrel A/B, ~10 min
.venv/Scripts/python.exe -m bench.showcase_run --scenario corvus  # mine A/B (48 h), writes showcase_run_corvus.json
.venv/Scripts/python.exe -m bench.showcase_run --scenario aquila --seed 1   # steel group A/B (96 h), showcase_run_aquila_s1.json
.venv/Scripts/python.exe scenarios/aquila/generate.py; .venv/Scripts/python.exe scenarios/aquila/make_scenario.py
.venv/Scripts/python.exe -m bench.policy_in_twin --scenario aquila   # plan vs the baseline's rules in the twin's own world (no truth, no events)
.venv/Scripts/python.exe -m bench.corporate --rounds 3 --reference --workers 6  # group hourly + the works' plants found at 15 min; toy check
.venv/Scripts/python.exe -m bench.corporate --regions --rounds 5 --workers 6     # the hourly group: whole vs its regional networks
.venv/Scripts/python.exe -m bench.corporate --plant-ablation                     # each plant alone: GA vs per-unit DPs
.venv/Scripts/python.exe -m bench.decomposition --seeds 3       # GA vs GA + exact sections, filter, native kernels
.venv/Scripts/python.exe -m bench.section_theory                # section DP vs exact MILP (HiGHS), ~25 min
.venv/Scripts/python.exe -m bench.section_theory --dual smooth --tag smooth   # tight certificate, ~35 min
.venv/Scripts/python.exe -m bench.showcase_run --scenario corvus --seed 1 [--no-branches --tag nobranches]
.venv/Scripts/python.exe -m bench.exact_branches                # structure classes + exact/sequential LP on the twins
.venv/Scripts/python.exe -m plyxis.optim.structure scenarios/corvus   # which branches are LP / SLP / convex / nonlinear
.venv/Scripts/python.exe bench/checks/pwl_slope.py                    # reverse-mode slopes vs finite differences (also tangent_bound.py, cold_plans_old_vs_new.py); from the repo root
.venv/Scripts/python.exe -m plyxis.native build                 # C++/OpenMP kernels (g++ / MinGW-w64); NumPy fallback
.venv/Scripts/python.exe scenarios/corvus/generate.py; .venv/Scripts/python.exe scenarios/corvus/make_scenario.py
.venv/Scripts/python.exe -m bench.twin_accuracy --hours 30 --full
.venv/Scripts/python.exe -m bench.solver_comparison --budget 60
.venv/Scripts/python.exe -m bench.neural_ablation --experts 60 --test 8   # retrains scenarios/kestrel/models/subsolver.pt
.venv/Scripts/python.exe -m bench.scaling --fields 1 2 4 8 --ga
.venv/Scripts/python.exe -m bench.solver_budget --cases 5   # live solver budget vs previous / large
.venv/Scripts/python.exe -m bench.backtest --scenario kestrel   # pilot gate G1: historian export -> twin -> unseen history
.venv/Scripts/python.exe -m plyxis lint scenarios/corvus/field.plx         # what compiles but will not do what you think
.venv/Scripts/python.exe -m plyxis structure scenarios/aquila_works/field.plx --sim corp96 --detail   # the block tree, and why
.venv/Scripts/python.exe -m bench.structure                    # the block tree of every twin -> structure.json
.venv/Scripts/python.exe -m plyxis backfill history.csv.gz --scenario kestrel   # calibrate a fresh twin from an export
.venv/Scripts/python.exe -m bench.risk                  # outcome distributions: sigma points vs Monte Carlo (both twins); --frontier
.venv/Scripts/python.exe -m bench.uncertainty_coverage --scenario kestrel --hours 48   # posterior intervals vs hidden truth
.venv/Scripts/python.exe -m bench.colgen                # column generation vs exact MILP by site size; --learn (learned prices)
.venv/Scripts/python.exe -m bench.reduction --sites 1 2 4 8 16 32 64   # reduction engine vs whole-model MILP as a twin grows; + filters on the 4 twins
.venv/Scripts/python.exe -m bench.reduction --parts slp      # sequential LP on the algebraic model vs the cascade (steel group)
.venv/Scripts/python.exe -m bench.console_run --scenario aquila   # the A/B as the console runs it (600x, plans in the background)
.venv/Scripts/python.exe -m bench.plan_quality           # one steel plan: cascade vs a better stored plan, restarts, exact polish, a step's errors
.venv/Scripts/python.exe -m bench.planner_speed          # plan time and value per twin: morning planner, current, quarter budget + polish (alone!)
.venv/Scripts/python.exe -m bench.showcase_run --scenario aquila --scenario-dir <copy> --tag x   # a scenario variant from a copy
.venv/Scripts/python.exe -m bench.report                # markdown tables from bench/results
```

Benchmarks that report solve times (decomposition, section_theory, solver_comparison, solver_budget, scaling, exact_branches) are
timing-sensitive: run them one at a time, and not while the console server or a web build is running. The showcase A/B has no wall-clock
budgets, so seeds can run side by side (three at the default thread count take about 35-55 min on the mine); its solve times then mean
nothing. It reads `field.plx` and the stdlib again at a twin rewrite, so never edit those under a running benchmark: run from a copy of
the code. Compare arms only at equal thread counts (`PLYXIS_THREADS`; 2 instead of 6 threads moved the same oil-field seed by USD 31 k,
a rerun at equal threads by USD 136). After a model change run `neural_ablation` first (it retrains `subsolver.pt`), then the
showcase and the rest. Blueprints that no case study instantiates do not touch the neural layout.

## Layout

- `plyxis/lang` PLX language: parser, elaborator, typed IR, compiler to a time-vectorised differentiable
  torch simulator (planning and physical semantics), reference interpreter. `stdlib/*.plx` are libraries.
- `plyxis/optim` decision space, problem, ALM, metaheuristics, partition, neural sub-solvers, cascade,
  explanations, dispatch, maintenance.
- `plyxis/data` connectors, plant emulator (hidden truth + protection layer + KPI accounting),
  estimation (MHE with L-BFGS, prediction step, adaptive prior release), anomaly, discovery, repo.
- `plyxis/runtime` engine (two worlds, plant-clock thread, planner thread), planner, twin manager; `advisory.py` operating
  modes, recommendations, audit log; `report.py` shift report, work orders; `whatif.py` what-if runs; `backfill.py`.
- `plyxis/data` also: `quality.py` data quality in front of the estimator, `industrial.py` PI Web API / Sparkplug B / OPC UA
  connectors (read only, tested against local stand-ins only), `historian.py` export format. `plyxis/lang/lint.py` model lint.
- `plyxis/optim/sections.py` deterministic sections (graph filter net, clustering, exact section search);
  `decomp.py` section tables + forest DP on the twin; `sectiondp.py` the model-agnostic core (forest DP,
  induced-forest moves, dual decomposition with certified bound); `structure.py` proves problem classes from
  the planning IR (DCP-style curvature); `branchlp.py` exact LP / sequential LP of those branches (HiGHS), `tangent_gap`;
  `plyxis/native` C++ kernels (`sections.cpp`: clustering, scoring, Viterbi, forest DP, coordinate ascent)
  with ctypes wrapper and NumPy reference versions. Theory: `docs/DECOMPOSITION.md`.
- The block tree of the asset graph: `plyxis/lang/blocks.py` `find_structure` cuts the graph at shared exchanges, time
  scales and large stocks, level by level (no notion of plant or line in it); `compile_hierarchy(split="auto")` takes
  its blocks at their own step. The earlier detectors in `plyxis/lang/hierarchy.py` - `detect_plants` (units whose
  durations, stated in time, need a finer step; grown through small buffers, cut at large stocks and stockless units),
  `detect_regions` (exchanges shared by several networks stay in the master), `compile_hierarchy(split=...)` splits one
  twin into the master and one program per block (a sim's `plants:` section still overrides with declared groups);
  `plyxis/optim/corporate.py` coordinates them by prices (Dantzig-Wolfe: probe-slot prices, block columns, block
  weights, commit; `workers` solves blocks side by side in processes); `plyxis/optim/unitdp.py` exact per-unit dynamic
  programs for sequence rules (gap / run / reset), the cascade's `discrete = "auto"` uses them wherever rules exist.
  `scenarios/aquila_works` = Aquila with the Duisburg works machine by machine (`steel.plx` "a works in detail";
  not a console scenario yet, no truth/baseline). `examples/plants_toy.plx` the toy both the test and the bench use.
- The exact layer: `plyxis/optim/algebra.py` the algebraic backend of the compiler (`linearize(pb, x)`: every node's
  value and sparse Jacobian in the lifted form - states and recursive delays as columns, so rows are local in time;
  exact where affine, piecewise terms of discrete decisions encoded exactly with indicators, first order elsewhere;
  `check()` against the simulator); `exact.py` HiGHS LP / MILP on any part of it (row scaling, elastic hard rows at
  the planner's price, soft slacks, trust region, local branching for integers, shadow prices) and `slp()` the
  trust-region sequential LP judged on the twin; `reduce.py` the reduction engine (filters contract the math graph:
  LP cells, discrete blocks under an integer cap, recourse, hubs, closure; Dantzig-Wolfe with exact block MILPs,
  Lagrangian bound, commit). Tests: `tests/test_reduce.py`; `tests/models/kilns.plx` the small discrete twin.
- `plyxis/optim/risk.py` planning under uncertainty (uncertain inputs as factors, sigma-point propagation of every limit and
  the plan value, Sobol remainder, back-offs, `RiskProblem` for the spread cap); the covariance comes from `Estimator.cov`
  (information filter around the MHE) and `Forecaster.log_sigma`. `plyxis/optim/colgen.py` column generation with exact
  per-asset DP (unit-commitment family, master LP, Lagrangian bound, polynomial integer step); `learnprice.py` learned prices.
- `plyxis/server` FastAPI + WebSocket (`/api/scenarios`, `POST /api/scenario` switches the case study).
  `web/` React console (Vite, TS, Tailwind 4, MapLibre, ECharts, Monaco, three.js for the 3D twin).
- Case studies: `scenarios/kestrel` (oil field, 72 h), `scenarios/corvus` (underground copper mine, 48 h,
  3D, ~USD 500 M/a OPEX and ~14 Mt/a measured; register/geometry generated by `generate.py`, scenario by `make_scenario.py`),
  `scenarios/aquila` (global steel group: 13 sites, 237 instances, 96 h, 2,476 decisions; library `stdlib/steel.plx`;
  register, truth and inventory values by `generate.py`, prices/events/demo by `make_scenario.py`, baseline = corporate S&OP
  + site control rooms in `legacy.py`; times are UTC, each power zone has its daily shape in local time).
  `scenarios/mining_demo` (open pit) and `enterprise` are solver demos without a console scenario.
  `bench/` evidence. `docs/` documentation.

## Console design system (keep it dark - the founder works at night)

Matches the logo (`web/public/logo.svg`) and plyxis.at:

| Token | Value | Use |
|---|---|---|
| bg | `#09090b` | page |
| panel / panel2 / panel3 | `#12121a` / `#16161f` / `#1a1a28` | cards, raised surfaces |
| line / line2 | `#232334` / `#2e2e44` | borders |
| ink / muted / dim | `#f4f4f5` / `#a1a1aa` / `#71717a` | text |
| brand (primary, PLYXIS world) | `#FE0265` (hover `#e0025a`, light `#ff65a3`) | primary actions, PLYXIS series |
| violet (secondary) | `#8B5CF6` | plans, optimizer, links |
| amber / gold | `#F59E0B` / `#D4AA00` | value, highlights (logo gold) |
| orange | `#D68519` | logo bottom bar, warnings |
| manual operations | neutral slate `#8b93a7` | the comparison world - never a brand colour |
| good / critical | `#34d399` / `#ef4444` | semantic status only |

Gradients: `135deg #FE0265 -> #8B5CF6` and the logo diagonal `#5a2ca0 -> #ff0066 -> #d4aa00`.
Fonts: Outfit (headings, hero numbers), Inter (UI text), JetBrains Mono (numbers, code, tags).

## Conventions

- Python 3.11+ (dev on 3.14), torch CPU. Keep generated-code paths vectorised; no per-instance Python loops
  in hot paths.
- PLX: string literals interpolate `{name}` from loops / discovery bindings; `site` is a kind keyword.
- Engine KPIs: realised accounting incl. inventories (model `final` terms) and remaining pump life;
  planning-only terms (`risk`, `flare_risk`, `pump_wear`) are excluded; carbon reported separately.
- UI numbers that change every tick are shown smoothed (rolling means); raw values stay available.
  Executive tiles on the Overview are cumulative (they converge); rates live in the charts.
- Charts: categorical slots in fixed order `SERIES` (pink, blue, orange, violet, aqua; validated for the
  dark surface); PLYXIS = brand pink, manual operations = slate; status colours only for state.
- Git: the user commits; do not commit unless asked. Never commit `__pycache__`, `.venv`, `web/dist`.

## Mechanisms worth knowing (and traps already hit)

- **Plans can lose their way, and why.** The ALM keeps its best iterate among those with violation up to `tol`. In the relaxation at 0.002 a poor but strictly feasible
  iterate beat a far better one the discrete stage and polish could still repair: about one mine plan in 50 fell from USD 7.5 M to 5.5-6.8 M of plan value, not reproducibly (the same seed and
  code once ended USD 360 k apart, a plan idled the mill for three hours), and the first plan with a capped grade window idled the mill for 7-8 h. `relax_tol` 0.01 (mine, `scenario.json`) fixes it: no such
  plan in 57, capped window from -4 % to +4.9 %, main result unchanged within noise, seeds closer together. Neutral on the oil field (kept at the default there). Safety nets that stay on: the running plan as a
  candidate after the polish (`kept_running_plan`), a second solve from another random stream (`planner.second_opinion`). When an A/B looks odd, read `runs[].objective` plan by plan first.
- **Operating modes** (`runtime/advisory.py`): autonomous, advisory, shadow. In advisory mode the plant runs `held` set points; a rejected or
  modified decision (set point or line-up) needs a reason, is held until the shift ends and is pinned for the planner
  (`Problem.set_pins`: every evaluation of every stage). A plan expires after 12 h unless a newer one is on its way (at 600x time compression a
  60 s solve is 10 plant hours: counting the expiry from the plant clock alone would stop the mine in a live demo).
- **Assays and margins.** Composite muck samples at the working places (scenario `assays`, Kalman update, same rule in both worlds; the baseline
  reconciles its block model) give `Estimator.uncertainty`; `Engine._update_margins` plans the head grade one standard error of the blend
  inside the window. After a twin rewrite the last plan's blend does not map: keep the margin (an even blend of all open stopes gave 0.002).
- **Zustand selectors must return stable values.** `useStore((s) => Object.keys(...))` returns a new array every time: React error #185, the page
  goes blank. Select the object and `useMemo` the rest.
- **Console start states** (`server/app.py`, `web/src/App.tsx`): `boot()` and `ticker()` are tasks nobody awaits, so
  each handles its own failures. A failed start, including a failed first plan (`_replan` only logs), goes to
  `/api/health.error` and a `boot_error` push; the console shows it with retry buttons instead of spinning for ever.
  No snapshots while booting: they used to flip the page between the starting screen and an empty console. Pushes are
  strict JSON (`_dumps`): one NaN token makes the browser's `JSON.parse` reject the whole message. If no snapshot
  arrives for 3 s the console polls `/api/snapshot`. It opens paused at the first hour for the guided demo, and
  `StartBar` says so: `$0` and `NaN` alone read as "stuck".
- **Anytime planning**: urgent triggers (disruption, notices, twin rewrite, asset back) publish a
  first response (short ALM from the shifted plan + neural proposal) in 1-2 s, then the full cascade
  (~10 s, `Planner.LIVE`) replaces it. The console holds the plant clock only until the first response.
- **Move suppression** (`Problem.set_move_reference`): decisions inside the next 3 h pay USD 1,500 per
  full-range change (quadratic) - stops plan churn between re-plans. Quarter weight after urgent triggers.
- **Feasibility restoration** in the cascade (`restore_iters`): only when the polished plan is still
  infeasible; frees discrete decisions, re-rounds, polishes. Reported under the polish stage.
- **Calibration runs off the plant clock** (`calibrate_compute` on a worker, `calibrate_apply` at the
  next step); headless runs (`sync_plans=True`) keep it inline for determinism.
- **Forecasts**: observations inside an announced window are not the new normal (excluded from the
  power level/profile and from gas persistence). A forecaster that extrapolates a price spike makes
  the planner shut the field for the rest of the horizon.
- **Scenario events**: `hours` on a `capacity` event = auto-restore after that time (engine
  `restores`); on a `price` event = override duration. `notice: true` updates the twin directly and
  calls `legacy.notice()` - both control rooms always receive notices at the same time.
- **Baseline parity**: when a scenario event is added, give the manual-operations model the standard
  response a competent control room would have (e.g. immediate curtailment on a midstream notice,
  the 75 % demand-response set point on an announced price cap) in the same change.
- **A competent baseline respects the limits its crew knows.** Check the legacy world's protection-layer
  `throttle_log` and its set points vs. what the plant did: the mine baseline set ~5,200 t/h of loader
  set points that the ventilation interlock cut to ~1,000 t/h (equal loader shares, blind to each level's
  air), which made PLYXIS look +34 % better. It now allocates within each level's air budget, runs the
  other fans at full speed after a fan trip and works to a tonnes target (the mill rate). A rule that
  only "works" because the plant's protection rescues it is not a competent rule.
- **Planner valuations must match the KPI accounting**: e.g. pump wear is priced at
  `pump_cost / (1 - fail_health)` per unit health because the KPI values remaining life from health 1
  down to `fail_health`. Check new cost terms against `_Kpi` before trusting a result.
- **Neural sub-solver features** = all parameters + states of each entity type, padded to the largest
  type. Adding a property to a blueprint shifts the layout: retrain (`bench.neural_ablation`).
  Loading skips mismatched tensors silently (the net then only adds a poor start).
- **Value bridge** (`Engine._bridge`): the operating-profit difference by realised term; sums exactly
  to `delta.profit`. Use it to explain every result before writing it into docs.
- **Large fan-outs as requests, not routing**: a pool feeding dozens of consumers (loader fleet -> 60 stopes,
  fan -> 23 districts) must be modelled as consumers *requesting* from the pool with a sum constraint at the
  pool. Routed split fractions over many destinations start at ~1/n each and gradients barely move them -
  the mine planner lost to the baseline until this was changed. Routing is right for few destinations
  (line-ups, 2-4 outlets).
- **Model incentives**: check every decision for perverse gradients before trusting a plan (a blast that
  scaled mucking down was used as a throttle; mucking now uses `min(muck, cap after re-entry)`). Terminal
  values must not let the plan create value by doing work or by parking rich material at the horizon end:
  realised ore inventories (broken ore, passes, bin, pad) are valued at cost (IAS 2), never at their metal -
  the first mine A/B showed a +$0.6 M "gain" that was exactly that (rich ore in the passes, less copper
  sold). Always read the bridge (and `inventory_terms`) before believing a result.
- **Flow-weighted attributes** (grades) explode when flows are slightly negative. Blend states with
  `pos(stock)`, and the estimator clamps assimilated measurements to constant state limits
  (`Estimator._const_lo`): a noisy reading of an empty bin is not a negative stock.
- **The flow-weighted attribute of no flow is 0.** A soft window on a blended grade must be weighted by the
  flow (`throughput * g >= throughput * grade_min`, penalty per tonne), otherwise an idle mill "violates"
  the lower limit every hour it stands: the mine planner milled through a 1,500 USD/MWh peak and rationed
  a full ROM pad for a whole night to avoid that phantom penalty.
- **Outages with a crew estimate**: `fail`/`trip` events with `hours` give the twin the estimated return
  (`eta_hours`, default `hours`; `eta: false` withholds it) via `Engine.outages` ->
  `Problem.set_state_schedule` (a `[1, n, T]` trajectory of the constant state `up`/`avail`). Without it
  the planner assumes every failed asset stays down for the whole horizon.
- **Terminal values: inventories at cost, buffers get reserves - not a value-to-go.** Planning-only `final`
  terms are named `value_to_go` (excluded from the KPI by `_Kpi.EXPECTATION_TERMS`); only broken ore uses
  one (half its cost in the plan so rings are fired just in time, full cost in the KPI). A value-to-go on
  the ROM pad at the full milling margin was tried and removed: it assumes spare mill capacity after the
  horizon. Linear (100 USD/t) it made the planner hoist at the hoist limit and idle the mill for eight hours;
  concave (working stock only) it still idled the mill for five hours of a price peak in a mill-bound mine
  (~7,000 t of milling lost to save USD 180 k of power). Which resource is the bottleneck decides what a
  terminal tonne is worth - the planner sees that inside its horizon by itself.
- **Buffers need a reserve for what is not announced**: a load-shifting planner fills or drains every buffer
  to the limit (sumps at 85-100 % to pump in cheap hours), so an unannounced pump failure floods the level.
  Sumps keep `reserve_h` hours of inflow free (soft `storage` constraint), the ROM pad keeps `reserve_t`
  (soft `working_stock`, priced far below the mill margin so an outage is bridged from it); overflow is
  charged in the KPI (`overflow_cost`, `cum.overflow_m3`) - it used to be free. A saturated level also hides
  the inflow from the estimator (any inflow fits a full sump), so the twin could not see it was under-pumping.
- **Events**: `fail`/`trip` with `hours` restore themselves (repair / restore_units); `assets: [...]`
  applies one event to many assets (a level closed by seismicity). Per-scenario planner budget in
  `scenario.json` -> `planner` (the mine uses `discrete: "ga"` since 6 Nov: without the section search +595,515 against
  +589,677, 3 seeds, and 5-10 s faster per plan), KPI metrics in `kpi` (drives the
  generic Overview), market tags in `market_tags`.
- **3D geometry**: `at = (lat, lon, z)`, `path` with z, `geojson("x.geojson")` keeps elevation. The 3D view
  appears in the console when any asset has a z coordinate.
- **Check slopes against finite differences.** `rt.pwl` had slope 0 at the top of its table: in float32 `(x - x0) / (x1 - x0)` can lie an ulp above 1, where `clamp` has derivative 0. A pump at
  full speed had no slope of its rate, only of its power and wear, and half of a plan's well speeds sit there, so every first-order stage on the oil field worked with a wrong gradient. It showed only because a sequential LP on reverse-mode
  slopes lost money where finite differences gained. Values did not change; the mine's gradients are identical (its only table is the recovery curve, the head grade stays inside it). `test_reverse_mode_agrees_with_finite_differences_at_the_bounds` guards it.
- **Distributions: what is exact, what is not.** One batched simulation at +-1 sigma per uncertainty factor gives every limit's
  spread per asset and hour within a few per cent of Monte Carlo on both twins. It does *not* give the plan value where a total
  over many assets meets a kink (flaring when many wells err together; the mine's grade window): per factor nobody crosses the
  kink, and a quadrature along the near-binding limits' directions did not fix it. The value's spread comes from samples
  (`bench/risk.py`). Deterministic propagation through a nonlinear twin is not a free lunch - say so when asked.
- **Chance constraints belong on consequential limits.** At 5 % on every limit the oil field lost about USD 45 k of its lead in 16 h:
  a never-observed capacity keeps its 15 % prior and the planner left a quarter of the battery liquid capacity unused. A flow
  over a capacity is cut by the protection layer (costs the cut flow); a state beyond its limit is a spill, flood or dry run.
  Default `risk_scope = "states"`.
- **An asset flagged by the anomaly detector** gets a wide covariance prior once (0.5 in log), not a drift added at every
  calibration: that made one well's gas-oil ratio 123 % uncertain within hours and the plan value's 5 % quantile -12 M.
- **Column generation's cost is the master LP, not the DPs.** On 640 units the DPs of all rounds took 0.1 s of 10 s; HiGHS
  solved the growing master from scratch each round (scipy has no warm start; highspy is not installed). Column management
  (keep the schedules in use, near zero reduced cost, the newest, all-off) cut it to 2.2 s at an equal gap.
- **Learned prices do not pay yet** (column generation, 60 units): the prediction saves 1.2 of the 7 rounds a perfect start saves.
  A feature for the excess demand at zero prices (one DP) made it worse (0.4); the start probably enters too weakly (one round of
  columns and the smoothing centre), not the features.
- **Structure first, then the method.** `structure.py` classifies with the discrete choices fixed and blend
  qualities (incoming and stored) frozen - the polish's situation. Pure-LP branches came out already optimal
  from the cascade (the LP only certifies them); the effect is sequential LP over the whole (piecewise) linear
  layer - positive on the mine's A/B, negative on the oil field's (see open work). It finishes the cascade
  but cannot replace it (half the cascade + LP is far worse). The finite-difference Jacobian dominates the
  time (~21 of 26 s on the mine); `scipy.linprog` raises "c must be a 1-D array" when a branch has no
  variables (routing splits are not in `space.blocks`).
- **Restarting the ALM polish from a converged plan breaks feasibility** (F falls even with the cascade's
  multipliers): never use "more polish" as a fair baseline without checking F and the violation.
- **Exact plans need margins.** An LP optimum sits exactly on the binding constraints as the twin sees them;
  every estimation error then becomes a violation in reality. With the LP finishing step the mine flooded a
  sump (reserve computed from an inflow the twin underestimated by 17 %: 92 m3 = 8 h of that error) and ran
  its head grade above the plan window (stope grades underestimated). Fixes: `pump_station.reserve_margin`
  (25 % more inflow in the reserve) and offset-free planning - `concentrator.head_factor` (analysed / computed
  head grade, `calibrate`d online from the on-stream analyser). Check every exact stage for this.
- **Benchmark a planner change over seeds** (`showcase_run --seed`): one 72 h A/B pair moved by +$27k and
  -$44k for the same change; only the mean over seeds says something.
- **Price paths are one random stream per price** (`runtime/scenario.py`). With one shared stream, advanced in
  the order the code first asked for each price and hour, any change in call order changed the realised
  scenario: two runs of different code versions saw different prices (the manual-operations result moved by
  0.5 % with nothing changed in it). Within a run both worlds always saw the same prices. Results produced
  before the fix (twin accuracy, solver budget and comparison, neural ablation, scaling, decomposition) used
  the old paths and are not reproducible bit for bit.
- **The forecaster only sees prices whose tags it knows.** The engine mapped market tags back to price names through
  the oil field's default `MARKET_TAGS`, not the scenario's `market_tags`: the mine's forecaster never received a price
  and planned with flat nominal copper and power prices (announced windows only). Fixed (Nov 2026,
  `test_forecaster_sees_the_scenarios_own_market_tags`); every Corvus result before the fix was planned that way.
  `price_history_days` (scenario) gives the forecaster a week of past prices from a separate random stream, so the first
  plan knows each zone's daily shape (Aquila; the conventional world knows its tariff's peak hours).
- **Tooling trap (this machine)**: the Bash tool collapses doubled backslashes inside heredocs - write
  Python/TS that contains backslashes with the Write/Edit tools, not via `python - <<EOF`. Leftover headless Edge
  instances from the screenshot scripts keep port 9333: a new screenshot then shows the old page (kill the
  `--remote-debugging-port=9333` processes). `TaskStop` on a Windows background job can leave its python child running.
- **Steel: one name, one kind.** Totals by name (KPI, value bridge) add every term of a name, so `cost ironmaking` and
  `emission ironmaking` booked kg of CO2 as dollars (the baseline showed -47 M in 6 h). The program now refuses a term name
  used for two kinds. The planner was never affected (its objective sums by kind).
- **Steel: strings interpolate loop names, not fields.** `tag = "{s.tag}.GRID"` in a table loop gave all 13 grids one tag
  (telemetry cannot be told apart); use `concat(s.tag, ".GRID")`. `plyxis lint` reports `duplicate-tag`.
- **Steel: what an inventory is worth decides what the plan does.** Below variable cost a plan sells the yards instead of
  melting; at full cost or NRV it makes slabs for the yard; variable cost capped at 0.98 x NRV is neutral. Even then a plan
  sells stock down instead of producing (the stock saves the production's carbon and the true cost above book value), so the
  yards hold the S&OP stock target (`reserve_t` = start stock) in both worlds - an A/B lead that comes from emptying yards
  is not a better decision. The floor is smooth (30 USD/h x shortfall^2 / target): as a hinge at the target, where every
  yard starts, any extra shipment paid the full penalty until mill, caster and furnace moved together - the first day's
  lanes stayed near their start (5,514 t/h against 6,250 t/h of contracts, USD -0.6 M after 3 h). Smooth, the plan ships
  6,784 t/h from the first hour and holds the yards at 100 %. The band has a top too (`stock_max` = 110 %, 0.5 USD/t/h above it,
  planning only): above the target extra stock cost nothing and, valued at standard cost, steel made for the yard in a cheap
  night hour (or with more scrap while the furnaces sit at turndown) booked a paper gain - PLYXIS made 15 % more crude steel
  than it shipped. A demand above capacity at full price (spot without a discount) made the
  whole group capacity-limited and every plan a sell-down: markets take contracts at ~80 % of capacity plus a discounted spot.
- **Steel: tight couplings stall first-order search.** With a torpedo-ladle hub of half an hour between blast furnace and
  converter (both with throughput decisions) the furnaces stayed at their start; the converter now blows all the hot
  metal it gets (pig iron beyond its capacity) and the furnace wind is the decision - the plan found the merit order.
- **Steel: a backlog that costs nothing until the horizon end is put off for ever.** Shipping now or tomorrow scored the
  same, the search left the lanes near their start (0), and the first plan shipped 6,063 t/h against 6,250 t/h of
  contracts - every rolling plan defers again, the A/B was USD 0.9 M behind after 8 h. `late_delivery` (2 USD per t
  and hour of backlog, planning-only like `risk`, in `_Kpi.EXPECTATION_TERMS`) and lanes starting at half capacity fix it;
  at 0.5 USD one seed's first plan (its relaxation ended at violation 0.1) still left European coil 20 % short, and the
  plans after it stayed in that basin (USD -2.3 M after 12 h). Aquila also uses the mine's `relax_tol` 0.01.
- **Steel: the baseline plans with the same carbon price.** In steel the internal carbon price (30 USD/t, ~1.9 t/t)
  is about as large as the operating margin, so a planner that prices it and a baseline that does not optimise different
  objectives: PLYXIS sold less spot, lost operating profit and would "win" after carbon. The S&OP's standard costs carry
  each product's standard CO2e (`co2` in the yard registers, `generate.co2_intensity`) at `legacy.carbon_price`.
- **Steel: an hourly stock penalty costs little in a plan's last hours.** Plans ran the yards down in their last day
  (crude steel 4,700 t/h, iron 3,450 t/h) and over-produced at their front to have the stock (7,560 t/h, converters at
  the scrap-rich limit) - the front is all a rolling plan executes, so the plant over-produced for ever. Every yard (product,
  DRI, scrap bay) now has a planning-only `final cost value_to_go = pos(reserve_t - stock) * restock` (200 USD/t). Scrap
  bays had no target at all: PLYXIS melted them down instead of buying (USD 11 M less scrap in 96 h).
- **Score the baseline's decisions in the plan's own objective before blaming the solver.** `bench.policy_in_twin` runs the
  rules on the engineering model (no truth, no events) and the plan on the same; projecting the rules' hourly set points
  onto the decision blocks and evaluating them in `Problem` shows which planning-only term the plan pays for. It found
  that the plan was "better" by USD 30 M in its objective while worse in the accounts: the steel baseline sold its arc
  furnaces' yards down (its S&OP booked them at 97 % while the tariff rule and the grid connection allowed ~85 %) and left
  unprofitable contracts unserved. Fixed in `legacy.py` (`_eaf_capacity`; contracts down to minus their damages) - a
  baseline that profits from emptying yards is as wrong as a planner that does.
- **Steel: lanes draw without `sat`.** A yard drawn below empty would create material in the plant; the emulator logs any
  state underflow (`PlantEmulator.underflow`, `protection` in every showcase result): an A/B is only valid at zero in both worlds.
- **Steel: Europe runs on scrap in the plan - not by choice, the search stalls there.** The plans run the European blast furnaces
  near their 60 % turndown, sell little spot and buy scrap into the bays; the wind's gradient is about zero, which was read as an
  optimum. It is not: a plan worth USD 2.7 M more in the same objective (`bench/results/arms/aquila_plan_better.pt`, found by
  accident by a short relaxation before the ALM fix) ships 3.9 % more, sells 45 % more spot (European coil, slab exports) and runs
  the French, Dutch and Spanish furnaces higher (`plan_quality_aquila.json`). And the operating gap is not the carbon price: with
  it at 0 in both worlds PLYXIS is still -0.46 M on the mean of three seeds (`showcase_run_aquila_nocarbon*.json`), shipping less.
  Why first order stalls: at a yard's stock target the smooth floor has zero slope, so neither a lane alone (the yard falls) nor
  a furnace alone (the yard rises) gains - only the joint move does, and it earns the spot margin. The LP on the algebraic model
  sees that move exactly (the sequential LP from the cascade's plan ships 6,900 t more in its first 15 iterations); its steps
  are cut short by kinks the first-order model gets wrong (spot depth through the backlog's `pos()`, the grid overrun charge).
- **The ALM stored its best iterate one optimiser step late** (`gradient.py`, fixed 5 Nov 2026). It scored `e.x`, which is `x`
  itself when nothing is pinned, and copied it after `opt.step()` had moved it in place: the "best" was always a step past the
  plan it was scored for, and a warm start was never kept (a feasible steel plan worth USD 1,394.7 M came back as 1,387.2 M at
  violation 0.29). `test_alm_returns_the_iterate_it_scored`. Every result before 5 Nov carries it. It also changed the
  coordinator's commit on the toy: the plant's best answer is all or nothing, and at a tracking weight of 1 it went to the extreme
  every time (market 25-32 % over its demand); `CorporateConfig.commit_track` = 16 keeps it near the mix (and a commit that still
  breaks a limit tries the pool's feasible schedules).
- **The console's lag is not what costs the steel group.** At 600x (`bench/console_run.py`) a plan takes 10-12 plant hours and the
  plant ran on 8 plans instead of 31, yet seed 0 ended where the waiting benchmark did (+0.10 M operating, +1.99 M after carbon;
  benchmark +0.18 / +1.95 M).
- **Steel: the end value of a yard shortfall is smooth too.** `value_to_go` (200 USD per t short at the plan's end) as a hinge at the
  target was the same trap as the hinged floor; quadratic up to 10 % of the target. Start plan +3.1 M in the twin's world, A/B within noise.
- **`bench.policy_in_twin` flatters a plan that ends below its targets.** One 96 h plan sold 36,700 t more steel out of the yards than
  the rules by its horizon end (about its whole operating lead at margin) and bought 77,700 t of scrap into the bays; a rolling
  plan never reaches its end. Read the `stocks` it reports before quoting the difference.
- **A first plan has nothing to be compared with** (`cold_starts`): the second opinion compares with the last plan and never fires for
  the first. One steel seed's first plan came out USD 2-6 M below two other random streams' and shipped 700 t/h less European coil
  than its contracts; every later plan started from it (seed -3.0 M; with two cold starts +0.18 M). Steel only so far.
- **Steel runs of the same seed differ** (thread scheduling changes the plans): a replay of seed 1 was USD 0.7 M ahead at hour 42
  where the benchmark run was 0.15 M behind. Only means over seeds, and only differences beyond about USD 0.7 M per seed, say anything.
- **`planner.runs` keeps the last 30 plans** (a deque): in a 96 h steel run the first of 31 plans is missing from `runs[]`; read
  `plans[]` (plan events) for the first plan.

- **Durations in time, not steps.** A step-relative model (`delay(tap, 1) + delay(tap, 2)`, `tundish = 40` steps,
  `hold 4`) means something else at every step, and nobody can tell which step it was written for. Write durations as
  properties with a time unit read through `delay(x, d)` (the compiler divides by `dt`; `delay(taps, cycle - dt)` for a
  window that includes the current step) and holds as `hold 30 [min]`: the detector derives each unit's step from them
  (the largest step dividing the sim's at which every duration is a whole number of steps). A `hold 30 [min]` used to
  parse as `30 [min]` = 0.5 h and silently became one step; `behavior.py` now reads a hold with a time unit as a duration.
- **What decides where a plant ends.** A receiver joins a plant if its stock holds less than `buffer_steps` (24) sim steps
  of the flow through it - measured on the twin at default and random decisions; senders into a plant always join (a
  flow decided outside would be an input it cannot plan); stockless receivers (markets, grids, the gas network without a
  holder) take the flow per sim step and are the fence. Only bounded states count as stocks: a market's contract
  backlog is an account. Duisburg splits at its 144 h slab yard into iron/steel/casting and hot rolling.
- **What makes a regional network.** Exchanges are sinks (take flows from >= 2 units, pass nothing on, no stock); the ones
  several pieces share stay in the master. A caster (two senders, no stock) is not one: it passes slabs on - counting
  it cut every site apart.
- **A plant's program is its input-output function.** Every connection that leaves a plant ends in a priced sink
  (`plx_bsink_*`: revenue = flow x price, tracking term for the commit); the master sees the plant through column slots
  (`plx_col_*`: flows and internal value per hour as exogenous state trajectories, a weight per 8-h block) and a simplex
  (weights add up to 1 every hour). Prices = d(master Lagrangian)/d(probe-slot flows). Every crossing must leave the
  plant: the detector takes every sender into the plant (a declared group must include them: a connection into a
  plant is refused). Crossings are one per port on the other side (fifteen units' power into one grid: one flow).
- **Column generation traps (all hit on the toy):** one start column -> infeasible master -> its multipliers measure the
  infeasibility and the next prices explode (1e10 reduced cost): start with several (zero prices = the plant alone, and
  the group's prices scaled); price from multipliers whenever the master's violation is below 0.05 (a master at 0.002
  counted as infeasible priced the market as open: 98 instead of 70 USD/t); smooth prices half-way; seed a new column at
  weight 0.2 in the master's warm start (at 0 a converged master never moves it); one weight per whole-horizon schedule
  is too rigid - weights per block; a mix of schedules is not a schedule - commit (track the mix, re-plan the group,
  re-price while the committed plan breaks a limit).
- **Per-unit DP traps.** Linearising one unit at a time swings it between all and nothing (no taps: every tap valuable;
  a tap every 45 min: none) - start from the relaxed plan rounded with error carried forward, then trust-region sweeps
  (a cost per changed decision, a sweep kept only if the repaired plan scores better). Three vessels with the same
  relaxed rate round to the same steps: stagger their errors and de-clash identical units where a limit is broken.
- **A heat schedule at part load is the hard case.** Below the group's prices the steel side wants to make little but
  its blast furnaces cannot go under 60 % wind: taps and casts must match around a ladle furnace of four heats. The
  rounded start broke the ladle's limits (0.11) and a value sweep over all units on one linearisation over-reacted
  (casters off, 50 % more taps, violation 1.8). Fixes in `unitdp.py`: a feasibility phase first (units one at a time
  on the gradient of the violations alone, kept only if the total falls, alternated with repairs), trust regions as a
  quantile of the gradient's size (the mean let whole campaigns flip). At the group's prices and above: feasible; at
  0.7x it still ends at 0.005-0.018, so the coordinator keeps such schedules out of the master (`col_tol`) and adds
  start columns at 1.3x / 1.6x when fewer than two keep the block's limits.
- **Phase I for column generation.** Regional networks all buy scrap as if it were plentiful: no mix of their start
  schedules keeps the shared scrap supply (master violation 0.39), and an infeasible master's multipliers are not used
  as prices, so nobody learns the scarcity. If the first master is infeasible, an artificial schedule per block (no flow
  at the fence, worse every hour than any real one) enters it. Only then: always on, it cost the toy USD 2,000.
- **A committed plan is feasible only if every block's schedule keeps its own limits** (`CorporatePlan.plant_violation`,
  `master_feasible`): the first detected-plants run reported "feasible" with the steel side's schedule at 0.0057.
- **Works: two decisions around a small buffer stall the search.** A power-plant load and a flare decided separately
  around a gas holder of eleven minutes: the plan flared up to a third of the process gas with the plant half idle; a
  mill rate against a delayed furnace charge through a 400 t roller table: it rolled half the mill's capacity, and more
  budget made it worse (value -6 M). Now the power plant burns what the furnaces leave (gas follows, no decision) and
  the furnaces decide charge and discharge with their content as the buffer, the mill rolls what they discharge: flaring
  0, pig iron 0-33 t/h instead of 155, feasible (`arms/corporate_plant_declared_plant.json`).
- **Step-by-step recursions are slow.** A state whose update is not affine in itself is scanned in Python, one step at
  a time: Duisburg at 384 steps took 157 ms per evaluation with four such scans. Written as running totals compared
  over a window (`ran - delay(ran, k)`), they vectorise (111 ms). Affine updates `x <- c * x + b` (c, b free of x: the
  mill's `wear <- (wear + rolled * dt) * (1 - change)`) are now an associative scan (`rt.affine_scan`, log2(T) steps):
  the hot-rolling plant went from 49 to 15 ms per evaluation and gradient, values identical (`PLX_NO_AFFINE=1` turns it
  off; the mine twin has 3 such states, the steel group 1 - same objectives at default and random plans, checked).
  `delay(x, k, init)`: `init` must be a constant or a property, `k` may be a property or a duration.

- **A first-order model at a kink drops the rule.** `pos(on(t-1) - on(t-2)) <= on` (a minimum run) linearised at
  "all on" sits exactly on its kink: the slope taken is the flat side, the rule vanished from the model and the
  whole MILP's plan broke it on the twin (kilns: violation 1.0). Piecewise terms (pos, min, max, abs, clamp) whose
  argument reads a discrete decision and only boxed columns are now exact (`Linearizer._max`: a column, an
  indicator, big-Ms from the decisions' full box - the layout does not depend on the reference). Terms of states stay
  first order (a state has no box).
- **Dantzig-Wolfe pricing must own its states.** The hot-strip mill's roll wear is updated from the reheating
  furnaces' flow; with the furnaces in the master that row was priced, the wear column was free in the mill's pricing
  problem and HiGHS reported it unbounded - and the round's bound fell below a plan the whole MILP had found. The
  closure filter moves every atom a block's definition rows read into the block (atom by atom - union-find would
  have swallowed the whole LP cell). A pricing problem that is still unbounded is solved in a wide box and its
  round's bound marked unproven.
- **The integer trust region is a flip budget, not the radius.** A radius below 1 freezes binaries (|u - u_ref| <=
  0.3); `solve(flips=k)` is local branching (sum |z - z_ref| <= k, decisions only, not indicators). Splitting one
  budget across blocks restricts the problem the engine solves - compare bounds only at equal budgets.
- **Scale the rows.** Tonnes, MW and USD 1e9 in one model: HiGHS's MILP failed numerically on the works twin
  (status 4) until every row was scaled to unit largest coefficient (duals are mapped back per raw unit).
- **The exact polish in the cascade** (`slp_iters`, last stage; 5 Nov evening): the whole twin's continuous decisions by
  sequential LP on the algebraic model, discrete ones fixed (`fix_discrete`, also exclusive line-ups and the indicators),
  each LP warm-started from the last HiGHS basis (`warm_lp`: scipy's bundled `scipy.optimize._highspy._core._Highs`, no
  extra package; every row kept so the dimensions are stable - half the LP time). HiGHS options (presolve off, parallel
  dual) do not matter; IPM wrongly reports the steel LP infeasible. On the mine the branch stage stays: restricted to its
  4,032 decisions the polish gained less in more time (the lifted model is far larger than the branch stage's reduced LP);
  on top of the branch stage it lost (+572,540 on 2 seeds against +589,677) and one of its LPs ran for over 15 minutes, so
  every polish and branch LP now has a 10 s limit (`slp_lp_time`; a timed-out LP is a rejected step) and the stage 60 s
  (`slp_time`); a warm start HiGHS ends 'Unknown' (steel, step 23) is retried once from scratch - it used to end the polish.
  On the steel group the trust region settles at 0.0018 and gains about the same every step (15 steps ~ 8 s; the first two
  steps at 0.02 / 0.006 are rejected). `slp_iters = None` (default): on for twins without discrete decisions, off otherwise (oil field neutral,
  mine negative); the steel group sets 15.
- **A shorter cascade plans the steel group better.** With the polish after it, a quarter of the live budget (relax 50,
  repair 20, ALM polish 30) gives better cold plans in 20 s instead of 46 s (`planner_speed.json`): the long relaxation
  drifts into the low-throughput basin. On the oil field a quarter budget loses (cold plan USD 3.81 M against 4.41 M), on the mine a warm one (-0.09 M), and
  the polish is neutral in the A/B (+402,831 against +403,607): there it stays off.
- **Each block prices on its own sub-model** (`Engine._cache`): slicing the whole model per pricing call made a round
  cost blocks x model size (16 kiln sites: 29 s; 21 s on sub-models, 16 s without the block sweeps, which never
  improved a plan and are off by default).

## Open work (as of the last benchmark run)

- **Exact layer / reduction engine (5 Nov 2026, BENCHMARKS.md "The exact layer").** The algebraic model and the sequential
  LP are wired into the planner (the cascade's exact polish, "Planner speed" in BENCHMARKS.md); the reduction engine is
  not. No speed-up to claim for the engine: on the kiln family HiGHS solves the whole MILP faster at most sizes (tight LP
  relaxation); the engine's bound is valid and its plans within 0.04 % from 8 sites on, but its time grows faster than
  linearly (commit 62 s of 110 s at 64 sites). Next, roughly in order: (1) the oil field's on/off wells - the sequential MILP stalls at an
  infeasible point (pump curves times on/off: a flip changes the model more than first order can say); start it
  from the cascade's relaxation instead of the default plan; (2) the master LP is the cost on large twins (scipy has
  no warm start; highspy would give one); (3) pricing in parallel processes; (4) nested contraction where a site's
  blocks no longer fit one MILP (DW inside a block); (5) convex terms (the yards' quadratic stock penalties) as cuts
  instead of trust-region steps; (6) A/B over seeds before anything is switched on.

- **Corvus variants predate the forecaster fix** (the mine planned with flat nominal prices). The three main seeds were re-run
  (4 Nov: +10.3 %, mean USD +583 k, unchanged); the variants (capped grade window, no branches, risk arms, relax_tol, exact, fast)
  were not, so the exact-branch effect on the mine compares runs across the fix.
- **Aquila: why PLYXIS trails (5 Nov, BENCHMARKS.md "Why PLYXIS trails on operating profit").** Not the console's lag, not the
  carbon price (no-carbon arm: the same gap): the plans stall in a low-throughput local optimum - a plan worth USD 2.7 M more ships
  3.9 % more, mostly spot - and in rolling operation the lead PLYXIS builds by hour 40-48 drains in the second half by shipping
  300-700 t/h less than the baseline. The ALM fix alone does not change the A/B (mean -0.96 M, seed 1 a shipping collapse of
  1,900 t/h from hour 38). The exact polish (`slp_iters` 15) is the best arm: operating -0.09 M (-0.4 %), after carbon +1.67 M on
  the mean of three seeds, seed 1's collapse gone - now on for Aquila (`scenario.json`; plans 23-32 s longer); within noise for
  seeds 0 and 2, so more seeds would tell. Next: (0) polish several candidates and keep the best - where the polish ends depends
  on its start (`plan_quality_aquila.json` "polish_starts": +0.16 to +3.55 M, best 1,395.95 M = 3.9 M above the cascade's plan);
  the first plan's two cold starts are the obvious pair; (1) Kestrel and Corvus with the polish (continuous twins only so far: the
  stage skips twins with discrete decisions); (2) the kinks that keep its steps
  small - the markets' backlog `pos()` (needs a box on the backlog state: interval propagation through the state updates, then
  indicators), the overrun charge and the spot discount (exact as epigraphs, `Linearizer(epigraph=True)`, opt-in: the blend's
  copper downgrade then mispredicts); (3) why seed 1's plan of hour 38 holds 1,900 t/h back (replay to hour 38 and read the plan's
  lanes and yards; a same-seed replay did not reproduce it, `arms/replay_aquila_s1_h38.json`); (4) the ALM fix on Kestrel and
  Corvus was re-run (3 seeds each): +403,607 vs +392,840 and +576,462 vs +583,368 - within noise, headline numbers kept; the other
  Kestrel/Corvus benchmarks (variants, neural ablation, scaling, ...) predate it; (5) the corporate benchmarks (`corporate*.json`)
  predate the fix and `commit_track` = 16.
- **Aquila: a small operating-profit lead** (5 Nov night, current code = quarter budget + polish, 6 seeds, 96 h: +1.9 %, USD +0.46 M; after carbon +1.64 M, 6 of 6 seeds
  ahead; CO2e -4.1 %; full budget + polish +0.26 / +1.88 M; before the fix and the polish -1.6 % / +1.39 M). Next, each with a 3-seed A/B: a fuel-rate penalty at low blast-furnace wind (in twin and truth); the 110 %
  stock-band top for scrap bays and DRI piles (bays reach 144 % of target when the market is a few dollars below book); whether
  the European strategy should keep a buffer for outages (risk-aware planning, `risk.py`). The demo texts claim no behaviour that
  was not measured - check any new one against a run. Not checked yet: what the plan does in the ERCOT window and after the
  European scrap jump (the replay tool logged markets and yards only).

- **Agreed plan (3 Nov 2026): planning under uncertainty, then decomposition, then learned prices.** Objective chosen by the founder: maximise expected profit
  subject to a cap on the variance of profit and on the probability of breaking each hard limit. Phases: (1) uncertainty sources (parameter covariance from the
  estimator fit, forecast errors, failure hazards; Monte Carlo only as test reference), (2) first-order propagation with exact correlations + sigma points at kinks,
  profit as a mixture over failures, (3) risk-aware planning replacing hand-set margins, (4) column generation with exact per-asset DP and a master LP (bound via
  Shapley-Folkman), (5) a net that predicts the prices. Nothing switched on without A/B over seeds.
  Status (3 Nov, BENCHMARKS.md "Planning under uncertainty" / "Column generation"): 1-3 built and measured; chance constraints
  off by default (state limits: oil field +USD 7 k within noise, mine -USD 37 k, mostly the stopes' broken-ore lower bound - a
  non-negativity the plant keeps itself, so a cut flow; every limit: two thirds of the oil field's early lead lost). `risk_scope="upper"`
  (overflow only) cut the mine's cost to -USD 11 k (3 seeds), still no gain; next: replace the sump reserve's hand-set margin by
  the propagated one, and test on a scenario whose events actually strain a tank. Open: the wells'
  rate posterior is overconfident (z rms 2.4: pump-curve model error the fit cannot see); covariance matching (on the rates only,
  as it finds) fixes the tails but makes the core far too wide, and exposed that the estimator's 5 % step prior and the
  filter's 1 % drift disagree - it stays off; the value-spread cap needs the founder's number (frontier in
  risk.json). 4-5 built as a model-agnostic core with one asset family and measured against exact MILP; not yet derived from
  a PLX twin (tanks and sumps as blocks with priced flow balances) - the next step for C.
- **The mine's lead and its baseline's head grade.** The founder chose the 1.35-1.45 % window for the showcase (+10.3 %; +4.9 % capped at 1.40 % stays the documented sensitivity). But the baseline lands at 1.37 %, 0.03 below its own 1.40 % target and below the orebody's
  true 1.416 %, and never reads the mill's on-stream analyser (`OSA.HEAD`), while PLYXIS corrects from it (`head_factor`). A control room would see that offset and act on it. If it reached 1.40 %, about USD 206 k of the USD 583 k lead would go (rough arithmetic, not measured).
  Give `legacy.py` analyser feedback and re-run the mine A/B before quoting +10.3 % outside. The twin also has no in-situ depletion, so 48 h cannot show what a richer blend does to later grade.
- **`relax_tol` as the default?** Neutral on the oil field (+389,642 against +392,180, 3 seeds, before the slope fix), better on the mine. Making it the default means re-running the Kestrel benchmarks.
- **Convex stage on the oil field** (`--exact convex --exact-margin 0.10`): +USD 136 k in one plan including the rest of the layer, but the A/B was stopped at hour 24 of 72 with no difference (+162,505 against +163,053). Run it to the end before switching it on.
  `branchlp.tangent_gap` is a measured estimate, not a proof: the analysis proves convexity per asset type, not through the kinks downstream.
- **Cold plans on the oil field after the slope fix**: 3 of 4 seeds better, one -USD 326 k (`arms/cold_plans_old_vs_new.log`). Warm A/B is neutral. More seeds would tell whether the spread really widened; it matters for a console's first plan.
- **Exact branches**: on for the mine (+USD 44 k / 48 h, 3 seeds; the stage-off arm predates the forecaster fix, +47 k at equal code), off for the oil field (before the slope fix: -USD 7 k / 72 h with 10 % margins on state limits,
  3 seeds; -USD 48 k without). Single-plan gains in the twin's objective do not predict this (`bench.exact_branches`) - always decide with A/B over seeds.
- **Section search** has no measurable gain, also not with charge-up crews on the mine or after the polish (`late_sections`; +0.5 % in one mine plan, not A/B tested there; oil field A/B +USD 2.7 k, noise).
  Off on the mine since 6 Nov (`showcase_run_corvus_ga*.json` +595,515 against `_final` +589,677 with it, 3 seeds).
  The scaling claim holds. Crew limits rarely bind in a 48 h horizon; a model with a binding discrete resource would be the test.
- **Certificates.** Smoothed dual decomposition (`sectiondp.solve(dual="smooth")`) proves 1.7-3.7 % up to
  90,000 blocks at 6-9x the time of the subgradient bound (2.5-19 %); the default stays the fast one.
- **Connectors** have only met local stand-ins. Next: a real broker, PI server or OPC UA server; TLS for MQTT, Kerberos for PI, OPC UA subscriptions.
- **Backtest gate G1** counts an unannounced step change against the frozen twin (SWD-2's injectivity, the only failure): decide with the first pilot whether the gate is
  meant frozen or with live estimation.
- **Not re-run after the plan guard**: `neural_ablation`, `solver_comparison`, `scaling`, `twin_accuracy` (Kestrel-only; solver_comparison, scaling and twin_accuracy never pass a warm start, neural_ablation does, and re-running it retrains `subsolver.pt`).
- **Not re-run after the `pwl` slope fix** (Kestrel only; the mine's gradients are identical): the same four, `solver_budget`, `decomposition`, the `core` showcase run and the oil-field comparison arms (exact, `relax_tol`, sections). They stay labelled as measured before it.
- **Plants found in one twin - next** (4 Nov, BENCHMARKS.md "One twin, plants found in it": the detector's two works
  plants committed USD 1,395.6 M feasible against 1,375.8 M at violation 0.022 for the whole twin at 15 min; 363 vs
  1,140 s; the run before, same code but the commit retry: 1,393.3 M). Open, roughly in order: (1) the mill runs at
  525 of 1,350 t/h (the mix asked for 612) - whether a fuller mill is the group's optimum; a commit whose schedule
  misses the limits falls back to a pool schedule that can be far from the mix (397 t/h in one run); (2) part-load heat schedules (0.005-0.018 at 0.7x prices);
  (3) regional networks coordinated by prices do not converge in 5 rounds (shared market depth and scrap supply bind;
  Phase I keeps masters near-feasible but leaning on the artificial schedules; 12 rounds: reduced costs plateau at
  USD 400 M, commit infeasible - more rounds do not help): a stabilised dual, a resource-directive split of the shared
  limits, or blocks solved in one batched evaluation (small blocks cost nearly as much per evaluation as the whole);
  (4) plants inside regional networks (side by side only: `split="both"` compiles, not benchmarked); (5) more plants
  (Velsen, Dunkirk at machine level); (6) truth model and a works-level baseline for an A/B, then a console scenario;
  (7) a Lagrangian bound; (8) more machines (oxygen plant, coke ovens, sinter, cranes, grades and cast sequences).
- **The affine scan changed compiled programs of the mine (3 states) and the steel group (1)**: same objectives and
  violations at default and random plans (checked with `PLX_NO_AFFINE=1`), so the showcase results stand; not re-run.
- **Not built**: SSO/RBAC (decisions carry a free-text name), Dockerfile never built (no Docker on the development machine), mobile view, SAP PM / Maximo
  interface proper (flat file only), write-back, job queue for solves, constraint editor beyond the PLX editor.
