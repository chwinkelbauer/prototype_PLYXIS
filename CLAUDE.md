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
.venv/Scripts/python.exe -m pytest tests -q            # about 100 tests, ~5 min
.venv/Scripts/python.exe -m plyxis serve --port 8000   # console at http://127.0.0.1:8000 (needs web/dist)
cd web && npx vite build                               # rebuild console; npx tsc -b --noEmit to type-check
.venv/Scripts/python.exe -m bench.showcase_run --hours 72      # Kestrel A/B, ~10 min
.venv/Scripts/python.exe -m bench.showcase_run --scenario corvus  # mine A/B (48 h), writes showcase_run_corvus.json
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
.venv/Scripts/python.exe -m plyxis backfill history.csv.gz --scenario kestrel   # calibrate a fresh twin from an export
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
- `plyxis/server` FastAPI + WebSocket (`/api/scenarios`, `POST /api/scenario` switches the case study).
  `web/` React console (Vite, TS, Tailwind 4, MapLibre, ECharts, Monaco, three.js for the 3D twin).
- Case studies: `scenarios/kestrel` (oil field, 72 h), `scenarios/corvus` (underground copper mine, 48 h,
  3D, ~USD 500 M/a OPEX and ~14 Mt/a measured; register/geometry generated by `generate.py`, scenario by `make_scenario.py`).
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
  `scenario.json` -> `planner` (the mine uses `discrete: "ga+sections"`), KPI metrics in `kpi` (drives the
  generic Overview), market tags in `market_tags`.
- **3D geometry**: `at = (lat, lon, z)`, `path` with z, `geojson("x.geojson")` keeps elevation. The 3D view
  appears in the console when any asset has a z coordinate.
- **Check slopes against finite differences.** `rt.pwl` had slope 0 at the top of its table: in float32 `(x - x0) / (x1 - x0)` can lie an ulp above 1, where `clamp` has derivative 0. A pump at
  full speed had no slope of its rate, only of its power and wear, and half of a plan's well speeds sit there, so every first-order stage on the oil field worked with a wrong gradient. It showed only because a sequential LP on reverse-mode
  slopes lost money where finite differences gained. Values did not change; the mine's gradients are identical (its only table is the recovery curve, the head grade stays inside it). `test_reverse_mode_agrees_with_finite_differences_at_the_bounds` guards it.
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
- **Tooling trap (this machine)**: the Bash tool collapses doubled backslashes inside heredocs - write
  Python/TS that contains backslashes with the Write/Edit tools, not via `python - <<EOF`.

## Open work (as of the last benchmark run)

- **Which head-grade window the mine claims.** +10.3 % operating profit at 1.35-1.45 %, about four fifths of the copper gain is head grade at the top of it; +4.9 % with the window capped at 1.40 %.
  The founder should decide which number the showcase leads with.
- **`relax_tol` as the default?** Neutral on the oil field (+389,642 against +392,180, 3 seeds, before the slope fix), better on the mine. Making it the default means re-running the Kestrel benchmarks.
- **Convex stage on the oil field** (`--exact convex --exact-margin 0.10`): +USD 136 k in one plan including the rest of the layer, but the A/B was stopped at hour 24 of 72 with no difference (+162,505 against +163,053). Run it to the end before switching it on.
  `branchlp.tangent_gap` is a measured estimate, not a proof: the analysis proves convexity per asset type, not through the kinks downstream.
- **Cold plans on the oil field after the slope fix**: 3 of 4 seeds better, one -USD 326 k (`arms/cold_plans_old_vs_new.log`). Warm A/B is neutral. More seeds would tell whether the spread really widened; it matters for a console's first plan.
- **Exact branches**: on for the mine (+USD 47 k / 48 h, 3 seeds), off for the oil field (before the slope fix: -USD 7 k / 72 h with 10 % margins on state limits,
  3 seeds; -USD 48 k without). Single-plan gains in the twin's objective do not predict this (`bench.exact_branches`) - always decide with A/B over seeds.
- **Section search** has no measurable gain, also not with charge-up crews on the mine or after the polish (`late_sections`; +0.5 % in one mine plan, not A/B tested there; oil field A/B +USD 2.7 k, noise).
  The scaling claim holds. Crew limits rarely bind in a 48 h horizon; a model with a binding discrete resource would be the test.
- **Certificates.** Smoothed dual decomposition (`sectiondp.solve(dual="smooth")`) proves 1.7-3.7 % up to
  90,000 blocks at 6-9x the time of the subgradient bound (2.5-19 %); the default stays the fast one.
- **Connectors** have only met local stand-ins. Next: a real broker, PI server or OPC UA server; TLS for MQTT, Kerberos for PI, OPC UA subscriptions.
- **Backtest gate G1** counts an unannounced step change against the frozen twin (SWD-2's injectivity, the only failure): decide with the first pilot whether the gate is
  meant frozen or with live estimation.
- **Not re-run after the plan guard**: `neural_ablation`, `solver_comparison`, `scaling`, `twin_accuracy` (Kestrel-only; solver_comparison, scaling and twin_accuracy never pass a warm start, neural_ablation does, and re-running it retrains `subsolver.pt`).
- **Not re-run after the `pwl` slope fix** (Kestrel only; the mine's gradients are identical): the same four, `solver_budget`, `decomposition`, the `core` showcase run and the oil-field comparison arms (exact, `relax_tol`, sections). They stay labelled as measured before it.
- **Not built**: SSO/RBAC (decisions carry a free-text name), Dockerfile never built (no Docker on the development machine), mobile view, SAP PM / Maximo
  interface proper (flat file only), write-back, job queue for solves, constraint editor beyond the PLX editor.
