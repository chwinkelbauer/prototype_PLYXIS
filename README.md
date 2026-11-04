# PLYXIS - prototype 2

**An operating system for industrial sites: write the site as code, keep it equal to reality
with live data, and let a layered optimizer run it.**

PLYXIS has four parts, all in this repository:

| | Part | What it does |
|---|---|---|
| A | **PLX language + compiler** (`plyxis/lang`) | Sites are written as code: entities with properties, states, decisions, flows, behaviour, constraints, costs and GIS geometry - nothing is predefined. The compiler turns a model into a batched, differentiable, time-vectorised simulator. |
| B | **Data integration** (`plyxis/data`) | Connectors, tag mapping from the model, moving-horizon estimation, anomaly detection and discovery. The twin recalibrates itself and *rewrites its own code* when new assets appear. |
| C | **Optimizer** (`plyxis/optim`, `plyxis/native`) | Decomposes the twin, uses neural sub-solvers trained on the twin as fast subfunctions, then augmented-Lagrangian optimisation, memetic discrete search, an exact section search (a learned graph filter cuts the site into small sections whose discrete decisions are enumerated exactly; C++/OpenMP kernels) and an exact polish. 40,000 decisions in under a minute on a laptop CPU. |
| D | **Console** (`web/`, `plyxis/server`) | Live operations view for field staff and executives: map, 3D twin, schedule with reasons, shift page (autonomous, advisory and shadow modes, decisions, handover report, audit trail), optimizer with what-if runs, twin studio (code editor, model review), data integration, benchmarks. Switch between case studies in the top bar. |

**No AI wrapper.** PLYXIS does not call any external AI model or service. Every learned
component (the neural sub-solvers) is trained locally from the twin by the platform itself;
explanations come from gradients and shadow prices, not from a language model. The whole stack
runs offline.

## Quick start (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1     # venv, Python deps, web build, tests
powershell -ExecutionPolicy Bypass -File scripts\start.ps1     # http://127.0.0.1:8000
```

The server compiles the twin and computes the first plan before the console opens; the starting screen
shows the solver stage and the elapsed time, and says so if the server is unreachable or the start failed.
The scenario then waits, paused at its first hour, so the guided demo can begin at step 1: press **Start**
to run the plant clock (or start the server with `-Autoplay` / `--autoplay`).

Manual setup on any OS (Python 3.11+, Node 18+):

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # (.venv/bin on Linux/macOS)
cd web && npm install && npx vite build && cd ..
python -m plyxis serve --port 8000
```

Command line:

```bash
python -m plyxis compile scenarios/kestrel/field.plx --show-code   # compile, print stats / generated code
python -m plyxis lint scenarios/corvus/field.plx                   # what compiles but will not do what you think
python -m plyxis backfill history.csv.gz --scenario kestrel        # calibrate a fresh twin from a historian export
python -m plyxis optimize scenarios/mining_demo/pit.plx            # optimise any PLX model
python -m plyxis run --hours 24                                    # headless A/B showcase run
python -m pytest tests -q                                          # test-suite
```

## Case studies

| | Kestrel Field (oil & gas) | Corvus Mine (underground copper) | Aquila Steel (global steel group) |
|---|---|---|---|
| Site | 96 ESP wells, 12 pads, 3 tank batteries, 2 compressor stations, gas plant, trucking, SWD | 186 stopes on 23 levels (520-1,060 m deep), ore passes, trucks, 2 crushers, skip hoist, 3 main fans, 10 pump stations, concentrator | 13 sites on four continents: 15 blast furnaces, 7 converter shops, 10 arc furnaces, 4 DRI plants, 17 casters, 21 rolling and coating lines, 78 yards, 12 markets, 29 sales and transfer lanes, 11 power zones, 5 gas hubs |
| Twin | 2D GIS, 123 lines of PLX + asset register | 3D (elevation on every asset, openings from a 3D GeoJSON design export), 146 lines of PLX + register, 395 instances | world map, 173 lines of PLX + register on a 405-line steel library, 237 instances |
| Planning problem | 72 h hourly, 4,983 decisions | 48 h hourly, ~5,000 decisions (744 discrete ring firings) | 96 h hourly, 2,476 decisions for the whole group |
| Scenario | 72 h: compressor trips, gas plant upset, ESP wear, injectivity loss, negative gas price, ERCOT price cap, new well | 48 h: main fan trip, seismic level closure, hoist stop, loader breakdowns, critical-peak power, pump failure, new stope | 96 h: blast furnace off wind, port closure, ERCOT scarcity pricing, gas curtailment, strip mill fault, customers pulling orders forward, scrap price jump, arc furnace trip |
| Scale | ~18,000 bopd | ~14 Mt/a milled, operating cost ~USD 500 M/a (baseline, measured: 493-504 M) | 73 Mt/a crude steel capacity (nameplate), contracts at ~80 % of it |

## The showcase: Kestrel field

A synthetic 96-well Permian (Delaware Basin) field: 12 pads, 3 tank batteries, 2 compressor
stations with flares, 3 salt-water disposal wells, a gas plant, oil and water trucking, 4 lease
crews and 2 workover rigs. The plant is emulated from a hidden ground-truth model whose
parameters differ from the engineering model by realistic margins; the twin only sees SCADA
telemetry. Two worlds run on the same truth, the same events and the same noise: one operated by
PLYXIS, one by a rule-based model of today's manual operations.

Over 72 hours the scenario throws compressor trips, a gas plant upset that cuts the field's
takeaway by a third for ten hours, a fast-wearing ESP, an unannounced loss of disposal
injectivity, a day of negative gas prices, two hours of power at ERCOT's 5,000 USD/MWh cap, and a
new well that appears only as SCADA tags. The console has a guided demo (bottom right) that walks
through each event.

### Results at a glance (laptop CPU, all synthetic)

| | |
|---|---|
| Kestrel A/B, 72 h vs. competent manual operations | operating profit +14.4 % (mean of 3 seeds, +13.8 to +14.7 %; +USD 393 k per 72 h), after carbon +15.1 %, oil +11 %, cash operating cost -10 %, flaring -84 %, CO2e -30 %, no unplanned pump failure (baseline: 1); 93 of 96 plans feasible; without the two newest events +9.5 % |
| Corvus mine A/B, 48 h vs. competent mine control | operating profit +10.3 % (mean of 3 seeds, +9.8 to +10.8 %; +USD 583 k per 48 h): copper +6.4 %, production cost per t of copper -7.0 %, cash cost about equal (-0.4 %); about four fifths of the copper is head grade, near the top of the plan's window (1.43 %, conventional 1.37 %, limit 1.45 %). With the window capped at 1.40 % the lead is +4.9 %. Baseline operating cost about USD 500 M/a |
| Aquila steel group A/B, 96 h vs. a competent corporate S&OP and site control rooms | no operating-profit lead: -1.6 % (mean of 3 seeds, -4.9 to +0.8 %; USD -0.42 M per 96 h on about USD 24 M). After the group's internal carbon price (the objective both worlds plan with) USD +1.39 M per 96 h, all three seeds ahead, CO2e -6.2 %: the plan runs Europe on scrap instead of hot metal (blast furnaces near turndown, which the model lets them do at no efficiency cost). Merit-order allocation by standard cost leaves little to gain in a market-limited group |
| Backtest, pilot gate G1 | a twin calibrated on a raw historian export and then frozen predicts unseen history: the mine passes; the oil field fails on one signal (disposal pressure 16.9 % vs. limit 10 %, an unannounced loss of injectivity at one well, which the live twin re-estimates within 3 h), all others pass |
| Twin accuracy | well deliverability error 11.9 % → 2.4 %; SWD-2 injectivity loss re-estimated within 3 h to within 1 % |
| Solver, 4,983 decisions | cascade + neural sub-solvers: feasible plan in 10.6 s, 4.9 % better than the best metaheuristic (GA) after 60 s |
| Neural sub-solvers | better re-plans in 7 of 8 held-out disruptions (median +4.6 %), 20 ms per proposal |
| Deterministic sections | graph filter + exact section enumeration: sections stay at ≤ 3 assets while their number grows 76 → 688 (1 → 8 fields); one exact sweep 1.0 s → 3.2 s; no measurable plan gain yet (oil field -0.0 %, mine +0.0 %) |
| Decomposition theory, models with known optimum | section DP within 0.5-1.9 % of the optimum where HiGHS can prove it, about 1 ms per block from 120 to 90,000 blocks; the exact MILP's best after 2-4 min is 7-13 % worse from 1,200 blocks on; every plan proven within 1.7-3.7 % of the optimum by the smoothed dual (about 6 ms per block) |
| Structure recognition | the twins' equations prove 98.6 % of the mine's decisions linear or piecewise linear (given discrete choices and frozen grades), 7.5 % of the oil field's, and the oil field's wells (92.5 %) convex; sequential LP over that layer adds USD 47 k per 48 h to the mine's A/B (3 seeds) and, with 10 % margins on tank limits, lowers the oil field's by USD 7 k per 72 h (3 seeds), so it runs on the mine only |
| Scale, 768 wells / 39,864 decisions | feasible in 38 s; same 432 lines of generated simulator code as one field; GA infeasible at equal time |
| Planning under uncertainty | the spread of every limit per asset and hour from one batched simulation, within a few per cent of Monte Carlo on both twins (median ratio 0.986-0.993); chance constraints on tank and sump limits: oil field +USD 7 k (within noise), mine -USD 37 k (mostly a stock's lower bound that the plant keeps by itself), so they stay off; on every limit they cost two thirds of the oil field's early lead |
| Decomposition into exact asset sub-problems | column generation with an exact DP per asset: 16-30 rounds from 10 to 2,560 assets, gap to the proven bound from 2-4 % down to 0.01 %, 0.2-2.5 s from 160 to 1,280 assets; the exact MILP stops at its time limit from 40 assets. Model-agnostic core, not yet derived from the twins |

Method, caveats and how the A/B is valued: [docs/BENCHMARKS.md](docs/BENCHMARKS.md) and
[docs/SHOWCASE.md](docs/SHOWCASE.md).

## Documentation

| Document | Content |
|---|---|
| [docs/PILOT_PLAN_OIL_GAS.md](docs/PILOT_PLAN_OIL_GAS.md) | building and implementation plan for the first oil & gas pilots |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | system architecture, code map, design decisions |
| [docs/PLX_LANGUAGE.md](docs/PLX_LANGUAGE.md) | language reference |
| [docs/OPTIMIZER.md](docs/OPTIMIZER.md) | problem formulation, cascade, neural sub-solvers |
| [docs/DECOMPOSITION.md](docs/DECOMPOSITION.md) | decomposition theory: structure recognition, exact branches, section DP, column generation, bounds |
| [docs/DATA_INTEGRATION.md](docs/DATA_INTEGRATION.md) | connectors, estimation, anomaly detection, discovery |
| [docs/SHOWCASE.md](docs/SHOWCASE.md) | demo script for investors and customers, legacy-operations assumptions |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | solver comparison, neural ablation, scaling, showcase run |
| [docs/ANALYSIS_PROTO1.md](docs/ANALYSIS_PROTO1.md) | what prototype 1 was and what changed |

## Repository layout

```
plyxis/lang        PLX lexer, parser, library, elaborator, typed IR, compiler, runtime, interpreter
plyxis/lang/stdlib standard packages written in PLX (core, oilgas, mining, steel)
plyxis/optim       decision space, problem, ALM, metaheuristics, partition, neural sub-solvers, cascade,
                   explanations, dispatch, maintenance planning
plyxis/data        connectors, plant emulator, estimation, anomaly detection, discovery, twin repo, forecasts
plyxis/runtime     live engine (two worlds), planner, twin manager, scenario, views
plyxis/server      FastAPI + WebSocket API, serves the console
web/               React console (Vite, TypeScript, Tailwind, MapLibre, ECharts, Monaco)
scenarios/         kestrel (oil field showcase), corvus (underground mine), aquila (global steel group),
                   mining_demo (copper pit), enterprise (generated multi-field)
bench/             benchmarks and their results (bench/results)
tests/             pytest suite
docs/              documentation
```

All data in this repository is synthetic.
