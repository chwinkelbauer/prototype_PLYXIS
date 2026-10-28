# PLYXIS - prototype 2

**An operating system for industrial sites: write the site as code, keep it equal to reality
with live data, and let a layered optimizer run it.**

PLYXIS has four parts, all in this repository:

| | Part | What it does |
|---|---|---|
| A | **PLX language + compiler** (`plyxis/lang`) | Sites are written as code: entities with properties, states, decisions, flows, behaviour, constraints, costs and GIS geometry - nothing is predefined. The compiler turns a model into a batched, differentiable, time-vectorised simulator. |
| B | **Data integration** (`plyxis/data`) | Connectors, tag mapping from the model, moving-horizon estimation, anomaly detection and discovery. The twin recalibrates itself and *rewrites its own code* when new assets appear. |
| C | **Optimizer** (`plyxis/optim`, `plyxis/native`) | Decomposes the twin, uses neural sub-solvers trained on the twin as fast subfunctions, then augmented-Lagrangian optimisation, memetic discrete search, an exact section search (a learned graph filter cuts the site into small sections whose discrete decisions are enumerated exactly; C++/OpenMP kernels) and an exact polish. 40,000 decisions in under a minute on a laptop CPU. |
| D | **Console** (`web/`, `plyxis/server`) | Live operations view for field staff and executives: map, 3D twin, schedule with reasons, optimizer, twin studio (code editor), data integration, benchmarks. Switch between case studies in the top bar. |

**No AI wrapper.** PLYXIS does not call any external AI model or service. Every learned
component (the neural sub-solvers) is trained locally from the twin by the platform itself;
explanations come from gradients and shadow prices, not from a language model. The whole stack
runs offline.

## Quick start (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1     # venv, Python deps, web build, tests
powershell -ExecutionPolicy Bypass -File scripts\start.ps1     # http://127.0.0.1:8000
```

Manual setup on any OS (Python 3.11+, Node 18+):

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # (.venv/bin on Linux/macOS)
cd web && npm install && npx vite build && cd ..
python -m plyxis serve --port 8000
```

Command line:

```bash
python -m plyxis compile scenarios/kestrel/field.plx --show-code   # compile, print stats / generated code
python -m plyxis optimize scenarios/mining_demo/pit.plx            # optimise any PLX model
python -m plyxis run --hours 24                                    # headless A/B showcase run
python -m pytest tests -q                                          # test-suite
```

## Case studies

| | Kestrel Field (oil & gas) | Corvus Mine (underground copper) |
|---|---|---|
| Site | 96 ESP wells, 12 pads, 3 tank batteries, 2 compressor stations, gas plant, trucking, SWD | 186 stopes on 23 levels (520-1,060 m deep), ore passes, trucks, 2 crushers, skip hoist, 3 main fans, 10 pump stations, concentrator |
| Twin | 2D GIS, 123 lines of PLX + asset register | 3D (elevation on every asset, openings from a 3D GeoJSON design export), 135 lines of PLX + register, 392 instances |
| Planning problem | 72 h hourly, 4,983 decisions | 48 h hourly, ~5,000 decisions (744 discrete ring firings) |
| Scenario | 72 h: compressor trips, gas plant upset, ESP wear, injectivity loss, negative gas price, ERCOT price cap, new well | 48 h: main fan trip, seismic level closure, hoist stop, loader breakdowns, critical-peak power, pump failure, new stope |
| Scale | ~18,000 bopd | ~14 Mt/a milled, operating cost ~USD 580 M/a (baseline, measured) |

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
| Kestrel A/B, 72 h vs. competent manual operations | operating profit +13.7 % (+USD 385 k), after carbon +14.3 %, oil +10 %, cash operating cost -9.3 %, flaring -82 %, CO2e -31 %, no unplanned pump failure (baseline: 1); all 32 plans feasible |
| Corvus mine A/B, 48 h vs. competent mine control | operating profit +12.3 % (+USD 647 k) at equal mill throughput and equal cash cost - from head grade inside the mine plan's window (assay-calibrated twin 1.449 % vs. block-model grade control 1.368 %, window 1.35-1.45 %): copper +7.1 %, production cost per t of copper -7.4 %. Baseline operating cost USD 581 M/a |
| Twin accuracy | well deliverability error 11.9 % → 2.4 %; SWD-2 injectivity loss re-estimated within 3 h to within 1 % |
| Solver, 4,983 decisions | cascade + neural sub-solvers: feasible plan in 10.5 s, 5.6 % better than the best metaheuristic (GA) after 60 s |
| Neural sub-solvers | better re-plans in 5 of 8 held-out disruptions (2 ties, median +3.0 %), 20 ms per proposal |
| Deterministic sections | graph filter + exact section enumeration: sections stay at ≤ 3 assets while their number grows 76 → 688 (1 → 8 fields); one exact sweep 0.9 s → 3.2 s; no measurable plan gain yet (oil field -0.0 %, mine -0.4 %, within run-to-run noise) |
| Scale, 768 wells / 39,864 decisions | feasible in 38 s; same 432 lines of generated simulator code as one field; GA infeasible at equal time |

Method, caveats and how the A/B is valued: [docs/BENCHMARKS.md](docs/BENCHMARKS.md) and
[docs/SHOWCASE.md](docs/SHOWCASE.md).

## Documentation

| Document | Content |
|---|---|
| [docs/PILOT_PLAN_OIL_GAS.md](docs/PILOT_PLAN_OIL_GAS.md) | building and implementation plan for the first oil & gas pilots |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | system architecture, code map, design decisions |
| [docs/PLX_LANGUAGE.md](docs/PLX_LANGUAGE.md) | language reference |
| [docs/OPTIMIZER.md](docs/OPTIMIZER.md) | problem formulation, cascade, neural sub-solvers |
| [docs/DATA_INTEGRATION.md](docs/DATA_INTEGRATION.md) | connectors, estimation, anomaly detection, discovery |
| [docs/SHOWCASE.md](docs/SHOWCASE.md) | demo script for investors and customers, legacy-operations assumptions |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | solver comparison, neural ablation, scaling, showcase run |
| [docs/ANALYSIS_PROTO1.md](docs/ANALYSIS_PROTO1.md) | what prototype 1 was and what changed |

## Repository layout

```
plyxis/lang        PLX lexer, parser, library, elaborator, typed IR, compiler, runtime, interpreter
plyxis/lang/stdlib standard packages written in PLX (core, oilgas, mining)
plyxis/optim       decision space, problem, ALM, metaheuristics, partition, neural sub-solvers, cascade,
                   explanations, dispatch, maintenance planning
plyxis/data        connectors, plant emulator, estimation, anomaly detection, discovery, twin repo, forecasts
plyxis/runtime     live engine (two worlds), planner, twin manager, scenario, views
plyxis/server      FastAPI + WebSocket API, serves the console
web/               React console (Vite, TypeScript, Tailwind, MapLibre, ECharts, Monaco)
scenarios/         kestrel (oil field showcase), mining_demo (copper pit), enterprise (generated multi-field)
bench/             benchmarks and their results (bench/results)
tests/             pytest suite
docs/              documentation
```

All data in this repository is synthetic.
