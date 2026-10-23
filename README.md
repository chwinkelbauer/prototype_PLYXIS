# PLYXIS - prototype 2

**An operating system for industrial sites: write the site as code, keep it equal to reality
with live data, and let a layered optimizer run it.**

PLYXIS has four parts, all in this repository:

| | Part | What it does |
|---|---|---|
| A | **PLX language + compiler** (`plyxis/lang`) | Sites are written as code: entities with properties, states, decisions, flows, behaviour, constraints, costs and GIS geometry - nothing is predefined. The compiler turns a model into a batched, differentiable, time-vectorised simulator. |
| B | **Data integration** (`plyxis/data`) | Connectors, tag mapping from the model, moving-horizon estimation, anomaly detection and discovery. The twin recalibrates itself and *rewrites its own code* when new assets appear. |
| C | **Optimizer** (`plyxis/optim`) | Decomposes the twin, uses neural sub-solvers trained on the twin as fast subfunctions, then augmented-Lagrangian optimisation, memetic discrete search and an exact polish. 40,000 decisions in under a minute on a laptop CPU. |
| D | **Console** (`web/`, `plyxis/server`) | Live operations view for field staff and executives: map, schedule with reasons, optimizer, twin studio (code editor), data integration, benchmarks. |

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

## The showcase: Kestrel field

A synthetic 96-well Permian (Delaware Basin) field: 12 pads, 3 tank batteries, 2 compressor
stations with flares, 3 salt-water disposal wells, a gas plant, oil and water trucking, 4 lease
crews and 2 workover rigs. The plant is emulated from a hidden ground-truth model whose
parameters differ from the engineering model by realistic margins; the twin only sees SCADA
telemetry. Two worlds run on the same truth, the same events and the same noise: one operated by
PLYXIS, one by a rule-based model of today's manual operations.

Over 72 hours the scenario throws compressor trips, a fast-wearing ESP, an unannounced loss of
disposal injectivity, a negative gas price, and a new well that appears only as SCADA tags.
The console has a guided demo (bottom right) that walks through each event.

### Results at a glance (laptop CPU, all synthetic)

| | |
|---|---|
| A/B over 72 h vs. competent manual operations | profit after carbon +2.4 % (operating profit +2.0 %), flaring -93 %, CO2e -35 %, no unplanned pump failure (baseline: 1) |
| Twin accuracy | well deliverability error 11.9 % → 2.4 %; SWD-2 injectivity loss re-estimated within 3 h to within 1 % |
| Solver, 4,983 decisions | cascade + neural sub-solvers: feasible plan in 14 s, 8.7 % better than the best metaheuristic (GA) after 60 s |
| Neural sub-solvers | better re-plans in 7 of 8 held-out disruptions (median +2.5 %), 20 ms per proposal |
| Scale, 768 wells / 39,864 decisions | feasible in 55 s; same 429 lines of generated simulator code as one field; GA infeasible at equal time |

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
