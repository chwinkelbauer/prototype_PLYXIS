# PLYXIS platform architecture

PLYXIS turns an industrial site into **code** (a PLX model), compiles that code into a
**differentiable, batched simulator**, keeps it equal to reality with **live data**, and
solves the resulting planning problem with a **layered optimizer that uses neural
sub-solvers trained on the twin itself**. No external AI service and no pretrained model
is involved anywhere: every learned component is trained locally, from the twin, by the
platform.

```
                 engineers                       field / enterprise systems
                    |                                       |
           PLX source (git-like repo)          SCADA / historian / market feeds
                    |                                       |
   +----------------v-----------------+     +---------------v----------------+
   |  A  PLX language + compiler      |     |  B  Data integration           |
   |  parse -> elaborate (netlist,    |<----|  connectors -> TagMap          |
   |  GIS) -> typed IR -> schedule -> |code |  MHE estimator (calibration)   |
   |  generated torch simulator       |     |  anomaly detection, discovery  |
   +----------------+-----------------+     |  -> writes live.plx /          |
                    | Program               |     calibration.plx            |
                    v (differentiable)      +---------------^----------------+
   +----------------------------------+                     | telemetry
   |  C  Optimizer cascade            |                     |
   |  decompose -> neural sub-solvers |      setpoints  +---+---------------+
   |  -> augmented Lagrangian ->      |---------------->| plant (emulator)  |
   |  memetic discrete search ->      |   instructions  | or real operators |
   |  polish; shadow prices           |                 +-------------------+
   +----------------+-----------------+
                    | plan, explanations
                    v
   +----------------------------------+
   |  D  Console (React) via FastAPI  |
   |  overview, map, schedule, why,   |
   |  optimizer, twin studio, data    |
   +----------------------------------+
```

## Code map

| Package | Purpose | Key modules |
|---|---|---|
| `plyxis/lang` | PLX language, compiler, runtime library | `lexer`, `parser` (recursive descent + Pratt, error recovery), `library` (packages, inheritance), `elaborate` (instances, `for`/`if` generation, tables, GIS, flat netlist), `behavior` (typed IR, units), `compiler` (signal graph, SCC scheduling, code generation), `program` (compiled simulator API), `interp` (reference interpreter), `units`, `printer` |
| `plyxis/lang/stdlib` | Standard packages written in PLX | `core.plx`, `oilgas.plx` (wells, batteries, compression, flares, SWD, trucking, sales, rigs, crews), `mining.plx` |
| `plyxis/optim` | Optimization | `space` (decision vector), `problem` (objective, constraints, scaling), `gradient` (augmented Lagrangian, shadow prices), `partition` (subsystems), `neural` (graph sub-solvers), `metaheur` (PSO, gravitational search, GA, DE, cooperative variants), `cascade` (the layered solver), `explain`, `dispatch`, `maintenance`, `plan` |
| `plyxis/data` | Data integration | `connectors`, `emulator` (synthetic "real" plant), `estimation` (TagMap, LiveTwin, MHE), `anomaly`, `discovery`, `repo` (versioned twin source), `forecast` |
| `plyxis/runtime` | Live operation | `engine` (two worlds, plant clock, events, KPIs), `planner` (rolling horizon, warm starts, online learning), `twin_manager` (code generation, hot swap), `scenario`, `views` |
| `plyxis/server` | API | FastAPI REST + WebSocket push; serves the built console |
| `web/` | Console | Vite, React 18, TypeScript, Tailwind, MapLibre, ECharts, Monaco (PLX syntax) |
| `scenarios/` | Twins | `kestrel/` (96-well Permian field, showcase), `mining_demo/` (copper pit), `enterprise/` (generated multi-field company) |
| `bench/` | Evidence | neural ablation, solver comparison, scaling, full showcase run |

## A. From PLX source to a simulator

1. **Parse** (`parser.py`). Hand-written recursive descent with a Pratt expression parser.
   Errors carry file/line/column, a hint and "did you mean" suggestions; the parser recovers
   at statement boundaries so a file reports all its errors at once. Legacy syntax from the
   draft DSL (`str3`) is recognised and reported with a migration hint.
2. **Library** (`library.py`). Packages, `use`, inheritance (`define esp_well as well`),
   environment inheritance (`define kestrel_live as kestrel`: contents, constraints and
   discovery rules are inherited).
3. **Elaborate** (`elaborate.py`). Executes the `contains:` sections: `for`/`if` generation,
   `table("wells.csv")`, name templates (`P{pad}_W{well}`), composite entities (flattened with
   junction nodes), `connect(a, b) via flowline {...}` (links become instances with their own
   behavior, e.g. pressure-limited flowlines), wildcard ports specialised by what flows through
   them, GIS coordinates/lines (lat/lon, great-circle lengths). The result is a **flat netlist**
   of instances, ports and edges.
4. **Behavior IR** (`behavior.py`). Every entity type's statements become a typed IR: locals,
   port flows, next-state updates (`s <- e`), constraints (`a <= b`, `soft(w)`), and objective
   terms (`cost`, `revenue`, `emission`, `metric`, `final`). Units are converted to canonical
   units at compile time, dimensions are checked (warnings, not errors).
5. **Schedule & generate** (`compiler.py`). Instances of the same type at the same graph level
   form a *group* (one tensor row per instance). The global signal graph is analysed with
   Tarjan's SCC algorithm:
   - acyclic signals are computed **for all time steps at once** (vectorised over time),
   - pure accumulators (`s <- s + f(...)`) become a `cumsum`,
   - only true feedback loops through state run as a sequential scan.

   The output is plain Python/PyTorch source (`Program.source`), batched over scenarios `B`,
   vectorised over time `T`, differentiable end-to-end.
6. **Two semantics from one model.** The *planning* program treats state limits as
   constraints the optimizer must respect; the *physical* program (used by the emulator and the
   estimator) saturates them like reality (a full tank stops accepting fluid). Both share the
   same grouping so tensors are interchangeable.

The compiled simulator is checked against an independent reference interpreter (`interp.py`)
to machine precision (`tests/test_compiler.py`).

## B. Keeping the twin equal to reality

* **Connectors** turn sources into `Telemetry`/`Birth` messages (emulated SCADA, market feeds,
  HTTP push, CSV historian replay). A pilot swaps in OPC UA / MQTT Sparkplug B / PI Web API
  clients behind the same interface.
* **TagMap** is derived from the model's `telemetry:` sections - the model declares what is
  measured, nothing is hard-coded.
* **Estimator**: direct assimilation (levels, run status, unit availability), well-test
  updates, and moving-horizon estimation that replays the twin over the last window with the
  controls actually applied and fits calibratable properties and unmeasured states
  (e.g. pump health) by gradient descent through the physical program.
* **Anomaly detection**: innovations vs. the twin, thresholds and trends declared in PLX,
  status changes.
* **Discovery**: unknown tags matching a `discover:` rule produce a PLX code proposal
  (with coordinates from the device's birth certificate). Applied proposals are committed to
  `live.plx`, calibrations to `calibration.plx`, both versioned in the twin repository.
  Recompilation transfers the live state by asset path.

Details: [DATA_INTEGRATION.md](DATA_INTEGRATION.md).

## C. Planning

The planner rolls a 72 h horizon at hourly resolution. Every re-plan (scheduled every 3 h, or
immediately after a disruption, a twin rewrite or a market notice) runs the cascade on the
current twin, warm-started from the previous plan shifted in time (and remapped by asset path
when the twin's structure changed). Details: [OPTIMIZER.md](OPTIMIZER.md).

## D. Runtime and console

* `Engine` runs **two worlds** on the same hidden ground truth, the same events and the same
  noise: one operated by PLYXIS, one by a rule-based model of today's manual operations
  (`scenarios/kestrel/legacy.py`). KPIs are realised accounting (sales minus cash costs plus
  change in inventory value), not model predictions.
* Threads: a plant-clock thread advances both plants; the planner solves in its own thread;
  the asyncio server only serves requests and pushes snapshots. During urgent re-plans the plant
  clock is held, so compressed demo time does not penalise the planner for wall-clock solve time
  (the UI shows a banner while the clock is held).
* The console is a static build served by the API process (`web/dist`). Data arrives via
  REST plus a WebSocket snapshot stream.

## Design decisions

| Decision | Why |
|---|---|
| The twin is code, compiled to a differentiable simulator | Gradients through the twin give fast solvers, self-supervised training of sub-solvers, calibration (MHE) and explanations (shadow prices) from one artefact |
| Time-vectorised code generation (SCC analysis) | A 72-step, 4,983-variable field simulates in ~10 ms on a laptop CPU (16 candidate plans in ~16 ms), so about a thousand candidate plans per second |
| Groups by (type, level) | Adding wells adds rows, not code: 1 to 8 fields compile to the same 17 groups and 424 generated lines |
| Neural nets only as sub-solvers inside a verified cascade | Learned policies are fast but not trusted blindly: every plan is checked by exact simulation and repaired/polished by optimisation |
| No external AI | Deterministic, auditable, deployable air-gapped at OT sites; no data leaves the customer |
| Physical vs. planning semantics | The same model plans (constraints) and emulates/estimates (saturation) |
| Twin edits are commits | Auditable management of change for every structural or calibration update |
