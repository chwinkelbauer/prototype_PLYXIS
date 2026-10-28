# Analysis of prototype 1 (`proto1_ag`) and what changed

## What prototype 1 was

A front-end demonstrator (Next.js, Tailwind, Monaco) built to the brief in `prompt.txt` and
`prototype_minfunction.docx`:

| Part | Implementation in proto1 |
|---|---|
| DSL | `str3`/`test`: a VHDL-like draft (`define x as ent/pkg/env/sim`, sections `properties`, `nodes`, `behavior`, `contains`, `geom`), parsed in the browser by `dslParser.ts` driven by a keyword map (`dsl-definition-map.json`) into a JSON "entity registry" |
| Digital twin | 2D canvas with drag-and-drop trucks/excavators and flow lines; mock data (`mockData.ts`) |
| Optimizer | `simulation/engine.py` and the console: a timed sequence printing the four stages graviSearch → PSO → GA → DEoptim with randomly generated "costs"; nothing was optimised |
| Analytics | hard-coded KPIs (OPEX −11 %, availability +15 %, utilisation +30.8 %) and a manual vs. optimised Gantt toggle |

It communicated the vision well (business deck `PLYXIS_ir_acc.pdf`: DSL → twin → layered
optimizer → ERP/operations, cloud-first, gain-share pricing) but none of the core claims were
backed by working technology.

## Findings

1. **The DSL draft had the right ideas** - entities as blueprints, packages, environments that
   instantiate objects, nodes as typed in/out flows, behaviour as code, GIS geometry, a
   simulation block with an objective. These concepts are kept in PLX.
2. **It had no semantics.** Expressions like `.in$.$amount <- inA$.$amount - outX$.$amount + ...`
   mixed R syntax, implicit iteration and undeclared names; there was no type system, no units,
   no notion of time step, no distinction between decisions, states and parameters, and no way to
   say what is a constraint vs. an assignment. A model could not be simulated, let alone optimised.
3. **The keyword-map parser** (change the map, not the code) cannot carry real semantics. A
   language needs a grammar, scoping, types and a compiler; extensibility belongs in libraries
   written *in* the language (packages), not in the parser.
4. **The 4-stage metaheuristic pipeline does not scale.** Black-box population methods need a
   number of evaluations that grows with dimension and rarely satisfy many coupled capacity
   constraints. Measured on the Kestrel twin (4,983 decisions): PSO and gravitational search end
   infeasible after 60 s, GA needs the full 60 s to reach the quality the PLYXIS cascade reaches
   in 18 s, and the cascade with neural sub-solvers is 10 % better still
   ([BENCHMARKS.md](BENCHMARKS.md)).
5. **"Differential equation solver for candidate evaluation" (DEoptim)** conflated two things:
   DEoptim is *differential evolution* (another metaheuristic), not an ODE solver. The real need -
   evaluating a candidate schedule against physics - is met by simulating the compiled twin.
6. **Mock data everywhere.** Investors and customers increasingly ask for a live system; a demo
   that cannot be poked (inject a failure, change a model line, ask "why") does not survive
   technical due diligence.

## What prototype 2 keeps, replaces and adds

| Topic | proto1 | proto2 |
|---|---|---|
| Language | draft syntax, browser keyword parser | PLX: full grammar, units, types, decisions/states/parameters, constraints, objective terms, packages, inheritance, `for`/`if` generation, tables, GIS, discovery rules; compiler to a batched differentiable simulator; reference interpreter; diagnostics with hints ([PLX_LANGUAGE.md](PLX_LANGUAGE.md)) |
| Entities | predefined trucks/excavators | none predefined: standard libraries (`oilgas`, `mining`) are ordinary PLX code; any asset is a few lines |
| Twin | canvas mock | compiled netlist with real geometry (MapLibre), live state, versioned source |
| Data | none | connectors, tag mapping from the model, estimation (MHE), anomaly detection, discovery that rewrites the model |
| Optimizer | scripted animation | cascade: decomposition, neural sub-solvers trained on the twin, augmented Lagrangian, memetic discrete search, polish, shadow prices; the four metaheuristics are implemented, vectorised and benchmarked |
| Explanations | canned "agent's reason" strings | derived from gradients and shadow prices of the actual plan |
| KPIs | hard-coded percentages | realised accounting of a closed-loop A/B run against a rule-based model of today's operations |
| AI dependency | n/a | none: no external AI model or service; all learning happens locally on the twin |

The deck's value assumptions should be re-framed with what the showcase measures against a
competent baseline over 72 hours on the synthetic field: +13.7 % operating profit (+14.3 % after carbon
cost), -9.3 % cash operating cost, +10 % oil, -82 % flaring, -31 % emissions and no unplanned pump failure.
The scenario is deliberately dense in events (two compressor trips, a gas plant upset, an injectivity loss,
a negative-price day and a power price cap in three days), so these figures must not be annualised
linearly; the cash-cost effect is consistent with the deck's "OPEX -7 to -13 %", but a headline for
customers needs real pilot data - see [PILOT_PLAN_OIL_GAS.md](PILOT_PLAN_OIL_GAS.md).
