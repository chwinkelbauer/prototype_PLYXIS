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
.venv/Scripts/python.exe -m pytest tests -q            # 45+ tests, ~1 min
.venv/Scripts/python.exe -m plyxis serve --port 8000   # console at http://127.0.0.1:8000 (needs web/dist)
cd web && npx vite build                               # rebuild console; npx tsc -b --noEmit to type-check
.venv/Scripts/python.exe -m bench.showcase_run --hours 72      # A/B run, ~7 min
.venv/Scripts/python.exe -m bench.twin_accuracy --hours 30 --full
.venv/Scripts/python.exe -m bench.solver_comparison --budget 60
.venv/Scripts/python.exe -m bench.neural_ablation --experts 60 --test 8   # retrains scenarios/kestrel/models/subsolver.pt
.venv/Scripts/python.exe -m bench.scaling --fields 1 2 4 8 --ga
.venv/Scripts/python.exe -m bench.solver_budget --cases 5   # live solver budget vs previous / large
.venv/Scripts/python.exe -m bench.report                # markdown tables from bench/results
```

Long benchmarks are CPU-bound and timing-sensitive (wall-clock budgets): run them one at a time, and
not while the console server or a web build is running. After a model change run `neural_ablation`
first (it retrains `subsolver.pt`), then the showcase and the rest.

## Layout

- `plyxis/lang` PLX language: parser, elaborator, typed IR, compiler to a time-vectorised differentiable
  torch simulator (planning and physical semantics), reference interpreter. `stdlib/*.plx` are libraries.
- `plyxis/optim` decision space, problem, ALM, metaheuristics, partition, neural sub-solvers, cascade,
  explanations, dispatch, maintenance.
- `plyxis/data` connectors, plant emulator (hidden truth + protection layer + KPI accounting),
  estimation (MHE with L-BFGS, prediction step, adaptive prior release), anomaly, discovery, repo.
- `plyxis/runtime` engine (two worlds, plant-clock thread, planner thread), planner, twin manager.
- `plyxis/server` FastAPI + WebSocket. `web/` React console (Vite, TS, Tailwind 4, MapLibre, ECharts,
  Monaco). `scenarios/kestrel` showcase field. `bench/` evidence. `docs/` documentation.

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
- **Planner valuations must match the KPI accounting**: e.g. pump wear is priced at
  `pump_cost / (1 - fail_health)` per unit health because the KPI values remaining life from health 1
  down to `fail_health`. Check new cost terms against `_Kpi` before trusting a result.
- **Neural sub-solver features** = all parameters + states of each entity type, padded to the largest
  type. Adding a property to a blueprint shifts the layout: retrain (`bench.neural_ablation`).
  Loading skips mismatched tensors silently (the net then only adds a poor start).
- **Value bridge** (`Engine._bridge`): the operating-profit difference by realised term; sums exactly
  to `delta.profit`. Use it to explain every result before writing it into docs.
