# The PLYXIS optimizer

## Problem

Any compiled PLX twin defines one optimisation problem:

```
maximise   J(u) = sum_t [ revenue - cost - carbon_price * emission ](u, t) + final terms
subject to h_c(u, t) <= 0                 hard constraints (capacities, pressures, tank limits ...)
           soft constraints               penalised with their declared price, e.g. soft(500 [USD/Mscf])
           u in decision space            continuous, integer, boolean, routing, hold blocks
```

* **Decision space** (`space.py`). Every `decision:` of every instance becomes a block of
  normalised variables `u in [0, 1]` with one value per *hold block* (`hold 6` = the decision
  can change every 6 steps). `int`/`bool` decisions are relaxed during the continuous stages and
  made integral later. Every edge leaving a *routed* out-port becomes a routing variable;
  `exclusive` ports (one destination at a time, e.g. a tank battery line-up) are relaxed
  splits that are annealed to one-hot. One representation is shared by all solvers.
* **Scaling** (`problem.py`). Constraint residuals are signed and scaled by magnitude probes
  of both sides (from the generated code), so a pressure in kPa and a gas rate in m3/h have
  comparable violations. The objective is scaled by its magnitude at the start point.
* **Evaluation**. One call simulates a whole population `[B, N]` over the horizon in one
  batched, time-vectorised forward pass (≈10 ms for the 72 h Kestrel field, 4,983 variables, on a laptop CPU).
  Gradients cost a small multiple of one simulation regardless of `N` (reverse-mode autodiff).

The problem is model-agnostic: the same code optimises the oil field, the copper pit
(`scenarios/mining_demo`) and the generated multi-field enterprise.

## Why not a plain metaheuristic

The original concept (graviSearch → PSO → GA → DEoptim) evaluates candidates with black-box
simulations. On a 5,000-dimensional, heavily constrained problem that fails: population
methods need a number of evaluations that grows with the dimension and almost never land in the
feasible region of coupled capacity constraints. `bench/solver_comparison.py` measures exactly
this at equal wall time (results in [BENCHMARKS.md](BENCHMARKS.md)). The metaheuristics are kept
- vectorised, and in a cooperative (separable) form - as one ingredient of the discrete stage,
not as the main engine.

## The cascade (`cascade.py`)

| Stage | What | Cost |
|---|---|---|
| 0 Compile & decompose | decision space, subsystem partition (`partition.py`), scales | ms |
| 1 Neural sub-solvers | learned best response of every subsystem to the current context and constraint prices; dual coordination of the prices | ~25 ms per forward |
| 2 Coordinated relaxation | multi-start augmented Lagrangian (PHR) with projected Adam on the full twin; integer/boolean/exclusive decisions annealed towards integrality | seconds |
| 3 Discrete search | memetic GA: randomised roundings of the relaxed solution, cooperative recombination per subsystem (separable fitness), batched gradient repair | seconds |
| 4 Polish & verify | continuous refinement with discrete decisions fixed, exact (non-relaxed) simulation, shadow prices | seconds |

Starts for stage 2 are the previous plan shifted in time (rolling horizon), the neural plan
and the default plan, plus perturbed copies of them (uniform noise of 12-22 % of each variable's
range) up to 8 starts. All starts are optimised as one batch, so extra starts cost little wall
time, and on a non-convex problem they are the cheapest lever there is. When the model has no
discrete decisions, stage 3 is skipped and its budget goes to the polish.

**Live configuration** (`runtime/planner.py`): relaxation 200 iterations, 8 starts, discrete search
12 candidates × 1 generation with 80 repair iterations, polish 120 iterations: about 10 s for the
4,983 decisions of the Kestrel twin on a laptop CPU. The budget was set by an ablation on sampled
event situations (`bench/solver_budget.py`): the live configuration is worth +2.6 % per plan over the
previous one (10.2 s instead of 7.1 s); a 3.4x larger budget adds only another 0.3 % - the live budget sits
at the knee of the curve (results in [BENCHMARKS.md](BENCHMARKS.md)). The mine uses its own budget
(`scenario.json` -> `planner`: 120 relaxation iterations, 4 starts, GA + exact sections, ~45-55 s for
~5,000 decisions over 48 h); a first response is live within ~10 s.

### Augmented Lagrangian (`gradient.py`)

Powell-Hestenes-Rockafellar form on signed, scaled residuals, minimised with projected Adam
(cosine learning-rate schedule) over a whole batch of starts at once. Every `outer` iterations
the multipliers are updated (`lambda <- relu(lambda + rho * h)`), and the penalty weight `rho`
triples only when the worst violation has not at least halved. All residuals of a batch are
flattened into one tensor with an equality mask, so penalty, violation and multiplier update are
a handful of vectorised operations per iteration, and per-constraint violations are only
materialised when someone asks for them (reports, explanations). The multipliers at the optimum are
**shadow prices**: the marginal value of one more unit of each capacity, per hour. They drive
the bottleneck view and the explanations.

### Neural sub-solvers (`neural.py`)

A typed spatio-temporal graph network over the netlist: every instance is a node, every flow
edge is a graph edge, every node carries a time axis (one slot per 4 planning steps).

* Inputs: local parameters and states (per-type encoders), type embedding, exogenous inputs
  over time (prices, hour of day), constraint prices `lambda`.
* 3 message-passing rounds along the flow edges (upstream and downstream messages,
  LayerNorm, GELU MLP update) plus a depthwise temporal convolution per round.
* Outputs: one head per (entity type, decision), plus a routing head on edge pairs. Heads are
  **residuals around a reference plan** (zero-initialised), so an untrained network reproduces
  the reference and training can only move it where it helps.

Training uses the differentiable twin, never labelled data and never an external model:

1. *Self-supervised*: maximise `J/J_scale - sum lambda h - mu * relu(h)^2` over randomised
   operating contexts (drifted parameters, failures, tank levels, price scenarios).
2. *Expert iteration / imitation*: the cascade solves realistic event situations sampled from the
   scenario's event types (trips, injectivity loss, price shocks, new wells), and the network
   is trained to imitate those plans. In live operation the planner keeps learning from its own
   solutions (`Planner.learn`).

At solve time the network gives an instant plan after any event and acts inside a
dual-decomposition loop (`lambda <- relu(lambda + eta * h(x(lambda)))`, one forward plus one
simulation per iteration). Its output is always handed to stages 2-4: a learned policy is a
starting point, never the final word.

**Measured effect** (8 held-out event situations, equal budgets,
`bench/results/neural_ablation.json`, retrained on the current model): the network proposes a plan for
the whole field in about 20 ms. That plan is not feasible as it stands (0/8) - it is a starting point.
After the same 1.4 s gradient repair it is feasible in 8 of 8 cases, against 5 of 8 when the previous plan
is repaired (objective within 0.2 %). Inside the full cascade the neural start wins 5 of 8 cases, ties 2
and loses 1 (median +3.0 %, mean +3.8 %, range -3.2 % to +15.2 %) at the same solve time. Earlier
training runs on previous versions of the model showed the same direction with a different spread.

### Deterministic sections: learned graph filter + exact section search (`sections.py`, `plyxis/native`)

Discrete operating decisions (which stopes to fire this shift, which wells run, how many compressor units,
which battery a pad is lined up to) make planning a scheduling problem: exact solution time grows
exponentially with the number of discrete decisions. Most of them, however, interact only with a few
neighbours in the twin's graph. The section stage exploits that:

1. **Graph filter.** A polynomial graph-filter network - two layers of `sum_k A^k X W_k` over the twin's
   normalised instance graph (K = 3 hops), i.e. learned spectral filters - embeds every asset and predicts
   the coupling strength of two assets' discrete decisions. Labels come from the twin itself: exact
   finite-difference cross terms of the penalised objective, `F(x+a+b) - F(x+a) - F(x+b) + F(x)`. The
   network is trained on the twin at the first solve (about 2 s, no external data, no external model);
   afterwards one forward pass (~2 ms) replaces hundreds of probing simulations.
2. **Sections.** Union-find clustering merges assets along the strongest predicted couplings as long as a
   section's joint discrete options per time window stay within a budget (64). Weak couplings are cut and
   coordinated by prices.
3. **Exact section search.** For each time window, all sections vary at once in ONE batched simulation
   (batch element 0 keeps the current plan): each section's joint options are enumerated exactly. A
   candidate's value is the exact change of the section's own penalised objective plus the linearised
   effect on the rest of the site (gradient of the other sections' terms, obtained with one backward pass
   per colour of a graph colouring of the section interaction graph). The best option per section is
   applied jointly and verified on the full twin with backtracking.

Work: exponential only inside a section (≤ 64 options), linear in the number of sections - the batch size
does not grow with the site. In the cascade the stage runs as an exact large-neighbourhood search on the
GA's best plan (`discrete="ga+sections"`): its result is accepted only if the penalised score improves the
GA's plan. Run alone instead of the GA (`discrete="sections"`) it is weaker on fields where discrete and
continuous decisions are tightly coupled (switching a well changes the optimal speeds of its neighbours,
which the section move holds fixed). **Measured so far:** the scaling holds (sections stay at 3 assets
while the site grows 8x), but the plan value is not measurably better than the GA's on either case study
(-0.0 % oil field, -0.4 % mine, within run-to-run noise; [BENCHMARKS.md](BENCHMARKS.md)). The discrete
choices of these two models interact too little for an exact search to find more than the GA does.

**Native kernels** (`plyxis/native/sections.cpp`, C++17 + OpenMP, loaded with ctypes, built by
`python -m plyxis.native build` with g++/MinGW-w64): union-find clustering under the budget, per-section
candidate scoring (segment sums over instances and decisions), exact argmax, and a Viterbi dynamic
programme over time windows with switching costs. Every kernel has a NumPy reference implementation with
identical results (used when no compiler is available, and in the benchmark).

Related work this builds on: learned neighbourhood selection for large-neighbourhood search on MIPs
(Sonnerat et al., *Learning a Large Neighborhood Search Algorithm for Mixed Integer Programs*, 2021;
two-layer LNS with a graph transformer, 2024), data-driven Dantzig-Wolfe decomposition (Basso & Ceselli,
*Math. Prog. Computation* 2023; Parmentier, *Learning when to use a decomposition*) and dynamic programming
over tree decompositions, which is exponential in the decomposition width but linear in the problem size
(Bodlaender & Koster). PLYXIS differs in where the structure comes from: the graph is the twin's netlist,
and the coupling labels are measured on the differentiable twin rather than on a MIP formulation.

### Metaheuristics (`metaheur.py`)

graviSearch (gravitational search), PSO, GA and DE, fully vectorised (one generation = one
batched simulation). Each has a cooperative mode: if the fitness returns one score per
subsystem `[P, K]`, selection happens per subsystem - K small searches in parallel instead of
one huge one (cooperative coevolution). The cascade uses this inside stage 3.

## Rolling horizon and live operation (`runtime/planner.py`)

* Re-plan every 3 h of plant time, immediately after a disruption, a twin rewrite, an engineer
  edit or a market notice.
* Warm start: the previous plan shifted by the elapsed time; after structural rewrites the
  decision vector is remapped by asset path (`remap_x`), new assets start from defaults.
* Assets with an active work order are excluded from new work-over assignments
  (`worked_assets`).
* Hold blocks and `route_hold` keep plans executable: no valve chatter, no line-up changes
  every hour.
* **Anytime planning.** After an urgent trigger (disruption, midstream or market notice, twin
  rewrite, an asset back in service) the planner first publishes a *first response*: 60
  augmented-Lagrangian iterations from the shifted previous plan and the neural proposal, with
  the discrete decisions kept. It is live within 1-2 s and releases the console's clock hold; the
  full cascade keeps running in the background and replaces it when it is done.
* **Move suppression.** As in model predictive control, decisions that take effect within the
  next 3 h are penalised for deviating from the plan being executed (USD 1,500 per full-range
  change, quadratic, so small corrections are cheap; a quarter of that after an urgent trigger).
  Without it, two near-equivalent optima could swap at every re-plan and set points would jump
  for no operational reason.
* **Known outages.** When a hoist, fan, pump or loader fails with a crew estimate of the repair time, the
  twin's status of that asset becomes a trajectory over the horizon (down until the estimate, then back:
  `Problem.set_state_schedule`). Without it the planner assumes a failed asset stays down for the whole
  48 h horizon - in the mine it then rationed a full ROM pad through a 5-hour hoist stop.
* **Terminal values and reserves.** The planning horizon ends; the plant does not. Inventories at the
  horizon end are valued like the KPI - at cost - so a plan cannot create value by mining ahead or by
  parking ore. Buffers carry soft reserves for what is not announced instead: sumps keep room for 8 h of
  inflow (a pump failure), the ROM pad keeps a working stock that bridges a hoist stop. A terminal value at
  the full milling margin was tried and removed: it is only right when the mill has spare capacity after
  the horizon, and in the mill-bound mine it made the planner idle the mill through a price peak. The only
  planning-only terminal term left (`value_to_go`, never in the realised KPIs) values broken ore at half
  its blasting cost, so rings are fired just in time.

## From plan to action

* `dispatch.py` turns the plan into instructions (set points with start/end times), tank
  battery line-ups, truck trips per lane, and crew/rig work orders (`maintenance.py`).
* `explain.py` answers *why* for any asset and decision:
  `dL/du = sum_k dTerm_k/du + sum_c lambda_c dh_c/du`, i.e. the marginal value decomposed into
  objective terms (oil sales, power, trucking, emissions ...) and priced bottlenecks, rendered by
  deterministic templates.
