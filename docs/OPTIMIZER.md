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
and the default plan. When the model has no discrete decisions, stage 3 is skipped and its
budget goes to the polish.

### Augmented Lagrangian (`gradient.py`)

Powell-Hestenes-Rockafellar form on signed, scaled residuals, minimised with projected Adam
(cosine learning-rate schedule) over a whole batch of starts at once. Every `outer` iterations
the multipliers are updated (`lambda <- relu(lambda + rho * h)`), and the penalty weight `rho`
triples only when the worst violation has not at least halved. The multipliers at the optimum are
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
`bench/results/neural_ablation.json`): the network proposes a plan for the whole field in about
20 ms. That plan is not feasible as it stands (0/8) - it is a starting point. After the same 1.8 s
gradient repair, it is worth 3.6 % more than repairing the previous plan (both feasible in 8/8
cases). Inside the full cascade, the neural start wins on 7 of 8 cases (median +2.5 %, mean
+2.0 %; one case -0.5 %) at the same solve time. An earlier training run on a previous version of
the model showed the same direction with a larger spread (one case +141 %, where the cascade alone
was stuck in a poor local optimum).

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

## From plan to action

* `dispatch.py` turns the plan into instructions (set points with start/end times), tank
  battery line-ups, truck trips per lane, and crew/rig work orders (`maintenance.py`).
* `explain.py` answers *why* for any asset and decision:
  `dL/du = sum_k dTerm_k/du + sum_c lambda_c dh_c/du`, i.e. the marginal value decomposed into
  objective terms (oil sales, power, trucking, emissions ...) and priced bottlenecks, rendered by
  deterministic templates.
