# Decomposition theory: sections as exact functions

This document is the mathematical basis of the decomposition solvers (`plyxis/optim/structure.py`,
`branchlp.py`, `sections.py`, `decomp.py`, `sectiondp.py`, kernels in `plyxis/native`). The question it
answers: how can a site plan - which stopes fire in which shift, which wells run, how much air each level
gets, when the pumps run - be optimised when exact optimisation of such problems takes time exponential in
the number of decisions? Measured results are in [BENCHMARKS.md](BENCHMARKS.md).

The short answer has two parts. **What kind of problem each branch is:** the twin's equations prove which
branches are problems with exact, fast, classical methods - a linear programme, a storage schedule, a chain
over time - and those branches are solved by those methods (section 7). **How the rest interacts:** the
twin's graph shows which decisions interact; decisions that interact strongly are grouped into small
**sections**, each solved *exactly* and turned into a table - a deterministic input-output function - and
the sections are coordinated by **dynamic programming** along forests of their interactions and by
**prices** for the capacities they share. The work grows linearly with the site and exponentially only with
the size of a section; what the decomposition leaves out is bounded, and every plan carries a certified
bound on how far it can be from the optimum.

---

## 1. The problem

A compiled PLX twin defines decisions `u = (c, z)`: continuous set points `c` and discrete choices `z`
(integers, booleans, exclusive line-ups). The planner maximises the penalised plan value

```
F(c, z) = J(c, z) / J_scale - (mu / T) * sum of constraint violations
```

where `J` is the revenue minus cost of the plan over the horizon (`Problem.evaluate`). The discrete
choices are grouped into **units** `i` (one asset's choice in one time window) with finite option sets
`O_i`. With the continuous part fixed, `F` is a function of `z` alone. Maximising it exactly is NP-hard in
general (it contains max-cut and machine scheduling); brute force costs `prod_i |O_i|` evaluations.

## 2. Interaction structure

**Expansion.** Around an incumbent `z0`, every set function has the exact (Moebius/ANOVA) expansion

```
F(z) = F(z0) + sum_i d_i(z_i) + sum_{i<j} d_ij(z_i, z_j) + R3(z)
d_i(o)      = F(z0 with z_i = o) - F(z0)
d_ij(o, p)  = F(z0 with z_i = o, z_j = p) - F(z0 with z_i = o) - F(z0 with z_j = p) + F(z0)
```

`d_ij` is the **cross term**: how much the value of `i`'s choice depends on `j`'s. The **interaction graph**
has an edge `(i, j)` where `d_ij` is not identically zero; its weight `omega_ij` is the range
`max d_ij - min d_ij`.

**Locality (why twins are decomposable).** A twin's terms are computed along its netlist (flows from asset
to asset) and through time (state updates). Two decisions can only have a cross term if some term of the
plan depends on both, i.e. if they share a descendant in the computation graph within the horizon. In a
site, most decisions share descendants only with their neighbours (wells on one pad, stopes on one level,
the same asset in consecutive shifts); the exceptions are **hubs** - shared capacities such as the mill,
the hoist, a loader fleet, a crew - which touch many decisions but in a simple way (they add up their use).

**Learned graph filter.** The interaction weights are estimated from the netlist by a polynomial graph
filter `H = sum_{k<=K} A^k X W_k` (two layers, `K = 3` hops; learned spectral filters on the normalised
instance graph). It is trained on exact cross terms measured on the twin (finite differences as above)
and then replaces hundreds of probing simulations by one forward pass. It learns real structure where
there is some (rank correlation 0.73 with the exact couplings on the oil field) and reports when there
is none (0.03 on the mine, whose discrete choices barely interact).

## 3. Sections, blocks, regions

* **Sections.** Units are merged into sections along the strongest predicted couplings (union-find in
  weight order) as long as a section's joint options stay within a budget `B` (64). This is a greedy
  heuristic for the size-constrained minimum-cut partition - the partition that leaves the least
  interaction between sections (that problem itself is NP-hard; the greedy is not optimal, which the
  bounds below account for).
* **Blocks.** A block `b` is a section in one time window; its options are the joint options of the
  section's units in that window, `|O_b| <= B`.
* **Regions.** A section's region is its assets plus the assets only it touches (its *halo*: e.g. an ore
  pass fed by stopes of one section). Assets touched by several sections are *shared*; assets touched by
  more than `h` (12) sections are *hubs*. Two sections are **neighbours** if their regions touch or share
  an asset.

## 4. Section functions - the deterministic input-output functions

For a block `b` of section `c` define its table

```
u_b(o) = [ change of the penalised terms of region R_c when block b takes option o ]
       + < pi_b , dx_b(o) >,           pi_b = gradient of (F - F_{R_c}) w.r.t. the decisions of b
```

The first part is exact (simulated); the second prices the block's effect on everything outside its
region - the shared assets and hubs - at their marginal value. `u_b` is the section's **input-output
function**: given the prices `pi` (the input), its best response `argmax_o u_b(o)` and its value are a
lookup in a table of at most `B` entries. As a function of the prices, the block's value
`phi_b(pi) = max_o [ v_b(o) + <pi, dx_b(o)> ]` is the maximum of affine functions - convex and piecewise
linear - and `dx_b` at the maximiser is its subgradient. This is what makes price coordination work.

**Proposition 1 (exact attribution in batches).** If no two blocks of a batch belong to the same or to
neighbouring sections, every block's region terms in one batched simulation change only because of that
block, so all their tables are read off one simulation exactly. *Condition:* a region's terms depend only
on decisions of its own and of neighbouring regions, and on hubs. Regions, halos and the neighbour relation
are constructed to make this true for flow networks; influence through hubs enters only through the prices,
and the verification step (section 8) guards whatever the model misses. In the code a greedy colouring of
the blocks plans the batches (`Decomposer._unary_plan`).

## 5. Coordinating sections: dynamic programming on a forest

The model the solver maximises around the incumbent is

```
M(z) = sum_b u_b(z_b) + sum_{(a,b) in T} psi_ab(z_a, z_b)
```

with `psi_ab` the exact cross-term tables of coupled blocks (measured on the union of their regions and
shared assets), for the edges of a forest `T` of the block interaction graph: **time edges** (a section
in consecutive windows) and **space edges** (neighbouring sections in one window).

**Theorem 1 (exact forest DP).** If `T` is a forest, `max_z M(z)` is attained by max-sum message passing:
eliminate leaves towards the roots, `m_{v->p}(z_p) = max_{z_v} [psi_pv(z_p, z_v) + u_v(z_v) + sum_c m_{c->v}(z_v)]`,
then back-track. Time `O(sum_{(a,b) in T} |O_a| |O_b|) <= O(n B^2)`, memory `O(n B)`.
*Proof:* on a tree the maximisation over a leaf depends only on its parent's choice, so eliminating it
replaces it by a unary term on the parent without changing the optimum; induction over the leaves
(nonserial dynamic programming, Bertele & Brioschi 1972; max-product on trees, Pearl 1988). The kernel is
`px_forest_dp` (C++), checked against brute force on 200 random forests (`tests/test_sectiondp.py`).

**Corollary (where the exponential goes).** The work is exponential only in the section size
(`B = prod` of a section's unit options) and linear in the number of blocks. For a fixed budget `B` the
discrete stage of a site with `n` blocks costs `O(n B^2)` - this is the claimed reduction: exponential
growth inside small sections, linear growth over the site.

**Theorem 2 (treewidth).** More generally, if the block interaction graph has treewidth `w`, `M` with all
its edges is maximised exactly in `O(n B^(w+1))` by variable elimination along a tree decomposition
(Bodlaender & Koster 2008). Gathering networks and ore-flow trees have `w = 1`; a grid of `s` sections by
`W` windows has `w <= min(s, W)`; graphs that embed in the plane have `w = O(sqrt n)`, giving
`2^O(sqrt(n) log B)` - sub-exponential. The implementation uses forests (`w = 1`) and handles the remaining
edges by the bounds and searches below; elimination with larger width is the natural extension where a
site's structure calls for it.

**Choice of the forest.** Of all spanning forests, the one that maximises the summed interaction ranges of
its edges (Kruskal on the ranges) minimises what is left outside it - and that is exactly the error bound
of Theorem 3.

## 6. What the decomposition leaves out - bounds

**Theorem 3 (forest bound).** If `z_T` maximises the forest model `M_T` and the full pairwise model is
`M = M_T + sum_{e not in T} psi_e`, then for the optimum `z*`

```
M(z*) - M(z_T) <= sum_{e not in T} range(psi_e).
```

*Proof:* `M(z*) = M_T(z*) + sum_off psi_e(z*) <= M_T(z_T) + sum_off max psi_e` and
`M(z_T) >= M_T(z_T) + sum_off min psi_e`; subtract. Hence the maximum-weight spanning forest minimises the
bound.

**Theorem 4 (block-wise optima).** If `z^` is block-wise optimal (no single block can improve `M`), then
`M(z*) - M(z^) <= sum_{(a,b) in E} kappa_ab` with `kappa_ab = max_{o,p} [psi(o,p) - psi(o,z^_b) - psi(z^_a,p) + psi(z^_a,z^_b)]`.
*Proof:* move from `z^` to `z*` one block at a time; each step's gain equals its gain at `z^` (which is
`<= 0`) plus the cross terms with the blocks already moved, each bounded by `kappa`. Block coordinate
ascent therefore loses at most the interaction it ignores - small when the sections cut weak couplings.

**Theorem 5 (dual certificate).** Cover the edges by forests `F_1..F_K` (e.g. all time chains, all network
trees) and split every table: `sum_k theta_kb = u_b - <mu, r_b>` with prices `mu >= 0` on the shared
capacities `sum_b r_b(z_b) <= C`. For every such `theta, mu`

```
D(theta, mu) = sum_k max_z [ sum_b theta_kb(z_b) + sum_{e in F_k} psi_e(z) ] + <mu, C>  >=  max { M(z) : r(z) <= C }.
```

*Proof:* for any feasible `z`, the copies evaluated at the same `z` add up to `M(z) - <mu, r(z) - C>` `>= M(z)`,
and each copy's maximum is at least its value at `z`. Each copy is a forest problem (Theorem 1), so `D` is
computed exactly; minimising it over `theta, mu` (dual decomposition; Komodakis, Paragios & Tziritas 2011)
tightens it, and at its minimum it equals the bound of the LP relaxation over the local polytope
(Wainwright, Jaakkola & Willsky 2005). So every plan comes with a **certified gap** `(D - M(z)) / D`.
Plain subgradient steps converge slowly (`O(1/sqrt t)`). **Smoothing** fixes that without giving up
validity: replacing every copy's `max` by `tau log sum exp(./tau)` gives `D_tau >= D >= max M` for every
`tau > 0` (log-sum-exp is never below the max), and `D_tau` is smooth - its gradient with respect to the
shares is the difference of the copies' node marginals, computed by sum-product message passing on each
forest (`px_forest_logz`). It is minimised by L-BFGS-B with `tau` lowered in stages (Jojic, Gould & Koller
2010; Savchynskyy et al. 2011); every value evaluated is a valid bound, and so is the plain dual at each
stage's optimum (`sectiondp.solve(dual="smooth")`).

**Theorem 6 (shared capacities - why pricing hubs is accurate on large sites).** For a separable problem
`max sum_b f_b(z_b)` subject to `H` shared capacities `sum_b r_b(z_b) <= C`, the Lagrangian duality gap is at
most `(H + 1) max_b rho_b`, where `rho_b` is the non-convexity of block `b` (Shapley-Folkman; Aubin & Ekeland
1976; Udell & Boyd 2016). The gap does not grow with the number of blocks: with a handful of hubs (mill,
hoist, fleets, crews) and hundreds of sections, coordinating the hubs by prices loses a vanishing share of
the value. This is why hubs are priced rather than put into sections.

**Rest of the site.** The prices `pi_b` are a first-order model of everything outside a block's region;
its error is second order in the block's change (`<= L |dx|^2 / 2` for an `L`-Lipschitz gradient) and is
largest where a move hits a saturating capacity. The verification below makes every accepted move a real
improvement regardless.

## 7. Recognising branches that classical methods solve exactly

The section DP uses only the *shape* of the interactions. Many branches of a twin are, in addition, a
problem *class* for which the literature has exact, fast, deterministic methods - and the twin's equations
say which, because the PLX compiler holds every equation symbolically. `structure.py` classifies every
expression of the planning program with the composition rules of disciplined convex programming (Grant &
Boyd 2006): a sum of affine parts is affine, a non-negative multiple of a convex part is convex, `max` of
convex parts is convex, `min` of concave parts is concave, `pos`/`abs` of an affine part is convex and
piecewise affine, a product of two variables is bilinear, a power `x^k` (k >= 1) of a non-negative affine
part is convex; a piecewise-linear table of a decision whose whole range lies inside one of its segments
is affine there; signs come from the actual parameter values, decision bounds and initial states. Flows
are affine where their sources are; a state is affine if its update is (linear dynamics), and a state may be
convex or concave if its update reproduces that curvature from a state that already has it (checked by
induction) and every use of the state complies; a flow-weighted quality of a blend (an ore grade, an oil
gravity) is a ratio of flows - the pooling problem.

A branch is a connected set of assets (along the twin's flows) with one class:

| class (proven from the equations) | method | guarantee |
|---|---|---|
| affine, continuous | linear programme, HiGHS dual simplex | global optimum, exact shadow prices, finite |
| affine with integer / boolean decisions | MILP on small branches, section DP on large ones | exact / certified gap |
| piecewise affine (given the discrete choices and frozen qualities) | sequential LP with a trust region, Jacobian reuse | monotone (verified) |
| affine once discrete choices and blend qualities are fixed | successive LP (Haverly 1978; refinery blending practice) | monotone (verified) |
| convex (given the discrete choices, frozen qualities and the current sign of prices: cost convex, revenue concave, flows affine) | sequential LP on tangent planes | monotone (verified); tangent bound, see below |
| nonlinear | the cascade | local optimum, verified |

**Proposition 2 (the LP of an affine branch is exact).** If every term, constraint, dynamic and flow of a
branch is affine in its decisions with all other decisions fixed, the twin's plan value restricted to the
branch is `c.d - (mu/T) sum relu(h + J d) - sum w relu(h_s + J_s d) - move-suppression(d)`, with `c, J, J_s`
constant. The `relu` terms and the quadratic move-suppression term are represented exactly by slack
variables and (to any accuracy) by tangent cuts, so one linear programme gives the global optimum of the
branch, and its duals are the exact marginal values of the branch's constraints. Because `J` is constant,
it is obtained exactly from finite differences of the twin itself (`branchlp.jacobian`, sparse, a few
batched simulations) - no separate LP model has to be written or kept in sync with the twin.

**Coupling.** A branch interacts with the rest of the site through flows into shared assets and through
site-wide aggregates (total power under a grid limit). With the rest fixed these enter the LP exactly
when they are affine (they are part of the Jacobian), otherwise through the verification step. Branches
are solved after the cascade's polish (`CascadeConfig.exact_branches`), and a result is kept only if the
full twin confirms it.

**Conditional classes.** The cascade fixes the discrete choices before its continuous polish, and blend
qualities change slowly between plans. The analyser therefore also classifies each asset type with its
discrete decisions held fixed and its blend qualities frozen (both the incoming qualities and the stored
average quality of a bin or pass). A stope is then piecewise affine (a fixed ring firing turns the bilinear
`blast x broken` into a linear term), an ore pass is affine. For such layers the method is **sequential LP**:
the LP of section 7 at the current point with a trust region, verified on the twin, repeated; the Jacobian
(finite differences with a small step, local slopes) is reused while steps verify - the slopes of a
piecewise-affine function only change where a kink is crossed - and recomputed after a rejected step.

**The convex class and how close a plan is.** With on/off fixed an oil well is not generically nonlinear. Its rate
is linear in its pump speed (the pump curve has one segment on the speed range), its power (`speed^3`) and wear
(`speed^4`) are convex, its health is concave in the speeds, and the failure risk `exp(5 (1 - health))` is convex in
them. The verdict states what it rests on: the fixed discrete choices, the frozen qualities and a non-negative power
price. For a function that is concave in the decisions, the first-order model overestimates everywhere, so the LP
without a trust region gives a bound: `F(x) <= F(x0) + max_d [ grad.d - penalties(d) ] + penalties(0)`
(`branchlp.tangent_gap`). The analysis proves this per asset type, not for a whole twin (a well's oil flows on through
batteries and compressors with kinks), so the number is a **measured bound, not a theorem**, for the twin as a whole.
It was attacked on an oil-field plan from before the slope fix (`bench/checks/tangent_bound.py`, `arms/tangent_bound_check.log`):
sequential LP from three perturbed starts stayed below it, and the midpoint condition held on all 64 random chords. The
finite-difference slopes make it slightly approximate as well.

**Exact plans need margins.** An LP optimum is a vertex: it sits exactly on the binding constraints - as the
twin sees them. Wherever the twin's estimate is off, that becomes a violation in reality. The mine showed
it twice: with the LP finishing step, a sump was planned exactly on its storage reserve computed from an
inflow the twin underestimated by 17 %, and the unannounced pump failure then flooded it (92 m3, almost
exactly 8 h of the inflow error); and the mill feed was planned exactly on the head-grade window's upper
limit with stope grades the twin underestimated, so the real head grade ended above the window. The
first-order polish had stayed off those limits by luck, not by design. Two standard remedies are built in:
a **design margin** where an uncertainty is structural (the sump reserve is computed with 25 % more
inflow), and **offset-free planning** where a measured output exists (Pannocchia & Rawlings 2003): the
concentrator's `head_factor` - the ratio of the on-stream analyser's head grade to the blend the twin
computes - is estimated online like any calibrated property, and the plan's grade window applies to the
corrected grade.
The head grade is also planned inside the window, by one standard error of the blend the last plan mines
(`concentrator.grade_margin`, standard errors from the assays: `Estimator.uncertainty`); with that the realised head grade stays inside it.

**On the case-study twins** (BENCHMARKS.md): 98.6 % of the mine's decisions are proven linear (33.7 %) or
piecewise linear given the discrete choices and grades (64.9 %); the oil field's wells are genuinely nonlinear
(7.5 % of its decisions are in linear classes). The pure LP branches turn out to be solved already by the
cascade (the LP certifies them). Sequential LP over the whole recognised layer adds USD 47,000 per 48 h to
the mine's A/B result (three seeds, all positive) and lowers the oil field's by USD 47,800 per 72 h (three
seeds, all negative): there it drives the tank batteries to their limits. With a 10 % margin on state limits
(`exact_margin`) the oil-field loss shrinks to USD 7,400 but does not turn into a gain. A gain in the twin's objective
for one plan predicts neither - only the A/B over seeds decides, and the stage runs on the mine only.

## 8. The algorithms as implemented

**On the twin** (`decomp.py`, `discrete="ga+decomp"` / `"decomp"` in the cascade):

1. prices `pi` of the rest of the site: one backward pass per colour of the section graph;
2. unary tables of all blocks: batched simulations planned by the block colouring (Proposition 1);
3. pair tables on the maximum-weight spanning forest (time and space edges), for each block's top-`q` options
   (`q = 6`), batched over edges whose sections and neighbourhoods are disjoint;
4. forest DP (Theorem 1) - the exact maximiser of the model;
5. **verification as a batched line search**: prefixes of the DP move (blocks sorted by their predicted gain)
   and every forest component on its own are simulated in one batch on the full twin; the best real
   improvement is taken. Every accepted step increases `F`, so the sweeps terminate (finite option sets);
6. repeat around the new incumbent until the verified gain is below the model's resolution.

Event-local re-planning: after an event, only blocks of sections within one neighbourhood of the changed
assets are re-solved (`solve(focus=...)`); the other sections keep their choices.

**On explicit table models** (`sectiondp.py`, the core used by the theory benchmark):

1. start: the strongest forest solved exactly, repaired to the capacities, block coordinate ascent;
2. **exact moves over induced forests**: draw a random set `S` of blocks whose induced interaction graph is
   a forest; conditioned on the blocks outside `S`, the plan value over `S` is a forest problem, so the best
   joint change of all of `S` (thousands of blocks at once) is found exactly; shared capacities by bisection
   on their price. Monotone (a large-neighbourhood search whose neighbourhood is solved exactly);
3. dual decomposition over a forest cover for the certificate (Theorem 5), Polyak steps towards the best
   plan, majority votes of the copies offered as further plans.

Native kernels (C++17/OpenMP, NumPy references with identical results): forest DP, block coordinate ascent
with capacities, clustering, candidate scoring.

## 9. Complexity per sweep (twin)

| step | simulations / work |
|---|---|
| prices of the rest | (colours of the section graph + 1) backward passes |
| unary tables | (block colours) x (largest option set) samples, batched |
| pair tables | (edge classes) x q^2 samples, batched |
| forest DP | O(n q^2) arithmetic (microseconds per block) |
| verification | one batch of <= 64 candidate plans |

On the twins the batched simulation costs about 1 ms per sample, so a sweep is a few seconds; on both
twins the discrete layer around the relaxed plan is weakly coupled, and the exact method reaches the same
plan value as the simpler per-window section search (BENCHMARKS.md). The gains of the method appear where
discrete choices interact - measured on scheduling models with known optima in BENCHMARKS.md.

## 10. What is proven, what is heuristic

| claim | status |
|---|---|
| forest DP is exact; `O(n B^2)` | proven (Theorem 1), tested against brute force |
| the certificate is a valid upper bound | proven (Theorem 5), tested against brute force (both dual methods) |
| the smoothed certificate is tight in practice | measured: 1.7-3.7 % up to 90,000 blocks, true gap smaller |
| what the forest leaves out is bounded by the off-forest ranges | proven (Theorem 3) |
| pricing a few hubs loses a share that vanishes with the number of sections | proven for separable problems (Theorem 6); twins are not exactly separable |
| tables read off one batched simulation are exact | under the locality condition of Proposition 1 |
| the sections found by the graph filter are good | heuristic (greedy partition, learned weights) |
| the twin's discrete layer around the relaxed plan is weakly coupled | measured (BENCHMARKS.md), model-dependent |
| a branch classified affine is affine | proven from the equations (sound rules; conservative: some affine branches are reported as nonlinear) |
| the LP of an affine branch is its global optimum | proven (Proposition 2), verified on the twin |
| sequential LP improves the realised result | measured: yes on the mine (+USD 47 k / 48 h, 3 seeds), no on the oil field (-USD 7 k / 72 h with 10 % margins on state limits, -USD 48 k without, 3 seeds) |

## Literature

* U. Bertele, F. Brioschi, *Nonserial Dynamic Programming*, Academic Press 1972.
* J. Pearl, *Probabilistic Reasoning in Intelligent Systems*, 1988 (max-product on trees).
* H. L. Bodlaender, A. M. C. A. Koster, Combinatorial optimization on graphs of bounded treewidth,
  *The Computer Journal* 51 (2008).
* M. J. Wainwright, T. S. Jaakkola, A. S. Willsky, MAP estimation via agreement on trees,
  *IEEE Trans. Inf. Theory* 51 (2005).
* N. Komodakis, N. Paragios, G. Tziritas, MRF energy minimization and beyond via dual decomposition,
  *IEEE TPAMI* 33 (2011).
* J.-P. Aubin, I. Ekeland, Estimates of the duality gap in nonconvex optimization, *Math. Oper. Res.* 1 (1976);
  M. Udell, S. Boyd, Bounding duality gap for separable problems with linear constraints,
  *Comput. Optim. Appl.* 64 (2016).
* V. Jojic, S. Gould, D. Koller, Accelerated dual decomposition for MAP inference, ICML 2010;
  B. Savchynskyy, J. Kappes, S. Schmidt, C. Schnoerr, A study of Nesterov's scheme for Lagrangian
  decomposition and MAP labeling, CVPR 2011 (smoothed dual decomposition).
* G. Pannocchia, J. B. Rawlings, Disturbance models for offset-free model-predictive control,
  *AIChE J.* 49 (2003) (offset-free MPC: an estimated output disturbance).
* M. L. Fisher, The Lagrangian relaxation method for solving integer programming problems,
  *Management Science* 27 (1981).
* A. R. Curtis, M. J. D. Powell, J. K. Reid, On the estimation of sparse Jacobian matrices,
  *IMA J. Appl. Math.* 13 (1974) (colouring for simultaneous evaluation).
* P. Shaw, Using constraint programming and local search methods to solve vehicle routing problems, CP 1998
  (large-neighbourhood search); J. Sonnerat et al., Learning a large neighborhood search algorithm for mixed
  integer programs, 2021 (learned neighbourhoods).
* M. Grant, S. Boyd, Y. Ye, Disciplined convex programming, in *Global Optimization*, Springer 2006.
* C. A. Haverly, Studies of the behavior of recursion for the pooling problem, *ACM SIGMAP Bull.* 25 (1978).
* Q. Huangfu, J. A. J. Hall, Parallelizing the dual revised simplex method, *Math. Prog. Comp.* 10 (2018) (HiGHS).
* D. I. Shuman et al., The emerging field of signal processing on graphs, *IEEE Signal Proc. Mag.* 30 (2013)
  (polynomial graph filters).
