"""Is the tangent-plane bound of the wells' layer a real bound? Bound at the cascade plan and at the SLP end point,
then the best plan value found by line searches, random points and SLP from other starts must stay below it."""
import os
import sys
import time

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from plyxis.lang import compile_model  # noqa: E402
from plyxis.optim import branchlp as BL  # noqa: E402
from plyxis.optim.cascade import Cascade, CascadeConfig  # noqa: E402
from plyxis.optim.problem import Problem  # noqa: E402

torch.set_num_threads(3)
m = compile_model([os.path.join("scenarios", "kestrel", "field.plx")])
pb = Problem(m.program)
cfg = CascadeConfig(relax_iters=200, population=12, generations=1, repair_iters=80, polish_iters=120, multistart=8, use_neural=False)
res = Cascade(pb, config=cfg).solve()
x0 = res.x.detach().clone()
var = BL.class_variables(pb, ("convex",))
print("plan", round(res.objective), "violation", round(res.max_violation, 4), "wells decisions", len(var), flush=True)
J = pb.J_scale


def usd(f):
    return f * J


for name, step in (("step 0.02", 0.02), ("step 0.005", 0.005)):
    jac = BL.jacobian(pb, x0, var, step=step)
    gap = BL.tangent_gap(pb, x0, var, jac)
    print(f"{name}: F {jac['F0']:.5f} gap {gap:.5f} = USD {usd(gap):,.0f}  bound F {jac['F0'] + gap:.5f}", flush=True)
bound0 = jac["F0"] + gap
t = time.perf_counter()
br = BL.solve_branches(pb, x0, var, affine=False, iters=8)
print(f"SLP: F {br.F:.5f} (+USD {usd(br.F - br.F0):,.0f}) {br.iterations} iters {time.perf_counter() - t:.1f}s; below the bound from the plan: {br.F <= bound0}", flush=True)
jac = BL.jacobian(pb, br.x, var, step=0.005)
gap = BL.tangent_gap(pb, br.x, var, jac)
print(f"gap at the SLP point: {gap:.5f} = USD {usd(gap):,.0f}, bound F {br.F + gap:.5f}", flush=True)
bound1 = br.F + gap
# attack the bound: other starts inside the convex layer must not end above it
rng = np.random.default_rng(0)
best = br.F
for k, sc in enumerate((0.05, 0.15, 0.4)):
    x = br.x.clone()
    x[0, var] = torch.as_tensor(np.clip(x[0, var].numpy() + rng.normal(0, sc, len(var)), 0, 1), dtype=x.dtype)
    F1 = float(BL._F(pb, x)[0])
    r = BL.solve_branches(pb, x, var, affine=False, iters=8)
    best = max(best, r.F)
    print(f"start {k}: noise {sc}: F {F1:.5f} -> SLP {r.F:.5f}  (bound {bound1:.5f}, excess USD {usd(r.F - bound1):,.0f})", flush=True)
# concavity along random chords: F(mid) >= (F(a) + F(b)) / 2 for a concave function
bad, worst = 0, 0.0
for k in range(64):
    a, b = br.x.clone(), br.x.clone()
    a[0, var] = torch.as_tensor(np.clip(a[0, var].numpy() + rng.normal(0, 0.1, len(var)), 0, 1), dtype=x.dtype)
    b[0, var] = torch.as_tensor(np.clip(b[0, var].numpy() + rng.normal(0, 0.1, len(var)), 0, 1), dtype=x.dtype)
    Fs = BL._F(pb, torch.cat([a, b, 0.5 * (a + b)]))
    d = float(Fs[2]) - 0.5 * float(Fs[0] + Fs[1])
    if d < -1e-5:
        bad += 1
        worst = min(worst, d)
print(f"midpoint concavity: {bad} of 64 chords violate it, worst USD {usd(worst):,.0f}", flush=True)
