"""The derivative of rt.pwl at the end of its table, and what it does to the oil field's gradient and cold plans."""
import os
import sys
import time

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from plyxis.lang import compile_model, rt  # noqa: E402
from plyxis.optim import branchlp as BL  # noqa: E402
from plyxis.optim.cascade import Cascade, CascadeConfig  # noqa: E402
from plyxis.optim.problem import Problem  # noqa: E402

torch.set_num_threads(3)
for u in (1.0, 0.999, 0.5, 0.0):
    lo, hi = torch.tensor(0.35), torch.tensor(1.0)
    x = (lo + (hi - lo) * torch.tensor(u)).requires_grad_(True)
    y = rt.pwl(x, (0.35, 1.0), torch.tensor(0.22), torch.tensor(1.0))
    y.backward()
    print(f"u {u}: speed {float(x):.9f} pwl {float(y):.7f} slope {float(x.grad):.4f} (true 1.2000)", flush=True)
m = compile_model([os.path.join("scenarios", "kestrel", "field.plx")])
pb = Problem(m.program)
cfg = CascadeConfig(relax_iters=200, population=12, generations=1, repair_iters=80, polish_iters=120, multistart=8, use_neural=False)
t = time.perf_counter()
res = Cascade(pb, config=cfg).solve()
x0 = res.x.detach().clone()
print(f"cold plan {res.objective:,.0f} violation {res.max_violation:.4f} in {time.perf_counter() - t:.1f}s", flush=True)
var = BL.class_variables(pb, ("convex",))
print(f"wells at the top of the speed range: {(x0[0, var] > 0.999).float().mean():.3f}, just below it (0.95-0.999): {((x0[0, var] > 0.95) & (x0[0, var] <= 0.999)).float().mean():.3f}, at the bottom: {(x0[0, var] < 0.001).float().mean():.3f}", flush=True)
fd = BL.jacobian(pb, x0, var, step=0.02)
BL._PATTERN.clear()
BL.jacobian_structured(pb, x0, var, step=0.02)
st = BL.jacobian_structured(pb, x0, var, step=0.02)
g1, g2 = fd["grad"], st["grad"]
print(f"objective gradient: |fd| {np.abs(g1).sum():.5f} |reverse| {np.abs(g2).sum():.5f} |diff| {np.abs(g1 - g2).sum():.5f} corr {np.corrcoef(g1, g2)[0, 1]:.5f}", flush=True)
J = pb.J_scale
F0 = float(BL._F(pb, x0)[0])
for name, jac in (("finite differences", fd), ("structured", st)):
    for tr in (None, 0.25):
        xn, s, lp = BL.solve_lp(pb, x0, var, jac, tr)
        al = torch.tensor([1.0, 0.5, 0.25, 0.1]).view(-1, 1)
        Fs = BL._F(pb, x0 + al * (xn - x0))
        print(f"{name:20s} trust {tr}: F gain along the step {[round((float(f) - F0) * J) for f in Fs]}", flush=True)
BL._PATTERN.clear()
br = BL.solve_branches(pb, x0, var, affine=False, iters=6)
gap = BL.tangent_gap(pb, br.x, var, BL.jacobian(pb, br.x, var, step=0.005))
print(f"sequential LP on the wells: gain USD {(br.F - F0) * J:+,.0f} gap USD {gap * J:,.0f} in {br.seconds:.1f}s iterations {br.iterations}", flush=True)
