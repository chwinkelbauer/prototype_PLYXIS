"""Why does the structured Jacobian give a worse sequential LP than finite differences, at the same point?
Compares both Jacobians at a cascade plan of the oil field, on the wells' speed decisions."""
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
x0 = Cascade(pb, config=cfg).solve().x.detach().clone()
var = BL.class_variables(pb, ("convex",))
J = pb.J_scale
t = time.perf_counter()
fd = BL.jacobian(pb, x0, var, step=0.02)
print(f"finite differences {time.perf_counter() - t:.1f}s nnz {fd['J'].nnz} rows {len(fd['meta'])}", flush=True)
BL._PATTERN.clear()
BL.jacobian_structured(pb, x0, var, step=0.02)            # fills the pattern
t = time.perf_counter()
st = BL.jacobian_structured(pb, x0, var, step=0.02)
print(f"structured {time.perf_counter() - t:.1f}s nnz {st['J'].nnz} colours {st['colours']} dense rows {st['dense_rows']}", flush=True)
g1, g2 = fd["grad"], st["grad"]
print(f"objective gradient: |fd| {np.abs(g1).sum():.5f} |st| {np.abs(g2).sum():.5f} |diff| {np.abs(g1 - g2).sum():.5f} max diff {np.abs(g1 - g2).max():.6f} corr {np.corrcoef(g1, g2)[0, 1]:.5f}", flush=True)
D = (fd["J"] - st["J"]).tocsr()
d = np.abs(D).sum(1).A1
print(f"rows: |J_fd| {np.abs(fd['J']).sum():.3f} |diff| {d.sum():.3f}; rows with |diff| > 1e-3: {(d > 1e-3).sum()}", flush=True)
by = {}
for i in np.argsort(-d)[:4000]:
    if d[i] > 1e-4:
        k = fd["meta"][i][:3]
        by[k] = by.get(k, 0.0) + d[i]
for k, v in sorted(by.items(), key=lambda kv: -kv[1])[:12]:
    print("   ", k, round(v, 4), flush=True)
# one LP on each, verified on the twin (same point, same rows otherwise)
F0 = float(BL._F(pb, x0)[0])
for name, jac in (("finite differences", fd), ("structured", st), ("fd slopes + reverse gradient", dict(fd, grad=g2)), ("structured rows + fd gradient", dict(st, grad=g1))):
    for tr in (None, 0.25):
        xn, s, lp = BL.solve_lp(pb, x0, var, jac, tr)
        al = torch.tensor([1.0, 0.5, 0.25, 0.1]).view(-1, 1)
        Fs = BL._F(pb, x0 + al * (xn - x0))
        print(f"{name:30s} trust {tr}: LP {s:.1f}s  F gain along the step {[round((float(f) - F0) * J) for f in Fs]}  mean |d| {np.abs((xn - x0)[0, var].numpy()).mean():.3f}", flush=True)
