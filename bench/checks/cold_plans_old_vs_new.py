"""Cold plans of the oil field with the old and the new pwl slope, same seeds and budget (no neural start)."""
import dataclasses
import json
import os
import sys
import time

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
import torch  # noqa: E402

from plyxis.lang import compile_model, rt  # noqa: E402
from plyxis.optim import branchlp as BL  # noqa: E402
from plyxis.optim.cascade import Cascade  # noqa: E402
from plyxis.optim.problem import Problem  # noqa: E402
from plyxis.runtime.planner import Planner  # noqa: E402

torch.set_num_threads(4)


def old_pwl(x, xs, *ys):
    x = rt.T(x)
    n = len(xs)
    out = None
    xc = torch.clamp(x, min=xs[0], max=xs[-1])
    for i in range(n - 1):
        x0, x1 = xs[i], xs[i + 1]
        w = torch.clamp((xc - x0) / max(x1 - x0, 1e-12), 0.0, 1.0)
        seg = (1 - w) * ys[i] + w * ys[i + 1] if i == 0 else w * (ys[i + 1] - ys[i])
        out = seg if out is None else out + seg
    return out


new_pwl = rt.pwl
sd = os.path.join("scenarios", "kestrel")
sc = json.load(open(os.path.join(sd, "scenario.json")))
m = compile_model([os.path.join(sd, f) for f in sc["twin"]["files"]], env=sc["twin"]["env"])
pb = Problem(m.program)
pb.calibrate_scales()
tot = {"old": [], "new": []}
for seed in range(4):
    for name, f in (("old", old_pwl), ("new", new_pwl)):
        rt.pwl = f
        cfg = dataclasses.replace(Planner.LIVE, **sc.get("planner", {}), use_neural=False, seed=seed)
        t = time.perf_counter()
        r = Cascade(pb, config=cfg).solve()
        s = time.perf_counter() - t
        rt.pwl = new_pwl
        Fv = float(BL._F(pb, r.x[:1])[0])
        tot[name].append(r.objective)
        print(f"seed {seed} {name}: objective {r.objective:,.0f} F {Fv:.5f} violation {r.max_violation:.4f} evals {r.evals if hasattr(r, 'evals') else '?'} {s:.1f}s", flush=True)
print({k: round(sum(v) / len(v)) for k, v in tot.items()}, flush=True)
