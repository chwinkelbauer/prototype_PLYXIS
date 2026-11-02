"""Does the pwl fix change the mine's gradients? Old rt.pwl against the new one, at several points, plus the values."""
import json
import os
import sys

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
import torch  # noqa: E402

from plyxis.lang import compile_model, rt  # noqa: E402
from plyxis.optim.problem import Problem  # noqa: E402

torch.set_num_threads(2)


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
for scen in ("corvus", "kestrel"):
    sd = os.path.join("scenarios", scen)
    sc = json.load(open(os.path.join(sd, "scenario.json")))
    m = compile_model([os.path.join(sd, f) for f in sc["twin"]["files"]], env=sc["twin"]["env"])
    pb = Problem(m.program)
    pb.calibrate_scales()
    torch.manual_seed(0)
    x0 = pb.space.default_u(1)
    pts = [("default", x0)] + [(f"noise {s}", (x0 + s * torch.randn_like(x0)).clamp(0, 1)) for s in (0.05, 0.2)] + [("all at top", torch.ones_like(x0))]
    for name, x in pts:
        res = []
        for f in (old_pwl, new_pwl):
            rt.pwl = f
            X = x.clone().requires_grad_(True)
            e = pb.evaluate(X, relax=False)
            g = torch.autograd.grad(e.objective.sum(), X, allow_unused=True)[0]
            res.append((float(e.objective[0]), g[0].clone()))
        rt.pwl = new_pwl
        d = (res[0][1] - res[1][1]).abs()
        print(f"{scen:8s} {name:12s} objective old {res[0][0]:,.3f} new {res[1][0]:,.3f} | gradient: largest change {float(d.max()):.4g} of largest {float(res[1][1].abs().max()):.4g}, changed entries {int((d > 1e-6 * res[1][1].abs().max()).sum())} of {d.numel()}", flush=True)
