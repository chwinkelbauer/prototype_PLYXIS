"""Runtime helpers used by generated PLX simulation code.

All helpers accept Python floats or torch tensors and are differentiable where
it makes sense (rounding ops use straight-through estimators).
"""
from __future__ import annotations

import builtins as _bi

import torch

EPS = 1e-9
_F = torch.float32


def set_dtype(dt) -> None:
    global _F
    _F = dt


def T(x):
    return x if isinstance(x, torch.Tensor) else torch.tensor(float(x), dtype=_F)


def div(a, b):
    if isinstance(b, torch.Tensor):
        b = torch.where(b.abs() < 1e-12, torch.full_like(b, 1e-12) * torch.where(b < 0, -1.0, 1.0), b)
        return a / b
    if b == 0:
        b = 1e-12
    return a / b


def safediv(a, b):
    """a / b for non-negative denominators (flow-weighted averages)."""
    if isinstance(b, torch.Tensor):
        return a / b.clamp(min=EPS)
    return a / max(b, EPS)


def pow(a, b):
    if not isinstance(b, torch.Tensor) and float(b).is_integer():
        return a ** int(b)
    if isinstance(a, torch.Tensor):
        return a.clamp(min=0.0) ** b
    return max(a, 0.0) ** b


def _bin(fn, a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return fn(T(a), T(b)).to(_F)
    return 1.0 if fn(torch.tensor(a), torch.tensor(b)).item() else 0.0


def lt(a, b): return _bin(torch.lt, a, b)
def le(a, b): return _bin(torch.le, a, b)
def gt(a, b): return _bin(torch.gt, a, b)
def ge(a, b): return _bin(torch.ge, a, b)
def eq(a, b): return _bin(torch.eq, a, b)
def ne(a, b): return _bin(torch.ne, a, b)


def or_(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return torch.clamp(T(a) + T(b), max=1.0)
    return min(1.0, a + b)


def min2(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return torch.minimum(T(a), T(b))
    return min(a, b)


def max2(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return torch.maximum(T(a), T(b))
    return max(a, b)


def minn(*xs):
    out = xs[0]
    for x in xs[1:]:
        out = min2(out, x)
    return out


def maxn(*xs):
    out = xs[0]
    for x in xs[1:]:
        out = max2(out, x)
    return out


def abs(a): return torch.abs(a) if isinstance(a, torch.Tensor) else _bi.abs(a)
def relu(a): return torch.relu(a) if isinstance(a, torch.Tensor) else max(a, 0.0)
def negpart(a): return torch.clamp(a, max=0.0) if isinstance(a, torch.Tensor) else min(a, 0.0)


def sqrt(a): return torch.sqrt(torch.clamp(a, min=0.0)) if isinstance(a, torch.Tensor) else max(a, 0.0) ** 0.5
def exp(a): return torch.exp(torch.clamp(T(a), max=80.0))
def log(a): return torch.log(torch.clamp(T(a), min=1e-12))
def log10(a): return torch.log10(torch.clamp(T(a), min=1e-12))
def sin(a): return torch.sin(T(a))
def cos(a): return torch.cos(T(a))
def tanh(a): return torch.tanh(T(a))
def sigmoid(a): return torch.sigmoid(T(a))


def sign(a): return torch.sign(T(a))
def step(a): return (T(a) > 0).to(_F)


def _st(x, y):
    """straight-through: forward y, gradient of x."""
    return x + (y - x).detach()


def floor(a): a = T(a); return _st(a, torch.floor(a))
def ceil(a): a = T(a); return _st(a, torch.ceil(a))
def round(a): a = T(a); return _st(a, torch.round(a))


def clamp(x, lo, hi):
    return min2(max2(x, lo), hi)


def where(c, a, b):
    if isinstance(c, torch.Tensor):
        return torch.where(c > 0.5, T(a), T(b))
    return a if c > 0.5 else b


def smin(a, b, k):
    a, b = T(a), T(b)
    return -torch.logaddexp(-k * a, -k * b) / k


def smax(a, b, k):
    a, b = T(a), T(b)
    return torch.logaddexp(k * a, k * b) / k


def pwl(x, xs, *ys):
    """Piecewise-linear interpolation with constant breakpoints xs and (possibly tensor) ys."""
    x = T(x)
    n = len(xs)
    out = None
    # linear extrapolation is clamped to the end values
    xc = torch.clamp(x, min=xs[0], max=xs[-1])
    for i in range(n - 1):
        x0, x1 = xs[i], xs[i + 1]
        w = torch.clamp((xc - x0) / max(x1 - x0, 1e-12), 0.0, 1.0)
        seg = (1 - w) * ys[i] + w * ys[i + 1] if i == 0 else w * (ys[i + 1] - ys[i])
        out = seg if out is None else out + seg
    return out


def delay(x, k, init, T_len):
    """Vectorised delay along the last (time) axis: y[t] = x[t-k] (init before start)."""
    x = T(x)
    if isinstance(k, torch.Tensor):
        # per-instance integer delays: k shaped [Bp, n, 1]
        kk = k.round().long().clamp(min=1)
        Kmax = int(kk.max().item())
        x = x.expand(*x.shape[:-1], T_len) if x.shape[-1] == 1 else x
        pad = (T(init) * torch.ones(*x.shape[:-1], Kmax, dtype=x.dtype)).expand(*x.shape[:-1], Kmax)
        cat = torch.cat([pad, x], -1)
        idx = torch.arange(T_len).view(1, 1, -1) + Kmax - kk
        idx = idx.expand(*x.shape[:-1], T_len)
        return torch.gather(cat, -1, idx)
    k = int(k)
    if x.shape[-1] == 1:
        x = x.expand(*x.shape[:-1], T_len)
    if k >= T_len:
        return (T(init) * torch.ones_like(x))
    pad = T(init) * torch.ones(*x.shape[:-1], k, dtype=x.dtype)
    return torch.cat([pad.expand(*x.shape[:-1], k), x[..., :T_len - k]], -1)


def shift(nxt, x0):
    """State trajectory from next-values: x[0]=x0, x[t]=next[t-1]."""
    nxt = T(nxt)
    x0 = T(x0)
    if nxt.dim() == 3:
        B = max(nxt.shape[0], x0.shape[0])
        n = max(nxt.shape[1], x0.shape[1])
        nxt = nxt.expand(B, n, nxt.shape[-1])
        x0 = x0.expand(B, n)
        return torch.cat([x0.unsqueeze(-1), nxt[..., :-1]], -1)
    return nxt


def state_input(x, T_len):
    """Exogenous state: [B, n] (constant over the horizon) or [B, n, T] (given trajectory)."""
    x = T(x)
    if x.dim() == 3:
        return x
    return x.unsqueeze(-1).expand(*x.shape, T_len)


def state_final(x, B, n):
    x = T(x)
    if x.dim() == 3:
        x = x[..., -1]
    return full(x, B, n)


def excumsum(inc):
    """exclusive cumulative sum along time."""
    c = torch.cumsum(inc, -1)
    return c - inc


def full(x, B, n, T_len=None):
    x = T(x)
    if T_len is None:
        if x.dim() == 0:
            return x.expand(B, n)
        return x.expand(B, n) if x.dim() == 2 else x.reshape(x.shape[0], -1).expand(B, n)
    if x.dim() == 0:
        return x.expand(B, n, T_len)
    return x.expand(B, n, T_len)


def gather_sum(B, n, T_len, sources, R):
    """Sum flows from (src tensor, src_idx, dst_idx, routing_idx|None) into a [B, n(, T)] in-port tensor."""
    shape = (B, n) if T_len is None else (B, n, T_len)
    acc = torch.zeros(shape, dtype=_F)
    for src, sidx, didx, ridx in sources:
        s = full(src, B, src.shape[1] if isinstance(src, torch.Tensor) and src.dim() >= 2 else 1, T_len) \
            if isinstance(src, torch.Tensor) else torch.full((B, 1) if T_len is None else (B, 1, T_len), float(src))
        if isinstance(src, torch.Tensor) and src.dim() >= 2:
            s = s.index_select(1, sidx)
        else:
            s = s.expand(B, len(sidx)) if T_len is None else s.expand(B, len(sidx), T_len)
        if ridx is not None:
            s = s * R.index_select(1, ridx)
        acc = acc.index_add(1, didx, s)
    return acc


def agg(parts, how, B, T_len):
    """Aggregate selected instances of several groups -> [B, 1, T] (or [.,1,1] when static)."""
    vals = []
    cnt = 0
    for x, idx in parts:
        x = T(x)
        if x.dim() < 2:
            x = x.view(1, 1, 1)
        sel = x.index_select(1, idx) if idx is not None else x
        cnt += sel.shape[1]
        if how in ("sum", "mean", "value", "count"):
            vals.append(sel.sum(1, keepdim=True))
        elif how == "max":
            vals.append(sel.amax(1, keepdim=True))
        elif how == "min":
            vals.append(sel.amin(1, keepdim=True))
    if how == "count":
        return float(cnt)
    if not vals:
        return 0.0
    out = vals[0]
    for v in vals[1:]:
        if how in ("sum", "mean", "value"):
            out = out + v
        elif how == "max":
            out = torch.maximum(out, v)
        else:
            out = torch.minimum(out, v)
    if how == "mean":
        out = out / max(cnt, 1)
    return out
