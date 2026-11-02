"""Model lint: compiles fine, but will it do what the engineer thinks? Checks that catch the traps of real twins.

    python -m plyxis lint scenarios/corvus/field.plx

    no-effect         a decision that moves neither the objective nor any constraint (a port not connected, a typo)
    unused-property   a property no equation reads
    unconnected-input an input port without a connection: its flow is silently zero
    no-tag            an asset type with telemetry, an asset without a tag: no data will ever arrive
    unobservable      a property to calibrate on an asset without telemetry: nothing can correct it
    unpriced-soft     a soft constraint with price zero never binds
    terminal-value    a `final revenue` term: the plan can create value by parking material at the horizon end,
                      so check it is valued at cost (listed for review, not an error)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch

from .behavior import IR, walk_refs


@dataclass
class Finding:
    severity: str          # warning | info
    code: str
    type: str
    asset: Optional[str]
    message: str


def _walk_any(value, out: set) -> None:
    if isinstance(value, IR):
        walk_refs(value, out)
    elif isinstance(value, dict):
        for v in value.values():
            _walk_any(v, out)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _walk_any(v, out)
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        for k, v in vars(value).items():
            if k not in ("loc", "type"):
                _walk_any(v, out)


def _used_names(ir) -> set:
    refs: set = set()
    for part in (ir.nodes, ir.cons, ir.terms, ir.finals, ir.states, ir.decisions, ir.aggs):
        _walk_any(part, refs)
    return {name for _, name in refs}


def lint(model, gradient_points: int = 3, seed: int = 0) -> list:
    """All findings for a compiled model. The gradient check simulates the twin, so it needs a `sim`."""
    prog, flat = model.program, model.flat
    out: list = []
    groups = [g for g in prog.groups if not g.is_env]
    seen_types = set()
    for g in groups:
        ir, tname = g.ir, g.ir.type.short
        if tname not in seen_types:
            seen_types.add(tname)
            used = _used_names(ir)
            for pname, decl in ir.type.params.items():
                if pname not in used and pname not in ("length",) and not decl.is_input:
                    out.append(Finding("info", "unused-property", tname, None,
                                       f"{tname}.{pname}: no equation reads this property"))
            for ck, c in ir.cons.items():
                if getattr(c, "soft", None) is None:
                    continue
                w = prog._bounds_cache()[g.gid].get(f"soft:{ck}", None)
                if w is not None and float(torch.as_tensor(w, dtype=torch.float32).abs().max()) <= 0.0:
                    out.append(Finding("warning", "unpriced-soft", tname, None,
                                       f"{tname}.{ck.split(':', 1)[-1]}: soft constraint with price 0 never binds"))
            for key in ir.finals:
                if "revenue" in str(key):
                    out.append(Finding("info", "terminal-value", tname, None,
                                       f"{tname}: terminal value {str(key).split(':')[-1]}: check it is at cost, not at margin"))
        for li, iid in enumerate(g.inst_ids):
            inst = flat.instances[iid]
            if inst.is_junction or inst.is_link:
                continue
            for port in (flat.ports[p] if isinstance(p, int) else p for p in _ports(inst)):
                if port.direction == "in" and not port.edges and port.name in ir.type.ports:
                    out.append(Finding("warning", "unconnected-input", tname, inst.path,
                                       f"{inst.path}.{port.name}: input without a connection, its flow is zero"))
            if ir.telemetry and not inst.tag:
                out.append(Finding("warning", "no-tag", tname, inst.path, f"{inst.path}: has telemetry signals but no tag"))
            cal = [p for p, d in ir.type.params.items() if d.calibrate is not None]
            if cal and not ir.telemetry:
                out.append(Finding("info", "unobservable", tname, inst.path,
                                   f"{inst.path}: {', '.join(cal)} marked to calibrate, but this type has no telemetry"))
    if prog.flat.sim is not None and gradient_points > 0:
        out += _no_effect(model, gradient_points, seed)
    return _merge(out)


def _ports(inst) -> list:
    p = inst.ports
    return list(p.values()) if isinstance(p, dict) else list(p)


def _merge(findings: list) -> list:
    """Many instances with the same finding read as one line."""
    by: dict = {}
    for f in findings:
        by.setdefault((f.severity, f.code, f.type, f.asset is None), []).append(f)
    out = []
    for group in by.values():
        if len(group) > 3:
            f = group[0]
            out.append(Finding(f.severity, f.code, f.type, None, f"{f.message} (and {len(group) - 1} more assets of type {f.type})"))
        else:
            out += group
    out.sort(key=lambda f: (f.severity != "warning", f.code, f.type))
    return out


def _no_effect(model, points: int, seed: int) -> list:
    from ..optim.problem import Problem
    pb = Problem(model.program)
    sp = pb.space
    gen = torch.Generator().manual_seed(seed)
    seen = torch.zeros(sp.N, dtype=torch.bool)
    for k in range(points):
        x = (sp.default_u(1) if k == 0 else torch.rand(1, sp.N, generator=gen)).requires_grad_(True)
        e = pb.evaluate(x, relax=True)
        total = e.objective.sum() / pb.J_scale
        for h in e.resid.values():
            total = total + h.sum()
        g, = torch.autograd.grad(total, x, allow_unused=True)
        if g is not None:
            seen |= (g[0].abs() > 0.0)
    out = []
    for b in sp.blocks:
        grp = model.program.gmap[b.gid]
        if grp.is_env:
            continue
        u = seen[b.offset:b.offset + b.size].view(b.n, b.H).any(dim=1)
        dead = torch.nonzero(~u).reshape(-1).tolist()
        if len(dead) == b.n:                     # a modelling error: no instance of the type reacts
            out.append(Finding("warning", "no-effect", grp.ir.type.short, None,
                               f"{grp.ir.type.short}.{b.name}: this decision moves neither the objective nor any constraint"))
            continue
        for li in dead:                          # usually the start state (a closed stope, an empty tank)
            inst = model.flat.instances[grp.inst_ids[li]]
            out.append(Finding("info", "no-effect", grp.ir.type.short, inst.path,
                               f"{inst.path}.{b.name}: no effect from the model's start state"))
    return out


def as_dicts(findings: list) -> list:
    return [asdict(f) for f in findings]
