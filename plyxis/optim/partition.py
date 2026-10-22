"""Decomposition of a twin into subsystems.

A subsystem is a set of instances whose decisions are mostly local (e.g. one
well pad with its wells and outgoing flowlines; one tank battery; one
compressor station). Subsystems interact only through flows on the edges that
cross their boundary - this is what price coordination and the separable
population searches exploit.

The partition is derived automatically from the netlist. A solver hint
(`partition = "pad"` in the sim block) names the entity type that anchors
gathering subsystems.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class Subsystem:
    sid: int
    name: str
    anchor: Optional[int]          # instance id that anchors it
    instances: list = field(default_factory=list)
    kind: str = ""


@dataclass
class Partition:
    subsystems: list
    of_instance: dict              # instance id -> subsystem id
    env_sid: int
    var_sid: Optional[torch.Tensor] = None   # flat decision variable -> subsystem id

    @property
    def K(self) -> int:
        return len(self.subsystems)

    def describe(self, flat) -> list:
        out = []
        for s in self.subsystems:
            types: dict = {}
            for iid in s.instances:
                t = flat.instances[iid].type.short
                types[t] = types.get(t, 0) + 1
            out.append({"id": s.sid, "name": s.name, "kind": s.kind, "size": len(s.instances), "types": types})
        return out


def build_partition(program, space=None, hint: Optional[str] = None) -> Partition:
    flat = program.flat
    hint = hint or flat.sim.solver.get("partition")
    n = len(flat.instances)
    succ = [[] for _ in range(n)]
    pred = [[] for _ in range(n)]
    for e in flat.edges:
        a, b = flat.ports[e.src].inst, flat.ports[e.dst].inst
        succ[a].append(b)
        pred[b].append(a)
    owner = [-1] * n
    subs: list = []

    def new(name, anchor, kind):
        s = Subsystem(len(subs), name, anchor, [], kind)
        subs.append(s)
        return s

    def claim(iid, s):
        if owner[iid] == -1:
            owner[iid] = s.sid
            s.instances.append(iid)

    # 1) anchored gathering subsystems: anchor + exclusive upstream tree + its outgoing links
    if hint:
        for inst in flat.instances:
            if inst.is_junction or not inst.type.is_a(str(hint)):
                continue
            s = new(inst.label or inst.path, inst.id, str(hint))
            claim(inst.id, s)
            stack = list(pred[inst.id])
            while stack:
                j = stack.pop()
                if owner[j] != -1:
                    continue
                if all(owner[k] == s.sid or k == inst.id for k in succ[j]):
                    claim(j, s)
                    stack.extend(pred[j])
            for j in succ[inst.id]:
                if flat.instances[j].is_link and len(pred[j]) == 1:
                    claim(j, s)
    # 2) every remaining non-link instance anchors its own subsystem (+ its exclusive outgoing links)
    for inst in flat.instances:
        if owner[inst.id] != -1 or inst.is_link or inst.is_junction:
            continue
        s = new(inst.label or inst.path, inst.id, inst.type.short)
        claim(inst.id, s)
        for j in succ[inst.id]:
            if flat.instances[j].is_link and all(owner[k] in (-1, s.sid) for k in pred[j]):
                claim(j, s)
    # 3) leftovers (links fed by several, junctions): attach to their upstream owner
    changed = True
    while changed:
        changed = False
        for inst in flat.instances:
            if owner[inst.id] != -1:
                continue
            cand = [owner[k] for k in pred[inst.id] if owner[k] != -1] or \
                   [owner[k] for k in succ[inst.id] if owner[k] != -1]
            if cand:
                claim(inst.id, subs[cand[0]])
                changed = True
    for inst in flat.instances:
        if owner[inst.id] == -1:
            claim(inst.id, new(inst.path, inst.id, inst.type.short))
    env = new("site-wide", None, "env")
    part = Partition(subs, {i: owner[i] for i in range(n)}, env.sid)
    if space is not None:
        part.var_sid = variable_subsystems(program, space, part)
    return part


def variable_subsystems(program, space, part: Partition) -> torch.Tensor:
    """Subsystem id of every flat decision variable."""
    vs = torch.full((space.N,), part.env_sid, dtype=torch.long)
    for b in space.blocks:
        g = program.gmap[b.gid]
        ids = torch.tensor([part.of_instance[i] for i in g.inst_ids], dtype=torch.long) if not g.is_env \
            else torch.tensor([part.env_sid])
        vs[b.offset:b.offset + b.size] = ids.view(-1, 1).expand(b.n, b.H).reshape(-1)
    if space.route is not None:
        r = space.route
        sid = torch.zeros(r.E, dtype=torch.long)
        for rp in program.routed:
            sid[rp.ridx] = part.of_instance[rp.inst]
        vs[r.offset:r.offset + r.size] = sid.view(-1, 1).expand(r.E, r.H).reshape(-1)
    return vs


def subsystem_fitness(problem, e, part: Partition, mu: float = 10.0) -> torch.Tensor:
    """Penalised objective attributed to each subsystem: [B, K]."""
    per_inst = problem.per_instance(e, mu)                 # [B, n_inst + 1]
    B = per_inst.shape[0]
    idx = torch.tensor([part.of_instance[i] for i in range(problem.n_inst - 1)] + [part.env_sid],
                       dtype=torch.long)
    return torch.zeros(B, part.K).index_add(1, idx, per_inst)
