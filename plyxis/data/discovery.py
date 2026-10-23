"""Topology discovery: unknown telemetry -> proposed twin code.

Discovery rules live in the environment's `discover:` section:

    rule new_well: on "KST.P{pad}.W{well}.FLOW.LIQ" {
      init P{pad}_W{well} as esp_well { tag = "KST.P{pad}.W{well}"; };
      connect(P{pad}_W{well}.prod, PAD{pad}.wells);
    };

When a tag matches a rule and the asset is not in the twin yet, the rule body is
instantiated with the captured names and printed as ordinary PLX. Metadata from the
device's birth certificate (coordinates, type) is merged in. The proposal can be
approved by an engineer or applied automatically - either way it becomes a commit
in the twin's version history.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..lang import ast as A
from ..lang import printer
from ..lang.consteval import Scope


@dataclass
class Proposal:
    pid: str
    rule: str
    tag: str
    bindings: dict
    code: str
    assets: list
    t: float
    status: str = "pending"          # pending | applied | rejected
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.pid, "rule": self.rule, "tag": self.tag, "bindings": self.bindings, "code": self.code,
                "assets": self.assets, "t": self.t, "status": self.status, "meta": self.meta}


def _pattern(p: str) -> re.Pattern:
    out, i = "", 0
    for m in re.finditer(r"\{(\w+)\}", p):
        out += re.escape(p[i:m.start()]) + f"(?P<{m.group(1)}>[A-Za-z0-9_]+?)"
        i = m.end()
    out += re.escape(p[i:])
    return re.compile("^" + out + "$")


class Discovery:
    def __init__(self, rules: list, known_paths: set, tag_prefixes: set):
        self.rules = [(r, _pattern(r.pattern)) for r in rules]
        self.known = set(known_paths)
        self.prefixes = set(tag_prefixes)
        self.proposals: dict = {}
        self.seen_unknown: set = set()
        self.births: dict = {}

    def note_birth(self, prefix: str, meta: dict) -> None:
        self.births[prefix] = meta

    def observe(self, tag: str, t: float) -> Optional[Proposal]:
        if any(tag.startswith(p + ".") for p in self.prefixes):
            return None
        if tag in self.seen_unknown:
            return None
        self.seen_unknown.add(tag)
        for rule, rx in self.rules:
            m = rx.match(tag)
            if not m:
                continue
            bind = m.groupdict()
            scope = Scope(dict(bind))
            stmts = []
            assets = []
            prefix = None
            for st in rule.body:
                if isinstance(st, A.InitStmt):
                    name = printer.expr(st.name, scope)
                    if name in self.known:
                        return None
                    assets.append(name)
                    over = list(st.overrides)
                    names = {o.name for o in over}
                    for o in over:
                        if o.name == "tag":
                            prefix = printer.expr(o.value, scope).strip('"')
                    meta = self.births.get(prefix or "", {})
                    if "at" not in names and meta.get("lat") is not None:
                        over.append(A.Override("at", A.TupleExpr([A.Num(round(meta["lat"], 6)),
                                                                  A.Num(round(meta["lon"], 6))])))
                    st = A.InitStmt(st.name, st.type, over, loc=st.loc)
                stmts.append(printer.stmt(st, 4, scope))
            pid = f"D{len(self.proposals) + 1:03d}"
            code = f"    ## discovered from tag {tag} (rule '{rule.name}')\n" + "\n".join(stmts)
            p = Proposal(pid, rule.name, tag, bind, code, assets, t,
                         meta=self.births.get(prefix or "", {}))
            self.proposals[pid] = p
            if prefix:
                self.prefixes.add(prefix)
            return p
        return None

    def mark_known(self, paths: list, prefixes: list) -> None:
        self.known.update(paths)
        self.prefixes.update(prefixes)
