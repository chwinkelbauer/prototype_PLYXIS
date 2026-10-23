"""Twin-as-code version history.

The twin is PLX source. Every change - an engineer's edit in Twin Studio, a
calibration overlay written by the estimator, a discovered asset - is a commit
with author, message, sim time and a unified diff. Reality changes the code;
the code is always the single source of truth.
"""
from __future__ import annotations

import difflib
import time
from dataclasses import dataclass, field


@dataclass
class Commit:
    cid: str
    t_sim: float
    wall: float
    author: str
    message: str
    diff: str
    files: list
    stats: dict = field(default_factory=dict)

    def to_dict(self, with_diff: bool = False) -> dict:
        d = {"id": self.cid, "t": self.t_sim, "wall": self.wall, "author": self.author, "message": self.message,
             "files": self.files, "stats": self.stats}
        if with_diff:
            d["diff"] = self.diff
        return d


class TwinRepo:
    def __init__(self, files: dict):
        self.files = dict(files)
        self.commits: list = []
        self._commit({}, "initial twin", "engineering", 0.0, initial=True)

    def _commit(self, changes: dict, message: str, author: str, t_sim: float, initial: bool = False) -> Commit:
        diffs, names, add, rem = [], [], 0, 0
        for name, new in changes.items():
            old = self.files.get(name, "")
            if old == new:
                continue
            d = list(difflib.unified_diff(old.splitlines(), new.splitlines(), f"a/{name}", f"b/{name}", lineterm=""))
            add += sum(1 for l in d if l.startswith("+") and not l.startswith("+++"))
            rem += sum(1 for l in d if l.startswith("-") and not l.startswith("---"))
            diffs.append("\n".join(d))
            names.append(name)
            self.files[name] = new
        if not names and not initial:
            return None
        c = Commit(f"c{len(self.commits):04d}", t_sim, time.time(), author, message, "\n".join(diffs),
                   names or list(self.files), {"added": add, "removed": rem})
        self.commits.append(c)
        return c

    def commit(self, changes: dict, message: str, author: str, t_sim: float):
        return self._commit(changes, message, author, t_sim)

    def log(self, limit: int = 100) -> list:
        return [c.to_dict() for c in reversed(self.commits[-limit:])]

    def get(self, cid: str):
        return next((c for c in self.commits if c.cid == cid), None)
