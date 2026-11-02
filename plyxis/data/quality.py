"""Data quality in front of the estimator: bad readings must not become twin state.

    stale         a tag that reported before has stopped reporting
    frozen        an analog signal repeats exactly (real instruments have noise; a stuck transmitter does not)
    out of range  a reading outside the physical limits declared in PLX, beyond any instrument noise

Flagged readings stay visible in the tag list but do not reach estimation or anomaly detection; the twin keeps
predicting that signal from its model. A reading is released again as soon as the signal behaves.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


STATUS = (":up", ":avail", ":running")


@dataclass
class TagQuality:
    last: float = math.nan
    last_t: float = -math.inf
    repeats: int = 0
    issue: Optional[str] = None
    since: float = 0.0


class QualityMonitor:
    def __init__(self, tagmap, limits: Optional[dict] = None, frozen_after: int = 12, stale_after_h: float = 1.0,
                 tolerance: float = 0.2):
        self.tags = tagmap
        self.limits = limits or {}          # (gid, state) -> constant lower limit from the model
        self.frozen_after = frozen_after    # samples; 12 at the data step of 10 minutes = 2 h
        self.stale_after_h = stale_after_h
        self.tolerance = tolerance          # relative margin around a limit before a reading counts as impossible
        self.state: dict = {}
        self.flagged_total = 0
        self.events: list = []              # (t, tag, issue) on every new flag

    def _flag(self, q: TagQuality, tag: str, issue: str, t: float) -> None:
        if q.issue != issue:
            q.issue, q.since = issue, t
            self.events.append((t, tag, issue))

    def filter(self, t: float, telemetry: list) -> list:
        """The readings the estimator may use. Direct samples (`test`) pass: they have their own error model."""
        out = []
        for m in telemetry:
            info = self.tags.get(m.tag)
            if info is None or m.quality == "test":
                out.append(m)
                continue
            q = self.state.setdefault(m.tag, TagQuality())
            if m.quality == "bad" or m.value != m.value:     # the source itself says so (PI, OPC UA, Sparkplug null)
                q.last_t = t
                self._flag(q, m.tag, "bad at the source", t)
                self.flagged_total += 1
                m.quality = "bad"
                continue
            analog = not info.key.endswith(STATUS)
            q.repeats = q.repeats + 1 if (analog and m.value == q.last and m.value != 0.0) else 0
            q.last, q.last_t = m.value, t
            bad = None
            if q.repeats >= self.frozen_after:
                bad = "frozen"
            elif self._out_of_range(info, m.value):
                bad = "out of range"
            if bad is None:
                q.issue = None
                out.append(m)
                continue
            self._flag(q, m.tag, bad, t)
            self.flagged_total += 1
            m.quality = "bad"
        for tag, q in self.state.items():
            if q.issue is None and t - q.last_t > self.stale_after_h:
                self._flag(q, tag, "stale", t)
            elif q.issue == "stale" and t - q.last_t <= self.stale_after_h:
                q.issue = None
        return out

    def _out_of_range(self, info, value: float) -> bool:
        if not info.key.startswith("state:"):
            return False
        lo = self.limits.get((info.gid, info.key.split(":", 1)[1]))
        if lo is None:
            return False
        from ..lang import units as U
        v = value
        if info.unit:
            try:
                v = U.to_canonical(value, info.unit)
            except Exception:  # noqa: BLE001
                return False
        return v < lo - self.tolerance * max(abs(lo), 1.0)

    def drain(self) -> list:
        ev, self.events = self.events, []
        return ev

    def bad_tags(self) -> set:
        return {t for t, q in list(self.state.items()) if q.issue}

    def summary(self) -> dict:
        issues = [{"tag": t, "issue": q.issue, "since": q.since} for t, q in list(self.state.items()) if q.issue]
        issues.sort(key=lambda i: i["since"], reverse=True)
        return {"tags": len(self.state), "bad_now": len(issues), "flagged_total": self.flagged_total, "issues": issues[:200]}
