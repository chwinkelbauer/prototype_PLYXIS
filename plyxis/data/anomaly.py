"""Anomaly detection on telemetry - innovations against the twin, thresholds, trends, status changes.

* model mismatch: EWMA of normalised innovations (measured - twin prediction)
* thresholds:     telemetry options `warn` / `alarm` declared in PLX
* trends:         least-squares slope over the recent window -> predicted time to alarm
                  (predictive maintenance: 'ESP failure expected in ~9 h')
* status:         run/availability transitions (trips, failures, restarts)
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Alert:
    t: float
    severity: str          # info | warning | critical
    kind: str              # mismatch | threshold | trend | status
    asset: str
    label: str
    tag: str
    message: str
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"t": self.t, "severity": self.severity, "kind": self.kind, "asset": self.asset, "label": self.label,
                "tag": self.tag, "message": self.message, "data": self.data}


SIGNAL_NAMES = {"vib": "vibration", "p_inj": "injection pressure", "liquid": "liquid rate", "gas_in": "gas rate",
                "motor_kw": "motor load", "oil_lvl": "oil tank level", "wat_lvl": "water tank level"}


def _sig(info) -> str:
    name = info.key.split(":")[-1]
    return SIGNAL_NAMES.get(name, name.replace("_", " "))


class AnomalyDetector:
    def __init__(self, tagmap, flat, window: int = 18, z_alarm: float = 4.0, ewma: float = 0.3):
        self.tags = tagmap
        self.flat = flat
        self.window = window
        self.z_alarm = z_alarm
        self.alpha = ewma
        self.ewma: dict = {}
        self.hist: dict = {}
        self.active: dict = {}           # (tag, kind) -> t raised
        self.predicted: dict = {}       # asset path -> (t, hours to alarm) from significant trends
        self.last_status: dict = {}

    def _label(self, path: str) -> str:
        iid = self.flat.by_path.get(path)
        if iid is None:
            return path
        inst = self.flat.instances[iid]
        return inst.label or path

    def update(self, t: float, telemetry: list, predictions: dict) -> list:
        alerts = []
        for m in telemetry:
            info = self.tags.get(m.tag)
            if info is None or m.quality == "test":
                continue
            v = m.value
            h = self.hist.setdefault(m.tag, deque(maxlen=self.window))
            h.append((t, v))
            kind = info.key
            label = self._label(info.path)
            # status transitions
            if kind.endswith(":up") or kind.endswith(":avail"):
                prev = self.last_status.get(m.tag)
                self.last_status[m.tag] = v
                if prev is not None and abs(prev - v) > 0.5:
                    if v < prev:
                        what = "tripped" if kind.endswith(":avail") else "stopped / failed"
                        alerts.append(Alert(t, "critical", "status", info.path, label, m.tag,
                                            f"{label}: {what} ({prev:.0f} -> {v:.0f})",
                                            {"from": prev, "to": v}))
                    else:
                        alerts.append(Alert(t, "info", "status", info.path, label, m.tag,
                                            f"{label}: back in service ({prev:.0f} -> {v:.0f})", {"from": prev, "to": v}))
                continue
            # thresholds
            warn, alarm = info.opts.get("warn"), info.opts.get("alarm")
            if alarm is not None and v >= alarm and (m.tag, "alarm") not in self.active:
                self.active[(m.tag, "alarm")] = t
                alerts.append(Alert(t, "critical", "threshold", info.path, label, m.tag,
                                    f"{label}: {_sig(info)} {v:.1f} above alarm level {alarm}", {"value": v}))
            elif warn is not None and v >= warn and (m.tag, "warn") not in self.active:
                self.active[(m.tag, "warn")] = t
                alerts.append(Alert(t, "warning", "threshold", info.path, label, m.tag,
                                    f"{label}: {_sig(info)} {v:.1f} above warning level {warn}", {"value": v}))
            # trends -> predicted time to alarm
            if alarm is not None and len(h) >= max(6, self.window // 2) and (m.tag, "trend") not in self.active:
                ts = [p[0] for p in h]
                vs = [p[1] for p in h]
                n = len(ts)
                mt, mv = sum(ts) / n, sum(vs) / n
                var = sum((x - mt) ** 2 for x in ts)
                if var > 1e-9:
                    slope = sum((x - mt) * (y - mv) for x, y in zip(ts, vs)) / var
                    res = [y - (mv + slope * (x - mt)) for x, y in zip(ts, vs)]
                    s_res = math.sqrt(sum(r * r for r in res) / max(1, n - 2))
                    t_stat = slope * math.sqrt(var) / max(s_res, 1e-9)
                    level = warn if warn is not None else alarm * 0.6
                    if slope > 1e-6 and v < alarm and t_stat > 6.0 and mv > 0.85 * level:
                        eta = (alarm - v) / slope
                        if eta < 48:
                            self.active[(m.tag, "trend")] = t
                            self.predicted[info.path] = (t, eta)
                            alerts.append(Alert(t, "warning", "trend", info.path, label, m.tag,
                                                f"{label}: rising {_sig(info)} - alarm level expected in "
                                                f"~{eta:.0f} h (predictive maintenance)",
                                                {"slope_per_h": slope, "eta_h": eta}))
            # innovations vs twin
            p = predictions.get((info.gid, info.li, info.key))
            if p is not None:
                from ..lang import units as U
                pv = p
                if info.unit:
                    try:
                        pv = U.from_canonical(p, info.unit)
                    except Exception:  # noqa: BLE001
                        pass
                sigma = 0.03 * abs(v) + 0.01 * (abs(pv) + 1.0)
                z = (v - pv) / sigma
                e = self.ewma.get(m.tag, 0.0) * (1 - self.alpha) + self.alpha * z
                self.ewma[m.tag] = e
                key = (m.tag, "mismatch")
                if abs(e) > self.z_alarm and key not in self.active:
                    self.active[key] = t
                    alerts.append(Alert(t, "warning", "mismatch", info.path, label, m.tag,
                                        f"{label}: measured {v:,.1f} vs twin {pv:,.1f} ({m.tag.split('.', 2)[-1]}) - "
                                        "twin recalibration triggered", {"measured": v, "twin": pv, "z": e}))
                elif abs(e) < 1.0 and key in self.active:
                    del self.active[key]
        return alerts

    def clear(self, tag_prefix: str) -> None:
        for k in [k for k in self.active if k[0].startswith(tag_prefix)]:
            del self.active[k]
