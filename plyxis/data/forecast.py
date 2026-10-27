"""Forecasts of exogenous inputs (prices) over the planning horizon.

Deliberately simple and transparent: persistence for crude and gas, a learned
daily profile scaled to the current level for power. Announced events (e.g. a
pipeline notice that pushes the hub price negative for a day) are layered on top.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import torch


class Forecaster:
    def __init__(self, inputs: list, nominal: dict, profile_days: int = 7):
        self.inputs = list(inputs)
        self.nominal = dict(nominal)
        self.hist = {k: deque(maxlen=24 * profile_days * 6) for k in inputs}
        self.announced: dict = {}       # name -> list of (start, end, value)

    def observe(self, name: str, t: float, value: float) -> None:
        if name in self.hist:
            self.hist[name].append((t, value))

    def announce(self, name: str, start: float, end: float, value: float) -> None:
        self.announced.setdefault(name, []).append((start, end, value))

    def _announced_at(self, name: str, t: float) -> bool:
        return any(s <= t < e for s, e, _ in self.announced.get(name, []))

    def forecast(self, t0: float, T: int, dt: float) -> dict:
        out = {}
        hours = torch.tensor([t0 + i * dt for i in range(T)])
        for k in self.inputs:
            # an announced transient (a negative-price day, a scarcity hour) is not the new normal:
            # level and profile come from the observations it does not explain
            h = [(t, v) for t, v in self.hist[k] if not self._announced_at(k, t)] if k in self.announced \
                else self.hist[k]
            cur = h[-1][1] if h else self.nominal[k]
            if "power" in k and len(h) >= 24:
                prof = torch.zeros(24)
                cnt = torch.zeros(24)
                for t, v in h:
                    i = int(t % 24)
                    prof[i] += v
                    cnt[i] += 1
                prof = torch.where(cnt > 0, prof / cnt.clamp(min=1), torch.full((24,), float(cur)))
                level = sum(v for _, v in list(h)[-6:]) / max(1, len(list(h)[-6:]))
                hour_now = int(t0 % 24)
                scale = level / max(float(prof[hour_now]), 1e-6) if float(prof[hour_now]) > 1e-6 else 1.0
                scale = 0.5 * scale + 0.5           # mean-revert the deviation
                ser = prof[(hours % 24).long()] * scale
            else:
                ser = torch.full((T,), float(cur))
            for s, e, v in self.announced.get(k, []):
                m = (hours >= s) & (hours < e)
                ser = torch.where(m, torch.full_like(ser, v), ser)
            out[k] = ser
        return out
