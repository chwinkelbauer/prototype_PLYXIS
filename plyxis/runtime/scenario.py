"""Scenario definition: twin sources, hidden truth, prices, scripted events, guided demo."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Scenario:
    root: str
    name: str
    description: str
    twin_files: list
    twin_env: str
    twin_sim: Optional[str]
    truth_files: list
    truth_env: str
    hidden: list
    dt_emu: float
    replan_every: float
    calibrate_every: float
    start_hour: float
    end_hour: float
    prices: dict
    events: list
    legacy_cfg: dict
    demo: list
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, root: str) -> "Scenario":
        with open(os.path.join(root, "scenario.json"), encoding="utf-8") as fh:
            d = json.load(fh)
        tw, tr, tm = d["twin"], d["truth"], d["time"]
        return cls(root=root, name=d["name"], description=d.get("description", ""),
                   twin_files=[os.path.join(root, f) for f in tw["files"]], twin_env=tw["env"], twin_sim=tw.get("sim"),
                   truth_files=[os.path.join(root, f) for f in tr["files"]], truth_env=tr["env"],
                   hidden=tr.get("hidden", []), dt_emu=tm["dt_emulator_h"], replan_every=tm["replan_every_h"],
                   calibrate_every=tm["calibrate_every_h"], start_hour=tm.get("start_hour", 0.0),
                   end_hour=tm.get("end_hour", 1e9),
                   prices=d.get("prices", {}), events=sorted(d.get("events", []), key=lambda e: e["t"]),
                   legacy_cfg=d.get("legacy", {}), demo=d.get("demo", []), raw=d)

    def legacy_class(self):
        path = os.path.join(self.root, "legacy.py")
        if not os.path.exists(path):
            return None
        spec = importlib.util.spec_from_file_location("scenario_legacy", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "LegacyOperators", None)


class PriceModel:
    """Realised market prices (deterministic for a given seed). Returned in the units of the
    scenario definition; the emulator converts to canonical units."""

    def __init__(self, cfg: dict, seed: int = 11, start_hour: float = 0.0):
        from ..lang import units as U
        self.cfg = cfg
        self.U = U
        self.start = start_hour
        self.rng = random.Random(seed)
        self.cache: dict = {}
        self.walk: dict = {k: 0.0 for k in cfg}
        self.last_hour = {k: -1 for k in cfg}

    def display(self, name: str, t: float) -> float:
        c = self.cfg.get(name)
        if c is None:
            return 0.0
        h = int(math.floor(t))
        key = (name, h)
        if key not in self.cache:
            while self.last_hour[name] < h:
                self.last_hour[name] += 1
                self.walk[name] = 0.98 * self.walk[name] + self.rng.gauss(0, c.get("walk", 0.0))
                hh = self.last_hour[name]
                v = c["base"] + self.walk[name]
                if "daily" in c:
                    v = c["base"] * c["daily"][int((hh + self.start) % 24)] * (1 + 0.15 * self.walk[name])
                    if self.rng.random() < c.get("spike_prob", 0.0):
                        v *= 4.0
                self.cache[(name, hh)] = v
        return self.cache[key]

    def canonical(self, name: str, t: float) -> float:
        c = self.cfg.get(name, {})
        v = self.display(name, t)
        u = c.get("unit")
        return self.U.to_canonical(v, u) if u else v

    def unit(self, name: str) -> Optional[str]:
        return self.cfg.get(name, {}).get("unit")
