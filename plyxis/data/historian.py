"""Historian exports: raw history as one compressed CSV, the input of backfill and backtests.

    t, kind, name, value, unit

    tag      name = tag                         a measurement
    test     name = tag                         a direct sample (well test, assay)
    control  name = path|decision               set point as the control system reports it
    route    name = source|port|dest|dest port  line-up split
    price    name = price name                  market price (canonical units)
    birth    name = tag prefix                  a device announced itself; `unit` holds its metadata as JSON

A pilot's historian replication produces the same file (tag mapping is code, see PLX `telemetry:`), so everything
downstream sees only what a customer's systems can deliver.
"""
from __future__ import annotations

import csv
import gzip
import json
from typing import Iterator, Optional

from .emulator import Birth, Telemetry


class HistoryWriter:
    def __init__(self, path: str):
        self.fh = gzip.open(path, "wt", newline="", encoding="utf-8")
        self.w = csv.writer(self.fh)
        self.w.writerow(["t", "kind", "name", "value", "unit"])

    def telemetry(self, msgs: list) -> None:
        for m in msgs:
            if isinstance(m, Birth):
                self.w.writerow([m.t, "birth", m.prefix, "", json.dumps(m.meta)])
            else:
                self.w.writerow([m.t, "test" if m.quality == "test" else "tag", m.tag, m.value, m.unit or ""])

    def controls(self, t: float, values: dict, routes: dict) -> None:
        for (path, name), v in values.items():
            self.w.writerow([t, "control", f"{path}|{name}", round(v, 6), ""])
        for key, v in routes.items():
            self.w.writerow([t, "route", "|".join(key), round(v, 6), ""])

    def prices(self, t: float, prices: dict) -> None:
        for name, v in prices.items():
            self.w.writerow([t, "price", name, v, ""])

    def close(self) -> None:
        self.fh.close()


class HistoryStep:
    __slots__ = ("t", "telemetry", "births", "controls", "routes", "prices")

    def __init__(self, t: float):
        self.t, self.telemetry, self.births, self.controls, self.routes, self.prices = t, [], [], {}, {}, {}


def read_history(path: str, last_seen: Optional[dict] = None) -> Iterator[HistoryStep]:
    """Steps in time order. A reading without a value (a gap in the export) is skipped. Controls and prices that are
    missing from a step keep their last value, as a historian's last-known-value query would return them."""
    controls: dict = {}
    routes: dict = {}
    prices: dict = {}
    step: Optional[HistoryStep] = None
    with gzip.open(path, "rt", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            t = float(r["t"])
            if step is None or t > step.t + 1e-9:
                if step is not None:
                    step.controls, step.routes, step.prices = dict(controls), dict(routes), dict(prices)
                    yield step
                step = HistoryStep(t)
            kind, name = r["kind"], r["name"]
            if kind == "birth":
                step.births.append(Birth(name, t, json.loads(r["unit"] or "{}")))
                continue
            if r["value"] == "":
                continue
            v = float(r["value"])
            if kind == "tag" or kind == "test":
                step.telemetry.append(Telemetry(name, v, r["unit"] or None, t, "test" if kind == "test" else "good"))
            elif kind == "control":
                path_, dec = name.split("|")
                controls[(path_, dec)] = v
            elif kind == "route":
                routes[tuple(name.split("|"))] = v
            elif kind == "price":
                prices[name] = v
    if step is not None:
        step.controls, step.routes, step.prices = dict(controls), dict(routes), dict(prices)
        yield step
