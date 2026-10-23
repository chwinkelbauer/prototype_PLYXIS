"""Connectors: how data enters PLYXIS.

A connector turns an external source into a stream of `Telemetry` / `Birth`
messages. The runtime only depends on this interface, so a pilot replaces the
emulator connector with an OPC UA, MQTT/Sparkplug B or historian (PI Web API)
client without touching estimation, optimisation or the GUI.

Included here:
  EmulatorConnector   - synthetic SCADA gateway in front of the plant emulator
  MarketConnector     - price feeds (crude, gas hub, power)
  HttpPushConnector   - external systems POST telemetry to /api/ingest
  CsvReplayConnector  - replay a historian export (t, tag, value[, unit])
"""
from __future__ import annotations

import csv
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from .emulator import Birth, Telemetry


@dataclass
class ConnectorStats:
    messages: int = 0
    last_t: Optional[float] = None
    last_wall: Optional[float] = None
    errors: int = 0
    rate: float = 0.0


class Connector:
    kind = "generic"

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.stats = ConnectorStats()
        self.status = "connected"
        self._window: deque = deque(maxlen=50)

    def _account(self, msgs: list, t: Optional[float]) -> list:
        n = len(msgs)
        self.stats.messages += n
        now = time.time()
        if n:
            self.stats.last_t = t
            self.stats.last_wall = now
        self._window.append((now, n))
        if len(self._window) > 1:
            span = self._window[-1][0] - self._window[0][0]
            self.stats.rate = sum(c for _, c in self._window) / span if span > 0 else 0.0
        return msgs

    def poll(self, t: float) -> list:
        return []

    def info(self) -> dict:
        return {"name": self.name, "kind": self.kind, "status": self.status, "description": self.description,
                "messages": self.stats.messages, "last_t": self.stats.last_t, "rate_per_s": round(self.stats.rate, 1),
                "errors": self.stats.errors}


class EmulatorConnector(Connector):
    kind = "scada-gateway"

    def __init__(self, name: str, description: str = "Synthetic SCADA gateway (OPC UA / MQTT stand-in)"):
        super().__init__(name, description)
        self.queue: deque = deque()

    def push(self, msgs: list) -> None:
        self.queue.extend(msgs)

    def poll(self, t: float) -> list:
        out = list(self.queue)
        self.queue.clear()
        return self._account(out, t)


class MarketConnector(Connector):
    kind = "market-data"

    def __init__(self, name: str, price_fn: Callable, tags: dict, description: str = "Market data feed"):
        super().__init__(name, description)
        self.price_fn = price_fn
        self.tags = tags          # env input name -> tag
        self.last: dict = {}

    def poll(self, t: float) -> list:
        out = []
        for name, tag in self.tags.items():
            v = self.price_fn(name, t)
            self.last[name] = v
            out.append(Telemetry(tag, float(v), None, t))
        return self._account(out, t)


class HttpPushConnector(Connector):
    kind = "http-push"

    def __init__(self, name: str = "http-ingest", description: str = "REST ingestion endpoint (POST /api/ingest)"):
        super().__init__(name, description)
        self.queue: deque = deque(maxlen=100000)

    def submit(self, records: list, t: float) -> int:
        n = 0
        for r in records:
            if "prefix" in r and "meta" in r:
                self.queue.append(Birth(r["prefix"], float(r.get("t", t)), dict(r["meta"])))
            else:
                self.queue.append(Telemetry(str(r["tag"]), float(r["value"]), r.get("unit"), float(r.get("t", t))))
            n += 1
        return n

    def poll(self, t: float) -> list:
        out = list(self.queue)
        self.queue.clear()
        return self._account(out, t)


class CsvReplayConnector(Connector):
    kind = "historian-replay"

    def __init__(self, name: str, path: str, description: str = "Historian export replay"):
        super().__init__(name, description)
        rows = []
        with open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                rows.append(Telemetry(r["tag"], float(r["value"]), r.get("unit") or None, float(r["t"])))
        rows.sort(key=lambda m: m.t)
        self.rows = deque(rows)

    def poll(self, t: float) -> list:
        out = []
        while self.rows and self.rows[0].t <= t:
            out.append(self.rows.popleft())
        return self._account(out, t)
