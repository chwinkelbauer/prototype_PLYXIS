"""Connectors to a customer's plant systems (read only; pilot phases 1-3, docs/PILOT_PLAN_OIL_GAS.md section 5).

    PiWebApiConnector   OSIsoft / AVEVA PI Web API: polls `streamsets/value` over HTTPS
    SparkplugConnector  MQTT Sparkplug B: subscribes to a group; births announce devices, deaths mark them gone
    OpcUaConnector      OPC UA: subscribes to nodes, needs the optional `asyncua` package

They produce the same `Telemetry` / `Birth` messages as the emulated gateway, with the source's quality flag, so
nothing downstream changes. Standard library only, except OPC UA (its binary protocol is not worth re-implementing).
What has been tested against what is stated in docs/DATA_INTEGRATION.md: none of them has met a real plant yet.
"""
from __future__ import annotations

import base64
import json
import socket
import ssl
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from typing import Callable, Optional

from .connectors import Connector
from .emulator import Birth, Telemetry


class ConnectorUnavailable(RuntimeError):
    pass


# ----------------------------------------------------------------------------- PI Web API
class PiWebApiConnector(Connector):
    """`tags` maps the PLX tag to the PI point's WebId; `units` optionally gives the tag's unit."""
    kind = "pi-web-api"

    def __init__(self, name: str, base_url: str, tags: dict, username: Optional[str] = None, password: Optional[str] = None,
                 period_s: float = 10.0, timeout_s: float = 10.0, units: Optional[dict] = None,
                 verify_tls: bool = True, clock: Callable = time.monotonic,
                 description: str = "PI Web API, read only"):
        super().__init__(name, description)
        self.base = base_url.rstrip("/")
        self.tags, self.units = dict(tags), units or {}
        self.by_webid = {v: k for k, v in tags.items()}
        self.period, self.timeout = period_s, timeout_s
        self.clock, self.last_poll = clock, -float("inf")
        self.headers = {"Accept": "application/json"}
        if username is not None:
            token = base64.b64encode(f"{username}:{password or ''}".encode()).decode()
            self.headers["Authorization"] = f"Basic {token}"
        self.ctx = None if verify_tls else ssl._create_unverified_context()   # only for a lab server with a private CA

    def _url(self) -> str:
        return f"{self.base}/streamsets/value?" + urllib.parse.urlencode([("webId", w) for w in self.tags.values()])

    def poll(self, t: float) -> list:
        now = self.clock()
        if now - self.last_poll < self.period or not self.tags:
            return self._account([], t)
        self.last_poll = now
        try:
            req = urllib.request.Request(self._url(), headers=self.headers)
            with urllib.request.urlopen(req, timeout=self.timeout, context=self.ctx) as resp:
                items = json.loads(resp.read().decode("utf-8")).get("Items", [])
            self.status = "connected"
        except (urllib.error.URLError, OSError, ValueError) as ex:
            self.status = f"error: {ex}"
            self.stats.errors += 1
            return self._account([], t)
        out = []
        for it in items:
            tag = self.by_webid.get(it.get("WebId"))
            val = it.get("Value") or {}
            v = val.get("Value")
            if isinstance(v, dict):         # digital state: {"Name": ..., "Value": n}
                v = v.get("Value")
            if tag is None or not isinstance(v, (int, float)) or isinstance(v, bool):
                if tag is not None:
                    self.stats.errors += 1
                continue
            good = val.get("Good", True) and not val.get("Questionable", False)
            out.append(Telemetry(tag, float(v), self.units.get(tag) or val.get("UnitsAbbreviation") or None, t,
                                 "good" if good else "bad"))
        return self._account(out, t)


# ----------------------------------------------------------------------------- Sparkplug B payload (protobuf wire format)
def _varint(b: bytes, i: int) -> tuple:
    shift, out = 0, 0
    while True:
        c = b[i]
        i += 1
        out |= (c & 0x7f) << shift
        if not c & 0x80:
            return out, i
        shift += 7


def _fields(b: bytes):
    i = 0
    while i < len(b):
        key, i = _varint(b, i)
        num, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 1:
            v, i = b[i:i + 8], i + 8
        elif wt == 2:
            n, i = _varint(b, i)
            v, i = b[i:i + n], i + n
        elif wt == 5:
            v, i = b[i:i + 4], i + 4
        else:
            raise ValueError(f"unsupported wire type {wt}")
        yield num, v


def _signed(v: int, bits: int) -> int:
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


# Sparkplug B data types of the metric (specification, table of data types)
_INT = {1: 8, 2: 16, 3: 32, 4: 64}          # Int8..Int64 are carried as two's complement in int/long_value


def decode_sparkplug(payload: bytes) -> dict:
    """{timestamp, seq, metrics: [{name, alias, timestamp, datatype, value, is_null}]}. Metrics of a type without a
    numeric value (strings, datasets, templates) come back with value None."""
    out: dict = {"timestamp": None, "seq": None, "metrics": []}
    for num, v in _fields(payload):
        if num == 1:
            out["timestamp"] = v
        elif num == 3:
            out["seq"] = v
        elif num == 2:
            m: dict = {"name": None, "alias": None, "timestamp": None, "datatype": None, "value": None, "is_null": False}
            raw = None
            for n2, v2 in _fields(v):
                if n2 == 1:
                    m["name"] = v2.decode("utf-8")
                elif n2 == 2:
                    m["alias"] = v2
                elif n2 == 3:
                    m["timestamp"] = v2
                elif n2 == 4:
                    m["datatype"] = v2
                elif n2 == 7:
                    m["is_null"] = bool(v2)
                elif n2 in (10, 11):
                    raw = v2
                elif n2 == 12:
                    m["value"] = struct.unpack("<f", v2)[0]
                elif n2 == 13:
                    m["value"] = struct.unpack("<d", v2)[0]
                elif n2 == 14:
                    m["value"] = float(bool(v2))
            if raw is not None:
                bits = _INT.get(m["datatype"])
                m["value"] = float(_signed(raw, bits) if bits else raw)
            out["metrics"].append(m)
    return out


# ----------------------------------------------------------------------------- MQTT 3.1.1 (QoS 0 subscriber)
def _remaining(n: int) -> bytes:
    out = bytearray()
    while True:
        d = n % 128
        n //= 128
        out.append(d | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _mqtt_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack(">H", len(b)) + b


class SparkplugConnector(Connector):
    """Subscribes to spBv1.0/<group>/#. Tag = <prefix>.<metric with '/' as '.'>, prefix from `prefix_of(edge, device)`."""
    kind = "mqtt-sparkplug-b"

    def __init__(self, name: str, host: str, port: int = 1883, group: str = "+", client_id: str = "plyxis-read",
                 username: Optional[str] = None, password: Optional[str] = None, keepalive_s: int = 30,
                 prefix_of: Optional[Callable] = None, units: Optional[dict] = None, reconnect_s: float = 5.0,
                 description: str = "MQTT Sparkplug B, read only"):
        super().__init__(name, description)
        self.host, self.port, self.group, self.client_id = host, port, group, client_id
        self.username, self.password, self.keepalive = username, password, keepalive_s
        self.prefix_of = prefix_of or (lambda edge, device: device or edge)
        self.units = units or {}
        self.reconnect_s = reconnect_s
        self.queue: deque = deque(maxlen=200000)
        self.aliases: dict = {}             # (edge, device) -> {alias: metric name}
        self.seq: dict = {}                 # edge -> last sequence number
        self.dead: set = set()              # (edge, device) that sent a death certificate
        self._stop = False
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self.status = "starting"

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name=f"sparkplug-{self.name}", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop = True
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass

    def poll(self, t: float) -> list:
        self.start()
        out = []
        while self.queue:
            out.append(self.queue.popleft())
        for m in out:
            m.t = t
        return self._account(out, t)

    # -------------------------------------------------------------- socket loop
    def _run(self) -> None:
        while not self._stop:
            try:
                self._session()
            except (OSError, ValueError, struct.error) as ex:
                if self._stop:
                    return
                self.status = f"disconnected: {ex}"
                self.stats.errors += 1
            time.sleep(self.reconnect_s)

    def _read(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise OSError("connection closed by the broker")
            buf += chunk
        return buf

    def _packet(self) -> tuple:
        head = self._read(1)[0]
        n, mult = 0, 1
        while True:
            c = self._read(1)[0]
            n += (c & 0x7f) * mult
            mult *= 128
            if not c & 0x80:
                break
        return head >> 4, self._read(n)

    def _session(self) -> None:
        self._sock = socket.create_connection((self.host, self.port), timeout=10.0)
        flags = 0x02 | (0x80 if self.username else 0) | (0x40 if self.password else 0)      # clean session
        body = _mqtt_str("MQTT") + bytes([4, flags]) + struct.pack(">H", self.keepalive) + _mqtt_str(self.client_id)
        if self.username:
            body += _mqtt_str(self.username)
        if self.password:
            body += _mqtt_str(self.password)
        self._sock.sendall(b"\x10" + _remaining(len(body)) + body)
        kind, data = self._packet()
        if kind != 2 or len(data) < 2 or data[1] != 0:
            raise OSError(f"broker refused the connection (code {data[1] if len(data) > 1 else '?'})")
        topic = f"spBv1.0/{self.group}/#"
        sub = struct.pack(">H", 1) + _mqtt_str(topic) + b"\x00"
        self._sock.sendall(b"\x82" + _remaining(len(sub)) + sub)
        self.status = "connected"
        self._sock.settimeout(max(1.0, self.keepalive / 2.0))
        last_ping = time.monotonic()
        while not self._stop:
            try:
                kind, data = self._packet()
            except socket.timeout:
                kind, data = None, b""
            if time.monotonic() - last_ping > self.keepalive / 2.0:
                self._sock.sendall(b"\xc0\x00")
                last_ping = time.monotonic()
            if kind == 3:
                n = struct.unpack(">H", data[:2])[0]
                self.on_message(data[2:2 + n].decode("utf-8"), data[2 + n:])

    # -------------------------------------------------------------- Sparkplug
    def on_message(self, topic: str, payload: bytes) -> None:
        parts = topic.split("/")
        if len(parts) < 4 or parts[0] != "spBv1.0":
            return
        kind, edge = parts[2], parts[3]
        device = parts[4] if len(parts) > 4 else None
        node = (edge, device)
        if kind in ("NCMD", "DCMD", "STATE"):
            return                           # commands are not ours: this connector never writes
        if kind in ("NDEATH", "DDEATH"):
            self.dead.add(node)
            if kind == "NDEATH":
                self.dead |= {k for k in self.aliases if k[0] == edge}
            return
        try:
            p = decode_sparkplug(payload)
        except (ValueError, IndexError, struct.error):
            self.stats.errors += 1
            return
        if p["seq"] is not None:             # a gap means lost messages: count it, the quality monitor sees the effect
            prev = self.seq.get(edge)
            if prev is not None and kind not in ("NBIRTH",) and p["seq"] != (prev + 1) % 256:
                self.stats.errors += 1
            self.seq[edge] = p["seq"]
        prefix = self.prefix_of(edge, device)
        alias = self.aliases.setdefault(node, {})
        if kind in ("NBIRTH", "DBIRTH"):
            self.dead.discard(node)
            alias.clear()
            self.queue.append(Birth(prefix, 0.0, {"edge": edge, "device": device, "type": "sparkplug", "label": prefix}))
        if kind not in ("NBIRTH", "DBIRTH", "NDATA", "DDATA"):
            return
        for m in p["metrics"]:
            if m["name"] and m["alias"] is not None:
                alias[m["alias"]] = m["name"]
            name = m["name"] or alias.get(m["alias"])
            if name is None:                 # data before its birth certificate: cannot be mapped
                self.stats.errors += 1
                continue
            if name.startswith("bdSeq") or name.startswith("Node Control/"):
                continue
            tag = f"{prefix}.{name.replace('/', '.')}"
            if m["is_null"]:
                self.queue.append(Telemetry(tag, float("nan"), self.units.get(tag), 0.0, "bad"))
            elif m["value"] is not None:
                self.queue.append(Telemetry(tag, float(m["value"]), self.units.get(tag), 0.0))


# ----------------------------------------------------------------------------- OPC UA
class OpcUaConnector(Connector):
    """`nodes` maps the PLX tag to the node id (e.g. 'ns=2;s=PAD07.W03.FLOW'). Polls the values through asyncua's
    synchronous client; subscriptions are the next step once a real server is available to test against."""
    kind = "opc-ua"

    def __init__(self, name: str, url: str, nodes: dict, units: Optional[dict] = None, period_s: float = 10.0,
                 username: Optional[str] = None, password: Optional[str] = None, clock: Callable = time.monotonic,
                 description: str = "OPC UA client, read only"):
        super().__init__(name, description)
        try:
            from asyncua.sync import Client
        except ImportError as ex:
            raise ConnectorUnavailable("OPC UA needs the optional package: pip install asyncua") from ex
        self.client = Client(url)
        if username:
            self.client.set_user(username)
            self.client.set_password(password or "")
        self.tags, self.units, self.period, self.clock = dict(nodes), units or {}, period_s, clock
        self.last_poll, self.handles = -float("inf"), None

    def poll(self, t: float) -> list:
        now = self.clock()
        if now - self.last_poll < self.period:
            return self._account([], t)
        self.last_poll = now
        out = []
        try:
            if self.handles is None:
                self.client.connect()
                self.handles = {tag: self.client.get_node(nid) for tag, nid in self.tags.items()}
            for tag, node in self.handles.items():
                dv = node.read_data_value()
                good = dv.StatusCode is None or dv.StatusCode.is_good()
                v = dv.Value.Value if dv.Value is not None else None
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out.append(Telemetry(tag, float(v), self.units.get(tag), t, "good" if good else "bad"))
            self.status = "connected"
        except Exception as ex:  # noqa: BLE001 - the library raises many kinds; a lost server must not stop the plant loop
            self.status = f"error: {ex}"
            self.stats.errors += 1
            self.handles = None
        return self._account(out, t)


def build_connector(cfg: dict) -> Connector:
    """From a scenario's `connectors` list: {"type": "pi-web-api" | "sparkplug" | "opc-ua", "name": ..., ...}."""
    cfg = dict(cfg)
    kind = cfg.pop("type")
    if kind == "pi-web-api":
        return PiWebApiConnector(**cfg)
    if kind == "sparkplug":
        return SparkplugConnector(**cfg)
    if kind == "opc-ua":
        return OpcUaConnector(**cfg)
    raise ValueError(f"unknown connector type {kind}")
