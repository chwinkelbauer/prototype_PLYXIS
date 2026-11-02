"""Industrial connectors against local protocol stand-ins (no plant systems involved), historian export and replay."""
import json
import math
import os
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from plyxis.data.emulator import Birth, Telemetry
from plyxis.data.historian import HistoryWriter, read_history
from plyxis.data.industrial import ConnectorUnavailable, OpcUaConnector, PiWebApiConnector, SparkplugConnector, build_connector, decode_sparkplug


# ------------------------------------------------------------------ PI Web API
class _Pi(BaseHTTPRequestHandler):
    seen: list = []
    fail = False

    def log_message(self, *a):
        pass

    def do_GET(self):
        _Pi.seen.append((self.path, self.headers.get("Authorization")))
        if _Pi.fail:
            self.send_response(503)
            self.end_headers()
            return
        body = {"Items": [
            {"WebId": "W1", "Name": "FLOW", "Value": {"Value": 12.5, "UnitsAbbreviation": "bbl/d", "Good": True, "Questionable": False}},
            {"WebId": "W2", "Name": "LVL", "Value": {"Value": 7.0, "Good": False}},
            {"WebId": "W3", "Name": "RUN", "Value": {"Value": {"Name": "On", "Value": 1}, "Good": True}},
            {"WebId": "W4", "Name": "TXT", "Value": {"Value": "n/a", "Good": True}},
        ]}
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def pi_server():
    srv = HTTPServer(("127.0.0.1", 0), _Pi)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _Pi.seen, _Pi.fail = [], False
    yield f"http://127.0.0.1:{srv.server_port}/piwebapi"
    srv.shutdown()


def test_pi_web_api_reads_values_quality_and_digital_states(pi_server):
    c = PiWebApiConnector("pi", pi_server, {"A.FLOW": "W1", "A.LVL": "W2", "A.RUN": "W3", "A.TXT": "W4"}, username="u", password="p", period_s=0.0)
    msgs = {m.tag: m for m in c.poll(5.0)}
    assert msgs["A.FLOW"].value == 12.5 and msgs["A.FLOW"].unit == "bbl/d" and msgs["A.FLOW"].quality == "good" and msgs["A.FLOW"].t == 5.0
    assert msgs["A.LVL"].quality == "bad"
    assert msgs["A.RUN"].value == 1.0
    assert "A.TXT" not in msgs and c.stats.errors == 1
    path, auth = _Pi.seen[0]
    assert path.startswith("/piwebapi/streamsets/value?webId=W1") and auth.startswith("Basic ")


def test_pi_web_api_survives_a_server_error_and_respects_its_period(pi_server):
    t = {"v": 0.0}
    c = PiWebApiConnector("pi", pi_server, {"A.FLOW": "W1"}, period_s=10.0, clock=lambda: t["v"])
    assert c.poll(1.0) and len(_Pi.seen) == 1
    assert c.poll(1.1) == [] and len(_Pi.seen) == 1         # within its period: no request
    _Pi.fail = True
    t["v"] = 20.0
    assert c.poll(1.2) == [] and c.status.startswith("error") and c.stats.errors == 1


# ------------------------------------------------------------------ Sparkplug B encoder (test side of the wire)
def _vi(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7f
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _f(num: int, wt: int, v: bytes) -> bytes:
    return _vi(num << 3 | wt) + v


def metric(name=None, alias=None, double=None, floatv=None, longv=None, intv=None, datatype=None, null=False, boolean=None) -> bytes:
    b = b""
    if name is not None:
        s = name.encode()
        b += _f(1, 2, _vi(len(s)) + s)
    if alias is not None:
        b += _f(2, 0, _vi(alias))
    if datatype is not None:
        b += _f(4, 0, _vi(datatype))
    if null:
        b += _f(7, 0, _vi(1))
    if intv is not None:
        b += _f(10, 0, _vi(intv))
    if longv is not None:
        b += _f(11, 0, _vi(longv))
    if floatv is not None:
        b += _f(12, 5, struct.pack("<f", floatv))
    if double is not None:
        b += _f(13, 1, struct.pack("<d", double))
    if boolean is not None:
        b += _f(14, 0, _vi(int(boolean)))
    return b


def payload(metrics: list, seq: int = 0) -> bytes:
    b = _f(1, 0, _vi(1700000000000))
    for m in metrics:
        b += _f(2, 2, _vi(len(m)) + m)
    return b + _f(3, 0, _vi(seq))


def test_sparkplug_decoder_handles_the_value_types():
    p = decode_sparkplug(payload([metric("a", 1, double=2.5), metric("b", 2, floatv=1.5), metric("c", 3, longv=70000, datatype=8),
                                  metric("neg", 4, intv=(1 << 32) - 3, datatype=3), metric("on", 5, boolean=True), metric("gone", 6, null=True)], seq=7))
    m = {x["name"]: x for x in p["metrics"]}
    assert p["seq"] == 7 and m["a"]["value"] == 2.5 and m["b"]["value"] == 1.5 and m["c"]["value"] == 70000.0
    assert m["neg"]["value"] == -3.0 and m["on"]["value"] == 1.0 and m["gone"]["is_null"] and m["gone"]["value"] is None


def test_sparkplug_birth_alias_data_and_death():
    c = SparkplugConnector("mqtt", "unused")
    c.on_message("spBv1.0/F/DBIRTH/GW1/KST.P07.W11", payload([metric("FLOW/LIQ", 11, double=100.0, datatype=10)], seq=0))
    c.on_message("spBv1.0/F/DDATA/GW1/KST.P07.W11", payload([metric(alias=11, double=120.0)], seq=1))
    c.on_message("spBv1.0/F/DDATA/GW1/KST.P07.W11", payload([metric(alias=99, double=1.0)], seq=2))       # no birth: unmapped
    c.on_message("spBv1.0/F/DCMD/GW1/KST.P07.W11", b"ignored")                                            # never ours
    c.on_message("spBv1.0/F/DDATA/GW1/KST.P07.W11", payload([metric(alias=11, null=True)], seq=5))        # sequence gap
    c.on_message("spBv1.0/F/DDEATH/GW1/KST.P07.W11", payload([]))
    assert ("GW1", "KST.P07.W11") in c.dead
    msgs = list(c.queue)
    assert isinstance(msgs[0], Birth) and msgs[0].prefix == "KST.P07.W11"
    assert [m.value for m in msgs[1:3]] == [100.0, 120.0] and msgs[1].tag == "KST.P07.W11.FLOW.LIQ"
    assert msgs[3].quality == "bad" and math.isnan(msgs[3].value)
    assert c.stats.errors == 2                                        # one unmapped metric, one sequence gap


class _Broker(threading.Thread):
    """Just enough of an MQTT 3.1.1 broker for one subscriber: CONNACK, SUBACK, then the given publishes."""

    def __init__(self, messages):
        super().__init__(daemon=True)
        self.messages, self.sub = messages, None
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]

    @staticmethod
    def _rd(c):
        h = c.recv(1)
        if not h:
            return None, b""
        n, mult = 0, 1
        while True:
            b = c.recv(1)[0]
            n += (b & 0x7f) * mult
            mult *= 128
            if not b & 0x80:
                break
        data = b""
        while len(data) < n:
            data += c.recv(n - len(data))
        return h[0] >> 4, data

    def run(self):
        c, _ = self.srv.accept()
        assert self._rd(c)[0] == 1
        c.sendall(b"\x20\x02\x00\x00")
        kind, data = self._rd(c)
        assert kind == 8
        n = struct.unpack(">H", data[2:4])[0]
        self.sub = data[4:4 + n].decode()
        c.sendall(b"\x90\x03\x00\x01\x00")
        for topic, body in self.messages:
            t = topic.encode()
            frame = struct.pack(">H", len(t)) + t + body
            ln, rem = len(frame), bytearray()
            while True:
                d = ln % 128
                ln //= 128
                rem.append(d | (0x80 if ln else 0))
                if not ln:
                    break
            c.sendall(b"\x30" + bytes(rem) + frame)
        time.sleep(1.0)
        c.close()


def test_sparkplug_over_a_real_socket():
    msgs = [("spBv1.0/F/NBIRTH/GW1", payload([metric("Node Control/Rebirth", 1, boolean=False), metric("TK/LVL", 2, double=55.5)], seq=0)),
            ("spBv1.0/F/NDATA/GW1", payload([metric(alias=2, double=56.5)], seq=1))]
    b = _Broker(msgs)
    b.start()
    c = SparkplugConnector("mqtt", "127.0.0.1", b.port, group="F", reconnect_s=60.0)
    c.start()
    got, t0 = [], time.time()
    while len(got) < 3 and time.time() - t0 < 5.0:
        got += c.poll(1.0)
        time.sleep(0.02)
    c.stop()
    assert b.sub == "spBv1.0/F/#"
    assert isinstance(got[0], Birth)
    assert [(m.tag, m.value, m.t) for m in got[1:]] == [("GW1.TK.LVL", 55.5, 1.0), ("GW1.TK.LVL", 56.5, 1.0)]


def test_opc_ua_reports_a_missing_library_clearly():
    try:
        import asyncua  # noqa: F401
    except ImportError:
        with pytest.raises(ConnectorUnavailable, match="pip install asyncua"):
            OpcUaConnector("ua", "opc.tcp://localhost:4840", {"A": "ns=2;s=A"})


def test_opc_ua_against_a_local_server():
    pytest.importorskip("asyncua")
    from asyncua import ua
    from asyncua.sync import Server
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    srv = Server()
    srv.set_endpoint(f"opc.tcp://127.0.0.1:{port}/plyxis")
    srv.set_security_policy([ua.SecurityPolicyType.NoSecurity])
    idx = srv.register_namespace("urn:plyxis:test")
    obj = srv.nodes.objects.add_object(idx, "Well")
    flow = obj.add_variable(ua.NodeId("A.FLOW", idx), "FLOW", 12.5)
    text = obj.add_variable(ua.NodeId("A.TXT", idx), "TXT", "not a number")
    srv.start()
    try:
        t = {"v": 0.0}
        c = OpcUaConnector("ua", f"opc.tcp://127.0.0.1:{port}/plyxis", {"A.FLOW": f"ns={idx};s=A.FLOW", "A.TXT": f"ns={idx};s=A.TXT"},
                           units={"A.FLOW": "bbl/d"}, period_s=10.0, clock=lambda: t["v"])
        msgs = c.poll(3.0)
        assert [(m.tag, m.value, m.unit, m.quality, m.t) for m in msgs] == [("A.FLOW", 12.5, "bbl/d", "good", 3.0)]
        assert c.poll(3.1) == []                        # within its period
        flow.write_value(20.0)
        t["v"] = 11.0
        assert c.poll(4.0)[0].value == 20.0
        c.client.disconnect()
        assert text is not None
    finally:
        srv.stop()


def test_build_connector_from_configuration():
    c = build_connector({"type": "sparkplug", "name": "m", "host": "h"})
    assert isinstance(c, SparkplugConnector)
    with pytest.raises(ValueError):
        build_connector({"type": "smoke-signal", "name": "x"})


# ------------------------------------------------------------------ historian export -> replay
def test_history_roundtrip_keeps_last_known_controls_and_skips_gaps(tmp_path):
    path = str(tmp_path / "h.csv.gz")
    w = HistoryWriter(path)
    w.telemetry([Telemetry("A.B", 1.5, "bbl", 1.0), Telemetry("A.T", 0.9, None, 1.0, "test"), Birth("A", 1.0, {"type": "well"})])
    w.controls(1.0, {("A", "speed"): 0.5}, {("A", "out", "B", "in"): 1.0})
    w.prices(1.0, {"oil": 0.4})
    w.telemetry([Telemetry("A.B", 2.5, "bbl", 2.0)])
    w.close()
    steps = list(read_history(path))
    assert [s.t for s in steps] == [1.0, 2.0]
    a, b = steps
    assert [(m.tag, m.quality) for m in a.telemetry] == [("A.B", "good"), ("A.T", "test")] and a.births[0].meta["type"] == "well"
    assert b.controls == {("A", "speed"): 0.5} and b.routes == {("A", "out", "B", "in"): 1.0} and b.prices == {"oil": 0.4}


def test_backtest_replay_and_backfill_run_on_the_file_alone(tmp_path):
    from bench.backtest import export_history, replay
    from plyxis.runtime.backfill import backfill
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sdir = os.path.join(root, "scenarios", "kestrel")
    path = str(tmp_path / "h.csv.gz")
    assert export_history(sdir, 6.0, path)["steps"] == 36
    gate = {"signals": [("Wells", ".FLOW.LIQ", 50.0)], "levels": [("Oil", "oil_lvl", "oil_tank", 50.0)]}
    r = replay(sdir, path, 2.0, gate)
    assert r["signals"][0]["n"] > 0 and r["signals"][1]["n"] > 0
    assert r["signals"][0]["pass"] and r["gate_passed"]
    b = backfill(sdir, path)
    assert b["steps"] == 36 and b["calibrations"] >= 4 and b["properties"] > 0
    assert any(line.startswith("set kestrel_live.") for line in b["calibration"].splitlines())
