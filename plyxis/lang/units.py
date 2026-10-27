"""Units of measure for PLX.

Literals annotated with a unit (``q_max = 1200 [bbl/d]``) are converted into a
coherent internal system so that expressions stay dimensionally consistent:

    length m · mass kg · time h · currency USD · energy kWh · pressure kPa · temperature K · count

Time is canonical in *hours* because operations are planned in hours; rates
declared per day are therefore divided by 24 internally and converted back for
display. Energy and pressure are kept as independent base dimensions (a
pragmatic choice for operations models; they are never converted into each
other implicitly).
"""
from __future__ import annotations

import functools
import math
import re
from typing import Optional

BASE = ("L", "M", "T", "CUR", "E", "P", "TEMP", "N")
ZERO = (0,) * len(BASE)


def _d(**kw) -> tuple:
    return tuple(kw.get(b, 0) for b in BASE)


L3 = _d(L=3)
UNITS: dict[str, tuple[float, tuple]] = {
    # dimensionless
    "1": (1.0, ZERO), "%": (0.01, ZERO), "frac": (1.0, ZERO), "ppm": (1e-6, ZERO), "-": (1.0, ZERO),
    "rpm": (60.0, _d(T=-1)), "Hz": (3600.0, _d(T=-1)),
    # length
    "m": (1.0, _d(L=1)), "km": (1000.0, _d(L=1)), "mi": (1609.344, _d(L=1)), "ft": (0.3048, _d(L=1)),
    "in": (0.0254, _d(L=1)), "cm": (0.01, _d(L=1)), "mm": (0.001, _d(L=1)),
    "m2": (1.0, _d(L=2)), "km2": (1e6, _d(L=2)), "acre": (4046.8564224, _d(L=2)), "ha": (1e4, _d(L=2)),
    # time (canonical: hour)
    "s": (1 / 3600, _d(T=1)), "sec": (1 / 3600, _d(T=1)), "min": (1 / 60, _d(T=1)),
    "h": (1.0, _d(T=1)), "hr": (1.0, _d(T=1)), "d": (24.0, _d(T=1)), "day": (24.0, _d(T=1)),
    "wk": (168.0, _d(T=1)), "mo": (730.0, _d(T=1)), "yr": (8760.0, _d(T=1)), "a": (8760.0, _d(T=1)),
    # volume
    "m3": (1.0, L3), "Sm3": (1.0, L3), "kSm3": (1e3, L3), "MMSm3": (1e6, L3), "L": (1e-3, L3),
    "bbl": (0.158987294928, L3), "kbbl": (158.987294928, L3), "Mbbl": (158.987294928, L3),
    "boe": (0.158987294928, L3), "gal": (0.003785411784, L3),
    "scf": (0.0283168466, L3), "Mscf": (28.3168466, L3), "MMscf": (28316.8466, L3), "Bcf": (28316846.6, L3),
    "mcf": (28.3168466, L3), "mmcf": (28316.8466, L3),
    # mass
    "kg": (1.0, _d(M=1)), "t": (1000.0, _d(M=1)), "tonne": (1000.0, _d(M=1)), "kt": (1e6, _d(M=1)),
    "lb": (0.45359237, _d(M=1)), "g": (1e-3, _d(M=1)),
    "kgCO2e": (1.0, _d(M=1)), "tCO2e": (1000.0, _d(M=1)), "gCO2e": (1e-3, _d(M=1)),
    # currency
    "USD": (1.0, _d(CUR=1)), "$": (1.0, _d(CUR=1)), "kUSD": (1e3, _d(CUR=1)), "MUSD": (1e6, _d(CUR=1)),
    "EUR": (1.08, _d(CUR=1)), "kEUR": (1080.0, _d(CUR=1)),
    # energy & power (canonical kWh, kW)
    "kWh": (1.0, _d(E=1)), "MWh": (1e3, _d(E=1)), "GWh": (1e6, _d(E=1)), "Wh": (1e-3, _d(E=1)),
    "J": (1 / 3.6e6, _d(E=1)), "kJ": (1 / 3600, _d(E=1)), "MJ": (1 / 3.6, _d(E=1)), "GJ": (1000 / 3.6, _d(E=1)),
    "Btu": (0.00029307107, _d(E=1)), "MMBtu": (293.07107, _d(E=1)),
    "W": (1e-3, _d(E=1, T=-1)), "kW": (1.0, _d(E=1, T=-1)), "MW": (1e3, _d(E=1, T=-1)),
    "hp": (0.745699872, _d(E=1, T=-1)),
    # pressure (canonical kPa)
    "Pa": (1e-3, _d(P=1)), "kPa": (1.0, _d(P=1)), "MPa": (1e3, _d(P=1)), "bar": (100.0, _d(P=1)),
    "psi": (6.894757293, _d(P=1)), "psig": (6.894757293, _d(P=1)), "psia": (6.894757293, _d(P=1)),
    "atm": (101.325, _d(P=1)),
    # temperature (differences only)
    "K": (1.0, _d(TEMP=1)), "degC": (1.0, _d(TEMP=1)), "C": (1.0, _d(TEMP=1)), "degF": (5 / 9, _d(TEMP=1)),
    # counts
    "each": (1.0, _d(N=1)), "ea": (1.0, _d(N=1)), "unit": (1.0, _d(N=1)), "units": (1.0, _d(N=1)),
    "truck": (1.0, _d(N=1)), "trucks": (1.0, _d(N=1)), "crew": (1.0, _d(N=1)), "trip": (1.0, _d(N=1)),
    "km/h": (1000.0, _d(L=1, T=-1)), "mph": (1609.344, _d(L=1, T=-1)),
    "mm/s": (3.6, _d(L=1, T=-1)), "m/s": (3600.0, _d(L=1, T=-1)),
}


class UnitError(Exception):
    pass


_TOKEN = re.compile(r"\s*(?:(?P<num>\d+(?:\.\d+)?)|(?P<name>[A-Za-z_$%][A-Za-z0-9_$%]*)|(?P<op>[*/^()·]))")


def _tokens(s: str):
    pos = 0
    out = []
    # composite names containing '/' that are atomic units
    for atomic in ("km/h", "mm/s", "m/s"):
        s = s.replace(atomic, atomic.replace("/", "_PER_"))
    while pos < len(s):
        m = _TOKEN.match(s, pos)
        if not m or m.end() == pos:
            if s[pos:].strip() == "":
                break
            raise UnitError(f"bad unit syntax near '{s[pos:]}'")
        pos = m.end()
        if m.group("num"):
            out.append(("num", m.group("num")))
        elif m.group("name"):
            out.append(("name", m.group("name").replace("_PER_", "/")))
        else:
            out.append(("op", m.group("op")))
    return out


@functools.lru_cache(maxsize=8192)
def parse_unit(text: str) -> tuple[float, tuple]:
    """Return (factor_to_canonical, dims) for a unit expression like 'bbl/d/psi'."""
    text = text.strip()
    if text in UNITS:
        return UNITS[text]
    toks = _tokens(text)
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else (None, None)

    def factor():
        nonlocal pos
        kind, val = peek()
        if kind == "op" and val == "(":
            pos += 1
            f, d = expr()
            if peek() != ("op", ")"):
                raise UnitError("missing ')' in unit")
            pos += 1
        elif kind == "name":
            pos += 1
            if val not in UNITS:
                raise UnitError(f"unknown unit '{val}'")
            f, d = UNITS[val]
        elif kind == "num":
            pos += 1
            f, d = float(val), ZERO
        else:
            raise UnitError(f"unexpected token in unit: {val}")
        if peek() == ("op", "^"):
            pos += 1
            k, v = peek()
            sign = 1
            if k == "op" and v == "-":
                pos += 1
                k, v = peek()
                sign = -1
            if k != "num":
                raise UnitError("expected exponent")
            pos += 1
            e = float(v) * sign
            f = f ** e
            d = tuple(x * e for x in d)
        return f, d

    def expr():
        nonlocal pos
        f, d = factor()
        while True:
            kind, val = peek()
            if kind == "op" and val in ("*", "·"):
                pos += 1
                f2, d2 = factor()
                f, d = f * f2, tuple(a + b for a, b in zip(d, d2))
            elif kind == "op" and val == "/":
                pos += 1
                f2, d2 = factor()
                f, d = f / f2, tuple(a - b for a, b in zip(d, d2))
            else:
                return f, d

    f, d = expr()
    if pos != len(toks):
        raise UnitError(f"trailing tokens in unit '{text}'")
    return f, d


def to_canonical(value: float, unit: Optional[str]) -> float:
    if not unit:
        return value
    f, _ = parse_unit(unit)
    return value * f


def from_canonical(value: float, unit: Optional[str]) -> float:
    if not unit:
        return value
    f, _ = parse_unit(unit)
    return value / f


def factor(unit: Optional[str]) -> float:
    if not unit:
        return 1.0
    return parse_unit(unit)[0]


def dims_of(unit: Optional[str]) -> Optional[tuple]:
    if not unit:
        return None
    try:
        return parse_unit(unit)[1]
    except UnitError:
        return None


def fmt_dims(d: Optional[tuple]) -> str:
    if d is None:
        return "?"
    if all(x == 0 for x in d):
        return "1"
    num = [f"{b}{'' if e == 1 else '^' + _fmt(e)}" for b, e in zip(BASE, d) if e > 0]
    den = [f"{b}{'' if e == -1 else '^' + _fmt(-e)}" for b, e in zip(BASE, d) if e < 0]
    s = "·".join(num) or "1"
    if den:
        s += "/" + "·".join(den)
    return s


def _fmt(e: float) -> str:
    return str(int(e)) if float(e).is_integer() else f"{e:g}"


def dims_mul(a, b):
    if a is None or b is None:
        return None
    return tuple(x + y for x, y in zip(a, b))


def dims_div(a, b):
    if a is None or b is None:
        return None
    return tuple(x - y for x, y in zip(a, b))


def dims_pow(a, e: float):
    if a is None:
        return None
    return tuple(x * e for x in a)


def dims_equal(a, b) -> bool:
    if a is None or b is None:
        return True
    return all(math.isclose(x, y, abs_tol=1e-9) for x, y in zip(a, b))


# display unit suggestion for a dimension (used when a signal has no declared unit)
_PREFERRED = {
    _d(L=3, T=-1): "bbl/d", _d(L=3): "bbl", _d(CUR=1, T=-1): "USD/h", _d(CUR=1): "USD",
    _d(E=1, T=-1): "kW", _d(E=1): "kWh", _d(P=1): "psi", _d(M=1, T=-1): "kgCO2e/h", _d(M=1): "kg",
    _d(T=1): "h", _d(L=1): "km", _d(N=1): "each",
}


def preferred_unit(d: Optional[tuple]) -> Optional[str]:
    if d is None:
        return None
    for k, v in _PREFERRED.items():
        if dims_equal(k, d):
            return v
    return None
