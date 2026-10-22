"""Diagnostics for the PLX compiler.

Every error carries a source location so the Twin Studio editor can place
markers exactly where the problem is (file, line, column, span).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Loc:
    file: str = "<input>"
    line: int = 0
    col: int = 0
    end_line: int = 0
    end_col: int = 0
    offset: int = 0
    end_offset: int = 0

    def span(self, other: "Loc") -> "Loc":
        return Loc(self.file, self.line, self.col, other.end_line, other.end_col,
                   self.offset, other.end_offset)

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.col}"


NOLOC = Loc()


@dataclass
class Diagnostic:
    severity: str          # "error" | "warning" | "info"
    message: str
    loc: Loc = NOLOC
    code: str = ""
    hint: str = ""

    def to_dict(self) -> dict:
        return {
            "severity": self.severity, "message": self.message, "code": self.code,
            "hint": self.hint, "file": self.loc.file, "line": self.loc.line,
            "col": self.loc.col, "end_line": self.loc.end_line or self.loc.line,
            "end_col": self.loc.end_col or (self.loc.col + 1),
        }

    def format(self, sources: Optional[dict] = None) -> str:
        s = f"{self.loc}: {self.severity}: {self.message}"
        if self.code:
            s += f" [{self.code}]"
        if sources and self.loc.file in sources and self.loc.line > 0:
            lines = sources[self.loc.file].splitlines()
            if 0 < self.loc.line <= len(lines):
                src = lines[self.loc.line - 1]
                width = max(1, (self.loc.end_col - self.loc.col) if self.loc.end_line == self.loc.line else 1)
                s += f"\n    {src}\n    {' ' * max(0, self.loc.col - 1)}{'^' * width}"
        if self.hint:
            s += f"\n    hint: {self.hint}"
        return s


class PlxError(Exception):
    """Raised when compilation cannot continue. Carries all diagnostics."""

    def __init__(self, diagnostics: list[Diagnostic], sources: Optional[dict] = None):
        self.diagnostics = diagnostics
        self.sources = sources or {}
        errs = [d for d in diagnostics if d.severity == "error"]
        head = errs[0] if errs else (diagnostics[0] if diagnostics else None)
        msg = head.format(self.sources) if head else "compilation failed"
        if len(errs) > 1:
            msg += f"\n(+{len(errs) - 1} more errors)"
        super().__init__(msg)


@dataclass
class Diagnostics:
    items: list[Diagnostic] = field(default_factory=list)
    sources: dict = field(default_factory=dict)

    def error(self, msg: str, loc: Loc = NOLOC, code: str = "", hint: str = "") -> None:
        self.items.append(Diagnostic("error", msg, loc, code, hint))

    def warning(self, msg: str, loc: Loc = NOLOC, code: str = "", hint: str = "") -> None:
        self.items.append(Diagnostic("warning", msg, loc, code, hint))

    def info(self, msg: str, loc: Loc = NOLOC, code: str = "") -> None:
        self.items.append(Diagnostic("info", msg, loc, code))

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.items if d.severity == "error"]

    @property
    def warnings(self) -> list[Diagnostic]:
        return [d for d in self.items if d.severity == "warning"]

    def has_errors(self) -> bool:
        return any(d.severity == "error" for d in self.items)

    def raise_if_errors(self) -> None:
        if self.has_errors():
            raise PlxError(self.items, self.sources)

    def extend(self, other: "Diagnostics") -> None:
        self.items.extend(other.items)
        self.sources.update(other.sources)


def suggest(name: str, candidates) -> str:
    """'did you mean' helper based on edit distance."""
    best, best_d = None, 10 ** 9
    for c in candidates:
        d = _lev(name.lower(), str(c).lower())
        if d < best_d:
            best, best_d = c, d
    if best is not None and best_d <= max(2, len(name) // 3):
        return f"did you mean '{best}'?"
    return ""


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
