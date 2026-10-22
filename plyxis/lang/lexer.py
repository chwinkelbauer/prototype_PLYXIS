"""PLX lexer.

Produces a flat token list. Whitespace is insignificant except that the parser
uses token adjacency (no whitespace between tokens) to recognise interpolated
names such as ``w{i}`` or ``pad{p}_w{j}``.
"""
from __future__ import annotations

from dataclasses import dataclass

from .errors import Diagnostics, Loc

MULTI_OPS = ("<-", "<=", ">=", "==", "!=", "->", "..", "&&", "||", "=>", "**")
SINGLE_OPS = set("+-*/^%<>=(){}[],;:.$!@?|&")


@dataclass
class Token:
    kind: str          # ident | number | string | op | eof
    value: str
    loc: Loc
    num: float = 0.0

    def is_op(self, *ops: str) -> bool:
        return self.kind == "op" and self.value in ops

    def is_ident(self, *names: str) -> bool:
        return self.kind == "ident" and (not names or self.value in names)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Token({self.kind},{self.value!r},{self.loc.line}:{self.loc.col})"


class Lexer:
    def __init__(self, text: str, file: str = "<input>", diags: Diagnostics | None = None):
        self.text = text
        self.file = file
        self.diags = diags if diags is not None else Diagnostics()
        self.pos = 0
        self.line = 1
        self.col = 1
        self.doc_comments: dict[int, str] = {}   # line -> text of '##' comments

    def _loc(self, start: int, sline: int, scol: int) -> Loc:
        return Loc(self.file, sline, scol, self.line, self.col, start, self.pos)

    def _advance(self, n: int = 1) -> None:
        for _ in range(n):
            if self.pos >= len(self.text):
                return
            if self.text[self.pos] == "\n":
                self.line += 1
                self.col = 1
            else:
                self.col += 1
            self.pos += 1

    def tokens(self) -> list[Token]:
        out: list[Token] = []
        t = self.text
        n = len(t)
        while True:
            # skip whitespace and comments
            while self.pos < n:
                c = t[self.pos]
                if c in " \t\r\n﻿":
                    self._advance()
                elif c == "#":
                    start = self.pos
                    line = self.line
                    while self.pos < n and t[self.pos] != "\n":
                        self._advance()
                    raw = t[start:self.pos]
                    if raw.startswith("##"):
                        self.doc_comments[line] = raw[2:].strip()
                else:
                    break
            if self.pos >= n:
                out.append(Token("eof", "", Loc(self.file, self.line, self.col, self.line, self.col, self.pos, self.pos)))
                return out
            start, sline, scol = self.pos, self.line, self.col
            c = t[self.pos]
            # identifiers
            if c.isalpha() or c == "_":
                while self.pos < n and (t[self.pos].isalnum() or t[self.pos] == "_"):
                    self._advance()
                out.append(Token("ident", t[start:self.pos], self._loc(start, sline, scol)))
                continue
            # numbers: 12, 1.5, .5, 1e-3 ; '1..5' is a range, not 1.
            if c.isdigit() or (c == "." and self.pos + 1 < n and t[self.pos + 1].isdigit()
                               and not (out and out[-1].is_op(".."))):
                while self.pos < n and t[self.pos].isdigit():
                    self._advance()
                if self.pos < n and t[self.pos] == "." and self.pos + 1 < n and t[self.pos + 1].isdigit():
                    self._advance()
                    while self.pos < n and t[self.pos].isdigit():
                        self._advance()
                if self.pos < n and t[self.pos] in "eE":
                    j = self.pos + 1
                    if j < n and t[j] in "+-":
                        j += 1
                    if j < n and t[j].isdigit():
                        self._advance(j - self.pos)
                        while self.pos < n and t[self.pos].isdigit():
                            self._advance()
                raw = t[start:self.pos]
                tok = Token("number", raw, self._loc(start, sline, scol))
                try:
                    tok.num = float(raw)
                except ValueError:
                    self.diags.error(f"invalid number '{raw}'", tok.loc, "E-LEX")
                out.append(tok)
                continue
            # strings
            if c == '"' or c == "'":
                q = c
                self._advance()
                buf = []
                closed = False
                while self.pos < n:
                    ch = t[self.pos]
                    if ch == "\\" and self.pos + 1 < n:
                        nxt = t[self.pos + 1]
                        buf.append({"n": "\n", "t": "\t", '"': '"', "'": "'", "\\": "\\"}.get(nxt, nxt))
                        self._advance(2)
                        continue
                    if ch == q:
                        self._advance()
                        closed = True
                        break
                    if ch == "\n":
                        break
                    buf.append(ch)
                    self._advance()
                loc = self._loc(start, sline, scol)
                if not closed:
                    self.diags.error("unterminated string literal", loc, "E-LEX")
                out.append(Token("string", "".join(buf), loc))
                continue
            # operators
            two = t[self.pos:self.pos + 2]
            if two in MULTI_OPS:
                self._advance(2)
                out.append(Token("op", two, self._loc(start, sline, scol)))
                continue
            if c in SINGLE_OPS:
                self._advance()
                out.append(Token("op", c, self._loc(start, sline, scol)))
                continue
            self._advance()
            self.diags.error(f"unexpected character '{c}'", self._loc(start, sline, scol), "E-LEX")


def tokenize(text: str, file: str = "<input>", diags: Diagnostics | None = None):
    lx = Lexer(text, file, diags)
    toks = lx.tokens()
    return toks, lx.doc_comments
