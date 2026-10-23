"""Abstract syntax tree for PLX.

The tree is deliberately close to the surface syntax so that the pretty
printer (``printer.py``) can regenerate readable source - the data-integration
layer rewrites twins by emitting PLX code, not by patching hidden JSON.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from .errors import Loc, NOLOC


# --------------------------------------------------------------------------- expressions
@dataclass
class Expr:
    loc: Loc = field(default=NOLOC, kw_only=True)


@dataclass
class Num(Expr):
    value: float
    unit: Optional[str] = None


@dataclass
class Str(Expr):
    value: str


@dataclass
class Bool(Expr):
    value: bool


@dataclass
class Name(Expr):
    id: str


@dataclass
class Template(Expr):
    """Interpolated identifier, e.g. ``w{i}`` -> parts ['w', Name(i)]."""
    parts: list  # list[str | Expr]

    def is_plain(self) -> bool:
        return all(isinstance(p, str) for p in self.parts)

    def plain(self) -> str:
        return "".join(p for p in self.parts if isinstance(p, str))


@dataclass
class Member(Expr):
    obj: Expr
    attr: Union[str, "Template"]


@dataclass
class Call(Expr):
    func: str
    args: list
    kwargs: dict = field(default_factory=dict)


@dataclass
class Unary(Expr):
    op: str
    operand: Expr


@dataclass
class Binary(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass
class Compare(Expr):
    ops: list          # list[str]
    operands: list     # list[Expr], len = len(ops)+1


@dataclass
class TupleExpr(Expr):
    items: list


@dataclass
class ListExpr(Expr):
    items: list


@dataclass
class Range(Expr):
    start: Expr
    end: Expr


@dataclass
class Wildcard(Expr):
    pass


# --------------------------------------------------------------------------- statements
@dataclass
class Stmt:
    loc: Loc = field(default=NOLOC, kw_only=True)


@dataclass
class Define(Stmt):
    """``target = value`` - a local signal, or an out-port channel assignment."""
    target: Expr
    value: Expr
    unit: Optional[str] = None


@dataclass
class Update(Stmt):
    """``state <- value`` - next-step state update (synchronous semantics)."""
    target: Expr
    value: Expr


@dataclass
class ConstraintStmt(Stmt):
    name: Optional[str]
    expr: Compare
    soft: Optional[Expr] = None


@dataclass
class TermStmt(Stmt):
    """``cost|revenue|emission|metric NAME = expr`` (rates per hour)."""
    kind: str
    name: str
    expr: Expr
    final: bool = False
    unit: Optional[str] = None


@dataclass
class Override:
    name: str
    value: Expr
    unit: Optional[str] = None
    loc: Loc = NOLOC


@dataclass
class InitStmt(Stmt):
    name: Template
    type: Template
    overrides: list = field(default_factory=list)   # list[Override]


@dataclass
class ConnectStmt(Stmt):
    src: Expr
    dst: Expr
    via: Optional[Template] = None
    overrides: list = field(default_factory=list)
    name: Optional[Template] = None


@dataclass
class ForStmt(Stmt):
    var: str
    iterable: Expr
    body: list


@dataclass
class IfStmt(Stmt):
    cond: Expr
    then: list
    orelse: list = field(default_factory=list)


@dataclass
class SetStmt(Stmt):
    path: Expr
    value: Expr
    unit: Optional[str] = None


@dataclass
class ExprStmt(Stmt):
    expr: Expr


@dataclass
class DecisionStmtLegacy(Stmt):
    """Old-style ``decision outX$.$amount(t);`` inside behavior."""
    text: str


# --------------------------------------------------------------------------- declarations
@dataclass
class ParamDecl:
    name: str
    default: Optional[Expr]
    unit: Optional[str] = None
    calibrate: Optional[tuple] = None   # (lo Expr, hi Expr)
    is_input: bool = False
    doc: str = ""
    loc: Loc = NOLOC


@dataclass
class StateDecl:
    name: str
    init: Expr
    unit: Optional[str] = None
    bounds: Optional[tuple] = None      # (lo Expr, hi Expr)
    doc: str = ""
    loc: Loc = NOLOC


@dataclass
class DecisionDecl:
    name: str
    domain: str = "real"                # real | int | bool
    lo: Optional[Expr] = None
    hi: Optional[Expr] = None
    unit: Optional[str] = None
    default: Optional[Expr] = None
    hold: Optional[Expr] = None
    hold_unit: Optional[str] = None
    doc: str = ""
    loc: Loc = NOLOC
    default_unit: Optional[str] = None


@dataclass
class PortDecl:
    direction: str                      # in | out
    name: str
    resources: list                     # list[str]; ['*'] = wildcard
    exclusive: bool = False
    doc: str = ""
    loc: Loc = NOLOC


@dataclass
class TelemetryDecl:
    signal: str
    tag: str
    unit: Optional[str] = None
    options: dict = field(default_factory=dict)   # name -> Expr
    loc: Loc = NOLOC


@dataclass
class RuleDecl:
    name: str
    pattern: str
    body: list
    when: Optional[Expr] = None
    loc: Loc = NOLOC


@dataclass
class Definition:
    name: str
    loc: Loc = field(default=NOLOC, kw_only=True)
    doc: str = field(default="", kw_only=True)


@dataclass
class EntityDef(Definition):
    kind: str = "ent"                   # ent | obj | env
    base: Optional[str] = None          # parent entity type (inheritance)
    attrs: dict = field(default_factory=dict)          # flags before sections (part=TRUE...)
    params: list = field(default_factory=list)
    states: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    ports: list = field(default_factory=list)
    behavior: list = field(default_factory=list)
    contains: list = field(default_factory=list)
    geom: dict = field(default_factory=dict)
    telemetry: list = field(default_factory=list)
    rules: list = field(default_factory=list)
    constraints: list = field(default_factory=list)    # env-level constraints


@dataclass
class ResourceDef(Definition):
    unit: Optional[str] = None
    rate_unit: Optional[str] = None
    attrs: dict = field(default_factory=dict)          # attr name -> default Expr
    props: dict = field(default_factory=dict)


@dataclass
class GroupDef(Definition):
    members: list = field(default_factory=list)        # list[str]


@dataclass
class FunctionDef(Definition):
    args: list = field(default_factory=list)
    body: Optional[Expr] = None


@dataclass
class SimDef(Definition):
    env: Optional[str] = None
    settings: dict = field(default_factory=dict)       # name -> (Expr, unit)
    inputs: list = field(default_factory=list)         # list[Define]
    objective: Optional[tuple] = None                  # (sense, Expr)
    solver: dict = field(default_factory=dict)
    legacy_behavior: list = field(default_factory=list)


@dataclass
class PackageDef(Definition):
    items: list = field(default_factory=list)


@dataclass
class UseDef(Definition):
    path: str = ""


@dataclass
class Program:
    items: list
    file: str = "<input>"
    docs: dict = field(default_factory=dict)
