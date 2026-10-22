"""PLX - the PLYXIS twin description language.

    from plyxis.lang import compile_model
    model = compile_model(["site.plx"], env="kestrel")
    raw = model.program.simulate(model.program.default_decisions())
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Optional

from .errors import Diagnostic, Diagnostics, PlxError

STDLIB_DIR = os.path.join(os.path.dirname(__file__), "stdlib")


def stdlib_files() -> list[str]:
    return sorted(glob.glob(os.path.join(STDLIB_DIR, "*.plx")))


@dataclass
class CompiledModel:
    program: "object"
    physical: Optional["object"]
    flat: "object"
    lib: "object"
    diagnostics: Diagnostics
    sources: dict = field(default_factory=dict)


def compile_model(files: list[str], env: Optional[str] = None, sim: Optional[str] = None,
                  sources: Optional[dict] = None, physical: bool = False, include_stdlib: bool = True,
                  diags: Optional[Diagnostics] = None) -> CompiledModel:
    """Parse, elaborate and compile PLX sources into an executable twin.

    files    - list of paths (or virtual names present in `sources`)
    env      - environment to elaborate (default: the only/first 'env' definition)
    sim      - simulation spec (default: the one bound to the env)
    physical - also compile the 'physical' variant used by the plant emulator
               (state bounds saturate instead of being constraints)
    """
    from .compiler import Compiler
    from .elaborate import Elaborator
    from .library import load_library

    diags = diags if diags is not None else Diagnostics()
    all_files = (stdlib_files() if include_stdlib else []) + list(files)
    lib = load_library(all_files, diags, sources)
    diags.raise_if_errors()
    if env is None:
        envs = [q for q, d in lib.defs.items() if getattr(d, "kind", None) == "env"]
        if not envs:
            diags.error("no environment ('define ... as env') found")
            diags.raise_if_errors()
        env = envs[-1]
    flat = Elaborator(lib, diags).elaborate(env, sim)
    diags.raise_if_errors()
    prog = Compiler(flat, diags, physical=False).compile()
    phys = None
    if physical:
        phys = Compiler(flat, diags, physical=True).compile()
    return CompiledModel(prog, phys, flat, lib, diags, dict(diags.sources))


def check_source(files: list[str], sources: Optional[dict] = None, env: Optional[str] = None) -> list[dict]:
    """Compile and return diagnostics as dicts (never raises) - used by the editor."""
    diags = Diagnostics()
    try:
        compile_model(files, env=env, sources=sources, diags=diags)
    except PlxError:
        pass
    except Exception as ex:  # noqa: BLE001 - surface internal errors to the editor as well
        diags.error(f"internal compiler error: {ex}")
    return [d.to_dict() for d in diags.items]


__all__ = ["compile_model", "check_source", "CompiledModel", "PlxError", "Diagnostics", "Diagnostic"]
