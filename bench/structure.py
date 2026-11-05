"""The block tree of every case study's asset graph (plyxis/lang/blocks.py): what can be solved on its own, why, and
which blocks the plan uses as input-output functions.

    python -m bench.structure

Writes bench/results/structure.json: per twin the tree (name, units, step, decisions per step, flows at its fence,
structure class, exact per-unit methods, the rule that cut it and why, used or not) and the time to find it.
"""
from __future__ import annotations

import json
import os
import time

from plyxis.lang import compile_model
from plyxis.lang.blocks import find_structure, render

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TWINS = {"kestrel": ("kestrel", None), "corvus": ("corvus", None), "aquila": ("aquila", None),
         "aquila_works": ("aquila_works", "corp96")}


def _tree(b, flat) -> dict:
    return {"name": b.name, "units": len(b.members), "step_min": round(b.step * 60, 3), "decisions_per_step": b.channels,
            "fence_flows": len(b.interface), "class": b.cls, "exact": b.exact, "rule": b.rule, "why": b.why[:12],
            "used": b.used, "note": b.note, "depth": b.depth,
            "single_units": [c.name for c in b.children if len(c.members) == 1 and not c.children],
            "children": [_tree(c, flat) for c in b.children if not (len(c.members) == 1 and not c.children)]}


def main() -> None:
    out = {}
    for key, (folder, sim) in TWINS.items():
        t = time.perf_counter()
        m = compile_model([os.path.join(ROOT, "scenarios", folder, "field.plx")], sim=sim)
        root = find_structure(m)
        sec = time.perf_counter() - t
        blocks = list(root.walk())[1:]
        out[key] = {"seconds": sec, "blocks": len(blocks), "used": [b.name for b in blocks if b.used],
                    "by_depth": {d: sum(1 for b in blocks if b.depth == d) for d in sorted({b.depth for b in blocks})},
                    "tree": _tree(root, m.flat)}
        print(f"== {key} ({sec:.1f} s)\n" + render(root, m.flat), flush=True)
    os.makedirs(os.path.join(ROOT, "bench", "results"), exist_ok=True)
    with open(os.path.join(ROOT, "bench", "results", "structure.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
