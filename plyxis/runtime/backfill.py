"""Backfill: from a raw historian export to a calibrated twin (data/historian.py describes the file).

The file is replayed through the same estimator the live twin uses: states assimilated, properties re-estimated every
calibration period, unknown tags turned into proposals by the discovery rules. The result is a `calibration.plx` of
`set` lines, a reviewable change to the twin like any other.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch

from ..data.historian import read_history
from .scenario import Scenario
from .twin_manager import TwinManager


def backfill(sdir: str, path: str, progress: Optional[Callable] = None, iters: int = 25) -> dict:
    sc = Scenario.load(sdir)
    twin = TwinManager(sc.twin_files, sc.twin_env, sc.twin_sim, dt_data=sc.dt_emu,
                       test_sigma=(sc.raw.get("assays") or {}).get("sigma"))
    first, last_cal, steps, calibrations, discovered = None, None, 0, 0, []
    t = 0.0
    for step in read_history(path):
        t = step.t
        first = t if first is None else first
        last_cal = t if last_cal is None else last_cal
        steps += 1
        for b in step.births:
            twin.discovery.note_birth(b.prefix, b.meta)
        tele = []
        for m in step.telemetry:
            if twin.tags.get(m.tag) is None:
                prop = twin.discovery.observe(m.tag, t) if m.quality != "test" else None
                if prop is not None and twin.apply_discovery(prop, t, author="auto: backfill").get("ok"):
                    discovered.extend(prop.assets)
                continue
            tele.append(m)
        est = twin.estimator
        D, R = twin.live.controls_from_paths(step.controls, step.routes)
        E = {k: torch.tensor([step.prices.get(k, float(twin.model.program.E0[k]))]) for k in twin.model.program.env_inputs}
        est.ingest(t, tele, D, R, E)
        est.propagate()
        if t - last_cal >= sc.calibrate_every - 1e-9:
            last_cal = t
            est.calibrate(iters=iters)
            calibrations += 1
        if progress:
            progress(t)
    twin.write_calibration(t)
    text = twin.repo.files.get("calibration.plx", "")
    return {"calibration": text, "steps": steps, "hours": 0.0 if first is None else t - first, "calibrations": calibrations,
            "properties": sum(1 for line in text.splitlines() if line.startswith("set ")), "discovered": discovered}
