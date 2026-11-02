"""PLYXIS command line.

    python -m plyxis serve [--port 8000] [--speed 600] [--autoplay] [--no-neural]
    python -m plyxis compile <file.plx> [--env NAME] [--show-code]
    python -m plyxis lint <file.plx> [--env NAME]     checks beyond the compiler (dead decisions, unconnected inputs, ...)
    python -m plyxis optimize <file.plx> [--env NAME]
    python -m plyxis run [--hours 24]                 headless showcase run (A/B KPIs)
    python -m plyxis backfill <history.csv.gz> --scenario NAME [--out calibration.plx]
                                                      calibrate a fresh twin from a historian export (data/historian.py)
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="plyxis")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--speed", type=float, default=600.0)
    s.add_argument("--autoplay", action="store_true")
    s.add_argument("--no-neural", action="store_true")
    s.add_argument("--scenario", default=None)
    c = sub.add_parser("compile")
    c.add_argument("files", nargs="+")
    c.add_argument("--env")
    c.add_argument("--show-code", action="store_true")
    l = sub.add_parser("lint")
    l.add_argument("files", nargs="+")
    l.add_argument("--env")
    o = sub.add_parser("optimize")
    o.add_argument("files", nargs="+")
    o.add_argument("--env")
    b = sub.add_parser("backfill")
    b.add_argument("history")
    b.add_argument("--scenario", required=True)
    b.add_argument("--out", default="calibration.plx")
    r = sub.add_parser("run")
    r.add_argument("--hours", type=float, default=24.0)
    args = ap.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn
        from .server.app import create_app
        app = create_app(args.scenario, speed=args.speed, autoplay=args.autoplay, use_neural=not args.no_neural)
        print(f"PLYXIS console: http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    elif args.cmd == "compile":
        from .lang import compile_model
        from .lang.errors import PlxError
        t0 = time.perf_counter()
        try:
            m = compile_model(args.files, env=args.env)
        except PlxError as ex:
            for d in ex.diagnostics:
                print(d.format(ex.sources))
            sys.exit(1)
        for d in m.diagnostics.items:
            if d.severity != "info":
                print(d.format(m.diagnostics.sources))
        print(f"compiled in {time.perf_counter() - t0:.2f}s: {m.program.summary()}")
        if args.show_code:
            print(m.program.source)
    elif args.cmd == "lint":
        from .lang import compile_model
        from .lang.errors import PlxError
        from .lang.lint import lint
        try:
            m = compile_model(args.files, env=args.env)
        except PlxError as ex:
            for d in ex.diagnostics:
                print(d.format(ex.sources))
            sys.exit(2)
        found = lint(m)
        for f in found:
            print(f"{f.severity:8s}{f.code:20s}{f.message}")
        warnings = sum(1 for f in found if f.severity == "warning")
        print(f"{warnings} warnings, {len(found) - warnings} for review")
        sys.exit(1 if warnings else 0)
    elif args.cmd == "backfill":
        from .runtime.backfill import backfill
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        res = backfill(os.path.join(root, "scenarios", args.scenario), args.history, progress=lambda t: None)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(res["calibration"])
        print(f"{res['hours']:.0f} h replayed, {res['steps']} steps, {res['calibrations']} calibrations, "
              f"{res['properties']} properties differ from the engineering values, assets discovered: {res['discovered'] or 'none'}")
        print(f"calibration written to {args.out}: add it to the twin's files in scenario.json")
    elif args.cmd == "optimize":
        from .lang import compile_model
        from .optim.cascade import Cascade
        from .optim.problem import Problem
        m = compile_model(args.files, env=args.env)
        pb = Problem(m.program)
        res = Cascade(pb).solve(callback=lambda r: print(r) if r.get("event") or r.get("stage") == "done" else None)
        print(f"objective {res.objective:,.0f}  max violation {res.max_violation:.4f}  in {res.seconds:.1f}s")
        for st in res.stages:
            print("  ", st)
    elif args.cmd == "run":
        from .runtime.engine import Engine
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        eng = Engine(os.path.join(root, "scenarios", "kestrel"), sync_plans=True)

        def prog(e):
            k = e.kpis()
            print(f"{e.clock_str()}  value captured {k['delta']['profit']:12,.0f} USD   "
                  f"oil +{k['delta']['oil_bbl']:9,.0f} bbl   flaring {k['delta']['flared_mscf']:+9,.0f} Mscf")
        eng.run_headless(args.hours, progress=prog)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
