"""Planner: rolling-horizon re-optimisation of the live twin.

* builds the optimisation problem from the live estimate (parameters, states) and forecasts
* warm-starts from the previous plan shifted in time (and remapped after twin rewrites)
* runs the cascade (neural sub-solvers -> relaxation -> discrete search -> polish)
* derives maintenance jobs (counterfactual value) and schedules rigs / crews
* online expert iteration: every plan becomes training data for the neural sub-solvers
"""
from __future__ import annotations

import dataclasses
import os
import threading
import time
from collections import deque
from typing import Callable, Optional

import torch

from ..optim import maintenance as MAINT
from ..optim import neural as NN
from ..optim.cascade import Cascade, CascadeConfig, CascadeResult
from ..optim.gradient import alm, shadow_prices
from ..optim.partition import build_partition
from ..optim.plan import Plan, build_plan
from ..optim.problem import Problem
from ..optim.space import DecisionSpace


def remap_x(old_space: DecisionSpace, new_space: DecisionSpace, x: torch.Tensor) -> torch.Tensor:
    """Carry a plan across a recompilation (by asset path and decision name)."""
    out = new_space.default_u(1)
    po, pn = old_space.split(x), new_space.split(out)
    op, np_ = old_space.program, new_space.program
    for b in new_space.blocks:
        g = np_.gmap[b.gid]
        for li, iid in enumerate(g.inst_ids):
            path = np_.flat.instances[iid].path
            oiid = op.flat.by_path.get(path)
            if oiid is None:
                continue
            ogid, oli = op.inst_group[oiid]
            if (ogid, b.name) in po and po[(ogid, b.name)].shape[2] == b.H:
                pn[(b.gid, b.name)][0, li] = po[(ogid, b.name)][0, oli]
    if new_space.route is not None and old_space.route is not None and new_space.route.H == old_space.route.H:
        okey = _route_keys(op)
        for key, r in _route_keys(np_).items():
            if key in okey:
                pn["route"][0, r] = po["route"][0, okey[key]]
    return out


def _route_keys(prog) -> dict:
    flat = prog.flat
    out = {}
    for rp in prog.routed:
        src = flat.instances[rp.inst].path
        for r, eid in zip(rp.ridx, rp.edges):
            dp = flat.ports[flat.edges[eid].dst]
            out[(src, rp.name, flat.instances[dp.inst].path, dp.name)] = r
    return out


class Planner:
    # live budget (~10 s for the Kestrel twin on a laptop CPU), set by bench/solver_budget.py
    LIVE = CascadeConfig(relax_iters=200, population=12, generations=1, repair_iters=80, polish_iters=120,
                         multistart=8, neural_iters=1)

    def __init__(self, twin, cfg: Optional[CascadeConfig] = None, net_path: Optional[str] = None,
                 online_learning: bool = True):
        self.twin = twin
        self.cfg = cfg or dataclasses.replace(self.LIVE)
        self.net_path = net_path
        self.online = online_learning
        self.move_lock_h = 3.0          # move suppression: decisions starting within this window ...
        self.move_cost = 1500.0         # ... cost USD per unit^2 of normalised change to revise
        self.plan: Optional[Plan] = None
        self.plans: deque = deque(maxlen=50)
        self.runs: deque = deque(maxlen=30)
        self.progress: list = []
        self.running = False
        self.pending: Optional[str] = None
        self.lock = threading.Lock()
        self._version = None
        self.base_pb: Optional[Problem] = None
        self.net = None
        self.net_info: dict = {"loaded": False, "trained_on": 0, "updates": 0}
        self.experts: deque = deque(maxlen=200)
        self.committed_jobs: dict = {}        # job id -> job dict (started / done)
        self.pid = 0
        self.decoded = None

    # ------------------------------------------------------------------ problem
    def _ensure_problem(self) -> None:
        if self._version == self.twin.version and self.base_pb is not None:
            return
        old_pb = self.base_pb
        prog = self.twin.model.program
        self.base_pb = Problem(prog)
        self.part = build_partition(prog, self.base_pb.space)
        old_net = self.net
        if old_net is not None or (self.net_path and os.path.exists(self.net_path)):
            graph = NN.TwinGraph(self.base_pb)
            base_x = None
            if old_net is not None and old_pb is not None:
                base_x = remap_x(old_pb.space, self.base_pb.space, torch.sigmoid(old_net.base_logit))
            net = NN.SubSolverNet(graph, d=32, rounds=3, slot=4, base_x=base_x)
            state = old_net.state_dict() if old_net is not None else \
                torch.load(self.net_path, map_location="cpu", weights_only=False)["state"]
            own = net.state_dict()
            state = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
            missing, unexpected = net.load_state_dict(state, strict=False)
            self.net = net
            self.net_info["loaded"] = True
            self.net_info["missing"] = len([m for m in missing if m != "base_logit"])
        self._version = self.twin.version

    def problem(self, t_clock: float, E: dict) -> Problem:
        self._ensure_problem()
        live = self.twin.live
        P = {g: {k: v.clone() for k, v in d.items()} for g, d in live.P.items()}
        S = {g: {k: v.clone() for k, v in d.items()} for g, d in live.S.items()}
        return self.base_pb.with_context(P=P, S0=S, E=E, t0=t_clock)

    # ------------------------------------------------------------------ solve
    def solve(self, t_clock: float, E: dict, trigger: str, busy: Optional[dict] = None,
              callback: Optional[Callable] = None, worked_assets: Optional[set] = None,
              predicted: Optional[dict] = None, urgent: bool = False, quick_first: bool = False,
              on_interim: Optional[Callable] = None) -> Plan:
        with self.lock:
            self.running = True
            self.progress = []
            t_wall = time.time()
            try:
                pb = self.problem(t_clock, E)
                warm = None
                if self.plan is not None:
                    prev_space = getattr(self, "_plan_space", None)
                    x = self.plan.x
                    if prev_space is not None and prev_space is not pb.space:
                        x = remap_x(prev_space, pb.space, x)
                    warm = pb.space.shift(x, t_clock - self.plan.t0)
                    # move suppression: keep what the plant executes now unless a change is worth it
                    # (lighter after disruptions, which need fast reactions)
                    pb.set_move_reference(warm, lock_hours=self.move_lock_h,
                                          cost_usd=self.move_cost * (0.25 if urgent else 1.0))

                def cb(rec):
                    self.progress.append(rec)
                    if callback:
                        callback(rec)
                cfg = self.cfg
                start = warm
                if quick_first and warm is not None:
                    # anytime planning: a first response in about a second, the full cascade continues
                    q = self._first_response(pb, warm, cb)
                    if q is not None:
                        interim = build_plan(pb, q, self.pid + 1, f"{trigger} - first response", t_wall,
                                             jobs=list(self.plan.jobs) if self.plan is not None else [])
                        D0, R0 = pb.space.decode(q.x, pb.P, relax=False)
                        interim.decoded = ({g: {k: v.detach() for k, v in d.items()} for g, d in D0.items()},
                                           R0.detach() if R0 is not None else None)
                        interim.problem = pb
                        interim.interim = True
                        self.plan = interim
                        self._plan_space = pb.space
                        start = q.x
                        pb.set_move_reference(q.x, lock_hours=self.move_lock_h, cost_usd=self.move_cost * 0.25)
                        if on_interim:
                            on_interim(interim)
                cas = Cascade(pb, net=self.net, partition=self.part, config=cfg)
                res = cas.solve(warm=start, callback=cb)
                # maintenance: value repairs with the twin, schedule rigs and crews
                jobs = MAINT.find_jobs(pb, res.x, predicted=predicted)
                jobs = [j for j in jobs if j.jid not in self.committed_jobs and
                        j.asset_path not in (worked_assets or set())]
                resources = MAINT.resources_of(pb, busy)
                sched = MAINT.schedule(jobs, resources, now=t_clock) if jobs else []
                self.pid += 1
                plan = build_plan(pb, res, self.pid, trigger, t_wall, jobs=sched)
                D, R = pb.space.decode(res.x, pb.P, relax=False)
                plan.decoded = ({g: {k: v.detach() for k, v in d.items()} for g, d in D.items()},
                                R.detach() if R is not None else None)
                plan.problem = pb
                self.plan = plan
                self._plan_space = pb.space
                self.plans.append(plan)
                self.runs.append({"id": plan.pid, "t": t_clock, "trigger": trigger, "objective": plan.objective,
                                  "feasible": plan.feasible, "seconds": res.seconds, "stages": res.stages,
                                  "evals": res.evals, "history": res.history[-400:]})
                if self.online and self.net is not None:
                    ctx = NN.Context(pb.P, pb.S0, {k: v.view(1, -1) if isinstance(v, torch.Tensor) else v
                                                   for k, v in pb.E.items()}, None, t_clock, 1)
                    self.experts.append((ctx, res.x.detach().clone()))
                return plan
            finally:
                self.running = False

    def _first_response(self, pb: Problem, warm: torch.Tensor, cb: Optional[Callable] = None) -> Optional[CascadeResult]:
        """Fast feasible re-plan after a disruption: repair the running plan (and the neural proposal)
        with a short gradient run on the exact twin, discrete decisions held."""
        t = time.perf_counter()
        sp = pb.space
        mask = sp.discrete_mask()
        starts = [sp.round_discrete(warm.view(1, -1))]
        if self.net is not None and self.cfg.use_neural:
            try:
                from ..optim.neural import dual_coordination
                starts.append(sp.round_discrete(dual_coordination(pb, self.net, iters=1).x))
            except Exception:  # noqa: BLE001 - the neural start is optional
                pass
        r = alm(pb, torch.cat(starts, 0), iters=60, lr=0.02, rho=0.5, outer=10, mask=~mask, relax=False,
                stage="first response", callback=cb)
        secs = time.perf_counter() - t
        return CascadeResult(r.x, r.objective, r.max_violation, r.feasible, shadow_prices(pb, r.multipliers),
                             r.multipliers, r.history, [{"stage": "first response", "seconds": secs,
                                                         "objective": r.objective, "max_violation": r.max_violation}],
                             secs, r.evals)

    def learn(self, steps: int = 25) -> Optional[dict]:
        """Online expert iteration on the plans produced so far (runs after a plan is published)."""
        if self.net is None or len(self.experts) < 4:
            return None
        with self.lock:
            pb = self.base_pb
            ex = list(self.experts)
            if any(e[1].shape[1] != pb.space.N for e in ex):
                ex = [e for e in ex if e[1].shape[1] == pb.space.N]
                if len(ex) < 4:
                    return None
            try:
                logs = NN.train_imitation(pb, self.net, ex, steps=steps, B=min(4, len(ex)), lr=5e-4, ss_weight=0.0)
            except Exception as exc:  # noqa: BLE001 - learning must never break operations
                return {"error": str(exc)}
            self.net_info["updates"] += 1
            self.net_info["trained_on"] = len(ex)
            return {"steps": steps, "examples": len(ex), "loss": logs[-1]["loss"] if logs else None}

    # ------------------------------------------------------------------ execution helpers
    def setpoints_at(self, t_clock: float) -> tuple:
        """Plan decisions at sim time t as {(path, decision): value} and route splits by key."""
        plan = self.plan
        if plan is None or getattr(plan, "decoded", None) is None:
            return {}, {}
        pb = plan.problem
        prog = pb.prog
        D, R = plan.decoded
        k = int(max(0, min(pb.T - 1, (t_clock - plan.t0) // pb.dt)))
        vals = {}
        for g in prog.groups:
            if g.is_env:
                continue
            for dname, t in D.get(g.gid, {}).items():
                col = t[0, :, k] if t.shape[-1] > k else t[0, :, -1]
                for li, iid in enumerate(g.inst_ids):
                    vals[(prog.flat.instances[iid].path, dname)] = float(col[li])
        routes = {}
        if R is not None:
            for key, r in _route_keys(prog).items():
                routes[key] = float(R[0, r, min(k, R.shape[-1] - 1)])
        return vals, routes
