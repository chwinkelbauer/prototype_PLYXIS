"""Section dynamic programming (plyxis/optim/sectiondp.py) against brute force: exact on forests, a valid
certified bound on general graphs and with shared capacities (docs/DECOMPOSITION.md, theorems 2-4)."""
import itertools

import numpy as np

from plyxis.optim import sectiondp as SD


def _model(rng, n, q, extra=0, hub=False):
    edges = [tuple(sorted((int(rng.integers(0, v)), v))) for v in range(1, n)]   # a random tree
    free = [(a, b) for a in range(n) for b in range(a + 1, n) if (a, b) not in edges]
    for i in rng.permutation(len(free))[:extra]:                                 # plus edges that close cycles
        edges.append(free[int(i)])
    m = SD.TableModel([rng.normal(size=q) for _ in range(n)], edges,
                      [rng.normal(size=(q, q)) for _ in edges])
    if hub:
        m.usage = [[np.arange(q, dtype=float) % 3 for _ in range(n)]]
        m.cap = np.array([0.6 * n])
    return m


def _brute(m, q):
    best = -np.inf
    for z in itertools.product(range(q), repeat=m.n):
        z = np.array(z)
        if SD.feasible(m, z):
            best = max(best, SD.value(m, z))
    return best


def test_forest_dp_is_exact_on_trees():
    rng = np.random.default_rng(0)
    for _ in range(40):
        n, q = int(rng.integers(2, 8)), int(rng.integers(2, 4))
        m = _model(rng, n, q)
        s = SD.solve(m)
        opt = _brute(m, q)
        assert abs(s.value - opt) < 1e-9
        assert s.upper_bound >= opt - 1e-9 and s.gap < 1e-9          # certified optimal


def test_bound_holds_with_cycles_and_capacity():
    rng = np.random.default_rng(1)
    for _ in range(40):
        n, q = int(rng.integers(3, 8)), 3
        m = _model(rng, n, q, extra=int(rng.integers(1, 4)), hub=bool(rng.random() < 0.5))
        s = SD.solve(m)
        opt = _brute(m, q)
        assert s.feasible and s.value <= opt + 1e-9
        assert s.upper_bound >= opt - 1e-9                            # the certificate is never wrong


def test_smoothed_dual_is_a_valid_bound():
    rng = np.random.default_rng(2)
    for _ in range(12):
        n, q = int(rng.integers(3, 7)), 3
        m = _model(rng, n, q, extra=2, hub=True)
        s = SD.solve(m, dual="smooth")
        opt = _brute(m, q)
        assert s.feasible and s.value <= opt + 1e-9
        assert s.upper_bound >= opt - 1e-7                       # tau log-sum-exp >= max: never below the optimum
