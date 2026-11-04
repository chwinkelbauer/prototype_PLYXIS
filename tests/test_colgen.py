"""Column generation with exact DP sub-problems (plyxis/optim/colgen.py) and learned prices (learnprice.py)."""
import itertools

import numpy as np
import pytest


def test_dp_is_the_exact_best_response():
    """The vectorised DP finds, for every asset, the best schedule at given prices - checked by brute force."""
    from plyxis.optim import colgen as CG
    fam, _ = CG.random_family(3, T=12, L=3, hold=6, seg=2, seed=1)
    pi = np.random.default_rng(0).uniform(0, 30, (1, 12))
    _, _, _, red = fam.best_response(pi)
    for b in range(3):
        one = CG.UnitFamily(fam.value[b:b + 1], fam.use[b:b + 1], fam.restart[b:b + 1], 6, 2, fam.was_on[b:b + 1])
        best = -np.inf
        for on in itertools.product([0, 1], repeat=2):
            for lev in itertools.product(range(3), repeat=6):
                L = np.array([[lev[s] if on[s // 3] else -1 for s in range(6)]])
                best = max(best, one.evaluate(L, pi)[3][0])
        assert red[b] == pytest.approx(best, rel=1e-9)


def test_bound_and_plan_against_exact_milp():
    """A feasible plan, a valid upper bound (above the MILP optimum), and a small gap."""
    from plyxis.optim import colgen as CG
    fam, cap = CG.random_family(10, seed=10)                   # small enough for the MILP to prove its optimum
    r = CG.solve(fam, cap)
    ex = CG.exact_milp(fam, cap, seconds=60)
    assert r.violation <= 1e-9
    assert ex["status"] == 0                                   # proven optimal
    assert r.bound >= ex["value"] - 1e-6 * abs(ex["value"])    # the Lagrangian bound is a bound
    assert r.value <= ex["value"] + 1e-6 * abs(ex["value"])
    assert (ex["value"] - r.value) / ex["value"] < 0.05
    assert r.prices.shape == cap.shape and (r.prices >= 0).all()


def test_gap_shrinks_with_size():
    """Shapley-Folkman: the absolute gap does not grow with the number of assets, so the relative gap falls."""
    from plyxis.optim import colgen as CG
    small = CG.solve(*CG.random_family(10, seed=1))
    large = CG.solve(*CG.random_family(160, seed=1))
    assert large.gap < small.gap


def test_learned_prices_seed_column_generation():
    from plyxis.optim import colgen as CG
    from plyxis.optim import learnprice as LP
    data = []
    for s in range(6):
        fam, cap = CG.random_family(30, seed=100 + s, cap_frac=0.4 + 0.05 * s)
        data.append(LP.target(fam, cap, CG.solve(fam, cap).prices))
    net = LP.fit(data, epochs=200)
    fam, cap = CG.random_family(30, seed=999, cap_frac=0.5)
    pi = LP.predict(net, fam, cap)
    assert pi.shape == cap.shape and (pi >= 0).all()
    warm, cold = CG.solve(fam, cap, warm_pi=pi), CG.solve(fam, cap)
    assert warm.violation <= 1e-9 and warm.bound >= warm.value
    assert abs(warm.value - cold.value) / cold.value < 0.02       # a start changes the effort, not the answer
