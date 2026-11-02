"""Native (C++/OpenMP) kernels of the deterministic-section solver, with NumPy reference versions.

    python -m plyxis.native build        # compiles sections.cpp with g++ (MinGW-w64 on Windows)

The library is optional: every routine has a NumPy implementation with identical results, used when
the compiled library is missing (`available()` is False) or when `force_numpy(True)` is set (benchmarks).
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from typing import Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "_plyxis_native" + (".dll" if os.name == "nt" else ".so"))
SRC = os.path.join(HERE, "sections.cpp")
_lib: Optional[ctypes.CDLL] = None
_tried = False
_force_numpy = False

_i32p = ctypes.POINTER(ctypes.c_int32)
_f32p = ctypes.POINTER(ctypes.c_float)
_f64p = ctypes.POINTER(ctypes.c_double)
_u8p = ctypes.POINTER(ctypes.c_uint8)


def build(verbose: bool = True) -> bool:
    cxx = shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        if verbose:
            print("no C++ compiler found (g++ / clang++); the NumPy kernels are used")
        return False
    cmd = [cxx, "-O3", "-march=native", "-fopenmp", "-shared", "-std=c++17", "-o", LIB, SRC]
    if os.name == "nt":
        cmd[1:1] = ["-static-libgcc", "-static-libstdc++", "-static"]
    else:
        cmd.insert(1, "-fPIC")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if verbose:
        print(" ".join(cmd))
        print(r.stdout + r.stderr if r.returncode else f"built {LIB}")
    return r.returncode == 0


def _load() -> Optional[ctypes.CDLL]:
    global _lib, _tried
    if _tried:
        return _lib
    _tried = True
    if not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(SRC):
        if not build(verbose=False):
            return None
    try:
        lib = ctypes.CDLL(LIB)
    except OSError:
        return None
    lib.px_cluster.restype = ctypes.c_int
    lib.px_cluster.argtypes = [ctypes.c_int, ctypes.c_int, _i32p, _i32p, _f32p, _f64p, ctypes.c_double, ctypes.c_float, _i32p]
    lib.px_section_scores.argtypes = [ctypes.c_int] * 4 + [_f32p, _i32p, _f32p, _f32p, _i32p, _f32p]
    lib.px_argmax_sections.argtypes = [ctypes.c_int, ctypes.c_int, _f32p, _i32p, _i32p, _f32p]
    lib.px_viterbi.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, _f32p, _f32p, _u8p, _i32p]
    lib.px_forest_dp.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, _f64p, _f64p, _i32p, _i32p, _i32p,
                                 _u8p, _i32p]
    lib.px_coordinate_ascent.restype = ctypes.c_int
    lib.px_coordinate_ascent.argtypes = [ctypes.c_int] * 4 + [_f64p, _i32p, _f64p, _f64p, _f64p, _i32p, ctypes.c_int]
    if hasattr(lib, "px_forest_logz"):
        lib.px_forest_logz.restype = ctypes.c_double
        lib.px_forest_logz.argtypes = [ctypes.c_int, ctypes.c_int, _f64p, _f64p, _i32p, _i32p, _i32p, _u8p,
                                       ctypes.c_double, _f64p]
    _lib = lib
    return _lib


def available() -> bool:
    return not _force_numpy and _load() is not None


def force_numpy(flag: bool) -> None:
    global _force_numpy
    _force_numpy = bool(flag)


def _p(a: np.ndarray, t):
    return a.ctypes.data_as(t)


# ------------------------------------------------------------------ clustering
def cluster(n_nodes: int, src, dst, w, node_logcost, log_budget: float, min_w: float = 0.0) -> np.ndarray:
    src = np.ascontiguousarray(src, dtype=np.int32)
    dst = np.ascontiguousarray(dst, dtype=np.int32)
    w = np.ascontiguousarray(w, dtype=np.float32)
    cost = np.ascontiguousarray(node_logcost, dtype=np.float64)
    out = np.zeros(n_nodes, dtype=np.int32)
    if available():
        _lib.px_cluster(n_nodes, len(src), _p(src, _i32p), _p(dst, _i32p), _p(w, _f32p), _p(cost, _f64p),
                        float(log_budget), float(min_w), _p(out, _i32p))
        return out
    parent = np.arange(n_nodes)
    c = cost.copy()

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for k in np.argsort(-w, kind="stable"):
        if w[k] < min_w:
            break
        a, b = root(src[k]), root(dst[k])
        if a == b or c[a] + c[b] > log_budget:
            continue
        if a < b:
            a, b = b, a
        parent[a] = b
        c[b] += c[a]
    ids: dict = {}
    for i in range(n_nodes):
        out[i] = ids.setdefault(root(i), len(ids))
    return out


# ------------------------------------------------------------------ candidate scoring
def section_scores(dl: np.ndarray, inst_sec: np.ndarray, dx: np.ndarray, g: np.ndarray, dec_sec: np.ndarray,
                   n_sections: int) -> np.ndarray:
    """scores[b, c] = sum_{i in c} dl[b, i] + sum_{n in c} g[n] dx[b, n]."""
    dl = np.ascontiguousarray(dl, dtype=np.float32)
    dx = np.ascontiguousarray(dx, dtype=np.float32)
    g = np.ascontiguousarray(g, dtype=np.float32)
    inst_sec = np.ascontiguousarray(inst_sec, dtype=np.int32)
    dec_sec = np.ascontiguousarray(dec_sec, dtype=np.int32)
    B, I = dl.shape
    N = dx.shape[1]
    if available():
        out = np.zeros((B, n_sections), dtype=np.float32)
        _lib.px_section_scores(B, I, N, n_sections, _p(dl, _f32p), _p(inst_sec, _i32p), _p(dx, _f32p), _p(g, _f32p),
                               _p(dec_sec, _i32p), _p(out, _f32p))
        return out
    out = np.zeros((B, n_sections), dtype=np.float64)
    mi = inst_sec >= 0
    np.add.at(out.T, inst_sec[mi], dl[:, mi].T.astype(np.float64))
    md = dec_sec >= 0
    np.add.at(out.T, dec_sec[md], (dx[:, md] * g[md]).T.astype(np.float64))
    return out.astype(np.float32)


def argmax_sections(scores: np.ndarray, n_cand: np.ndarray):
    scores = np.ascontiguousarray(scores, dtype=np.float32)
    n_cand = np.ascontiguousarray(n_cand, dtype=np.int32)
    B, C = scores.shape
    best = np.zeros(C, dtype=np.int32)
    gain = np.zeros(C, dtype=np.float32)
    if available():
        _lib.px_argmax_sections(B, C, _p(scores, _f32p), _p(n_cand, _i32p), _p(best, _i32p), _p(gain, _f32p))
        return best, gain
    mask = np.arange(B)[:, None] < np.clip(n_cand, 1, B)[None, :]
    s = np.where(mask, scores, -np.inf)
    best = s.argmax(0).astype(np.int32)
    gain = (s[best, np.arange(C)] - scores[0]).astype(np.float32)
    return best, gain


# ------------------------------------------------------------------ dynamic programme over time windows
def viterbi(R: np.ndarray, S: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    """R [C, K, A] rewards, S [C, A, A] switching costs -> best action per window [C, K]."""
    R = np.ascontiguousarray(R, dtype=np.float32)
    C, K, A = R.shape
    S = np.ascontiguousarray(S, dtype=np.float32)
    valid = np.ones((C, K, A), dtype=np.uint8) if valid is None else np.ascontiguousarray(valid, dtype=np.uint8)
    path = np.zeros((C, K), dtype=np.int32)
    if available():
        _lib.px_viterbi(C, K, A, _p(R, _f32p), _p(S, _f32p), _p(valid, _u8p), _p(path, _i32p))
        return path
    NEG = -1e30
    V = np.where(valid[:, 0], R[:, 0], NEG)
    back = np.zeros((C, K, A), dtype=np.int32)
    for k in range(1, K):
        cand = V[:, :, None] - S                      # [C, prev, next]
        cand = np.where(V[:, :, None] <= NEG, NEG, cand)
        back[:, k] = cand.argmax(1)
        best = cand.max(1)
        V = np.where(valid[:, k] & (best > NEG), best + R[:, k], NEG)
    a = V.argmax(1)
    for k in range(K - 1, -1, -1):
        path[:, k] = a
        if k > 0:
            a = back[np.arange(C), k, a]
    return path


# ------------------------------------------------------------------ max-sum dynamic programme on a forest
def forest_dp(U: np.ndarray, P: np.ndarray, post: np.ndarray, parent: np.ndarray, pedge: np.ndarray,
              flip: np.ndarray) -> np.ndarray:
    """Exact argmax of sum_v U[v, z_v] + sum_k P[k, z_a, z_b] on a forest.
    U [n, q], P [m, q, q] (P[k] indexed [z_a, z_b] of edge k = (a, b)); post = nodes children-first;
    parent[v] (-1 for roots), pedge[v] = edge to the parent, flip[v] = 1 if the parent is b of that edge."""
    U = np.ascontiguousarray(U, dtype=np.float64)
    P = np.ascontiguousarray(P, dtype=np.float64)
    n, q = U.shape
    post = np.ascontiguousarray(post, dtype=np.int32)
    parent = np.ascontiguousarray(parent, dtype=np.int32)
    pedge = np.ascontiguousarray(pedge, dtype=np.int32)
    flip = np.ascontiguousarray(flip, dtype=np.uint8)
    z = np.zeros(n, dtype=np.int32)
    if n == 0:
        return z
    if available():
        _lib.px_forest_dp(n, q, len(P), _p(U, _f64p), _p(P, _f64p), _p(post, _i32p), _p(parent, _i32p),
                          _p(pedge, _i32p), _p(flip, _u8p), _p(z, _i32p))
        return z
    acc = U.copy()
    arg = np.zeros((n, q), dtype=np.int32)
    for v in post:
        p = parent[v]
        if p < 0:
            continue
        T = P[pedge[v]].T if flip[v] else P[pedge[v]]          # [z_parent, z_child]
        s = T + acc[v][None, :]
        arg[v] = s.argmax(1)
        acc[p] += s.max(1)
    for v in post[::-1]:                                        # parents before children
        p = parent[v]
        z[v] = int(acc[v].argmax()) if p < 0 else int(arg[v][z[p]])
    return z


# ------------------------------------------------------------------ smoothed forest value (sum-product)
def forest_logz(U: np.ndarray, P: np.ndarray, post: np.ndarray, parent: np.ndarray, pedge: np.ndarray,
                flip: np.ndarray, tau: float) -> tuple:
    """tau * log sum_z exp((sum_v U[v,z_v] + sum_k P[k,z_a,z_b]) / tau) on a forest, and the node marginals
    [n, q]. It is >= the max-sum value for every tau > 0 (a smooth upper bound); its gradient is the marginals."""
    U = np.ascontiguousarray(U, dtype=np.float64)
    P = np.ascontiguousarray(P, dtype=np.float64)
    n, q = U.shape
    post = np.ascontiguousarray(post, dtype=np.int32)
    parent = np.ascontiguousarray(parent, dtype=np.int32)
    pedge = np.ascontiguousarray(pedge, dtype=np.int32)
    flip = np.ascontiguousarray(flip, dtype=np.uint8)
    marg = np.zeros((n, q))
    if available() and hasattr(_lib, "px_forest_logz"):
        v = _lib.px_forest_logz(n, q, _p(U, _f64p), _p(P, _f64p), _p(post, _i32p), _p(parent, _i32p),
                                _p(pedge, _i32p), _p(flip, _u8p), float(tau), _p(marg, _f64p))
        return float(v), marg

    def lse(a, axis=None):
        m = np.max(a, axis=axis, keepdims=True)
        m = np.where(m <= -1e299, 0.0, m)
        return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(a - m), axis=axis))
    acc = np.where(U <= -1e29, -1e300, U / tau)
    up = np.zeros((n, q))
    for v in post:
        p = parent[v]
        if p < 0:
            continue
        T = (P[pedge[v]].T if flip[v] else P[pedge[v]]) / tau        # [z_parent, z_child]
        up[v] = lse(T + acc[v][None, :], axis=1)
        acc[p] += up[v]
    down = np.zeros((n, q))
    logz = 0.0
    for v in post[::-1]:
        p = parent[v]
        if p < 0:
            logz += float(lse(acc[v]))
        else:
            T = (P[pedge[v]].T if flip[v] else P[pedge[v]]) / tau
            down[v] = lse(T + (acc[p] - up[v] + down[p])[:, None], axis=0)
    b = acc + down
    marg = np.exp(b - lse(b, axis=1)[:, None])
    return tau * logz, marg


# ------------------------------------------------------------------ block coordinate ascent
def coordinate_ascent(U: np.ndarray, E: np.ndarray, P: np.ndarray, R: np.ndarray, cap: np.ndarray,
                      z: np.ndarray, sweeps: int) -> np.ndarray:
    """Gauss-Seidel over blocks, exact per block given its neighbours, within the shared capacities
    (a move may not raise a capacity that is, or would be, exceeded). U [n,q], E [m,2], P [m,q,q],
    R [H,n,q], cap [H]. Returns the improved z."""
    U = np.ascontiguousarray(U, dtype=np.float64)
    n, q = U.shape
    E = np.ascontiguousarray(np.asarray(E).reshape(-1, 2), dtype=np.int32)
    m = len(E)
    P = np.ascontiguousarray(P if m else np.zeros((1, q, q)), dtype=np.float64)
    R = np.ascontiguousarray(R, dtype=np.float64).reshape(-1, n, q)
    H = R.shape[0]
    cap = np.ascontiguousarray(cap, dtype=np.float64).reshape(-1)
    z = np.ascontiguousarray(z, dtype=np.int32).copy()
    if available():
        _lib.px_coordinate_ascent(n, q, m, H, _p(U, _f64p), _p(E, _i32p), _p(P, _f64p),
                                  _p(R if H else np.zeros(1), _f64p), _p(cap if H else np.zeros(1), _f64p),
                                  _p(z, _i32p), int(sweeps))
        return z.astype(np.int64)
    nb = [[] for _ in range(n)]
    for k in range(m):
        nb[E[k, 0]].append((k, 0))
        nb[E[k, 1]].append((k, 1))
    use = R[:, np.arange(n), z].sum(1) if H else np.zeros(0)
    for _ in range(sweeps):
        changed = False
        for b in range(n):
            s = U[b].copy()
            for k, side in nb[b]:
                s += P[k][:, z[E[k, 1]]] if side == 0 else P[k][z[E[k, 0]], :]
            cur = int(z[b])
            best, bv = cur, s[cur]
            for j in range(q):
                if U[b, j] <= -1e29 or s[j] <= bv + 1e-12:
                    continue
                if any(use[h] - R[h, b, cur] + R[h, b, j] > cap[h] + 1e-9 and R[h, b, j] > R[h, b, cur]
                       for h in range(H)):
                    continue
                best, bv = j, s[j]
            if best != cur:
                use = use + R[:, b, best] - R[:, b, cur] if H else use
                z[b] = best
                changed = True
        if not changed:
            break
    return z.astype(np.int64)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        sys.exit(0 if build() else 1)
    print("native kernels available:", available(), LIB)
