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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        sys.exit(0 if build() else 1)
    print("native kernels available:", available(), LIB)
