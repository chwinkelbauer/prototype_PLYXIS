// PLYXIS native kernels for the deterministic-section solver (plyxis/optim/sections.py).
//
//   px_cluster          union-find clustering of the twin graph: merge along the strongest predicted
//                       couplings while a section's joint action space stays within the budget
//   px_section_scores   value of every candidate of every section: own exact Lagrangian change plus the
//                       linearised effect on the rest of the site (segment sums over instances / decisions)
//   px_argmax_sections  best candidate per section (exact enumeration)
//   px_viterbi          dynamic programme over time windows with switching costs, per section
//
// Built with:  g++ -O3 -march=native -fopenmp -shared -o _plyxis_native.(dll|so) sections.cpp
// No dependencies; called through ctypes (plyxis/native/__init__.py), which also holds a NumPy
// reference implementation of every routine.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#ifdef _WIN32
#define PX_API extern "C" __declspec(dllexport)
#else
#define PX_API extern "C"
#endif

static int find_root(std::vector<int>& p, int x) {
    while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; }
    return x;
}

// Edges sorted by weight (descending); a merge is accepted if the summed log action-space of the two
// sections stays <= log_budget and the edge is stronger than min_w. Returns the number of sections.
PX_API int px_cluster(int n_nodes, int n_edges, const int* src, const int* dst, const float* w,
                      const double* node_logcost, double log_budget, float min_w, int* out_label) {
    std::vector<int> order(n_edges);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return w[a] > w[b]; });
    std::vector<int> parent(n_nodes);
    std::iota(parent.begin(), parent.end(), 0);
    std::vector<double> cost(node_logcost, node_logcost + n_nodes);
    for (int k : order) {
        if (w[k] < min_w) break;
        int a = find_root(parent, src[k]), b = find_root(parent, dst[k]);
        if (a == b) continue;
        if (cost[a] + cost[b] > log_budget) continue;
        if (a < b) std::swap(a, b);
        parent[a] = b;
        cost[b] += cost[a];
    }
    std::vector<int> id(n_nodes, -1);
    int n = 0;
    for (int i = 0; i < n_nodes; ++i) {
        int r = find_root(parent, i);
        if (id[r] < 0) id[r] = n++;
        out_label[i] = id[r];
    }
    return n;
}

// out[b, c] = sum_{i in c} dl[b, i] + sum_{n in c} g[n] * dx[b, n]   (sections < 0 are ignored)
PX_API void px_section_scores(int B, int I, int N, int C, const float* dl, const int* inst_sec,
                              const float* dx, const float* g, const int* dec_sec, float* out) {
#pragma omp parallel for schedule(static)
    for (int b = 0; b < B; ++b) {
        float* o = out + (int64_t)b * C;
        for (int c = 0; c < C; ++c) o[c] = 0.0f;
        const float* d = dl + (int64_t)b * I;
        for (int i = 0; i < I; ++i) { int c = inst_sec[i]; if (c >= 0) o[c] += d[i]; }
        const float* x = dx + (int64_t)b * N;
        for (int n = 0; n < N; ++n) {
            float v = x[n];
            if (v != 0.0f) { int c = dec_sec[n]; if (c >= 0) o[c] += g[n] * v; }
        }
    }
}

// best[c] = argmax_{b < n_cand[c]} scores[b, c]; gain[c] = scores[best, c] - scores[0, c] (b = 0 is "keep")
PX_API void px_argmax_sections(int B, int C, const float* scores, const int* n_cand, int* best, float* gain) {
#pragma omp parallel for schedule(static)
    for (int c = 0; c < C; ++c) {
        int nb = std::min(B, std::max(1, n_cand[c]));
        int bi = 0;
        float bv = scores[c];
        for (int b = 1; b < nb; ++b) {
            float v = scores[(int64_t)b * C + c];
            if (v > bv) { bv = v; bi = b; }
        }
        best[c] = bi;
        gain[c] = bv - scores[c];
    }
}

// Per section c: maximise sum_k R[c,k,a_k] - S[c,a_{k-1},a_k] over valid actions (Viterbi).
PX_API void px_viterbi(int C, int K, int A, const float* R, const float* S, const uint8_t* valid, int* path) {
#pragma omp parallel for schedule(dynamic, 4)
    for (int c = 0; c < C; ++c) {
        const float NEG = -1e30f;
        std::vector<float> V(A), Vn(A);
        std::vector<int> back((size_t)K * A, 0);
        const float* Rc = R + (int64_t)c * K * A;
        const float* Sc = S + (int64_t)c * A * A;
        const uint8_t* vc = valid + (int64_t)c * K * A;
        for (int a = 0; a < A; ++a) V[a] = vc[a] ? Rc[a] : NEG;
        for (int k = 1; k < K; ++k) {
            for (int a = 0; a < A; ++a) {
                if (!vc[(int64_t)k * A + a]) { Vn[a] = NEG; continue; }
                float best = NEG; int arg = 0;
                for (int p = 0; p < A; ++p) {
                    if (V[p] <= NEG) continue;
                    float v = V[p] - Sc[(int64_t)p * A + a];
                    if (v > best) { best = v; arg = p; }
                }
                Vn[a] = best <= NEG ? NEG : best + Rc[(int64_t)k * A + a];
                back[(size_t)k * A + a] = arg;
            }
            std::swap(V, Vn);
        }
        int a = (int)(std::max_element(V.begin(), V.end()) - V.begin());
        for (int k = K - 1; k >= 0; --k) {
            path[(int64_t)c * K + k] = a;
            if (k > 0) a = back[(size_t)k * A + a];
        }
    }
}

PX_API int px_version() { return 1; }
