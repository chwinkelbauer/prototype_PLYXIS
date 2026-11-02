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

// Max-sum dynamic programme on a forest (exact): maximise sum_v U[v,z_v] + sum_k P[k,z_a,z_b].
// post = nodes with children before parents; parent[v] < 0 for roots; pedge[v] = edge to the parent;
// flip[v] = 1 if the parent is the edge's second node (P[k] is indexed [z_a, z_b]). O(m q^2).
PX_API void px_forest_dp(int n, int q, int m, const double* U, const double* P, const int* post,
                         const int* parent, const int* pedge, const uint8_t* flip, int* z) {
    std::vector<double> acc(U, U + (size_t)n * q);
    std::vector<int> arg((size_t)n * q, 0);
    for (int t = 0; t < n; ++t) {
        int v = post[t], p = parent[v];
        if (p < 0) continue;
        const double* Pk = P + (size_t)pedge[v] * q * q;
        const double* av = acc.data() + (size_t)v * q;
        double* ap = acc.data() + (size_t)p * q;
        int* gv = arg.data() + (size_t)v * q;
        for (int zp = 0; zp < q; ++zp) {
            double best = -1e300; int bi = 0;
            for (int zc = 0; zc < q; ++zc) {
                double tv = flip[v] ? Pk[(size_t)zc * q + zp] : Pk[(size_t)zp * q + zc];
                double s = tv + av[zc];
                if (s > best) { best = s; bi = zc; }
            }
            gv[zp] = bi;
            ap[zp] += best;
        }
    }
    for (int t = n - 1; t >= 0; --t) {
        int v = post[t], p = parent[v];
        if (p < 0) {
            const double* av = acc.data() + (size_t)v * q;
            z[v] = (int)(std::max_element(av, av + q) - av);
        } else {
            z[v] = arg[(size_t)v * q + z[p]];
        }
    }
    (void)m;
}

// Block coordinate ascent (Gauss-Seidel, exact per block) on a pairwise table model with shared capacities:
// every block in turn takes its best option given its neighbours, among the options that keep every
// capacity h within cap[h] (given the other blocks' use). U [n,q] (<= -1e29: no option), E [m,2], P [m,q,q],
// R [H,n,q], cap [H]; z [n] in/out. Monotone in the objective; returns the number of sweeps run.
PX_API int px_coordinate_ascent(int n, int q, int m, int H, const double* U, const int* E, const double* P,
                                const double* R, const double* cap, int* z, int sweeps) {
    std::vector<int> deg(n + 1, 0);
    for (int k = 0; k < m; ++k) { deg[E[2 * k] + 1]++; deg[E[2 * k + 1] + 1]++; }
    for (int i = 0; i < n; ++i) deg[i + 1] += deg[i];
    std::vector<int> nb_edge(2 * (size_t)m), nb_side(2 * (size_t)m), fill(deg.begin(), deg.end() - 1);
    for (int k = 0; k < m; ++k) {
        int a = E[2 * k], b = E[2 * k + 1];
        nb_edge[fill[a]] = k; nb_side[fill[a]++] = 0;
        nb_edge[fill[b]] = k; nb_side[fill[b]++] = 1;
    }
    std::vector<double> use(H, 0.0);
    for (int h = 0; h < H; ++h)
        for (int b = 0; b < n; ++b) use[h] += R[((size_t)h * n + b) * q + z[b]];
    std::vector<double> s(q);
    int sw = 0;
    for (; sw < sweeps; ++sw) {
        bool changed = false;
        for (int b = 0; b < n; ++b) {
            const double* Ub = U + (size_t)b * q;
            for (int j = 0; j < q; ++j) s[j] = Ub[j];
            for (int t = deg[b]; t < deg[b + 1]; ++t) {
                int k = nb_edge[t];
                const double* Pk = P + (size_t)k * q * q;
                if (nb_side[t] == 0) { int o = z[E[2 * k + 1]]; for (int j = 0; j < q; ++j) s[j] += Pk[(size_t)j * q + o]; }
                else                 { int o = z[E[2 * k]];     for (int j = 0; j < q; ++j) s[j] += Pk[(size_t)o * q + j]; }
            }
            int cur = z[b], best = cur;
            double bv = s[cur];
            for (int j = 0; j < q; ++j) {
                if (Ub[j] <= -1e29 || s[j] <= bv + 1e-12) continue;
                bool ok = true;
                for (int h = 0; h < H && ok; ++h) {
                    const double* Rh = R + ((size_t)h * n + b) * q;
                    if (use[h] - Rh[cur] + Rh[j] > cap[h] + 1e-9 && Rh[j] > Rh[cur]) ok = false;
                }
                if (ok) { bv = s[j]; best = j; }
            }
            if (best != cur) {
                for (int h = 0; h < H; ++h) {
                    const double* Rh = R + ((size_t)h * n + b) * q;
                    use[h] += Rh[best] - Rh[cur];
                }
                z[b] = best;
                changed = true;
            }
        }
        if (!changed) { ++sw; break; }
    }
    return sw;
}

// Sum-product on a forest at temperature tau (log space): returns tau * log Z of
// exp((sum_v U[v,z_v] + sum_k P[k,z_a,z_b]) / tau) and writes the node marginals marg [n,q].
// tau * log Z >= max (it is a smooth upper bound of the max-sum value); its gradient w.r.t. U is marg.
static inline double lse(const double* a, int q) {
    double m = -1e300;
    for (int i = 0; i < q; ++i) m = std::max(m, a[i]);
    if (m <= -1e299) return m;
    double s = 0.0;
    for (int i = 0; i < q; ++i) s += std::exp(a[i] - m);
    return m + std::log(s);
}

PX_API double px_forest_logz(int n, int q, const double* U, const double* P, const int* post, const int* parent,
                             const int* pedge, const uint8_t* flip, double tau, double* marg) {
    const double inv = 1.0 / tau;
    std::vector<double> acc((size_t)n * q), up((size_t)n * q, 0.0), down((size_t)n * q, 0.0), tmp(q);
    for (size_t i = 0; i < (size_t)n * q; ++i) acc[i] = U[i] <= -1e29 ? -1e300 : U[i] * inv;
    // upward pass: message v -> parent, stored in up[v]
    for (int t = 0; t < n; ++t) {
        int v = post[t], p = parent[v];
        if (p < 0) continue;
        const double* Pk = P + (size_t)pedge[v] * q * q;
        double* uv = up.data() + (size_t)v * q;
        const double* av = acc.data() + (size_t)v * q;
        for (int zp = 0; zp < q; ++zp) {
            for (int zc = 0; zc < q; ++zc) {
                double pv = flip[v] ? Pk[(size_t)zc * q + zp] : Pk[(size_t)zp * q + zc];
                tmp[zc] = pv * inv + av[zc];
            }
            uv[zp] = lse(tmp.data(), q);
        }
        double* ap = acc.data() + (size_t)p * q;
        for (int zp = 0; zp < q; ++zp) ap[zp] += uv[zp];
    }
    double logz = 0.0;
    // downward pass: message parent -> v (down[v]), then the marginals
    for (int t = n - 1; t >= 0; --t) {
        int v = post[t], p = parent[v];
        double* av = acc.data() + (size_t)v * q;
        if (p < 0) {
            logz += lse(av, q);
        } else {
            const double* Pk = P + (size_t)pedge[v] * q * q;
            const double* ap = acc.data() + (size_t)p * q;
            const double* dp = down.data() + (size_t)p * q;
            const double* uv = up.data() + (size_t)v * q;
            double* dv = down.data() + (size_t)v * q;
            for (int zc = 0; zc < q; ++zc) {
                for (int zp = 0; zp < q; ++zp) {
                    double pv = flip[v] ? Pk[(size_t)zc * q + zp] : Pk[(size_t)zp * q + zc];
                    tmp[zp] = pv * inv + ap[zp] - uv[zp] + dp[zp];     // parent's belief without v's message
                }
                dv[zc] = lse(tmp.data(), q);
            }
        }
    }
    for (int v = 0; v < n; ++v) {
        double* av = acc.data() + (size_t)v * q;
        const double* dv = down.data() + (size_t)v * q;
        for (int z = 0; z < q; ++z) tmp[z] = av[z] + dv[z];
        double l = lse(tmp.data(), q);
        for (int z = 0; z < q; ++z) marg[(size_t)v * q + z] = l <= -1e299 ? 0.0 : std::exp(tmp[z] - l);
    }
    return tau * logz;
}

PX_API int px_version() { return 4; }
