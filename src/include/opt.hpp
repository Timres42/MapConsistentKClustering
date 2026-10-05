// "opt" algorithm: the exact optimum of the center-identity variant of
// label-consistent k-median, solved with Gurobi (C API).
//
// Variant: given a historical clustering H (every point p has a historical
// center point h(p)) and a budget b, find k center points C and an assignment
// sigma minimizing sum_p d(p, sigma(p)) such that at most b points have
// sigma(p) != h(p). "Center" means the explicit point: if h(p) is not in C,
// p is counted, even if it stays in the "same" cluster slot.
// Without a historical clustering (hist == nullptr) this is plain k-median.
//
// Two exact formulations are available (both give the same optimum):
//
// "benders" (default) - Benders / radius formulation, after Duran-Mateluna,
// Ales & Elloumi, "An efficient Benders decomposition for the p-median
// problem", EJOR 2023, extended by the budget. Only 3n variables:
//   y_j      1 if point j is a center
//   s_i      1 if point i stays at its historical center h(i)
//   theta_i  cost of point i
//   min  sum_i theta_i
//   s.t. sum_j y_j = k,   s_i <= y_{h(i)},   sum_i s_i >= n - b
//        theta_i >= D - sum_{j: d(i,j) < D} (D - d(i,j)) y_j
//                       + [D < d(i,h(i))] (d(i,h(i)) - D) s_i     for every threshold D
//   For integer y the strongest of these cuts equals dist_i + s_i (d(i,h(i)) - dist_i),
//   with dist_i the distance to the nearest open center: d(i, h(i)) if i stays,
//   dist_i if it moves. Only y is binary: for fixed y the remaining problem in s
//   (choose >= n - b points to stay, unit weights) has an integral LP optimum, so
//   s can stay continuous without changing the optimum.
//   The cuts are generated on demand: first in a cutting-plane loop on the LP
//   relaxation, then as lazy constraints / user cuts inside branch-and-bound.
//
// "classic" - the compact assignment MIP with n^2 + n variables:
//   min  sum_ij d(i,j) x_ij
//   s.t. sum_j x_ij = 1,  x_ij <= y_j,  sum_j y_j = k,  sum_i x_{i,h(i)} >= n - b
//        y_j binary, x_{i,h(i)} binary, all other x_ij in [0, 1]
//   Kept for cross-checking; much slower beyond a few hundred points.
//
// Build with -DUSE_GUROBI and Gurobi's include/lib paths (see Makefile).
#pragma once
#include "core.hpp"
#include "form.hpp"  // MappedClustering
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <atomic>

struct OptInfo {
    bool solved = false;      // true if a MIP solution was obtained
    bool optimal = false;     // proven optimal (within MIPGap)
    int status = -1;          // Gurobi status code
    double mip_gap = std::numeric_limits<double>::quiet_NaN();
    double bound = std::numeric_limits<double>::quiet_NaN();  // best lower bound
    double runtime = 0.0;
    int budget_used = 0;      // points whose center point differs from h(p)
    // benders diagnostics
    int lp_rounds = 0;        // cutting-plane rounds on the LP relaxation
    long cuts = 0;            // Benders cuts added in total
    double lp_bound = std::numeric_limits<double>::quiet_NaN();  // LP relaxation value
    double lp_time = 0.0;     // seconds spent on the LP relaxation
};

// Number of points whose center point differs from their historical center point.
inline int center_point_changes(const Clustering& hist, const Clustering& c) {
    int cnt = 0;
    for (int p = 0; p < hist.n; ++p)
        if (hist.center_of(p) != c.center_of(p)) ++cnt;
    return cnt;
}

// Put the chosen center points into k slots: a center that is also a historical
// center keeps that historical slot, the others fill the remaining slots in order.
inline std::vector<int> order_centers_by_slot(const std::vector<int>& centers,
                                              const Clustering* hist, int k) {
    if (!hist) return centers;
    std::vector<int> slots(k, -1);
    std::vector<char> placed(centers.size(), 0);
    for (int s = 0; s < k; ++s)
        for (size_t c = 0; c < centers.size(); ++c)
            if (!placed[c] && centers[c] == hist->centers[s]) { slots[s] = centers[c]; placed[c] = 1; break; }
    size_t next = 0;
    for (int s = 0; s < k; ++s) {
        if (slots[s] != -1) continue;
        while (next < centers.size() && placed[next]) ++next;
        slots[s] = centers[next];
        placed[next] = 1;
    }
    return slots;
}

// Exact optimal assignment for FIXED center points under the variant.
// Points whose historical center is not open must move (to their nearest center);
// the rest of the budget goes to the points with the largest savings
// d(p, h(p)) - d(p, nearest center). Returns false if the forced moves exceed b.
inline bool assign_with_budget(const Points& P, const Clustering* hist,
                               const std::vector<int>& centers, int budget,
                               std::vector<int>& assignment) {
    int k = (int)centers.size();
    std::vector<int> nearest = assign_points(P, centers);
    if (!hist) { assignment = nearest; return true; }

    std::unordered_map<int, int> slot_of;  // center point -> slot (first occurrence)
    for (int s = k - 1; s >= 0; --s) slot_of[centers[s]] = s;

    assignment.assign(P.n, 0);
    int forced = 0;
    struct Move { double savings; int p; };
    std::vector<Move> moves;
    for (int p = 0; p < P.n; ++p) {
        int hc = hist->center_of(p);
        auto it = slot_of.find(hc);
        if (it == slot_of.end()) { assignment[p] = nearest[p]; ++forced; continue; }
        assignment[p] = it->second;
        double savings = dist(P, p, hc) - dist(P, p, centers[nearest[p]]);
        if (savings > 0) moves.push_back({savings, p});
    }
    if (forced > budget) return false;
    std::stable_sort(moves.begin(), moves.end(),
                     [](const Move& a, const Move& b) { return a.savings > b.savings; });
    int allowed = std::min(budget - forced, (int)moves.size());
    for (int m = 0; m < allowed; ++m) assignment[moves[m].p] = nearest[moves[m].p];
    return true;
}

#ifdef USE_GUROBI
extern "C" {
#include "gurobi_c.h"
}

inline void grb_check(int err, GRBenv* env, const char* what) {
    if (err) {
        std::string msg = std::string("Gurobi error in ") + what + ": " + (env ? GRBgeterrormsg(env) : "")
                          + " (code " + std::to_string(err) + ")";
        throw std::runtime_error(msg);
    }
}

inline double opt_seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// Best feasible solution among the given center sets (empty centers if none is feasible).
struct OptStart {
    std::vector<int> centers, assignment;
    double cost = std::numeric_limits<double>::infinity();
};

inline OptStart opt_best_start(const Points& P, int k, const Clustering* hist, int budget,
                               const std::vector<std::vector<int>>& start_center_sets) {
    OptStart best;
    for (const auto& cs : start_center_sets) {
        if ((int)cs.size() != k) continue;
        std::vector<int> ordered = order_centers_by_slot(cs, hist, k);
        std::vector<int> a;
        if (!assign_with_budget(P, hist, ordered, budget, a)) continue;
        double c = Clustering(P, ordered, a).cost();
        if (c < best.cost) { best.cost = c; best.centers = ordered; best.assignment = a; }
    }
    return best;
}

// Turns the solver's center set into the final clustering (exact assignment for
// those centers), or falls back to the start solution / historical clustering.
inline MappedClustering opt_finish(const Points& P, int k, const Clustering* hist, int budget,
                                   const std::vector<int>& centers, const OptStart& start,
                                   std::chrono::steady_clock::time_point t0, OptInfo& info) {
    Clustering result;
    if (info.solved) {
        std::vector<int> ordered = order_centers_by_slot(centers, hist, k);
        std::vector<int> assignment;
        if (assign_with_budget(P, hist, ordered, budget, assignment))
            result = Clustering(P, ordered, assignment);
        else
            info.solved = false;
    }
    if (!info.solved) {
        if (!start.centers.empty()) result = Clustering(P, start.centers, start.assignment);
        else if (hist) result = *hist;
        else throw std::runtime_error("opt: no solution found and no start solution available");
        std::fprintf(stderr, "  [opt] no MIP solution (status %d), using best start solution\n", info.status);
    }
    info.budget_used = hist ? center_point_changes(*hist, result) : 0;
    info.runtime = opt_seconds_since(t0);
    std::vector<int> identity(k);
    std::iota(identity.begin(), identity.end(), 0);
    return {result, identity};
}

// Picks the k largest y values as centers.
inline std::vector<int> opt_top_k(const std::vector<double>& y, int k) {
    std::vector<int> order(y.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return y[a] > y[b]; });
    std::vector<int> centers(order.begin(), order.begin() + k);
    std::sort(centers.begin(), centers.end());
    return centers;
}

inline GRBenv* opt_new_env() {
    GRBenv* env = nullptr;
    grb_check(GRBemptyenv(&env), nullptr, "GRBemptyenv");
    grb_check(GRBsetintparam(env, "OutputFlag", 0), env, "OutputFlag");
    grb_check(GRBstartenv(env), env, "GRBstartenv (license?)");
    return env;
}

// ---------------------------------------------------------------------------
// "classic": compact assignment formulation with n^2 + n variables.
// ---------------------------------------------------------------------------
inline MappedClustering opt_lcc_median_classic(
        const Points& P, int k, const Clustering* hist, int budget,
        const std::vector<std::vector<int>>& start_center_sets,
        double time_limit, double mip_gap, int threads, OptInfo& info) {
    auto t0 = std::chrono::steady_clock::now();
    const int n = P.n;
    const size_t nn = (size_t)n * n;
    const size_t nvars = nn + n;
    auto X = [n](int i, int j) { return (int)((size_t)i * n + j); };
    auto Y = [nn](int j) { return (int)(nn + j); };
    if (nvars > (size_t)std::numeric_limits<int>::max())
        throw std::runtime_error("opt: instance too large for the classic MIP (n^2 + n > INT_MAX)");

    std::vector<int> hcenter(n, -1);
    if (hist) for (int p = 0; p < n; ++p) hcenter[p] = hist->center_of(p);
    OptStart start = opt_best_start(P, k, hist, budget, start_center_sets);

    GRBenv* env = opt_new_env();
    GRBmodel* model = nullptr;
    {
        std::vector<double> obj(nvars, 0.0), lb(nvars, 0.0), ub(nvars, 1.0);
        std::vector<char> vtype(nvars, GRB_CONTINUOUS);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) obj[X(i, j)] = dist(P, i, j);
        for (int j = 0; j < n; ++j) vtype[Y(j)] = GRB_BINARY;
        if (hist) for (int i = 0; i < n; ++i) vtype[X(i, hcenter[i])] = GRB_BINARY;
        grb_check(GRBnewmodel(env, &model, "opt_classic", (int)nvars, obj.data(), lb.data(), ub.data(),
                              vtype.data(), nullptr), env, "GRBnewmodel");
    }
    GRBenv* menv = GRBgetenv(model);
    grb_check(GRBsetdblparam(menv, "TimeLimit", time_limit), menv, "TimeLimit");
    grb_check(GRBsetdblparam(menv, "MIPGap", mip_gap), menv, "MIPGap");
    grb_check(GRBsetintparam(menv, "Threads", threads), menv, "Threads");

    {
        std::vector<int> ind(n);
        std::vector<double> ones(n, 1.0);
        int cind[2];
        double cval[2] = {1.0, -1.0};
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) ind[j] = X(i, j);
            grb_check(GRBaddconstr(model, n, ind.data(), ones.data(), GRB_EQUAL, 1.0, nullptr), menv, "assign");
            for (int j = 0; j < n; ++j) {
                cind[0] = X(i, j); cind[1] = Y(j);
                grb_check(GRBaddconstr(model, 2, cind, cval, GRB_LESS_EQUAL, 0.0, nullptr), menv, "open");
            }
        }
        for (int j = 0; j < n; ++j) ind[j] = Y(j);
        grb_check(GRBaddconstr(model, n, ind.data(), ones.data(), GRB_EQUAL, (double)k, nullptr), menv, "k");
        if (hist) {
            for (int i = 0; i < n; ++i) ind[i] = X(i, hcenter[i]);
            grb_check(GRBaddconstr(model, n, ind.data(), ones.data(), GRB_GREATER_EQUAL,
                                   (double)(n - budget), nullptr), menv, "budget");
        }
    }

    if (!start.centers.empty()) {
        std::vector<double> x0(nvars, 0.0);
        for (int c : start.centers) x0[Y(c)] = 1.0;
        for (int i = 0; i < n; ++i) x0[X(i, start.centers[start.assignment[i]])] = 1.0;
        grb_check(GRBsetdblattrarray(model, "Start", 0, (int)nvars, x0.data()), menv, "Start");
    }

    grb_check(GRBoptimize(model), menv, "GRBoptimize");
    int status = 0, solcount = 0;
    GRBgetintattr(model, "Status", &status);
    GRBgetintattr(model, "SolCount", &solcount);
    info.status = status;
    info.optimal = (status == GRB_OPTIMAL);

    std::vector<int> centers;
    if (solcount > 0) {
        GRBgetdblattr(model, "MIPGap", &info.mip_gap);
        GRBgetdblattr(model, "ObjBound", &info.bound);
        std::vector<double> y(n);
        grb_check(GRBgetdblattrarray(model, "X", Y(0), n, y.data()), menv, "X");
        centers = opt_top_k(y, k);
        info.solved = true;
    }
    GRBfreemodel(model);
    GRBfreeenv(env);
    return opt_finish(P, k, hist, budget, centers, start, t0, info);
}

// ---------------------------------------------------------------------------
// "benders": radius / Benders formulation with on-demand cuts.
// ---------------------------------------------------------------------------
struct OptBendersCtx {
    int n = 0, k = 0;
    bool has_hist = false;
    std::vector<double> dh;      // d(i, h(i))
    std::vector<int> nb;         // n x n: neighbors of i sorted by distance (row i)
    std::vector<double> nd;      // n x n: the matching distances
    int Y0 = 0, T0 = 0, S0 = 0, nvars = 0;
    double tol = 1e-6;           // relative violation tolerance
    std::atomic<long> cuts{0};
};

// Most violated cut of point i at x (indexed like the model's variables).
// theta_i + sum_{j: d_ij < D} (D - d_ij) y_j - [D < dh_i] (dh_i - D) s_i >= D
// Returns the violation (<= 0: no violated cut) and fills ind/val/rhs.
inline double opt_benders_separate(const OptBendersCtx& C, int i, const double* x,
                                   std::vector<int>& ind, std::vector<double>& val, double& rhs) {
    const int n = C.n;
    const int* nb = &C.nb[(size_t)i * n];
    const double* nd = &C.nd[(size_t)i * n];
    const double s = C.has_hist ? x[C.S0 + i] : 0.0;
    const double dh = C.has_hist ? C.dh[i] : 0.0;

    // g(D) = D (1 - cum) + wsum + [D < dh] (dh - D) s is concave in D with
    // breakpoints at the distances; maximize it by scanning the sorted list.
    double cum = 0.0, wsum = 0.0, best = -std::numeric_limits<double>::infinity();
    double bestD = 0.0;
    int bestp = 0, p = 0;
    while (p < n) {
        const double D = nd[p];
        double g = D * (1.0 - cum) + wsum + ((C.has_hist && D < dh) ? (dh - D) * s : 0.0);
        if (g > best) { best = g; bestD = D; bestp = p; }
        while (p < n && nd[p] == D) {
            double yv = x[C.Y0 + nb[p]];
            cum += yv;
            wsum += nd[p] * yv;
            ++p;
        }
        if (cum >= 1.0 - 1e-12) break;  // slope <= 0 from here on
    }
    const double theta = x[C.T0 + i];
    const double viol = best - theta;
    if (viol <= C.tol * std::max(1.0, std::fabs(best))) return 0.0;

    ind.clear(); val.clear();
    ind.push_back(C.T0 + i); val.push_back(1.0);
    for (int q = 0; q < bestp; ++q) {
        double coef = bestD - nd[q];
        if (coef > 0) { ind.push_back(C.Y0 + nb[q]); val.push_back(coef); }
    }
    if (C.has_hist && bestD < dh) { ind.push_back(C.S0 + i); val.push_back(-(dh - bestD)); }
    rhs = bestD;
    return viol;
}

// Callback: lazy constraints at new incumbents, user cuts at fractional nodes.
inline int __stdcall opt_benders_callback(GRBmodel* /*model*/, void* cbdata, int where, void* usrdata) {
    OptBendersCtx& C = *static_cast<OptBendersCtx*>(usrdata);
    if (where != GRB_CB_MIPSOL && where != GRB_CB_MIPNODE) return 0;
    std::vector<double> x(C.nvars);
    if (where == GRB_CB_MIPSOL) {
        if (GRBcbget(cbdata, where, GRB_CB_MIPSOL_SOL, x.data())) return 0;
    } else {
        int st = 0;
        if (GRBcbget(cbdata, where, GRB_CB_MIPNODE_STATUS, &st) || st != GRB_OPTIMAL) return 0;
        if (GRBcbget(cbdata, where, GRB_CB_MIPNODE_REL, x.data())) return 0;
    }
    std::vector<int> ind;
    std::vector<double> val;
    double rhs = 0.0;
    for (int i = 0; i < C.n; ++i) {
        if (opt_benders_separate(C, i, x.data(), ind, val, rhs) <= 0.0) continue;
        int err = (where == GRB_CB_MIPSOL)
            ? GRBcblazy(cbdata, (int)ind.size(), ind.data(), val.data(), GRB_GREATER_EQUAL, rhs)
            : GRBcbcut(cbdata, (int)ind.size(), ind.data(), val.data(), GRB_GREATER_EQUAL, rhs);
        if (err) return err;
        C.cuts++;
    }
    return 0;
}

inline MappedClustering opt_lcc_median_benders(
        const Points& P, int k, const Clustering* hist, int budget,
        const std::vector<std::vector<int>>& start_center_sets,
        double time_limit, double mip_gap, int threads, OptInfo& info) {
    auto t0 = std::chrono::steady_clock::now();
    const int n = P.n;
    if ((size_t)n * n > (size_t)std::numeric_limits<int>::max())
        throw std::runtime_error("opt: instance too large");

    OptBendersCtx C;
    C.n = n; C.k = k; C.has_hist = (hist != nullptr);
    C.Y0 = 0; C.T0 = n; C.S0 = 2 * n;
    C.nvars = C.has_hist ? 3 * n : 2 * n;
    std::vector<int> hcenter(n, -1);
    if (hist) {
        C.dh.resize(n);
        for (int i = 0; i < n; ++i) { hcenter[i] = hist->center_of(i); C.dh[i] = dist(P, i, hcenter[i]); }
    }
    // Sorted neighbor lists (n^2 ints + n^2 doubles).
    C.nb.resize((size_t)n * n);
    C.nd.resize((size_t)n * n);
    {
        std::vector<std::pair<double, int>> row(n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) row[j] = {dist(P, i, j), j};
            std::sort(row.begin(), row.end());
            for (int j = 0; j < n; ++j) { C.nb[(size_t)i * n + j] = row[j].second; C.nd[(size_t)i * n + j] = row[j].first; }
        }
    }
    OptStart start = opt_best_start(P, k, hist, budget, start_center_sets);

    GRBenv* env = opt_new_env();
    GRBmodel* model = nullptr;
    {
        std::vector<double> obj(C.nvars, 0.0), lb(C.nvars, 0.0), ub(C.nvars, 1.0);
        for (int i = 0; i < n; ++i) { obj[C.T0 + i] = 1.0; ub[C.T0 + i] = GRB_INFINITY; }
        std::vector<char> vtype(C.nvars, GRB_CONTINUOUS);  // LP relaxation first
        grb_check(GRBnewmodel(env, &model, "opt_benders", C.nvars, obj.data(), lb.data(), ub.data(),
                              vtype.data(), nullptr), env, "GRBnewmodel");
    }
    GRBenv* menv = GRBgetenv(model);
    grb_check(GRBsetintparam(menv, "Threads", threads), menv, "Threads");
    {
        std::vector<int> ind(n);
        std::vector<double> ones(n, 1.0);
        for (int j = 0; j < n; ++j) ind[j] = C.Y0 + j;
        grb_check(GRBaddconstr(model, n, ind.data(), ones.data(), GRB_EQUAL, (double)k, nullptr), menv, "k");
        if (hist) {
            for (int i = 0; i < n; ++i) ind[i] = C.S0 + i;
            grb_check(GRBaddconstr(model, n, ind.data(), ones.data(), GRB_GREATER_EQUAL,
                                   (double)(n - budget), nullptr), menv, "budget");
            int cind[2];
            double cval[2] = {1.0, -1.0};
            for (int i = 0; i < n; ++i) {
                cind[0] = C.S0 + i; cind[1] = C.Y0 + hcenter[i];
                grb_check(GRBaddconstr(model, 2, cind, cval, GRB_LESS_EQUAL, 0.0, nullptr), menv, "stay");
            }
        }
    }

    // Start solution as a full variable vector (y, theta, s).
    std::vector<double> x0;
    if (!start.centers.empty()) {
        x0.assign(C.nvars, 0.0);
        for (int c : start.centers) x0[C.Y0 + c] = 1.0;
        for (int i = 0; i < n; ++i) {
            int c = start.centers[start.assignment[i]];
            x0[C.T0 + i] = dist(P, i, c);
            if (hist && c == hcenter[i]) x0[C.S0 + i] = 1.0;
        }
    }

    std::vector<int> ind;
    std::vector<double> val;
    double rhs = 0.0;
    auto add_cuts_at = [&](const std::vector<double>& x, bool force) {
        int added = 0;
        std::vector<double> xx = x;
        for (int i = 0; i < n; ++i) {
            if (force) xx[C.T0 + i] = -1.0;  // every point gets its tight cut
            if (opt_benders_separate(C, i, xx.data(), ind, val, rhs) <= 0.0) continue;
            grb_check(GRBaddconstr(model, (int)ind.size(), ind.data(), val.data(), GRB_GREATER_EQUAL, rhs, nullptr),
                      menv, "cut");
            ++added;
        }
        C.cuts += added;
        return added;
    };
    if (!x0.empty()) add_cuts_at(x0, true);

    // Phase 1: cutting-plane loop on the LP relaxation.
    std::vector<double> x(C.nvars);
    const int max_rounds = 1000;
    double lp_obj = std::numeric_limits<double>::quiet_NaN();
    for (int round = 0; round < max_rounds; ++round) {
        double remaining = time_limit - opt_seconds_since(t0);
        if (remaining <= 0) break;
        grb_check(GRBsetdblparam(menv, "TimeLimit", remaining), menv, "TimeLimit");
        grb_check(GRBoptimize(model), menv, "GRBoptimize (LP)");
        int st = 0;
        GRBgetintattr(model, "Status", &st);
        if (st != GRB_OPTIMAL) break;
        GRBgetdblattr(model, "ObjVal", &lp_obj);
        grb_check(GRBgetdblattrarray(model, "X", 0, C.nvars, x.data()), menv, "X (LP)");
        info.lp_rounds = round + 1;
        if (add_cuts_at(x, false) == 0) { info.lp_bound = lp_obj; break; }
    }
    info.lp_time = opt_seconds_since(t0);

    // Phase 2: branch-and-cut with y binary (s stays continuous, see top of file).
    {
        std::vector<char> bin(n, GRB_BINARY);
        grb_check(GRBsetcharattrarray(model, "VType", C.Y0, n, bin.data()), menv, "VType y");
    }
    if (!x0.empty()) grb_check(GRBsetdblattrarray(model, "Start", 0, C.nvars, x0.data()), menv, "Start");
    grb_check(GRBsetintparam(menv, "LazyConstraints", 1), menv, "LazyConstraints");
    grb_check(GRBsetintparam(menv, "PreCrush", 1), menv, "PreCrush");
    grb_check(GRBsetdblparam(menv, "MIPGap", mip_gap), menv, "MIPGap");
    grb_check(GRBsetcallbackfunc(model, opt_benders_callback, &C), menv, "callback");
    double remaining = std::max(1.0, time_limit - opt_seconds_since(t0));
    grb_check(GRBsetdblparam(menv, "TimeLimit", remaining), menv, "TimeLimit");
    grb_check(GRBoptimize(model), menv, "GRBoptimize (MIP)");

    int status = 0, solcount = 0;
    GRBgetintattr(model, "Status", &status);
    GRBgetintattr(model, "SolCount", &solcount);
    info.status = status;
    info.optimal = (status == GRB_OPTIMAL);
    std::vector<int> centers;
    if (solcount > 0) {
        GRBgetdblattr(model, "MIPGap", &info.mip_gap);
        GRBgetdblattr(model, "ObjBound", &info.bound);
        std::vector<double> y(n);
        grb_check(GRBgetdblattrarray(model, "X", C.Y0, n, y.data()), menv, "X");
        centers = opt_top_k(y, k);
        info.solved = true;
    }
    info.cuts = C.cuts.load();
    GRBfreemodel(model);
    GRBfreeenv(env);
    return opt_finish(P, k, hist, budget, centers, start, t0, info);
}

// hist == nullptr: plain k-median. start_center_sets: center sets used to build
// MIP start solutions (the best feasible one is passed to Gurobi).
// formulation: "benders" (default) or "classic".
inline MappedClustering opt_lcc_median_center_identity(
        const Points& P, int k, const Clustering* hist, int budget,
        const std::vector<std::vector<int>>& start_center_sets,
        double time_limit, double mip_gap, int threads, OptInfo& info,
        const std::string& formulation = "benders") {
    if (formulation == "classic")
        return opt_lcc_median_classic(P, k, hist, budget, start_center_sets, time_limit, mip_gap, threads, info);
    if (formulation != "benders")
        throw std::invalid_argument("unknown --opt_formulation '" + formulation + "' (benders|classic)");
    return opt_lcc_median_benders(P, k, hist, budget, start_center_sets, time_limit, mip_gap, threads, info);
}
#endif  // USE_GUROBI