// Exact k-median baseline (--heuristic optimal), solved with Gurobi (C API).
//
// Centers are input points, like in the heuristics. Uses the Benders formulation of
// Duran-Mateluna, Ales & Elloumi, "An efficient Benders decomposition for the p-median
// problem", EJOR 308(1), 2023. Variables: y_j = 1 if point j is a center (binary),
// theta_i = cost of point i.
//
//   min sum theta_i
//   s.t. sum_j y_j = k
//        theta_i >= D - sum_{j: d_ij < D} (D - d_ij) y_j       for all i and distances D
//
// Cuts are generated on demand: first in a cutting-plane loop on the LP relaxation,
// then by a callback (lazy constraints at incumbents, user cuts at fractional nodes).
//
// Build with -DUSE_GUROBI and Gurobi's include/lib paths (see Makefile).
#pragma once
#ifdef USE_GUROBI
#include "core.hpp"
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

extern "C" {
#include "gurobi_c.h"
}

struct OptimalInfo {
    bool optimal = false;  // proven optimal (within mip_gap); false e.g. at the time limit
    double mip_gap = std::numeric_limits<double>::quiet_NaN();
};

inline void gurobi_check(int err, GRBenv* env, const char* what) {
    if (err) {
        std::string msg = std::string("Gurobi error in ") + what + ": " + (env ? GRBgeterrormsg(env) : "")
                          + " (code " + std::to_string(err) + ")";
        throw std::runtime_error(msg);
    }
}

struct KMedianCuts {
    int n = 0;
    std::vector<int> nb;     // n x n: neighbors of i sorted by distance (row i)
    std::vector<double> nd;  // n x n: the matching distances
    double tol = 1e-6;       // relative violation tolerance

    int y_var(int j) const { return j; }
    int theta_var(int i) const { return n + i; }

    // Most violated cut of point i at x. Returns the violation (<= 0: no violated cut)
    // and fills ind/val/rhs. g(D) = D (1 - cum) + wsum is concave in D with breakpoints
    // at the distances, so it is maximized by scanning the sorted list.
    double separate(int i, const double* x, std::vector<int>& ind, std::vector<double>& val, double& rhs) const {
        const int* nbi = &nb[(size_t)i * n];
        const double* ndi = &nd[(size_t)i * n];
        double cum = 0.0, wsum = 0.0, best = -std::numeric_limits<double>::infinity();
        double best_d = 0.0;
        int best_p = 0, p = 0;
        while (p < n) {
            const double D = ndi[p];
            double g = D * (1.0 - cum) + wsum;
            if (g > best) { best = g; best_d = D; best_p = p; }
            while (p < n && ndi[p] == D) {
                double yv = x[y_var(nbi[p])];
                cum += yv;
                wsum += ndi[p] * yv;
                ++p;
            }
            if (cum >= 1.0 - 1e-12) break;  // slope <= 0 from here on
        }
        const double viol = best - x[theta_var(i)];
        if (viol <= tol * std::max(1.0, std::fabs(best))) return 0.0;

        ind.clear(); val.clear();
        ind.push_back(theta_var(i)); val.push_back(1.0);
        for (int q = 0; q < best_p; ++q) {
            double coef = best_d - ndi[q];
            if (coef > 0) { ind.push_back(y_var(nbi[q])); val.push_back(coef); }
        }
        rhs = best_d;
        return viol;
    }
};

// Callback: lazy constraints at new incumbents, user cuts at fractional nodes.
inline int __stdcall kmedian_cut_callback(GRBmodel* /*model*/, void* cbdata, int where, void* usrdata) {
    const KMedianCuts& C = *static_cast<const KMedianCuts*>(usrdata);
    if (where != GRB_CB_MIPSOL && where != GRB_CB_MIPNODE) return 0;
    std::vector<double> x(2 * C.n);
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
        if (C.separate(i, x.data(), ind, val, rhs) <= 0.0) continue;
        int err = (where == GRB_CB_MIPSOL)
            ? GRBcblazy(cbdata, (int)ind.size(), ind.data(), val.data(), GRB_GREATER_EQUAL, rhs)
            : GRBcbcut(cbdata, (int)ind.size(), ind.data(), val.data(), GRB_GREATER_EQUAL, rhs);
        if (err) return err;
    }
    return 0;
}

// Exact k-median of P. `start` is a heuristic solution used as MIP start and returned
// if Gurobi finds nothing better within the time limit; check info.optimal.
inline Clustering optimal_kmedian(const Points& P, int k, const Clustering& start,
                                  double time_limit, double mip_gap, int threads, OptimalInfo& info) {
    auto t0 = std::chrono::steady_clock::now();
    auto seconds_left = [&]() {
        return time_limit - std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };
    const int n = P.n;
    const int nvars = 2 * n;

    KMedianCuts C;
    C.n = n;
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

    GRBenv* env = nullptr;
    gurobi_check(GRBemptyenv(&env), nullptr, "GRBemptyenv");
    gurobi_check(GRBsetintparam(env, "OutputFlag", 0), env, "OutputFlag");
    gurobi_check(GRBstartenv(env), env, "GRBstartenv (license?)");

    GRBmodel* model = nullptr;
    {
        std::vector<double> obj(nvars, 0.0), lb(nvars, 0.0), ub(nvars, 1.0);
        for (int i = 0; i < n; ++i) { obj[C.theta_var(i)] = 1.0; ub[C.theta_var(i)] = GRB_INFINITY; }
        std::vector<char> vtype(nvars, GRB_CONTINUOUS);  // LP relaxation first
        gurobi_check(GRBnewmodel(env, &model, "optimal_kmedian", nvars, obj.data(), lb.data(), ub.data(),
                                 vtype.data(), nullptr), env, "GRBnewmodel");
    }
    GRBenv* menv = GRBgetenv(model);
    gurobi_check(GRBsetintparam(menv, "Threads", threads), menv, "Threads");
    {
        std::vector<int> ind(n);
        std::vector<double> ones(n, 1.0);
        std::iota(ind.begin(), ind.end(), 0);
        gurobi_check(GRBaddconstr(model, n, ind.data(), ones.data(), GRB_EQUAL, (double)k, nullptr), menv, "k");
    }

    std::vector<int> ind;
    std::vector<double> val;
    double rhs = 0.0;
    auto add_cuts_at = [&](const std::vector<double>& x) {
        int added = 0;
        for (int i = 0; i < n; ++i) {
            if (C.separate(i, x.data(), ind, val, rhs) <= 0.0) continue;
            gurobi_check(GRBaddconstr(model, (int)ind.size(), ind.data(), val.data(), GRB_GREATER_EQUAL, rhs, nullptr),
                         menv, "cut");
            ++added;
        }
        return added;
    };

    // Start solution as a full variable vector (y, theta), with its tight cuts.
    std::vector<double> x0(nvars, 0.0);
    for (int c : start.centers) x0[C.y_var(c)] = 1.0;
    add_cuts_at(x0);  // theta is still 0, so every point gets its tight cut
    for (int i = 0; i < n; ++i) x0[C.theta_var(i)] = dist(P, i, start.center_of(i));

    // Phase 1: cutting-plane loop on the LP relaxation.
    std::vector<double> x(nvars);
    const int max_rounds = 1000;
    for (int round = 0; round < max_rounds; ++round) {
        double remaining = seconds_left();
        if (remaining <= 0) break;
        gurobi_check(GRBsetdblparam(menv, "TimeLimit", remaining), menv, "TimeLimit");
        gurobi_check(GRBoptimize(model), menv, "GRBoptimize (LP)");
        int st = 0;
        GRBgetintattr(model, "Status", &st);
        if (st != GRB_OPTIMAL) break;
        gurobi_check(GRBgetdblattrarray(model, "X", 0, nvars, x.data()), menv, "X (LP)");
        if (add_cuts_at(x) == 0) break;
    }

    // Phase 2: branch-and-cut with y binary.
    {
        std::vector<char> bin(n, GRB_BINARY);
        gurobi_check(GRBsetcharattrarray(model, "VType", 0, n, bin.data()), menv, "VType y");
    }
    gurobi_check(GRBsetdblattrarray(model, "Start", 0, nvars, x0.data()), menv, "Start");
    gurobi_check(GRBsetintparam(menv, "LazyConstraints", 1), menv, "LazyConstraints");
    gurobi_check(GRBsetintparam(menv, "PreCrush", 1), menv, "PreCrush");
    gurobi_check(GRBsetdblparam(menv, "MIPGap", mip_gap), menv, "MIPGap");
    gurobi_check(GRBsetcallbackfunc(model, kmedian_cut_callback, &C), menv, "callback");
    gurobi_check(GRBsetdblparam(menv, "TimeLimit", std::max(1.0, seconds_left())), menv, "TimeLimit");
    gurobi_check(GRBoptimize(model), menv, "GRBoptimize (MIP)");

    int status = 0, solcount = 0;
    GRBgetintattr(model, "Status", &status);
    GRBgetintattr(model, "SolCount", &solcount);
    info.optimal = (status == GRB_OPTIMAL);

    Clustering result = start;
    if (solcount > 0) {
        GRBgetdblattr(model, "MIPGap", &info.mip_gap);
        std::vector<double> y(n);
        gurobi_check(GRBgetdblattrarray(model, "X", 0, n, y.data()), menv, "X");
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return y[a] > y[b]; });
        std::vector<int> centers(order.begin(), order.begin() + k);
        std::sort(centers.begin(), centers.end());
        Clustering solved(P, centers, assign_points(P, centers));
        // The cut tolerance lets theta undercut the true cost slightly, so compare true costs.
        if (solved.cost() < start.cost()) result = solved;
    } else {
        std::fprintf(stderr, "  [optimal] no MIP solution (status %d), using the start solution\n", status);
    }
    GRBfreemodel(model);
    GRBfreeenv(env);
    return result;
}

#endif  // USE_GUROBI
