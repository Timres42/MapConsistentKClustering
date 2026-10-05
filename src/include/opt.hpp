// "opt" algorithm: the exact optimum of the center-identity variant of
// label-consistent k-median, solved as a MIP with Gurobi (C API).
//
// Variant: given a historical clustering H (every point p has a historical
// center point h(p)) and a budget b, find k center points C and an assignment
// sigma minimizing sum_p d(p, sigma(p)) such that at most b points have
// sigma(p) != h(p). "Center" means the explicit point: if h(p) is not in C,
// p is counted, even if it stays in the "same" cluster slot.
// Without a historical clustering (hist == nullptr) this is plain k-median.
//
// MIP (n = number of points, every point is a candidate center):
//   min  sum_ij d(i,j) x_ij
//   s.t. sum_j x_ij = 1                 for all i
//        x_ij <= y_j                    for all i, j
//        sum_j y_j = k
//        sum_i x_{i,h(i)} >= n - b      (only with a historical clustering)
//        y_j binary, x_{i,h(i)} binary, all other x_ij in [0, 1]
// The other x_ij may stay continuous: for fixed y and fixed stay-variables
// an optimal assignment sends each remaining point to its nearest open center.
//
// The model has n^2 + n variables and about n^2 constraints, so the exact
// solve is only practical for instances up to a few thousand points.
//
// Build with -DUSE_GUROBI and Gurobi's include/lib paths (see Makefile notes).
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

struct OptInfo {
    bool solved = false;      // true if a MIP solution was obtained
    bool optimal = false;     // proven optimal (within MIPGap)
    int status = -1;          // Gurobi status code
    double mip_gap = std::numeric_limits<double>::quiet_NaN();
    double bound = std::numeric_limits<double>::quiet_NaN();  // best lower bound
    double runtime = 0.0;
    int budget_used = 0;      // points whose center point differs from h(p)
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

// hist == nullptr: plain k-median. start_center_sets: center sets used to build
// MIP start solutions (the best feasible one is passed to Gurobi).
inline MappedClustering opt_lcc_median_center_identity(
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
        throw std::runtime_error("opt: instance too large for the MIP (n^2 + n > INT_MAX)");

    std::vector<int> hcenter(n, -1);
    if (hist) for (int p = 0; p < n; ++p) hcenter[p] = hist->center_of(p);

    // Best feasible start solution among the given center sets.
    std::vector<int> start_centers, start_assign;
    double start_cost = std::numeric_limits<double>::infinity();
    for (const auto& cs : start_center_sets) {
        if ((int)cs.size() != k) continue;
        std::vector<int> ordered = order_centers_by_slot(cs, hist, k);
        std::vector<int> a;
        if (!assign_with_budget(P, hist, ordered, budget, a)) continue;
        double c = Clustering(P, ordered, a).cost();
        if (c < start_cost) { start_cost = c; start_centers = ordered; start_assign = a; }
    }

    GRBenv* env = nullptr;
    GRBmodel* model = nullptr;
    grb_check(GRBemptyenv(&env), nullptr, "GRBemptyenv");
    grb_check(GRBsetintparam(env, "OutputFlag", 0), env, "OutputFlag");
    grb_check(GRBstartenv(env), env, "GRBstartenv (license?)");

    {
        std::vector<double> obj(nvars, 0.0), lb(nvars, 0.0), ub(nvars, 1.0);
        std::vector<char> vtype(nvars, GRB_CONTINUOUS);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) obj[X(i, j)] = dist(P, i, j);
        for (int j = 0; j < n; ++j) vtype[Y(j)] = GRB_BINARY;
        if (hist) for (int i = 0; i < n; ++i) vtype[X(i, hcenter[i])] = GRB_BINARY;
        grb_check(GRBnewmodel(env, &model, "opt_lcc", (int)nvars, obj.data(), lb.data(), ub.data(),
                              vtype.data(), nullptr), env, "GRBnewmodel");
    }
    GRBenv* menv = GRBgetenv(model);
    grb_check(GRBsetdblparam(menv, "TimeLimit", time_limit), menv, "TimeLimit");
    grb_check(GRBsetdblparam(menv, "MIPGap", mip_gap), menv, "MIPGap");
    grb_check(GRBsetintparam(menv, "Threads", threads), menv, "Threads");  // 0 = all cores

    {
        // sum_j x_ij = 1 and x_ij - y_j <= 0, added row by row to keep memory low.
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

    if (!start_centers.empty()) {
        std::vector<double> start(nvars, 0.0);
        for (int c : start_centers) start[Y(c)] = 1.0;
        for (int i = 0; i < n; ++i) start[X(i, start_centers[start_assign[i]])] = 1.0;
        grb_check(GRBsetdblattrarray(model, "Start", 0, (int)nvars, start.data()), menv, "Start");
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
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return y[a] > y[b]; });
        centers.assign(order.begin(), order.begin() + k);
        std::sort(centers.begin(), centers.end());
        info.solved = true;
    }
    GRBfreemodel(model);
    GRBfreeenv(env);

    // Rebuild the assignment exactly for the chosen centers (same optimum, no
    // rounding noise from the continuous x). Fall back to the start solution.
    Clustering result;
    std::vector<int> assignment;
    if (info.solved) {
        std::vector<int> ordered = order_centers_by_slot(centers, hist, k);
        if (assign_with_budget(P, hist, ordered, budget, assignment))
            result = Clustering(P, ordered, assignment);
        else
            info.solved = false;
    }
    if (!info.solved) {
        if (!start_centers.empty()) result = Clustering(P, start_centers, start_assign);
        else if (hist) result = *hist;
        else throw std::runtime_error("opt: no solution found and no start solution available");
        std::fprintf(stderr, "  [opt] no MIP solution (status %d), using best start solution\n", status);
    }

    info.budget_used = hist ? center_point_changes(*hist, result) : 0;
    info.runtime = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::vector<int> identity(k);
    std::iota(identity.begin(), identity.end(), 0);
    return {result, identity};
}
#endif  // USE_GUROBI