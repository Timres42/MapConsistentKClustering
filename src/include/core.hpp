// Core clustering data structures and reductions.
//
// Points/Clustering mirror the shapes defined in src/include/clustering_classes.py
// (the Python side only uses those for setup and I/O; the cost/assignment
// logic itself is implemented here, not duplicated in Python).
//
// All sums accumulate sequentially, and all argmin/min lookups break ties
// toward the lowest index.
#pragma once
#include <vector>
#include <cmath>
#include <cstddef>
#include <limits>

struct Points {
    int n = 0;
    int d = 0;
    std::vector<double> data;  // row-major, n*d
    const double* row(int i) const { return &data[(size_t)i * d]; }
};

// Euclidean distance: sqrt of the sequential sum of squared differences.
inline double euclidean(const double* a, const double* b, int d) {
    double s = 0.0;
    for (int j = 0; j < d; ++j) {
        double diff = a[j] - b[j];
        s += diff * diff;
    }
    return std::sqrt(s);
}

inline double dist(const Points& P, int i, int j) {
    return euclidean(P.row(i), P.row(j), P.d);
}

// argmin over a vector, returning the first minimum.
inline int argmin_first(const std::vector<double>& v) {
    int best = 0;
    double bv = v[0];
    for (int i = 1; i < (int)v.size(); ++i)
        if (v[i] < bv) { bv = v[i]; best = i; }
    return best;
}

// Assign every point to its nearest center (index into `centers`), matching
// helpers._assign_points (ties broken toward the first minimum).
inline std::vector<int> assign_points(const Points& P, const std::vector<int>& centers) {
    std::vector<int> assignment(P.n, 0);
    for (int i = 0; i < P.n; ++i) {
        double best = std::numeric_limits<double>::infinity();
        int arg = 0;
        for (int c = 0; c < (int)centers.size(); ++c) {
            double dd = dist(P, i, centers[c]);
            if (dd < best) { best = dd; arg = c; }
        }
        assignment[i] = arg;
    }
    return assignment;
}

struct Clustering {
    const Points* P = nullptr;
    std::vector<int> centers;     // global point indices, length k
    std::vector<int> assignment;  // index into `centers`, length n
    int n = 0;
    int k = 0;

    Clustering() = default;
    Clustering(const Points& p, std::vector<int> c, std::vector<int> a)
        : P(&p), centers(std::move(c)), assignment(std::move(a)) {
        n = (int)assignment.size();
        k = (int)centers.size();
    }

    int center_of(int p) const { return centers[assignment[p]]; }

    // k-median cost: sequential accumulation, matching Clustering.cost().
    double cost() const {
        if (n == 0) return 0.0;
        double total = 0.0;
        for (int p = 0; p < n; ++p) total += dist(*P, p, center_of(p));
        return total;
    }

    std::vector<int> cluster_sizes() const {
        std::vector<int> sizes(k, 0);
        for (int p = 0; p < n; ++p) sizes[assignment[p]]++;
        return sizes;
    }

    std::vector<int> points_in_cluster(int i) const {
        std::vector<int> res;
        for (int p = 0; p < n; ++p) if (assignment[p] == i) res.push_back(p);
        return res;
    }
};

// helpers._compute_cost: sum over points of min distance to any center (seq).
inline double compute_cost(const Points& P, const std::vector<int>& centers) {
    double total = 0.0;
    for (int i = 0; i < P.n; ++i) {
        double m = std::numeric_limits<double>::infinity();
        for (int c : centers) {
            double dd = dist(P, i, c);
            if (dd < m) m = dd;
        }
        total += m;
    }
    return total;
}

// helpers.cluster_distance: count points whose mapped historical label differs
// from clustering_b's label. mapping is applied to A's labels before comparing.
inline int cluster_distance(const Clustering& a, const Clustering& b,
                            const std::vector<int>& mapping) {
    int diff = 0;
    for (int p = 0; p < a.n; ++p)
        if (mapping[a.assignment[p]] != b.assignment[p]) diff++;
    return diff;
}