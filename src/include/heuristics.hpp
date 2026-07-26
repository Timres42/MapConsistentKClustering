// Heuristic initializers used to seed the baseline / historical clustering.
// kmedian_plus_plus is standard k-median++ seeding; kmedianppwpost runs a
// post-processing pass that recenters each cluster on its medoid. These are
// the only two heuristics implemented (selected via --heuristic in
// propagation.cpp / propagation_setup.py's HEURISTIC_CHOICES).
#pragma once
#include "core.hpp"
#include <vector>
#include <set>
#include <limits>
#include <cstdio>
#include <functional>
#include <random>
#include <stdexcept>

int choice(int n, std::mt19937& generator) {
    if (n <= 0) {
        throw std::invalid_argument("n must be positive");
    }

    std::uniform_int_distribution<int> distribution(0, n - 1);
    return distribution(generator);
}

int64_t choice_p(
    const std::vector<double>& probabilities,
    std::mt19937& generator
) {
    if (probabilities.empty()) {
        throw std::invalid_argument("probabilities must not be empty");
    }

    std::discrete_distribution<std::size_t> distribution(
        probabilities.begin(),
        probabilities.end()
    );

    return static_cast<int64_t>(distribution(generator));
}

inline Clustering kmedian_plus_plus(const Points& P, int k, std::mt19937& rng) {
    int n = P.n;
    if (n == 0) return Clustering(P, {}, {});
    std::vector<int> centers = {choice(n,rng)};
    for (int step = 0; step < k - 1; ++step) {
        std::vector<double> min_distances(n);
        for (int i = 0; i < n; ++i) {
            double m = std::numeric_limits<double>::infinity();
            for (int c : centers) { double dd = dist(P, i, c); if (dd < m) m = dd; }
            min_distances[i] = m;
        }
        //Compute sum
        double total = 0.0;
        for (size_t i = 0; i < min_distances.size(); ++i) total += min_distances[i];

        int next_center;
        if (total == 0.0) {
            std::set<int> cset(centers.begin(), centers.end());
            std::vector<int> remaining;
            for (int i = 0; i < n; ++i) if (!cset.count(i)) remaining.push_back(i);
            if (!remaining.empty()) next_center = remaining[choice((int)remaining.size(),rng)];
            else next_center = choice(n,rng);
        } else {
            std::vector<double> probs(n);
            for (int i = 0; i < n; ++i) probs[i] = min_distances[i] / total;
            next_center = choice_p(probs,rng);
        }
        centers.push_back(next_center);
    }
    std::vector<int> assignment = assign_points(P, centers);
    return Clustering(P, centers, assignment);
}

inline Clustering kmedianppwpost(const Points& P, int k, std::mt19937& rng) {
    Clustering initial = kmedian_plus_plus(P, k, rng);
    const std::vector<int>& centers = initial.centers;
    const std::vector<int>& assignment = initial.assignment;

    std::vector<int> new_centers;
    new_centers.reserve(centers.size());
    for (int center_idx : centers) {
        // Find the first slot in `centers` equal to this center (a center id
        // can repeat if kmedian_plus_plus picked the same point twice).
        int first_pos = 0;
        for (int j = 0; j < (int)centers.size(); ++j) if (centers[j] == center_idx) { first_pos = j; break; }
        std::vector<int> cluster_pts;
        for (int p = 0; p < P.n; ++p) if (assignment[p] == first_pos) cluster_pts.push_back(p);
        if (cluster_pts.empty()) { new_centers.push_back(center_idx); continue; }
        std::vector<double> total_distances(cluster_pts.size());
        for (size_t a = 0; a < cluster_pts.size(); ++a) {
            double s = 0.0;  // Python builtin sum(): sequential.
            for (int other : cluster_pts) s += dist(P, cluster_pts[a], other);
            total_distances[a] = s;
        }
        int best = argmin_first(total_distances);
        new_centers.push_back(cluster_pts[best]);
    }
    std::vector<int> new_assignment = assign_points(P, new_centers);
    return Clustering(P, new_centers, new_assignment);
}

// Heuristic dispatch by name.
enum class HeuristicKind {KMpp, KMppPost};

inline Clustering run_heuristic(HeuristicKind kind, const Points& P, int k, std::mt19937& rng) {
    switch (kind) {
        case HeuristicKind::KMpp:        return kmedian_plus_plus(P, k, rng);
        case HeuristicKind::KMppPost:    return kmedianppwpost(P, k, rng);
    }
    return kmedianppwpost(P, k, rng);
}
