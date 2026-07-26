// "us" algorithm: poly_lcc_median_fct, the label-consistent clustering
// routine that greedily grows each historical cluster toward its cheapest
// heuristic center and settles the budget split with a knapsack (see
// knapsack.hpp). This C++ version is the only implementation - there is no
// separate Python reference to keep in sync with.
#pragma once
#include "core.hpp"
#include "knapsack.hpp"
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>

struct MappedClustering {
    Clustering clustering;
    std::vector<int> mapping;  // f_hat
};

inline MappedClustering poly_lcc_median_fct(const Points& P, int k,
                                            const Clustering& hist_clust, int budget,
                                            const Clustering& heuristic_clust) {
    struct Entry { double cost; std::vector<int> points; int c_idx; };

    std::vector<std::vector<Entry>> culm_opt(k);
    for (int i = 0; i < k; ++i) {
        std::vector<int> Hi = hist_clust.points_in_cluster(i);
        int hi = (int)Hi.size();
        std::vector<Entry> curr(hi + 1);
        curr[0] = {0.0, {}, 0};
        for (int j = 1; j <= hi; ++j) curr[j] = {std::numeric_limits<double>::infinity(), {}, 0};

        for (int c_idx = 0; c_idx < (int)heuristic_clust.centers.size(); ++c_idx) {
            int center_global = heuristic_clust.centers[c_idx];
            std::vector<double> delta(hi);
            for (int t = 0; t < hi; ++t) {
                int p = Hi[t];
                int c_heur = heuristic_clust.center_of(p);
                delta[t] = dist(P, p, center_global) - dist(P, p, c_heur);
            }
            // sorted(Hi, key=delta): stable ascending. Hi is already ascending,
            // so a stable sort of positions preserves index order on ties.
            std::vector<int> order(hi);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(),
                             [&](int a, int b) { return delta[a] < delta[b]; });

            double cum_cost = 0.0;
            std::vector<int> cum_points;
            for (int j = 0; j < hi; ++j) {
                int pos = order[j];
                cum_cost += delta[pos];
                cum_points.push_back(Hi[pos]);
                if (cum_cost < curr[j + 1].cost)
                    curr[j + 1] = {cum_cost, cum_points, c_idx};
            }
        }
        culm_opt[i] = std::move(curr);
    }

    // Build knapsack groups: one per historical cluster.
    std::vector<std::vector<Item>> mck_list(k);
    for (int i = 0; i < k; ++i) {
        int hi = (int)hist_clust.points_in_cluster(i).size();
        double max_cost = -std::numeric_limits<double>::infinity();
        for (int m = 0; m <= hi; ++m) max_cost = std::max(max_cost, culm_opt[i][m].cost);
        for (int j = 0; j <= hi; ++j) {
            double cost = culm_opt[i][j].cost;
            int chosen_c_idx = culm_opt[i][j].c_idx;
            int weight = hi - j;
            double value = max_cost - cost;
            mck_list[i].push_back({weight, value, chosen_c_idx});
        }
    }
    MCKResult mck_sol = multiple_choice_knapsack(mck_list, budget);

    std::vector<int> f_hat;
    std::vector<int> new_centers = heuristic_clust.centers;
    std::vector<int> new_assignment = assign_points(P, new_centers);

    for (int i = 0; i < k; ++i) {
        int chosen_idx = mck_sol.choice_idx[i];
        const Item& chosen_item = mck_list[i][chosen_idx];
        const std::vector<int>& chosen_points = culm_opt[i][chosen_idx].points;
        int chosen_c_idx = chosen_item.label;
        f_hat.push_back(chosen_c_idx);
        for (int p : chosen_points) new_assignment[p] = chosen_c_idx;
    }

    Clustering new_clustering(P, new_centers, new_assignment);
    if (new_clustering.cost() > hist_clust.cost()) {
        new_clustering = hist_clust;
        f_hat.resize(k);
        for (int i = 0; i < k; ++i) f_hat[i] = i;
    }
    return {new_clustering, f_hat};
}
