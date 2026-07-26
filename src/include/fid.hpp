// "fid" algorithm: poly_lcc_median_f_id, the label-consistent clustering
// routine that keeps center identities fixed (f = id): it decides which
// historical clusters to keep vs. replace, picks replacement centers greedily,
// then reassigns a regret-ranked subset of points within budget. This C++
// version is the only implementation - there is no separate Python reference
// to keep in sync with.
#pragma once
#include "core.hpp"
#include "form.hpp"  // for MappedClustering
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>

inline MappedClustering poly_lcc_median_f_id(const Points& P, int k,
                                             const Clustering& hist_clust, int budget,
                                             const Clustering& heuristic_sol) {
    printf("Running FID on n=%d many points and budget b=%d",P.n,budget);

    int n = P.n;
    const double INF = std::numeric_limits<double>::infinity();

    double best_total_cost = INF;
    Clustering best_clustering;
    bool have_best = false;

    std::vector<int> cluster_sizes = hist_clust.cluster_sizes();
    // sorted(range(k), key=size): stable ascending.
    std::vector<int> sorted_slots(k);
    std::iota(sorted_slots.begin(), sorted_slots.end(), 0);
    std::stable_sort(sorted_slots.begin(), sorted_slots.end(),
                     [&](int a, int b) { return cluster_sizes[a] < cluster_sizes[b]; });

    for (int kp = 0; kp <= k; ++kp) {
        std::vector<int> slots_to_remove(sorted_slots.begin(), sorted_slots.begin() + kp);
        std::vector<int> slots_to_keep(sorted_slots.begin() + kp, sorted_slots.end());

        int required = 0;
        for (int i : slots_to_remove) required += cluster_sizes[i];
        if (required > budget) continue;

        std::vector<int> candidate_centers(k, -1);  // -1 == Python None
        for (int i : slots_to_keep) candidate_centers[i] = hist_clust.centers[i];

        std::vector<int> chosen_new_centers;
        std::vector<int> available(heuristic_sol.centers.begin(), heuristic_sol.centers.end());

        for (int r = 0; r < kp; ++r) {
            std::vector<int> base_centers;
            for (int i = 0; i < k; ++i) if (candidate_centers[i] != -1) base_centers.push_back(candidate_centers[i]);
            for (int c : chosen_new_centers) base_centers.push_back(c);

            int best_cand = -1;
            double best_cand_cost = INF;
            for (int cand : available) {
                std::vector<int> current = base_centers;
                current.push_back(cand);
                double total_cost = 0.0;
                for (int p = 0; p < n; ++p) {
                    double m = INF;
                    for (int c : current) { double dd = dist(P, p, c); if (dd < m) m = dd; }
                    total_cost += m;
                }
                if (total_cost < best_cand_cost) { best_cand_cost = total_cost; best_cand = cand; }
            }
            if (best_cand != -1) {
                chosen_new_centers.push_back(best_cand);
                // list.remove: erase first occurrence.
                for (auto it = available.begin(); it != available.end(); ++it)
                    if (*it == best_cand) { available.erase(it); break; }
            }
        }

        for (size_t z = 0; z < slots_to_remove.size() && z < chosen_new_centers.size(); ++z)
            candidate_centers[slots_to_remove[z]] = chosen_new_centers[z];

        std::vector<int> current_assignment = hist_clust.assignment;

        struct Regret { double savings; int p; int new_slot; };
        std::vector<Regret> regret_savings;
        for (int p = 0; p < n; ++p) {
            int hist_slot = hist_clust.assignment[p];
            double cost_if_stay = dist(P, p, candidate_centers[hist_slot]);
            int best_center_idx = 0;
            double bcv = dist(P, p, candidate_centers[0]);
            for (int i = 1; i < k; ++i) {
                double dd = dist(P, p, candidate_centers[i]);
                if (dd < bcv) { bcv = dd; best_center_idx = i; }
            }
            double cost_if_move = dist(P, p, candidate_centers[best_center_idx]);
            double savings = cost_if_stay - cost_if_move;
            if (savings > 0 && best_center_idx != hist_slot)
                regret_savings.push_back({savings, p, best_center_idx});
        }
        // sort by savings descending, stable (preserves ascending-p insertion order on ties).
        std::stable_sort(regret_savings.begin(), regret_savings.end(),
                         [](const Regret& a, const Regret& b) { return a.savings > b.savings; });
        int allowed_moves = std::min(budget, (int)regret_savings.size());
        for (int m = 0; m < allowed_moves; ++m)
            current_assignment[regret_savings[m].p] = regret_savings[m].new_slot;

        Clustering eval_clustering(P, candidate_centers, current_assignment);
        double eval_cost = eval_clustering.cost();
        if (eval_cost < best_total_cost) {
            best_total_cost = eval_cost;
            best_clustering = eval_clustering;
            have_best = true;
        }
    }

    std::vector<int> identity(k);
    for (int i = 0; i < k; ++i) identity[i] = i;
    if (!have_best || best_clustering.cost() > hist_clust.cost())
        best_clustering = hist_clust;
    return {best_clustering, identity};
}
