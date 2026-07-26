// Multiple-choice knapsack DP: pick exactly one item per group to maximize
// total value subject to a shared weight capacity. Used by form.hpp to decide,
// per historical cluster, how many points to reassign under the budget.
#pragma once
#include <vector>
#include <limits>

struct Item {
    int weight;
    double value;   // Python passes floats through the NamedTuple's int annotations.
    int label;      // stored as the integer that Python stringified into `label`.
};

// Returns, per group, the index of the chosen item (choices[i]).
struct MCKResult {
    double best_value;
    std::vector<int> choice_idx;  // choice_idx[i] = index into groups[i]
};

inline MCKResult multiple_choice_knapsack(const std::vector<std::vector<Item>>& groups, int capacity) {
    int m = (int)groups.size();
    const double NEG_INF = -std::numeric_limits<double>::infinity();

    std::vector<std::vector<double>> dp(m + 1, std::vector<double>(capacity + 1, NEG_INF));
    for (int w = 0; w <= capacity; ++w) dp[0][w] = 0.0;
    std::vector<std::vector<int>> choice(m + 1, std::vector<int>(capacity + 1, -1));

    for (int i = 1; i <= m; ++i) {
        const std::vector<Item>& group = groups[i - 1];
        for (int w = 0; w <= capacity; ++w) {
            double best_val = NEG_INF;
            int best_item_idx = -1;
            for (int idx = 0; idx < (int)group.size(); ++idx) {
                const Item& item = group[idx];
                if (item.weight <= w && dp[i - 1][w - item.weight] != NEG_INF) {
                    double candidate = dp[i - 1][w - item.weight] + item.value;
                    if (candidate > best_val) { best_val = candidate; best_item_idx = idx; }
                }
            }
            dp[i][w] = best_val;
            choice[i][w] = best_item_idx;
        }
    }

    MCKResult res;
    res.best_value = dp[m][capacity];
    res.choice_idx.assign(m, -1);
    int w = capacity;
    for (int i = m; i >= 1; --i) {
        int idx = choice[i][w];
        res.choice_idx[i - 1] = idx;
        w -= groups[i - 1][idx].weight;
    }
    return res;
}
