// C++ port of the main experiment runner (src/propagation_setup.py).
//
// Data loading and plotting stay in Python. The Python driver serializes the
// drift sequence (points per timestep) to a binary file; this program reads it,
// runs the center-propagation experiment, and writes the results CSV that the
// Python plotting code consumes.
//
// Binary input format (little-endian):
//   int32  T                 number of timesteps
//   int32  d                 point dimension
//   repeated T times:
//     int32 n_t              number of points this timestep
//     float64 x n_t*d        points, row-major
#include "core.hpp"
#include "heuristics.hpp"
#include "form.hpp"
#include "fid.hpp"
#include "opt.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

//----  randomness init ----------------------------------------------------
std::mt19937 make_generator(bool reproducible) {
    if (reproducible) {
        return std::mt19937{42};
    }

    return std::mt19937{std::random_device{}()};
}

//---- argument parsing -----------------------------------------------------
static std::string get_arg(int argc, char** argv, const std::string& key, const std::string& def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (key == argv[i]) return argv[i + 1];
    return def;
}

static bool parse_bool(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (value == "true" || value == "1" || value == "yes" || value == "on")
        return true;

    if (value == "false" || value == "0" || value == "no" || value == "off")
        return false;

    throw std::invalid_argument("Invalid boolean value: " + value);
}


// ---- CSV helpers ----------------------------------------------------------
static std::string fmt(double v) {
    if (std::isnan(v)) return "nan";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return buf;
}

struct CsvWriter {
    FILE* f;
    size_t ncols = 0;  // set by the header row; shorter rows are padded with "nan"
    void row(const std::vector<std::string>& cells) {
        if (ncols == 0) ncols = cells.size();
        for (size_t i = 0; i < std::max(cells.size(), ncols); ++i) {
            if (i) std::fputc(',', f);
            std::fputs(i < cells.size() ? cells[i].c_str() : "nan", f);
        }
        std::fputc('\n', f);
        std::fflush(f);
    }
};

// ---- drift sequence loading ----------------------------------------------
static bool read_i32(FILE* f, int32_t& out) { return std::fread(&out, sizeof(int32_t), 1, f) == 1; }

static std::vector<Points> load_sequence(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open data file %s\n", path.c_str()); std::exit(1); }
    int32_t T, d;
    if (!read_i32(f, T) || !read_i32(f, d)) { std::fprintf(stderr, "bad header\n"); std::exit(1); }
    std::vector<Points> seq;
    seq.reserve(T);
    for (int t = 0; t < T; ++t) {
        int32_t n;
        if (!read_i32(f, n)) { std::fprintf(stderr, "bad timestep header\n"); std::exit(1); }
        Points P;
        P.n = n; P.d = d;
        P.data.resize((size_t)n * d);
        if (std::fread(P.data.data(), sizeof(double), (size_t)n * d, f) != (size_t)n * d) {
            std::fprintf(stderr, "bad point block\n"); std::exit(1);
        }
        seq.push_back(std::move(P));
    }
    std::fclose(f);
    return seq;
}

// ---- experiment components ------------------------------------------------
static HeuristicKind parse_heuristic(const std::string& name) {
    if (name == "kmedian_plus_plus") return HeuristicKind::KMpp;
    return HeuristicKind::KMppPost;  // default kmedianppwpost
}

// compute_baseline: best of n_restarts fresh-RNG heuristic runs.
static Clustering compute_baseline(const Points& P, int k, HeuristicKind kind,
                                   bool reproducible, int t_idx, double& best_cost_out,
                                   int n_restarts = 5) {
    double best_cost = std::numeric_limits<double>::infinity();
    Clustering best;
    std::random_device rd;
    for (int restart = 0; restart < n_restarts; ++restart) {
        unsigned int restart_seed = reproducible ? 42 + t_idx * 100 + restart : rd();
        std::mt19937 generator(restart_seed);
        Clustering cand = run_heuristic(kind, P, k, generator);
        double c = cand.cost();
        if (c < best_cost) { best_cost = c; best = cand; }
    }
    best_cost_out = best_cost;
    return best;
}

// create_historical_clustering: nearest-center assignment to given centers.
static Clustering create_historical_clustering(const Points& P, const std::vector<int>& center_indices) {
    std::vector<int> assignment = assign_points(P, center_indices);
    return Clustering(P, center_indices, assignment);
}

// ---- coordinate matching between consecutive instances ---------------------
// Points are identified by their exact coordinates. For every distinct coordinate
// we store how many points of the instance carry it, split by cluster label.
using LabelCounts = std::unordered_map<std::string, std::vector<int>>;

static std::string point_key(const Points& P, int p) {
    std::string key(sizeof(double) * P.d, '\0');
    for (int j = 0; j < P.d; ++j) {
        double v = P.row(p)[j] + 0.0;  // maps -0.0 to +0.0
        std::memcpy(&key[sizeof(double) * j], &v, sizeof(double));
    }
    return key;
}

static LabelCounts count_labels(const Points& P, const std::vector<int>& assignment, int k) {
    LabelCounts counts;
    counts.reserve((size_t)P.n);
    for (int p = 0; p < P.n; ++p) {
        std::vector<int>& c = counts[point_key(P, p)];
        if (c.empty()) c.assign(k, 0);
        c[assignment[p]]++;
    }
    return counts;
}

// Every point of the new instance whose coordinates already occur in the previous
// instance is counted. It counts as changed if no point with these coordinates had
// its new label in the previous instance (for unique coordinates: its label differs).
struct MatchStats { int matched = 0; int changed = 0; };

static MatchStats compare_labels(const LabelCounts& prev, const LabelCounts& curr) {
    MatchStats s;
    for (const auto& kv : curr) {
        auto it = prev.find(kv.first);
        if (it == prev.end()) continue;
        const std::vector<int>& a = it->second;
        const std::vector<int>& b = kv.second;
        for (size_t l = 0; l < b.size(); ++l) {
            s.matched += b[l];
            if (a[l] == 0) s.changed += b[l];
        }
    }
    return s;
}

int main(int argc, char** argv) {
    std::string data_path = get_arg(argc, argv, "--data", "");
    std::string out_path  = get_arg(argc, argv, "--out", "out.csv");
    std::string dataset   = get_arg(argc, argv, "--dataset", "unknown");
    std::string heur_name = get_arg(argc, argv, "--heuristic", "kmedianppwpost");
    std::string b_str     = get_arg(argc, argv, "--b_fraction", "0.2");
    int    k        = std::stoi(get_arg(argc, argv, "--k", "3"));
    double b_frac   = std::stod(b_str);
    //uint64_t seed   = (uint64_t)std::stoll(get_arg(argc, argv, "--seed", "42"));
    bool reproducible = parse_bool(get_arg(argc, argv, "--reproducible", "true"));
    HeuristicKind kind = parse_heuristic(heur_name);
    // OPT: exact optimum of the center-identity variant (needs Gurobi, see opt.hpp).
    bool use_opt          = parse_bool(get_arg(argc, argv, "--opt", "false"));
    double opt_time_limit = std::stod(get_arg(argc, argv, "--opt_time_limit", "3600"));
    double opt_mip_gap    = std::stod(get_arg(argc, argv, "--opt_mip_gap", "0"));
    int    opt_threads    = std::stoi(get_arg(argc, argv, "--opt_threads", "0"));  // 0 = all cores
    std::string opt_formulation = get_arg(argc, argv, "--opt_formulation", "benders");  // or "classic"
#ifndef USE_GUROBI
    (void)opt_time_limit; (void)opt_mip_gap; (void)opt_threads; (void)opt_formulation;
    if (use_opt) {
        std::fprintf(stderr, "--opt requires a build with Gurobi (-DUSE_GUROBI), see opt.hpp\n");
        return 1;
    }
#endif

    std::vector<Points> seq = load_sequence(data_path);
    int T = (int)seq.size();
    std::printf("Loaded %d timesteps with k=%d\n", T, k);

    FILE* f = std::fopen(out_path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "cannot open output %s\n", out_path.c_str()); return 1; }
    CsvWriter csv{f};
    csv.row({"dataset", "timestep", "num_points", "k", "method", "b_fraction",
             "baseline/sigma used", "b_abs_available", "historical_cost", "result_cost",
             "b_abs_used", "execution_time", "baseline_cost", "relative_cost_ratio",
             "prev_points", "prev_points_changed", "opt_optimal", "opt_mip_gap"});

    // --- Timestep 1: fresh baseline only ---
    const Points& P0 = seq[0];
    int n0 = P0.n;
    int b_abs0 = (int)(b_frac * n0);
    double t0_baseline_cost;
    auto baseline_start = std::chrono::steady_clock::now();
    Clustering baseline0 = compute_baseline(P0, k, kind, reproducible, 0, t0_baseline_cost);
    auto baseline_end = std::chrono::steady_clock::now();
    double baseline_elapsed = std::chrono::duration<double>(baseline_end - baseline_start).count();

    csv.row({dataset, "1", std::to_string(n0), std::to_string(k), "baseline_heuristic",
             b_str, heur_name, std::to_string(b_abs0), "nan", fmt(t0_baseline_cost),
             "nan", fmt(baseline_elapsed), fmt(t0_baseline_cost), "1.0", "nan", "nan"});

    // Initial clustering propagated by every algorithm: the k-median++ baseline, or
    // with --opt the exact k-median optimum of the first timestep.
    Clustering init_clust = baseline0;
#ifdef USE_GUROBI
    if (use_opt) {
        std::printf("  Computing exact k-median (opt) for timestep 1...\n");
        OptInfo info;
        MappedClustering opt0 = opt_lcc_median_center_identity(
            P0, k, nullptr, n0, {baseline0.centers}, opt_time_limit, opt_mip_gap, opt_threads, info,
            opt_formulation);
        init_clust = opt0.clustering;
        double opt0_cost = init_clust.cost();
        csv.row({dataset, "1", std::to_string(n0), std::to_string(k), "opt",
                 b_str, heur_name, std::to_string(b_abs0), "nan", fmt(opt0_cost),
                 "nan", fmt(info.runtime), fmt(t0_baseline_cost),
                 fmt(t0_baseline_cost > 0 ? opt0_cost / t0_baseline_cost : 1.0),
                 "nan", "nan", info.optimal ? "1" : "0", fmt(info.mip_gap)});
    }
#endif

    // Per-algorithm propagated center coordinates (order matches ALGOS).
    std::vector<std::string> ALGOS = {"form", "fid"};
    if (use_opt) ALGOS.push_back("opt");
    std::vector<std::vector<double>> algo_coords(ALGOS.size());  // each k*d, laid out row-major
    for (size_t a = 0; a < ALGOS.size(); ++a) {
        std::vector<double> coords((size_t)k * P0.d);
        for (int c = 0; c < k; ++c)
            std::memcpy(&coords[(size_t)c * P0.d], P0.row(init_clust.centers[c]), sizeof(double) * P0.d);
        algo_coords[a] = std::move(coords);
    }

    // FORM's labels on the previous instance X_{i-1}, grouped by coordinates. Labels
    // index FORM's centers, which are propagated in the same order, so label c in
    // X_{i-1} is label c of the historical clustering in X_i.
    LabelCounts form_prev_labels = count_labels(P0, init_clust.assignment, k);

    // --- Propagate across timesteps ---
    for (int t_idx = 1; t_idx < T; ++t_idx) {
        const Points& Pt = seq[t_idx];
        int n_t = Pt.n;
        int d = Pt.d;

        // Build the augmented instance: current points + every algo's previous centers.
        Points aug;
        aug.d = d;
        aug.n = n_t + (int)ALGOS.size() * k;
        aug.data.reserve((size_t)aug.n * d);
        aug.data.insert(aug.data.end(), Pt.data.begin(), Pt.data.end());
        std::vector<std::vector<int>> center_idx_map(ALGOS.size());
        int cursor = n_t;
        for (size_t a = 0; a < ALGOS.size(); ++a) {
            aug.data.insert(aug.data.end(), algo_coords[a].begin(), algo_coords[a].end());
            center_idx_map[a].resize(k);
            for (int c = 0; c < k; ++c) center_idx_map[a][c] = cursor + c;
            cursor += k;
        }

        // Compute b
        int b_abs = (int)(b_frac * aug.n);
        std::printf("\nTimestep %d/%d: n=%d, budget=%d\n", t_idx + 1, T, aug.n, b_abs);

        // Fresh baseline on the augmented instance.
        std::printf("  Computing baseline heuristic...\n");
        double best_baseline_cost;
        baseline_start = std::chrono::steady_clock::now();
        Clustering baseline = compute_baseline(aug, k, kind, reproducible, t_idx, best_baseline_cost);
        baseline_end = std::chrono::steady_clock::now();
        baseline_elapsed = std::chrono::duration<double>(baseline_end - baseline_start).count();

        csv.row({dataset, std::to_string(t_idx + 1), std::to_string(aug.n), std::to_string(k),
                 "baseline_heuristic", b_str, heur_name, std::to_string(b_abs), "nan",
                 fmt(best_baseline_cost), "nan", fmt(baseline_elapsed), fmt(best_baseline_cost), "1.0", "nan", "nan"});

        // Center sets of this timestep's solutions, used as MIP start candidates for opt.
        std::vector<std::vector<int>> opt_starts = {baseline.centers};

        for (size_t a = 0; a < ALGOS.size(); ++a) {
            const std::string& algo = ALGOS[a];
            std::printf("  Evaluating %s...\n", algo.c_str());

            auto hist_start = std::chrono::steady_clock::now();
            Clustering hist_clust = create_historical_clustering(aug, center_idx_map[a]);
            auto hist_end = std::chrono::steady_clock::now();
            double hist_elapsed = std::chrono::duration<double>(hist_end - hist_start).count();
            double hist_cost = hist_clust.cost();

            // Points of X_i whose coordinates already occur in X_{i-1}, and how many of
            // them changed their FORM center through the instance creation. Written to
            // the CSV in the "form" row (nan in all other rows).
            std::string form_prev_points = "nan", form_prev_changed = "nan";
            if (algo == "form") {
                MatchStats ms = compare_labels(form_prev_labels,
                                               count_labels(aug, hist_clust.assignment, k));
                form_prev_points  = std::to_string(ms.matched);
                form_prev_changed = std::to_string(ms.changed);
            }

            double hist_ratio = best_baseline_cost > 0 ? hist_cost / best_baseline_cost : 1.0;
            csv.row({dataset, std::to_string(t_idx + 1), std::to_string(aug.n), std::to_string(k),
                     "hist_" + algo, b_str, heur_name, std::to_string(b_abs), fmt(hist_cost),
                     fmt(hist_cost), "0", fmt(hist_elapsed), fmt(best_baseline_cost), fmt(hist_ratio), "nan", "nan"});

            auto start = std::chrono::steady_clock::now();
            MappedClustering result;
            std::string opt_optimal = "nan", opt_gap = "nan";
            int opt_budget_used = -1;
            if (algo == "form") {
                result = poly_lcc_median_fct(aug, k, hist_clust, b_abs, baseline);
            } else if (algo == "fid") {
                result = poly_lcc_median_f_id(aug, k, hist_clust, b_abs, baseline);
            } else {
#ifdef USE_GUROBI
                OptInfo info;
                opt_starts.push_back(hist_clust.centers);
                result = opt_lcc_median_center_identity(aug, k, &hist_clust, b_abs, opt_starts,
                                                        opt_time_limit, opt_mip_gap, opt_threads, info,
                                                        opt_formulation);
                opt_optimal = info.optimal ? "1" : "0";
                opt_gap = fmt(info.mip_gap);
                opt_budget_used = info.budget_used;
                std::printf("    opt (%s): status %d, gap %.3g, %.1fs", opt_formulation.c_str(),
                            info.status, info.mip_gap, info.runtime);
                if (opt_formulation == "benders")
                    std::printf(" (LP %.1fs, %d rounds, %ld cuts)", info.lp_time, info.lp_rounds, info.cuts);
                std::printf("\n");
#endif
            }
            auto end = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            opt_starts.push_back(result.clustering.centers);

            double result_cost = result.clustering.cost();
            // For opt, b_abs_used is the variant's budget: points whose center point changed.
            int c_dist = (algo == "opt") ? opt_budget_used
                                         : cluster_distance(hist_clust, result.clustering, result.mapping);
            double relative_ratio = best_baseline_cost > 0 ? result_cost / best_baseline_cost : 1.0;

            // Remember FORM's final labels on X_i for the next instance creation.
            if (algo == "form")
                form_prev_labels = count_labels(aug, result.clustering.assignment, k);

            csv.row({dataset, std::to_string(t_idx + 1), std::to_string(aug.n), std::to_string(k),
                     algo, b_str, heur_name, std::to_string(b_abs), fmt(hist_cost),
                     fmt(result_cost), std::to_string(c_dist), fmt(elapsed),
                     fmt(best_baseline_cost), fmt(relative_ratio),
                     form_prev_points, form_prev_changed, opt_optimal, opt_gap});

            // Propagate this algorithm's resulting centers to the next timestep.
            std::vector<double> coords((size_t)k * d);
            for (int c = 0; c < k; ++c)
                std::memcpy(&coords[(size_t)c * d], aug.row(result.clustering.centers[c]), sizeof(double) * d);
            algo_coords[a] = std::move(coords);
        }
    }

    std::fclose(f);
    std::printf("\nExperiment complete. Results saved to: %s\n", out_path.c_str());
    return 0;
}