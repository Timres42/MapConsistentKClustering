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
#include <cerrno>
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
    std::vector<std::string> tail;  // cells appended to every data row (e.g. the seed)
    void row(const std::vector<std::string>& in) {
        std::vector<std::string> cells = in;
        if (ncols == 0) {
            ncols = cells.size();
        } else {
            while (cells.size() + tail.size() < ncols) cells.push_back("nan");
            cells.insert(cells.end(), tail.begin(), tail.end());
        }
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
    if (!f) { std::fprintf(stderr, "cannot open data file %s: %s (errno %d)\n", path.c_str(), std::strerror(errno), errno); std::exit(1); }
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
    return HeuristicKind::KMppPost;  // default kmedianppwpost, also the MIP start of "optimal"
}

// compute_baseline: best of n_restarts fresh-RNG heuristic runs.
// Seed protocol:
//   seed >= 0 (--seed S): restart r at timestep t uses mt19937(seed_seq{S, t, r}),
//                         so every (S, t, r) gets its own, non-overlapping stream.
//   seed <  0 (default):  legacy seeds 42 + 100 t + r (reproduces earlier results),
//                         or std::random_device with --reproducible false.
static Clustering compute_baseline(const Points& P, int k, HeuristicKind kind,
                                   bool reproducible, int t_idx, double& best_cost_out,
                                   long long seed = -1, int n_restarts = 5) {
    double best_cost = std::numeric_limits<double>::infinity();
    Clustering best;
    std::random_device rd;
    for (int restart = 0; restart < n_restarts; ++restart) {
        std::mt19937 generator;
        if (seed >= 0) {
            std::seed_seq seq{(unsigned)(seed & 0xffffffffLL), (unsigned)(seed >> 32),
                              (unsigned)t_idx, (unsigned)restart};
            generator.seed(seq);
        } else {
            unsigned int restart_seed = reproducible ? 42 + t_idx * 100 + restart : rd();
            generator.seed(restart_seed);
        }
        Clustering cand = run_heuristic(kind, P, k, generator);
        double c = cand.cost();
        if (c < best_cost) { best_cost = c; best = cand; }
    }
    best_cost_out = best_cost;
    return best;
}

// Baseline as selected by --heuristic: compute_baseline, or with "optimal" the exact
// k-median (opt_kmedian), warm-started from compute_baseline's solution. The CSV cells say
// whether the exact solve proved optimality and at which gap ("nan" for the heuristics).
struct OptSettings { double time_limit; double mip_gap; int threads; std::string formulation; };
struct BaselineStatus { std::string optimal = "nan"; std::string mip_gap = "nan"; };

static Clustering select_baseline(const Points& P, int k, HeuristicKind kind, bool reproducible, int t_idx,
                                  long long seed, bool optimal_baseline, const OptSettings& opt,
                                  double& cost_out, BaselineStatus& status) {
    Clustering baseline = compute_baseline(P, k, kind, reproducible, t_idx, cost_out, seed);
#ifdef USE_GUROBI
    if (optimal_baseline) {
        OptInfo info;
        baseline = opt_kmedian(P, k, {baseline.centers}, opt.time_limit, opt.mip_gap, opt.threads, info,
                               opt.formulation).clustering;
        cost_out = baseline.cost();
        status.optimal = info.optimal ? "1" : "0";
        status.mip_gap = fmt(info.mip_gap);
        if (!info.optimal)
            std::fprintf(stderr, "  [optimal] timestep %d: optimality not proven, gap %g\n", t_idx + 1, info.mip_gap);
    }
#endif
    return baseline;
}

#ifdef USE_GUROBI
// --reference optimal: the exact k-median of the instance, warm-started from the baseline.
// All costs are normalized to it, while FORM and FID still receive the heuristic baseline.
static Clustering exact_reference(const Points& P, int k, const Clustering& baseline, int t_idx,
                                  const OptSettings& opt, OptInfo& info) {
    Clustering reference = opt_kmedian(P, k, {baseline.centers}, opt.time_limit, opt.mip_gap, opt.threads,
                                       info, opt.formulation).clustering;
    if (!info.optimal)
        std::fprintf(stderr, "  [reference] timestep %d: optimality not proven, gap %g\n", t_idx + 1, info.mip_gap);
    return reference;
}
#endif

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
    // Master seed of the k-median++ randomness (see compute_baseline); -1 = legacy seeds.
    const std::string seed_str = get_arg(argc, argv, "--seed", "-1");
    const long long seed = std::stoll(seed_str);
    HeuristicKind kind = parse_heuristic(heur_name);
    // Exact reference algorithms (need Gurobi, see opt.hpp):
    //   opt_fid - optimum of the center-identity variant, on the instance FID solves
    //   opt_mc  - optimum of the mapping variant, on the instance FORM solves
    // (with --instances shared both are FORM's instance).
    bool use_opt          = parse_bool(get_arg(argc, argv, "--opt", "false"));
    const bool optimal_baseline = (heur_name == "optimal");
    // Cost all ratios are normalized to: the baseline itself, or with "optimal" the exact
    // k-median of each instance (row "opt_kmedian"); FORM and FID still get the baseline.
    std::string reference = get_arg(argc, argv, "--reference", "baseline");
    if (reference != "baseline" && reference != "optimal") {
        std::fprintf(stderr, "--reference must be baseline or optimal (got '%s')\n", reference.c_str());
        return 1;
    }
    const bool optimal_reference = (reference == "optimal" && !optimal_baseline);
    // Instances of the timesteps: "separate" - FORM and FID each propagate their centers and
    // solve their own instance; "shared" - only FORM propagates, and all algorithms solve
    // FORM's instance with FORM's historical clustering.
    std::string instances = get_arg(argc, argv, "--instances", "separate");
    if (instances != "separate" && instances != "shared") {
        std::fprintf(stderr, "--instances must be separate or shared (got '%s')\n", instances.c_str());
        return 1;
    }
    const bool shared_instance = (instances == "shared");
    std::string opt_algos = get_arg(argc, argv, "--opt_algos", "fid,mc");
    const bool run_opt_fid = use_opt && opt_algos.find("fid") != std::string::npos;
    const bool run_opt_mc  = use_opt && opt_algos.find("mc") != std::string::npos;
    if (use_opt && !run_opt_fid && !run_opt_mc) {
        std::fprintf(stderr, "--opt_algos must contain fid and/or mc (got '%s')\n", opt_algos.c_str());
        return 1;
    }
    double opt_time_limit = std::stod(get_arg(argc, argv, "--opt_time_limit", "3600"));
    double opt_mip_gap    = std::stod(get_arg(argc, argv, "--opt_mip_gap", "0"));
    int    opt_threads    = std::stoi(get_arg(argc, argv, "--opt_threads", "0"));  // 0 = all cores
    std::string opt_formulation = get_arg(argc, argv, "--opt_formulation", "benders");  // or "classic"
    const OptSettings opt_settings{opt_time_limit, opt_mip_gap, opt_threads, opt_formulation};
#ifndef USE_GUROBI
    (void)opt_time_limit; (void)opt_mip_gap; (void)opt_threads; (void)opt_formulation; (void)run_opt_fid; (void)run_opt_mc;
    if (use_opt || optimal_baseline || optimal_reference) {
        std::fprintf(stderr, "--opt, --heuristic optimal and --reference optimal require a build with Gurobi "
                             "(-DUSE_GUROBI), see opt.hpp\n");
        return 1;
    }
#endif

    std::vector<Points> seq = load_sequence(data_path);
    int T = (int)seq.size();
    std::printf("Loaded %d timesteps with k=%d\n", T, k);

    FILE* f = std::fopen(out_path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "cannot open output %s: %s (errno %d)\n", out_path.c_str(), std::strerror(errno), errno); return 1; }
    CsvWriter csv{f, 0, {}};
    csv.row({"dataset", "timestep", "num_points", "k", "method", "b_fraction",
             "baseline/sigma used", "b_abs_available", "historical_cost", "result_cost",
             "b_abs_used", "execution_time", "baseline_cost", "relative_cost_ratio",
             "prev_points", "prev_points_changed", "opt_optimal", "opt_mip_gap", "seed"});
    csv.tail = {seed >= 0 ? std::to_string(seed) : std::string("nan")};

    // --- Timestep 1: fresh baseline only ---
    const Points& P0 = seq[0];
    int n0 = P0.n;
    int b_abs0 = (int)(b_frac * n0);
    double t0_baseline_cost;
    BaselineStatus t0_status;
    auto baseline_start = std::chrono::steady_clock::now();
    Clustering baseline0 = select_baseline(P0, k, kind, reproducible, 0, seed, optimal_baseline, opt_settings,
                                           t0_baseline_cost, t0_status);
    auto baseline_end = std::chrono::steady_clock::now();
    double baseline_elapsed = std::chrono::duration<double>(baseline_end - baseline_start).count();

    // Row of the exact reference (--reference optimal), written after the baseline row.
    std::vector<std::string> reference_row;
    double t0_reference_cost = t0_baseline_cost;
#ifdef USE_GUROBI
    if (optimal_reference) {
        OptInfo info;
        t0_reference_cost = exact_reference(P0, k, baseline0, 0, opt_settings, info).cost();
        reference_row = {dataset, "1", std::to_string(n0), std::to_string(k), "opt_kmedian",
                         b_str, heur_name, std::to_string(b_abs0), "nan", fmt(t0_reference_cost),
                         "nan", fmt(info.runtime), fmt(t0_reference_cost), "1.0", "nan", "nan",
                         info.optimal ? "1" : "0", fmt(info.mip_gap)};
    }
#endif

    csv.row({dataset, "1", std::to_string(n0), std::to_string(k), "baseline_heuristic",
             b_str, heur_name, std::to_string(b_abs0), "nan", fmt(t0_baseline_cost),
             "nan", fmt(baseline_elapsed), fmt(t0_reference_cost),
             fmt(t0_reference_cost > 0 ? t0_baseline_cost / t0_reference_cost : 1.0), "nan", "nan",
             t0_status.optimal, t0_status.mip_gap});
    if (!reference_row.empty()) csv.row(reference_row);

    // Initial clustering propagated by every algorithm: the k-median++ baseline, or
    // with --opt the exact k-median optimum of the first timestep (row "opt_kmedian"),
    // which an optimal baseline already is. With --reference optimal the algorithms
    // keep the baseline, and the exact k-median is only the reference.
    Clustering init_clust = baseline0;
#ifdef USE_GUROBI
    if (use_opt && !optimal_baseline && !optimal_reference) {
        std::printf("  Computing exact k-median (opt_kmedian) for timestep 1...\n");
        OptInfo info;
        MappedClustering opt0 = opt_kmedian(P0, k, {baseline0.centers}, opt_time_limit, opt_mip_gap,
                                            opt_threads, info, opt_formulation);
        init_clust = opt0.clustering;
        double opt0_cost = init_clust.cost();
        csv.row({dataset, "1", std::to_string(n0), std::to_string(k), "opt_kmedian",
                 b_str, heur_name, std::to_string(b_abs0), "nan", fmt(opt0_cost),
                 "nan", fmt(info.runtime), fmt(t0_baseline_cost),
                 fmt(t0_baseline_cost > 0 ? opt0_cost / t0_baseline_cost : 1.0),
                 "nan", "nan", info.optimal ? "1" : "0", fmt(info.mip_gap)});
    }
#endif

    // Only FORM and FID propagate their centers (with --instances shared only FORM). Each
    // algorithm solves the instance of instance_of (index into algo_coords); the centers of
    // the other algorithms are dropped.
    std::vector<std::string> ALGOS = {"form", "fid"};
    const size_t fid_instance = shared_instance ? 0 : 1;
    std::vector<size_t> instance_of = {0, fid_instance};
    std::vector<std::vector<double>> algo_coords(shared_instance ? 1 : 2);  // each k*d, laid out row-major
    if (run_opt_fid) { ALGOS.push_back("opt_fid"); instance_of.push_back(fid_instance); }
    if (run_opt_mc) { ALGOS.push_back("opt_mc"); instance_of.push_back(0); }
    for (size_t a = 0; a < algo_coords.size(); ++a) {
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

        // Build the augmented instance: current points + the previous centers of FORM and FID
        // (with --instances shared only FORM's).
        Points aug;
        aug.d = d;
        aug.n = n_t + (int)algo_coords.size() * k;
        aug.data.reserve((size_t)aug.n * d);
        aug.data.insert(aug.data.end(), Pt.data.begin(), Pt.data.end());
        std::vector<std::vector<int>> center_idx_map(algo_coords.size());
        int cursor = n_t;
        for (size_t a = 0; a < algo_coords.size(); ++a) {
            aug.data.insert(aug.data.end(), algo_coords[a].begin(), algo_coords[a].end());
            center_idx_map[a].resize(k);
            for (int c = 0; c < k; ++c) center_idx_map[a][c] = cursor + c;
            cursor += k;
        }

        // Compute b
        int b_abs = (int)(b_frac * aug.n);
        std::printf("\nTimestep %d/%d: n=%d, budget=%d\n", t_idx + 1, T, aug.n, b_abs);

        // Fresh baseline on the augmented instance.
        std::printf("  Computing baseline %s...\n", heur_name.c_str());
        double best_baseline_cost;
        BaselineStatus baseline_status;
        baseline_start = std::chrono::steady_clock::now();
        Clustering baseline = select_baseline(aug, k, kind, reproducible, t_idx, seed, optimal_baseline, opt_settings,
                                              best_baseline_cost, baseline_status);
        baseline_end = std::chrono::steady_clock::now();
        baseline_elapsed = std::chrono::duration<double>(baseline_end - baseline_start).count();

        // Center sets of this timestep's solutions, used as MIP start candidates for opt.
        std::vector<std::vector<int>> opt_starts = {baseline.centers};

        // Cost all rows of this timestep are normalized to.
        double reference_cost = best_baseline_cost;
        reference_row.clear();
#ifdef USE_GUROBI
        if (optimal_reference) {
            std::printf("  Computing exact k-median reference...\n");
            OptInfo info;
            Clustering reference_clust = exact_reference(aug, k, baseline, t_idx, opt_settings, info);
            reference_cost = reference_clust.cost();
            opt_starts.push_back(reference_clust.centers);
            reference_row = {dataset, std::to_string(t_idx + 1), std::to_string(aug.n), std::to_string(k),
                             "opt_kmedian", b_str, heur_name, std::to_string(b_abs), "nan",
                             fmt(reference_cost), "nan", fmt(info.runtime), fmt(reference_cost), "1.0",
                             "nan", "nan", info.optimal ? "1" : "0", fmt(info.mip_gap)};
        }
#endif

        csv.row({dataset, std::to_string(t_idx + 1), std::to_string(aug.n), std::to_string(k),
                 "baseline_heuristic", b_str, heur_name, std::to_string(b_abs), "nan",
                 fmt(best_baseline_cost), "nan", fmt(baseline_elapsed), fmt(reference_cost),
                 fmt(reference_cost > 0 ? best_baseline_cost / reference_cost : 1.0), "nan", "nan",
                 baseline_status.optimal, baseline_status.mip_gap});
        if (!reference_row.empty()) csv.row(reference_row);

        for (size_t a = 0; a < ALGOS.size(); ++a) {
            const std::string& algo = ALGOS[a];
            std::printf("  Evaluating %s...\n", algo.c_str());

            auto hist_start = std::chrono::steady_clock::now();
            Clustering hist_clust = create_historical_clustering(aug, center_idx_map[instance_of[a]]);
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

            // Algorithms that share an instance share its historical clustering, already in the
            // CSV (with --instances shared only hist_form).
            if (a < algo_coords.size()) {
                double hist_ratio = reference_cost > 0 ? hist_cost / reference_cost : 1.0;
                csv.row({dataset, std::to_string(t_idx + 1), std::to_string(aug.n), std::to_string(k),
                         "hist_" + algo, b_str, heur_name, std::to_string(b_abs), fmt(hist_cost),
                         fmt(hist_cost), "0", fmt(hist_elapsed), fmt(reference_cost), fmt(hist_ratio), "nan", "nan"});
            }

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
                // MIP start candidates: historical, baseline and this timestep's other results.
                OptInfo info;
                std::vector<std::vector<int>> starts = opt_starts;
                starts.push_back(hist_clust.centers);
                const bool classic = (algo == "opt_fid" && opt_formulation == "classic");
                if (algo == "opt_fid")
                    result = opt_fid(aug, k, hist_clust, b_abs, starts, opt_time_limit, opt_mip_gap,
                                     opt_threads, info, opt_formulation);
                else
                    result = opt_mc(aug, k, hist_clust, b_abs, starts, opt_time_limit, opt_mip_gap,
                                    opt_threads, info);
                opt_optimal = info.optimal ? "1" : "0";
                opt_gap = fmt(info.mip_gap);
                opt_budget_used = info.budget_used;
                std::printf("    %s%s: status %d, gap %.3g, %.1fs", algo.c_str(), classic ? " (classic)" : "",
                            info.status, info.mip_gap, info.runtime);
                if (!classic)
                    std::printf(" (LP %.1fs, %d rounds, %ld cuts)", info.lp_time, info.lp_rounds, info.cuts);
                std::printf("\n");
#endif
            }
            auto end = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            opt_starts.push_back(result.clustering.centers);

            double result_cost = result.clustering.cost();
            // b_abs_used is each algorithm's own budget: label changes under its mapping
            // (form, fid, opt_mc) or changed center points (opt_fid).
            int c_dist = (algo.rfind("opt_", 0) == 0) ? opt_budget_used
                                                      : cluster_distance(hist_clust, result.clustering, result.mapping);
            double relative_ratio = reference_cost > 0 ? result_cost / reference_cost : 1.0;

            // Remember FORM's final labels on X_i for the next instance creation.
            if (algo == "form")
                form_prev_labels = count_labels(aug, result.clustering.assignment, k);

            csv.row({dataset, std::to_string(t_idx + 1), std::to_string(aug.n), std::to_string(k),
                     algo, b_str, heur_name, std::to_string(b_abs), fmt(hist_cost),
                     fmt(result_cost), std::to_string(c_dist), fmt(elapsed),
                     fmt(reference_cost), fmt(relative_ratio),
                     form_prev_points, form_prev_changed, opt_optimal, opt_gap});

            // Propagate the resulting centers of FORM and FID (shared: FORM) to the next timestep.
            if (a < algo_coords.size()) {
                std::vector<double> coords((size_t)k * d);
                for (int c = 0; c < k; ++c)
                    std::memcpy(&coords[(size_t)c * d], aug.row(result.clustering.centers[c]), sizeof(double) * d);
                algo_coords[a] = std::move(coords);
            }
        }
    }

    std::fclose(f);
    std::printf("\nExperiment complete. Results saved to: %s\n", out_path.c_str());
    return 0;
}