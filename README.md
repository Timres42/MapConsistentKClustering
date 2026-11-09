# Label Consistent Clustering for k-Median under Mapping

A research prototype for label-consistent clustering under mapping, evaluated on temporal
drift datasets via **center propagation**. Python loads and slices a dataset into a sequence
of timesteps; a compiled C++ core does the actual clustering (heuristics, the two
label-consistent algorithms `form` and `fid`, optionally the exact algorithms `opt_mc` and
`opt_fid` - the optima of the problems FORM and FID solve heuristically - via Gurobi, and cost
bookkeeping) and writes a results CSV;
Python optionally plots it.

## How it works (architecture)

The heavy computation lives in C++, not Python:

1. **Python (`src/propagation_setup.py`)** loads a dataset via a `data/loader/*` adapter and
   `TemporalDriftPipeline` (`src/include/drift_classes.py`), which slices it into `timesteps`
   many `ClusteringInstance`s (`src/include/clustering_classes.py`). It serializes their points
   to a small binary file (little-endian `int32 T, int32 d`, then per timestep `int32 n` +
   `n*d float64` points, row-major) and builds `build/propagation` if needed (`make`).
2. **C++ (`src/propagation.cpp` + `src/include/*.hpp`)** reads that binary file and, per
   timestep: computes a fresh heuristic baseline clustering, builds a historical clustering
   by assigning the current points to each algorithm's *previous* timestep centers, runs
   `form` (`form.hpp`, knapsack-based reassignment, `knapsack.hpp`), `fid` (`fid.hpp`,
   identity-mapped reassignment) and, with `--opt`, `opt_mc` / `opt_fid` (`opt.hpp`, exact via
   Gurobi, see [Exact algorithms](#exact-algorithms-opt_mc-opt_fid)) under a point budget
   `b_abs = b_fraction * n`, and appends a row per (timestep, method) to the output CSV. Each
   algorithm's resulting centers are then propagated forward to seed the next timestep's
   historical clustering.
3. **Python** optionally plots the CSV's `relative_cost_ratio` column (cost relative to the
   unconstrained heuristic baseline) via `plots/scripts/plots_propagation_ratio_bviolin.py`.

There is no Python fallback for the clustering algorithms themselves - `src/include/helpers.py`
only keeps the `euclidean` distance function used while building `ClusteringInstance`s, and
`Clustering`/`ClusteringWithMapping` in `clustering_classes.py` are unused vestiges of an
earlier Python-only implementation.

## Project structure

### Root level
- `README.md`: Project documentation
- `Makefile`: Builds `build/propagation` from `src/propagation.cpp` and `src/include/*.hpp`,
  with Gurobi enabled automatically when `GUROBI_HOME` is set; `make test-opt` runs the
  brute-force check of the exact algorithms
- `run_experiments.sh`: Batch-runs `src.propagation_setup` across datasets, `k` values, and
  budget fractions, optionally in parallel, optionally with `opt_mc` / `opt_fid` (`RUN_OPT=1`) and optionally
  plotting afterwards
- `src/wildfire_node_scaling.py`: Runs the wildfire point-fraction scaling experiment for
  fractions 0.1-1.0, `k` 5 and 10, and `b` 0.0-1.0
- `src/seed_study.py`: Runs the seed study (FORM, FID and baseline with 30 independent
  k-median++ seeds per configuration, see [Seed study](#seed-study-variability-of-k-median))
- `tests/test_opt.cpp`: Compares `opt_kmedian`, `opt_fid` (both formulations) and `opt_mc` with
  brute-force enumeration on small random instances, and checks `opt_mc <= opt_fid`,
  `opt_mc <= FORM` and `opt_fid <= FID` for the same historical clustering

### `src/` - C++ core + Python driver
- `propagation.cpp`: Entry point of the compiled core. Loads the binary drift sequence, runs
  the baseline heuristic, `form`, `fid` and (with `--opt`) `opt_fid` / `opt_mc` per timestep with center
  propagation, writes the CSV.
- `propagation_setup.py`: Python driver - dataset loading, serialization, invoking the C++
  binary, and (optionally) plotting. Run as `python -m src.propagation_setup ...`.
- **`include/`** - C++ headers (the actual algorithm implementations) and small Python helpers
  - `core.hpp`: `Points`/`Clustering` structs, Euclidean distance, cost/assignment reductions
  - `heuristics.hpp`: `kmedian_plus_plus` (k-median++ seeding) and `kmedianppwpost` (adds a
    medoid-recentering pass); selected via `--heuristic`
  - `knapsack.hpp`: Multiple-choice knapsack DP, a subroutine of `form.hpp`
  - `form.hpp`: The `form` algorithm (`poly_lcc_median_fct`) - knapsack-based budgeted reassignment
  - `fid.hpp`: The `fid` algorithm (`poly_lcc_median_f_id`) - identity-mapped, regret-ranked
    budgeted reassignment
  - `opt.hpp`: The exact algorithms `opt_mc` (optimum of FORM's problem), `opt_fid` (optimum
    of FID's problem) and `opt_kmedian` (exact k-median, for timestep 1 and `--heuristic optimal`), solved with Gurobi's C API
    via a Benders/radius formulation with on-demand cuts; `opt_fid` can also use the classic
    assignment MIP. Only compiled with `-DUSE_GUROBI` (set by the `Makefile` when `GUROBI_HOME`
    is set)
  - `clustering_classes.py`: `ClusteringInstance` (used to hand points/k/seed to the pipeline)
    plus legacy `Clustering`/`ClusteringWithMapping` dataclasses (see note above)
  - `drift_classes.py`: `BaseTemporalDataset` (adapter interface) and `TemporalDriftPipeline`
    (slices a dataset into per-timestep `ClusteringInstance`s)
  - `helpers.py`: `euclidean` distance function

### `data/` - Dataset loaders
- `loader/`: One `BaseTemporalDataset` subclass per dataset - `uber.py`, `wildfire.py`,
  `household_power.py`, `online_retail.py`, `twitter.py` (see Datasets below)

### `plots/scripts/` - Plotting and evaluation
- `evaluate_center_changes.py`: Prints, per dataset, `k` and `b`, how many points of X_i already
  existed in X_{i-1} and how many of them changed their FORM center through the instance
  creation (mean, standard deviation, max; absolute and in % of all points)
- `plots_propagation_ratio_bviolin.py`: Used by `propagation_setup.py` and `run_experiments.sh`
  to plot `relative_cost_ratio` per method (line, box and budget-used violin plot) for a given
  budget fraction; includes `opt_fid` (green) and `opt_mc` (red) when present (hollow markers
  where Gurobi hit the time limit; the timestep-1 `opt_kmedian` row is left out; older results
  with method `opt` are read as `opt_fid`, and their `hist_opt_*` rows are shown as well)
- `compare_runtimes.py`: Compares the runtime of `form`, `fid`, `opt_fid` and `opt_mc` per dataset, `k` and
  `b` (terminal table plus log-scale figures, see [Comparing runtimes](#comparing-runtimes))
- `plot_wildfire_node_scaling.py`: Creates a separate total-FORM-runtime plot for each
  `(k, b)` combination from the wildfire point-fraction experiment, or with `--summary` one
  figure of the runtime over the number of points
- `plots_propagation_runtime.py`: Total and per-timestep runtime of FORM per dataset, side by side
- `approx_factor_form.py`: FORM / OPT_MC cost ratios as a text report and, with `--tex`, a tex table
- `result_files.py`: Finds a result CSV, preferring the first available of several baselines
- `evaluate_seeds.py`: Evaluates the seed study (mean / sd / 95 % CI across seeds, paired
  FORM vs FID comparison with Wilcoxon test, tables and figures)
- `plots_propagation_ratio_bviolin_multiple.py`, `plots_propagation_ratio_multiple.py`: Used by
  `run_plots_for_paper.sh` to build multi-panel figures across several runs at once; `--opt`
  reads the results of `--opt` runs and adds `opt_fid` and `opt_mc`
- `plots_propagation_ratio.py`: Not currently invoked by any script - an older, single-panel
  variant superseded by the `_bviolin` version
- `maps/`: Standalone scripts to render a dataset's raw geography (`plot_uber_nyc.py`,
  `plot_wildfire.py`, `plot_twitter_geo.py`), each with a matching `run_*_window.sh` wrapper

### Output directories (created on demand, gitignored)
- `results/`: CSV files written by `propagation.cpp` (`run_experiments.sh` uses
  `results/propagation/<dataset>`, or `results/propagation_opt/<dataset>` with `RUN_OPT=1`)
- `plots/`: Generated plot images (everything under `plots/` except `plots/scripts/`)
- `build/`: The compiled `propagation` binary

## Requirements

- A C++17 compiler (`g++` or `clang++` - see `Makefile`'s `CXX`)
- Python 3.x with: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`
- `openpyxl` (only needed to read the Online Retail dataset's `.xlsx` file via pandas)
- Gurobi with a valid license (only needed for `--opt` and `--heuristic optimal`, see
  [Optional: Gurobi](#optional-gurobi-only-for-opt_mc--opt_fid--heuristic-optimal))

```bash
pip install numpy pandas scipy matplotlib seaborn openpyxl
```

`make` is invoked automatically by `propagation_setup.py` (it rebuilds `build/propagation`
whenever a source file or the `Makefile` is newer than the binary, or when `--opt` is requested
but the binary was built without Gurobi), but you can also build it directly:

```bash
make
```

### Wildfire runtime scaling

Run the complete 220-run grid (FORM, FID, and baseline are all recorded by the propagation
driver):

```bash
python -m src.wildfire_node_scaling
python plots/scripts/plot_wildfire_node_scaling.py
```

The baseline is the exact k-median, or `kmedianppwpost` for runs where it fails or Gurobi is
missing. Each result CSV is stored below
`results/wildfire_node_scaling/fraction_<fraction>/`; the plots are written to
`plots/wildfire_node_scaling/`. Wildfire points are sampled uniformly without replacement
within each daily slice. The fixed seed makes samples reproducible and nested across
fractions, while retaining at least one point in each nonempty day. FORM runtime totals
sum the per-timestep `execution_time` values in the CSV. Each runtime plot also overlays
a per-`(k, b)` through-origin least-squares fit of `c * sum_t(n_t * log(n_t))`, using each
FORM row's `num_points` value.

### Seed study (variability of k-median++)

Measures how much the randomness of the k-median++ initialization affects the baseline, FORM
and FID (no exact algorithms):

```bash
python -m src.seed_study                     # wildfire, k 5/10, b 0.0-1.0, seeds 1-30 (660 runs)
python plots/scripts/evaluate_seeds.py       # tables + figures in plots/seeds/
```

**Seed protocol.** The input data is fixed (data seed 42, identical for all runs); only the
k-median++ randomness of the C++ core varies. Each run has a master seed `S` from a fixed,
pre-specified list (default 1-30); restart `r` (of 5) of the baseline at timestep `t` uses
`std::mt19937(std::seed_seq{S, t, r})`, so all `(S, t, r)` streams are distinct. All runs are
reported. `results/seeds/<dataset>/protocol.json` records the configuration (seeds, grid,
restarts, git commit, platform); every CSV has a `seed` column and `_s<S>` in its name. Runs
whose CSV is complete are skipped, so an interrupted study can be resumed by rerunning it.
Options: `--seeds 1-30`, `--k`, `--b`, `--timesteps`, `--jobs` (parallel runs, default half
the cores), `--results_dir` (default `results/seeds`).

**Statistics.** The unit of analysis is one run (seed): metrics are averaged over timesteps
2..T per run. Across the seeds of each `(k, b)`, `evaluate_seeds.py` reports mean, sample
standard deviation, 95 % confidence interval of the mean (t-distribution), median, IQR,
min/max and coefficient of variation; a paired FORM vs FID comparison per seed (both see the
same baseline within a run): mean difference with 95 % CI, number of seeds where FORM is
better / equal / worse, Wilcoxon signed-rank test; and the coefficient of variation of the
baseline cost across seeds per timestep. Outputs in `plots/seeds/`: `seeds_runs_*.csv`
(per run), `seeds_summary_*.csv`, `seeds_paired_*.csv`, `seeds_timesteps_*.csv`, a LaTeX
table `seeds_table_*.tex`, and figures `seeds_<dataset>_k<k>.png` (every seed as a dot, mean
with 95 % CI, paired difference) and `seeds_<dataset>_k<k>_timesteps.png` (mean ± sd per
timestep).

Single runs accept the same option: `python -m src.propagation_setup ... --seed 7`. Without
`--seed`, the legacy fixed seeds (`42 + 100 t + r`) are used, so earlier results stay
reproducible.

### Optional: Gurobi (only for `opt_mc` / `opt_fid` / `--heuristic optimal`)

The exact algorithms need [Gurobi](https://www.gurobi.com) (version 10 or newer; free academic licenses are
available). The build uses Gurobi's **C API** (`libgurobiXYZ`), so the separate C++ wrapper
library is not needed. Point `GUROBI_HOME` at the installation's platform directory (the one
containing `include/` and `lib/`); the `Makefile` then compiles with `-DUSE_GUROBI` and links
the library it finds there:

```bash
export GUROBI_HOME=/Library/gurobi1300/macos_universal2   # macOS (adjust the version)
export GUROBI_HOME=/opt/gurobi1300/linux64                 # Linux
make            # prints "Built with Gurobi (gurobi130): --opt available"
make test-opt   # optional: check the exact algorithms against brute force
```

Gurobi also needs a valid license (`grbgetkey`, or `GRB_LICENSE_FILE`). The pip license that
comes with `pip install gurobipy` only works from Python and cannot be used here.
`make USE_GUROBI=0` builds without Gurobi even if `GUROBI_HOME` is set, and
`GUROBI_LIB=gurobi120` overrides the auto-detected library name. Switching Gurobi on or off
triggers a rebuild automatically.

## Datasets

| name               | source                             | how to get the data |
|--------------------|-------------------------------------|----------------------|
| `uber`             | NYC Uber Pickups (Kaggle)           | place `uber-raw-data-apr14.csv` at `data/uber/uber-raw-data-apr14.csv` |
| `wildfire`         | US Wildfire FPA-FOD (Kaggle/sqlite) | place `FPA_FOD_20170508.sqlite` at `data/wildfire/FPA_FOD_20170508.sqlite` |
| `household_power`  | UCI dataset #235                    | auto-downloaded to `data/household_power/household_power.txt` if missing |
| `online_retail`    | UCI dataset #352                    | auto-downloaded to `data/online_retail/online_retail.xlsx` if missing |
| `twitter`          | UCI dataset #1050                   | auto-downloaded to `data/twitter/twitter.csv` if missing |

`uber` and `wildfire` come from sources without a stable direct-download URL, so those files
need to be placed manually; the other three are fetched automatically on first use.

## Usage

### Run a single experiment

```bash
python -m src.propagation_setup \
  --dataset uber --k 5 --b_fraction 0.2 --timesteps 12 --heuristic kmedianppwpost
```

Key arguments (see `propagation_setup.py`'s `argparse` block for the full list and defaults):
- `--dataset`: one of `uber`, `wildfire`, `household_power`, `online_retail`, `twitter`
- `--k`: number of clusters/centers (unrelated to the kernel count in filenames)
- `--num_kernels`: kernel/job count to record in the output filename (default: `1`; the
  batch runner sets it to `MAX_JOBS`)
- `--b_fraction`: reassignment budget as a fraction of points per timestep, recomputed per timestep
- `--timesteps`: number of sequential timesteps to evaluate
- `--heuristic`: baseline clustering - `kmedian_plus_plus`, `kmedianppwpost` (default; these are
  the only two heuristics implemented in `heuristics.hpp`), or `optimal` for the exact k-median
  (`opt_kmedian` in `opt.hpp`, centers are input points). `form` and `fid` then use it as their baseline, and
  `relative_cost_ratio` is relative to it
- `--results_dir` / `--plots_dir`: output locations (default `results/propagation`, `plots/out`)
- `--no_plot`: skip plotting after the run
- `--opt`: also run the exact algorithms (needs a Gurobi build, see above and
  [Exact algorithms](#exact-algorithms-opt_mc-opt_fid))
- `--opt_algos`: which exact algorithms to run with `--opt`: `mc`, `fid` or `fid,mc` (default)
- `--opt_time_limit`: Gurobi time limit in seconds per solve (default `3600`)
- `--opt_mip_gap`: relative MIP gap at which Gurobi stops (default `0` = proven optimum)
- `--opt_threads`: Gurobi threads per solve (default `0` = all cores; lower it when running jobs
  in parallel)
- `--opt_formulation`: `benders` (default) or `classic` for `opt_fid` / `opt_kmedian`, see
  [Exact algorithms](#exact-algorithms-opt_mc-opt_fid)
- `--seed`: master seed of the k-median++ randomness (see [Seed study](#seed-study-variability-of-k-median));
  default: legacy fixed seeds. Adds `_s<seed>` to the file name

This writes a CSV to
`<results_dir>/[<timesteps>]_<dataset>_k<k>_kernels<num_kernels>_base-<heuristic>_b<b_fraction>.csv`
(with `_opt` and/or `_s<seed>` appended before `.csv` when `--opt` / `--seed` is given) and, unless `--no_plot` is given,
plots it with `plots_propagation_ratio_bviolin.py`.

### Batch experiments

`run_experiments.sh` sweeps datasets x `k` values x budget fractions (see the arrays at the top
of the script to adjust them) and runs them in parallel, bounded by `MAX_JOBS`:

```bash
bash run_experiments.sh

# Customize parallelism and enable plotting after each batch:
MAX_JOBS=4 RUN_PLOTS=1 ./run_experiments.sh

# Include the exact algorithms (needs GUROBI_HOME, see Requirements):
RUN_OPT=1 OPT_TIME_LIMIT=1800 ./run_experiments.sh
```

Environment variables for the exact algorithms:
- `RUN_OPT=1`: pass `--opt` to every run. These results go to `results/propagation_opt` and
  `plots/propagation_opt` (override with `RESULTS_DIR` / `PLOTS_DIR`), because `--opt` changes
  the initial clustering of all algorithms (see below).
- `OPT_ALGOS` (default `fid,mc`), `OPT_TIME_LIMIT` (default `3600`), `OPT_MIP_GAP` (default
  `0`), `OPT_FORMULATION` (default `benders`): passed to `--opt_algos`, `--opt_time_limit`,
  `--opt_mip_gap` and `--opt_formulation`.
- `OPT_THREADS`: Gurobi threads per run; defaults to `CPU cores / MAX_JOBS` so the parallel runs
  do not compete for the same cores.

The batch runner records `MAX_JOBS` in each CSV filename as `_kernels<MAX_JOBS>`.

### Exact algorithms (opt_mc, opt_fid)

Each exact algorithm solves exactly the instance of its heuristic in every timestep: `opt_mc`
gets FORM's historical clustering (every point of X_i assigned to the nearest of FORM's previous
centers) and `opt_fid` FID's, each with the same budget `b`.

**`opt_mc` - optimum of FORM's problem (mapping variant).** Find `k` center points, an
assignment `sigma` and a function `f` from the historical clusters to the new centers (the
*anchor* of each historical cluster; not necessarily injective) with minimum k-median cost such
that at most `b` points change their assignment modulo `f`, i.e. `sigma(p) != f(H(p))`.
For a fixed center set the best anchors and movers follow from exactly FORM's multiple-choice
knapsack; FORM is therefore optimal for the baseline's centers, and `opt_mc` additionally
optimizes the center set. So `opt_mc <= FORM` in every timestep.

**`opt_fid` - optimum of FID's problem (center-identity variant).** Every point `p` has its
historical center *point* `h(p)`; at most `b` points may end up at a different center point.
A point whose historical center is replaced counts against the budget even if it stays in the
"same" cluster. FID stays within this budget, so `opt_fid <= FID` in every timestep. `opt_mc`
and `opt_fid` start from different historical clusterings, so they are not ordered in general.

**Formulation.** Both use a Benders/radius formulation after Duran-Mateluna, Ales & Elloumi,
*An efficient Benders decomposition for the p-median problem* (EJOR 2023), extended by the
budget: `y_j` = point `j` is a center (binary), `theta_i` = cost of point `i`, `s_i` = point `i`
stays (continuous), and for `opt_mc` `z_cj` = `j` is the anchor of historical cluster `c` (binary).

```
opt_fid:  min sum_i theta_i
          s.t. sum_j y_j = k,   s_i <= y_{h(i)},   sum_i s_i >= n - b
               theta_i >= D - sum_{j: d(i,j) < D} (D - d(i,j)) y_j
                             + [D < d(i,h(i))] (d(i,h(i)) - D) s_i                 for all D

opt_mc:   min sum_i theta_i
          s.t. sum_j y_j = k,   sum_j z_cj = 1,   z_cj <= y_j,   sum_i s_i >= n - b
               theta_i >= D1 - sum_j (D1 - d(i,j))^+ y_j + (D2 - D1) s_i
                             - sum_j [(D2 - d(i,j))^+ - (D1 - d(i,j))^+] z_{c(i)j}   for all D1 <= D2
```

The `opt_mc` inequalities are the LP duals of the per-point assignment subproblem (with `z`
fixed to the historical center point they reduce to the `opt_fid` ones). For integral `y` (and
`z`) the strongest inequality gives exactly the cost of a point: distance to its historical
center / anchor if it stays, to the nearest center if it moves. `s` can stay continuous: for
fixed `y` (and `z`) the remaining problem (keep at least `n - b` points, unit weights) has an
integral LP optimum. The inequalities are generated on demand - first in a cutting-plane loop
on the LP relaxation, then by a Gurobi callback (lazy constraints at incumbents, user cuts at
fractional nodes). A parallel Gurobi solve can end on an incumbent whose lazy inequalities it just
received, so the solve is repeated until the final solution satisfies all of them (up to a total
violation of `1e-6` relative to its cost). Model size: `3n` variables plus `k*n` anchor variables for `opt_mc`; memory
mainly two `n x n` neighbor tables (about 40 MB at `n = 1750`).

`opt_fid` (and `opt_kmedian`) can alternatively use the compact assignment MIP with `n^2 + n`
variables (`--opt_formulation classic`), kept for cross-checking - at `n = 1750` its LP
relaxation alone needs several GB of memory per run and often does not finish in time.

The best feasible solution among the historical, baseline and this timestep's other results is
passed to Gurobi as a start solution (for `opt_mc` including `opt_fid`'s result). The final
clustering is recomputed exactly for the chosen centers.

With `--opt`:
- **Timestep 1**: the exact (unconstrained) k-median of X_1 (`opt_kmedian`) replaces the
  k-median++ baseline as the initial clustering of **all** algorithms, and is written as an
  `opt_kmedian` row for timestep 1. With `--heuristic optimal` the baseline already is this
  optimum, so it is reused and no `opt_kmedian` row is written.
- **Later timesteps**: only `form` and `fid` propagate their centers, so each X_i contains the
  previous centers of these two. The centers of `opt_fid` and `opt_mc` are not added to the next
  instance, so `--opt_algos` does not change the instances. Apart from the initial clustering
  above, `--opt` does not change them either.
- **Scale**: instances of a few thousand points are practical; the time goes into
  branch-and-bound over the center choice (for `opt_mc` also the anchors). Use
  `--opt_time_limit` and check `opt_optimal` / `opt_mip_gap` in the CSV. The progress line per
  solve shows the LP phase (`LP <seconds>, <rounds> rounds, <cuts> cuts`).

### Output format

Each row of the results CSV is one (timestep, method) evaluation, with columns: `dataset`,
`timestep`, `num_points`, `k`, `method` (`baseline_heuristic`; `hist_form`/`hist_fid` for the
historical clustering, shared with `opt_mc`/`opt_fid`; `form`/`fid`/`opt_fid`/`opt_mc` for
the algorithm's result; `opt_kmedian` for the exact k-median of timestep 1), `b_fraction`,
the heuristic used, `b_abs_available`, `historical_cost`, `result_cost`, `b_abs_used`,
`execution_time`, `baseline_cost`, `relative_cost_ratio` (cost normalized to the fresh
baseline), and:
- `prev_points`, `prev_points_changed` (only in `form` rows): points of X_i whose coordinates
  already occur in X_{i-1}, and how many of them changed their FORM center through the
  instance creation (nearest-center assignment to the propagated centers)
- `opt_optimal`, `opt_mip_gap` (only in `opt_kmedian`/`opt_fid`/`opt_mc` rows and the
  `baseline_heuristic` rows of `--heuristic optimal` runs): `1` if Gurobi proved optimality (`0` if
  stopped by the time limit), and the final relative MIP gap
- `seed`: the master seed given with `--seed` (in every row; `nan` for the legacy fixed seeds)

All other cells of these columns are `nan`. The method name stays `baseline_heuristic` for an
optimal baseline, so the plotting scripts work unchanged. `execution_time` is measured in seconds for each
row's method, including the baseline heuristic, historical clustering and all algorithms.
`b_abs_used` is each algorithm's own budget: for `form`, `fid` and `opt_mc` the points whose
label changed under the algorithm's mapping; for `opt_fid` the points whose center point
differs from their historical center point. (Results from before the rename call `opt_fid`
just `opt`.)

### Evaluating center changes from the instance creation

```bash
python plots/scripts/evaluate_center_changes.py                    # all datasets, k = 5 10 15 20
python plots/scripts/evaluate_center_changes.py --datasets uber --k 5 10
python plots/scripts/evaluate_center_changes.py --results_dir results/propagation_opt
```

Prints the `prev_points` / `prev_points_changed` statistics per dataset, `k` and `b`. Files with
the same dataset, `k` and `b` in one directory are pooled, so keep `--opt` runs in their own
directory (as `run_experiments.sh` does).

### Comparing runtimes

```bash
python plots/scripts/compare_runtimes.py                          # results/propagation_opt, k = 5 10 15 20
python plots/scripts/compare_runtimes.py --datasets uber --k 5 10
python plots/scripts/compare_runtimes.py --csv results/propagation_opt/uber/*.csv --per_timestep
```

Prints per dataset, `k` and `b` the median / mean / max runtime per timestep of `form`, `fid`,
`opt_fid` and `opt_mc`, how many times slower each exact algorithm is than the heuristic for
the same problem (median `opt_fid`/`fid` and `opt_mc`/`form`), and how many exact solves hit
the time limit (their runtime equals the limit). Writes the runtime table to
`plots/runtime/runtimes`, as well as `plots/runtime/runtime_<dataset>.png` (median runtime
over `b`, log scale, one panel per `k`) and, with `--per_timestep`, one figure per CSV. The
timestep-1 `opt_kmedian` row is excluded.

### Experiments and plots for the paper

```bash
bash run_experiments_plots_paper.sh       # needs a Gurobi build
bash plots/scripts/run_plots_for_paper.sh
```

The first script runs exactly the experiments the second one needs and writes them to
`results/paper` (`no_opt/`, `opt/`, `wildfire_node_scaling/`; logs in `logs/`). It first checks that the
compiled program may write there and stops with a hint if not (macOS can refuse this for `~/Documents`). It remembers finished
runs in `.done/`, so running it again continues after an interruption; delete that folder to start
over. Ctrl+C or killing the script also stops all its running experiments; only `kill -9` cannot be
handled. Runs that fail are listed at the end. Every run uses the exact k-median baseline
(`--heuristic optimal`), and `kmedianppwpost` if that run fails; the plots prefer the `optimal`
result of each `(dataset, k, b)` and print a message when they use the other one. `MAX_JOBS`,
`OPT_THREADS`, `OPT_TIME_LIMIT`, `OPT_MIP_GAP`, `TIMESTEPS` and `RESULTS_ROOT` can be set as
environment variables. The runs for FORM's runtime measurements finish before the exact
algorithms start, so these don't disturb the timings.

The second script writes to `plots/paper`:

| File | Content |
|---|---|
| `stacked_wildfire_k5_t30_b0.0_0.1_0.5.png` | Cost ratio, its distribution and budget used per `b`, with `opt_fid` and `opt_mc` |
| `combined_wildfire-k10_wildfire-k20_online_retail-k10_online_retail-k20_t30.png` | Cost ratios of `form`, `fid`, their historical clusterings, `opt_fid` and `opt_mc` |
| `stacked_<dataset>_k<k>_t30_b0.0_0.01_0.1_0.5_1.0.png` | Per dataset: `k` = 5 and 10 with `opt_fid` and `opt_mc`, `k` = 20 without |
| `runtime_form_<dataset>.png` | Total and per-timestep runtime of FORM |
| `runtime_form_wildfire_vs_n.png` | FORM runtime over the number of points on wildfire |
| `approx_factor_form.tex` (and `.txt`) | Table of the FORM / OPT_MC cost ratio |

## Notes

- This repository is intended for experimentation and research rather than production use.
- Only the C++ core in `src/include/*.hpp` computes clusterings; there is no separate Python
  implementation of the algorithms to keep in sync.