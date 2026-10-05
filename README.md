# Label Consistent Clustering for k-Median under Mapping

A research prototype for label-consistent clustering under mapping, evaluated on temporal
drift datasets via **center propagation**. Python loads and slices a dataset into a sequence
of timesteps; a compiled C++ core does the actual clustering (heuristics, the two
label-consistent algorithms `form` and `fid`, optionally the exact optimum `opt` of the
center-identity variant via Gurobi, and cost bookkeeping) and writes a results CSV;
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
   identity-mapped reassignment) and, with `--opt`, `opt` (`opt.hpp`, exact MIP via Gurobi,
   see [Exact optimum (opt)](#exact-optimum-opt)) under a point budget
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
  brute-force check of `opt`
- `run_experiments.sh`: Batch-runs `src.propagation_setup` across datasets, `k` values, and
  budget fractions, optionally in parallel, optionally with `opt` (`RUN_OPT=1`) and optionally
  plotting afterwards
- `tests/test_opt.cpp`: Compares `opt` with brute-force enumeration on small random instances

### `src/` - C++ core + Python driver
- `propagation.cpp`: Entry point of the compiled core. Loads the binary drift sequence, runs
  the baseline heuristic, `form`, `fid` and (with `--opt`) `opt` per timestep with center
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
  - `opt.hpp`: The `opt` algorithm (`opt_lcc_median_center_identity`) - exact optimum of the
    center-identity variant as a MIP solved with Gurobi's C API; only compiled with
    `-DUSE_GUROBI` (set by the `Makefile` when `GUROBI_HOME` is set)
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
  to plot `relative_cost_ratio` per method as violin plots for a given budget fraction
- `plots_propagation_ratio_bviolin_multiple.py`, `plots_propagation_ratio_multiple.py`: Used by
  `run_plots_for_paper.sh` to build multi-panel figures across several runs at once
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

```bash
pip install numpy pandas scipy matplotlib seaborn openpyxl
```

`make` is invoked automatically by `propagation_setup.py` (it rebuilds `build/propagation`
whenever a source file or the `Makefile` is newer than the binary, or when `--opt` is requested
but the binary was built without Gurobi), but you can also build it directly:

```bash
make
```

### Optional: Gurobi (only for `opt`)

`opt` needs [Gurobi](https://www.gurobi.com) (version 10 or newer; free academic licenses are
available). The build uses Gurobi's **C API** (`libgurobiXYZ`), so the separate C++ wrapper
library is not needed. Point `GUROBI_HOME` at the installation's platform directory (the one
containing `include/` and `lib/`); the `Makefile` then compiles with `-DUSE_GUROBI` and links
the library it finds there:

```bash
export GUROBI_HOME=/Library/gurobi1300/macos_universal2   # macOS (adjust the version)
export GUROBI_HOME=/opt/gurobi1300/linux64                 # Linux
make            # prints "Built with Gurobi (gurobi130): --opt available"
make test-opt   # optional: check opt against brute force on small instances
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
- `--heuristic`: `kmedian_plus_plus` or `kmedianppwpost` (default; these are the only two
  heuristics implemented in `heuristics.hpp`)
- `--results_dir` / `--plots_dir`: output locations (default `results/propagation`, `plots/out`)
- `--no_plot`: skip plotting after the run
- `--opt`: also run `opt` (needs a Gurobi build, see above and
  [Exact optimum (opt)](#exact-optimum-opt))
- `--opt_time_limit`: Gurobi time limit in seconds per `opt` solve (default `3600`)
- `--opt_mip_gap`: relative MIP gap at which Gurobi stops (default `0` = proven optimum)
- `--opt_threads`: Gurobi threads per solve (default `0` = all cores)

This writes a CSV to
`<results_dir>/[<timesteps>]_<dataset>_k<k>_kernels<num_kernels>_base-<heuristic>_b<b_fraction>.csv`
(with `_opt` appended before `.csv` when `--opt` is given) and, unless `--no_plot` is given,
plots it with `plots_propagation_ratio_bviolin.py`.

### Batch experiments

`run_experiments.sh` sweeps datasets x `k` values x budget fractions (see the arrays at the top
of the script to adjust them) and runs them in parallel, bounded by `MAX_JOBS`:

```bash
bash run_experiments.sh

# Customize parallelism and enable plotting after each batch:
MAX_JOBS=4 RUN_PLOTS=1 ./run_experiments.sh

# Include opt (needs GUROBI_HOME, see Requirements):
RUN_OPT=1 OPT_TIME_LIMIT=1800 ./run_experiments.sh
```

Environment variables for `opt`:
- `RUN_OPT=1`: pass `--opt` to every run. These results go to `results/propagation_opt` and
  `plots/propagation_opt` (override with `RESULTS_DIR` / `PLOTS_DIR`), because `opt` changes
  the initial clustering and the instances of all algorithms (see below).
- `OPT_TIME_LIMIT` (default `3600`), `OPT_MIP_GAP` (default `0`): passed to `--opt_time_limit`
  and `--opt_mip_gap`.
- `OPT_THREADS`: Gurobi threads per run; defaults to `CPU cores / MAX_JOBS` so the parallel runs
  do not compete for the same cores.

The batch runner records `MAX_JOBS` in each CSV filename as `_kernels<MAX_JOBS>`.
To plot these batch results later with `run_plots_for_paper.sh`, pass the same value
through `NUM_KERNELS` (for example, `NUM_KERNELS=4 bash plots/scripts/run_plots_for_paper.sh`).

### Exact optimum (opt)

`opt` is the exact optimum of the **center-identity variant** of label-consistent k-median,
the problem `fid` approximates heuristically: given the historical clustering (every point `p`
has a historical center *point* `h(p)`) and the budget `b`, find `k` center points and an
assignment with minimum k-median cost such that at most `b` points are not assigned to their
historical center point. A point whose historical center is replaced counts against the budget
even if it stays in the "same" cluster.

`opt.hpp` solves this as a MIP (every point is a candidate center):

```
min  sum_ij d(i,j) x_ij
s.t. sum_j x_ij = 1               for every point i
     x_ij <= y_j                  for every i, j
     sum_j y_j = k
     sum_i x_{i,h(i)} >= n - b
     y_j, x_{i,h(i)} binary; all other x_ij in [0, 1]
```

The best feasible center set among the historical, baseline, `form` and `fid` solutions of the
timestep is passed to Gurobi as a start solution. The final assignment is recomputed exactly for
the chosen centers (forced moves for replaced centers, the remaining budget to the largest savings).

With `--opt`:
- **Timestep 1**: `opt` computes the exact (unconstrained) k-median of X_1. This clustering,
  instead of the k-median++ baseline, is the initial historical clustering of **all**
  algorithms (`form`, `fid`, `opt`), and it is written as an `opt` row for timestep 1.
- **Later timesteps**: `opt` propagates its own centers like `form` and `fid`, so each X_i
  also contains `opt`'s `k` previous centers. Results with and without `--opt` are therefore not
  directly comparable.
- **Scale**: the model has `n^2 + n` variables and about `n^2` constraints. Exact solves are
  practical for instances up to a few thousand points; use `--opt_time_limit` and check
  `opt_optimal` / `opt_mip_gap` in the CSV.

### Output format

Each row of the results CSV is one (timestep, method) evaluation, with columns: `dataset`,
`timestep`, `num_points`, `k`, `method` (`baseline_heuristic`, `hist_form`/`hist_fid`/`hist_opt`
for the historical clustering, or `form`/`fid`/`opt` for the algorithm's result), `b_fraction`,
the heuristic used, `b_abs_available`, `historical_cost`, `result_cost`, `b_abs_used`,
`execution_time`, `baseline_cost`, `relative_cost_ratio` (cost normalized to the fresh
baseline), and:
- `prev_points`, `prev_points_changed` (only in `form` rows): points of X_i whose coordinates
  already occur in X_{i-1}, and how many of them changed their FORM center through the
  instance creation (nearest-center assignment to the propagated centers)
- `opt_optimal`, `opt_mip_gap` (only in `opt` rows): `1` if Gurobi proved optimality (`0` if
  stopped by the time limit), and the final relative MIP gap

All other cells of these columns are `nan`. `execution_time` is measured in seconds for each
row's method, including the baseline heuristic, historical clustering, `form`, `fid` and `opt`.
For `form`/`fid`, `b_abs_used` counts points whose label changed under the algorithm's mapping;
for `opt` it counts points whose center point differs from their historical center point (the
variant's budget).

### Evaluating center changes from the instance creation

```bash
python plots/scripts/evaluate_center_changes.py                    # all datasets, k = 5 10 15 20
python plots/scripts/evaluate_center_changes.py --datasets uber --k 5 10
python plots/scripts/evaluate_center_changes.py --results_dir results/propagation_opt
```

Prints the `prev_points` / `prev_points_changed` statistics per dataset, `k` and `b`. Files with
the same dataset, `k` and `b` in one directory are pooled, so keep `opt` runs in their own
directory (as `run_experiments.sh` does).

### Plotting for the paper

```bash
bash plots/scripts/run_plots_for_paper.sh
```

## Notes

- This repository is intended for experimentation and research rather than production use.
- Only the C++ core in `src/include/*.hpp` computes clusterings; there is no separate Python
  implementation of the algorithms to keep in sync.