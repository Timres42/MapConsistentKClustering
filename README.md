# Label Consistent Clustering for k-Median under Mapping

A research prototype for label-consistent clustering under mapping, evaluated on temporal
drift datasets via **center propagation**. Python loads and slices a dataset into a sequence
of timesteps; a compiled C++ core does the actual clustering (heuristics, the two
label-consistent algorithms `form` and `fid`, and cost bookkeeping) and writes a results CSV;
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
   `form` (`form.hpp`, knapsack-based reassignment, `knapsack.hpp`) and `fid` (`fid.hpp`,
   identity-mapped reassignment) under a point budget `b_abs = b_fraction * n`, and appends a
   row per (timestep, method) to the output CSV. Each algorithm's resulting centers are then
   propagated forward to seed the next timestep's historical clustering.
3. **Python** optionally plots the CSV's `relative_cost_ratio` column (cost relative to the
   unconstrained heuristic baseline) via `plots/scripts/plots_propagation_ratio_bviolin.py`.

There is no Python fallback for the clustering algorithms themselves - `src/include/helpers.py`
only keeps the `euclidean` distance function used while building `ClusteringInstance`s, and
`Clustering`/`ClusteringWithMapping` in `clustering_classes.py` are unused vestiges of an
earlier Python-only implementation.

## Project structure

### Root level
- `README.md`: Project documentation
- `Makefile`: Builds `build/propagation` from `src/propagation.cpp` and `src/include/*.hpp`
- `run_experiments.sh`: Batch-runs `src.propagation_setup` across datasets, `k` values, and
  budget fractions, optionally in parallel and optionally plotting afterwards

### `src/` - C++ core + Python driver
- `propagation.cpp`: Entry point of the compiled core. Loads the binary drift sequence, runs
  the baseline heuristic, `form`, and `fid` per timestep with center propagation, writes the CSV.
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
  - `clustering_classes.py`: `ClusteringInstance` (used to hand points/k/seed to the pipeline)
    plus legacy `Clustering`/`ClusteringWithMapping` dataclasses (see note above)
  - `drift_classes.py`: `BaseTemporalDataset` (adapter interface) and `TemporalDriftPipeline`
    (slices a dataset into per-timestep `ClusteringInstance`s)
  - `helpers.py`: `euclidean` distance function

### `data/` - Dataset loaders
- `loader/`: One `BaseTemporalDataset` subclass per dataset - `uber.py`, `wildfire.py`,
  `household_power.py`, `online_retail.py`, `twitter.py` (see Datasets below)

### `plots/scripts/` - Plotting
- `plots_propagation_ratio_bviolin.py`: Used by `propagation_setup.py` and `run_experiments.sh`
  to plot `relative_cost_ratio` per method as violin plots for a given budget fraction
- `plots_propagation_ratio_bviolin_multiple.py`, `plots_propagation_ratio_multiple.py`: Used by
  `run_plots_for_paper.sh` to build multi-panel figures across several runs at once
- `plots_propagation_ratio.py`: Not currently invoked by any script - an older, single-panel
  variant superseded by the `_bviolin` version
- `maps/`: Standalone scripts to render a dataset's raw geography (`plot_uber_nyc.py`,
  `plot_wildfire.py`, `plot_twitter_geo.py`), each with a matching `run_*_window.sh` wrapper

### Output directories (created on demand, gitignored)
- `results/`: CSV files written by `propagation.cpp`
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
whenever a source file is newer than the binary), but you can also build it directly:

```bash
make
```

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
- `--k`: number of clusters
- `--b_fraction`: reassignment budget as a fraction of points per timestep, recomputed per timestep
- `--timesteps`: number of sequential timesteps to evaluate
- `--heuristic`: `kmedian_plus_plus` or `kmedianppwpost` (default; these are the only two
  heuristics implemented in `heuristics.hpp`)
- `--results_dir` / `--plots_dir`: output locations (default `results/propagation`, `plots/out`)
- `--no_plot`: skip plotting after the run

This writes a CSV to `<results_dir>/[<timesteps>]_<dataset>_k<k>_base-<heuristic>_b<b_fraction>.csv`
and, unless `--no_plot` is given, plots it with `plots_propagation_ratio_bviolin.py`.

### Batch experiments

`run_experiments.sh` sweeps datasets x `k` values x budget fractions (see the arrays at the top
of the script to adjust them) and runs them in parallel, bounded by `MAX_JOBS`:

```bash
bash run_experiments.sh

# Customize parallelism and enable plotting after each batch:
MAX_JOBS=4 RUN_PLOTS=1 ./run_experiments.sh
```

### Output format

Each row of the results CSV is one (timestep, method) evaluation, with columns: `dataset`,
`timestep`, `num_points`, `k`, `method` (`baseline_heuristic`, `hist_us`/`hist_fid` for the
historical clustering, or `form`/`fid` for the algorithm's result), `b_fraction`, the heuristic
used, `b_abs_available`, `historical_cost`, `result_cost`, `b_abs_used`, `execution_time`,
`baseline_cost`, and `relative_cost_ratio` (cost normalized to the fresh baseline).

### Plotting for the paper

```bash
bash plots/scripts/run_plots_for_paper.sh
```

## Notes

- This repository is intended for experimentation and research rather than production use.
- Only the C++ core in `src/include/*.hpp` computes clusterings; there is no separate Python
  implementation of the algorithms to keep in sync.
