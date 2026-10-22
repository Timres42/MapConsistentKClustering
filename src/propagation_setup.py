"""Temporal drift experiment with center propagation across timesteps.

Thin Python driver around the C++ core. Data loading (via the dataset loaders /
drift pipeline) and plotting stay in Python; the actual experiment - baselines,
historical clusterings, the ``form`` and ``fid`` algorithms and CSV writing - runs
in the compiled C++ program ``cpp/propagation`` (see ``cpp/`` for the source).

The drift sequence produced by the Python loaders is serialized to a temporary
binary file, handed to the C++ binary, which writes the results CSV; the CSV is
then optionally plotted with the existing Python plotting code.

Run as ``python -m src.propagation_setup ...`` from the repo root.
"""

import os
import sys
import struct
import subprocess
import tempfile
import argparse
import numpy as np

# Dataset loaders and the drift pipeline (stay in Python).
from src.include.drift_classes import TemporalDriftPipeline
from data.loader.uber import UberPickupsDataset
from data.loader.wildfire import USWildfireDataset
from data.loader.household_power import HouseholdPowerDataset
from data.loader.online_retail import OnlineRetailDataset
from data.loader.twitter import TwitterGeospatialDataset
from src.include.clustering_classes import ClusteringInstance

# Plotting (stays in Python).
from plots.scripts.plots_propagation_ratio_bviolin import plot_relative_costs

# Heuristic names accepted by the C++ core.
HEURISTIC_CHOICES = ["kmedian_plus_plus", "kmedianppwpost"]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CPP_DIR = REPO_ROOT
CPP_BINARY = os.path.join(REPO_ROOT, "build", "propagation")


def load_drift_sequence(
    dataset_name: str, k: int, seed: int, timesteps: int, point_fraction: float = 1.0
) -> list:
    """Load a temporal drift sequence for the specified dataset."""

    if dataset_name == "uber":
        print("Loading NYC Uber Pickups Dataset...")
        dataset_adapter = UberPickupsDataset(data_path="data/uber/uber-raw-data-apr14.csv", seed=seed)
        pipeline = TemporalDriftPipeline(k=k, seed=seed, timesteps=timesteps)
        drift_sequence = pipeline.execute(dataset_adapter)

    elif dataset_name == "wildfire":
        print("Loading US Wildfire Dataset...")
        dataset_adapter = USWildfireDataset(
            start_date="2006-06-01",
            end_date="2006-06-30",
            seed=seed,
            point_fraction=point_fraction,
        )
        pipeline = TemporalDriftPipeline(k=k, seed=seed, timesteps=timesteps)
        drift_sequence = pipeline.execute(dataset_adapter)

    elif dataset_name == "household_power":
        print("Loading Household Power Dataset...")
        dataset_adapter = HouseholdPowerDataset(data_path="data/household_power/household_power.txt", seed=seed)
        pipeline = TemporalDriftPipeline(k=k, seed=seed, timesteps=timesteps)
        drift_sequence = pipeline.execute(dataset_adapter)

    elif dataset_name == "online_retail":
        print("Loading Online Retail Dataset...")
        dataset_adapter = OnlineRetailDataset(data_path="data/online_retail/online_retail.xlsx", seed=seed)
        pipeline = TemporalDriftPipeline(k=k, seed=seed, timesteps=timesteps)
        drift_sequence = pipeline.execute(dataset_adapter)

    elif dataset_name == "twitter":
        # Geo data on the US but still Euclidean distance (not distance on the globe).
        print("Loading Twitter Dataset...")
        dataset_adapter = TwitterGeospatialDataset(data_path="data/twitter/twitter.csv", seed=seed)
        pipeline = TemporalDriftPipeline(k=k, seed=seed, timesteps=timesteps)
        drift_sequence = pipeline.execute(dataset_adapter)

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    return drift_sequence


def serialize_sequence(drift_sequence: list, path: str) -> None:
    """Write the drift sequence to the binary format the C++ core reads.

    Layout (little-endian): int32 T, int32 d, then per timestep int32 n and
    n*d float64 points (row-major).
    """
    d = int(drift_sequence[0].points.shape[1])
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(drift_sequence)))
        f.write(struct.pack("<i", d))
        for inst in drift_sequence:
            pts = np.ascontiguousarray(inst.points, dtype=np.float64)
            if pts.shape[1] != d:
                raise ValueError("All timesteps must share the same dimension.")
            f.write(struct.pack("<i", pts.shape[0]))
            f.write(pts.tobytes())


def built_with_gurobi() -> bool:
    """True if the last `make` enabled Gurobi (recorded by the Makefile in build/.build_config)."""
    config = os.path.join(os.path.dirname(CPP_BINARY), ".build_config")
    return os.path.exists(config) and "-DUSE_GUROBI" in open(config).read()


def ensure_binary(need_opt: bool = False) -> str:
    """Build the C++ core if needed and return the path to the executable."""
    src_files = [
        os.path.join(REPO_ROOT, "Makefile"),
        os.path.join(REPO_ROOT, "src", "propagation.cpp"),
        os.path.join(REPO_ROOT, "src", "include", "core.hpp"),
        os.path.join(REPO_ROOT, "src", "include", "heuristics.hpp"),
        os.path.join(REPO_ROOT, "src", "include", "knapsack.hpp"),
        os.path.join(REPO_ROOT, "src", "include", "form.hpp"),
        os.path.join(REPO_ROOT, "src", "include", "fid.hpp"),
        os.path.join(REPO_ROOT, "src", "include", "opt.hpp"),
    ]
    needs_build = not os.path.exists(CPP_BINARY)
    if not needs_build:
        bin_mtime = os.path.getmtime(CPP_BINARY)
        for sp in src_files:
            if os.path.exists(sp) and os.path.getmtime(sp) > bin_mtime:
                needs_build = True
                break
    if need_opt and not built_with_gurobi():
        needs_build = True  # e.g. built before GUROBI_HOME was set
    if needs_build:
        print("Building C++ core...")
        subprocess.run(["make", "-C", CPP_DIR], check=True)
    if need_opt and not built_with_gurobi():
        raise SystemExit("--opt needs a Gurobi build: set GUROBI_HOME to your Gurobi installation "
                         "(see README, 'Exact optimum (opt)') and rerun.")
    return CPP_BINARY


def run_cpp_experiment(
    drift_sequence: list,
    output_file: str,
    dataset_name: str,
    k: int,
    b_fraction: float,
    heuristic: str,
    opt: bool = False,
    opt_time_limit: float = 3600,
    opt_mip_gap: float = 0.0,
    opt_threads: int = 0,
    opt_formulation: str = "benders",
    opt_algos: str = "fid,mc",
    quiet: bool = False,
    seed: int = None,
    data_path: str = None,
) -> None:
    """Serialize a drift sequence and run the C++ propagation core.

    seed: master seed of the k-median++ randomness in the C++ core (None = legacy
    fixed seeds). data_path: an already serialized drift sequence to reuse (then
    drift_sequence may be None and the file is kept).
    """
    binary = ensure_binary(need_opt=opt)
    own_file = data_path is None
    if own_file:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
            data_path = tf.name
        serialize_sequence(drift_sequence, data_path)
    cmd = [binary, "--data", data_path, "--out", output_file,
           "--dataset", dataset_name, "--k", str(k),
           "--b_fraction", str(b_fraction), "--heuristic", heuristic,
           "--opt", str(opt).lower(), "--opt_time_limit", str(opt_time_limit),
           "--opt_mip_gap", str(opt_mip_gap), "--opt_threads", str(opt_threads),
           "--opt_formulation", opt_formulation, "--opt_algos", opt_algos]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    # Keep normal setup progress visible, but allow batch drivers to suppress it.
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        print(f"Binary failed, inspect data at: {data_path}")
        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=result.stdout, stderr=result.stderr
        )

    # Keep the serialized input on failure for debugging.
    if own_file:
        os.remove(data_path)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate label-consistent clustering with center propagation across temporal datasets."
    )
    parser.add_argument("--dataset", type=str, default="uber",
                        help="Dataset identifier ('uber', 'wildfire', 'household_power', 'online_retail', 'twitter')")
    parser.add_argument("--k", type=int, default=3, help="Number of clusters")
    parser.add_argument("--num_kernels", type=int, default=1,
                        help="Number of kernels/jobs to record in the output filename")
    parser.add_argument("--b_fraction", type=float, default=0.2,
                        help="Budget fraction (b / n) - recalculated per timestep")
    parser.add_argument("--timesteps", type=int, default=10,
                        help="Number of sequential time steps to evaluate")
    parser.add_argument("--point_fraction", type=float, default=1.0,
                        help="For wildfire, retain this fraction of points per daily slice (0 < fraction <= 1)")
    parser.add_argument("--heuristic", type=str, default="kmedianppwpost",
                        choices=HEURISTIC_CHOICES, help="Heuristic strategy")
    parser.add_argument("--results_dir", type=str, default="results/propagation",
                        help="Output directory")
    parser.add_argument("--plots_dir", type=str, default="plots/out",
                        help="Output directory for the generated plot")
    parser.add_argument("--opt", action="store_true",
                        help="Also run the exact algorithms opt_fid / opt_mc (needs a Gurobi build); the exact "
                             "k-median of timestep 1 becomes the initial clustering for all algorithms")
    parser.add_argument("--opt_algos", default="fid,mc",
                        help="Which exact algorithms to run with --opt: 'fid' (optimum of FID's problem), "
                             "'mc' (optimum of FORM's problem) or 'fid,mc' (default)")
    parser.add_argument("--opt_time_limit", type=float, default=3600,
                        help="Gurobi time limit in seconds per opt solve")
    parser.add_argument("--opt_mip_gap", type=float, default=0.0,
                        help="Relative MIP gap at which Gurobi stops (0 = proven optimum)")
    parser.add_argument("--opt_threads", type=int, default=0,
                        help="Gurobi threads per opt solve (0 = all cores)")
    parser.add_argument("--opt_formulation", choices=["benders", "classic"], default="benders",
                        help="MIP formulation for opt_fid / opt_kmedian: 'benders' (cuts on demand; default) "
                             "or 'classic' (n^2 + n variables; for cross-checking). opt_mc always uses benders")
    parser.add_argument("--seed", type=int, default=None,
                        help="Master seed of the k-median++ randomness in the C++ core (restart r at "
                             "timestep t uses seed_seq(seed, t, r)); default: legacy fixed seeds. "
                             "The data seed stays 42, so the input data is identical for every seed")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip generating the plot after the experiment finishes")
    args = parser.parse_args()

    print(f"Running propagation setup on {args.dataset} with k={args.k}, b={args.b_fraction}, "
          f"timesteps={args.timesteps} and heuristic={args.heuristic}")

    os.makedirs(args.results_dir, exist_ok=True)
    seed = 42

    # Load drift sequence (Python loaders).
    drift_sequence = load_drift_sequence(
        args.dataset, args.k, seed, args.timesteps, args.point_fraction
    )
    print(f"Loaded {len(drift_sequence)} timesteps with k={args.k}")

    # Output CSV path records cluster count separately from kernel count.
    output_file = os.path.join(
        args.results_dir,
        f"[{args.timesteps}]_{args.dataset}_k{args.k}_kernels{args.num_kernels}"
        f"_base-{args.heuristic}_b{args.b_fraction}{'_opt' if args.opt else ''}"
        f"{f'_s{args.seed}' if args.seed is not None else ''}.csv")

    run_cpp_experiment(
        drift_sequence=drift_sequence,
        output_file=output_file,
        dataset_name=args.dataset,
        k=args.k,
        b_fraction=args.b_fraction,
        heuristic=args.heuristic,
        opt=args.opt,
        opt_time_limit=args.opt_time_limit,
        opt_mip_gap=args.opt_mip_gap,
        opt_threads=args.opt_threads,
        opt_formulation=args.opt_formulation,
        opt_algos=args.opt_algos,
        seed=args.seed,
    )

    print(f"\nExperiment complete. Results saved to: {output_file}")

    if not args.no_plot:
        print("\nGenerating plot...")
        plot_relative_costs(output_file, args.plots_dir)


if __name__ == "__main__":
    main()