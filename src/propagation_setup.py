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


def load_drift_sequence(dataset_name: str, k: int, seed: int, timesteps: int) -> list:
    """Load a temporal drift sequence for the specified dataset."""

    if dataset_name == "uber":
        print("Loading NYC Uber Pickups Dataset...")
        dataset_adapter = UberPickupsDataset(data_path="data/uber/uber-raw-data-apr14.csv", seed=seed)
        pipeline = TemporalDriftPipeline(k=k, seed=seed, timesteps=timesteps)
        drift_sequence = pipeline.execute(dataset_adapter)

    elif dataset_name == "wildfire":
        print("Loading US Wildfire Dataset...")
        dataset_adapter = USWildfireDataset(start_date="2006-06-01", end_date="2006-06-30", seed=seed)
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


def ensure_binary() -> str:
    """Build the C++ core if needed and return the path to the executable."""
    src_files = [
        os.path.join(REPO_ROOT, "src", "propagation.cpp"),
        os.path.join(REPO_ROOT, "src", "include", "core.hpp"),
        os.path.join(REPO_ROOT, "src", "include", "heuristics.hpp"),
        os.path.join(REPO_ROOT, "src", "include", "knapsack.hpp"),
        os.path.join(REPO_ROOT, "src", "include", "form.hpp"),
        os.path.join(REPO_ROOT, "src", "include", "fid.hpp"),
    ]
    needs_build = not os.path.exists(CPP_BINARY)
    if not needs_build:
        bin_mtime = os.path.getmtime(CPP_BINARY)
        for sp in src_files:
            if os.path.exists(sp) and os.path.getmtime(sp) > bin_mtime:
                needs_build = True
                break
    if needs_build:
        print("Building C++ core...")
        subprocess.run(["make", "-C", CPP_DIR], check=True)
    return CPP_BINARY


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate label-consistent clustering with center propagation across temporal datasets."
    )
    parser.add_argument("--dataset", type=str, default="uber",
                        help="Dataset identifier ('uber', 'wildfire', 'household_power', 'online_retail', 'twitter')")
    parser.add_argument("--k", type=int, default=3, help="Number of clusters")
    parser.add_argument("--b_fraction", type=float, default=0.2,
                        help="Budget fraction (b / n) - recalculated per timestep")
    parser.add_argument("--timesteps", type=int, default=10,
                        help="Number of sequential time steps to evaluate")
    parser.add_argument("--heuristic", type=str, default="kmedianppwpost",
                        choices=HEURISTIC_CHOICES, help="Heuristic strategy")
    parser.add_argument("--results_dir", type=str, default="results/propagation",
                        help="Output directory")
    parser.add_argument("--plots_dir", type=str, default="plots/out",
                        help="Output directory for the generated plot")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip generating the plot after the experiment finishes")
    args = parser.parse_args()

    print(f"Running propagation setup on {args.dataset} with k={args.k}, b={args.b_fraction}, "
          f"timesteps={args.timesteps} and heuristic={args.heuristic}")

    os.makedirs(args.results_dir, exist_ok=True)
    seed = 42

    # Load drift sequence (Python loaders).
    drift_sequence = load_drift_sequence(args.dataset, args.k, seed, args.timesteps)
    print(f"Loaded {len(drift_sequence)} timesteps with k={args.k}")

    # Output CSV path (unchanged naming).
    output_file = os.path.join(
        args.results_dir,
        f"[{args.timesteps}]_{args.dataset}_k{args.k}_base-{args.heuristic}_b{args.b_fraction}.csv")

    # Serialize the sequence and run the C++ core.
    binary = ensure_binary()
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
        data_path = tf.name
    serialize_sequence(drift_sequence, data_path)
    cmd = [binary, "--data", data_path, "--out", output_file,
           "--dataset", args.dataset, "--k", str(args.k),
           "--b_fraction", str(args.b_fraction), "--heuristic", args.heuristic]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print(f"Binary failed, inspect data at: {data_path}")
        raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout, stderr=result.stderr)

    # Only clean up on success; keep the temp file around on failure for debugging.
    os.remove(data_path)

    print(f"\nExperiment complete. Results saved to: {output_file}")

    if not args.no_plot:
        print("\nGenerating plot...")
        plot_relative_costs(output_file, args.plots_dir)


if __name__ == "__main__":
    main()