"""Run wildfire propagation experiments at progressively larger point counts."""

import argparse
import os

from data.loader.wildfire import USWildfireDataset
from src.include.drift_classes import TemporalDriftPipeline
from src.propagation_setup import run_cpp_experiment

POINT_FRACTIONS = [round(i / 10, 1) for i in range(1, 11)]
K_VALUES = [5, 10]
B_FRACTIONS = [round(i / 10, 1) for i in range(11)]
TIMESTEPS = 30
SEED = 42
HEURISTIC = "kmedianppwpost"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run wildfire FORM/FID/baseline experiments for the node-scaling study."
    )
    parser.add_argument(
        "--results_dir",
        default="results/wildfire_node_scaling",
        help="Root folder for the per-fraction result directories",
    )
    args = parser.parse_args()
    results_root = os.path.abspath(args.results_dir)

    total = len(POINT_FRACTIONS) * len(K_VALUES) * len(B_FRACTIONS)
    run = 0
    for point_fraction in POINT_FRACTIONS:
        results_dir = os.path.join(results_root, f"fraction_{point_fraction}")
        os.makedirs(results_dir, exist_ok=True)
        dataset = USWildfireDataset(
            start_date="2006-06-01",
            end_date="2006-06-30",
            seed=SEED,
            point_fraction=point_fraction,
        )
        for k in K_VALUES:
            pipeline = TemporalDriftPipeline(k=k, seed=SEED, timesteps=TIMESTEPS)
            drift_sequence = pipeline.execute(dataset)
            for b_fraction in B_FRACTIONS:
                run += 1
                output_file = os.path.join(
                    results_dir,
                    f"[{TIMESTEPS}]_wildfire_k{k}_kernels1"
                    f"_base-{HEURISTIC}_b{b_fraction}.csv",
                )
                print(
                    f"[{run}/{total}] wildfire point_fraction={point_fraction}, "
                    f"k={k}, b={b_fraction}",
                    flush=True,
                )
                run_cpp_experiment(
                    drift_sequence=drift_sequence,
                    output_file=output_file,
                    dataset_name="wildfire",
                    k=k,
                    b_fraction=b_fraction,
                    heuristic=HEURISTIC,
                    quiet=True,
                )


if __name__ == "__main__":
    main()
