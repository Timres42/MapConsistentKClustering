"""Run wildfire propagation experiments at progressively larger point counts."""

import argparse
import os
import subprocess

from data.loader.wildfire import USWildfireDataset
from src.include.drift_classes import TemporalDriftPipeline
from src.propagation_setup import run_cpp_experiment

POINT_FRACTIONS = [round(i / 10, 1) for i in range(1, 11)]
K_VALUES = [5, 10]
B_FRACTIONS = [round(i / 10, 1) for i in range(11)]
TIMESTEPS = 30
SEED = 42
# The exact k-median baseline, and kmedianppwpost for runs where it fails.
HEURISTICS = ["optimal", "kmedianppwpost"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run wildfire FORM/FID/baseline experiments for the node-scaling study."
    )
    parser.add_argument(
        "--results_dir",
        default="results/wildfire_node_scaling",
        help="Root folder for the per-fraction result directories",
    )
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--opt_time_limit", type=float, default=3600,
                        help="Gurobi time limit in seconds per solve")
    parser.add_argument("--opt_mip_gap", type=float, default=0.0,
                        help="Relative MIP gap at which Gurobi stops (0 = proven optimum)")
    parser.add_argument("--heuristics", nargs="+", default=HEURISTICS,
                        help="Baselines, each tried if the previous one fails")
    parser.add_argument("--reference", choices=["baseline", "optimal"], default="baseline",
                        help="Cost the ratios are normalized to (see src.propagation_setup)")
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
            pipeline = TemporalDriftPipeline(k=k, seed=SEED, timesteps=args.timesteps)
            drift_sequence = pipeline.execute(dataset)
            for b_fraction in B_FRACTIONS:
                run += 1
                print(
                    f"[{run}/{total}] wildfire point_fraction={point_fraction}, "
                    f"k={k}, b={b_fraction}",
                    flush=True,
                )
                for heuristic in args.heuristics:
                    output_file = os.path.join(
                        results_dir,
                        f"[{args.timesteps}]_wildfire_k{k}_kernels1"
                        f"_base-{heuristic}_b{b_fraction}.csv",
                    )
                    try:
                        run_cpp_experiment(
                            drift_sequence=drift_sequence,
                            output_file=output_file,
                            dataset_name="wildfire",
                            k=k,
                            b_fraction=b_fraction,
                            heuristic=heuristic,
                            opt_time_limit=args.opt_time_limit,
                            opt_mip_gap=args.opt_mip_gap,
                            reference=args.reference,
                            quiet=True,
                        )
                        break
                    except (subprocess.CalledProcessError, SystemExit):  # SystemExit: no Gurobi build
                        # A partial result must not be picked up by the plots.
                        if os.path.exists(output_file):
                            os.remove(output_file)
                        if heuristic == args.heuristics[-1]:
                            raise
                        print(f"Baseline {heuristic} failed, trying the next one", flush=True)


if __name__ == "__main__":
    main()
