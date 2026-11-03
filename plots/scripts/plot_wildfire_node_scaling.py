"""Plot wildfire FORM runtimes with a fitted n log(n) scaling model."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POINT_FRACTIONS = [round(i / 10, 1) for i in range(1, 11)]
K_VALUES = [5, 10]
B_FRACTIONS = [round(i / 10, 1) for i in range(11)]
TIMESTEPS = 30
HEURISTIC = "kmedianppwpost"


def load_form_metrics(csv_path: Path) -> tuple[float, float]:
    """Return total FORM runtime and summed per-timestep n*log(n) workload."""
    results = pd.read_csv(csv_path)
    form = results.loc[results["method"] == "form", ["num_points", "execution_time"]]
    form = form.dropna()
    if form.empty:
        raise ValueError(f"No FORM runtime rows found in {csv_path}")

    point_counts = form["num_points"].to_numpy(dtype=float)
    if np.any(point_counts <= 0):
        raise ValueError(f"FORM rows must have positive point counts in {csv_path}")
    workload = np.sum(point_counts * np.log(point_counts))
    runtime = form["execution_time"].sum()
    return float(runtime), float(workload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot total wildfire FORM runtime for each point fraction, k, and b."
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results/wildfire_node_scaling"),
        help="Root folder created by src.wildfire_node_scaling",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("plots/wildfire_node_scaling"),
        help="Folder for the generated per-(k,b) plots",
    )
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--num_kernels", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(POINT_FRACTIONS))
    colors = plt.cm.viridis(np.linspace(0, 1, len(POINT_FRACTIONS)))

    for k in K_VALUES:
        for b_fraction in B_FRACTIONS:
            runtimes = []
            workloads = []
            for point_fraction in POINT_FRACTIONS:
                csv_path = (
                    args.results_dir
                    / f"fraction_{point_fraction}"
                    / (
                        f"[{args.timesteps}]_wildfire_k{k}_kernels{args.num_kernels}"
                        f"_base-{HEURISTIC}_b{b_fraction}.csv"
                    )
                )
                if not csv_path.exists():
                    print(f"Missing result: {csv_path}")
                    runtimes.append(np.nan)
                    workloads.append(np.nan)
                    continue

                runtime, workload = load_form_metrics(csv_path)
                runtimes.append(runtime)
                workloads.append(workload)

            runtimes_array = np.asarray(runtimes)
            workloads_array = np.asarray(workloads)
            valid = (
                np.isfinite(runtimes_array)
                & np.isfinite(workloads_array)
                & (workloads_array > 0)
            )
            if not valid.any():
                raise ValueError(f"No valid FORM results to fit for k={k}, b={b_fraction}")
            coefficient = np.dot(
                workloads_array[valid], runtimes_array[valid]
            ) / np.dot(workloads_array[valid], workloads_array[valid])
            fitted_runtimes = coefficient * workloads_array

            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.bar(x, runtimes, color=colors, width=0.8, label="Measured")
            ax.plot(
                x,
                fitted_runtimes,
                color="black",
                marker="o",
                linewidth=1.8,
                label=rf"$c\sum_t n_t\log(n_t)$, $c={coefficient:.2e}$",
            )
            ax.set_xticks(x)
            ax.set_xticklabels([str(fraction) for fraction in POINT_FRACTIONS])
            ax.set_xlabel("Fraction of wildfire points retained")
            ax.set_ylabel("Total FORM runtime [s]")
            ax.set_title(f"Wildfire FORM runtime (k={k}, b={b_fraction})")
            ax.grid(axis="y", alpha=0.3)
            ax.legend()
            fig.tight_layout()
            output_path = args.output_dir / f"runtime_form_k{k}_b{b_fraction}.png"
            fig.savefig(output_path, dpi=150)
            plt.close(fig)


if __name__ == "__main__":
    main()
