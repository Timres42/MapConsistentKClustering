"""Runtime of FORM per dataset: total runtime and per-timestep distribution, side by side."""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from result_files import find_csv

DATASETS = ["twitter", "online_retail", "household_power", "wildfire", "uber"]
K_VALUES = [5, 10, 15, 20]
B_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
METHOD = "form"


def load_runtimes(results_dir, ds, k, b, timesteps, heuristics):
    """Return the per-timestep runtimes of METHOD, or None if the file is missing."""
    csv_path = find_csv(Path(results_dir) / ds, ds, k, timesteps, heuristics, b)
    if csv_path is None:
        print(f"missing: {ds} k={k} b={b}")
        return None
    df = pd.read_csv(csv_path)
    return df.loc[df["method"] == METHOD, "execution_time"].to_numpy()


def plot_dataset(ds, results_dir, output_dir, timesteps, heuristics):
    runtimes = {(k, b): load_runtimes(results_dir, ds, k, b, timesteps, heuristics)
                for k in K_VALUES for b in B_FRACTIONS}

    width = 0.8 / len(B_FRACTIONS)
    x = np.arange(len(K_VALUES))
    colors = plt.cm.viridis(np.linspace(0, 1, len(B_FRACTIONS)))
    offset = lambda i: (i - (len(B_FRACTIONS) - 1) / 2) * width

    fig, (ax_total, ax_box) = plt.subplots(1, 2, figsize=(14, 4.5))

    # --- Left: total runtime, grouped bars (x = k, bars = b) ---
    for i, b in enumerate(B_FRACTIONS):
        totals = [runtimes[k, b].sum() if runtimes[k, b] is not None else np.nan
                  for k in K_VALUES]
        ax_total.bar(x + offset(i), totals, width, label=f"b = {b}", color=colors[i])
    ax_total.set_ylabel(f"Total runtime of {METHOD.upper()} over {timesteps} timesteps [s]")
    ax_total.set_title("Total runtime")

    # --- Right: box plot of per-timestep runtimes, grouped (x = k, boxes = b) ---
    for i, b in enumerate(B_FRACTIONS):
        data, pos = [], []
        for j, k in enumerate(K_VALUES):
            t = runtimes[k, b]
            if t is not None and len(t) > 0:
                data.append(t)
                pos.append(x[j] + offset(i))
        if not data:
            continue
        ax_box.boxplot(data, positions=pos, widths=width * 0.85, patch_artist=True,
                       boxprops=dict(facecolor=colors[i], alpha=0.8),
                       medianprops=dict(color="black"),
                       flierprops=dict(marker="o", markersize=3, alpha=0.6))
    ax_box.set_ylabel(f"Runtime of {METHOD.upper()} per timestep [s]")
    ax_box.set_title("Distribution of runtimes per timestep")

    for ax in (ax_total, ax_box):
        ax.set_xticks(x)
        ax.set_xticklabels([f"k = {k}" for k in K_VALUES])
        ax.set_xlim(-0.5, len(K_VALUES) - 0.5)
        ax.grid(axis="y", alpha=0.3)

    # The legend sits above both panels, so it cannot cover any data.
    handles, labels = ax_total.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.legend(handles, labels, loc="upper center", ncol=len(B_FRACTIONS))

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"runtime_{METHOD}_{ds}.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", default="results/propagation",
                        help="Folder with one subfolder per dataset")
    parser.add_argument("--output_dir", default="plots/paper")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--timesteps", type=int, default=30)
    parser.add_argument("--heuristic", nargs="+", default=["kmedianppwpost"],
                        help="Baselines in order of preference")
    args = parser.parse_args()

    for ds in args.datasets:
        plot_dataset(ds, args.results_dir, args.output_dir, args.timesteps, args.heuristic)


if __name__ == "__main__":
    main()
