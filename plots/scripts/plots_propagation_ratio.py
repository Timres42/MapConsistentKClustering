#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

B_FRACTIONS = (0.0, 0.1, 0.5)

PALETTE = {
    "form": "#1f77b4",
    "hist_form": "#7fb3d9",
    "fid": "#ff7f0e",
    "hist_fid": "#ffbb78",
}

LINESTYLES = {
    "baseline_heuristic": "-",
    "form": "-",
    "hist_form": "--",
    "fid": "-",
    "hist_fid": "--",
}

METHODS_TO_PLOT = ["form", "hist_form", "fid", "hist_fid"]
NON_HIST_METHODS = ["form", "fid"]
METHODS_TO_NAMES = {"form":"FORM", "hist_form":"HIST FORM", "fid":"FID", "hist_fid":"HIST FID"}

DATASET_NAMES = {
    "wildfire": "Wildfire Dataset",
    "uber": "Uber Dataset",
    "twitter": "Twitter Dataset",
    "online_retail": "Online Retail Dataset",
    "household_power": "Household Power Dataset",
}


def csv_path_for(results_dir, dataset, k, timesteps, heuristic, b):
    filename = f"[{timesteps}]_{dataset}_k{k}_base-{heuristic}_b{b}.csv"
    return Path(results_dir) / dataset / filename


def load_dataframes(results_dir, dataset, k, timesteps, heuristic):
    dataframes = {}

    for b in B_FRACTIONS:
        csv_path = csv_path_for(
            results_dir, dataset, k, timesteps, heuristic, b
        )

        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV for b={b}: {csv_path}")

        df = pd.read_csv(csv_path)

        if df.empty:
            raise ValueError(f"CSV is empty: {csv_path}")

        if "relative_cost_ratio" not in df.columns:
            raise ValueError(
                f"'relative_cost_ratio' column missing from: {csv_path}"
            )

        dataframes[b] = df

    return dataframes


def compute_shared_y_limits(dataframes):
    values = []

    for b, df in dataframes.items():
        # Ignore b = 0.0 when determining the y-axis limits.
        if b == 0.0:
            continue

        # Only use the actual US and FID results,
        # not hist_us, hist_fid, or baseline.
        filtered = df[df["method"].isin(["form", "fid"])]

        ratios = filtered["relative_cost_ratio"].dropna()

        if not ratios.empty:
            values.append(ratios)

    if not values:
        raise ValueError(
            "No valid relative_cost_ratio values found for "
            "b != 0.0 and methods 'form'/'fid'."
        )

    all_values = pd.concat(values, ignore_index=True)

    y_min = float(all_values.min())
    y_max = float(all_values.max())

    span = y_max - y_min

    if span > 0:
        padding = 0.05 * span
    else:
        padding = 0.05 * abs(y_max) if y_max != 0 else 0.05

    return y_min - padding, y_max + padding


def plot_panel(ax, df, b, y_min, y_max, show_right_axis=False):
    ax.axhline(
        y=1.0,
        color="black",
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
        label="Baseline (ratio=1)",
    )

    for method in METHODS_TO_PLOT:
        method_data = df[df["method"] == method].sort_values("timestep")

        if method_data.empty:
            continue

        ax.plot(
            method_data["timestep"],
            method_data["relative_cost_ratio"],
            label=METHODS_TO_NAMES[method],
            color=PALETTE.get(method, "gray"),
            linestyle=LINESTYLES.get(method, "-"),
            linewidth=2,
        )

    ax.set_xlabel("Timestep")
    ax.set_title(f"b = {b}")
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)

    points_data = None

    for method in NON_HIST_METHODS:
        method_data = df[df["method"] == method].sort_values("timestep")
        if not method_data.empty and "num_points" in method_data.columns:
            points_data = method_data
            break

    if points_data is not None:
        ax_points = ax.twinx()

        ax_points.plot(
            points_data["timestep"],
            points_data["num_points"],
            color="gray",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label="Number of points",
        )

        if show_right_axis:
            ax_points.set_ylabel("Number of Points")
        else:
            # Hide right-side ticks and labels for the first two plots.
            ax_points.tick_params(
                axis="y",
                #right=False,
                labelright=False,
            )

        ax_points.grid(False)


def main():
    parser = argparse.ArgumentParser(
        description="Create three side-by-side Plot-1 panels for b=0.0, 0.1, and 0.5."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--timesteps", required=True, type=int)
    parser.add_argument("--heuristic", default="kmedianppwpost")
    parser.add_argument("--results_dir", default="results/propagation")
    parser.add_argument("--output_dir", default="plots/paper")
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
    })

    dataframes = load_dataframes(
        args.results_dir,
        args.dataset,
        args.k,
        args.timesteps,
        args.heuristic,
    )

    y_min, y_max = compute_shared_y_limits(dataframes)

    fig, axes = plt.subplots(1, 3, figsize=(21, 3), sharey=True)

    for i,(ax, b) in enumerate(zip(axes, B_FRACTIONS)):
        plot_panel(ax, dataframes[b], b, y_min, y_max,show_right_axis=(i == len(axes) - 1))

    axes[0].set_ylabel("Relative Cost Ratio")

    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 1.02),
    )

    dataset_title = DATASET_NAMES.get(args.dataset, args.dataset)
    fig.suptitle(
        f"{dataset_title}, k = {args.k}, timesteps = {args.timesteps}",
        fontsize=14,
        y=1.08,
    )

    plt.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)

    output_path = (
        Path(args.output_dir)
        / f"combined_{args.dataset}_k{args.k}_t{args.timesteps}_b0.0_0.1_0.5.png"
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()