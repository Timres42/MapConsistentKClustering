#!/usr/bin/env python3
"""
Create one combined figure for a single dataset/k combination and several b values.

Each row corresponds to one b value.
The three columns reproduce the three plots from plots_for_paper.py:
  1. Relative cost ratio over timesteps + number of points
  2. Distribution of relative cost ratios
  3. Percentage of budget used

The b value is shown once per row on the left, rotated by 90 degrees.
Only one global legend is shown at the top.

Example:
    python plot_b_values_stacked.py \
        --dataset wildfirejune \
        --k 10 \
        --timesteps 50 \
        --b_values 0.0 0.1 0.5 1.0
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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

METHOD_NAMES = {
    "form": "FORM",
    "hist_form": "HIST FORM",
    "fid": "FID",
    "hist_fid": "HIST FID",
}

NON_HIST_METHODS = ["form", "fid"]

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


def load_dataframes(results_dir, dataset, k, timesteps, heuristic, b_values):
    dataframes = {}

    for b in b_values:
        path = csv_path_for(
            results_dir,
            dataset,
            k,
            timesteps,
            heuristic,
            b,
        )

        if not path.exists():
            raise FileNotFoundError(f"Missing CSV for b={b}: {path}")

        df = pd.read_csv(path)

        if df.empty:
            raise ValueError(f"CSV is empty: {path}")

        dataframes[b] = df

    return dataframes


def plot_row(ax1, ax2, ax3, df, b, show_left_ratio_axis, show_right_points_axis):
    # ============================================================
    # Plot 1: Relative cost ratio over timesteps
    # ============================================================
    ax1.axhline(
        y=1.0,
        color="black",
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
        label="Baseline (ratio=1)",
    )

    for method in METHODS_TO_PLOT:
        method_data = df[df["method"] == method].sort_values("timestep")

        if not method_data.empty and "relative_cost_ratio" in df.columns:
            ax1.plot(
                method_data["timestep"],
                method_data["relative_cost_ratio"],
                label=METHOD_NAMES[method],
                color=PALETTE.get(method, "gray"),
                linestyle=LINESTYLES.get(method, "-"),
                linewidth=2,
            )

    ax1.set_xlabel("Timestep")
    ax1.set_title("Cost Ratios over Time / Number of Datapoints")
    ax1.grid(True, alpha=0.3)

    if show_left_ratio_axis:
        ax1.set_ylabel("Relative Cost Ratio")
    else:
        ax1.tick_params(axis="y", left=False, labelleft=False)

    # Add number of points on secondary right axis.
    points_data = None

    for method in NON_HIST_METHODS:
        method_data = df[df["method"] == method].sort_values("timestep")

        if not method_data.empty and "num_points" in method_data.columns:
            points_data = method_data
            break

    if points_data is not None:
        ax1_right = ax1.twinx()

        ax1_right.plot(
            points_data["timestep"],
            points_data["num_points"],
            color="gray",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label="Number of points",
        )

        if show_right_points_axis:
            ax1_right.set_ylabel("Number of Datapoints")
        else:
            ax1_right.tick_params(
                axis="y",
                right=False,
                labelright=False,
            )

        ax1_right.grid(False)

    # ============================================================
    # Plot 2: Distribution of relative costs
    # ============================================================
    if "relative_cost_ratio" in df.columns:
        method_ratios = (
            df.groupby("method")["relative_cost_ratio"]
            .apply(list)
            .to_dict()
        )

        box_data = []
        box_labels = []
        box_colors = []

        for method in METHODS_TO_PLOT:
            if method in method_ratios:
                ratios = [
                    r
                    for r in method_ratios[method]
                    if pd.notna(r) and np.isfinite(r)
                ]

                if ratios:
                    box_data.append(ratios)
                    box_labels.append(METHOD_NAMES[method])
                    box_colors.append(PALETTE.get(method, "gray"))

        bp = ax2.boxplot(
            box_data,
            tick_labels=box_labels,
            patch_artist=True,
        )

        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax2.axhline(
            y=1.0,
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )

        ax2.set_ylabel("Relative Cost Ratio")
        ax2.set_title("Distribution of Cost Ratios")
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.set_xticklabels(
            box_labels,
            rotation=15,
            ha="right",
        )

    # ============================================================
    # Plot 3: Distribution of budget used (violin plot, FORM vs FID)
    # ============================================================
    # A time-series line here jumps sharply between values because
    # percentage_used only takes on a couple of distinct levels across
    # timesteps, which straight interpolated lines make look misleading.
    # A violin instead shows the full distribution of values per method,
    # which is a more honest representation of "mostly at X%, sometimes Y%".
    violin_data = []
    violin_labels = []
    violin_colors = []

    for method in NON_HIST_METHODS:
        method_data = df[df["method"] == method].sort_values("timestep")

        if (
            not method_data.empty
            and "b_abs_used" in df.columns
            and "num_points" in df.columns
        ):
            percentage_used = (
                method_data["b_abs_used"]
                / method_data["num_points"]
            )
            percentage_used = percentage_used[np.isfinite(percentage_used)]

            if not percentage_used.empty:
                violin_data.append(percentage_used.values)
                violin_labels.append(METHOD_NAMES[method])
                violin_colors.append(PALETTE.get(method, "gray"))

    if violin_data:
        parts = ax3.violinplot(
            violin_data,
            showmeans=True,
            showextrema=True,
        )

        for body, color in zip(parts["bodies"], violin_colors):
            body.set_facecolor(color)
            body.set_alpha(0.6)
            body.set_edgecolor("black")
            body.set_linewidth(1)

        for key in ("cmeans", "cmaxes", "cmins", "cbars"):
            if key in parts:
                parts[key].set_color("black")
                parts[key].set_alpha(0.7)

        ax3.set_xticks(range(1, len(violin_labels) + 1))
        ax3.set_xticklabels(violin_labels)

    ax3.set_title(f"Distribution of Budget Used (max possible = {b})")
    ax3.set_ylabel("Budget Used (%)")
    ax3.grid(True, alpha=0.3, axis="y")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create stacked rows of the three paper plots "
            "for several b values."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset identifier.",
    )
    parser.add_argument(
        "--k",
        required=True,
        type=int,
        help="Number of clusters.",
    )
    parser.add_argument(
        "--timesteps",
        required=True,
        type=int,
        help="Number of timesteps.",
    )
    parser.add_argument(
        "--b_values",
        nargs="+",
        required=True,
        type=float,
        help="List of b values, e.g. 0.0 0.1 0.5 1.0",
    )
    parser.add_argument(
        "--heuristic",
        default="kmedianppwpost",
    )
    parser.add_argument(
        "--results_dir",
        default="results/propagation",
    )
    parser.add_argument(
        "--output_dir",
        default="plots/paper",
    )

    args = parser.parse_args()

    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
        }
    )

    dataframes = load_dataframes(
        results_dir=args.results_dir,
        dataset=args.dataset,
        k=args.k,
        timesteps=args.timesteps,
        heuristic=args.heuristic,
        b_values=args.b_values,
    )

    n_rows = len(args.b_values)

    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(21, 3 * n_rows),
        squeeze=False,
    )

    for row_idx, b in enumerate(args.b_values):
        plot_row(
            axes[row_idx, 0],
            axes[row_idx, 1],
            axes[row_idx, 2],
            dataframes[b],
            b,
            show_left_ratio_axis=True,
            show_right_points_axis=True,
        )

    # Single global legend at the very top.
    handles, labels = axes[0, 0].get_legend_handles_labels()

    # Add the number-of-points line to the global legend manually.
    from matplotlib.lines import Line2D

    handles.append(
        Line2D(
            [0],
            [0],
            color="gray",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
        )
    )
    labels.append("Number of Datapoints")

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.995),
    )

    dataset_title = DATASET_NAMES.get(
        args.dataset,
        args.dataset,
    )

    plt.subplots_adjust(
        left=0.08,
        top=0.92,
        hspace=0.55,
        wspace=0.28,
    )

    # Add each b value once on the far left of its row.
    for row_idx, b in enumerate(args.b_values):
        row_box = axes[row_idx, 0].get_position()
        y_center = (row_box.y0 + row_box.y1) / 2

        fig.text(
            0.04,
            y_center,
            f"b = {b}",
            rotation=90,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )


    os.makedirs(args.output_dir, exist_ok=True)

    b_suffix = "_".join(str(b) for b in args.b_values)

    output_path = (
        Path(args.output_dir)
        / (
            f"stacked_{args.dataset}_"
            f"k{args.k}_t{args.timesteps}_"
            f"b{b_suffix}.png"
        )
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Plot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()