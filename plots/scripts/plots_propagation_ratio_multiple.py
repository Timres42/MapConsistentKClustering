#!/usr/bin/env python3
"""
Create a multi-row figure for several (dataset, k) combinations.

Each row corresponds to one (dataset, k) combination.
Each row contains three plots for b = 0.0, 0.1, and 0.5.

Example:
    python plot_all_datasets_all_k_values.py \
        --datasets wildfirejune uber4hmerged \
        --k_values 3 5 10 \
        --timesteps 50

This creates rows in this order:
    (wildfirejune, 3)
    (wildfirejune, 5)
    (wildfirejune, 10)
    (uber4hmerged, 3)
    (uber4hmerged, 5)
    (uber4hmerged, 10)

Pairwise mode:
    Pass --pairwise to instead pair --datasets and --k_values element-wise
    (row i = datasets[i] with k_values[i]). --datasets and --k_values must
    then have the same length.

    python plot_all_datasets_all_k_values.py \
        --pairwise \
        --datasets wildfirejune uber4hmerged \
        --k_values 5 10 \
        --timesteps 50

    This creates rows in this order:
        (wildfirejune, 5)
        (uber4hmerged, 10)

    --global_y_min / --global_y_max also accept either a single value
    (applied to every row) or one value per row, e.g.:
        --global_y_min 0.8 0.9   # row 0 uses 0.8, row 1 uses 0.9
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

B_FRACTIONS: Tuple[float, ...] = (0.0, 0.1, 0.5)

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
METHODS_TO_NAMES = {"form": "FORM", "hist_form": "HIST FORM", "fid": "FID", "hist_fid": "HIST FID"}

DATASET_NAMES = {
    "wildfire": "Wildfire Dataset",
    "uber": "Uber Dataset",
    "twitter": "Twitter Dataset",
    "online_retail": "OnlineRetail",
    "household_power": "Househ. Power",
}

# Layout constants used both for subplots_adjust and for row-title placement.
# Keeping these as named constants (instead of calling subplots_adjust after
# the titles are placed) is what keeps the row titles aligned with their rows.
FIG_TOP = 0.94
FIG_LEFT = 0.12  # reserve horizontal room for the rotated row labels
FIG_HSPACE = 0.55
FIG_WSPACE = 0.12
ROW_TITLE_OFFSET = 0.05  # horizontal gap between a row's left axis and its rotated label


def csv_path_for(results_dir: str, dataset: str, k: int, timesteps: int, heuristic: str, b: float) -> Path:
    filename = f"[{timesteps}]_{dataset}_k{k}_base-{heuristic}_b{b}.csv"
    return Path(results_dir) / dataset / filename


def load_dataframes(
    results_dir: str, dataset: str, k: int, timesteps: int, heuristic: str
) -> Dict[float, pd.DataFrame]:
    dataframes: Dict[float, pd.DataFrame] = {}

    for b in B_FRACTIONS:
        csv_path = csv_path_for(results_dir, dataset, k, timesteps, heuristic, b)

        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV for dataset={dataset}, k={k}, b={b}: {csv_path}")

        df = pd.read_csv(csv_path)

        if df.empty:
            raise ValueError(f"CSV is empty: {csv_path}")

        if "relative_cost_ratio" not in df.columns:
            raise ValueError(f"'relative_cost_ratio' column missing from: {csv_path}")

        dataframes[b] = df

    return dataframes


def compute_shared_y_limits(dataframes: Dict[float, pd.DataFrame]) -> Tuple[float, float]:
    """
    Compute y limits using only:
      - b != 0.0
      - methods 'form' and 'fid'
    """
    values = []

    for b, df in dataframes.items():
        if b == 0.0:
            continue

        filtered = df[df["method"].isin(["form", "fid"])]
        ratios = filtered["relative_cost_ratio"].dropna()

        if not ratios.empty:
            values.append(ratios)

    if not values:
        raise ValueError("No valid relative_cost_ratio values found for b != 0.0 and methods 'form'/'fid'.")

    all_values = pd.concat(values, ignore_index=True)

    y_min = float(all_values.min())
    y_max = float(all_values.max())

    span = y_max - y_min
    padding = 0.05 * span if span > 0 else (0.05 * abs(y_max) if y_max != 0 else 0.05)

    return y_min - padding, y_max + padding


def plot_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    b: float,
    y_min: float,
    y_max: float,
    show_left_axis: bool = False,
    show_right_axis: bool = False,
) -> None:
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

        # BUG FIX: the method's display name must be passed as `label=`, not as
        # a positional argument. Positionally, matplotlib's `Axes.plot` treats
        # the third argument as a style/format string (e.g. "-o", "r--"), so
        # passing "FORM" / "HIST FORM" there is invalid and breaks the legend.
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

    if show_left_axis:
        ax.set_ylabel("Relative Cost Ratio")
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)

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
            label="Number of Datapoints",
            color="gray",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
        )

        if show_right_axis:
            ax_points.set_ylabel("Number of Datapoints")
        else:
            ax_points.tick_params(axis="y", labelright=False)

        ax_points.grid(False)


def add_row_title(fig: plt.Figure, axes: "plt.Axes", row_idx: int, text: str) -> None:
    """
    Place a title to the LEFT of a row of subplots, rotated 90 degrees
    (like a seaborn FacetGrid row label).

    Must be called AFTER fig.subplots_adjust(...) has been applied, since
    ax.get_position() reflects whatever layout is currently active. Calling
    this before subplots_adjust (as the original script did) bakes in stale
    axis positions, which is why the row titles ended up misaligned once the
    final hspace/top adjustment was applied.
    """
    left_bbox = axes[row_idx, 0].get_position()

    # Vertically center the label on the row's own y-span.
    row_title_y = (left_bbox.y0 + left_bbox.y1) / 2

    # Anchor x just to the left of the row's leftmost axis (which already
    # has y-axis tick labels / "Relative cost ratio" on it).
    row_title_x = left_bbox.x0 - ROW_TITLE_OFFSET

    fig.text(
        row_title_x,
        row_title_y,
        text,
        ha="center",
        va="center",
        rotation=90,
        fontsize=14,
        fontweight="bold",
    )


def build_combined_legend(fig: plt.Figure, axes) -> None:
    """
    Collect legend entries from BOTH the primary axis and the twin (num_points)
    axis of the top-left panel, de-duplicating by label. The original script
    only pulled handles from axes[0, 0], which misses the "Number of Datapoints"
    line living on the twinx axis.
    """
    handles, labels = axes[0, 0].get_legend_handles_labels()

    for child in axes[0, 0].figure.axes:
        if child is axes[0, 0]:
            continue
        # twinx axes share the same position as their parent; a cheap way to
        # find "the twin of axes[0,0]" is to check for matching bounding box.
        if child.bbox.bounds == axes[0, 0].bbox.bounds and child is not axes[0, 0]:
            extra_handles, extra_labels = child.get_legend_handles_labels()
            for h, l in zip(extra_handles, extra_labels):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
            break

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.995),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create rows of three side-by-side relative-cost plots for multiple dataset/k combinations."
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset names. In pairwise mode, one per row; otherwise combined with every k value.",
    )
    parser.add_argument(
        "--k_values",
        nargs="+",
        required=True,
        type=int,
        help="k values. In pairwise mode, one per row (must match --datasets length); otherwise combined with every dataset.",
    )
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help=(
            "Pair --datasets and --k_values element-wise instead of taking their cartesian "
            "product: row i uses datasets[i] with k_values[i]. Requires --datasets and "
            "--k_values to have the same length."
        ),
    )
    parser.add_argument("--timesteps", required=True, type=int)
    parser.add_argument("--heuristic", default="kmedianppwpost")
    parser.add_argument("--results_dir", default="results/propagation")
    parser.add_argument("--output_dir", default="plots/paper")
    parser.add_argument(
        "--global_y_min",
        type=float,
        nargs="+",
        default=[1.0],
        help="Either a single value applied to every row, or one value per row (must match the number of rows).",
    )
    parser.add_argument(
        "--global_y_max",
        type=float,
        nargs="+",
        default=[1.0],
        help="Either a single value applied to every row, or one value per row (must match the number of rows).",
    )

    args = parser.parse_args()

    if args.pairwise:
        if len(args.datasets) != len(args.k_values):
            parser.error(
                f"--pairwise requires --datasets and --k_values to have the same length "
                f"(got {len(args.datasets)} datasets and {len(args.k_values)} k values)."
            )
        # Element-wise pairing: row i is (datasets[i], k_values[i]).
        combinations: List[Tuple[str, int]] = list(zip(args.datasets, args.k_values))
    else:
        # Cartesian product in dataset-first order:
        # all k values for the first dataset, then all k values for the next dataset.
        combinations = [(dataset, k) for dataset in args.datasets for k in args.k_values]

    n_rows = len(combinations)

    def _resolve_per_row(values: List[float], name: str) -> List[float]:
        """Broadcast a single value to every row, or validate a per-row list."""
        if len(values) == 1:
            return values * n_rows
        if len(values) != n_rows:
            parser.error(
                f"--{name} must be either a single value or provide exactly one value per "
                f"row ({n_rows} rows), got {len(values)} values."
            )
        return values

    global_y_min_per_row = _resolve_per_row(args.global_y_min, "global_y_min")
    global_y_max_per_row = _resolve_per_row(args.global_y_max, "global_y_max")

    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13})

    # Load all data first, so a missing/broken CSV fails fast before any plotting work.
    all_data = {
        (dataset, k): load_dataframes(args.results_dir, dataset, k, args.timesteps, args.heuristic)
        for dataset, k in combinations
    }

    # One row per (dataset, k), three columns for b values.
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 3 * n_rows), squeeze=False)

    for row_idx, (dataset, k) in enumerate(combinations):
        dataframes = all_data[(dataset, k)]

        # Each row gets its own y-range.
        y_min, y_max = compute_shared_y_limits(dataframes)
        y_min = min(y_min, global_y_min_per_row[row_idx])
        y_max = max(y_max, global_y_max_per_row[row_idx])


        for col_idx, b in enumerate(B_FRACTIONS):
            plot_panel(
                ax=axes[row_idx, col_idx],
                df=dataframes[b],
                b=b,
                y_min=y_min,
                y_max=y_max,
                show_left_axis=(col_idx == 0),
                show_right_axis=(col_idx == 2),
            )

    # Apply the final layout BEFORE computing any row-title positions, so
    # get_position() reflects where the rows actually end up.
    plt.subplots_adjust(top=FIG_TOP, left=FIG_LEFT, hspace=FIG_HSPACE, wspace=FIG_WSPACE)

    for row_idx, (dataset, k) in enumerate(combinations):
        dataset_title = DATASET_NAMES.get(dataset, dataset)
        add_row_title(fig, axes, row_idx, f"{dataset_title}, k = {k}")

    build_combined_legend(fig, axes)

    os.makedirs(args.output_dir, exist_ok=True)

    combination_suffix = "_".join(f"{dataset}-k{k}" for dataset, k in combinations)
    output_path = Path(args.output_dir) / f"combined_{combination_suffix}_t{args.timesteps}.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"Plot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()