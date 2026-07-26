"""Visualization of center-propagation drift experiment results.

Plots relative cost ratios (normalized to unconstrained heuristic) across timesteps
for label-consistent clustering algorithms and baselines in temporal drift scenarios.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


PALETTE = {
    "form": "#1f77b4",
    "hist_form": "#7fb3d9",
    "fid": "#ff7f0e",
    "hist_fid": "#ffbb78",
}

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13})


def extract_b_value(df: pd.DataFrame, csv_path: str) -> str:
    """
    Determine the budget fraction (b) used for this experiment run.
    
    Preference order:
    1. The 'b_fraction' column in the CSV, if present and non-empty.
    2. Parsed out of the filename (handles both 'b0.2' and 'b0_2' styles,
       e.g. '[30]_wildfire_k5_b0.2.csv').
    3. '?' if neither source yields a value.
    
    :param df: Loaded results dataframe
    :param csv_path: Path to the CSV file (used for filename fallback)
    :return: String representation of b, or '?' if it can't be determined
    """
    if 'b_fraction' in df.columns:
        valid = df['b_fraction'].dropna()
        if not valid.empty:
            return str(valid.iloc[0])
    
    basename = os.path.basename(csv_path)
    match = re.search(r'b(\d+(?:[._]\d+)?)', basename)
    if match:
        return match.group(1).replace('_', '.')
    
    return '?'

def extract_heuristic_name(df: pd.DataFrame, csv_path: str) -> str:
    """
    Determine the heuristic name used for this experiment run.
    
    Preference order:
    1. The 'baseline/sigma used' column in the CSV, if present and non-empty.
    2. Parsed out of the filename (handles both 'base-heuristic' and 'base_heuristic' styles,
       e.g. '[30]_wildfire_k5_b0.2_base-local_search.csv').
    3. '?' if neither source yields a value.
    
    :param df: Loaded results dataframe
    :param csv_path: Path to the CSV file (used for filename fallback)
    :return: String representation of heuristic name, or '?' if it can't be determined
    """
    if 'baseline/sigma used' in df.columns:
        valid = df['baseline/sigma used'].dropna()
        if not valid.empty:
            return str(valid.iloc[0])
    
    basename = os.path.basename(csv_path)
    match = re.search(r'base[-_](\w+)', basename)
    if match:
        return match.group(1)
    
    return '?'


def plot_relative_costs(csv_path: str, output_dir: str = "plots", 
                        y_min: float = None, y_max: float = None):
    """
    Plot relative cost ratios from drift propagation experiment.
    
    :param csv_path: Path to CSV results file
    :param output_dir: Directory to save plots
    :param y_min: Optional lower bound for the y-axis (both subplots). If
                  None, falls back to the previous default (0.8).
    :param y_max: Optional upper bound for the y-axis (both subplots). If
                  None, falls back to the previous default
                  (max(2.0, max relative_cost_ratio * 1.1)).
    """
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return
    
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"Error: {csv_path} is empty.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metadata from filename or dataframe
    dataset = df['dataset'].iloc[0] if 'dataset' in df.columns else 'unknown'
    k = df['k'].iloc[0] if 'k' in df.columns else 3
    b_val = extract_b_value(df, csv_path)
    heuristic = extract_heuristic_name(df, csv_path)
    
    # Define color palette and markers.
    # hist_us/hist_fid share the hue of us/fid respectively (lighter shade) so
    # it reads as "same algorithm, before vs. after re-optimization".
    palette = {
        "form": "#1f77b4",                   # Blue
        "hist_form": "#7fb3d9",              # Light blue
        "fid": "#ff7f0e",                  # Orange
        "hist_fid": "#ffbb78",             # Light orange
    }
    
    markers = {
        "baseline_heuristic": "^",
        "form": "o",
        "hist_form": "o",
        "fid": "s",
        "hist_fid": "s",
    }
    
    linestyles = {
        "baseline_heuristic": "-",
        "form": "-",
        "hist_form": "--",
        "fid": "-",
        "hist_fid": "--",
    }
    dataset_names = {
        "wildfire": "Wildfire Dataset",
        "uber": "Uber Dataset",
        "twitter": "Twitter Dataset",
        "online_retail": "Online Retail Dataset",
        "household_power": "Household Power Dataset",
    }
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 4))
    #Overall title for the figure
    fig.suptitle(f"{dataset_names[dataset]}, k = {k}, b = {b_val}", fontsize=14)
    
    # ============ Plot 1: Relative Cost Ratio over Timesteps ============
    methods_to_plot = ["form", "hist_form", "fid", "hist_fid"]
    methods_to_names={"form":"FORM", "hist_form":"HIST FORM", "fid":"FID", "hist_fid":"HIST FID"}
    non_hist_methods_to_plot = [method for method in methods_to_plot if not method.startswith("hist_")]
    
    ax1.axhline(y=1.0, color='black', linestyle=':', linewidth=1.5, alpha=0.7, label='Baseline (ratio=1)')
    for method in methods_to_plot:
        method_data = df[df['method'] == method].sort_values('timestep')
        if not method_data.empty and 'relative_cost_ratio' in df.columns:
            ax1.plot(method_data['timestep'], 
                    method_data['relative_cost_ratio'],
                    #marker=markers.get(method, 'o'),
                    label=methods_to_names[method],
                    color=palette.get(method, 'gray'),
                    linestyle=linestyles.get(method, '-'),
                    linewidth=2,
                    #markersize=8
                    )
    
    ax1.set_xlabel('Timestep', fontsize=12)
    ax1.set_ylabel('Relative Cost Ratio', fontsize=12)
    ax1.set_title(f'Cost Ratios over Time / Number of Datapoints', fontsize=13)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # y-axis bounds: use the user-supplied value if given, else the previous default
    effective_y_min = y_min if y_min is not None else 0.8
    effective_y_max = y_max if y_max is not None else max(2.0, df['relative_cost_ratio'].max() * 1.1)
    ax1.set_ylim([effective_y_min, effective_y_max])

    # Add number of points connected via line and with scale on the right y-axis
    # Compute the number of points for each timestep (they should be the same across methods, but we take the first available method)
    num_points = None
    for method in non_hist_methods_to_plot:
        method_data = df[df['method'] == method].sort_values('timestep')
        if not method_data.empty and 'num_points' in df.columns:
            num_points = method_data['num_points']
            break
    ax1_2 = ax1.twinx()
    ax1_2.plot(method_data['timestep'], num_points, color=palette.get(method, 'gray'), linestyle=':', linewidth=1.5, alpha=0.7, label=f'Num points ({method})')
    ax1_2.set_ylabel('Number of Points', fontsize=12)
    ax1_2.grid(False)
    
    # ============ Plot 2: Distribution of Relative Costs ============
    if 'relative_cost_ratio' in df.columns:
        method_ratios = df.groupby('method')['relative_cost_ratio'].apply(list).to_dict()
        
        box_data = []
        box_labels = []
        box_colors = []
        
        for method in methods_to_plot:
            if method in method_ratios:
                ratios = [r for r in method_ratios[method] if pd.notna(r) and np.isfinite(r)]
                if ratios:
                    box_data.append(ratios)
                    box_labels.append(methods_to_names[method])
                    box_colors.append(palette.get(method, 'gray'))
        
        # NOTE: matplotlib >= 3.9 renamed the `labels` kwarg to `tick_labels`
        # (and it was removed entirely as of 3.10, which is what's installed here).
        bp = ax2.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.set_ylabel('Relative Cost Ratio', fontsize=12)
        ax2.set_title(f'Distribution of Cost Ratios', fontsize=13)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xticklabels(box_labels, rotation=15, ha='right')
        
        # Only override the box plot's autoscaled y-axis if the user explicitly
        # asked for bounds; otherwise leave it auto-scaled to the box data.
        if y_min is not None or y_max is not None:
            box_ymin, box_ymax = ax2.get_ylim()
            ax2.set_ylim([
                y_min if y_min is not None else box_ymin,
                y_max if y_max is not None else box_ymax
            ])

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

    for method in non_hist_methods_to_plot:
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
                violin_labels.append(methods_to_names[method])
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

    ax3.set_title(f"Distribution of Budget Used (max possible = {b_val})")
    ax3.set_ylabel("Budget Used (%)")
    ax3.grid(True, alpha=0.3, axis="y")
           

    
    plt.tight_layout()
    
    csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
    output_path = os.path.join(output_dir, f"plot_{csv_basename}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()
    
    # ============ Summary Statistics ============
    print("\n" + "="*70)
    print("RELATIVE COST RATIO SUMMARY STATISTICS")
    print("="*70)
    
    for method in methods_to_plot:
        method_data = df[df['method'] == method]['relative_cost_ratio']
        if not method_data.empty:
            valid_ratios = method_data.dropna()
            if len(valid_ratios) > 0:
                print(f"\n{method}:")
                print(f"  Mean relative cost:  {valid_ratios.mean():.4f}")
                print(f"  Std dev:             {valid_ratios.std():.4f}")
                print(f"  Min:                 {valid_ratios.min():.4f}")
                print(f"  Max:                 {valid_ratios.max():.4f}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot relative cost ratios from drift propagation experiments.")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV results file")
    parser.add_argument("--output_dir", type=str, default="plots/out", help="Output directory for plots")
    parser.add_argument("--y_min", type=float, default=None, help="Lower bound for the y-axis (both subplots)")
    parser.add_argument("--y_max", type=float, default=None, help="Upper bound for the y-axis (both subplots)")
    args = parser.parse_args()
    
    plot_relative_costs(args.csv, args.output_dir, y_min=args.y_min, y_max=args.y_max)