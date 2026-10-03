import os
from pathlib import Path

from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("results/propagation")    # folder containing the CSVs
OUT_DIR = "plots/propagation"
DATASETS = ["twitter", "online_retail", "household_power", "wildfire", "uber"]
K_VALUES = [5, 10, 15, 20]
B_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
HEURISTIC = "kmedianppwpost"
TIMESTEPS = 30
METHOD = "form"


def load_runtimes(ds, k, b):
    """Return the per-timestep runtimes of METHOD, or None if the file is missing."""
    csv_path = (
        RESULTS_DIR
        / ds
        / f"[{TIMESTEPS}]_{ds}_k{k}_kernels4_base-{HEURISTIC}_b{b}.csv"
    )
    if not csv_path.exists():
        print(f"missing: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    return df.loc[df["method"] == METHOD, "execution_time"].to_numpy()


for ds in DATASETS:
    out_dir = os.path.join(OUT_DIR, ds)
    os.makedirs(out_dir, exist_ok=True)
    runtimes = {(k, b): load_runtimes(ds, k, b) for k in K_VALUES for b in B_FRACTIONS}

    width = 0.8 / len(B_FRACTIONS)
    x = np.arange(len(K_VALUES))
    colors = plt.cm.viridis(np.linspace(0, 1, len(B_FRACTIONS)))

    # --- Plot 1: total runtime, grouped bars (x = k, bars = b) ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, b in enumerate(B_FRACTIONS):
        totals = [runtimes[k, b].sum() if runtimes[k, b] is not None else np.nan
                  for k in K_VALUES]
        ax.bar(x + (i - (len(B_FRACTIONS) - 1) / 2) * width, totals, width,
               label=f"b = {b}", color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels([f"k = {k}" for k in K_VALUES])
    ax.set_ylabel(f"Total runtime of '{METHOD}' over {TIMESTEPS} timesteps [s]")
    ax.set_title(f"{ds}: runtime of {METHOD}")
    ax.legend(title="b fraction")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"runtime_{METHOD}_{ds}.png"), dpi=150)
    plt.close(fig)

    # --- Plot 2: box plot of per-timestep runtimes, grouped (x = k, boxes = b) ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, b in enumerate(B_FRACTIONS):
        data, pos = [], []
        for j, k in enumerate(K_VALUES):
            t = runtimes[k, b]
            if t is not None and len(t) > 0:
                data.append(t)
                pos.append(x[j] + (i - (len(B_FRACTIONS) - 1) / 2) * width)
        if not data:
            continue
        ax.boxplot(data, positions=pos, widths=width * 0.85, patch_artist=True,
                   boxprops=dict(facecolor=colors[i], alpha=0.8),
                   medianprops=dict(color="black"),
                   flierprops=dict(marker="o", markersize=3, alpha=0.6))
        ax.plot([], [], "s", color=colors[i], label=f"b = {b}")  # legend handle
    ax.set_xticks(x)
    ax.set_xticklabels([f"k = {k}" for k in K_VALUES])
    ax.set_xlim(-0.5, len(K_VALUES) - 0.5)
    ax.set_ylabel(f"Runtime of '{METHOD}' per timestep [s]")
    ax.set_title(f"{ds}: distribution of {METHOD} runtimes per timestep")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(title="b fraction")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"runtime_box_{METHOD}_{ds}.png"), dpi=150)
    plt.close(fig)