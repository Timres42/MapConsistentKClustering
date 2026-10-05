"""Compare the runtimes of FORM, FID, OPT FID and OPT MC.

Reads the results CSVs in <results_dir>/<dataset>/ and uses the `execution_time` column of the
`form`, `fid`, `opt_fid` and `opt_mc` rows (seconds per timestep). Older results call opt_fid
just `opt`. The exact k-median row of timestep 1 (opt_kmedian) is a different problem and is
left out; all methods are compared on timesteps >= 2.

Terminal output, per dataset and k, one row per b:
  median / mean / max runtime per method, how many times slower each exact algorithm is than
  the heuristic for the same problem (OPT FID vs FID, OPT MC vs FORM; medians), and how many
  exact solves hit the Gurobi time limit (not proven optimal - their runtime is the limit).

Figures (<output_dir>/runtime_<dataset>.png): one panel per k, median runtime over b on a
log scale with the 25-75 % range shaded. Hollow markers: at least one exact solve timed out.
With --per_timestep, additionally one figure per CSV with the runtime of every timestep.

Usage:
    python plots/scripts/compare_runtimes.py                                  # results/propagation_opt
    python plots/scripts/compare_runtimes.py --datasets uber --k 5 10
    python plots/scripts/compare_runtimes.py --csv results/propagation_opt/uber/*.csv --per_timestep
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METHODS = ["form", "fid", "opt_fid", "opt_mc"]
OPT_METHODS = ["opt_fid", "opt_mc"]
NAMES = {"form": "FORM", "fid": "FID", "opt_fid": "OPT FID", "opt_mc": "OPT MC"}
# Same colors as plots_propagation_ratio_bviolin.py
COLORS = {"form": "#1f77b4", "fid": "#ff7f0e", "opt_fid": "#2ca02c", "opt_mc": "#d62728"}
MARKERS = {"form": "o", "fid": "s", "opt_fid": "D", "opt_mc": "v"}
DATASET_NAMES = {
    "wildfire": "Wildfire Dataset",
    "uber": "Uber Dataset",
    "twitter": "Twitter Dataset",
    "online_retail": "Online Retail Dataset",
    "household_power": "Household Power Dataset",
}


def load(paths):
    frames = []
    for path in paths:
        df = pd.read_csv(path, na_values=["nan"])
        if "execution_time" not in df.columns:
            print(f"  [skipped] {path}: no execution_time column")
            continue
        df["method"] = df["method"].replace({"opt": "opt_fid"})   # older results
        df = df[df["method"].isin(METHODS) & (df["timestep"] >= 2)].copy()
        if "opt_optimal" not in df.columns:
            df["opt_optimal"] = np.nan
        df["source"] = path
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fmt_time(seconds: float) -> str:
    if pd.isna(seconds):
        return "-"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    if seconds < 120:
        return f"{seconds:.1f} s"
    return f"{seconds / 60:.1f} min"


def print_tables(data: pd.DataFrame, datasets, k_values):
    col = 30
    present = [m for m in METHODS if (data["method"] == m).any()]
    opts = [m for m in OPT_METHODS if m in present]
    for dataset in datasets:
        sub_d = data[data["dataset"] == dataset]
        print("\n" + "=" * 132)
        print(f"Dataset: {dataset}   (runtime per timestep: median / mean / max)")
        print("=" * 132)
        for k in k_values:
            sub_k = sub_d[sub_d["k"] == k]
            print(f"\nk = {k}")
            if sub_k.empty:
                print("  no results")
                continue
            ratio_hdr = {"opt_fid": "OPTFID/FID", "opt_mc": "OPTMC/FORM"}
            header = f"  {'b':<6} {'T':>3} | " + " | ".join(f"{NAMES[m]:^{col}}" for m in present) \
                     + "".join(f" | {ratio_hdr[m]:>11}" for m in opts) \
                     + "".join(f" | {NAMES[m] + ' timed out':>17}" for m in opts)
            print(header)
            for b in sorted(sub_k["b_fraction"].unique()):
                rows = sub_k[sub_k["b_fraction"] == b]
                cells, med = [], {}
                for m in present:
                    t = rows.loc[rows["method"] == m, "execution_time"].dropna()
                    med[m] = t.median() if len(t) else np.nan
                    cells.append(f"{fmt_time(t.median()) if len(t) else '-':>9} / "
                                 f"{fmt_time(t.mean()) if len(t) else '-':>8} / "
                                 f"{fmt_time(t.max()) if len(t) else '-':>8}")
                def ratio(a, c):
                    if not (a > 0 and c > 0):
                        return f"{'-':>11}"
                    r = a / c
                    return f"{r:>10,.0f}x" if r >= 10 else f"{r:>10.2f}x"
                heur_of = {"opt_fid": "fid", "opt_mc": "form"}
                ratios = "".join(f" | {ratio(med.get(m, np.nan), med.get(heur_of[m], np.nan))}" for m in opts)
                timed = ""
                for m in opts:
                    r_ = rows[rows["method"] == m]
                    t_ = f"{int((r_['opt_optimal'] == 0).sum())}/{len(r_)}" if len(r_) else "-"
                    timed += f" | {t_:>17}"
                T = rows.loc[rows["method"] == "form", "timestep"].nunique()
                print(f"  {b:<6g} {T:>3} | " + " | ".join(f"{c:^{col}}" for c in cells) + ratios + timed)


def plot_overview(data: pd.DataFrame, dataset: str, k_values, out_dir: str):
    import matplotlib.pyplot as plt

    sub_d = data[data["dataset"] == dataset]
    ks = [k for k in k_values if (sub_d["k"] == k).any()]
    if not ks:
        return
    fig, axes = plt.subplots(1, len(ks), figsize=(5.5 * len(ks), 4.2), sharey=True, squeeze=False)
    for ax, k in zip(axes[0], ks):
        sub_k = sub_d[sub_d["k"] == k]
        for m in METHODS:
            rows = sub_k[sub_k["method"] == m].dropna(subset=["execution_time"])
            if rows.empty:
                continue
            g = rows.groupby("b_fraction")["execution_time"]
            stats = pd.DataFrame({"med": g.median(), "q1": g.quantile(0.25), "q3": g.quantile(0.75)})
            stats = stats[stats["med"] > 0]
            if stats.empty:
                continue
            ax.fill_between(stats.index, stats["q1"].clip(lower=1e-7), stats["q3"], color=COLORS[m], alpha=0.15, lw=0)
            ax.plot(stats.index, stats["med"], color=COLORS[m], lw=2, marker=MARKERS[m], ms=7, label=NAMES[m])
            if m in OPT_METHODS:
                timed = rows.groupby("b_fraction")["opt_optimal"].apply(lambda s: (s == 0).any())
                tb = [b for b in stats.index if timed.get(b, False)]
                if tb:
                    ax.scatter(tb, stats.loc[tb, "med"], s=60, marker=MARKERS[m], facecolors="white",
                               edgecolors=COLORS[m], linewidths=1.5, zorder=3,
                               label=f"{NAMES[m]}: some solves timed out")
        ax.set_yscale("log")
        ax.set_title(f"k = {k}")
        ax.set_xlabel("budget fraction b")
        ax.grid(True, which="major", alpha=0.3)
    axes[0][0].set_ylabel("runtime per timestep [s] (median, 25-75 %)")
    handles = {}
    for ax in axes[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    axes[0][-1].legend(handles.values(), handles.keys(), loc="best", fontsize=9)
    fig.suptitle(f"Runtime of the algorithms - {DATASET_NAMES.get(dataset, dataset)}", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, f"runtime_{dataset}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure: {path}")


def plot_per_timestep(data: pd.DataFrame, out_dir: str):
    import matplotlib.pyplot as plt

    for source, rows in data.groupby("source"):
        fig, ax = plt.subplots(figsize=(7, 4))
        for m in METHODS:
            r = rows[rows["method"] == m].sort_values("timestep")
            r = r[r["execution_time"] > 0]
            if r.empty:
                continue
            ax.plot(r["timestep"], r["execution_time"], color=COLORS[m], lw=2, marker=MARKERS[m], ms=6, label=NAMES[m])
            if m in OPT_METHODS:
                t = r[r["opt_optimal"] == 0]
                if len(t):
                    ax.scatter(t["timestep"], t["execution_time"], s=50, marker=MARKERS[m], facecolors="white",
                               edgecolors=COLORS[m], linewidths=1.5, zorder=3, label=f"{NAMES[m]} timed out")
        first = rows.iloc[0]
        ax.set_yscale("log")
        ax.set_xlabel("timestep")
        ax.set_ylabel("runtime [s]")
        ax.set_title(f"{DATASET_NAMES.get(first['dataset'], first['dataset'])}, k = {first['k']}, "
                     f"b = {first['b_fraction']:g}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        stem = os.path.splitext(os.path.basename(source))[0]
        path = os.path.join(out_dir, f"runtime_{stem}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results_dir", default=os.path.join(REPO_ROOT, "results", "propagation_opt"),
                        help="Directory with one subdirectory per dataset")
    parser.add_argument("--datasets", nargs="*", default=None, help="Datasets (default: every subdirectory)")
    parser.add_argument("--k", nargs="*", type=int, default=[5, 10, 15, 20], help="k values to report")
    parser.add_argument("--csv", nargs="*", default=None, help="Explicit CSV files instead of --results_dir")
    parser.add_argument("--output_dir", default=os.path.join(REPO_ROOT, "plots", "runtime"))
    parser.add_argument("--per_timestep", action="store_true", help="Also plot every CSV's runtime per timestep")
    parser.add_argument("--no_plot", action="store_true", help="Only print the tables")
    args = parser.parse_args()

    if args.csv:
        paths = args.csv
    else:
        datasets = args.datasets or sorted(d for d in os.listdir(args.results_dir)
                                           if os.path.isdir(os.path.join(args.results_dir, d)))
        paths = sorted(p for d in datasets for p in glob.glob(os.path.join(args.results_dir, d, "*.csv")))
    data = load(paths)
    if data.empty:
        print("No results found.")
        return
    if not data["method"].isin(OPT_METHODS).any():
        print("Note: no opt_fid / opt_mc rows found (runs without --opt); comparing FORM and FID only.")
    datasets = args.datasets or sorted(data["dataset"].unique())
    k_values = args.k if not args.csv else sorted(data["k"].unique())

    multi = data.groupby(["dataset", "k", "b_fraction"])["source"].nunique()
    for (ds, k, b), cnt in multi[multi > 1].items():
        print(f"  [note] {ds} k={k} b={b:g}: timesteps pooled from {cnt} files")

    print_tables(data, datasets, k_values)

    if not args.no_plot:
        os.makedirs(args.output_dir, exist_ok=True)
        for dataset in datasets:
            plot_overview(data, dataset, k_values, args.output_dir)
        if args.per_timestep:
            plot_per_timestep(data, args.output_dir)


if __name__ == "__main__":
    main()