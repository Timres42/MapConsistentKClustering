"""Evaluate the seed study (src/seed_study.py): variability caused by the k-median++ randomness.

Unit of analysis: one run (= one seed). For every run, metrics are averaged over timesteps
2..T (timestep 1 only has the baseline):
  ratio_<m>   mean relative_cost_ratio of method m (form, fid, hist_form, hist_fid)
  budget_<m>  mean fraction of points counted against the budget (b_abs_used / num_points)
  cost_<m>    mean absolute cost; baseline_cost is the mean baseline (k-median++) cost
  time_<m>    total runtime in seconds (form, fid, baseline)

Across the seeds of one configuration (dataset, k, b) it reports mean, sample standard
deviation, 95 % confidence interval of the mean (t-distribution, n-1 df), median, IQR,
min/max and coefficient of variation; and a paired comparison FORM vs FID per seed
(both see the same baseline in a run): mean difference with 95 % CI, number of seeds
where FORM is better / equal / worse, and the Wilcoxon signed-rank test.
Per timestep it also reports the coefficient of variation of the baseline cost across seeds.

Outputs (default plots/seeds/):
  seeds_runs_<dataset>.csv        one row per run (per-run metrics)
  seeds_summary_<dataset>.csv     statistics per (k, b, metric)
  seeds_paired_<dataset>.csv      FORM vs FID per (k, b)
  seeds_timesteps_<dataset>.csv   mean / sd across seeds per (k, b, timestep)
  seeds_table_<dataset>.tex       LaTeX table: mean +- sd of FORM / FID ratios
  seeds_<dataset>_k<k>.png        ratio and paired difference over b (every seed as a dot)
  seeds_<dataset>_k<k>_timesteps.png   mean +- sd band per timestep, one panel per b

Usage:
    python plots/scripts/evaluate_seeds.py
    python plots/scripts/evaluate_seeds.py --dataset wildfire --results_dir results/seeds --no_plot
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METHODS = ["form", "fid", "hist_form", "hist_fid"]
NAMES = {"form": "FORM", "fid": "FID", "hist_form": "HIST FORM", "hist_fid": "HIST FID"}
COLORS = {"form": "#1f77b4", "fid": "#ff7f0e", "hist_form": "#7fb3d9", "hist_fid": "#ffbb78"}  # as in the other plots
DIFF_COLOR = "#52514e"


# ---------------------------------------------------------------- loading
def load_runs(paths):
    rows, ts_rows = [], []
    for path in paths:
        df = pd.read_csv(path, na_values=["nan"])
        if "seed" in df.columns and df["seed"].notna().any():
            seed = int(df["seed"].dropna().iloc[0])
        else:
            m = re.search(r"_s(\d+)\.csv$", path)
            if not m:
                print(f"  [skipped] {path}: no seed")
                continue
            seed = int(m.group(1))
        k = int(df["k"].iloc[0])
        b = float(df["b_fraction"].iloc[0])
        T = int(df["timestep"].max())
        d = df[df["timestep"] >= 2]
        run = {"dataset": df["dataset"].iloc[0], "k": k, "b": b, "seed": seed, "timesteps": T}
        for m in METHODS:
            r = d[d["method"] == m]
            run[f"ratio_{m}"] = r["relative_cost_ratio"].mean()
            run[f"cost_{m}"] = r["result_cost"].mean()
            if m in ("form", "fid"):
                run[f"budget_{m}"] = (r["b_abs_used"] / r["num_points"]).mean()
                run[f"time_{m}"] = r["execution_time"].sum()
        base = df[df["method"] == "baseline_heuristic"]
        run["baseline_cost"] = base.loc[base["timestep"] >= 2, "result_cost"].mean()
        run["time_baseline"] = base["execution_time"].sum()
        rows.append(run)
        # per-timestep values for the bands / baseline CV
        piv = df[df["method"].isin(["form", "fid", "baseline_heuristic"])].pivot_table(
            index="timestep", columns="method", values=["relative_cost_ratio", "result_cost"], aggfunc="first")
        for t, r in piv.iterrows():
            ts_rows.append({"k": k, "b": b, "seed": seed, "timestep": t,
                            "ratio_form": r.get(("relative_cost_ratio", "form"), np.nan),
                            "ratio_fid": r.get(("relative_cost_ratio", "fid"), np.nan),
                            "baseline_cost": r.get(("result_cost", "baseline_heuristic"), np.nan)})
    return pd.DataFrame(rows), pd.DataFrame(ts_rows)


# ---------------------------------------------------------------- statistics
def describe(x: pd.Series) -> dict:
    x = x.dropna()
    n = len(x)
    mean = x.mean() if n else np.nan
    sd = x.std(ddof=1) if n > 1 else np.nan
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
    return {"n": n, "mean": mean, "sd": sd, "ci95_lo": mean - half, "ci95_hi": mean + half,
            "median": x.median() if n else np.nan, "q1": x.quantile(0.25) if n else np.nan,
            "q3": x.quantile(0.75) if n else np.nan, "min": x.min() if n else np.nan,
            "max": x.max() if n else np.nan, "cv": sd / abs(mean) if n > 1 and mean else np.nan}


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in runs.columns if c.split("_")[0] in ("ratio", "budget", "cost", "time") or c == "baseline_cost"]
    out = []
    for (k, b), g in runs.groupby(["k", "b"]):
        for m in metrics:
            out.append({"k": k, "b": b, "metric": m, **describe(g[m])})
    return pd.DataFrame(out)


def paired(runs: pd.DataFrame, a="ratio_form", c="ratio_fid") -> pd.DataFrame:
    out = []
    for (k, b), g in runs.groupby(["k", "b"]):
        g = g.dropna(subset=[a, c])
        diff = g[a] - g[c]
        d = describe(diff)
        tol = 1e-12 * max(1.0, float(g[[a, c]].abs().max().max()) if len(g) else 1.0)
        better = int((diff < -tol).sum()); worse = int((diff > tol).sum()); equal = len(diff) - better - worse
        p = np.nan
        if (diff.abs() > tol).sum() >= 1:
            try:
                p = stats.wilcoxon(g[a], g[c], zero_method="wilcox").pvalue
            except ValueError:
                p = np.nan
        out.append({"k": k, "b": b, "n": d["n"], "mean_diff": d["mean"], "sd_diff": d["sd"],
                    "ci95_lo": d["ci95_lo"], "ci95_hi": d["ci95_hi"],
                    "form_better": better, "equal": equal, "fid_better": worse, "wilcoxon_p": p})
    return pd.DataFrame(out)


def per_timestep(ts: pd.DataFrame) -> pd.DataFrame:
    g = ts.groupby(["k", "b", "timestep"])
    out = g.agg(n=("seed", "nunique"),
                ratio_form_mean=("ratio_form", "mean"), ratio_form_sd=("ratio_form", "std"),
                ratio_fid_mean=("ratio_fid", "mean"), ratio_fid_sd=("ratio_fid", "std"),
                baseline_mean=("baseline_cost", "mean"), baseline_sd=("baseline_cost", "std")).reset_index()
    out["baseline_cv"] = out["baseline_sd"] / out["baseline_mean"]
    return out


# ---------------------------------------------------------------- output
def fmt_ms(mean, sd, digits=4):
    if pd.isna(mean):
        return "-"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}" if pd.notna(sd) else f"{mean:.{digits}f}"


def print_tables(summary, pair, tsstats, dataset):
    s = summary.set_index(["k", "b", "metric"])
    for k in sorted(summary["k"].unique()):
        print("\n" + "=" * 150)
        n = int(s.loc[(k, slice(None), "ratio_form"), "n"].max())
        print(f"Dataset: {dataset}, k = {k}   ({n} seeds per b; per-run mean over timesteps 2..T; mean ± sd [95% CI])")
        print("=" * 150)
        print(f"  {'b':<5} | {'FORM ratio':^34} | {'FID ratio':^34} | {'FORM - FID':^27} | {'FORM/=/FID':^11} | "
              f"{'Wilcoxon p':>10} | {'baseline CV':>11}")
        for b in sorted(summary.loc[summary["k"] == k, "b"].unique()):
            cells = []
            for m in ("ratio_form", "ratio_fid"):
                r = s.loc[(k, b, m)]
                cells.append(f"{fmt_ms(r['mean'], r['sd'])} [{r['ci95_lo']:.4f}, {r['ci95_hi']:.4f}]"
                             if pd.notna(r["sd"]) else fmt_ms(r["mean"], r["sd"]))
            p = pair[(pair["k"] == k) & (pair["b"] == b)].iloc[0]
            diff = (f"{p['mean_diff']:+.4f} [{p['ci95_lo']:+.4f}, {p['ci95_hi']:+.4f}]"
                    if pd.notna(p["ci95_lo"]) else f"{p['mean_diff']:+.4f}")
            wins = f"{int(p['form_better'])}/{int(p['equal'])}/{int(p['fid_better'])}"
            pv = f"{p['wilcoxon_p']:.2g}" if pd.notna(p["wilcoxon_p"]) else "-"
            cv = tsstats[(tsstats["k"] == k) & (tsstats["b"] == b)]["baseline_cv"].mean()
            print(f"  {b:<5g} | {cells[0]:^34} | {cells[1]:^34} | {diff:^27} | {wins:^11} | {pv:>10} | {cv:>10.2%}")
    print("\n  FORM/=/FID: number of seeds where FORM's mean ratio is lower / equal / higher than FID's.")
    print("  baseline CV: coefficient of variation of the baseline cost across seeds, averaged over timesteps.")


def write_latex(summary, pair, path):
    s = summary.set_index(["k", "b", "metric"])
    lines = [r"\begin{tabular}{rrcccr}", r"\toprule",
             r"$k$ & $b$ & FORM & FID & FORM $-$ FID [95\% CI] & FORM better \\", r"\midrule"]
    for k in sorted(summary["k"].unique()):
        for b in sorted(summary.loc[summary["k"] == k, "b"].unique()):
            f, d = s.loc[(k, b, "ratio_form")], s.loc[(k, b, "ratio_fid")]
            p = pair[(pair["k"] == k) & (pair["b"] == b)].iloc[0]
            ci = (f"[{p['ci95_lo']:+.3f}, {p['ci95_hi']:+.3f}]" if pd.notna(p["ci95_lo"]) else "")
            lines.append(f"{k} & {b:g} & ${f['mean']:.3f} \\pm {f['sd']:.3f}$ & ${d['mean']:.3f} \\pm {d['sd']:.3f}$"
                         f" & ${p['mean_diff']:+.3f}$ {ci} & {int(p['form_better'])}/{int(p['n'])} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def plot_overview(runs, pair, dataset, k, out_dir):
    import matplotlib.pyplot as plt

    g = runs[runs["k"] == k]
    bs = sorted(g["b"].unique())
    step = min(np.diff(bs)) if len(bs) > 1 else 0.1
    off = 0.18 * step
    rng = np.random.default_rng(0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    for m, dx in (("form", -off), ("fid", off)):
        col = f"ratio_{m}"
        for b in bs:
            v = g.loc[g["b"] == b, col].dropna()
            ax1.scatter(b + dx + rng.uniform(-0.25, 0.25, len(v)) * off, v, s=10, color=COLORS[m],
                        alpha=0.35, linewidths=0)
        st = g.groupby("b")[col].apply(lambda x: pd.Series(describe(x))).unstack()
        ax1.errorbar(st.index + dx, st["mean"], yerr=[st["mean"] - st["ci95_lo"], st["ci95_hi"] - st["mean"]],
                     fmt="o-", color=COLORS[m], lw=2, ms=6, capsize=3, label=f"{NAMES[m]} (mean, 95% CI)")
    ax1.axhline(1.0, color="black", ls=":", lw=1.2, alpha=0.7)
    ax1.set_xlabel("budget fraction b"); ax1.set_ylabel("relative cost ratio (mean over timesteps)")
    ax1.set_title(f"Per-run mean ratio, one dot per seed (n = {g['seed'].nunique()})", fontsize=11)
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    p = pair[pair["k"] == k].sort_values("b")
    diffs = g.assign(diff=g["ratio_form"] - g["ratio_fid"])
    for b in bs:
        v = diffs.loc[diffs["b"] == b, "diff"].dropna()
        ax2.scatter(b + rng.uniform(-0.25, 0.25, len(v)) * step, v, s=10, color=DIFF_COLOR, alpha=0.35, linewidths=0)
    ax2.errorbar(p["b"], p["mean_diff"], yerr=[p["mean_diff"] - p["ci95_lo"], p["ci95_hi"] - p["mean_diff"]],
                 fmt="o-", color=DIFF_COLOR, lw=2, ms=6, capsize=3, label="mean, 95% CI")
    ax2.axhline(0.0, color="black", ls=":", lw=1.2, alpha=0.7)
    ax2.set_xlabel("budget fraction b"); ax2.set_ylabel("FORM − FID (paired per seed)")
    ax2.set_title("Paired difference (< 0: FORM better)", fontsize=11)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    fig.suptitle(f"Seed study - {dataset}, k = {k}", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, f"seeds_{dataset}_k{k}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Figure: {path}")


def plot_timesteps(tsstats, dataset, k, out_dir):
    import matplotlib.pyplot as plt

    g = tsstats[tsstats["k"] == k]
    bs = sorted(g["b"].unique())
    ncol = 4
    nrow = int(np.ceil(len(bs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.9 * nrow), sharex=True, sharey=True, squeeze=False)
    for ax, b in zip(axes.flat, bs):
        h = g[(g["b"] == b) & (g["timestep"] >= 2)].sort_values("timestep")
        for m in ("form", "fid"):
            mu, sd = h[f"ratio_{m}_mean"], h[f"ratio_{m}_sd"].fillna(0)
            ax.fill_between(h["timestep"], mu - sd, mu + sd, color=COLORS[m], alpha=0.2, lw=0)
            ax.plot(h["timestep"], mu, color=COLORS[m], lw=1.8, label=NAMES[m])
        ax.axhline(1.0, color="black", ls=":", lw=1, alpha=0.6)
        ax.set_title(f"b = {b:g}", fontsize=10); ax.grid(alpha=0.3)
    for ax in axes.flat[len(bs):]:
        ax.axis("off")
    axes[0][0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("timestep")
    for row in axes:
        row[0].set_ylabel("relative cost ratio")
    fig.suptitle(f"Mean ± sd across seeds per timestep - {dataset}, k = {k}", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, f"seeds_{dataset}_k{k}_timesteps.png")
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Figure: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default="wildfire")
    parser.add_argument("--results_dir", default=os.path.join(REPO_ROOT, "results", "seeds"))
    parser.add_argument("--output_dir", default=os.path.join(REPO_ROOT, "plots", "seeds"))
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    in_dir = os.path.join(args.results_dir, args.dataset)
    paths = sorted(glob.glob(os.path.join(in_dir, "*_s*.csv")))
    if not paths:
        print(f"No seed-study results in {in_dir} (run: python -m src.seed_study)")
        return
    proto_path = os.path.join(in_dir, "protocol.json")
    if os.path.exists(proto_path):
        proto = json.load(open(proto_path))
        print(f"Protocol: {proto.get('runs_per_configuration')} seeds {proto.get('run_seeds', [])[:3]}..., "
              f"data seed {proto.get('data_seed')}, {proto.get('baseline_restarts_per_timestep')} restarts/timestep, "
              f"rng {proto.get('rng')}")

    runs, ts = load_runs(paths)
    counts = runs.groupby(["k", "b"])["seed"].nunique()
    if counts.nunique() > 1:
        print("  [note] unequal number of seeds per configuration:")
        print(counts.to_string())

    summary, pair, tsstats = summarize(runs), paired(runs), per_timestep(ts)
    print_tables(summary, pair, tsstats, args.dataset)

    os.makedirs(args.output_dir, exist_ok=True)
    runs.to_csv(os.path.join(args.output_dir, f"seeds_runs_{args.dataset}.csv"), index=False)
    summary.to_csv(os.path.join(args.output_dir, f"seeds_summary_{args.dataset}.csv"), index=False)
    pair.to_csv(os.path.join(args.output_dir, f"seeds_paired_{args.dataset}.csv"), index=False)
    tsstats.to_csv(os.path.join(args.output_dir, f"seeds_timesteps_{args.dataset}.csv"), index=False)
    write_latex(summary, pair, os.path.join(args.output_dir, f"seeds_table_{args.dataset}.tex"))
    print(f"\nTables written to {args.output_dir}/seeds_*_{args.dataset}.csv and seeds_table_{args.dataset}.tex")

    if not args.no_plot:
        for k in sorted(runs["k"].unique()):
            plot_overview(runs, pair, args.dataset, k, args.output_dir)
            plot_timesteps(tsstats, args.dataset, k, args.output_dir)


if __name__ == "__main__":
    main()