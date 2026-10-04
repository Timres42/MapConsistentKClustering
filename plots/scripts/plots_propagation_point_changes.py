"""Print statistics on how the instance creation changes FORM's center assignments.

Reads the results CSVs in results/propagation/<dataset>/ and uses the columns
`prev_points` (points of X_i whose coordinates already occur in X_{i-1}) and
`prev_points_changed` (how many of them changed their FORM center when X_i was
built), which are only filled in the `method == "form"` rows.

For each dataset, each k and each b_fraction it prints over all timesteps:
  - points still there:  mean, standard deviation, max
  - points changed:      mean, standard deviation, max
each as an absolute number and as a percentage of all points of the instance
(num_points of that timestep; for the percentages, mean/std/max are taken over
the per-timestep percentages).

Usage (from anywhere):
    python plots/scripts/evaluate_center_changes.py
    python plots/scripts/evaluate_center_changes.py --datasets uber wildfire --k 5 10
"""

import argparse
import glob
import os

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RESULTS = os.path.join(REPO_ROOT, "results", "propagation")
DEFAULT_K = [5, 10, 15, 20]


def load_form_rows(path: str):
    """FORM rows of one results CSV, or None if the CSV has no center-change columns."""
    df = pd.read_csv(path, na_values=["nan"])
    if not {"prev_points", "prev_points_changed"} <= set(df.columns):
        print(f"  [skipped] {os.path.relpath(path, REPO_ROOT)}: no prev_points columns "
              f"(produced before the binary was updated)")
        return None
    form = df[df["method"] == "form"].dropna(subset=["prev_points", "prev_points_changed"]).copy()
    form["source"] = os.path.basename(path)
    return form


def stats(values: pd.Series):
    # Sample standard deviation over the timesteps (0 if there is only one timestep).
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    return values.mean(), std, values.max()


def fmt_row(b, T, kept, kept_pct, changed, changed_pct) -> str:
    def block(abs_stats, pct_stats):
        m, s, mx = abs_stats
        pm, ps, pmx = pct_stats
        return (f"{m:9.1f} {s:8.1f} {mx:7.0f} | "
                f"{pm:7.2f}% {ps:6.2f}% {pmx:7.2f}%")
    return f"  {b:<6} {T:>3} || {block(kept, kept_pct)} || {block(changed, changed_pct)}"


_BLOCK = f"{'mean':>9} {'std':>8} {'max':>7} | {'mean%':>8} {'std%':>7} {'max%':>8}"
HEADER = (
    f"  {'':<10} || {'points still there':^54} || {'points changed center':^54}\n"
    f"  {'b':<6} {'T':>3} || {_BLOCK} || {_BLOCK}"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS,
                        help="Directory containing one subdirectory per dataset")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Datasets to evaluate (default: every subdirectory)")
    parser.add_argument("--k", nargs="*", type=int, default=DEFAULT_K, help="k values to report")
    args = parser.parse_args()

    if args.datasets:
        datasets = args.datasets
    else:
        datasets = sorted(d for d in os.listdir(args.results_dir)
                          if os.path.isdir(os.path.join(args.results_dir, d)))
    if not datasets:
        print(f"No dataset directories found in {args.results_dir}")
        return

    for dataset in datasets:
        print("\n" + "=" * 128)
        print(f"Dataset: {dataset}")
        print("=" * 128)
        files = sorted(glob.glob(os.path.join(args.results_dir, dataset, "*.csv")))
        frames = [f for f in (load_form_rows(p) for p in files) if f is not None and not f.empty]
        if not frames:
            print("  no results with center-change columns")
            continue
        data = pd.concat(frames, ignore_index=True)

        for k in args.k:
            sub_k = data[data["k"] == k]
            print(f"\nk = {k}")
            if sub_k.empty:
                print("  no results")
                continue
            print(HEADER)
            for b in sorted(sub_k["b_fraction"].unique()):
                rows = sub_k[sub_k["b_fraction"] == b]
                sources = rows["source"].unique()
                if len(sources) > 1:
                    print(f"  [note] b={b}: timesteps pooled from {len(sources)} files: {', '.join(sources)}")
                kept = rows["prev_points"].astype(float)
                changed = rows["prev_points_changed"].astype(float)
                total = rows["num_points"].astype(float)
                print(fmt_row(f"{b:g}", len(rows),
                              stats(kept), stats(100.0 * kept / total),
                              stats(changed), stats(100.0 * changed / total)))


if __name__ == "__main__":
    main()