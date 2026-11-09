#!/usr/bin/env python3
"""Write FORM/OPT_MC cost ratios for available propagation opt results, optionally as a tex table."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NESTED_RESULTS = REPO_ROOT / "results" / "propagation" / "opts"
SIBLING_RESULTS = REPO_ROOT / "results" / "propagation_opts"
DEFAULT_RESULTS = NESTED_RESULTS if NESTED_RESULTS.is_dir() else SIBLING_RESULTS
DEFAULT_OUTPUT = REPO_ROOT / "plots" / "approx_factor_form.txt"
REQUIRED_COLUMNS = {"dataset", "timestep", "k", "method", "b_fraction", "result_cost"}
# Default of \kgap, the extra space between the columns of different k in the tex table. It is the
# width of an empty column between them, so that the rules under the k headers stop before it.
K_GAP = "1em"
GAP_COLUMN = "@{\\hspace{\\tabcolsep}}p{\\kgap}@{\\hspace{\\tabcolsep}}"


def join_blocks(cells: list[str], block_size: int) -> str:
    """Join the cells of a tex row, with an empty cell for the gap column between blocks."""
    blocks = [cells[i:i + block_size] for i in range(0, len(cells), block_size)]
    return " & & ".join(" & ".join(block) for block in blocks)
DATASET_NAMES = {
    "wildfire": "Wildfire",
    "uber": "Uber",
    "twitter": "Twitter",
    "online_retail": "Online Retail",
    "household_power": "Household Power",
}


def load_ratios(
    results_dir: Path,
) -> tuple[dict[tuple[str, int, float], list[float]], dict[tuple[str, int], int]]:
    """Return FORM/OPT_MC cost ratios paired by timestep, grouped by dataset, k, and b, and
    per dataset and k the number of paired OPT_MC solves that did not prove optimality."""
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    ratios = defaultdict(list)
    unproven = defaultdict(int)
    csv_paths = sorted(results_dir.rglob("*.csv"))
    for path in csv_paths:
        with path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"Missing columns in {path}: {', '.join(sorted(missing))}"
                )

            grouped_rows = defaultdict(lambda: {"form": {}, "opt_mc": {}})
            not_optimal = set()
            for row in reader:
                method = row["method"]
                if method not in ("form", "opt_mc"):
                    continue
                dataset = row["dataset"]
                k = int(row["k"])
                if k not in (5, 10):
                    continue
                b = round(float(row["b_fraction"]), 6)
                timestep = int(row["timestep"])
                method_rows = grouped_rows[(dataset, k, b)][method]
                if timestep in method_rows:
                    raise ValueError(
                        f"Duplicate {method} row for {dataset} k={k} b={b:g} "
                        f"timestep={timestep} in {path}"
                    )
                method_rows[timestep] = row["result_cost"]
                if method == "opt_mc" and row.get("opt_optimal") == "0":
                    not_optimal.add((dataset, k, b, timestep))

        for (dataset, k, b), methods in grouped_rows.items():
            ratios_for_b = []
            for timestep in methods["form"].keys() & methods["opt_mc"].keys():
                try:
                    form_cost = float(methods["form"][timestep])
                    opt_mc_cost = float(methods["opt_mc"][timestep])
                except ValueError:
                    continue
                if not math.isfinite(form_cost) or not math.isfinite(opt_mc_cost):
                    continue
                if opt_mc_cost == 0:
                    continue
                ratios_for_b.append(form_cost / opt_mc_cost)
                if (dataset, k, b, timestep) in not_optimal:
                    unproven[(str(dataset), k)] += 1

            if ratios_for_b:
                ratios[(str(dataset), k, b)].extend(ratios_for_b)

    return ratios, unproven


def format_report(ratios: dict[tuple[str, int, float], list[float]]) -> str:
    lines = [
        "# FORM / OPT_MC cost ratio; each b value is averaged over paired timesteps."
    ]
    by_dataset_k = defaultdict(list)
    for (dataset, k, b), values in ratios.items():
        by_dataset_k[(dataset, k)].append((b, sum(values) / len(values)))

    for (dataset, k), b_values in sorted(by_dataset_k.items()):
        b_values.sort()
        values = [ratio for _, ratio in b_values]
        b_fields = " | ".join(f"b={b:g}: {ratio:.6g}" for b, ratio in b_values)
        lines.append(
            f"{dataset} k={k} | {b_fields} | average={sum(values) / len(values):.6g}"
            f" | max={max(values):.6g}"
        )

    return "\n".join(lines) + "\n"


def format_tex_table(
    ratios: dict[tuple[str, int, float], list[float]],
    unproven: dict[tuple[str, int], int],
) -> str:
    """Booktabs table: a block of b columns per k, side by side; one row per dataset with the mean
    ratio, followed by the mean and the worst timestep over all datasets."""
    b_values = sorted({b for _, _, b in ratios})
    dataset_order = {name: i for i, name in enumerate(DATASET_NAMES)}
    datasets = sorted({dataset for dataset, _, _ in ratios},
                      key=lambda dataset: (dataset_order.get(dataset, len(dataset_order)), dataset))
    k_values = sorted({k for _, k, _ in ratios})
    columns = [(k, b) for k in k_values for b in b_values]

    n = len(b_values)
    k_header = join_blocks([f"\\multicolumn{{{n}}}{{c}}{{$k={k}$}}" for k in k_values], 1)
    k_rules = " ".join(
        f"\\cmidrule(lr){{{2 + i * (n + 1)}-{1 + i * (n + 1) + n}}}" for i in range(len(k_values))
    )
    b_header = join_blocks([f"$b={b:g}$" for _ in k_values for b in b_values], n)
    column_spec = "l" + GAP_COLUMN.join("r" * n for _ in k_values)
    lines = [
        "% Requires \\usepackage{booktabs}.",
        "\\begin{table*}[t]",
        "\\centering",
        f"\\providecommand{{\\kgap}}{{{K_GAP}}}% Extra space between the columns of different k.",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        f" & {k_header} \\\\",
        k_rules,
        f"Dataset & {b_header} \\\\",
        "\\midrule",
    ]
    for dataset in datasets:
        cells = []
        for k, b in columns:
            values = ratios.get((dataset, k, b), [])
            cells.append(f"{sum(values) / len(values):.4f}" if values else "--")
        name = DATASET_NAMES.get(dataset, dataset.replace("_", "\\_"))
        marker = "$^{\\dagger}$" if any(unproven.get((dataset, k)) for k in k_values) else ""
        lines.append(f"{name}{marker} & {join_blocks(cells, n)} \\\\")

    column_ratios = [[r for dataset in datasets for r in ratios.get((dataset, k, b), [])]
                     for k, b in columns]
    lines.append("\\midrule")
    lines.append("Mean & " + join_blocks(
        [f"{sum(values) / len(values):.4f}" if values else "--" for values in column_ratios], n) + " \\\\")
    lines.append("Max & " + join_blocks(
        [f"{max(values):.4f}" if values else "--" for values in column_ratios], n) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]

    caption = (
        "Approximation factor of FORM: cost of FORM divided by the cost of OPT\\_MC, averaged "
        "over all timesteps. For each $k$ and $b$, Mean averages over the timesteps of all "
        "datasets, Max is the largest ratio of a single timestep."
    )
    if unproven:
        solves = sum(unproven.values())
        caption += (
            f" $^{{\\dagger}}$Includes OPT\\_MC solves stopped by the time limit ({solves} in total), "
            "where the ratio is relative to the best solution found rather than the optimum."
        )
    lines += [f"\\caption{{{caption}}}", "\\label{tab:approx-factor-form}", "\\end{table*}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Opt-enabled propagation results directory",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tex", type=Path, default=None, help="Also write a tex table to this file")
    args = parser.parse_args()

    ratios, unproven = load_ratios(args.results_dir)
    if not ratios:
        raise ValueError(
            f"No paired FORM and OPT_MC costs found in {args.results_dir} "
            "for k=5 or k=10."
        )
    report = format_report(ratios)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Approximation-factor report saved to: {args.output}")
    if args.tex:
        args.tex.parent.mkdir(parents=True, exist_ok=True)
        args.tex.write_text(format_tex_table(ratios, unproven), encoding="utf-8")
        print(f"Approximation-factor table saved to: {args.tex}")


if __name__ == "__main__":
    main()
