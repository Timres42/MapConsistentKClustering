#!/usr/bin/env python3
"""Write OPT_MC/FORM and OPT_FID/FID runtime factors of propagation opt results, optionally as tex tables.

The factor of a configuration is the total runtime of the exact algorithm divided by the total
runtime of its heuristic, over the timesteps where both have a result. Ratios of totals are used
because the heuristics run in fractions of a millisecond, so per-timestep ratios are dominated by
timer noise.
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO_ROOT / "results" / "paper" / "opt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "plots"
REQUIRED_COLUMNS = {"dataset", "timestep", "k", "method", "b_fraction", "execution_time"}
# Exact algorithm, its heuristic, and their names in tex.
PAIRS = {
    "form": ("opt_mc", "OPT\\_MC", "FORM"),
    "fid": ("opt_fid", "OPT\\_FID", "FID"),
}
# Default of \kgap, the extra space between the columns of different k in the tex table. It is the
# width of an empty column between them, so that the rules under the k headers stop before it.
K_GAP = "1em"
GAP_COLUMN = "@{\\hspace{\\tabcolsep}}p{\\kgap}@{\\hspace{\\tabcolsep}}"
DATASET_NAMES = {
    "wildfire": "Wildfire",
    "uber": "Uber",
    "twitter": "Twitter",
    "online_retail": "Online Retail",
    "household_power": "Household Power",
}


def load_runtimes(results_dir: Path, heuristic: str, k_values: set[int]):
    """Return per dataset, k and b the summed runtimes (exact, heuristic) over paired timesteps,
    and per dataset and k the number of paired exact solves stopped by the time limit."""
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    exact = PAIRS[heuristic][0]
    totals = {}
    unproven = defaultdict(int)
    for path in sorted(results_dir.rglob("*.csv")):
        with path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"Missing columns in {path}: {', '.join(sorted(missing))}")

            grouped_rows = defaultdict(lambda: {heuristic: {}, exact: {}})
            not_optimal = set()
            for row in reader:
                method = row["method"]
                if method not in (heuristic, exact):
                    continue
                k = int(row["k"])
                if k not in k_values:
                    continue
                key = (row["dataset"], k, round(float(row["b_fraction"]), 6))
                timestep = int(row["timestep"])
                method_rows = grouped_rows[key][method]
                if timestep in method_rows:
                    raise ValueError(
                        f"Duplicate {method} row for {key[0]} k={k} b={key[2]:g} "
                        f"timestep={timestep} in {path}"
                    )
                method_rows[timestep] = row["execution_time"]
                if method == exact and row.get("opt_optimal") == "0":
                    not_optimal.add((key, timestep))

        for key, methods in grouped_rows.items():
            exact_total = heuristic_total = 0.0
            min_ratio = math.inf
            for timestep in methods[heuristic].keys() & methods[exact].keys():
                try:
                    exact_time = float(methods[exact][timestep])
                    heuristic_time = float(methods[heuristic][timestep])
                except ValueError:
                    continue
                if not math.isfinite(exact_time) or not math.isfinite(heuristic_time):
                    continue
                exact_total += exact_time
                heuristic_total += heuristic_time
                if heuristic_time > 0:
                    min_ratio = min(min_ratio, exact_time / heuristic_time)
                if (key, timestep) in not_optimal:
                    unproven[key[:2]] += 1
            if heuristic_total > 0:
                if key in totals:
                    raise ValueError(f"Several result files for {key[0]} k={key[1]} b={key[2]:g}")
                totals[key] = (exact_total, heuristic_total, min_ratio)

    return totals, unproven


def join_blocks(cells: list[str], block_size: int) -> str:
    """Join the cells of a tex row, with an empty cell for the gap column between blocks."""
    blocks = [cells[i:i + block_size] for i in range(0, len(cells), block_size)]
    return " & & ".join(" & ".join(block) for block in blocks)


def factor(runtimes) -> float:
    """Factor of the summed runtimes of (exact total, heuristic total, ...) tuples."""
    return sum(r[0] for r in runtimes) / sum(r[1] for r in runtimes)


def format_report(totals, heuristic: str) -> str:
    exact = PAIRS[heuristic][0]
    lines = [f"# {exact.upper()} / {heuristic.upper()} runtime factor: total runtimes over paired timesteps."]
    by_dataset_k = defaultdict(list)
    for (dataset, k, b), runtimes in totals.items():
        by_dataset_k[(dataset, k)].append((b, runtimes))

    for (dataset, k), b_values in sorted(by_dataset_k.items()):
        b_values.sort()
        b_fields = " | ".join(f"b={b:g}: {factor([r]):.4g}" for b, r in b_values)
        lines.append(f"{dataset} k={k} | {b_fields} | all b={factor([r for _, r in b_values]):.4g}")
    return "\n".join(lines) + "\n"


def tex_number(value: float) -> str:
    """Two significant digits, as a power of ten."""
    exponent = math.floor(math.log10(value))
    mantissa = value / 10 ** exponent
    if round(mantissa, 1) >= 10:
        mantissa, exponent = mantissa / 10, exponent + 1
    return f"${mantissa:.1f}{{\\cdot}}10^{{{exponent}}}$"


def format_tex_table(totals, unproven, heuristic: str) -> str:
    """Booktabs table: a block of b columns per k, side by side; one row per dataset with the
    factor, followed by the mean factor over all datasets and the smallest factor of a timestep."""
    exact, exact_name, heuristic_name = PAIRS[heuristic]
    b_values = sorted({b for _, _, b in totals})
    dataset_order = {name: i for i, name in enumerate(DATASET_NAMES)}
    datasets = sorted({dataset for dataset, _, _ in totals},
                      key=lambda dataset: (dataset_order.get(dataset, len(dataset_order)), dataset))
    k_values = sorted({k for _, k, _ in totals})
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
        cells = [tex_number(factor([totals[(dataset, k, b)]])) if (dataset, k, b) in totals else "--"
                 for k, b in columns]
        name = DATASET_NAMES.get(dataset, dataset.replace("_", "\\_"))
        marker = "$^{\\dagger}$" if any(unproven.get((dataset, k)) for k in k_values) else ""
        lines.append(f"{name}{marker} & {join_blocks(cells, n)} \\\\")

    column_runtimes = [[totals[(dataset, k, b)] for dataset in datasets if (dataset, k, b) in totals]
                       for k, b in columns]
    lines.append("\\midrule")
    lines.append("Mean & " + join_blocks([
        tex_number(sum(factor([r]) for r in runtimes) / len(runtimes)) if runtimes else "--"
        for runtimes in column_runtimes], n) + " \\\\")
    lines.append("Min & " + join_blocks([
        tex_number(min(r[2] for r in runtimes)) if runtimes else "--"
        for runtimes in column_runtimes], n) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]

    caption = (
        f"Runtime factor of {exact_name} over {heuristic_name}: total runtime of {exact_name} "
        f"divided by the total runtime of {heuristic_name} over all timesteps. For each $k$ and "
        "$b$, Mean averages the factors of all datasets, Min is the smallest factor of a single "
        "timestep."
    )
    if unproven:
        solves = sum(unproven.values())
        caption += (
            f" $^{{\\dagger}}$Includes {exact_name} solves stopped by the time limit "
            f"({solves} in total), so the factor is a lower bound."
        )
    lines += [f"\\caption{{{caption}}}", f"\\label{{tab:runtime-factor-{heuristic}}}", "\\end{table*}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS,
                        help="Opt-enabled propagation results directory")
    parser.add_argument("--heuristic", choices=sorted(PAIRS), required=True,
                        help="form: OPT_MC / FORM, fid: OPT_FID / FID")
    parser.add_argument("--k_values", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--output", type=Path, default=None,
                        help="Text report (default: plots/runtime_factor_<heuristic>.txt)")
    parser.add_argument("--tex", type=Path, default=None, help="Also write a tex table to this file")
    args = parser.parse_args()

    totals, unproven = load_runtimes(args.results_dir, args.heuristic, set(args.k_values))
    if not totals:
        raise ValueError(
            f"No paired {args.heuristic} and {PAIRS[args.heuristic][0]} runtimes found in "
            f"{args.results_dir} for k in {args.k_values}."
        )
    output = args.output or DEFAULT_OUTPUT_DIR / f"runtime_factor_{args.heuristic}.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(format_report(totals, args.heuristic), encoding="utf-8")
    print(f"Runtime-factor report saved to: {output}")
    if args.tex:
        args.tex.parent.mkdir(parents=True, exist_ok=True)
        args.tex.write_text(format_tex_table(totals, unproven, args.heuristic), encoding="utf-8")
        print(f"Runtime-factor table saved to: {args.tex}")


if __name__ == "__main__":
    main()
