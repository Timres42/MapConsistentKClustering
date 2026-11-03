#!/usr/bin/env python3
"""Write FORM/OPT_MC cost ratios for available propagation opt results."""

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


def load_ratios(results_dir: Path) -> dict[tuple[str, int, float], list[float]]:
    """Return FORM/OPT_MC cost ratios paired by timestep, grouped by dataset, k, and b."""
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    ratios = defaultdict(list)
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

            if ratios_for_b:
                ratios[(str(dataset), k, b)].extend(ratios_for_b)

    return ratios


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
        b_fields = " | ".join(f"b={b:.1f}: {ratio:.6g}" for b, ratio in b_values)
        lines.append(
            f"{dataset} k={k} | {b_fields} | average={sum(values) / len(values):.6g}"
            f" | max={max(values):.6g}"
        )

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
    args = parser.parse_args()

    ratios = load_ratios(args.results_dir)
    if not ratios:
        raise ValueError(
            f"No paired FORM and OPT_MC costs found in {args.results_dir} "
            "for k=5 or k=10."
        )
    report = format_report(ratios)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Approximation-factor report saved to: {args.output}")


if __name__ == "__main__":
    main()
