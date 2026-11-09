"""Locate result CSVs, preferring the first baseline in a list that has a result."""

import glob
import os
from pathlib import Path
from typing import List, Optional


def reference_label(df) -> str:
    """Legend label of the ratio-1 line: the optimal k-median when the ratios are normalized to
    it (--heuristic optimal, or --reference optimal with an opt_kmedian row per timestep)."""
    optimal_baseline = (df["baseline/sigma used"] == "optimal").all() if "baseline/sigma used" in df else False
    optimal_reference = ((df["method"] == "opt_kmedian") & (df["timestep"] > 1)).any()
    return "Optimal k-median (ratio=1)" if optimal_baseline or optimal_reference else "Baseline (ratio=1)"


def find_csv(
    directory,
    dataset: str,
    k: int,
    timesteps: int,
    heuristics: List[str],
    b: float,
    num_kernels: Optional[int] = None,
    opt: bool = False,
) -> Optional[Path]:
    """Return the CSV of the first heuristic with a result, or None.

    num_kernels None matches any kernel count (newest file wins); opt selects the
    files of --opt runs, whose instances differ from runs without --opt.
    """
    kernels = "*" if num_kernels is None else str(num_kernels)
    suffix = "_opt" if opt else ""
    for heuristic in heuristics:
        pattern = (
            glob.escape(f"{directory}/[{timesteps}]_{dataset}_k{k}_kernels")
            + kernels
            + glob.escape(f"_base-{heuristic}_b{b}{suffix}.csv")
        )
        matches = glob.glob(pattern)
        if matches:
            if heuristic != heuristics[0]:
                print(f"No {heuristics[0]} baseline for {dataset} k={k} b={b}: using {heuristic}")
            return Path(max(matches, key=os.path.getmtime))
    return None
