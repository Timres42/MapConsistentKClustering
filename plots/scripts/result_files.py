"""Locate result CSVs, preferring the first baseline in a list that has a result."""

import glob
import os
from pathlib import Path
from typing import List, Optional


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
