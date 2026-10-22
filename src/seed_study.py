"""Seed study: effect of the k-median++ randomness on FORM, FID and the baseline.

Protocol
--------
* Input data is fixed: the dataset is loaded once with the data seed 42 (as in all other
  experiments), so every run sees exactly the same drift sequence.
* Only the k-median++ randomness of the C++ core varies. Each run gets a master seed S from
  a fixed, pre-specified list (default 1..30); restart r (of 5) at timestep t uses
  mt19937(seed_seq{S, t, r}), so all (S, t, r) streams are distinct.
* Grid (default): wildfire, k in {5, 10}, b in {0.0, 0.1, ..., 1.0}, 30 timesteps,
  heuristic kmedianppwpost, no exact algorithms -> 2 x 11 x 30 = 660 runs.
* Every run is kept and reported (no selection of seeds).

Results go to <results_dir>/<dataset>/[T]_<dataset>_k<k>_kernels1_base-<heuristic>_b<b>_s<S>.csv
(the CSV has a `seed` column), plus protocol.json with the full configuration. Runs whose
CSV already contains all timesteps are skipped, so an interrupted study can be resumed.

Evaluate with:  python plots/scripts/evaluate_seeds.py

Run from the repo root:
    python -m src.seed_study                      # full default grid, parallel
    python -m src.seed_study --jobs 4 --seeds 1-10 --k 5 --b 0.1 0.5
"""

import argparse
import datetime
import json
import os
import platform
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from src.propagation_setup import (ensure_binary, load_drift_sequence, run_cpp_experiment,
                                   serialize_sequence)

DATA_SEED = 42  # fixed: identical input data for every run
DEFAULT_SEEDS = "1-30"
DEFAULT_K = [5, 10]
DEFAULT_B = [round(i / 10, 1) for i in range(11)]
N_RESTARTS = 5  # restarts of the baseline heuristic per timestep (compute_baseline in propagation.cpp)


def parse_seeds(text: str) -> list:
    """'1-30' or '1,2,5' or '1-10,20'."""
    seeds = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    if any(s < 0 for s in seeds):
        raise ValueError("seeds must be >= 0")
    return sorted(set(seeds))


def is_complete(path: str, timesteps: int) -> bool:
    """True if the CSV exists and contains the FORM/FID rows of the last timestep."""
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path, usecols=["timestep", "method"])
    except Exception:
        return False
    last = df[df["timestep"] == timesteps]
    return {"form", "fid"} <= set(last["method"])


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default="wildfire")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS, help="Seed list, e.g. '1-30' or '1,2,7' (default 1-30)")
    parser.add_argument("--k", nargs="+", type=int, default=DEFAULT_K)
    parser.add_argument("--b", nargs="+", type=float, default=DEFAULT_B)
    parser.add_argument("--timesteps", type=int, default=30)
    parser.add_argument("--heuristic", default="kmedianppwpost", choices=["kmedian_plus_plus", "kmedianppwpost"])
    parser.add_argument("--point_fraction", type=float, default=1.0)
    parser.add_argument("--results_dir", default="results/seeds")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="Runs in parallel (default: half the CPU cores)")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    out_dir = os.path.join(os.path.abspath(args.results_dir), args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    protocol = {
        "description": "Effect of k-median++ randomness on FORM, FID and the baseline",
        "dataset": args.dataset, "point_fraction": args.point_fraction, "timesteps": args.timesteps,
        "k_values": args.k, "b_fractions": args.b, "heuristic": args.heuristic,
        "baseline_restarts_per_timestep": N_RESTARTS,
        "data_seed": DATA_SEED, "run_seeds": seeds, "runs_per_configuration": len(seeds),
        "rng": "std::mt19937 seeded with std::seed_seq{seed_lo, seed_hi, timestep, restart}",
        "exact_algorithms": False,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(), "platform": platform.platform(),
    }
    with open(os.path.join(out_dir, "protocol.json"), "w") as f:
        json.dump(protocol, f, indent=2)

    ensure_binary()  # build once before the parallel runs

    jobs = []
    tmp_files = []
    for k in args.k:
        print(f"Loading {args.dataset} (data seed {DATA_SEED}) for k={k} ...", flush=True)
        seq = load_drift_sequence(args.dataset, k, DATA_SEED, args.timesteps, args.point_fraction)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
            data_path = tf.name
        serialize_sequence(seq, data_path)
        tmp_files.append(data_path)
        for b in args.b:
            for s in seeds:
                out = os.path.join(out_dir, f"[{args.timesteps}]_{args.dataset}_k{k}_kernels1"
                                            f"_base-{args.heuristic}_b{b}_s{s}.csv")
                if is_complete(out, args.timesteps):
                    continue
                jobs.append(dict(k=k, b=b, seed=s, out=out, data_path=data_path))

    total = len(args.k) * len(args.b) * len(seeds)
    print(f"{total} runs in the grid, {total - len(jobs)} already complete, {len(jobs)} to run "
          f"with {args.jobs} in parallel.", flush=True)

    def run(job):
        run_cpp_experiment(drift_sequence=None, output_file=job["out"], dataset_name=args.dataset,
                           k=job["k"], b_fraction=job["b"], heuristic=args.heuristic, quiet=True,
                           seed=job["seed"], data_path=job["data_path"])
        return job

    failed = 0
    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run, j) for j in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    j = fut.result()
                    print(f"[{i}/{len(jobs)}] done: k={j['k']} b={j['b']} seed={j['seed']}", flush=True)
                except Exception as e:  # keep going; the run can be resumed later
                    failed += 1
                    print(f"[{i}/{len(jobs)}] FAILED: {e}", flush=True)
    finally:
        for p in tmp_files:
            if os.path.exists(p):
                os.remove(p)

    print(f"\nSeed study finished: {len(jobs) - failed} runs done, {failed} failed. Results: {out_dir}")
    if failed:
        print("Rerun the same command to retry the failed runs.")


if __name__ == "__main__":
    main()