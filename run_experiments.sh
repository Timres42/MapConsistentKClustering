#!/bin/bash

DATASETS=(
#  "twitter"
#  "online_retail"
#  "household_power"
#  "wildfire"
  "uber"
)

K_VALUES=(5)
B_FRACTIONS=(0.0)
HEURISTICS=("kmedianppwpost")
TIMESTEPS_VALUES=(30)

RESULTS_DIR="results/propagation"
PLOTS_DIR="plots/propagation"

mkdir -p "$RESULTS_DIR"
mkdir -p "$PLOTS_DIR"

# Limit the number of experiments running at the same time.
# Override manually, for example:
#   MAX_JOBS=4 ./run_experiments_parallel.sh
#
# Default: use half the available CPU cores, capped at 4.
if [[ -z "${MAX_JOBS:-}" ]]; then
  if command -v sysctl >/dev/null 2>&1; then
    CPU_COUNT=$(sysctl -n hw.ncpu 2>/dev/null || echo 2)
  elif command -v nproc >/dev/null 2>&1; then
    CPU_COUNT=$(nproc)
  else
    CPU_COUNT=2
  fi

  MAX_JOBS=$((CPU_COUNT / 2))
  if (( MAX_JOBS < 1 )); then
    MAX_JOBS=1
  elif (( MAX_JOBS > 4 )); then
    MAX_JOBS=4
  fi
fi

echo "Running at most $MAX_JOBS experiments in parallel."

# Whether to also generate plots after each dataset/timestep/k/heuristic batch.
# Off by default. Turn on manually, for example:
#   RUN_PLOTS=1 ./run_experiments_parallel.sh
if [[ -z "${RUN_PLOTS:-}" ]]; then
  RUN_PLOTS=0
fi

if [[ "$RUN_PLOTS" == "1" ]]; then
  echo "Plot generation is ENABLED."
else
  echo "Plot generation is DISABLED (set RUN_PLOTS=1 to enable)."
fi

# Build once before starting parallel jobs so multiple Python processes do not
# try to invoke make at the same time.
make

echo
echo "Checking datasets..."
for ds in "${DATASETS[@]}"; do
  case "$ds" in
    uber)
      path="data/uber/uber-raw-data-apr14.csv"
      [[ -f "$path" ]] || { echo "Missing '$path' - no auto-download available for uber (no stable direct URL); place the file there manually."; exit 1; }
      ;;
    wildfire)
      path="data/wildfire/FPA_FOD_20170508.sqlite"
      [[ -f "$path" ]] || { echo "Missing '$path' - no auto-download available for wildfire (no stable direct URL); place the file there manually."; exit 1; }
      ;;
    household_power)
      python -c "from data.loader.household_power import HouseholdPowerDataset as D; D(data_path='data/household_power/household_power.txt').ensure_downloaded()" || exit 1
      ;;
    online_retail)
      python -c "from data.loader.online_retail import OnlineRetailDataset as D; D(data_path='data/online_retail/online_retail.xlsx').ensure_downloaded()" || exit 1
      ;;
    twitter)
      python -c "from data.loader.twitter import TwitterGeospatialDataset as D; D(data_path='data/twitter/twitter.csv').ensure_downloaded()" || exit 1
      ;;
    *)
      echo "Unknown dataset '$ds' - add a case for it in the dataset check block."
      exit 1
      ;;
  esac
done
echo "All datasets present."

for timesteps in "${TIMESTEPS_VALUES[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for k in "${K_VALUES[@]}"; do
      for heur in "${HEURISTICS[@]}"; do

        echo
        echo "Running dataset=$ds timesteps=$timesteps k=$k heuristic=$heur"

        # Run b-fraction experiments in bounded parallel batches.
        # This uses separate processes rather than threads, which is the usual
        # way to parallelize independent commands from Bash.
        pids=()

        for b in "${B_FRACTIONS[@]}"; do
          python -m src.propagation_setup \
            --dataset "$ds" \
            --k "$k" \
            --b_fraction "$b" \
            --timesteps "$timesteps" \
            --heuristic "$heur" \
            --no_plot \
            --results_dir "$RESULTS_DIR/$ds" &

          pids+=("$!")

          # Once MAX_JOBS processes are running, wait for the whole batch.
          # This avoids wait -n, so it also works with the older Bash shipped
          # by default on macOS.
          if (( ${#pids[@]} >= MAX_JOBS )); then
            for pid in "${pids[@]}"; do
              wait "$pid" || exit 1
            done
            pids=()
          fi
        done

        # Wait for any remaining processes in the final partial batch.
        for pid in "${pids[@]}"; do
          wait "$pid" || exit 1
        done

        if [[ "$RUN_PLOTS" == "1" ]]; then
          # Collect all CSV files for this dataset/timestep/k/heuristic
          # combination, varying only the b fraction.
          csv_files=()

          for b in "${B_FRACTIONS[@]}"; do
            csv="$RESULTS_DIR/$ds/[${timesteps}]_${ds}_k${k}_base-${heur}_b${b}.csv"

            if [[ -f "$csv" ]]; then
              csv_files+=("$csv")
            fi
          done

          # Skip if no files exist for this combination.
          if [[ ${#csv_files[@]} -eq 0 ]]; then
            echo "No files found for dataset=$ds timesteps=$timesteps k=$k heuristic=$heur"
            continue
          fi

          # Compute one shared y-axis range across all b fractions.
          read y_min y_max < <(
            python - "${csv_files[@]}" <<'PY'
import sys
import pandas as pd

values = []

for path in sys.argv[1:]:
    df = pd.read_csv(path)

    if "relative_cost_ratio" in df.columns:
        values.append(df["relative_cost_ratio"])

if not values:
    raise SystemExit("No relative_cost_ratio data found")

all_values = pd.concat(values, ignore_index=True).dropna()

if all_values.empty:
    raise SystemExit("relative_cost_ratio contains no valid values")

y_min = all_values.min()
y_max = all_values.max()

span = y_max - y_min

if span > 0:
    padding = 0.05 * span
else:
    padding = 0.05 * abs(y_max) if y_max != 0 else 0.05

print(y_min - padding, y_max + padding)
PY
          )

          echo "Shared y range across b values: [$y_min, $y_max]"

          # Plot sequentially to avoid adding unnecessary CPU and memory load.
          for b in "${B_FRACTIONS[@]}"; do
            csv="$RESULTS_DIR/$ds/[${timesteps}]_${ds}_k${k}_base-${heur}_b${b}.csv"

            if [[ ! -f "$csv" ]]; then
              echo "Skipping missing file: $csv"
              continue
            fi

            python plots/scripts/plots_propagation_ratio_bviolin.py \
              --csv "$csv" \
              --y_min "$y_min" \
              --y_max "$y_max" \
              --output_dir "$PLOTS_DIR/$ds"
          done
        fi
      done
    done
  done
done