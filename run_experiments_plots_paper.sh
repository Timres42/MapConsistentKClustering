#!/bin/bash
# Runs exactly the experiments needed by plots/scripts/run_plots_for_paper.sh.
#
# Finished runs are remembered in $RESULTS_ROOT/.done, so after an interruption the script
# continues where it stopped (delete that folder to start over). Logs: $RESULTS_ROOT/logs.
#
# The setup is chosen by two variables; each setup has its own results folder
# results/paper_<INSTANCES>_<BASELINE>:
#   INSTANCES=shared    only FORM propagates its centers; FORM, FID, OPT MC and OPT FID all
#                       solve FORM's instance (default)
#   INSTANCES=separate  FORM and FID propagate their own centers and solve their own instances;
#                       OPT MC solves FORM's instance, OPT FID FID's
#   BASELINE=kmedianppwpost  FORM and FID receive kmedianppwpost (also as the initial
#                            clustering) (default)
#   BASELINE=optimal         FORM and FID receive the exact k-median
# All costs are normalized to the exact k-median of each instance (row opt_kmedian, or the
# baseline itself with BASELINE=optimal).
#
# Overrides, for example:
#   INSTANCES=separate BASELINE=optimal ./run_experiments_plots_paper.sh
#   MAX_JOBS=4 OPT_TIME_LIMIT=1800 ./run_experiments_plots_paper.sh

# Stop all experiments when this script ends or is interrupted. Background jobs of a
# script ignore Ctrl+C, so they have to be stopped explicitly, including their children.
kill_tree() {
  local child children
  children=$(pgrep -P "$1")
  kill -TERM "$1" 2>/dev/null
  for child in $children; do
    kill_tree "$child"
  done
}
stop_experiments() {
  trap - EXIT INT TERM HUP
  local child
  for child in $(pgrep -P $$); do
    kill_tree "$child"
  done
}
trap stop_experiments EXIT
trap 'stop_experiments; exit 130' INT TERM HUP

DATASETS=(wildfire online_retail household_power twitter uber)

TIMESTEPS="${TIMESTEPS:-30}"
INSTANCES="${INSTANCES:-shared}"           # shared or separate, see above
BASELINE="${BASELINE:-kmedianppwpost}"     # optimal or kmedianppwpost, see above
RESULTS_ROOT="${RESULTS_ROOT:-results/paper_${INSTANCES}_${BASELINE}}"
# Baselines given to FORM and FID, each tried if the previous one fails.
read -r -a HEURISTICS <<< "${HEURISTICS:-$BASELINE}"
REFERENCE="${REFERENCE:-optimal}"          # cost the ratios are normalized to
if [[ "$INSTANCES" != shared && "$INSTANCES" != separate ]]; then
  echo "INSTANCES must be shared or separate (got '$INSTANCES')"
  exit 1
fi
OPT_TIME_LIMIT="${OPT_TIME_LIMIT:-3600}"   # seconds per Gurobi solve
OPT_MIP_GAP="${OPT_MIP_GAP:-0}"            # 0 = solve to proven optimality
# Experiments run only after all others, e.g. DEFER="opt_online_retail_k5_b0.1"
read -r -a DEFER <<< "${DEFER:-}"

BVIOLIN_B=(0.0 0.01 0.1 0.5 1.0)
COMBINED_B=(0.0 0.1 0.5)
RUNTIME_K=(5 10 15 20)
RUNTIME_B=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

DONE_DIR="$RESULTS_ROOT/.done"
LOG_DIR="$RESULTS_ROOT/logs"
FAILURES="$RESULTS_ROOT/failures.txt"
mkdir -p "$DONE_DIR" "$LOG_DIR"
: > "$FAILURES"

# Default: half the available CPU cores, capped at 4.
if command -v sysctl >/dev/null 2>&1 && sysctl -n hw.ncpu >/dev/null 2>&1; then
  CPU_COUNT=$(sysctl -n hw.ncpu)
elif command -v nproc >/dev/null 2>&1; then
  CPU_COUNT=$(nproc)
else
  CPU_COUNT=2
fi

if [[ -z "${MAX_JOBS:-}" ]]; then
  MAX_JOBS=$((CPU_COUNT / 2))
  if (( MAX_JOBS < 1 )); then
    MAX_JOBS=1
  elif (( MAX_JOBS > 4 )); then
    MAX_JOBS=4
  fi
fi

# Share the cores between the parallel jobs (Gurobi would otherwise use all of them per job).
if [[ -z "${OPT_THREADS:-}" ]]; then
  OPT_THREADS=$((CPU_COUNT / MAX_JOBS))
  (( OPT_THREADS < 1 )) && OPT_THREADS=1
fi

echo "Running at most $MAX_JOBS experiments in parallel, $OPT_THREADS Gurobi threads per job."

# Build once before starting parallel jobs so multiple Python processes do not
# try to invoke make at the same time.
make || exit 1

if ! grep -q -- "-DUSE_GUROBI" build/.build_config 2>/dev/null; then
  echo "These experiments need a Gurobi build: set GUROBI_HOME to your Gurobi installation (see README)."
  exit 1
fi

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
  esac
done
echo "All datasets present."

# run_one <no_opt|opt> <dataset> <k> <b>: starts one experiment once a job slot is free.
run_one() {
  local mode="$1" ds="$2" k="$3" b="$4"
  local key="${mode}_${ds}_k${k}_b${b}"
  [[ -f "$DONE_DIR/$key" ]] && return
  [[ -z "${RUN_DEFERRED:-}" && " ${DEFER[*]} " == *" $key "* ]] && return

  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
    sleep 1
  done

  (
    local opt_flag=() suffix="" heur
    if [[ "$mode" == "opt" ]]; then
      opt_flag=(--opt)
      suffix="_opt"
    fi
    local results_dir="$RESULTS_ROOT/$mode/$ds"
    for heur in "${HEURISTICS[@]}"; do
      echo "Starting $key (baseline $heur)"
      if python -m src.propagation_setup \
          --dataset "$ds" --k "$k" --b_fraction "$b" --timesteps "$TIMESTEPS" \
          --heuristic "$heur" --reference "$REFERENCE" --instances "$INSTANCES" --num_kernels "$MAX_JOBS" --no_plot \
          --results_dir "$results_dir" \
          --opt_time_limit "$OPT_TIME_LIMIT" --opt_mip_gap "$OPT_MIP_GAP" --opt_threads "$OPT_THREADS" \
          ${opt_flag[@]+"${opt_flag[@]}"} > "$LOG_DIR/$key.log" 2>&1; then
        python -c "import pathlib, sys; pathlib.Path(sys.argv[1]).touch()" "$DONE_DIR/$key"
        echo "Finished $key (baseline $heur)"
        exit 0
      fi
      # A partial result must not be picked up by the plots.
      rm -f "$results_dir/[${TIMESTEPS}]_${ds}_k${k}_kernels${MAX_JOBS}_base-${heur}_b${b}${suffix}.csv"
      echo "FAILED $key with baseline $heur, see $LOG_DIR/$key.log"
    done
    echo "$key" >> "$FAILURES"
  ) &
}

# --- Runs without the exact algorithms: runtime of FORM, and the k = 20 plots. ---
# They wait for each other only, so that the slow exact solves below don't disturb the
# measured runtimes.
for ds in "${DATASETS[@]}"; do
  for k in "${RUNTIME_K[@]}"; do
    for b in "${RUNTIME_B[@]}"; do
      run_one no_opt "$ds" "$k" "$b"
    done
  done
  # The only b of the k = 20 plots that is not part of the runtime grid.
  for b in "${BVIOLIN_B[@]}"; do
    [[ " ${RUNTIME_B[*]} " == *" $b "* ]] || run_one no_opt "$ds" 20 "$b"
  done
done
wait

# --- Wildfire runtime depending on the number of points (runs sequentially). ---
if [[ ! -f "$DONE_DIR/wildfire_node_scaling" ]]; then
  echo "Starting wildfire_node_scaling"
  # In the background, because a signal interrupts `wait` but not a foreground command.
  python -m src.wildfire_node_scaling \
      --results_dir "$RESULTS_ROOT/wildfire_node_scaling" \
      --timesteps "$TIMESTEPS" \
      --opt_time_limit "$OPT_TIME_LIMIT" --opt_mip_gap "$OPT_MIP_GAP" \
      --heuristics "${HEURISTICS[@]}" --reference "$REFERENCE" --instances "$INSTANCES" \
      > "$LOG_DIR/wildfire_node_scaling.log" 2>&1 &
  if wait $!; then
    python -c "import pathlib, sys; pathlib.Path(sys.argv[1]).touch()" "$DONE_DIR/wildfire_node_scaling"
    echo "Finished wildfire_node_scaling"
  else
    echo "FAILED wildfire_node_scaling, see $LOG_DIR/wildfire_node_scaling.log"
    echo "wildfire_node_scaling" >> "$FAILURES"
  fi
fi

# --- Runs with the exact algorithms opt_fid and opt_mc. ---
for k in 5 10; do
  for ds in "${DATASETS[@]}"; do
    for b in "${BVIOLIN_B[@]}"; do
      run_one opt "$ds" "$k" "$b"
    done
  done
done
for ds in wildfire online_retail; do
  for b in "${COMBINED_B[@]}"; do
    run_one opt "$ds" 20 "$b"
  done
done
wait

# --- Deferred experiments (keys of the form <mode>_<dataset>_k<k>_b<b>). ---
RUN_DEFERRED=1
for key in ${DEFER[@]+"${DEFER[@]}"}; do
  if [[ "$key" =~ ^(no_opt|opt)_(.+)_k([0-9]+)_b([0-9.]+)$ ]]; then
    run_one "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}" "${BASH_REMATCH[4]}"
  else
    echo "Ignoring DEFER entry '$key'"
  fi
done
wait

if [[ -s "$FAILURES" ]]; then
  echo
  echo "These experiments failed (listed in $FAILURES):"
  cat "$FAILURES"
  exit 1
fi

echo
echo "All experiments finished. Now run: bash plots/scripts/run_plots_for_paper.sh"
