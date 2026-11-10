#!/bin/bash
# Creates all figures and tables of the paper from the results of
# run_experiments_plots_paper.sh. Run from the repo root.
set -e

TIMESTEPS="${TIMESTEPS:-30}"
# The setup, as in run_experiments_plots_paper.sh, for example:
#   INSTANCES=separate BASELINE=optimal bash plots/scripts/run_plots_for_paper.sh
INSTANCES="${INSTANCES:-shared}"
BASELINE="${BASELINE:-kmedianppwpost}"
RESULTS_ROOT="${RESULTS_ROOT:-results/paper_${INSTANCES}_${BASELINE}}"
PLOTS_DIR="${PLOTS_DIR:-plots/paper/${INSTANCES}_${BASELINE}}"
OPT_DIR="$RESULTS_ROOT/opt"
NO_OPT_DIR="$RESULTS_ROOT/no_opt"
SCALING_DIR="$RESULTS_ROOT/wildfire_node_scaling"

DATASETS=(wildfire online_retail household_power twitter uber)

# Baselines of FORM and FID in the result file names, the first one with a result is used.
read -r -a BASELINES <<< "${BASELINES:-$BASELINE}"

# Wildfire, k = 5: cost ratio, its distribution and the budget used, one row per b.
python plots/scripts/plots_propagation_ratio_bviolin_multiple.py \
  --dataset wildfire \
  --k 5 \
  --opt \
  --timesteps "$TIMESTEPS" \
  --b_values 0.0 0.1 0.5 \
  --heuristic "${BASELINES[@]}" \
  --results_dir "$OPT_DIR" \
  --output_dir "$PLOTS_DIR"

# Cost ratios for wildfire and online_retail, k = 10 and 20.
python plots/scripts/plots_propagation_ratio_multiple.py \
  --datasets wildfire online_retail \
  --k_values 10 20 \
  --opt \
  --timesteps "$TIMESTEPS" \
  --global_y_max 1.75 \
  --heuristic "${BASELINES[@]}" \
  --results_dir "$OPT_DIR" \
  --output_dir "$PLOTS_DIR"

# Every dataset: k = 5 and 10 with the exact algorithms, k = 20 without.
for dataset in "${DATASETS[@]}"; do
  for k in 5 10; do
    python plots/scripts/plots_propagation_ratio_bviolin_multiple.py \
      --dataset "$dataset" \
      --k "$k" \
      --opt \
      --timesteps "$TIMESTEPS" \
      --b_values 0.0 0.01 0.1 0.5 1.0 \
      --heuristic "${BASELINES[@]}" \
      --results_dir "$OPT_DIR" \
      --output_dir "$PLOTS_DIR"
  done
  python plots/scripts/plots_propagation_ratio_bviolin_multiple.py \
    --dataset "$dataset" \
    --k 20 \
    --timesteps "$TIMESTEPS" \
    --b_values 0.0 0.01 0.1 0.5 1.0 \
    --heuristic "${BASELINES[@]}" \
    --results_dir "$NO_OPT_DIR" \
    --output_dir "$PLOTS_DIR"
done

# Every dataset: runtime of FORM.
python plots/scripts/plots_propagation_runtime.py \
  --datasets "${DATASETS[@]}" \
  --timesteps "$TIMESTEPS" \
  --heuristic "${BASELINES[@]}" \
  --results_dir "$NO_OPT_DIR" \
  --output_dir "$PLOTS_DIR"

# Wildfire: runtime of FORM depending on the number of points.
python plots/scripts/plot_wildfire_node_scaling.py \
  --summary \
  --timesteps "$TIMESTEPS" \
  --heuristic "${BASELINES[@]}" \
  --results_dir "$SCALING_DIR" \
  --output_dir "$PLOTS_DIR"

# Table of the approximation factor between FORM and OPT_MC.
python plots/scripts/approx_factor_form.py \
  --results_dir "$OPT_DIR" \
  --output "$PLOTS_DIR/approx_factor_form.txt" \
  --tex "$PLOTS_DIR/approx_factor_form.tex"

# Tables of the runtime factor between OPT_MC and FORM, and between OPT_FID and FID.
for heuristic in form fid; do
  python plots/scripts/runtime_factor.py \
    --heuristic "$heuristic" \
    --results_dir "$OPT_DIR" \
    --output "$PLOTS_DIR/runtime_factor_$heuristic.txt" \
    --tex "$PLOTS_DIR/runtime_factor_$heuristic.tex"
done
