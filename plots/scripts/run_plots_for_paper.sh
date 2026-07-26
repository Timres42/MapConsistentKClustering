#!/bin/bash

python plots/scripts/plots_propagation_ratio_bviolin_multiple.py \
  --dataset wildfire \
  --k 5 \
  --timesteps 30 \
  --b_values 0.0 0.1 0.5


python plots/scripts/plots_propagation_ratio_multiple.py \
  --datasets wildfire online_retail\
  --k_values 10 20 \
  --timesteps 30 \
  --global_y_max 1.75

for k in 5 10 20; do 
  for dataset in wildfire online_retail household_power twitter uber; do
    python plots/scripts/plots_propagation_ratio_bviolin_multiple.py \
      --dataset $dataset \
      --k $k \
      --timesteps 30 \
      --b_values 0.0 0.01 0.1 0.5 1.0
  done
done