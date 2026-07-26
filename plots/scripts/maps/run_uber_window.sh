#!/bin/bash

# Configuration lists
hour=($(seq 1 1 22))

# Nested loop to run combinations
for h in "${hour[@]}"; do
    curr_hour=$(printf "%02d" "$h")
    python plots/scripts/plot_uber_nyc.py --csv data/uber/uber-raw-data-apr14.csv --start "2014-04-01 ${curr_hour}:00" --end "2014-04-01 ${curr_hour}:59" --out "plots/uber/uber_map_2014-04-01_${curr_hour}.png"
done