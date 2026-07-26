#!/bin/bash

# Configuration lists
days=($(seq 1 1 29))

# Nested loop to run combinations
for day in "${days[@]}"; do
    start_day=$(printf "%02d" "$day")
    next_day=$(printf "%02d" "$((day+1))")
    python plots/scripts/plot_wildfire.py --db data/wildfire/FPA_FOD_20170508.sqlite --start "2006-06-${start_day}" --end "2006-06-${next_day}" --out "plots/wildfire/wildfires_map_wocause_2006-06-${start_day}.png"
done