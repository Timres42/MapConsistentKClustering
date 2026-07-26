#!/bin/bash

# Configuration lists
#minute=($(seq 0 1 58))
minute=(0 15 30 45)
# Nested loop to run combinations
for m in "${minute[@]}"; do
    curr_minute=$(printf "%02d" "$m")
    next_minute=$(printf "%02d" "$((m+14))")
    python3 plots/scripts/plot_twitter_geo.py --csv data/twitter/twitter.csv --start "2013-01-13 09:${curr_minute}" --end "2013-01-13 09:${next_minute}" --out "plots/maps/twitter/twitter_map-2013-01-13_09${curr_minute}.png"
done
