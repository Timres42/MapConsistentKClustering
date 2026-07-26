"""
Plot wildfires from the "1.88 Million US Wildfires" dataset that fall
inside a given time window.

Dataset: Short, K.C. 2017. Spatial wildfire occurrence data for the
United States, 1992-2015 [FPA_FOD_20170508].
https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires

Source file: FPA_FOD_20170508.sqlite (table: Fires)
Relevant columns used here:
    DISCOVERY_DATE  (stored as a Julian day number, e.g. 2453404.5)
    LATITUDE, LONGITUDE
    FIRE_SIZE       (acres)
    STAT_CAUSE_DESCR

Usage:
    python plot_wildfires.py --start "2006-06-01" --end "2006-09-01"

    If --db is omitted (or points to a path that doesn't exist), the
    dataset is fetched automatically via kagglehub, which downloads it
    once and caches it locally under ~/.cache/kagglehub for later runs.
    Requires a Kaggle account; kagglehub will prompt for credentials
    (or read them from ~/.kaggle/kaggle.json / KAGGLE_USERNAME +
    KAGGLE_KEY env vars) the first time it needs to download.

    You can still point at a local file explicitly to skip all of that:
        python plot_wildfires.py --db FPA_FOD_20170508.sqlite \
            --start "2006-06-01" --end "2006-09-01"

Alaska/Hawaii/Puerto Rico are excluded by default (--include-outlying to
keep them), since including them stretches the map and distorts the
Euclidean-looking plot; this only affects what's *shown*, not what's
read from the database.
"""

import argparse
import glob
import os
import sqlite3
import sys
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

# Approximate bounding box for the contiguous United States.
CONUS_BOUNDS = {
    "lon_min": -125.0,
    "lon_max": -66.0,
    "lat_min": 24.0,
    "lat_max": 50.0,
}

# Wider bounding box that also covers Alaska, Hawaii, and Puerto Rico,
# used instead of CONUS_BOUNDS when --include-outlying is passed.
ALL_US_BOUNDS = {
    "lon_min": -180.0,
    "lon_max": -64.0,
    "lat_min": 15.0,
    "lat_max": 72.0,
}

# FPA FOD stores DISCOVERY_DATE as a Julian day number. This is the
# offset needed to convert it to a standard Gregorian pandas Timestamp
# (Julian day 0 = noon, Jan 1, 4713 BCE; this constant is the Julian
# day number corresponding to 1970-01-01, the Unix epoch).
JULIAN_EPOCH_OFFSET = 2440587.5

# Fixed color per cause, assigned once from the full, known set of
# STAT_CAUSE_DESCR values in this dataset (sorted alphabetically for a
# stable, reproducible assignment). Using a fixed lookup instead of
# letting matplotlib/geopandas auto-assign colors per plot means a
# given cause (e.g. "Lightning") is always the same color, regardless
# of which other causes happen to appear in a particular time window.
KNOWN_CAUSES = [
    "Arson",
    "Campfire",
    "Children",
    "Debris Burning",
    "Equipment Use",
    "Fireworks",
    "Lightning",
    "Miscellaneous",
    "Missing/Undefined",
    "Powerline",
    "Railroad",
    "Smoking",
    "Structure",
]
_cause_cmap = plt.get_cmap("tab20")
CAUSE_COLORS = {cause: _cause_cmap(i / len(KNOWN_CAUSES)) for i, cause in enumerate(KNOWN_CAUSES)}
FALLBACK_CAUSE_COLOR = "black"  # for any cause value not in KNOWN_CAUSES


def color_for_cause(cause: str) -> str:
    return CAUSE_COLORS.get(cause, FALLBACK_CAUSE_COLOR)


LEGEND_MARKER_SIZE = 8  # fixed swatch size (points), independent of --size-by-fire


def build_cause_legend_handles(causes_present):
    """Proxy Line2D handles with a constant marker size, so the legend
    swatches never reflect the actual (possibly fire-size-scaled) point
    sizes plotted -- keeping the legend visually identical run to run."""
    from matplotlib.lines import Line2D
    return [
        Line2D([0], [0], marker="o", linestyle="", color=color_for_cause(cause),
               markersize=LEGEND_MARKER_SIZE, label=cause)
        for cause in causes_present
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot US wildfires within a time window.")
    parser.add_argument("--db", default=None,
                         help="Path to FPA_FOD_20170508.sqlite. If omitted or the path "
                              "doesn't exist, the dataset is downloaded automatically via "
                              "kagglehub.")
    parser.add_argument("--start", required=True, help="Window start, e.g. '2006-06-01'.")
    parser.add_argument("--end", required=True, help="Window end, e.g. '2006-09-01'.")
    parser.add_argument("--out", default="plots/wildfire/wildfires_map.png", help="Output image file.")
    parser.add_argument("--size-by-fire", action="store_true",
                         help="Scale marker size by FIRE_SIZE (acres) instead of a fixed size.")
    parser.add_argument("--color-by-cause", action="store_true",
                         help="Color points by STAT_CAUSE_DESCR instead of a single color.")
    parser.add_argument("--include-outlying", action="store_true",
                         help="Include Alaska, Hawaii, and Puerto Rico (stretches the map).")
    parser.add_argument("--point-size", type=float, default=6.0, help="Base marker size.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Marker transparency (0-1).")
    return parser.parse_args()


def resolve_db_path(db_arg: Optional[str]) -> str:
    """Return a usable path to FPA_FOD_20170508.sqlite, downloading the
    dataset via kagglehub if the given path is missing or wasn't given."""
    if db_arg and os.path.isfile(db_arg):
        return db_arg

    if db_arg:
        print(f"'{db_arg}' not found, falling back to kagglehub download...")

    try:
        import kagglehub
    except ImportError:
        sys.exit(
            "No local --db file found/given, and 'kagglehub' isn't installed to "
            "download it automatically.\n"
            "    pip install kagglehub\n"
            "or pass an existing file with --db path/to/FPA_FOD_20170508.sqlite"
        )

    print("Downloading/locating dataset via kagglehub (cached after the first run)...")
    dataset_dir = kagglehub.dataset_download("rtatman/188-million-us-wildfires")
    print(f"Dataset available at: {dataset_dir}")

    matches = glob.glob(os.path.join(dataset_dir, "**", "*.sqlite"), recursive=True)
    if not matches:
        sys.exit(
            f"No .sqlite file found under {dataset_dir}. "
            "The dataset layout may have changed; check kagglehub's output above."
        )
    return matches[0]


def load_data(db_path: str) -> pd.DataFrame:
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error as e:
        sys.exit(f"Error opening database: {e}")

    query = """
        SELECT DISCOVERY_DATE, LATITUDE, LONGITUDE, FIRE_SIZE,
               STAT_CAUSE_DESCR, STATE
        FROM Fires
        WHERE LATITUDE IS NOT NULL AND LONGITUDE IS NOT NULL
    """
    try:
        df = pd.read_sql_query(query, conn)
    except (pd.io.sql.DatabaseError, sqlite3.Error) as e:
        sys.exit(f"Error reading 'Fires' table: {e}")
    finally:
        conn.close()

    # Convert Julian day number -> pandas Timestamp.
    df["discovery_datetime"] = pd.to_datetime(
        df["DISCOVERY_DATE"] - JULIAN_EPOCH_OFFSET, unit="D", origin="unix"
    )
    return df


def filter_time_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    if start_ts >= end_ts:
        sys.exit("Error: --start must be earlier than --end.")
    windowed = df[(df["discovery_datetime"] >= start_ts) & (df["discovery_datetime"] <= end_ts)]
    return windowed.copy()


def restrict_to_conus(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["LONGITUDE"].between(CONUS_BOUNDS["lon_min"], CONUS_BOUNDS["lon_max"])
        & df["LATITUDE"].between(CONUS_BOUNDS["lat_min"], CONUS_BOUNDS["lat_max"])
    ]


def get_bounds(include_outlying: bool) -> dict:
    return ALL_US_BOUNDS if include_outlying else CONUS_BOUNDS


def plot_wildfires(df: pd.DataFrame, start: str, end: str, out_path: str,
                    size_by_fire: bool, color_by_cause: bool,
                    point_size: float, alpha: float, include_outlying: bool):
    fig, ax = plt.subplots(figsize=(12, 8))

    if df.empty:
        print("Warning: no wildfires found in the given time window.")

    sizes = point_size
    if size_by_fire:
        # FIRE_SIZE is in acres and heavily right-skewed; sqrt-scale it
        # so a few huge fires don't make every other point invisible.
        sizes = point_size + df["FIRE_SIZE"].fillna(0).pow(0.5) * 0.8

    use_basemap = False
    try:
        import contextily as cx
        use_basemap = True
    except ImportError:
        pass

    if use_basemap:
        import geopandas as gpd

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["LONGITUDE"], df["LATITUDE"]),
            crs="EPSG:4326",
        ).to_crs(epsg=3857)

        if color_by_cause:
            causes = gdf["STAT_CAUSE_DESCR"].fillna("Unknown")
            # Iterate causes in the same fixed order every time (not
            # whatever order .groupby() happens to yield), so overlapping
            # points are stacked identically across different windows too.
            present = [c for c in KNOWN_CAUSES if c in set(causes)] + \
                      sorted(set(causes) - set(KNOWN_CAUSES))
            for cause in present:
                mask = causes == cause
                s = sizes[mask.values] if size_by_fire else sizes
                gdf[mask].plot(ax=ax, markersize=s, alpha=alpha,
                                color=color_for_cause(cause))
            ax.legend(handles=build_cause_legend_handles(present),
                      loc="lower left", fontsize=7)
        else:
            gdf.plot(ax=ax, markersize=sizes, alpha=alpha, color="orangered")

        # Fixed crop: project the same lon/lat box every run and set it
        # as the axis limits, instead of letting geopandas autoscale to
        # whatever points happen to be in this particular time window.
        bounds = get_bounds(include_outlying)
        corners = gpd.GeoSeries(
            gpd.points_from_xy(
                [bounds["lon_min"], bounds["lon_max"]],
                [bounds["lat_min"], bounds["lat_max"]],
            ),
            crs="EPSG:4326",
        ).to_crs(epsg=3857)
        ax.set_xlim(corners.x.min(), corners.x.max())
        ax.set_ylim(corners.y.min(), corners.y.max())

        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
        ax.set_axis_off()
    else:
        if color_by_cause:
            causes = df["STAT_CAUSE_DESCR"].fillna("Unknown")
            present = [c for c in KNOWN_CAUSES if c in set(causes)] + \
                      sorted(set(causes) - set(KNOWN_CAUSES))
            for cause in present:
                group = df[causes == cause]
                s = sizes[group.index] if size_by_fire else sizes
                ax.scatter(group["LONGITUDE"], group["LATITUDE"], s=s,
                           alpha=alpha, linewidths=0,
                           color=color_for_cause(cause))
            ax.legend(handles=build_cause_legend_handles(present),
                      loc="lower left", fontsize=7)
        else:
            ax.scatter(df["LONGITUDE"], df["LATITUDE"], s=sizes,
                       alpha=alpha, color="orangered", linewidths=0)

        # Fixed crop, same reasoning as the basemap branch above.
        bounds = get_bounds(include_outlying)
        ax.set_xlim(bounds["lon_min"], bounds["lon_max"])
        ax.set_ylim(bounds["lat_min"], bounds["lat_max"])
        ax.set_aspect(1.3)
        ax.set_facecolor("#111111")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        print(
            "Note: 'contextily' (and 'geopandas') not installed, so no basemap "
            "tiles are shown. Install them for a real basemap:\n"
            "    pip install contextily geopandas"
        )

    ax.set_title(f"US wildfires\n{start} to {end}  (n={len(df)})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


def main():
    args = parse_args()
    db_path = resolve_db_path(args.db)
    df = load_data(db_path)
    windowed = filter_time_window(df, args.start, args.end)
    if not args.include_outlying:
        windowed = restrict_to_conus(windowed)
    plot_wildfires(windowed, args.start, args.end, args.out,
                    args.size_by_fire, args.color_by_cause,
                    args.point_size, args.alpha, args.include_outlying)


if __name__ == "__main__":
    main()