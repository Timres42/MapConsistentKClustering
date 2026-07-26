"""
Plot Uber NYC pickups that fall inside a given time window.

Dataset: Uber Pickups in New York City (FiveThirtyEight, Kaggle)
https://www.kaggle.com/datasets/fivethirtyeight/uber-pickups-in-new-york-city

Expected CSV columns (uber-raw-data-*.csv files):
    Date/Time, Lat, Lon, Base
Example row:
    4/1/2014 0:11:00,40.769,-73.9549,B02512

Usage:
    python plot_uber_pickups.py --start "2014-04-01 08:00" --end "2014-04-01 10:00"

    If --csv is omitted (or points to a path that doesn't exist), the
    matching monthly file is fetched automatically via kagglehub based
    on --start's year/month, and cached locally under ~/.cache/kagglehub
    for later runs. Requires a Kaggle account; kagglehub will prompt for
    credentials (or read them from ~/.kaggle/kaggle.json /
    KAGGLE_USERNAME + KAGGLE_KEY env vars) the first time it downloads.

    Auto-download only supports April-September 2014: those are the six
    months with Lat/Lon columns (uber-raw-data-{apr..sep}14.csv). The
    Jan-June 2015 file in this dataset uses TLC zone location IDs
    instead of coordinates, so it isn't compatible with this script --
    pass a pre-processed CSV with Lat/Lon via --csv if you need that
    period.

    You can still point at a local file explicitly to skip all of that:
        python plot_uber_pickups.py --csv uber-raw-data-apr14.csv \
            --start "2014-04-01 08:00" --end "2014-04-01 10:00"

If the optional 'contextily' package is installed, a real street-map tile
is drawn under the points. Otherwise the script falls back to a plain
scatter plot bounded by NYC's approximate lat/lon box, which still shows
the shape of the city via point density.
"""

import argparse
import glob
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

# Approximate bounding box for New York City (used for axis limits and
# as a sanity filter to drop stray/garbage coordinates in the raw data).
NYC_BOUNDS = {
    "lon_min": -74.05,
    "lon_max": -73.70,
    "lat_min": 40.55,
    "lat_max": 40.95,
}

# Auto-download only covers these months (the ones with Lat/Lon columns).
MONTH_TO_FILENAME = {
    (2014, 4): "uber-raw-data-apr14.csv",
    (2014, 5): "uber-raw-data-may14.csv",
    (2014, 6): "uber-raw-data-jun14.csv",
    (2014, 7): "uber-raw-data-jul14.csv",
    (2014, 8): "uber-raw-data-aug14.csv",
    (2014, 9): "uber-raw-data-sep14.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Uber NYC pickups within a time window.")
    parser.add_argument("--csv", default=None,
                         help="Path to the uber-raw-data-*.csv file. If omitted or the path "
                              "doesn't exist, the matching monthly file (Apr-Sep 2014 only) "
                              "is downloaded automatically via kagglehub based on --start.")
    parser.add_argument("--start", required=True, help="Window start, e.g. '2014-04-01 08:00'.")
    parser.add_argument("--end", required=True, help="Window end, e.g. '2014-04-01 10:00'.")
    parser.add_argument("--out", default="uber_pickups_map.png", help="Output image file.")
    parser.add_argument("--point-size", type=float, default=2.0, help="Marker size for each pickup.")
    parser.add_argument("--alpha", type=float, default=0.4, help="Marker transparency (0-1).")
    return parser.parse_args()


def resolve_csv_path(csv_arg, start: str) -> str:
    """Return a usable path to a uber-raw-data-*.csv file, downloading the
    dataset via kagglehub if the given path is missing or wasn't given."""
    if csv_arg and os.path.isfile(csv_arg):
        return csv_arg

    if csv_arg:
        print(f"'{csv_arg}' not found, falling back to kagglehub download...")

    start_ts = pd.to_datetime(start)
    key = (start_ts.year, start_ts.month)
    filename = MONTH_TO_FILENAME.get(key)
    if filename is None:
        sys.exit(
            f"--start falls in {start_ts.year}-{start_ts.month:02d}, which auto-download "
            "doesn't support (only April-September 2014 have Lat/Lon columns in this "
            "dataset). Pass a pre-processed CSV with Lat/Lon columns via --csv instead."
        )

    try:
        import kagglehub
    except ImportError:
        sys.exit(
            "No local --csv file found/given, and 'kagglehub' isn't installed to "
            "download it automatically.\n"
            "    pip install kagglehub\n"
            f"or pass an existing file with --csv path/to/{filename}"
        )

    print("Downloading/locating dataset via kagglehub (cached after the first run)...")
    dataset_dir = kagglehub.dataset_download("fivethirtyeight/uber-pickups-in-new-york-city")
    print(f"Dataset available at: {dataset_dir}")

    matches = glob.glob(os.path.join(dataset_dir, "**", filename), recursive=True)
    if not matches:
        sys.exit(
            f"Could not find '{filename}' under {dataset_dir}. "
            "The dataset layout may have changed; check kagglehub's output above."
        )
    return matches[0]


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = {"Date/Time", "Lat", "Lon"}
    missing = required_cols - set(df.columns)
    if missing:
        sys.exit(
            f"Error: CSV is missing expected column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df["Date/Time"] = pd.to_datetime(df["Date/Time"])

    # Drop rows with coordinates clearly outside NYC (bad geocodes).
    df = df[
        df["Lon"].between(NYC_BOUNDS["lon_min"], NYC_BOUNDS["lon_max"])
        & df["Lat"].between(NYC_BOUNDS["lat_min"], NYC_BOUNDS["lat_max"])
    ]
    return df


def filter_time_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    if start_ts >= end_ts:
        sys.exit("Error: --start must be earlier than --end.")
    windowed = df[(df["Date/Time"] >= start_ts) & (df["Date/Time"] <= end_ts)]
    return windowed


def plot_pickups(df: pd.DataFrame, start: str, end: str, out_path: str,
                  point_size: float, alpha: float):
    fig, ax = plt.subplots(figsize=(10, 10))

    if df.empty:
        print("Warning: no pickups found in the given time window.")

    use_basemap = False
    try:
        import contextily as cx
        use_basemap = True
    except ImportError:
        pass

    if use_basemap:
        # contextily expects Web Mercator (EPSG:3857) coordinates.
        import geopandas as gpd

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["Lon"], df["Lat"]),
            crs="EPSG:4326",
        ).to_crs(epsg=3857)

        gdf.plot(ax=ax, markersize=point_size, alpha=alpha, color="crimson")

        # Fixed crop: project the same NYC lon/lat box every run and set
        # it as the axis limits, instead of letting geopandas autoscale
        # to whatever points happen to be in this particular time window
        # (which is what made the map "jump" between different windows).
        corners = gpd.GeoSeries(
            gpd.points_from_xy(
                [NYC_BOUNDS["lon_min"], NYC_BOUNDS["lon_max"]],
                [NYC_BOUNDS["lat_min"], NYC_BOUNDS["lat_max"]],
            ),
            crs="EPSG:4326",
        ).to_crs(epsg=3857)
        ax.set_xlim(corners.x.min(), corners.x.max())
        ax.set_ylim(corners.y.min(), corners.y.max())

        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
        ax.set_axis_off()
    else:
        # Fallback: plain scatter in lat/lon space, bounded to NYC.
        ax.scatter(
            df["Lon"], df["Lat"],
            s=point_size, alpha=alpha, color="crimson", linewidths=0,
        )
        ax.set_xlim(NYC_BOUNDS["lon_min"], NYC_BOUNDS["lon_max"])
        ax.set_ylim(NYC_BOUNDS["lat_min"], NYC_BOUNDS["lat_max"])
        ax.set_aspect(1.3)  # rough correction for lat/lon distortion at NYC's latitude
        ax.set_facecolor("#111111")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        print(
            "Note: 'contextily' (and 'geopandas') not installed, so no street "
            "map tiles are shown. Install them for a real basemap:\n"
            "    pip install contextily geopandas"
        )

    ax.set_title(f"Uber pickups in NYC\n{start} to {end}  (n={len(df)})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


def main():
    args = parse_args()
    csv_path = resolve_csv_path(args.csv, args.start)
    df = load_data(csv_path)
    windowed = filter_time_window(df, args.start, args.end)
    plot_pickups(windowed, args.start, args.end, args.out, args.point_size, args.alpha)


if __name__ == "__main__":
    main()