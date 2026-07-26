"""
Plot geo-tagged Tweets that fall inside a given time window over the United States.

Dataset: Twitter Geospatial Data (UCI Machine Learning Repository, id 1050)
https://archive.ics.uci.edu/dataset/1050/twitter+geospatial+data

Expected CSV columns (raw file has no header):
    longitude, latitude, timestamp, timezone
Example row:
    -87.6298,41.8781,20130112081530,2

Usage:
    python plot_twitter_geo_us.py --start "2013-01-12 08:00" --end "2013-01-12 08:05"

    If --csv is omitted (or points to a path that doesn't exist), the raw
    dataset zip is downloaded directly from UCI and cached locally under
    ~/.cache/twitter_geospatial for later runs. No account/credentials
    required (unlike the Uber/kagglehub script) since UCI serves this file
    over a plain public URL.

    You can still point at a local file explicitly to skip all of that:
        python plot_twitter_geo_us.py --csv twitter.csv \
            --start "2013-01-12 08:00" --end "2013-01-12 08:05"

By default, only the contiguous United States is shown, using the same
approximate geographic bounds as the wildfire plotting script. Pass
--include-outlying to use the wider US bounding box covering Alaska, Hawaii,
and Puerto Rico as well.

If contextily/geopandas are installed, a basemap is drawn under the points.
Otherwise the script falls back to a plain longitude/latitude scatter plot.
"""

import argparse
import io
import os
import sys
import zipfile
import urllib.request

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

RAW_COLUMNS = ["longitude", "latitude", "timestamp", "timezone"]
DATASET_ZIP_URL = "https://archive.ics.uci.edu/static/public/1050/twitter+geospatial+data.zip"
CACHE_DIR = os.path.expanduser("~/.cache/twitter_geospatial")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot geo-tagged Tweets within a time window over the United States.")
    parser.add_argument("--csv", default=None,
                         help="Path to the raw twitter.csv file. If omitted or the path "
                              "doesn't exist, the dataset is downloaded automatically from UCI.")
    parser.add_argument("--start", required=True, help="Window start, e.g. '2013-01-12 08:00'.")
    parser.add_argument("--end", required=True, help="Window end, e.g. '2013-01-12 08:05'.")
    parser.add_argument("--out", default="plots/twitter/twitter_us_map.png", help="Output image file.")
    parser.add_argument("--point-size", type=float, default=1.0, help="Marker size for each Tweet.")
    parser.add_argument("--alpha", type=float, default=0.3, help="Marker transparency (0-1).")
    parser.add_argument("--include-outlying", action="store_true",
                         help="Include Alaska, Hawaii, and Puerto Rico (stretches the map).")
    return parser.parse_args()


def resolve_csv_path(csv_arg) -> str:
    """Return a usable path to the raw twitter.csv file, downloading the
    dataset directly from UCI if the given path is missing or wasn't given."""
    if csv_arg and os.path.isfile(csv_arg):
        return csv_arg

    if csv_arg:
        print(f"'{csv_arg}' not found, falling back to direct UCI download...")

    os.makedirs(CACHE_DIR, exist_ok=True)
    cached_csv = os.path.join(CACHE_DIR, "twitter.csv")
    if os.path.isfile(cached_csv):
        return cached_csv

    print(f"Downloading dataset from {DATASET_ZIP_URL} (cached after the first run)...")
    with urllib.request.urlopen(DATASET_ZIP_URL) as resp:
        zip_bytes = resp.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # The archive contains a single CSV; grab whichever file it is.
        csv_members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_members:
            sys.exit(f"No CSV found inside {DATASET_ZIP_URL}. Archive contents: {zf.namelist()}")
        with zf.open(csv_members[0]) as src, open(cached_csv, "wb") as dst:
            dst.write(src.read())

    return cached_csv

def restrict_to_conus(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["longitude"].between(CONUS_BOUNDS["lon_min"], CONUS_BOUNDS["lon_max"])
        & df["latitude"].between(CONUS_BOUNDS["lat_min"], CONUS_BOUNDS["lat_max"])
    ].copy()

def get_bounds(include_outlying: bool) -> dict:
    return ALL_US_BOUNDS if include_outlying else CONUS_BOUNDS

def load_data(csv_path: str) -> pd.DataFrame:
    probe = pd.read_csv(csv_path, nrows=1, header=None)
    first_row_is_header = not str(probe.iloc[0, 0]).replace('-', '').replace('.', '').isdigit()

    if first_row_is_header:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip().str.lower()
    else:
        df = pd.read_csv(csv_path, header=None, names=RAW_COLUMNS)

    df["datetime"] = pd.to_datetime(df["timestamp"].astype(str), format="%Y%m%d%H%M%S")

    # Drop rows with invalid geographic coordinates.
    df = df[
        df["longitude"].between(-180.0, 180.0)
        & df["latitude"].between(-90.0, 90.0)
    ].copy()
    return df


def filter_time_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    if start_ts >= end_ts:
        sys.exit("Error: --start must be earlier than --end.")
    windowed = df[(df["datetime"] >= start_ts) & (df["datetime"] <= end_ts)]
    return windowed


def plot_tweets(df: pd.DataFrame, start: str, end: str, out_path: str,
                 point_size: float, alpha: float, include_outlying: bool):
    fig, ax = plt.subplots(figsize=(12, 8))

    if df.empty:
        print("Warning: no Tweets found in the given time window.")

    use_basemap = False
    try:
        import contextily as cx
        import geopandas as gpd
        use_basemap = True
    except ImportError:
        pass

    bounds = get_bounds(include_outlying)

    if use_basemap:
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs="EPSG:4326",
        ).to_crs(epsg=3857)

        gdf.plot(
            ax=ax,
            markersize=point_size,
            alpha=alpha,
            color="crimson",
        )

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
        ax.scatter(
            df["longitude"],
            df["latitude"],
            s=point_size,
            alpha=alpha,
            color="crimson",
            linewidths=0,
        )

        ax.set_xlim(bounds["lon_min"], bounds["lon_max"])
        ax.set_ylim(bounds["lat_min"], bounds["lat_max"])
        ax.set_aspect(1.3)
        ax.set_facecolor("#111111")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        print(
            "Note: 'contextily' and/or 'geopandas' not installed, so no basemap "
            "tiles are shown. Install them for a real basemap:\n"
            "    pip install contextily geopandas"
        )

    region = "US" if include_outlying else "contiguous US"
    ax.set_title(f"Geo-tagged Tweets — {region}\n{start} to {end}  (n={len(df)})")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")

def main():
    args = parse_args()
    csv_path = resolve_csv_path(args.csv)
    df = load_data(csv_path)
    windowed = filter_time_window(df, args.start, args.end)

    if not args.include_outlying:
        windowed = restrict_to_conus(windowed)
    else:
        bounds = get_bounds(True)
        windowed = windowed[
            windowed["longitude"].between(bounds["lon_min"], bounds["lon_max"])
            & windowed["latitude"].between(bounds["lat_min"], bounds["lat_max"])
        ].copy()

    plot_tweets(
        windowed,
        args.start,
        args.end,
        args.out,
        args.point_size,
        args.alpha,
        args.include_outlying,
    )


if __name__ == "__main__":
    main()