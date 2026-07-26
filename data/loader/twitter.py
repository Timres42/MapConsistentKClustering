"""Twitter Geospatial Data adapter for temporal drift experiments.

Provides TwitterGeospatialDataset class for loading geo-tagged Tweet data
(UCI dataset #1050) and slicing it into per-minute temporal blocks.

Dataset: https://archive.ics.uci.edu/dataset/1050/twitter+geospatial+data
"""

import io
import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from typing import List
from src.include.drift_classes import BaseTemporalDataset

DATASET_ZIP_URL = (
   "https://archive.ics.uci.edu/static/public/1050/twitter+geospatial+data.zip"
)

class TwitterGeospatialDataset(BaseTemporalDataset):
    """
    Adapter for the UCI "Twitter Geospatial Data" dataset.
    Groups geo-tagged Tweets (contiguous US, Jan 12-18 2013) into per-minute
    temporal blocks based on their recorded timestamp.
    """

    # The raw file ships with no header row; columns are documented on the
    # UCI page in this exact order.
    _RAW_COLUMNS = ["longitude", "latitude", "timestamp", "timezone"]

    def __init__(self, data_path: str = "twitter.csv", seed: int = 42):
        super().__init__(seed)
        self.data_path = data_path
        self._cached_slices = None

    def _download(self) -> None:
        """Download the dataset zip from UCI and extract the .txt member to
        exactly self.data_path (creating parent directories if needed)."""
        parent_dir = os.path.dirname(self.data_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        print(f"'{self.data_path}' not found, downloading from {DATASET_ZIP_URL}...")
        with urllib.request.urlopen(DATASET_ZIP_URL) as resp:
            zip_bytes = resp.read()

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            csv_members = [n for n in zf.namelist() if n.lower().endswith('.csv')]
            if not csv_members:
                raise FileNotFoundError(
                    f"No .txt file found inside {DATASET_ZIP_URL}. "
                    f"Archive contents: {zf.namelist()}"
                )
            with zf.open(csv_members[0]) as src, open(self.data_path, 'wb') as dst:
                dst.write(src.read())

        print(f"Saved dataset to '{self.data_path}'.")

    def _load_raw(self) -> pd.DataFrame:
        # Detect whether the file already has a header row (e.g. if the
        # user re-saved it with column names) or is the raw, headerless
        # UCI export.
        probe = pd.read_csv(self.data_path, nrows=1, header=None)
        first_row_is_header = not str(probe.iloc[0, 0]).replace('-', '').replace('.', '').isdigit()

        if first_row_is_header:
            df = pd.read_csv(self.data_path)
            df.columns = df.columns.str.strip().str.lower()
        else:
            df = pd.read_csv(self.data_path, header=None, names=self._RAW_COLUMNS)
        return df

    def load_slices(self) -> List[np.ndarray]:
        if not os.path.exists(self.data_path):
            self._download()
        if self._cached_slices is not None:
            return self._cached_slices

        print(f"Loading Twitter geospatial dataset from {self.data_path}...")
        df = self._load_raw()

        # timestamp is an integer like 20130112000000 = 2013-01-12 00:00:00 CST
        df['datetime'] = pd.to_datetime(df['timestamp'].astype(str), format='%Y%m%d%H%M%S')
        df = df.dropna(subset=['latitude', 'longitude'])

        # Bucket every Tweet into its minute-of-day block. Using raw lon/lat
        # degrees as clustering features here (rather than a projected
        # Cartesian system like the NYC adapter): at global scale a single
        # flat projection introduces heavy distortion near the poles, and
        # US-only degree-space clustering is a reasonable approximation for
        # this dataset's footprint.
        df['minute_block'] = df['datetime'].dt.floor('min')

        grouped = df.groupby('minute_block', sort=True)

        slices = []
        features = ['longitude', 'latitude']
        for _, block_df in grouped:
            feature_matrix = block_df[features].to_numpy(dtype=float)
            if feature_matrix.shape[0] > 0:
                slices.append(feature_matrix)

        print(f"Loaded {len(slices)} sequential per-minute intervals.")
        self._cached_slices = slices
        return slices