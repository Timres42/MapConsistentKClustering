"""Individual Household Electric Power Consumption adapter for temporal drift experiments.

Provides HouseholdPowerDataset class for loading the UCI "Individual Household
Electric Power Consumption" dataset (id 235) and slicing it into daily
temporal blocks.

Dataset: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

Note on slicing granularity: the raw data is sampled once per minute over
~4 years (~2M rows), which is far too fine-grained to use directly as
per-minute "slices" the way the Uber/Twitter adapters do. This adapter
therefore buckets rows into daily blocks (same (date) grouping
convention used by UberPickupsDataset), which keeps slice sizes and count
reasonable. Adjust `freq` if a different granularity is needed.
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
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)


class HouseholdPowerDataset(BaseTemporalDataset):
    """
    Adapter for the UCI "Individual Household Electric Power Consumption" dataset.
    Groups minute-level power readings into dayly temporal blocks, using
    every numeric measurement column as a clustering feature.
    """

    def __init__(self, data_path: str = "household_power_consumption.txt",
                 seed: int = 42, freq: str = "D"):
        """
        :param data_path: path to household_power_consumption.txt. If this
                           file doesn't exist, it is downloaded from UCI and
                           written to exactly this path.
        :param seed: random seed
        :param freq: pandas offset alias controlling slice granularity
                      (default "h" = hourly blocks)
        """
        super().__init__(seed)
        self.data_path = data_path
        self.freq = freq
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
            txt_members = [n for n in zf.namelist() if n.lower().endswith('.txt')]
            if not txt_members:
                raise FileNotFoundError(
                    f"No .txt file found inside {DATASET_ZIP_URL}. "
                    f"Archive contents: {zf.namelist()}"
                )
            with zf.open(txt_members[0]) as src, open(self.data_path, 'wb') as dst:
                dst.write(src.read())

        print(f"Saved dataset to '{self.data_path}'.")

    def ensure_downloaded(self) -> None:
        if not os.path.exists(self.data_path):
            self._download()

    def load_slices(self) -> List[np.ndarray]:
        if not os.path.exists(self.data_path):
            self._download()
        if self._cached_slices is not None:
            return self._cached_slices

        print(f"Loading household power consumption dataset from {self.data_path}...")
        df = pd.read_csv(self.data_path, sep=';', na_values=['?'], low_memory=False)

        df.columns = df.columns.str.strip().str.lower()
        df['datetime'] = pd.to_datetime(
            df['date'] + ' ' + df['time'], format='%d/%m/%Y %H:%M:%S'
        )

        # Every column besides date/time/datetime is numeric per the UCI
        # variable table (global_active_power, global_reactive_power,
        # voltage, global_intensity, sub_metering_1/2/3). Select
        # programmatically rather than hardcoding names, so this keeps
        # working if the file's column set changes.
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Rows with missing measurements (~1.25% of the file, per the
        # dataset docs) are dropped rather than imputed, since np.stack
        # requires complete rows and imputation choices are out of scope here.
        df = df.dropna(subset=numeric_cols)

        df['time_block'] = df['datetime'].dt.floor(self.freq)
        grouped = df.groupby('time_block', sort=True)

        slices = []
        for _, block_df in grouped:
            feature_matrix = block_df[numeric_cols].to_numpy(dtype=float)
            if feature_matrix.shape[0] > 0:
                slices.append(feature_matrix)

        print(f"Loaded {len(slices)} sequential '{self.freq}' intervals "
              f"over {len(numeric_cols)} numeric features: {numeric_cols}")
        self._cached_slices = slices
        return slices