"""Online Retail dataset adapter for temporal drift experiments.

Provides OnlineRetailDataset class for loading the UCI "Online Retail"
dataset (id 352) and slicing it into daily temporal blocks.

Dataset: https://archive.ics.uci.edu/dataset/352/online+retail

Note on slicing granularity: transactions span 01/12/2010-09/12/2011
(~541K rows, ~373 days), so this adapter buckets rows by calendar day
(same grouping convention used by UberPickupsDataset, just at day rather
than hour granularity given the lower transaction density). Adjust `freq`
if a different granularity is needed.

Note on source file: UCI ships this dataset as "Online Retail.xlsx"
(not CSV), so `openpyxl` must be installed to read it (pip install openpyxl).
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
   "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
)

class OnlineRetailDataset(BaseTemporalDataset):
    """
    Adapter for the UCI "Online Retail" dataset.
    Groups transactions into daily temporal blocks, using every numeric
    column (Quantity, UnitPrice, CustomerID) as a clustering feature.
    """

    def __init__(self, data_path: str = "Online Retail.xlsx",
                 seed: int = 42, freq: str = "D"):
        """
        :param data_path: path to the Online Retail.xlsx file
        :param seed: random seed
        :param freq: pandas offset alias controlling slice granularity
                      (default "D" = daily blocks)
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
            xlsx_members = [n for n in zf.namelist() if n.lower().endswith('.xlsx')]
            if not xlsx_members:
                raise FileNotFoundError(
                    f"No .txt file found inside {DATASET_ZIP_URL}. "
                    f"Archive contents: {zf.namelist()}"
                )
            with zf.open(xlsx_members[0]) as src, open(self.data_path, 'wb') as dst:
                dst.write(src.read())

        print(f"Saved dataset to '{self.data_path}'.")

    def load_slices(self) -> List[np.ndarray]:
        if not os.path.exists(self.data_path):
            self._download()
        if self._cached_slices is not None:
            return self._cached_slices

        print(f"Loading Online Retail dataset from {self.data_path}...")
        if self.data_path.lower().endswith('.csv'):
            df = pd.read_csv(self.data_path, encoding='ISO-8859-1')
        else:
            df = pd.read_excel(self.data_path)

        df.columns = df.columns.str.strip().str.lower()
        df['invoicedate'] = pd.to_datetime(df['invoicedate'])

        # Quantity, UnitPrice, CustomerID are the numeric columns per the
        # UCI variable table; InvoiceNo/StockCode/Description/Country are
        # categorical/ID fields. Select programmatically so this stays
        # correct if the file's column set changes.
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # CustomerID is missing for ~25% of rows (guest/unlinked orders);
        # drop rows with any missing numeric value since np.stack requires
        # complete rows.
        df = df.dropna(subset=numeric_cols)

        df['time_block'] = df['invoicedate'].dt.floor(self.freq)
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