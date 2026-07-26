"""US Wildfire dataset adapter for temporal drift experiments.

Provides USWildfireDataset class for loading and preprocessing wildfire location data
across daily timesteps, with spatial coordinate matching and temporal aggregation.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from src.include.drift_classes import BaseTemporalDataset


class USWildfireDataset(BaseTemporalDataset):
    """
    Adapter for the US Wildfire Dataset (FPA FOD - Fire Program Analysis / Fire-Occurrence Database).
    Groups fire occurrence locations by calendar day, with spatial matching across consecutive days.
    """
    def __init__(self, 
                 db_path: str = "data/wildfire/FPA_FOD_20170508.sqlite",
                 start_date: str = "2006-06-01",
                 end_date: str = "2007-08-31",
                 merge: int = 1,
                 step: int = 1,
                 seed: int = 42):
        """
        Initialize the US Wildfire dataset adapter.
        
        :param db_path: Path to the SQLite database file
        :param start_date: Start date for temporal slice extraction (YYYY-MM-DD)
        :param end_date: End date for temporal slice extraction (YYYY-MM-DD)
        :param merge: Number of days to merge into each slice
        :param step: Step size for for starting point of slices. We first step and then slice. For example, if merge=3 and step=2, we will take days 1-3, then 3-5, then 5-7, etc.
        :param seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.db_path = db_path
        self.start_date = start_date
        self.end_date = end_date
        self.merge = merge
        self.step = step
        self._cached_slices = None
        self._validate_date_range()
    
    def _validate_date_range(self):
        """Validate that start_date and end_date are properly formatted."""
        try:
            self.start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
            self.end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD. Error: {e}")
        
        if self.start_dt > self.end_dt:
            raise ValueError(f"start_date ({self.start_date}) must be before end_date ({self.end_date})")
    
    def _project_coordinates(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """
        Projects Lat/Lon degrees into a flat Cartesian coordinate system (kilometers)
        using the center of the contiguous US as reference.
        
        :param lats: Array of latitude values
        :param lons: Array of longitude values
        :return: Array of shape (n, 2) with (x_km, y_km) coordinates
        """
        # Contiguous US center reference point (roughly Kansas)
        lat_ref = 39.5
        lon_ref = -98.5
        
        # Approximate meters per degree at US latitude
        lat_to_km = 111.05
        lon_to_km = 111.32 * np.cos(np.radians(lat_ref))
        
        # Convert to local kilometer offsets
        x_km = (lons - lon_ref) * lon_to_km
        y_km = (lats - lat_ref) * lat_to_km
        
        return np.column_stack([x_km, y_km])
    
    def load_slices(self) -> List[np.ndarray]:
        """
        Load and return a list of daily fire location matrices [X_1, X_2, ..., X_T].
        Each matrix has shape (n, 2) where columns are (x_km, y_km) projected coordinates.
        
        :return: List of numpy arrays, one per day in the date range
        """
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Wildfire database not found at '{self.db_path}'.")
        
        # If already cached, return immediately
        if self._cached_slices is not None:
            return self._cached_slices
        
        print(f"Loading wildfire data from {self.db_path}...")
        print(f"Date range: {self.start_date} to {self.end_date}")
        
        # Connect to SQLite database
        conn = sqlite3.connect(self.db_path)
        
        # Query: Select fire discovery dates and coordinates within date range
        query = """
        SELECT FIRE_YEAR, DISCOVERY_DOY, LATITUDE, LONGITUDE
        FROM Fires
        WHERE LATITUDE IS NOT NULL
          AND LONGITUDE IS NOT NULL
          AND LATITUDE != 0
          AND LONGITUDE != 0
        ORDER BY FIRE_YEAR, DISCOVERY_DOY
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            raise ValueError("No fire records found in the database for the specified date range.")
        
        print(f"Loaded {len(df)} total fire records from database.")
        
        # Convert DOY (day of year) to actual dates
        # Note: DOY is 1-indexed, so we need to add (DOY-1) days to Jan 1
        df['date'] = pd.to_datetime(
            df['FIRE_YEAR'].astype(str) + '-01-01'
        ) + pd.to_timedelta(df['DISCOVERY_DOY'] - 1, unit='D')
        
        # Filter for the specified date range
        df = df[(df['date'] >= self.start_dt) & (df['date'] <= self.end_dt)]
        
        if df.empty:
            raise ValueError(
                f"No fire records found in the specified date range "
                f"({self.start_date} to {self.end_date})."
            )
        
        print(f"Filtered to {len(df)} fires within date range.")
        
        # Project coordinates to kilometers
        coords_km = self._project_coordinates(df['LATITUDE'].values, df['LONGITUDE'].values)
        df['x_km'] = coords_km[:, 0]
        df['y_km'] = coords_km[:, 1]
        
        # Group by calendar date
        df['calendar_date'] = df['date'].dt.date
        daily_groups = df.groupby('calendar_date', sort=True)
        
        raw_slices = []
        date_index = {}
        
        for calendar_date, day_df in daily_groups:
            feature_matrix = day_df[['x_km', 'y_km']].to_numpy(dtype=float)
            if feature_matrix.shape[0] > 0:
                raw_slices.append(feature_matrix)
                date_index[len(raw_slices) - 1] = calendar_date
        
        print(f"Created {len(raw_slices)} daily intervals from {self.start_date} to {self.end_date}.")
        
        sizes = [len(s) for s in raw_slices]
        print(f"Slice sizes range from {min(sizes)} to {max(sizes)} fire locations (no cropping applied).")
        
        self._cached_slices = raw_slices
        return raw_slices