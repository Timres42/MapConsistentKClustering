"""Temporal drift data structures for time-series clustering experiments.

Provides BaseTemporalDataset, the abstract adapter interface implemented by
each loader in data/loader/ (uber, wildfire, household_power, online_retail,
twitter), and TemporalDriftPipeline, which slices a dataset's timesteps into
a sequence of ClusteringInstance objects for src/propagation_setup.py to
serialize and hand to the C++ core. It does not itself compute matchings
between timesteps or clusterings - that (center propagation) happens in
src/propagation.cpp.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# Assuming these exist in your project environment
from src.include.clustering_classes import ClusteringInstance
from src.include.helpers import euclidean


class BaseTemporalDataset(ABC):
    """
    Abstract Base Class acting as an adapter for different data sources.
    Handles data loading and point-to-point temporal matching.
    """
    def __init__(self, seed: int):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def load_slices(self) -> List[np.ndarray]:
        """
        Loads and returns a list of matrices [X_1, X_2, ..., X_T].
        Each matrix must have the exact same shape (n, d).
        """
        pass


class TemporalDriftPipeline:
    """
    Coordinates execution of label-consistent clustering loops over 
    dynamically shifting data slices using a specified dataset adapter.
    """
    def __init__(self, k: int, seed: int, timesteps: int, step: int = 1, merge: int = 1):
        """
        :param k: Number of clusters
        :param seed: Random seed
        :param timesteps: how many slices
        :param step: what step size should be used between the first datablocks of slices
        :param merge: how many datablocks should be merged in a slice
        """
        self.k = k
        self.seed = seed
        self.timesteps = timesteps
        self.step = step
        self.merge = merge

    def execute(self, dataset: BaseTemporalDataset) -> List[ClusteringInstance]:
        """
        Executes sequence: Clusters X_1 -> Tracks targets -> Feeds aligned X_t to next state.
        """
        slices = dataset.load_slices()
        instances: List[ClusteringInstance] = []

        for t, curr_idx in enumerate(range(0, self.timesteps * self.step, self.step)):
            start_idx = curr_idx
            end_idx = curr_idx + self.merge

            if end_idx > len(slices):
                raise IndexError(
                    f"Slice range [{start_idx}:{end_idx}] exceeds available "
                    f"data blocks ({len(slices)}). Reduce timesteps, step, or merge."
                )

            data_points = np.vstack(slices[start_idx:end_idx])
            inst = ClusteringInstance(
                points=data_points,
                k=self.k,
                distance=euclidean,
                seed=self.seed + t,
            )
            instances.append(inst)

        return instances
            
            