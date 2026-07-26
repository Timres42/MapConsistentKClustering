"""Core data structures for clustering instances and solutions.

Defines fundamental dataclasses including ClusteringInstance (input problem definition),
Clustering (solution with centers and point assignments), and ClusteringWithMapping
(clustering paired with a center-to-center mapping). Also includes type aliases for
point/center indices and distance functions.

Note: only ClusteringInstance is currently consumed elsewhere (by
drift_classes.py / propagation_setup.py, to describe a timestep's points
before they're serialized to the C++ core). Clustering and
ClusteringWithMapping are not referenced by any other module - the C++ core
(src/include/core.hpp) has its own equivalents and is what actually computes
clusterings now. Kept here for now as the closest thing to a spec of that
data shape; safe to remove if that stops being useful.
"""

from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar, Generic, Optional
import numpy as np

PointIdx = int  # The index of a point in the `points` array, not the index into `centers`.
CenterIdx = int  # The index of a center in the `centers` array, not the global index into `points`.

T = TypeVar("T")  # Type of a single data point.
DistanceFn = Callable[[T, T], float]


@dataclass
class ClusteringInstance(Generic[T]):
    """
    A dataclass to contain the clustering instance information, including the data points,
    number of clusters, and the distance function.
    """

    points: np.ndarray[tuple[int, ...], np.dtype[T]]
    k: int
    distance: DistanceFn
    seed: int
    # Allow rng to be passed in, defaulting to None
    rng: Optional[np.random.Generator] = None
    
    def __post_init__(self) -> None:
        # Only initialize a new RNG if one wasn't explicitly passed in
        if self.rng is None:
            self.rng = np.random.default_rng(self.seed)

@dataclass
class Clustering(Generic[T]):
    """
    A dataclass to contain the clustering information of centers and assignments.
    """
    clust_instance: ClusteringInstance[T]
    centers: np.ndarray  # shape (k,) - indices into `points`
    assignment: np.ndarray  # shape (n,) - assignment[i] = index into `centers`
    n: int = field(init=False)  # number of points
    k: int = field(init=False)  # number of centers


    def __post_init__(self) -> None:
        self.centers = np.asarray(self.centers, dtype=PointIdx)
        self.assignment = np.asarray(self.assignment, dtype=CenterIdx)
        self.n = self.assignment.shape[0]
        self.k = self.centers.shape[0]

        if self.centers.size > 0:
            valid_centers = (self.centers >= 0) & (self.centers < self.n)
            assert bool(np.all(valid_centers)), "Center indices must be valid indices into points"

        if self.assignment.size > 0:
            valid_assign = (self.assignment >= 0) & (self.assignment < self.k)
            assert bool(np.all(valid_assign)), "Assignment indices must be valid indices into centers"

    def center_of(self, p: PointIdx) -> PointIdx:
        """Global index (into `points`) of the center assigned to point p."""
        return self.centers[self.assignment[p]]

    def cluster_sizes(self) -> np.ndarray:
        return np.bincount(self.assignment, minlength=self.k)

    def points_in_cluster(self, i: CenterIdx) -> np.ndarray:
        """Global indices of points assigned to center i (0 <= i < k)."""
        return np.where(self.assignment == i)[0]
    

    def cost(self) -> float:
        """
        Compute the k-median cost of this clustering
        """
        if self.n == 0:
            return 0.0
        points = self.clust_instance.points
        distance = self.clust_instance.distance
        total = 0.0
        for p in range(self.n):
            c = self.center_of(p)
            total += distance(points[p], points[c])
        return total


    def __repr__(self) -> str:
        sizes = self.cluster_sizes() if self.k > 0 else np.array([])

        centers_str = np.array2string(self.centers, separator=", ", threshold=np.inf)

        return (
            f"Clustering(n={self.n}, k={self.k},\n"
            f"  centers={centers_str},\n"
       #     f"  assignment={assignment_str},\n"
            f"  cost={np.round(self.cost(),4)}\n"
            f")"
        )

    def print_assignment(self) -> str:
        sizes = self.cluster_sizes() if self.k > 0 else np.array([])
        size_str = ", ".join(f"{i}:{s}" for i, s in enumerate(sizes))

        centers_str = np.array2string(self.centers, separator=", ", threshold=np.inf)
        assignment_str = np.array2string(self.assignment, separator=", ", threshold=np.inf)

        return (
            f"Clustering(n={self.n}, k={self.k},\n"
            f"  centers={centers_str},\n"
            f"  assignment={assignment_str},\n"
            f"  cost={np.round(self.cost(),4)}\n"
            f")"
        )
    

ClusteringAlgo = Callable[[ClusteringInstance], Clustering]

@dataclass
class ClusteringWithMapping: 
    """
    A dataclass to contain the clustering information of centers and assignments, along with a mapping from historical CenterIdx to new CenterIdx.
    """
    clustering: Clustering
    mapping: np.ndarray  # shape (k,) - mapping from historical CenterIdx to new CenterIdx
    def __repr__(self) -> str:
        mapping_str = np.array2string(self.mapping, separator=", ", threshold=20)
        return (
            f"ClusteringWithMapping(\n"
            f"    clustering={self.clustering!r},\n"
            f"    mapping={mapping_str}\n"
            f")"
        )
__all__ = ["PointIdx", "CenterIdx", "T", "DistanceFn", "ClusteringInstance", "Clustering", "ClusteringAlgo"]
