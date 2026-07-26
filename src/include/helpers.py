"""Distance helper shared by the Python-side setup code.

Only `euclidean` remains here, used by drift_classes.py when building each
timestep's ClusteringInstance. Cost calculation, point-to-center assignment,
and cluster-distance metrics all now live exclusively in the C++ core
(src/include/core.hpp) - they were removed from here as dead code once the
propagation experiment stopped going through Python for those computations.
"""

import numpy as np

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))