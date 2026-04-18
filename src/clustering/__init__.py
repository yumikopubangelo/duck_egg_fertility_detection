"""Clustering algorithms package for duck egg fertility detection."""

from .awc import AdaptiveWeightedClustering

__all__ = ["AdaptiveWeightedClustering"]

try:
    from .kmeans_baseline import KMeansBaseline  # type: ignore

    __all__.append("KMeansBaseline")
except Exception:
    KMeansBaseline = None  # type: ignore

try:
    from .fuzzy_cmeans import FuzzyCMeans  # type: ignore

    __all__.append("FuzzyCMeans")
except Exception:
    FuzzyCMeans = None  # type: ignore

try:
    from .utils import ClusteringUtils  # type: ignore

    __all__.append("ClusteringUtils")
except Exception:
    ClusteringUtils = None  # type: ignore
