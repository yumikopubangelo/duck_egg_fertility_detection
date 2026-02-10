"""
Feature extraction package for duck egg fertility detection.
"""

from .classical_features import ClassicalFeatureExtractor
from .deep_features import DeepFeatureExtractor
from .fusion import FeatureFusion
from .utils import FeatureUtils

__all__ = [
    "ClassicalFeatureExtractor",
    "DeepFeatureExtractor",
    "FeatureFusion",
    "FeatureUtils"
]
