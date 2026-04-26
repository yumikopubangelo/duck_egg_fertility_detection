"""
Feature extraction package for duck egg fertility detection.
"""

from .classical_features import ClassicalFeatureExtractor
from .deep_features import DeepFeatureExtractor
from .fusion import FeatureFusion
from .hybrid_features import (
    HybridFeatureConfig,
    HybridFeatureExtractor,
    build_classical_metadata,
    build_default_feature_extractor,
    build_extractor_from_metadata,
    default_feature_metadata_path,
    load_feature_metadata,
    save_feature_metadata,
)
from .utils import FeatureUtils

__all__ = [
    "ClassicalFeatureExtractor",
    "DeepFeatureExtractor",
    "FeatureFusion",
    "FeatureUtils",
    "HybridFeatureConfig",
    "HybridFeatureExtractor",
    "build_classical_metadata",
    "build_default_feature_extractor",
    "build_extractor_from_metadata",
    "default_feature_metadata_path",
    "load_feature_metadata",
    "save_feature_metadata",
]
