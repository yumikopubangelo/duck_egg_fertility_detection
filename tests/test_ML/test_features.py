import numpy as np

from src.features.classical_features import ClassicalFeatureExtractor
from src.features.hybrid_features import (
    HybridFeatureConfig,
    HybridFeatureExtractor,
    build_extractor_from_metadata,
    load_feature_metadata,
    save_feature_metadata,
)


def _sample_image() -> np.ndarray:
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:96, :96]
    mask = (yy - 48) ** 2 + (xx - 48) ** 2 <= 22 ** 2
    image[mask] = (180, 220, 255)
    image[30:66, 45:51] = (255, 255, 255)
    return image


def test_classical_group_map_matches_feature_count():
    extractor = ClassicalFeatureExtractor()

    all_indices = []
    for idxs in extractor.group_map.values():
        all_indices.extend(idxs)

    assert len(extractor.feature_names) == 70
    assert "Tekstur GLCM" in extractor.group_map
    assert sorted(all_indices) == list(range(len(extractor.feature_names)))


def test_hybrid_extractor_without_deep_adds_glcm_and_morphology_features():
    extractor = HybridFeatureExtractor(
        HybridFeatureConfig(
            include_deep=False,
            preprocess_input=False,
        )
    )

    vector = extractor.extract(_sample_image())

    assert vector.shape == (len(extractor.feature_names),)
    assert len(extractor.feature_names) > 50
    assert "Tekstur GLCM" in extractor.group_map
    assert "Morfologi Mask" in extractor.group_map
    assert np.isfinite(vector).all()


def test_hybrid_metadata_roundtrip(tmp_path):
    extractor = HybridFeatureExtractor(
        HybridFeatureConfig(
            include_deep=False,
            preprocess_input=False,
        )
    )
    metadata_path = tmp_path / "feature_metadata.json"
    save_feature_metadata(metadata_path, extractor.metadata())

    metadata = load_feature_metadata(metadata_path)
    rebuilt = build_extractor_from_metadata(metadata)

    assert isinstance(rebuilt, HybridFeatureExtractor)
    assert rebuilt.feature_names == extractor.feature_names
