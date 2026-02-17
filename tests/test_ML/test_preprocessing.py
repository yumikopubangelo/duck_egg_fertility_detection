"""
Tests for preprocessing module.
"""

import pytest
import cv2
import numpy as np
from src.preprocessing import (
    DuckEggPreprocessor,
    PreprocessorConfig,
    apply_clahe,
    apply_clahe_he_hybrid,
    apply_homomorphic_fast,
    apply_bilateral,
    validate_image,
    convert_to_grayscale,
    normalize_image,
    compute_image_metrics,
    compute_enhancement_metrics,
)


def test_image_validation():
    """Test image validation function."""
    # Create test image
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    validated = validate_image(img)
    assert validated is not None
    assert validated.dtype == np.uint8
    assert validated.shape == (256, 256, 3)
    
    # Test invalid inputs
    with pytest.raises(ValueError):
        validate_image(None)
    
    with pytest.raises(ValueError):
        validate_image("not_an_image")


def test_grayscale_conversion():
    """Test grayscale conversion."""
    # RGB image
    img_rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    gray = convert_to_grayscale(img_rgb)
    assert gray.shape == (256, 256)
    assert gray.dtype == np.uint8
    
    # Grayscale image
    img_gray = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    gray = convert_to_grayscale(img_gray)
    assert gray.shape == (256, 256)
    assert gray.dtype == np.uint8


def test_clahe_apply():
    """Test CLAHE application."""
    img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    
    result = apply_clahe(img)
    assert result.shape == img.shape
    assert result.dtype == img.dtype
    assert np.all(result >= 0) and np.all(result <= 255)
    
    result = apply_clahe_he_hybrid(img)
    assert result.shape == img.shape
    assert result.dtype == img.dtype


def test_homomorphic_apply():
    """Test homomorphic filtering."""
    img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    
    result = apply_homomorphic_fast(img)
    assert result.shape == img.shape
    assert result.dtype == img.dtype


def test_bilateral_apply():
    """Test bilateral filter."""
    img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    
    result = apply_bilateral(img)
    assert result.shape == img.shape
    assert result.dtype == img.dtype


def test_preprocessor_init():
    """Test preprocessor initialization."""
    # Default configuration
    preprocessor = DuckEggPreprocessor()
    assert isinstance(preprocessor, DuckEggPreprocessor)
    
    # Custom configuration
    preprocessor = DuckEggPreprocessor(
        target_size=(512, 512),
        clahe_clip_limit=3.0,
        clahe_tile_size=(16, 16),
        use_homomorphic=False,
        use_clahe_he_hybrid=True,
        use_advanced_enhancement=True
    )
    assert preprocessor.target_size == (512, 512)
    assert preprocessor.clahe_clip_limit == 3.0
    assert preprocessor.clahe_tile_size == (16, 16)
    assert preprocessor.use_homomorphic is False
    assert preprocessor.use_clahe_he_hybrid is True
    assert preprocessor.use_advanced_enhancement is True


def test_preprocessor_config():
    """Test preprocessor config presets."""
    # Default config
    default = PreprocessorConfig.default()
    assert isinstance(default, DuckEggPreprocessor)
    
    # Light config
    light = PreprocessorConfig.light()
    assert isinstance(light, DuckEggPreprocessor)
    
    # Strong config
    strong = PreprocessorConfig.strong()
    assert isinstance(strong, DuckEggPreprocessor)
    
    # Paper2 style
    paper2 = PreprocessorConfig.paper2_style()
    assert isinstance(paper2, DuckEggPreprocessor)
    
    # Advanced config
    advanced = PreprocessorConfig.advanced()
    assert isinstance(advanced, DuckEggPreprocessor)


def test_preprocessor_process():
    """Test preprocessor pipeline."""
    img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    
    preprocessor = DuckEggPreprocessor()
    result = preprocessor.preprocess(img)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (256, 256)
    assert result.dtype == np.uint8
    assert np.all(result >= 0) and np.all(result <= 255)


def test_preprocessor_process_with_steps():
    """Test preprocessor with steps."""
    img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    
    preprocessor = DuckEggPreprocessor()
    steps = preprocessor.preprocess_with_steps(img)
    
    assert 'original' in steps
    assert 'resized' in steps
    assert 'grayscale' in steps
    assert 'clahe' in steps
    assert 'homomorphic' in steps
    assert 'denoised' in steps
    assert 'final' in steps
    assert 'metrics' in steps
    
    assert isinstance(steps['metrics'], dict)
    assert 'contrast_gain' in steps['metrics']
    assert 'entropy_gain' in steps['metrics']
    assert 'sharpness_gain' in steps['metrics']


def test_compute_metrics():
    """Test image metrics calculation."""
    img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    
    metrics = compute_image_metrics(img)
    assert isinstance(metrics, dict)
    assert 'mean' in metrics
    assert 'std' in metrics
    assert 'entropy' in metrics
    assert 'sharpness' in metrics


def test_compute_enhancement_metrics():
    """Test enhancement metrics."""
    img1 = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    
    metrics = compute_enhancement_metrics(img1, img2)
    assert isinstance(metrics, dict)
    assert 'contrast_gain' in metrics
    assert 'entropy_gain' in metrics
    assert 'sharpness_gain' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])