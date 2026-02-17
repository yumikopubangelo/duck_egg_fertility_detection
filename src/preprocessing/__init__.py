# src/preprocessing/__init__.py
from .preprocessor import DuckEggPreprocessor, PreprocessorConfig
from .clahe        import apply_clahe, apply_clahe_he_hybrid, compare_methods
from .homomorphic  import apply_homomorphic_fast
from .denoising    import apply_bilateral, compare_denoising_methods
from .utils        import validate_image, convert_to_grayscale, normalize_image, compute_image_metrics, compute_enhancement_metrics
from .advanced     import apply_multi_scale_retinex, apply_illumination_correction, apply_contrast_stretching

__all__ = [
    'DuckEggPreprocessor',
    'PreprocessorConfig',
    'apply_clahe',
    'apply_clahe_he_hybrid',
    'apply_homomorphic_fast',
    'apply_bilateral',
    'validate_image',
    'convert_to_grayscale',
    'normalize_image',
    'compute_image_metrics',
    'compute_enhancement_metrics',
    'apply_multi_scale_retinex',
    'apply_illumination_correction',
    'apply_contrast_stretching',
]