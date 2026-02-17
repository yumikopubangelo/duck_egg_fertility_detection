"""
Utility functions for image preprocessing module.

This module provides helper functions for various image processing tasks including:
- Image validation and normalization
- Feature extraction for quality assessment
- Performance metrics calculation
- Visualization utilities
"""

import cv2
import numpy as np
from typing import Optional, Union, Tuple


def validate_image(image: np.ndarray) -> np.ndarray:
    """
    Validate and preprocess input image for consistency.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Validated and normalized image
        
    Raises:
        ValueError: If image is invalid
    """
    if image is None:
        raise ValueError("Image is None!")
        
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(image)}")
        
    if len(image.shape) not in (2, 3):
        raise ValueError(f"Expected 2D (grayscale) or 3D (RGB/BGR) image, got shape {image.shape}")
        
    # Convert to uint8 if necessary
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
    return image


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB/BGR image to grayscale.
    
    Args:
        image: Input RGB/BGR or grayscale image
        
    Returns:
        Grayscale image
    """
    if len(image.shape) == 2:
        return image.copy()
        
    if len(image.shape) == 3:
        # Convert RGB to BGR if necessary (OpenCV uses BGR by default)
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            
    raise ValueError(f"Invalid image shape: {image.shape}")


def normalize_image(image: np.ndarray, target_min: int = 0, target_max: int = 255) -> np.ndarray:
    """
    Normalize image to specified range.
    
    Args:
        image: Input image
        target_min: Minimum value for normalization
        target_max: Maximum value for normalization
        
    Returns:
        Normalized image
    """
    if image.dtype == np.uint8 and target_min == 0 and target_max == 255:
        return image.copy()
        
    min_val = image.min()
    max_val = image.max()
    
    if max_val == min_val:
        return np.full_like(image, target_min, dtype=np.uint8)
        
    normalized = (image - min_val) * (target_max - target_min) / (max_val - min_val) + target_min
    return normalized.astype(np.uint8)


def compute_image_metrics(image: np.ndarray) -> dict:
    """
    Compute various image quality metrics.
    
    Args:
        image: Grayscale image
        
    Returns:
        Dictionary containing quality metrics
    """
    if len(image.shape) != 2:
        image = convert_to_grayscale(image)
        
    metrics = {}
    
    # Basic statistics
    metrics['mean'] = float(image.mean())
    metrics['std'] = float(image.std())
    metrics['min'] = float(image.min())
    metrics['max'] = float(image.max())
    
    # Contrast measures
    metrics['contrast'] = float(metrics['std'])
    metrics['contrast_range'] = metrics['max'] - metrics['min']
    
    # Entropy (information content)
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    hist = hist / (hist.sum() + 1e-8)
    hist = hist[hist > 0]
    metrics['entropy'] = float(-np.sum(hist * np.log2(hist)))
    
    # Sharpness (edge content)
    edges = cv2.Canny(image, 50, 150)
    metrics['sharpness'] = float(edges.mean() / 255)
    
    return metrics


def compute_enhancement_metrics(original: np.ndarray, enhanced: np.ndarray) -> dict:
    """
    Compute enhancement metrics comparing original and enhanced images.
    
    Args:
        original: Original grayscale image
        enhanced: Enhanced grayscale image
        
    Returns:
        Dictionary of enhancement metrics
    """
    original = convert_to_grayscale(original)
    enhanced = convert_to_grayscale(enhanced)
    
    orig_metrics = compute_image_metrics(original)
    enh_metrics = compute_image_metrics(enhanced)
    
    metrics = {}
    
    # Contrast improvement
    metrics['contrast_gain'] = enh_metrics['contrast'] / (orig_metrics['contrast'] + 1e-8)
    metrics['contrast_improvement'] = (enh_metrics['contrast'] - orig_metrics['contrast']) / (orig_metrics['contrast'] + 1e-8)
    
    # Entropy (information gain)
    metrics['entropy_gain'] = enh_metrics['entropy'] / (orig_metrics['entropy'] + 1e-8)
    metrics['entropy_improvement'] = (enh_metrics['entropy'] - orig_metrics['entropy']) / (orig_metrics['entropy'] + 1e-8)
    
    # Sharpness improvement
    metrics['sharpness_gain'] = enh_metrics['sharpness'] / (orig_metrics['sharpness'] + 1e-8)
    
    # Brightness adjustment
    metrics['brightness_change'] = (enh_metrics['mean'] - orig_metrics['mean']) / 255
    
    # PSNR (Peak Signal-to-Noise Ratio)
    mse = np.mean((original - enhanced) ** 2)
    if mse == 0:
        metrics['psnr'] = float('inf')
    else:
        metrics['psnr'] = 10 * np.log10((255 ** 2) / mse)
    
    return metrics


def crop_to_square(image: np.ndarray) -> np.ndarray:
    """
    Crop image to square aspect ratio (center crop).
    
    Args:
        image: Input image
        
    Returns:
        Cropped square image
    """
    height, width = image.shape[:2]
    crop_size = min(height, width)
    
    y_start = (height - crop_size) // 2
    y_end = y_start + crop_size
    x_start = (width - crop_size) // 2
    x_end = x_start + crop_size
    
    return image[y_start:y_end, x_start:x_end]


def resize_image(image: np.ndarray, target_size: Tuple[int, int], method: int = cv2.INTER_AREA) -> np.ndarray:
    """
    Resize image to target dimensions with specified interpolation.
    
    Args:
        image: Input image
        target_size: Target dimensions (width, height)
        method: Interpolation method
        
    Returns:
        Resized image
    """
    return cv2.resize(image, target_size, interpolation=method)


def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    """
    Apply gamma correction to image.
    
    Args:
        image: Input image
        gamma: Gamma value
        
    Returns:
        Gamma-corrected image
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def adaptive_gamma_correction(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Apply adaptive gamma correction based on image histogram.
    
    Args:
        image: Input grayscale image
        clip_limit: CLAHE clip limit
        
    Returns:
        Gamma-corrected image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(image)


def extract_regions_of_interest(image: np.ndarray, min_size: int = 100) -> list:
    """
    Extract regions of interest from image.
    
    Args:
        image: Input grayscale image
        min_size: Minimum size of regions to extract
        
    Returns:
        List of ROI coordinates and masks
    """
    # Threshold image
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract regions
    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_size:
            x, y, w, h = cv2.boundingRect(contour)
            regions.append({
                'bbox': (x, y, w, h),
                'mask': cv2.drawContours(np.zeros_like(image), [contour], -1, 255, -1)
            })
    
    return regions
