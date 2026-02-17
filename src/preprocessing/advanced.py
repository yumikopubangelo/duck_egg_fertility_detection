"""
Advanced image preprocessing methods for egg fertility detection.

This module implements state-of-the-art image enhancement and restoration
techniques specifically optimized for egg candling images.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from scipy import ndimage
from .utils import normalize_image


def apply_retinex_enhancement(image: np.ndarray, sigma: float = 100) -> np.ndarray:
    """
    Apply multi-scale retinex enhancement for improved dynamic range.
    
    Args:
        image: Grayscale image
        sigma: Gaussian blur sigma
        
    Returns:
        Enhanced image
    """
    img_float = np.float32(image) / 255.0
    img_log = np.log1p(img_float)
    
    blur = ndimage.gaussian_filter(img_float, sigma=sigma)
    log_blur = np.log1p(blur)
    
    retinex = img_log - log_blur
    retinex = np.exp(retinex) - 1
    
    return normalize_image(retinex)


def apply_multi_scale_retinex(image: np.ndarray, sigmas: list = [15, 80, 250]) -> np.ndarray:
    """
    Apply multi-scale retinex for better illumination normalization.
    
    Args:
        image: Grayscale image
        sigmas: List of sigma values for different scales
        
    Returns:
        Enhanced image
    """
    img_float = np.float32(image) / 255.0
    img_log = np.log1p(img_float)
    
    retinex = np.zeros_like(img_log)
    
    for sigma in sigmas:
        blur = ndimage.gaussian_filter(img_float, sigma=sigma)
        log_blur = np.log1p(blur)
        retinex += img_log - log_blur
        
    retinex /= len(sigmas)
    retinex = np.exp(retinex) - 1
    
    return normalize_image(retinex)


def apply_wavelet_denoising(image: np.ndarray, level: int = 3, threshold: float = 0.1) -> np.ndarray:
    """
    Apply wavelet-based denoising with edge preservation.
    
    Args:
        image: Grayscale image
        level: Wavelet decomposition level
        threshold: Threshold for noise reduction
        
    Returns:
        Denoised image
    """
    try:
        import pywt
        
        # Wavelet decomposition
        coeffs = pywt.wavedec2(image, 'db4', level=level)
        
        # Apply thresholding
        coeffs_thresholded = []
        for i, coeff in enumerate(coeffs):
            if i == 0:
                coeffs_thresholded.append(coeff)
            else:
                cH, cV, cD = coeff
                cH = soft_threshold(cH, threshold)
                cV = soft_threshold(cV, threshold)
                cD = soft_threshold(cD, threshold)
                coeffs_thresholded.append((cH, cV, cD))
        
        # Reconstruction
        reconstructed = pywt.waverec2(coeffs_thresholded, 'db4')
        
        return normalize_image(reconstructed)
    except ImportError:
        # Fallback to bilateral filter if pywt not available
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)


def soft_threshold(coeffs: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply soft thresholding to wavelet coefficients.
    
    Args:
        coeffs: Wavelet coefficients
        threshold: Threshold value
        
    Returns:
        Thresholded coefficients
    """
    return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)


def apply_dehazing(image: np.ndarray, omega: float = 0.95, t0: float = 0.1) -> np.ndarray:
    """
    Apply single image dehazing algorithm.
    
    Args:
        image: Grayscale image
        omega: Transmission estimation parameter
        t0: Minimum transmission
        
    Returns:
        Dehazed image
    """
    img_float = np.float32(image) / 255.0
    
    # Dark channel prior
    min_channel = np.min(img_float, axis=2) if len(img_float.shape) == 3 else img_float
    
    # Atmospheric light estimation
    A = estimate_atmospheric_light(img_float, min_channel)
    
    # Transmission estimation
    transmission = 1 - omega * min_channel
    
    # Clamp transmission
    transmission = np.maximum(transmission, t0)
    
    # Recover scene radiance
    if len(img_float.shape) == 3:
        dehazed = (img_float - A) / transmission[:, :, np.newaxis] + A
    else:
        dehazed = (img_float - A) / transmission + A
    
    return normalize_image(dehazed)


def estimate_atmospheric_light(image: np.ndarray, min_channel: np.ndarray, percent: float = 0.01) -> float:
    """
    Estimate atmospheric light from dark channel.
    
    Args:
        image: Original image
        min_channel: Dark channel image
        percent: Top percentile for estimation
        
    Returns:
        Atmospheric light value
    """
    img_float = np.float32(image)
    pixels = int(percent * image.shape[0] * image.shape[1])
    
    # Find brightest pixels in dark channel
    sorted_indices = np.argsort(min_channel.flatten())[::-1]
    
    # Get top pixels
    top_indices = sorted_indices[:pixels]
    
    # Estimate atmospheric light from RGB channels
    if len(img_float.shape) == 3:
        A = np.mean(img_float.reshape(-1, 3)[top_indices], axis=0)
        return np.max(A)
    else:
        A = np.mean(img_float.flatten()[top_indices])
        return A


def apply_illumination_correction(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """
    Apply illumination correction using homomorphic filtering.
    
    Args:
        image: Grayscale image
        kernel_size: Gaussian blur kernel size
        
    Returns:
        Corrected image
    """
    img_float = np.float32(image) / 255.0
    
    # Split into illumination and reflectance
    illumination = ndimage.gaussian_filter(img_float, sigma=kernel_size // 2)
    reflectance = img_float / (illumination + 1e-8)
    
    # Normalize illumination
    illumination_normalized = (illumination - illumination.min()) / (illumination.max() - illumination.min() + 1e-8)
    
    # Combine back
    corrected = reflectance * illumination_normalized
    
    return normalize_image(corrected)


def apply_super_resolution(image: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Apply super-resolution using interpolation and sharpening.
    
    Args:
        image: Input image
        scale: Scale factor
        
    Returns:
        Super-resolved image
    """
    # Resize with bicubic interpolation
    height, width = image.shape[:2]
    new_height, new_width = height * scale, width * scale
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Sharpen
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(resized, -1, kernel)
    
    return sharpened


def apply_contrast_stretching(image: np.ndarray, low_percentile: float = 2, high_percentile: float = 98) -> np.ndarray:
    """
    Apply contrast stretching using percentile normalization.
    
    Args:
        image: Grayscale image
        low_percentile: Lower percentile for contrast stretching
        high_percentile: Upper percentile for contrast stretching
        
    Returns:
        Contrast-stretched image
    """
    low_val = np.percentile(image, low_percentile)
    high_val = np.percentile(image, high_percentile)
    
    stretched = (image - low_val) * (255 / (high_val - low_val))
    stretched = np.clip(stretched, 0, 255)
    
    return stretched.astype(np.uint8)


def apply_histogram_specification(image: np.ndarray, target_mean: float = 128, target_std: float = 64) -> np.ndarray:
    """
    Apply histogram specification to normalize image histogram.
    
    Args:
        image: Grayscale image
        target_mean: Target mean intensity
        target_std: Target standard deviation
        
    Returns:
        Histogram-specified image
    """
    img_float = np.float32(image)
    
    # Calculate current statistics
    current_mean = img_float.mean()
    current_std = img_float.std()
    
    # Normalize to target mean and std
    normalized = ((img_float - current_mean) * (target_std / current_std)) + target_mean
    normalized = np.clip(normalized, 0, 255)
    
    return normalized.astype(np.uint8)


def apply_background_subtraction(image: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """
    Apply background subtraction to isolate eggs.
    
    Args:
        image: Grayscale image
        threshold: Background threshold
        
    Returns:
        Image with background subtracted
    """
    # Estimate background using Gaussian blur
    background = cv2.GaussianBlur(image, (21, 21), 0)
    
    # Subtract background
    subtracted = cv2.absdiff(image, background)
    
    # Threshold to binary
    _, mask = cv2.threshold(subtracted, int(threshold * 255), 255, cv2.THRESH_BINARY)
    
    # Apply mask to original image
    result = cv2.bitwise_and(image, mask)
    
    return result
