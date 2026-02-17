# src/preprocessing/clahe.py
"""
CLAHE - Contrast Limited Adaptive Histogram Equalization
=========================================================
Meningkatkan kontras LOKAL pada citra telur bebek.

Kenapa CLAHE bukan HE biasa?
  - HE biasa : enhance kontras GLOBAL → noise ikut di-amplify
  - CLAHE    : enhance per TILE kecil → lebih halus, noise terkontrol
  - Cocok untuk telur bebek yang cangkangnya tebal dan buram

Reference: Suhirman et al. (2022), Saifullah (2019)
"""

import cv2
import numpy as np


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE ke grayscale image.

    Args:
        image          : Grayscale image (H, W), dtype uint8
        clip_limit     : Threshold contrast (2.0-4.0)
                         Makin tinggi = kontras kuat tapi noise banyak
        tile_grid_size : Ukuran grid histogram

    Returns:
        Enhanced grayscale image (H, W), dtype uint8
    """
    if image is None:
        raise ValueError("Image is None. Periksa file path-nya!")
    if len(image.shape) != 2:
        raise ValueError(f"Butuh grayscale (H,W), dapat: {image.shape}")
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    return clahe.apply(image)


def apply_clahe_he_hybrid(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Hybrid CLAHE → HE (dari Paper Saifullah, 2019).

    Urutan: CLAHE dulu → HE setelahnya
      - CLAHE dulu : reduce noise, enhance local contrast
      - HE setelah : spread histogram lebih merata
      → Lebih jernih dari single method
    """
    clahe_result = apply_clahe(image, clip_limit, tile_grid_size)
    return cv2.equalizeHist(clahe_result)


def compare_methods(image):
    """
    Bandingkan semua metode - pakai di notebook untuk eksperimen.
    
    Returns:
        dict semua hasil enhancement
    """
    return {
        'original'        : image.copy(),
        'he_only'         : cv2.equalizeHist(image),
        'clahe_soft'      : apply_clahe(image, clip_limit=1.0),
        'clahe_medium'    : apply_clahe(image, clip_limit=2.0),
        'clahe_strong'    : apply_clahe(image, clip_limit=4.0),
        'clahe_he_hybrid' : apply_clahe_he_hybrid(image, clip_limit=2.0),
    }


def compute_metrics(original, enhanced):
    """
    Hitung metrik kualitas enhancement.

    Returns:
        dict: contrast, entropy, brightness, gain
    """
    def entropy(img):
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-8)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist)))

    return {
        'original_contrast' : float(original.std()),
        'enhanced_contrast' : float(enhanced.std()),
        'original_entropy'  : entropy(original),
        'enhanced_entropy'  : entropy(enhanced),
        'contrast_gain'     : float(enhanced.std()) / (float(original.std()) + 1e-8),
        'entropy_gain'      : entropy(enhanced) / (entropy(original) + 1e-8),
    }