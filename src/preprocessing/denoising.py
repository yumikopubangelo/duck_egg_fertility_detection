# src/preprocessing/denoising.py
"""
Denoising Methods untuk Citra Telur Bebek
==========================================
Hilangkan noise TANPA merusak edge (tepi embrio).

Kenapa penting?
  - Candling images sering punya noise (speckle, grain)
  - Gaussian blur: hilangkan noise TAPI edge juga blur
  - Bilateral filter: hilangkan noise, JAGA edge tetap tajam
    → Perfect untuk preserve batas embrio yang tipis!

Methods:
  1. Bilateral Filter (RECOMMENDED) 
     - Terbaik untuk edge preservation
     - Agak lambat tapi worth it
     
  2. Gaussian Blur
     - Cepat, tapi blur semua termasuk edge
     - Pakai sebagai baseline comparison
     
  3. Median Blur
     - Bagus untuk salt-and-pepper noise
     - Kurang baik untuk detail halus
     
  4. Non-Local Means (NLM)
     - Paling bagus hasilnya
     - Paling lambat (for batch processing, consider GPU)

Reference: Suhirman et al. (2022)
"""

import cv2
import numpy as np


def apply_bilateral(image, d=9, sigma_color=75, sigma_space=75):
    """
    Bilateral Filter - RECOMMENDED untuk telur bebek.

    Cara kerja:
      Setiap pixel di-smooth dengan tetangganya HANYA JIKA:
        1. Dekat secara spatial (sigma_space)
        2. Mirip warna/intensitas (sigma_color)
      → Pixel di tepi object = beda intensitas = TIDAK di-smooth
      → Edge tetap tajam!

    Args:
        image       : Grayscale image (H, W)
        d           : Diameter filter kernel (9 = 9x9 pixels)
                      Makin besar = lebih smooth tapi lebih lambat
        sigma_color : Intensitas color variation yang di-blur
                      75 = pixels dengan perbedaan intensitas > 75 TIDAK di-blur
        sigma_space : Spatial distance variation
                      75 = pixels dalam jarak 75px yang mirip warna di-blur

    Returns:
        Denoised image (H, W), uint8
    """
    if image is None:
        raise ValueError("Image is None!")
    if len(image.shape) != 2:
        raise ValueError(f"Butuh grayscale (H,W), dapat: {image.shape}")

    return cv2.bilateralFilter(
        image,
        d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space
    )


def apply_gaussian(image, kernel_size=(5, 5), sigma=0):
    """
    Gaussian Blur - simple denoising.

    Args:
        image       : Grayscale image
        kernel_size : Kernel size (harus ganjil, e.g. (3,3), (5,5), (7,7))
        sigma       : Gaussian standard deviation (0 = auto)

    Returns:
        Blurred image
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)


def apply_median(image, kernel_size=5):
    """
    Median Blur - bagus untuk salt-and-pepper noise.

    Args:
        image       : Grayscale image
        kernel_size : Kernel size (harus ganjil: 3, 5, 7)

    Returns:
        Filtered image
    """
    return cv2.medianBlur(image, kernel_size)


def apply_nlm(image, h=10, template_window=7, search_window=21):
    """
    Non-Local Means Denoising - kualitas terbaik, paling lambat.

    Args:
        image           : Grayscale image
        h               : Filter strength (10 = default, makin besar makin smooth)
        template_window : Patch size untuk comparison (harus ganjil)
        search_window   : Search area size (harus ganjil)

    Returns:
        Denoised image
    """
    return cv2.fastNlMeansDenoising(
        image,
        h=h,
        templateWindowSize=template_window,
        searchWindowSize=search_window
    )


def compare_denoising_methods(image):
    """
    Bandingkan semua denoising methods.
    Pakai di notebook untuk eksperimen.

    Returns:
        dict dengan semua hasil
    """
    return {
        'original'            : image.copy(),
        'bilateral_soft'      : apply_bilateral(image, d=5, sigma_color=50, sigma_space=50),
        'bilateral_medium'    : apply_bilateral(image, d=9, sigma_color=75, sigma_space=75),
        'bilateral_strong'    : apply_bilateral(image, d=15, sigma_color=150, sigma_space=150),
        'gaussian'            : apply_gaussian(image, kernel_size=(5, 5)),
        'median'              : apply_median(image, kernel_size=5),
        'nlm'                 : apply_nlm(image, h=10),
    }