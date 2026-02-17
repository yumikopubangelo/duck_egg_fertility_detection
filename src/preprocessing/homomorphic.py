# src/preprocessing/homomorphic.py
"""
Homomorphic Filtering
======================
Normalisasi pencahayaan yang tidak merata pada citra telur bebek.

Kenapa butuh ini?
  - Candling process sering bikin pencahayaan tidak merata
  - Satu sisi terang, satu sisi gelap
  - Homomorphic filter: pisahkan illumination vs reflectance
    → Kurangi illumination variation
    → Enhance reflectance (detail embrio)

Cara kerja:
  1. Log transform (pisahkan illumination & reflectance)
  2. FFT (ke frequency domain)
  3. Apply filter (enhance high-freq, suppress low-freq)
  4. IFFT (balik ke spatial domain)
  5. Exp transform (balik dari log)
"""

import cv2
import numpy as np


def apply_homomorphic(
    image,
    gamma_low=0.5,
    gamma_high=1.5,
    cutoff=30,
    sharpness=2.0
):
    """
    Apply homomorphic filter ke grayscale image.

    Args:
        image      : Grayscale image (H, W), dtype uint8
        gamma_low  : Weight untuk low frequencies (illumination)
                     0.3-0.7 → suppress illumination variations
        gamma_high : Weight untuk high frequencies (reflectance/detail)
                     1.2-2.0 → enhance details
        cutoff     : Frequency cutoff (D0)
                     Makin kecil = makin banyak low-freq yang di-suppress
        sharpness  : Sharpness of filter transition (c)

    Returns:
        Filtered grayscale image (H, W), dtype uint8
    """
    if image is None:
        raise ValueError("Image is None!")
    if len(image.shape) != 2:
        raise ValueError(f"Butuh grayscale (H,W), dapat: {image.shape}")

    rows, cols = image.shape

    # Step 1: Convert to float & apply log transform
    # f(x,y) = i(x,y) * r(x,y)  →  log: ln(i) + ln(r)
    img_float = np.float32(image) + 1.0  # +1 untuk hindari log(0)
    img_log   = np.log(img_float)

    # Step 2: DFT (Discrete Fourier Transform)
    dft        = cv2.dft(img_log, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift  = np.fft.fftshift(dft)

    # Step 3: Build Gaussian high-pass filter
    crow, ccol = rows // 2, cols // 2
    H = _build_homomorphic_filter(
        rows, cols, crow, ccol, cutoff, gamma_low, gamma_high, sharpness
    )

    # Apply filter (multiply in frequency domain)
    H_complex         = np.zeros_like(dft_shift)
    H_complex[:, :, 0] = H
    H_complex[:, :, 1] = H
    filtered_shift = dft_shift * H_complex

    # Step 4: IDFT
    filtered      = np.fft.ifftshift(filtered_shift)
    img_back      = cv2.idft(filtered, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    # Step 5: Exp transform (reverse log)
    img_exp = np.exp(img_back) - 1.0

    # Normalize to 0-255
    img_norm = cv2.normalize(img_exp, None, 0, 255, cv2.NORM_MINMAX)
    result   = np.uint8(img_norm)

    return result


def _build_homomorphic_filter(rows, cols, crow, ccol, D0, gamma_L, gamma_H, c):
    """
    Build Gaussian-based homomorphic filter.

    H(u,v) = (gamma_H - gamma_L) * [1 - exp(-c * D²/D0²)] + gamma_L

    Args:
        rows, cols  : Image dimensions
        crow, ccol  : Center of frequency domain
        D0          : Cutoff frequency
        gamma_L     : Low-frequency weight
        gamma_H     : High-frequency weight
        c           : Sharpness constant

    Returns:
        Filter array (H, W), float32
    """
    H = np.zeros((rows, cols), dtype=np.float32)

    for u in range(rows):
        for v in range(cols):
            D_squared = (u - crow) ** 2 + (v - ccol) ** 2
            H[u, v] = (gamma_H - gamma_L) * (
                1 - np.exp(-c * D_squared / (D0 ** 2 + 1e-8))
            ) + gamma_L

    return H


def apply_homomorphic_fast(image, gamma_low=0.5, gamma_high=1.5, cutoff=30):
    """
    Versi FAST homomorphic filter (vectorized, tanpa for loop).
    Pakai ini untuk production karena JAUH lebih cepat.

    Args:
        image      : Grayscale image
        gamma_low  : Low-freq weight
        gamma_high : High-freq weight
        cutoff     : Cutoff frequency D0

    Returns:
        Filtered image (H, W), uint8
    """
    if image is None or len(image.shape) != 2:
        raise ValueError("Butuh grayscale image (H, W)!")

    rows, cols = image.shape

    # Step 1: Log transform
    img_log = np.log1p(np.float32(image))

    # Step 2: FFT
    fft     = np.fft.fft2(img_log)
    fft_shifted = np.fft.fftshift(fft)

    # Step 3: Gaussian high-pass filter (vectorized)
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows).reshape(-1, 1) - crow
    v = np.arange(cols).reshape(1, -1) - ccol
    D_squared = u ** 2 + v ** 2

    # Homomorphic filter formula
    H = (gamma_high - gamma_low) * (
        1 - np.exp(-D_squared / (2 * cutoff ** 2))
    ) + gamma_low

    # Apply filter
    filtered_shift = fft_shifted * H

    # Step 4: IFFT
    filtered = np.fft.ifftshift(filtered_shift)
    img_back = np.real(np.fft.ifft2(filtered))

    # Step 5: Exp transform
    img_exp  = np.expm1(img_back)

    # Normalize
    img_norm = cv2.normalize(img_exp, None, 0, 255, cv2.NORM_MINMAX)
    result   = np.uint8(img_norm)

    return result