# src/preprocessing/preprocessor.py
"""
Main Preprocessing Pipeline untuk Telur Bebek
==============================================
Menggabungkan semua preprocessing steps:
  1. Resize         → Uniform size untuk U-Net
  2. Grayscale      → Konversi RGB ke grayscale
  3. CLAHE          → Enhance kontras lokal
  4. Homomorphic    → Normalisasi pencahayaan
  5. Bilateral      → Denoising, preserve edges

Alur (sesuai disertasi dosen):
  RGB image
    → Resize (256x256)
    → Grayscale
    → CLAHE
    → Homomorphic Filtering
    → Bilateral Denoising
    → Output (256x256, grayscale, preprocessed)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union

from .clahe       import apply_clahe, apply_clahe_he_hybrid
from .homomorphic import apply_homomorphic_fast
from .denoising   import apply_bilateral
from .utils       import (
    validate_image,
    convert_to_grayscale,
    normalize_image,
    compute_enhancement_metrics,
    gamma_correction as apply_gamma_correction,
)
from .advanced    import apply_multi_scale_retinex, apply_illumination_correction, apply_contrast_stretching


class DuckEggPreprocessor:
    """
    Main preprocessor untuk citra candling telur bebek.

    Usage:
        preprocessor = DuckEggPreprocessor()
        
        # Single image (numpy array)
        result = preprocessor.preprocess(img_array)
        
        # Single image (file path)
        result = preprocessor.preprocess_from_path("data/raw/fertile/egg_001.jpg")
        
        # Batch processing
        results = preprocessor.batch_preprocess("data/raw/fertile/")
    """

    def __init__(
        self,
        target_size: tuple         = (256, 256),
        clahe_clip_limit: float    = 2.0,
        clahe_tile_size: tuple     = (8, 8),
        use_homomorphic: bool      = True,
        homo_gamma_low: float      = 0.5,
        homo_gamma_high: float     = 1.5,
        homo_cutoff: int           = 30,
        bilateral_d: int           = 9,
        bilateral_sigma_color: int = 75,
        bilateral_sigma_space: int = 75,
        use_clahe_he_hybrid: bool  = False,
        use_advanced_enhancement: bool = False,
        use_illumination_correction: bool = False,
        use_contrast_stretching: bool = False,
        gamma_correction: float    = None,
    ):
        """
        Initialize preprocessor dengan hyperparameters.

        Args:
            target_size           : Output image size (W, H). Default (256,256) untuk U-Net
            clahe_clip_limit      : CLAHE clip limit (2.0-4.0)
            clahe_tile_size       : CLAHE tile grid size
            use_homomorphic       : Pakai homomorphic filter? (True recommended)
            homo_gamma_low        : Homomorphic low-freq weight (0.3-0.7)
            homo_gamma_high       : Homomorphic high-freq weight (1.2-2.0)
            homo_cutoff           : Homomorphic cutoff frequency
            bilateral_d           : Bilateral filter diameter
            bilateral_sigma_color : Bilateral color sigma
            bilateral_sigma_space : Bilateral space sigma
            use_clahe_he_hybrid   : Pakai CLAHE+HE hybrid? (dari Paper 2)
        """
        self.target_size           = target_size
        self.clahe_clip_limit      = clahe_clip_limit
        self.clahe_tile_size       = clahe_tile_size
        self.use_homomorphic       = use_homomorphic
        self.homo_gamma_low        = homo_gamma_low
        self.homo_gamma_high       = homo_gamma_high
        self.homo_cutoff           = homo_cutoff
        self.bilateral_d           = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.use_clahe_he_hybrid   = use_clahe_he_hybrid
        self.use_advanced_enhancement = use_advanced_enhancement
        self.use_illumination_correction = use_illumination_correction
        self.use_contrast_stretching = use_contrast_stretching
        self.gamma_correction = gamma_correction
        if self.gamma_correction is not None and self.gamma_correction <= 0:
            raise ValueError("gamma_correction harus > 0 jika diaktifkan")

    def _apply_advanced_pipeline(self, image: np.ndarray, steps: Optional[dict] = None) -> np.ndarray:
        """
        Terapkan advanced enhancement sesuai flag yang aktif.

        Args:
            image: Grayscale image.
            steps: dict untuk menyimpan intermediate outputs (optional).

        Returns:
            Enhanced grayscale image.
        """
        processed = image

        if not self.use_advanced_enhancement:
            return processed

        processed = apply_multi_scale_retinex(processed)
        if steps is not None:
            steps["retinex"] = processed.copy()

        if self.use_illumination_correction:
            processed = apply_illumination_correction(processed)
            if steps is not None:
                steps["illumination"] = processed.copy()

        if self.use_contrast_stretching:
            processed = apply_contrast_stretching(processed)
            if steps is not None:
                steps["contrast"] = processed.copy()

        return processed

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline untuk satu image.

        Args:
            image: RGB atau Grayscale image sebagai numpy array

        Returns:
            Preprocessed grayscale image (H, W), uint8
        """
        if image is None:
            raise ValueError("Image is None!")

        # Validate input
        img = validate_image(image)

        # --- Step 1: Resize ---
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)

        # --- Step 2: Convert ke Grayscale ---
        gray = convert_to_grayscale(img)

        # --- Step 3: CLAHE (atau CLAHE-HE hybrid) ---
        if self.use_clahe_he_hybrid:
            enhanced = apply_clahe_he_hybrid(
                gray,
                clip_limit=self.clahe_clip_limit,
                tile_grid_size=self.clahe_tile_size
            )
        else:
            enhanced = apply_clahe(
                gray,
                clip_limit=self.clahe_clip_limit,
                tile_grid_size=self.clahe_tile_size
            )

        # --- Step 4: Homomorphic Filtering (optional) ---
        if self.use_homomorphic:
            homo = apply_homomorphic_fast(
                enhanced,
                gamma_low=self.homo_gamma_low,
                gamma_high=self.homo_gamma_high,
                cutoff=self.homo_cutoff
            )
        else:
            homo = enhanced

        # --- Step 5: Advanced enhancement (optional) ---
        homo = self._apply_advanced_pipeline(homo)

        # --- Step 6: Bilateral Denoising ---
        denoised = apply_bilateral(
            homo,
            d=self.bilateral_d,
            sigma_color=self.bilateral_sigma_color,
            sigma_space=self.bilateral_sigma_space
        )

        # --- Step 7: Gamma correction (optional) ---
        if self.gamma_correction is not None:
            denoised = apply_gamma_correction(denoised, self.gamma_correction)

        # --- Step 8: Final normalization ---
        denoised = normalize_image(denoised)

        return denoised

    def preprocess_from_path(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load image dari file, lalu preprocess.

        Args:
            image_path: Path ke file gambar

        Returns:
            Preprocessed image
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {image_path}")

        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Gagal load image: {image_path}")

        return self.preprocess(img)

    def preprocess_with_steps(self, image: np.ndarray) -> dict:
        """
        Preprocess dan kembalikan SEMUA intermediate steps.
        Bagus untuk visualisasi di notebook / web dashboard.

        Returns:
            dict berisi setiap step preprocessing dan metrics
        """
        if image is None:
            raise ValueError("Image is None!")

        steps = {}

        # Step 1: Validate
        img = validate_image(image)
        steps['original'] = img.copy()

        # Step 2: Resize
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        steps['resized'] = img.copy()

        # Step 3: Grayscale
        gray = convert_to_grayscale(img)
        steps['grayscale'] = gray.copy()

        # Step 4: CLAHE
        if self.use_clahe_he_hybrid:
            enhanced = apply_clahe_he_hybrid(gray, self.clahe_clip_limit, self.clahe_tile_size)
        else:
            enhanced = apply_clahe(gray, self.clahe_clip_limit, self.clahe_tile_size)
        steps['clahe'] = enhanced.copy()

        # Step 5: Homomorphic
        if self.use_homomorphic:
            homo = apply_homomorphic_fast(enhanced, self.homo_gamma_low, self.homo_gamma_high, self.homo_cutoff)
        else:
            homo = enhanced.copy()
        steps['homomorphic'] = homo.copy()

        # Step 6: Advanced enhancement
        homo = self._apply_advanced_pipeline(homo, steps=steps)

        # Step 7: Bilateral
        denoised = apply_bilateral(homo, self.bilateral_d, self.bilateral_sigma_color, self.bilateral_sigma_space)
        steps['denoised'] = denoised.copy()

        # Step 8: Gamma correction
        if self.gamma_correction is not None:
            denoised = apply_gamma_correction(denoised, self.gamma_correction)
            steps['gamma_corrected'] = denoised.copy()

        # Step 9: Final normalization
        final = normalize_image(denoised)
        steps['final'] = final.copy()

        # Calculate metrics
        steps['metrics'] = compute_enhancement_metrics(steps['grayscale'], steps['final'])

        return steps

    def batch_preprocess(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        extensions: tuple = ('.jpg', '.jpeg', '.png'),
        verbose: bool = True
    ) -> list:
        """
        Batch process semua gambar dalam folder.

        Args:
            input_dir  : Folder input (berisi gambar)
            output_dir : Folder output (None = return list, tidak save)
            extensions : File extensions yang diproses
            verbose    : Print progress?

        Returns:
            List of (filename, preprocessed_image) jika output_dir None
            List of output paths jika output_dir diberikan
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory tidak ada: {input_dir}")

        # Cari semua image files
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

        if len(image_files) == 0:
            print(f"⚠️  Tidak ada gambar ditemukan di: {input_dir}")
            return []

        if verbose:
            print(f"📦 Processing {len(image_files)} images dari {input_dir}...")

        results = []

        for i, img_path in enumerate(sorted(image_files)):
            try:
                # Preprocess
                preprocessed = self.preprocess_from_path(img_path)

                if output_dir is not None:
                    # Save ke output folder
                    out_path = Path(output_dir) / img_path.name
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_path), preprocessed)
                    results.append(str(out_path))
                else:
                    results.append((img_path.name, preprocessed))

                if verbose:
                    print(f"  ✓ [{i+1}/{len(image_files)}] {img_path.name}")

            except Exception as e:
                print(f"  ✗ [{i+1}/{len(image_files)}] {img_path.name} - ERROR: {e}")
                continue

        if verbose:
            print(f"\n✅ Done! {len(results)}/{len(image_files)} berhasil diproses.")

        return results


class PreprocessorConfig:
    """
    Config presets yang bisa langsung dipakai.
    """

    @staticmethod
    def default():
        """Default config untuk telur bebek."""
        return DuckEggPreprocessor()

    @staticmethod
    def light():
        """Preprocessing ringan - untuk testing cepat."""
        return DuckEggPreprocessor(
            clahe_clip_limit=1.5,
            use_homomorphic=False,
            bilateral_d=5,
            use_advanced_enhancement=False
        )

    @staticmethod
    def strong():
        """Preprocessing kuat - untuk image kualitas rendah."""
        return DuckEggPreprocessor(
            clahe_clip_limit=4.0,
            use_homomorphic=True,
            homo_gamma_low=0.3,
            homo_gamma_high=2.0,
            bilateral_d=15,
            bilateral_sigma_color=150,
            bilateral_sigma_space=150,
            use_advanced_enhancement=True,
            use_illumination_correction=True,
            use_contrast_stretching=True
        )

    @staticmethod
    def paper2_style():
        """Mengikuti Paper 2 (Saifullah 2019) - CLAHE+HE hybrid."""
        return DuckEggPreprocessor(
            use_homomorphic=False,
            use_clahe_he_hybrid=True,
            clahe_clip_limit=2.0
        )

    @staticmethod
    def advanced():
        """Preprocessing advanced dengan retinex dan illumination correction."""
        return DuckEggPreprocessor(
            clahe_clip_limit=3.0,
            use_homomorphic=True,
            homo_gamma_low=0.4,
            homo_gamma_high=1.8,
            bilateral_d=11,
            bilateral_sigma_color=100,
            bilateral_sigma_space=100,
            use_advanced_enhancement=True,
            use_illumination_correction=True,
            use_contrast_stretching=True
        )

    @staticmethod
    def fast():
        """Preprocessing cepat untuk real-time applications."""
        return DuckEggPreprocessor(
            clahe_clip_limit=2.0,
            use_homomorphic=False,
            bilateral_d=3,
            use_advanced_enhancement=False
        )
