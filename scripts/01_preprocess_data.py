"""Batch preprocessing script for duck egg images."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import PreprocessorConfig


PRESET_FACTORIES = {
    "default": PreprocessorConfig.default,
    "light": PreprocessorConfig.light,
    "strong": PreprocessorConfig.strong,
    "paper2": PreprocessorConfig.paper2_style,
    "advanced": PreprocessorConfig.advanced,
    "fast": PreprocessorConfig.fast,
}


def iter_images(input_dir: Path):
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    for pattern in patterns:
        yield from input_dir.rglob(pattern)
        yield from input_dir.rglob(pattern.upper())


def main():
    parser = argparse.ArgumentParser(description="Batch preprocess duck egg images")
    parser.add_argument("--input-dir", required=True, help="Input image root directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for preprocessed images")
    parser.add_argument(
        "--preset",
        default="default",
        choices=sorted(PRESET_FACTORIES.keys()),
        help="Preprocessor preset name",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory tidak ditemukan: {input_dir}")

    preprocessor = PRESET_FACTORIES[args.preset]()
    image_paths = sorted(set(iter_images(input_dir)))

    if not image_paths:
        print(f"Tidak ada image ditemukan di {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    for idx, image_path in enumerate(image_paths, start=1):
        try:
            processed = preprocessor.preprocess_from_path(image_path)
            rel = image_path.relative_to(input_dir)
            out_path = output_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), processed)
            success += 1
            print(f"[{idx}/{len(image_paths)}] OK  {rel}")
        except Exception as exc:
            print(f"[{idx}/{len(image_paths)}] ERR {image_path.name} -> {exc}")

    print(f"Selesai: {success}/{len(image_paths)} gambar berhasil diproses")


if __name__ == "__main__":
    main()
