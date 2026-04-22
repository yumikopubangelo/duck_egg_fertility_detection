"""
Automated mask generation pipeline for duck egg segmentation.

Pipeline:
  data/{split}/fertile/   -> data/segmentation/{split}/images/  (copied)
  data/{split}/infertile/    data/segmentation/{split}/masks/   (generated)

Two-stage masking:
  Stage 1 - Egg boundary: Gaussian blur → Otsu → morph close → fill holes
            → largest contour → ellipse fill
  Stage 2 - Embryo region (inside egg): pixels darker than (mean - k*std)
            within egg boundary → morph cleanup

Fertile images  → Stage 2 embryo mask (the fertility indicator)
Infertile images → empty mask (all zeros) by default, or Stage 1 egg mask
                   if --infertile-mask=egg
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.segmentation.postprocessing import (
    constrain_mask_to_roi,
    fill_mask_holes,
    keep_largest_component,
    remove_small_components,
)

# Image enhancement helpers
from src.preprocessing.clahe import apply_clahe


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(directory: Path):
    files = []
    for p in sorted(directory.iterdir()):
        if p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return files


# ---------------------------------------------------------------------------
# Stage 1: egg boundary
# ---------------------------------------------------------------------------

_PROCESS_SIZE = 512  # resize to this before mask computation, then scale back


def _odd_kernel_size(value: int, minimum: int = 3, maximum: int | None = None) -> int:
    value = max(minimum, int(value))
    if value % 2 == 0:
        value += 1
    if maximum is not None and value > maximum:
        value = maximum if maximum % 2 == 1 else maximum - 1
    return max(minimum, value)


def _scaled_kernel_size(
    image_shape: Tuple[int, int],
    ratio: float,
    minimum: int = 3,
    maximum: int | None = None,
) -> int:
    base = int(round(min(image_shape[:2]) * ratio))
    return _odd_kernel_size(base, minimum=minimum, maximum=maximum)


def _resize_for_processing(image_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
    """Downscale to _PROCESS_SIZE on the longest side. Returns (resized, scale)."""
    h, w = image_bgr.shape[:2]
    longest = max(h, w)
    if longest <= _PROCESS_SIZE:
        return image_bgr, 1.0
    scale = _PROCESS_SIZE / longest
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def _scale_mask_back(mask_small: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
    h, w = original_shape[:2]
    return cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)


def detect_egg_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Return a conservative egg-region mask (uint8, 0/255)."""
    orig_shape = image_bgr.shape
    small, _ = _resize_for_processing(image_bgr)

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    a_channel = cv2.GaussianBlur(lab[:, :, 1], (11, 11), 0)
    red_minus_green = cv2.GaussianBlur(
        cv2.subtract(small[:, :, 2], small[:, :, 1]),
        (11, 11),
        0,
    )

    _, a_thresh = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, rg_thresh = cv2.threshold(red_minus_green, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_or(a_thresh, rg_thresh)

    close_size = _scaled_kernel_size(thresh.shape, ratio=0.04, minimum=11, maximum=31)
    open_size = _scaled_kernel_size(thresh.shape, ratio=0.015, minimum=5, maximum=13)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))

    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    egg_mask_small = fill_mask_holes(closed)
    egg_mask_small = keep_largest_component(egg_mask_small)
    egg_mask_small = cv2.morphologyEx(egg_mask_small, cv2.MORPH_OPEN, open_kernel, iterations=1)
    egg_mask_small = fill_mask_holes(egg_mask_small)
    egg_mask_small = keep_largest_component(egg_mask_small)

    min_valid_area = int(thresh.size * 0.02)
    if int(np.count_nonzero(egg_mask_small)) < min_valid_area:
        return np.zeros(orig_shape[:2], dtype=np.uint8)

    return _scale_mask_back(egg_mask_small, orig_shape)


# ---------------------------------------------------------------------------
# Stage 2: embryo region (fertility indicator inside egg)
# ---------------------------------------------------------------------------

def detect_embryo_mask(
    image_bgr: np.ndarray,
    egg_mask: np.ndarray,
    k: float = 0.5,
    min_area_ratio: float = 0.01,
) -> np.ndarray:
    """
    Return mask of dark regions inside the egg (embryo under candling).

    Uses CLAHE + Gaussian blur to better reveal low-contrast embryo core while
    preserving thin vascular structures for a separate detector.

    k        — sensitivity: lower k → more of the egg flagged as embryo
    min_area_ratio — embryo mask is zeroed if area < min_area_ratio * egg_area,
                     which prevents noise from being treated as embryo
    """
    orig_shape = image_bgr.shape
    small, _ = _resize_for_processing(image_bgr)
    egg_mask_small = cv2.resize(
        egg_mask,
        (small.shape[1], small.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    egg_mask_small = keep_largest_component(egg_mask_small)

    # Enhance local contrast to reveal subtle dark regions
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = apply_clahe(gray, clip_limit=2.0, tile_grid_size=(8, 8))
    gray = cv2.GaussianBlur(gray, (11, 11), 0)

    edge_margin_size = _scaled_kernel_size(gray.shape, ratio=0.02, minimum=5, maximum=19)
    edge_margin_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (edge_margin_size, edge_margin_size),
    )
    inner_egg_mask = cv2.erode(egg_mask_small, edge_margin_kernel, iterations=1)
    if not np.any(inner_egg_mask):
        inner_egg_mask = egg_mask_small.copy()

    egg_pixels = gray[inner_egg_mask > 0]
    if len(egg_pixels) == 0:
        return np.zeros_like(gray)

    mean_val = float(np.mean(egg_pixels))
    std_val = float(np.std(egg_pixels))
    threshold = mean_val - k * std_val

    dark = ((gray < threshold) & (inner_egg_mask > 0)).astype(np.uint8) * 255

    # morphological cleanup (use slightly smaller kernels to preserve thin vessels)
    morph_size = _scaled_kernel_size(gray.shape, ratio=0.02, minimum=5, maximum=15)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=1)
    dark = constrain_mask_to_roi(dark, inner_egg_mask)

    egg_area = float(np.sum(egg_mask_small > 0))
    min_component_area = max(32, int(egg_area * min_area_ratio * 0.35))
    dark = remove_small_components(
        dark,
        min_area=min_component_area,
        min_relative_area=0.25,
    )
    dark = constrain_mask_to_roi(dark, egg_mask_small)
    embryo_area = float(np.sum(dark > 0))
    if egg_area > 0 and (embryo_area / egg_area) < min_area_ratio:
        return np.zeros(orig_shape[:2], dtype=np.uint8)

    return _scale_mask_back(dark, orig_shape)


def detect_vascular_mask(
    image_bgr: np.ndarray,
    egg_mask: np.ndarray,
    scale_ratio: float = 0.01,
) -> np.ndarray:
    """
    Detect thin vascular structures inside the egg using CLAHE + black-hat
    morphological filtering. Returns uint8 mask (0/255).
    """
    orig_shape = image_bgr.shape
    small, _ = _resize_for_processing(image_bgr)
    egg_mask_small = cv2.resize(egg_mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    egg_mask_small = keep_largest_component(egg_mask_small)
    if not np.any(egg_mask_small):
        return np.zeros(orig_shape[:2], dtype=np.uint8)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = apply_clahe(gray, clip_limit=2.0, tile_grid_size=(8, 8))

    # Black-hat highlights small dark linear structures (vessels)
    kernel_size = _scaled_kernel_size(gray.shape, ratio=scale_ratio, minimum=7, maximum=31)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Normalize and threshold (Otsu) then constrain to egg
    _, vessel_thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    vessel_thresh = constrain_mask_to_roi(vessel_thresh, egg_mask_small)

    # Clean thin artifacts but keep line-like features
    thin_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    vessel_thresh = cv2.morphologyEx(vessel_thresh, cv2.MORPH_OPEN, thin_kernel, iterations=1)
    vessel_thresh = remove_small_components(vessel_thresh, min_area=8, min_relative_area=0.0)
    vessel_thresh = constrain_mask_to_roi(vessel_thresh, egg_mask_small)

    return _scale_mask_back(vessel_thresh, orig_shape)


# ---------------------------------------------------------------------------
# Visualization overlay
# ---------------------------------------------------------------------------

def make_overlay(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    # Vascularization (class 1) – red tint
    overlay[mask == 1] = (
        overlay[mask == 1] * 0.5 + np.array([0, 0, 200]) * 0.5
    ).astype(np.uint8)
    # Embryo core (class 2) – green tint
    overlay[mask == 2] = (
        overlay[mask == 2] * 0.5 + np.array([0, 200, 0]) * 0.5
    ).astype(np.uint8)
    return overlay


# ---------------------------------------------------------------------------
# Per-split processing
# ---------------------------------------------------------------------------

def process_split(
    split_dir: Path,
    out_images_dir: Path,
    out_masks_dir: Path,
    out_vis_dir: Path | None,
    k: float,
    min_area_ratio: float,
    infertile_mask_mode: str,
    mask_format: str = "multiclass",
) -> Tuple[int, int]:
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir.mkdir(parents=True, exist_ok=True)
    if out_vis_dir is not None:
        out_vis_dir.mkdir(parents=True, exist_ok=True)

    processed = skipped = 0

    for label, is_fertile in [("fertile", True), ("infertile", False)]:
        class_dir = split_dir / label
        if not class_dir.exists():
            continue

        images = list_images(class_dir)
        total = len(images)
        print(f"  [{label}] {total} images")

        for i, img_path in enumerate(images, 1):
            print(f"    {i}/{total} {img_path.name}", end="\r", flush=True)
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                print(f"\n    WARNING: cannot read {img_path.name}, skipping")
                skipped += 1
                continue

            egg_mask = detect_egg_mask(bgr)

            if is_fertile:
                embryo_mask = detect_embryo_mask(bgr, egg_mask, k=k, min_area_ratio=min_area_ratio)
                vascular_mask = detect_vascular_mask(bgr, egg_mask)

                if mask_format == "binary":
                    # legacy single-channel embryo mask
                    mask = embryo_mask
                else:
                    # multiclass: 0=background, 1=vascularization, 2=embryo core
                    multiclass = np.zeros_like(egg_mask, dtype=np.uint8)
                    multiclass[vascular_mask > 0] = 1
                    multiclass[embryo_mask > 0] = 2
                    mask = multiclass
            else:
                if mask_format == "binary":
                    mask = egg_mask if infertile_mask_mode == "egg" else np.zeros_like(egg_mask)
                else:
                    # infertile -> all background in multiclass setting
                    mask = np.zeros_like(egg_mask, dtype=np.uint8)

            dst_image = out_images_dir / img_path.name
            try:
                shutil.copy2(img_path, dst_image)
            except OSError as e:
                print(f"\n    WARNING: copy failed for {img_path.name}: {e}. Falling back to cv2 read/write")
                img_read = cv2.imread(str(img_path))
                if img_read is None:
                    print(f"\n    ERROR: cannot read {img_path.name} with cv2, skipping")
                    skipped += 1
                    continue
                if not cv2.imwrite(str(dst_image), img_read):
                    print(f"\n    ERROR: cv2 failed to write {dst_image}")
                    skipped += 1
                    continue

            dst_mask = out_masks_dir / (img_path.stem + ".png")
            # ensure uint8 writing
            cv2.imwrite(str(dst_mask), mask.astype(np.uint8))

            if out_vis_dir is not None:
                overlay = make_overlay(bgr, mask)
                cv2.imwrite(str(out_vis_dir / img_path.name), overlay)

            processed += 1

        print()  # newline after \r progress

    return processed, skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate segmentation masks from candled egg images"
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Root dir containing {train,val,test}/{fertile,infertile}/ (default: ./data)",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/segmentation",
        help="Root dir for segmentation output (default: ./data/segmentation)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to process (default: train val test)",
    )
    parser.add_argument(
        "--embryo-k",
        type=float,
        default=0.5,
        help=(
            "Sensitivity for embryo detection: pixels darker than "
            "(mean - k*std) inside egg are flagged as embryo. "
            "Lower = more sensitive (default: 0.5)"
        ),
    )
    parser.add_argument(
        "--min-embryo-area",
        type=float,
        default=0.01,
        help=(
            "Min embryo area as fraction of egg area. "
            "Smaller detections are discarded as noise (default: 0.01)"
        ),
    )
    parser.add_argument(
        "--infertile-mask",
        choices=["empty", "egg"],
        default="empty",
        help=(
            "Mask to assign infertile images: "
            "'empty' = all-zero mask (U-Net learns to output nothing), "
            "'egg' = egg boundary mask (default: empty)"
        ),
    )
    parser.add_argument(
        "--mask-format",
        choices=["binary", "multiclass"],
        default="multiclass",
        help=(
            "Output mask format: 'binary' produces single-channel 0/255 embryo masks; "
            "'multiclass' produces 0=background,1=vessels,2=embryo core (default: multiclass)"
        ),
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save mask overlay images to {output_dir}/{split}/viz/ for inspection",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    total_processed = total_skipped = 0

    for split in args.splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"Split '{split}' not found at {split_dir}, skipping.")
            continue

        print(f"\nProcessing split: {split}")
        out_images = output_dir / split / "images"
        out_masks = output_dir / split / "masks"
        out_vis = (output_dir / split / "viz") if args.visualize else None

        n, s = process_split(
            split_dir=split_dir,
            out_images_dir=out_images,
            out_masks_dir=out_masks,
            out_vis_dir=out_vis,
            k=args.embryo_k,
            min_area_ratio=args.min_embryo_area,
            infertile_mask_mode=args.infertile_mask,
            mask_format=args.mask_format,
        )
        total_processed += n
        total_skipped += s
        print(f"  Done: {n} processed, {s} skipped")

    print(f"\nTotal: {total_processed} processed, {total_skipped} skipped")
    print(f"Output: {output_dir.resolve()}")

    if args.visualize:
        print(
            "Visualizations saved — review them before training to tune --embryo-k "
            "and --min-embryo-area if masks look too aggressive or too sparse."
        )


if __name__ == "__main__":
    main()
