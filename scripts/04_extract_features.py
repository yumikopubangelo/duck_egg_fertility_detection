"""Extract handcrafted/deep features for AWC training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.features.classical_features import ClassicalFeatureExtractor
from src.features.deep_features import DeepFeatureExtractor
from src.features.fusion import FeatureFusion, FeatureFusionConfig


IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")


def list_images(directory: Path) -> List[Path]:
    files: List[Path] = []
    for pattern in IMAGE_PATTERNS:
        files.extend(directory.glob(pattern))
        files.extend(directory.glob(pattern.upper()))
    return sorted(set(files))


def load_split(data_root: Path, split: str) -> Tuple[List[np.ndarray], List[int]]:
    images: List[np.ndarray] = []
    labels: List[int] = []

    class_dirs = [("infertile", 0), ("fertile", 1)]
    for class_name, label in class_dirs:
        class_dir = data_root / split / class_name
        if not class_dir.exists():
            continue
        for image_path in list_images(class_dir):
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            images.append(img)
            labels.append(label)
    return images, labels


def main():
    parser = argparse.ArgumentParser(description="Extract features for AWC")
    parser.add_argument("--data-root", default="data", help="Dataset root containing train/test folders")
    parser.add_argument("--output-dir", default="data/features", help="Output directory for npy files")
    parser.add_argument("--use-deep", action="store_true", help="Enable deep feature extraction")
    parser.add_argument("--deep-backbone", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained deep backbone")
    parser.add_argument("--pca-components", type=int, default=None, help="Optional PCA components for fused features")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_images, train_labels = load_split(data_root, "train")
    test_images, test_labels = load_split(data_root, "test")

    if not train_images:
        raise RuntimeError(f"Tidak ada data train ditemukan di {data_root / 'train'}")
    if not test_images:
        raise RuntimeError(f"Tidak ada data test ditemukan di {data_root / 'test'}")

    print(f"Train samples: {len(train_images)}")
    print(f"Test samples: {len(test_images)}")

    classical_extractor = ClassicalFeatureExtractor()
    X_train_classical = classical_extractor.extract_batch(train_images)
    X_test_classical = classical_extractor.extract_batch(test_images)

    if args.use_deep:
        deep_extractor = DeepFeatureExtractor(
            backbone=args.deep_backbone,
            pretrained=args.pretrained,
        )
        X_train_deep = deep_extractor.extract_batch(train_images)
        X_test_deep = deep_extractor.extract_batch(test_images)

        fusion = FeatureFusion(
            FeatureFusionConfig(
                standardize=True,
                pca_components=args.pca_components,
                classical_weight=1.0,
                deep_weight=1.0,
            )
        )
        X_train = fusion.fit_transform(X_train_classical, X_train_deep)
        X_test = fusion.transform(X_test_classical, X_test_deep)
    else:
        X_train = X_train_classical
        X_test = X_test_classical

    y_train = np.asarray(train_labels, dtype=np.int64)
    y_test = np.asarray(test_labels, dtype=np.int64)

    np.save(output_dir / "awc_features.npy", X_train)
    np.save(output_dir / "awc_labels.npy", y_train)
    np.save(output_dir / "awc_test_features.npy", X_test)
    np.save(output_dir / "awc_test_labels.npy", y_test)

    print(f"Saved train features: {X_train.shape} -> {output_dir / 'awc_features.npy'}")
    print(f"Saved test features: {X_test.shape} -> {output_dir / 'awc_test_features.npy'}")


if __name__ == "__main__":
    main()
