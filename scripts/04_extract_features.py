"""Extract handcrafted/deep features for AWC training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.features.classical_features import ClassicalFeatureExtractor
from src.features.deep_features import DeepFeatureExtractor
from src.features.fusion import FeatureFusion, FeatureFusionConfig
from src.features.hybrid_features import (
    HybridFeatureConfig,
    HybridFeatureExtractor,
    build_classical_metadata,
    save_feature_metadata,
)
from src.utils.config import load_config


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


def _load_unet_defaults(config_path: str, checkpoint_override: str) -> Dict[str, object]:
    defaults: Dict[str, object] = {
        "unet_checkpoint": checkpoint_override,
        "unet_lightweight": True,
        "unet_n_channels": 3,
        "unet_n_classes": 3,
        "unet_bilinear": True,
        "unet_dropout_rate": 0.3,
    }

    cfg_path = Path(config_path)
    if cfg_path.exists():
        config = load_config(cfg_path)
        model_cfg = config.get("model", {})
        defaults.update(
            {
                "unet_checkpoint": checkpoint_override or str(model_cfg.get("unet_checkpoint", "")),
                "unet_lightweight": bool(model_cfg.get("lightweight", True)),
                "unet_n_channels": int(model_cfg.get("n_channels", 3)),
                "unet_n_classes": int(model_cfg.get("n_classes", 3)),
                "unet_bilinear": bool(model_cfg.get("bilinear", True)),
                "unet_dropout_rate": float(model_cfg.get("dropout_rate", 0.3)),
            }
        )

    return defaults


def main():
    parser = argparse.ArgumentParser(description="Extract features for AWC")
    parser.add_argument("--data-root", default="data", help="Dataset root containing train/test folders")
    parser.add_argument("--output-dir", default="data/features", help="Output directory for npy files")
    parser.add_argument("--mode", choices=["classical", "hybrid"], default="classical")
    parser.add_argument("--use-deep", action="store_true", help="Enable deep feature extraction for classical mode")
    parser.add_argument(
        "--deep-backbone",
        default="resnet18",
        choices=["resnet18", "resnet50", "unet_bottleneck"],
    )
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained deep backbone")
    parser.add_argument("--pca-components", type=int, default=None, help="Optional PCA components for fused features")
    parser.add_argument("--metadata-path", default="", help="Optional path to save feature metadata JSON")
    parser.add_argument("--unet-config", default="configs/evaluation_config.yaml")
    parser.add_argument("--unet-checkpoint", default="")
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

    metadata: Dict[str, object]

    if args.mode == "hybrid":
        unet_defaults = _load_unet_defaults(args.unet_config, args.unet_checkpoint)
        extractor = HybridFeatureExtractor(
            HybridFeatureConfig(
                preprocess_input=True,
                include_deep=True,
                deep_backbone="unet_bottleneck",
                unet_checkpoint=str(unet_defaults["unet_checkpoint"]),
                unet_lightweight=bool(unet_defaults["unet_lightweight"]),
                unet_n_channels=int(unet_defaults["unet_n_channels"]),
                unet_n_classes=int(unet_defaults["unet_n_classes"]),
                unet_bilinear=bool(unet_defaults["unet_bilinear"]),
                unet_dropout_rate=float(unet_defaults["unet_dropout_rate"]),
            )
        )
        X_train = extractor.extract_batch(train_images)
        X_test = extractor.extract_batch(test_images)
        metadata = extractor.metadata()
    else:
        classical_extractor = ClassicalFeatureExtractor()
        X_train_classical = classical_extractor.extract_batch(train_images)
        X_test_classical = classical_extractor.extract_batch(test_images)

        if args.use_deep:
            unet_defaults = _load_unet_defaults(args.unet_config, args.unet_checkpoint)
            deep_extractor = DeepFeatureExtractor(
                backbone=args.deep_backbone,
                pretrained=args.pretrained,
                checkpoint_path=unet_defaults["unet_checkpoint"] if args.deep_backbone == "unet_bottleneck" else None,
                lightweight=bool(unet_defaults["unet_lightweight"]),
                n_channels=int(unet_defaults["unet_n_channels"]),
                n_classes=int(unet_defaults["unet_n_classes"]),
                bilinear=bool(unet_defaults["unet_bilinear"]),
                dropout_rate=float(unet_defaults["unet_dropout_rate"]),
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
            feature_names = [f"fused_{idx:03d}" for idx in range(X_train.shape[1])]
            metadata = {
                "mode": "classical_deep_fusion",
                "feature_names": feature_names,
                "group_map": {"Fused Features": list(range(X_train.shape[1]))},
                "feature_count": int(X_train.shape[1]),
                "config": {
                    "deep_backbone": args.deep_backbone,
                    "pretrained": args.pretrained,
                    "pca_components": args.pca_components,
                },
            }
        else:
            X_train = X_train_classical
            X_test = X_test_classical
            metadata = build_classical_metadata()

    y_train = np.asarray(train_labels, dtype=np.int64)
    y_test = np.asarray(test_labels, dtype=np.int64)

    np.save(output_dir / "awc_features.npy", X_train)
    np.save(output_dir / "awc_labels.npy", y_train)
    np.save(output_dir / "awc_test_features.npy", X_test)
    np.save(output_dir / "awc_test_labels.npy", y_test)

    metadata_path = Path(args.metadata_path) if args.metadata_path else output_dir / "feature_metadata.json"
    save_feature_metadata(metadata_path, metadata)

    print(f"Saved train features: {X_train.shape} -> {output_dir / 'awc_features.npy'}")
    print(f"Saved test features: {X_test.shape} -> {output_dir / 'awc_test_features.npy'}")
    print(f"Saved feature metadata: {metadata_path}")


if __name__ == "__main__":
    main()
