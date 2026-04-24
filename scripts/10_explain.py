"""
XAI Explanation Script
======================
Generates Grad-CAM heatmaps (U-Net) and SHAP feature importance (AWC)
for the duck egg fertility detection system.

Usage:
    python scripts/10_explain.py --images data/preprocessed/test/fertile/IMG_8188.jpg
    python scripts/10_explain.py --images data/preprocessed/test/  # whole folder
    python scripts/10_explain.py --shap-only
    python scripts/10_explain.py --gradcam-only --layer bottleneck

Outputs (saved to results/xai/):
    gradcam/<image_name>_<layer>_<score_mode>.png  -- side-by-side panel
    shap/shap_summary.png                           -- feature importance bar chart
    shap/shap_waterfall_<i>.png                     -- per-sample waterfall
    shap/shap_report.json                           -- machine-readable importances
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("xai", level=logging.INFO)
OUT_DIR = Path("results/xai")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_unet(config: dict, device: str):
    from src.segmentation.unet import create_unet_for_eggs

    model_cfg = config.get("model", {})
    n_channels = model_cfg.get("n_channels", 3)
    n_classes  = model_cfg.get("n_classes", 1)
    bilinear   = model_cfg.get("bilinear", True)
    dropout    = model_cfg.get("dropout_rate", 0.0)
    lightweight = model_cfg.get("lightweight", False)

    model = create_unet_for_eggs(n_channels, n_classes, bilinear, dropout, lightweight)

    # Try multiple checkpoint locations — prefer config path, then newest best-epoch
    ckpt_paths = [
        model_cfg.get("unet_checkpoint", ""),
        "results/unet_training/checkpoints/best_epoch_39_20260423_084746.pth",
        "results/unet_training/checkpoints/checkpoint_epoch_40_20260423_085151.pth",
        "results/unet_training/checkpoints/best_epoch_39_20260421_095504.pth",
        "results/unet_training/checkpoints/final_epoch_1_20260417_133215.pth",
        "models/unet/model.pth",
    ]
    for ckpt in ckpt_paths:
        p = Path(ckpt)
        if p.exists() and p.stat().st_size > 0:
            checkpoint = torch.load(p, map_location=device)
            state = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state, strict=False)
            logger.info(f"U-Net loaded from {p}")
            break
    else:
        logger.warning("No U-Net checkpoint found — using random weights for demo.")

    return model.to(device).eval()


def _load_awc(config: dict):
    from src.clustering.awc import AdaptiveWeightedClustering

    ckpt_paths = [
        config.get("model", {}).get("awc_checkpoint", ""),
        "models/awc/awc_model.pkl",
    ]
    for ckpt in ckpt_paths:
        p = Path(ckpt)
        if p.exists() and p.stat().st_size > 0:
            model = AdaptiveWeightedClustering.load(str(p))
            logger.info(f"AWC loaded from {p}")
            return model
    raise FileNotFoundError("No AWC checkpoint found. Run train pipeline first.")


def _collect_images(paths: list[str]) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            images.extend(f for f in p.rglob("*") if f.suffix.lower() in exts)
        elif p.suffix.lower() in exts:
            images.append(p)
    return sorted(images)


# ---------------------------------------------------------------------------
# Grad-CAM runner
# ---------------------------------------------------------------------------

def run_gradcam(
    model,
    image_paths: list[Path],
    layer: str,
    score_mode: str,
    image_size: int,
    device: str,
    max_images: int = 20,
) -> None:
    from src.xai.gradcam import GradCAMExplainer

    explainer = GradCAMExplainer(model, layer=layer)
    out_dir = OUT_DIR / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = min(len(image_paths), max_images)
    logger.info(f"Grad-CAM: processing {n} images (layer={layer}, mode={score_mode})")

    for i, img_path in enumerate(image_paths[:n]):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            logger.warning(f"Cannot read {img_path}")
            continue

        heatmap, overlay, resized, confidence = explainer.explain_image(
            bgr, score_mode=score_mode, image_size=image_size, device=device
        )

        out_name = f"{img_path.stem}_{layer}_{score_mode}.png"
        out_path = str(out_dir / out_name)
        explainer.save(
            out_path,
            original_bgr=resized,
            heatmap=heatmap,
            overlay=overlay,
            label=img_path.parent.name,   # folder name = fertile/infertile
            confidence=confidence,
        )
        logger.info(f"  [{i+1}/{n}] Saved {out_path}")

    explainer.cleanup()
    logger.info(f"Grad-CAM done. Results in {out_dir}")


# ---------------------------------------------------------------------------
# SHAP runner
# ---------------------------------------------------------------------------

def run_shap(config: dict, awc_model, n_waterfall: int = 3) -> None:
    from src.xai.shap_explainer import SHAPExplainer

    feat_dir = Path("data/features")
    X_train_path = feat_dir / "awc_features.npy"
    X_test_path  = feat_dir / "awc_test_features.npy"
    y_test_path  = feat_dir / "awc_test_labels.npy"

    if not X_train_path.exists():
        logger.error(f"Feature file not found: {X_train_path}. Run step 4 first.")
        return

    X_train = np.load(str(X_train_path))
    X_test  = np.load(str(X_test_path)) if X_test_path.exists() else X_train[:10]

    logger.info(f"SHAP: background={len(X_train)} samples, explaining {len(X_test)} test samples")

    # Build optional feature names from config
    feat_names: list[str] | None = config.get("features", {}).get("feature_names", None)
    if feat_names and len(feat_names) != X_train.shape[1]:
        feat_names = None   # mismatch — use generic names

    explainer = SHAPExplainer(
        model=awc_model,
        feature_names=feat_names,
        background_samples=min(100, len(X_train)),
    )

    logger.info("Fitting KernelSHAP background ...")
    explainer.fit_explainer(X_train)

    logger.info("Computing SHAP values (this may take a few minutes) ...")
    shap_values = explainer.explain(X_test, nsamples=256)

    out_dir = OUT_DIR / "shap"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving summary bar chart ...")
    explainer.save_summary(str(out_dir / "shap_summary.png"))

    for i in range(min(n_waterfall, len(X_test))):
        label = ""
        if y_test_path.exists():
            y = np.load(str(y_test_path))
            label = "fertile" if y[i] == 1 else "infertile"
        explainer.save_waterfall(
            str(out_dir / f"shap_waterfall_{i}.png"),
            sample_idx=i,
            title=f"SHAP Waterfall — sample {i} ({label})",
        )

    explainer.save_report(str(out_dir / "shap_report.json"))

    top = explainer.top_features(k=10)
    logger.info("Top-10 most influential features (fertile prediction):")
    for rank, (name, score) in enumerate(top, 1):
        logger.info(f"  {rank:2d}. {name:<40s} {score:.5f}")

    logger.info(f"SHAP done. Results in {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="XAI: Grad-CAM + SHAP for egg fertility model")
    p.add_argument("--config", default="configs/evaluation_config.yaml")
    p.add_argument("--images", nargs="*", default=[],
                   help="Image files or directories for Grad-CAM")
    p.add_argument("--layer", default="bottleneck",
                   choices=["bottleneck", "down4", "down3", "down2"],
                   help="U-Net layer for Grad-CAM hook (down4 only on full model)")
    p.add_argument("--score-mode", default="fertile",
                   choices=["fertile", "infertile", "mean"],
                   help="Score function for Grad-CAM backprop")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--max-images", type=int, default=20,
                   help="Max images to process for Grad-CAM")
    p.add_argument("--n-waterfall", type=int, default=3,
                   help="Number of SHAP waterfall plots to save")
    p.add_argument("--gradcam-only", action="store_true")
    p.add_argument("--shap-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    config_path = Path(args.config)
    config = load_config(str(config_path)) if config_path.exists() else {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run_gc   = not args.shap_only
    run_shap_flag = not args.gradcam_only

    # ---- Grad-CAM ----
    if run_gc:
        image_paths = _collect_images(args.images) if args.images else _collect_images([
            "data/preprocessed/test/fertile",
            "data/preprocessed/test/infertile",
        ])

        if not image_paths:
            logger.warning("No images found for Grad-CAM. Pass --images <path>.")
        else:
            model = _load_unet(config, device)
            run_gradcam(
                model=model,
                image_paths=image_paths,
                layer=args.layer,
                score_mode=args.score_mode,
                image_size=args.image_size,
                device=device,
                max_images=args.max_images,
            )

    # ---- SHAP ----
    if run_shap_flag:
        try:
            awc_model = _load_awc(config)
            run_shap(config, awc_model, n_waterfall=args.n_waterfall)
        except FileNotFoundError as e:
            logger.error(str(e))


if __name__ == "__main__":
    main()