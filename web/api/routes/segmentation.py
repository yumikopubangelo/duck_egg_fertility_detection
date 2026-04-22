"""U-Net segmentation inference endpoint — overlay only, no DB storage."""

from __future__ import annotations

import base64
import traceback
from pathlib import Path

import cv2
import numpy as np
from flask import Blueprint, current_app, jsonify, request

segmentation_bp = Blueprint("segmentation", __name__)

_unet_model = None        # lazy-loaded once
_unet_n_classes = 1       # set from checkpoint config at load time
_unet_load_error: str | None = None

# Overlay colours: class_id → BGR
_CLASS_COLORS_MULTI = {
    1: (0,  50, 220),   # red-ish  → vascularization
    2: (0, 200,  60),   # green    → embryo core
}
_CLASS_COLORS_BINARY = {
    1: (0, 200,  60),   # green    → detected egg region
}

IMG_SIZE  = 256
NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _find_model_path(config: dict) -> Path | None:
    root = Path(__file__).resolve().parents[3]
    candidates = [
        Path(config.get("MODEL_FOLDER", "models")) / "unet" / "model.pth",
        root / "models" / "unet" / "model.pth",
        Path(config.get("MODEL_FOLDER", "models")) / "unet" / "unet_best.pth",
        root / "models" / "unet" / "unet_best.pth",
    ]
    for p in candidates:
        resolved = p if p.is_absolute() else root / p
        if resolved.exists() and resolved.stat().st_size > 0:
            return resolved
    return None


def _load_unet(model_path: Path):
    import torch
    from src.segmentation.unet import create_unet_for_eggs

    ckpt = torch.load(str(model_path), map_location="cpu")

    cfg = {}
    if isinstance(ckpt, dict):
        cfg   = ckpt.get("config", {}).get("model", {})
        state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    else:
        state = ckpt

    model = create_unet_for_eggs(
        n_channels=cfg.get("n_channels", 3),
        n_classes=cfg.get("n_classes", 1),
        bilinear=cfg.get("bilinear", True),
        dropout_rate=cfg.get("dropout_rate", 0.5),
        lightweight=cfg.get("lightweight", False),
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    model._n_classes = cfg.get("n_classes", 1)
    return model


def _get_unet():
    global _unet_model, _unet_load_error, _unet_n_classes
    if _unet_model is not None:
        return _unet_model, None

    path = _find_model_path(current_app.config)
    if path is None:
        root = Path(__file__).resolve().parents[3]
        checked = [
            str(root / "models" / "unet" / "model.pth"),
            str(root / "models" / "unet" / "unet_best.pth"),
        ]
        msg = "Model file tidak ditemukan. Dicari di: " + ", ".join(checked)
        _unet_load_error = msg
        return None, msg

    try:
        _unet_model = _load_unet(path)
        _unet_n_classes = getattr(_unet_model, "_n_classes", 1)
        current_app.logger.info(
            f"U-Net loaded from {path} (n_classes={_unet_n_classes})"
        )
        _unet_load_error = None
        return _unet_model, None
    except Exception as exc:
        msg = f"Gagal load model dari {path}: {exc}"
        _unet_load_error = msg
        current_app.logger.exception("Failed to load U-Net")
        return None, msg


def _infer(image_bgr: np.ndarray, model) -> np.ndarray:
    """Return predicted mask [H, W] with integer class indices."""
    import torch

    img = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - NORM_MEAN) / NORM_STD
    t   = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)  # [1,3,H,W]

    n_classes = getattr(model, "_n_classes", 1)

    with torch.no_grad():
        out = model(t)  # [1, n_classes, H, W]
        if n_classes == 1:
            # Binary: sigmoid → threshold at 0.5
            prob = torch.sigmoid(out).squeeze().numpy()
            mask = (prob >= 0.5).astype(np.uint8)
        else:
            mask = out.argmax(dim=1).squeeze().numpy().astype(np.uint8)

    return mask


def _make_overlay(image_bgr: np.ndarray, mask: np.ndarray, n_classes: int = 1, alpha: float = 0.45) -> np.ndarray:
    img_r  = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))
    color  = img_r.copy().astype(np.float32)
    colors = _CLASS_COLORS_BINARY if n_classes == 1 else _CLASS_COLORS_MULTI
    for cls, bgr in colors.items():
        color[mask == cls] = bgr
    blended = cv2.addWeighted(img_r.astype(np.float32), 1 - alpha, color, alpha, 0)
    return blended.astype(np.uint8)


def _to_b64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img_bgr)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def _class_areas(mask: np.ndarray, n_classes: int = 1) -> dict:
    total = mask.size
    if n_classes == 1:
        names = {0: "background", 1: "egg_region"}
        return {
            names[c]: round(float((mask == c).sum()) / total * 100, 2)
            for c in range(2)
        }
    names = {0: "background", 1: "vascularization", 2: "embryo"}
    return {
        names[c]: round(float((mask == c).sum()) / total * 100, 2)
        for c in range(3)
    }


@segmentation_bp.route("/segment", methods=["POST"])
def segment():
    """Run U-Net on an uploaded image, return overlay as base64. Not saved to DB."""
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "No file provided"}), 400

    try:
        data  = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Cannot decode image"}), 400

        model, load_err = _get_unet()
        if model is None:
            return jsonify({"error": load_err or "U-Net model tidak tersedia"}), 503

        n_cls   = getattr(model, "_n_classes", 1)
        mask    = _infer(image, model)
        overlay = _make_overlay(image, mask, n_classes=n_cls)
        resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        if n_cls == 1:
            class_colors = {
                "background": "#1e293b",
                "egg_region": "#16a34a",
            }
        else:
            class_colors = {
                "background":      "#1e293b",
                "vascularization": "#dc2626",
                "embryo":          "#16a34a",
            }

        return jsonify({
            "overlay_b64":  _to_b64(overlay),
            "original_b64": _to_b64(resized),
            "class_areas":  _class_areas(mask, n_classes=n_cls),
            "img_size":     IMG_SIZE,
            "n_classes":    n_cls,
            "class_colors": class_colors,
        }), 200

    except Exception as exc:
        current_app.logger.exception("Segmentation failed")
        return jsonify({"error": str(exc), "trace": traceback.format_exc()}), 500
