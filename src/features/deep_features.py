"""Deep feature extraction utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms
except Exception:  # pragma: no cover - graceful fallback when torch is unavailable
    torch = None
    nn = None
    F = None
    models = None
    transforms = None


class DeepFeatureExtractor:
    """
    Extract deep features from images using torchvision or U-Net bottleneck backbones.

    Falls back to compact handcrafted representation if torch/torchvision
    is unavailable in runtime environment.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = False,
        image_size: int | Tuple[int, int] = 224,
        device: Optional[str] = None,
        checkpoint_path: str | Path | None = None,
        lightweight: bool = True,
        n_channels: int = 3,
        n_classes: int = 3,
        bilinear: bool = True,
        dropout_rate: float = 0.0,
        strict_load: bool = False,
    ):
        self.backbone = backbone
        self.pretrained = pretrained
        self.image_size = image_size
        self.device = device or ("cuda" if torch is not None and torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.lightweight = lightweight
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dropout_rate = dropout_rate
        self.strict_load = strict_load

        self._model = None
        self._transform = None
        self._feature_dim = 64  # fallback dimension
        self._hook_output = None
        self._hook_handle = None

        if torch is not None and transforms is not None:
            self._build_model()

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @staticmethod
    def _image_size_tuple(image_size: int | Tuple[int, int]) -> Tuple[int, int]:
        if isinstance(image_size, tuple):
            return image_size
        return (image_size, image_size)

    def _default_unet_checkpoint(self) -> Path | None:
        root = Path(__file__).resolve().parents[2]
        ckpt_dir = root / "results" / "unet_training" / "checkpoints"
        candidates = []
        if ckpt_dir.exists():
            candidates.extend(sorted(ckpt_dir.glob("best_epoch_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True))
            candidates.extend(sorted(ckpt_dir.glob("final_epoch_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True))
        for candidate in candidates:
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate
        return None

    def _build_model(self) -> None:
        size = self._image_size_tuple(self.image_size)

        if self.backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if self.pretrained else None
            model = models.resnet18(weights=weights)
            self._feature_dim = model.fc.in_features
            model.fc = nn.Identity()
        elif self.backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if self.pretrained else None
            model = models.resnet50(weights=weights)
            self._feature_dim = model.fc.in_features
            model.fc = nn.Identity()
        elif self.backbone == "unet_bottleneck":
            from src.segmentation.unet import create_unet_for_eggs

            model = create_unet_for_eggs(
                n_channels=self.n_channels,
                n_classes=self.n_classes,
                bilinear=self.bilinear,
                dropout_rate=self.dropout_rate,
                lightweight=self.lightweight,
            )
            checkpoint = self.checkpoint_path or self._default_unet_checkpoint()
            if checkpoint is not None and checkpoint.exists() and checkpoint.stat().st_size > 0:
                state = torch.load(checkpoint, map_location=self.device)
                state_dict = state.get("model_state_dict", state)
                model.load_state_dict(state_dict, strict=self.strict_load)
                self.checkpoint_path = checkpoint
            self._feature_dim = 256 if self.lightweight else 512
            self._hook_handle = model.bottleneck.register_forward_hook(self._capture_bottleneck)
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")

        model.eval()
        model.to(self.device)

        self._model = model
        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _capture_bottleneck(self, _module, _inputs, output) -> None:
        if torch is not None:
            self._hook_output = output.detach()

    @staticmethod
    def _fallback_features(image: np.ndarray) -> np.ndarray:
        """Compact fallback descriptor when deep backend is unavailable."""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        vector = gray.astype(np.float32).flatten() / 255.0
        return vector[::16]  # 64 dims

    @staticmethod
    def _to_bgr(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image
        raise ValueError(f"Unsupported image shape: {image.shape}")

    def _mask_from_logits(self, logits: torch.Tensor, original_shape: Tuple[int, int]) -> np.ndarray:
        if logits.ndim != 4:
            return np.zeros(original_shape, dtype=np.uint8)

        if logits.shape[1] == 1:
            mask = (torch.sigmoid(logits)[0, 0] > 0.5).detach().cpu().numpy().astype(np.uint8)
        else:
            labels = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
            mask = (labels > 0).astype(np.uint8)

        if mask.shape != original_shape:
            mask = cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask.astype(np.uint8)

    def extract_with_mask(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if self._model is None or self._transform is None or torch is None:
            return self._fallback_features(image).astype(np.float32), None

        bgr = self._to_bgr(image)
        tensor = self._transform(bgr).unsqueeze(0).to(self.device)

        with torch.no_grad():
            self._hook_output = None
            features = self._model(tensor)
            if self.backbone == "unet_bottleneck":
                if self._hook_output is None:
                    raise RuntimeError("U-Net bottleneck hook did not capture activations")
                pooled = F.adaptive_avg_pool2d(self._hook_output, output_size=1)
                vector = pooled.flatten(1).squeeze(0).detach().cpu().numpy().astype(np.float32)
                mask = self._mask_from_logits(features, original_shape=bgr.shape[:2])
                return vector, mask

            if features.ndim > 2:
                features = torch.flatten(features, 1)
            vector = features.squeeze(0).detach().cpu().numpy().astype(np.float32)
            return vector, None

    def extract(self, image: np.ndarray) -> np.ndarray:
        vector, _ = self.extract_with_mask(image)
        return vector

    def extract_batch(self, images: Iterable[np.ndarray]) -> np.ndarray:
        vectors: List[np.ndarray] = [self.extract(img) for img in images]
        if not vectors:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        return np.vstack(vectors).astype(np.float32)

    def cleanup(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
