"""Deep feature extraction utilities."""

from __future__ import annotations

from typing import Iterable, List, Optional

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
except Exception:  # pragma: no cover - graceful fallback when torch is unavailable
    torch = None
    nn = None
    models = None
    transforms = None


class DeepFeatureExtractor:
    """
    Extract deep features from images using torchvision backbones.

    Falls back to compact handcrafted representation if torch/torchvision
    is unavailable in runtime environment.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = False,
        image_size: int = 224,
        device: Optional[str] = None,
    ):
        self.backbone = backbone
        self.pretrained = pretrained
        self.image_size = image_size
        self.device = device or ("cuda" if torch is not None and torch.cuda.is_available() else "cpu")

        self._model = None
        self._transform = None
        self._feature_dim = 64  # fallback dimension

        if torch is not None and models is not None and transforms is not None:
            self._build_model()

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def _build_model(self) -> None:
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
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")

        model.eval()
        model.to(self.device)

        self._model = model
        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

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

    def extract(self, image: np.ndarray) -> np.ndarray:
        if self._model is None or self._transform is None or torch is None:
            return self._fallback_features(image).astype(np.float32)

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        tensor = self._transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self._model(tensor)
            if features.ndim > 2:
                features = torch.flatten(features, 1)
            features = features.squeeze(0).detach().cpu().numpy().astype(np.float32)

        return features

    def extract_batch(self, images: Iterable[np.ndarray]) -> np.ndarray:
        vectors: List[np.ndarray] = [self.extract(img) for img in images]
        if not vectors:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        return np.vstack(vectors).astype(np.float32)
