"""MobileNetV3 end-to-end baseline for duck egg fertility classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import models, transforms
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    models = None
    transforms = None

from src.segmentation.data_loader import EggDataset


@dataclass
class MobileNetTrainingConfig:
    variant: str = "small"
    pretrained: bool = False
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 4
    num_workers: int = 0
    seed: int = 42
    device: str | None = None


class MobileNetV3Baseline:
    """End-to-end MobileNetV3 binary classifier."""

    def __init__(self, config: MobileNetTrainingConfig | None = None):
        if torch is None or models is None or transforms is None:
            raise RuntimeError("PyTorch and torchvision are required for MobileNetV3Baseline")

        self.config = config or MobileNetTrainingConfig()
        self.device = torch.device(
            self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = self._build_model()
        self.model.to(self.device)
        self.history: Dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def _build_model(self) -> nn.Module:
        variant = self.config.variant.lower()
        if variant == "small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if self.config.pretrained else None
            model = models.mobilenet_v3_small(weights=weights)
        elif variant == "large":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if self.config.pretrained else None
            model = models.mobilenet_v3_large(weights=weights)
        else:
            raise ValueError(f"Unsupported MobileNetV3 variant: {self.config.variant}")

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 2)
        return model

    def _build_transforms(self) -> tuple[transforms.Compose, transforms.Compose]:
        image_size = int(self.config.image_size)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                normalize,
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )
        return train_transform, eval_transform

    def _make_loader(
        self,
        fertile_dir: str,
        infertile_dir: str,
        transform,
        shuffle: bool,
    ) -> DataLoader:
        dataset = EggDataset(
            fertile_dir=fertile_dir,
            infertile_dir=infertile_dir,
            image_size=(self.config.image_size, self.config.image_size),
            transform=transform,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
        )

    @staticmethod
    def _batch_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
        predictions = torch.argmax(logits, dim=1)
        return float((predictions == targets).float().mean().item())

    def fit(
        self,
        train_fertile_dir: str,
        train_infertile_dir: str,
        val_fertile_dir: str,
        val_infertile_dir: str,
    ) -> Dict[str, list[float]]:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        train_transform, eval_transform = self._build_transforms()
        train_loader = self._make_loader(
            train_fertile_dir, train_infertile_dir, train_transform, shuffle=True
        )
        val_loader = self._make_loader(
            val_fertile_dir, val_infertile_dir, eval_transform, shuffle=False
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        for _epoch in range(self.config.epochs):
            self.model.train()
            running_loss = 0.0
            running_acc = 0.0
            sample_count = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.long().to(self.device)

                optimizer.zero_grad()
                logits = self.model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                batch_size = images.size(0)
                running_loss += float(loss.item()) * batch_size
                running_acc += self._batch_accuracy(logits, labels) * batch_size
                sample_count += batch_size

            train_loss = running_loss / max(sample_count, 1)
            train_acc = running_acc / max(sample_count, 1)
            val_loss, val_acc, _, _ = self.evaluate_loader(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.history

    def evaluate_loader(
        self,
        loader: DataLoader,
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        criterion = nn.CrossEntropyLoss()
        self.model.eval()

        running_loss = 0.0
        running_acc = 0.0
        sample_count = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.long().to(self.device)

                logits = self.model(images)
                loss = criterion(logits, labels)
                probs = torch.softmax(logits, dim=1)[:, 1]

                batch_size = images.size(0)
                running_loss += float(loss.item()) * batch_size
                running_acc += self._batch_accuracy(logits, labels) * batch_size
                sample_count += batch_size

                all_probs.append(probs.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())

        y_proba = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0,), dtype=np.float32)
        y_true = np.concatenate(all_labels, axis=0) if all_labels else np.empty((0,), dtype=np.int64)

        loss_value = running_loss / max(sample_count, 1)
        acc_value = running_acc / max(sample_count, 1)
        return loss_value, acc_value, y_true.astype(int), y_proba.astype(np.float32)

    def predict_dataset(
        self,
        fertile_dir: str,
        infertile_dir: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, eval_transform = self._build_transforms()
        loader = self._make_loader(fertile_dir, infertile_dir, eval_transform, shuffle=False)
        _loss, _acc, y_true, y_proba = self.evaluate_loader(loader)
        y_pred = (y_proba >= 0.5).astype(int)
        return y_true, y_pred, y_proba

    def predict_image(self, image: np.ndarray) -> tuple[int, float]:
        """Predict one image and return (label_id, fertile_probability)."""
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        _, eval_transform = self._build_transforms()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = eval_transform(rgb).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
            fertile_prob = float(torch.softmax(logits, dim=1)[0, 1].item())
        return int(fertile_prob >= 0.5), fertile_prob

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": self.config.__dict__,
                "history": self.history,
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "MobileNetV3Baseline":
        source = Path(path)
        checkpoint = torch.load(source, map_location=device or "cpu")
        config = MobileNetTrainingConfig(**checkpoint.get("config", {}))
        if device is not None:
            config.device = device
        model = cls(config)
        model.model.load_state_dict(checkpoint["state_dict"])
        model.history = checkpoint.get("history", model.history)
        model.model.eval()
        return model
