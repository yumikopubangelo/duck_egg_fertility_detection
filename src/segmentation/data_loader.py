"""Data loading module for egg classification and segmentation datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _list_images(directory: Path) -> List[Path]:
    files: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
        files.extend(sorted(directory.glob(ext)))
        files.extend(sorted(directory.glob(ext.upper())))
    return sorted(set(files))


class EggDataset(Dataset):
    """
    Flexible dataset:
    - Classification mode: provide `fertile_dir` and `infertile_dir`
    - Segmentation mode: provide `image_dir` and `mask_dir`
    """

    def __init__(
        self,
        fertile_dir: Optional[str] = None,
        infertile_dir: Optional[str] = None,
        image_size: Tuple[int, int] = (256, 256),
        transform: Optional[transforms.Compose] = None,
        image_dir: Optional[str] = None,
        mask_dir: Optional[str] = None,
        joint_transform: Optional[
            Callable[[Image.Image, Image.Image], Tuple[Image.Image, Image.Image]]
        ] = None,
    ):
        self.image_size = image_size
        self.transform = transform
        self.joint_transform = joint_transform
        self.to_tensor = transforms.ToTensor()
        self.mode = "classification"

        self.image_files: List[Path] = []
        self.labels: List[int] = []
        self.samples: List[Tuple[Path, Path]] = []

        if image_dir and mask_dir:
            self.mode = "segmentation"
            self.image_dir = Path(image_dir)
            self.mask_dir = Path(mask_dir)
            self._build_segmentation_samples()
        elif fertile_dir and infertile_dir:
            self.mode = "classification"
            self.fertile_dir = Path(fertile_dir)
            self.infertile_dir = Path(infertile_dir)
            self._build_classification_samples()
        else:
            raise ValueError(
                "Gunakan salah satu mode: "
                "(fertile_dir + infertile_dir) atau (image_dir + mask_dir)."
            )

    def _build_classification_samples(self) -> None:
        fertile_files = _list_images(self.fertile_dir)
        infertile_files = _list_images(self.infertile_dir)

        for file in fertile_files:
            self.image_files.append(file)
            self.labels.append(1)

        for file in infertile_files:
            self.image_files.append(file)
            self.labels.append(0)

    def _build_segmentation_samples(self) -> None:
        image_files = _list_images(self.image_dir)
        if not image_files:
            return

        mask_lookup = {}
        for mask_path in _list_images(self.mask_dir):
            mask_lookup[mask_path.stem] = mask_path

        for img_path in image_files:
            mask_path = mask_lookup.get(img_path.stem)
            if mask_path is not None:
                self.samples.append((img_path, mask_path))

    def __len__(self) -> int:
        if self.mode == "segmentation":
            return len(self.samples)
        return len(self.image_files)

    def _load_image_tensor(self, image: Image.Image) -> torch.Tensor:
        if self.transform:
            return self.transform(image)

        image = image.resize(self.image_size, Image.BILINEAR)
        return self.to_tensor(image)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "classification":
            image_path = self.image_files[idx]
            image = Image.open(image_path).convert("RGB")
            image_tensor = self._load_image_tensor(image)
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image_tensor, label_tensor

        image_path, mask_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.joint_transform is not None:
            image, mask = self.joint_transform(image, mask)

        image_tensor = self._load_image_tensor(image)
        # Resize mask using nearest neighbour to preserve class labels
        mask = mask.resize(self.image_size, Image.NEAREST)
        mask_np = np.array(mask, dtype=np.uint8)

        # Normalize to class indices 0/1/2 — always return LongTensor [H, W]
        # so every sample in a batch has the same shape regardless of class content.
        unique_vals = np.unique(mask_np)
        if set(unique_vals).issubset({0, 255}):
            mask_np = (mask_np > 0).astype(np.int64)
        elif 255 in unique_vals or 128 in unique_vals:
            mask_np = np.where(mask_np >= 200, 2, np.where(mask_np >= 100, 1, 0)).astype(np.int64)
        else:
            mask_np = mask_np.astype(np.int64)

        # Return (H, W) LongTensor — CrossEntropyLoss needs [B, H, W] class indices
        return image_tensor, torch.from_numpy(mask_np)
