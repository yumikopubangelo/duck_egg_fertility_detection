"""Data loading module for egg classification and segmentation datasets."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
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
    ):
        self.image_size = image_size
        self.transform = transform
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

        image_tensor = self._load_image_tensor(image)
        mask = mask.resize(self.image_size, Image.NEAREST)
        mask_tensor = self.to_tensor(mask)
        mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor
