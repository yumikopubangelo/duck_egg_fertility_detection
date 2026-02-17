"""
Data loading module for egg segmentation datasets.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional


class EggDataset(Dataset):
    """Dataset for egg images and masks"""
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_size: Tuple[int, int] = (256, 256),
        transform: Optional[transforms.Compose] = None
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.transform = transform
        
        # Get list of image files
        self.image_files = sorted(self.image_dir.glob('*.jpg')) + sorted(self.image_dir.glob('*.png'))
        self.mask_files = sorted(self.mask_dir.glob('*.png'))
        
        if len(self.image_files) != len(self.mask_files):
            raise ValueError("Number of images and masks must be equal")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Load mask
        mask_path = self.mask_files[idx]
        mask = Image.open(mask_path).convert('L')  # Grayscale
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        # Convert to tensors
        image_tensor = transforms.ToTensor()(image)
        mask_tensor = transforms.ToTensor()(mask)
        
        # For binary segmentation, convert mask to binary
        mask_tensor = (mask_tensor > 0.5).float()
        
        return image_tensor, mask_tensor
