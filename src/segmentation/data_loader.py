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
        fertile_dir: str,
        infertile_dir: str,
        image_size: Tuple[int, int] = (256, 256),
        transform: Optional[transforms.Compose] = None
    ):
        self.fertile_dir = Path(fertile_dir)
        self.infertile_dir = Path(infertile_dir)
        self.image_size = image_size
        self.transform = transform
        
        # Get list of image files from both classes
        fertile_files = sorted(self.fertile_dir.glob('*.jpg')) + sorted(self.fertile_dir.glob('*.png'))
        infertile_files = sorted(self.infertile_dir.glob('*.jpg')) + sorted(self.infertile_dir.glob('*.png'))
        
        # Create dataset with labels: 0 for infertile, 1 for fertile
        self.image_files = []
        self.labels = []
        
        for file in fertile_files:
            self.image_files.append(file)
            self.labels.append(1)  # Fertile
        
        for file in infertile_files:
            self.image_files.append(file)
            self.labels.append(0)  # Infertile
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image_tensor = self.transform(image)
        else:
            # Default transform if none provided
            image_tensor = transforms.ToTensor()(image)
        
        # Get label
        label = self.labels[idx]
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return image_tensor, label_tensor
