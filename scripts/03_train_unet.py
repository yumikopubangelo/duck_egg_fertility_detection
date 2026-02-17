"""
UNet Model Training Script

This script trains the UNet model for egg segmentation using PyTorch.
It includes data loading, model training, validation, and saving checkpoints.

Usage:
    python scripts/03_train_unet.py --config configs/unet_config.yaml

Configuration:
- Data paths and parameters
- Model architecture settings
- Training hyperparameters
- Output directories
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import logging
import sys
import time
from datetime import datetime
import numpy as np
from typing import Dict, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.segmentation.unet import UNet, create_unet_for_eggs, get_loss_function, count_parameters
from src.segmentation.data_loader import EggDataset
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.file_utils import create_directories


class TrainingMetrics:
    """Track training metrics"""
    
    def __init__(self):
        self.losses = []
        self.ious = []
        self.dice_coefficients = []
        self.learning_rates = []
    
    def update(self, loss: float, iou: float, dice: float, lr: float):
        self.losses.append(loss)
        self.ious.append(iou)
        self.dice_coefficients.append(dice)
        self.learning_rates.append(lr)
    
    def get_average(self) -> Dict[str, float]:
        return {
            'loss': np.mean(self.losses),
            'iou': np.mean(self.ious),
            'dice': np.mean(self.dice_coefficients),
            'lr': np.mean(self.learning_rates)
        }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
) -> TrainingMetrics:
    """Train model for one epoch"""
    
    model.train()
    metrics = TrainingMetrics()
    
    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            iou = calculate_iou(outputs, masks)
            dice = calculate_dice_coefficient(outputs, masks)
            lr = get_current_lr(optimizer)
        
        # Update metrics
        metrics.update(loss.item(), iou, dice, lr)
        
        # Log progress
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1} [{batch_idx * len(images)}/{len(dataloader.dataset)}] "
                  f"Loss: {loss.item():.4f}, IoU: {iou:.3f}, Dice: {dice:.3f}, LR: {lr:.6f}")
    
    return metrics


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer
) -> TrainingMetrics:
    """Validate model for one epoch"""
    
    model.eval()
    metrics = TrainingMetrics()
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            iou = calculate_iou(outputs, masks)
            dice = calculate_dice_coefficient(outputs, masks)
            lr = get_current_lr(optimizer)
            
            # Update metrics
            metrics.update(loss.item(), iou, dice, lr)
    
    return metrics


def calculate_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate Intersection over Union (IoU)"""
    pred = torch.sigmoid(pred) > 0.5
    
    intersection = (pred & target.byte()).float().sum()
    union = (pred | target.byte()).float().sum()
    
    if union == 0:
        return 1.0
    
    return (intersection / union).item()


def calculate_dice_coefficient(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate Dice coefficient"""
    pred = torch.sigmoid(pred) > 0.5
    
    intersection = 2.0 * (pred & target.byte()).float().sum()
    union = pred.float().sum() + target.float().sum()
    
    if union == 0:
        return 1.0
    
    return (intersection / union).item()


def get_current_lr(optimizer: optim.Optimizer) -> float:
    """Get current learning rate"""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: TrainingMetrics,
    config: Dict,
    checkpoint_dir: Path
) -> None:
    """Save model checkpoint"""
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics.get_average(),
        'config': config
    }
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"checkpoint_epoch_{epoch+1}_{timestamp}.pth"
    filepath = checkpoint_dir / filename
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='UNet Training Script')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to resume checkpoint')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    log_dir = Path(config['training']['log_dir'])
    create_directories(log_dir)
    logger = setup_logger(log_dir / 'training.log')
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create data transforms
    train_transform = transforms.Compose([
        transforms.Resize(config['data']['image_size']),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(config['data']['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = EggDataset(
        config['data']['train_image_dir'],
        config['data']['train_mask_dir'],
        config['data']['image_size'],
        train_transform
    )
    
    val_dataset = EggDataset(
        config['data']['val_image_dir'],
        config['data']['val_mask_dir'],
        config['data']['image_size'],
        val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # Create model
    model = create_unet_for_eggs(
        n_channels=3,
        n_classes=1,
        bilinear=config['model']['bilinear'],
        dropout_rate=config['model']['dropout_rate'],
        lightweight=config['model']['lightweight']
    )
    
    model = model.to(device)
    logger.info(f"Model created: {model.__class__.__name__}")
    logger.info(f"Model has {count_parameters(model):,} parameters")
    
    # Create loss function and optimizer
    criterion = get_loss_function(config['training']['loss_type'])
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['training']['lr_reduce_factor'],
        patience=config['training']['lr_patience'],
        verbose=True
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        logger.info(f"Resumed from checkpoint: {args.resume} (epoch {start_epoch})")
    
    # Create output directories
    output_dir = Path(config['training']['output_dir'])
    create_directories(output_dir)
    
    checkpoint_dir = output_dir / 'checkpoints'
    create_directories(checkpoint_dir)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
        logger.info(f"{'='*50}")
        
        # Train
        print(f"\n{'='*50}")
        print(f"Training Epoch {epoch + 1}/{config['training']['num_epochs']}")
        print(f"{'='*50}")
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, scheduler)
        
        # Validate
        print(f"\n{'='*50}")
        print(f"Validation Epoch {epoch + 1}/{config['training']['num_epochs']}")
        print(f"{'='*50}")
        
        val_metrics = validate_epoch(model, val_loader, criterion, device, optimizer)
        
        # Log metrics
        avg_train_metrics = train_metrics.get_average()
        avg_val_metrics = val_metrics.get_average()
        
        logger.info(f"Training - Loss: {avg_train_metrics['loss']:.4f}, "
                   f"IoU: {avg_train_metrics['iou']:.3f}, "
                   f"Dice: {avg_train_metrics['dice']:.3f}, "
                   f"LR: {avg_train_metrics['lr']:.6f}")
        
        logger.info(f"Validation - Loss: {avg_val_metrics['loss']:.4f}, "
                   f"IoU: {avg_val_metrics['iou']:.3f}, "
                   f"Dice: {avg_val_metrics['dice']:.3f}, "
                   f"LR: {avg_val_metrics['lr']:.6f}")