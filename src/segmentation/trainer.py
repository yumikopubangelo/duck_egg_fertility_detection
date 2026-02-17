"""
Training module for U-Net segmentation models.

This module provides functionality for training, validating, and testing U-Net models
for egg segmentation tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import time
import numpy as np
from sklearn.metrics import confusion_matrix

from .unet import UNet, create_unet, count_parameters
from .unet_lightweight import UNetLightWeight
from .data_loader import EggDataset
from .losses import get_loss_function


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UNetTrainer:
    """Trainer class for U-Net segmentation models"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_type: str = 'dice',
        optimizer_type: str = 'adam',
        learning_rate: float = 1e-4,
        device: Optional[torch.device] = None
    ):
        """
        Initialize UNetTrainer.
        
        Args:
            model: U-Net model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            loss_type: Type of loss function to use
            optimizer_type: Type of optimizer to use
            learning_rate: Learning rate for optimizer
            device: Device to use for training (CPU or GPU)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_type = loss_type
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
        
        self.criterion = get_loss_function(loss_type)
        self.optimizer = self._get_optimizer()
        
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_ious: List[float] = []
        self.val_dice_scores: List[float] = []
    
    def _get_optimizer(self) -> optim.Optimizer:
        """Get optimizer based on optimizer type"""
        if self.optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        elif self.optimizer_type == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
    
    def train_epoch(self) -> float:
        """Train the model for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for images, masks in self.train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        average_loss = total_loss / num_batches
        return average_loss
    
    def validate_epoch(self) -> Tuple[float, float, float]:
        """Validate the model for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_ious = []
        all_dice_scores = []
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Calculate metrics
                iou = self._calculate_iou(outputs, masks)
                dice = self._calculate_dice_coefficient(outputs, masks)
                all_ious.append(iou)
                all_dice_scores.append(dice)
        
        average_loss = total_loss / num_batches
        average_iou = np.mean(all_ious)
        average_dice = np.mean(all_dice_scores)
        
        return average_loss, average_iou, average_dice
    
    def _calculate_iou(self, outputs: torch.Tensor, masks: torch.Tensor) -> float:
        """Calculate Intersection over Union (IoU)"""
        outputs = torch.sigmoid(outputs) > 0.5
        masks = masks > 0.5
        
        intersection = (outputs & masks).sum().float()
        union = (outputs | masks).sum().float()
        
        if union == 0:
            return 1.0
        
        return (intersection / union).item()
    
    def _calculate_dice_coefficient(self, outputs: torch.Tensor, masks: torch.Tensor) -> float:
        """Calculate Dice coefficient"""
        outputs = torch.sigmoid(outputs) > 0.5
        masks = masks > 0.5
        
        intersection = 2.0 * (outputs & masks).sum().float()
        union = outputs.sum().float() + masks.sum().float()
        
        if union == 0:
            return 1.0
        
        return (intersection / union).item()
    
    def train(
        self,
        num_epochs: int,
        save_dir: str,
        patience: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of training epochs
            save_dir: Directory to save model checkpoints
            patience: Early stopping patience
            verbose: Whether to print training progress
        
        Returns:
            Training history dictionary
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Train and validate
            train_loss = self.train_epoch()
            val_loss, val_iou, val_dice = self.validate_epoch()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_ious.append(val_iou)
            self.val_dice_scores.append(val_dice)
            
            # Print progress
            if verbose:
                epoch_time = time.time() - epoch_start_time
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f} - "
                    f"Val Loss: {val_loss:.4f} - "
                    f"Val IoU: {val_iou:.4f} - "
                    f"Val Dice: {val_dice:.4f} - "
                    f"Time: {epoch_time:.2f}s"
                )
            
            # Check if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                
                # Save best model
                best_model_path = save_path / 'best_model.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_iou': val_iou,
                    'val_dice': val_dice
                }, best_model_path)
                
                logger.info(f"Best model saved to {best_model_path}")
            else:
                epochs_without_improvement += 1
                
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        total_time = time.time() - start_time
        
        # Save final model
        final_model_path = save_path / 'final_model.pth'
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_iou': val_iou,
            'val_dice': val_dice
        }, final_model_path)
        
        logger.info(f"Training completed in {total_time:.2f} seconds")
        logger.info(f"Final model saved to {final_model_path}")
        
        # Return training history
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_ious': self.val_ious,
            'val_dice_scores': self.val_dice_scores,
            'best_val_loss': best_val_loss,
            'total_epochs': epoch + 1,
            'total_time': total_time
        }
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: DataLoader for test data
        
        Returns:
            Evaluation metrics dictionary
        """
        self.model.eval()
        all_predictions = []
        all_masks = []
        
        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                predictions = torch.sigmoid(outputs) > 0.5
                
                all_predictions.extend(predictions.cpu().numpy())
                all_masks.extend(masks.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.array(all_predictions).flatten()
        all_masks = np.array(all_masks).flatten()
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(all_masks, all_predictions).ravel()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'confusion_matrix': {
                'true_positive': tp,
                'true_negative': tn,
                'false_positive': fp,
                'false_negative': fn
            }
        }


def create_trainer(
    model_config: Dict,
    data_config: Dict,
    training_config: Dict
) -> UNetTrainer:
    """
    Create a UNetTrainer instance from configuration dictionaries.
    
    Args:
        model_config: Model configuration
        data_config: Data configuration
        training_config: Training configuration
    
    Returns:
        UNetTrainer instance
    """
    # Create model
    model = create_unet(
        n_channels=model_config.get('n_channels', 3),
        n_classes=model_config.get('n_classes', 1),
        bilinear=model_config.get('bilinear', True),
        dropout_rate=model_config.get('dropout_rate', 0.0),
        lightweight=model_config.get('lightweight', False)
    )
    
    logger.info(f"Model created with {count_parameters(model):,} trainable parameters")
    
    # Create data loaders
    transform = transforms.Compose([
        transforms.Resize(data_config.get('image_size', (256, 256))),
        transforms.ToTensor()
    ])
    
    train_dataset = EggDataset(
        image_dir=data_config['train_image_dir'],
        mask_dir=data_config['train_mask_dir'],
        image_size=data_config.get('image_size', (256, 256)),
        transform=transform
    )
    
    val_dataset = EggDataset(
        image_dir=data_config['val_image_dir'],
        mask_dir=data_config['val_mask_dir'],
        image_size=data_config.get('image_size', (256, 256)),
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.get('batch_size', 16),
        shuffle=True,
        num_workers=training_config.get('num_workers', 0)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.get('batch_size', 16),
        shuffle=False,
        num_workers=training_config.get('num_workers', 0)
    )
    
    logger.info(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    
    # Create trainer
    trainer = UNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_type=training_config.get('loss_type', 'dice'),
        optimizer_type=training_config.get('optimizer_type', 'adam'),
        learning_rate=training_config.get('learning_rate', 1e-4)
    )
    
    return trainer
