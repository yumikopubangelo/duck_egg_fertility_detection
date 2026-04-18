"""
UNet training script for duck egg data.

This script supports two target formats:
- Segmentation targets: mask tensor shape [B, 1, H, W]
- Classification targets: scalar label shape [B]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.segmentation.data_loader import EggDataset
from src.segmentation.unet import create_unet_for_eggs, count_parameters, get_loss_function
from src.utils.config import load_config
from src.utils.file_utils import create_directories
from src.utils.logger import setup_logger


class TrainingMetrics:
    """Track training and validation metrics."""

    def __init__(self):
        self.losses = []
        self.ious = []
        self.dice_coefficients = []
        self.learning_rates = []
        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []

    def update(
        self,
        loss: float,
        iou: float,
        dice: float,
        lr: float,
        accuracy: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        f1_score: Optional[float] = None,
    ):
        self.losses.append(loss)
        self.ious.append(iou)
        self.dice_coefficients.append(dice)
        self.learning_rates.append(lr)
        if accuracy is not None:
            self.accuracies.append(accuracy)
        if precision is not None:
            self.precisions.append(precision)
        if recall is not None:
            self.recalls.append(recall)
        if f1_score is not None:
            self.f1_scores.append(f1_score)

    @staticmethod
    def _safe_mean(values):
        if not values:
            return float("nan")
        return float(np.mean(values))

    def get_average(self) -> Dict[str, float]:
        return {
            "loss": self._safe_mean(self.losses),
            "iou": self._safe_mean(self.ious),
            "dice": self._safe_mean(self.dice_coefficients),
            "lr": self._safe_mean(self.learning_rates),
            "accuracy": self._safe_mean(self.accuracies),
            "precision": self._safe_mean(self.precisions),
            "recall": self._safe_mean(self.recalls),
            "f1_score": self._safe_mean(self.f1_scores),
        }


def _is_classification_target(target: torch.Tensor) -> bool:
    return target.ndim <= 2 and (target.ndim == 1 or (target.ndim == 2 and target.shape[1] == 1))


def _reduce_logits_for_classification(outputs: torch.Tensor) -> torch.Tensor:
    if outputs.ndim == 4:
        return outputs.mean(dim=(2, 3)).squeeze(1)
    if outputs.ndim == 2 and outputs.shape[1] == 1:
        return outputs.squeeze(1)
    return outputs.view(-1)


def _classification_stats(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = probs > 0.5
    truth = target > 0.5

    tp = torch.logical_and(pred, truth).sum().item()
    fp = torch.logical_and(pred, ~truth).sum().item()
    fn = torch.logical_and(~pred, truth).sum().item()
    tn = torch.logical_and(~pred, ~truth).sum().item()

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "iou": float(iou),
        "dice": float(dice),
    }


def calculate_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate IoU for either segmentation tensors or classification logits."""
    if _is_classification_target(target):
        target = target.float().view(-1)
        logits = _reduce_logits_for_classification(pred)
        return _classification_stats(logits, target)["iou"]

    pred = torch.sigmoid(pred) > 0.5
    target = target > 0.5
    intersection = torch.logical_and(pred, target).sum().float()
    union = torch.logical_or(pred, target).sum().float()
    if union == 0:
        return 1.0
    return float((intersection / union).item())


def calculate_dice_coefficient(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate Dice for either segmentation tensors or classification logits."""
    if _is_classification_target(target):
        target = target.float().view(-1)
        logits = _reduce_logits_for_classification(pred)
        return _classification_stats(logits, target)["dice"]

    pred = torch.sigmoid(pred) > 0.5
    target = target > 0.5
    intersection = 2.0 * torch.logical_and(pred, target).sum().float()
    union = pred.sum().float() + target.sum().float()
    if union == 0:
        return 1.0
    return float((intersection / union).item())


def get_current_lr(optimizer: optim.Optimizer) -> float:
    for param_group in optimizer.param_groups:
        return float(param_group["lr"])
    return 0.0


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> TrainingMetrics:
    model.train()
    metrics = TrainingMetrics()

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)

        if _is_classification_target(targets):
            logits = _reduce_logits_for_classification(outputs)
            cls_target = targets.float().view(-1)
            loss = F.binary_cross_entropy_with_logits(logits, cls_target)
        else:
            loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            iou = calculate_iou(outputs, targets)
            dice = calculate_dice_coefficient(outputs, targets)
            lr = get_current_lr(optimizer)

            if _is_classification_target(targets):
                cls_metrics = _classification_stats(_reduce_logits_for_classification(outputs), targets.float().view(-1))
                metrics.update(
                    loss.item(),
                    iou,
                    dice,
                    lr,
                    accuracy=cls_metrics["accuracy"],
                    precision=cls_metrics["precision"],
                    recall=cls_metrics["recall"],
                    f1_score=cls_metrics["f1_score"],
                )
            else:
                metrics.update(loss.item(), iou, dice, lr)

        if batch_idx % 10 == 0:
            avg = metrics.get_average()
            print(
                f"Epoch {epoch+1} [{batch_idx * len(images)}/{len(dataloader.dataset)}] "
                f"Loss: {loss.item():.4f}, IoU: {iou:.3f}, Dice: {dice:.3f}, "
                f"Acc: {avg['accuracy']:.3f}, LR: {lr:.6f}"
            )

    return metrics


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer,
) -> TrainingMetrics:
    model.eval()
    metrics = TrainingMetrics()

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)

            if _is_classification_target(targets):
                logits = _reduce_logits_for_classification(outputs)
                cls_target = targets.float().view(-1)
                loss = F.binary_cross_entropy_with_logits(logits, cls_target)
            else:
                loss = criterion(outputs, targets)

            iou = calculate_iou(outputs, targets)
            dice = calculate_dice_coefficient(outputs, targets)
            lr = get_current_lr(optimizer)

            if _is_classification_target(targets):
                cls_metrics = _classification_stats(_reduce_logits_for_classification(outputs), targets.float().view(-1))
                metrics.update(
                    loss.item(),
                    iou,
                    dice,
                    lr,
                    accuracy=cls_metrics["accuracy"],
                    precision=cls_metrics["precision"],
                    recall=cls_metrics["recall"],
                    f1_score=cls_metrics["f1_score"],
                )
            else:
                metrics.update(loss.item(), iou, dice, lr)

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: TrainingMetrics,
    config: Dict,
    checkpoint_dir: Path,
    prefix: str = "checkpoint",
) -> Path:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics.get_average(),
        "config": config,
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = checkpoint_dir / f"{prefix}_epoch_{epoch+1}_{timestamp}.pth"
    torch.save(checkpoint, path)
    return path


def main():
    parser = argparse.ArgumentParser(description="UNet Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, help="Path to resume checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)
    log_dir = Path(config["training"]["log_dir"])
    create_directories(log_dir)
    logger = setup_logger(log_dir / "training.log")

    torch.manual_seed(config["training"]["seed"])
    np.random.seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_transform = transforms.Compose(
        [
            transforms.Resize(config["data"]["image_size"]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(config["data"]["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = EggDataset(
        fertile_dir=config["data"]["train_fertile_dir"],
        infertile_dir=config["data"]["train_infertile_dir"],
        image_size=tuple(config["data"]["image_size"]),
        transform=train_transform,
    )
    val_dataset = EggDataset(
        fertile_dir=config["data"]["val_fertile_dir"],
        infertile_dir=config["data"]["val_infertile_dir"],
        image_size=tuple(config["data"]["image_size"]),
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    model = create_unet_for_eggs(
        n_channels=3,
        n_classes=1,
        bilinear=config["model"]["bilinear"],
        dropout_rate=config["model"]["dropout_rate"],
        lightweight=config["model"]["lightweight"],
    ).to(device)

    logger.info(f"Model created: {model.__class__.__name__}")
    logger.info(f"Model has {count_parameters(model):,} parameters")

    criterion = get_loss_function(config["training"]["loss_type"])
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["training"]["lr_reduce_factor"],
        patience=config["training"]["lr_patience"],
    )

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        logger.info(f"Resumed from checkpoint: {args.resume} (epoch {start_epoch})")

    output_dir = Path(config["training"]["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    create_directories(checkpoint_dir)

    best_val_loss = float("inf")
    patience = int(config["training"].get("early_stopping_patience", 15))
    epochs_without_improvement = 0
    epoch = start_epoch - 1
    val_metrics = TrainingMetrics()

    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        logger.info(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_metrics = validate_epoch(model, val_loader, criterion, device, optimizer)

        avg_train = train_metrics.get_average()
        avg_val = val_metrics.get_average()
        scheduler.step(avg_val["loss"])

        logger.info(
            "Training - Loss: %.4f, IoU: %.3f, Dice: %.3f, Acc: %.3f, Precision: %.3f, Recall: %.3f, F1: %.3f, LR: %.6f",
            avg_train["loss"],
            avg_train["iou"],
            avg_train["dice"],
            avg_train["accuracy"],
            avg_train["precision"],
            avg_train["recall"],
            avg_train["f1_score"],
            avg_train["lr"],
        )
        logger.info(
            "Validation - Loss: %.4f, IoU: %.3f, Dice: %.3f, Acc: %.3f, Precision: %.3f, Recall: %.3f, F1: %.3f, LR: %.6f",
            avg_val["loss"],
            avg_val["iou"],
            avg_val["dice"],
            avg_val["accuracy"],
            avg_val["precision"],
            avg_val["recall"],
            avg_val["f1_score"],
            avg_val["lr"],
        )

        checkpoint_interval = int(config["training"].get("checkpoint_interval", 10))
        if (epoch + 1) % checkpoint_interval == 0:
            path = save_checkpoint(model, optimizer, epoch, val_metrics, config, checkpoint_dir)
            logger.info(f"Checkpoint saved: {path}")

        if avg_val["loss"] < best_val_loss:
            best_val_loss = avg_val["loss"]
            epochs_without_improvement = 0
            best_path = save_checkpoint(model, optimizer, epoch, val_metrics, config, checkpoint_dir, prefix="best")
            logger.info(f"Best checkpoint updated: {best_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping triggered.")
                break

    final_epoch = max(epoch, 0)
    final_path = save_checkpoint(model, optimizer, final_epoch, val_metrics, config, checkpoint_dir, prefix="final")
    logger.info(f"Training completed. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
