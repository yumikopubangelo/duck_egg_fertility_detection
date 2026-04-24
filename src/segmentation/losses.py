"""
Loss functions for segmentation tasks.

This module implements various loss functions commonly used in image segmentation,
including Dice loss, Focal loss, and combinations of losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation"""

    def __init__(self, smooth: float = 1.0):
        """
        Initialize DiceLoss.

        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss.

        Args:
            input: Model predictions (logits)
            target: Ground truth masks

        Returns:
            Dice loss value
        """
        input = torch.sigmoid(input)

        input_flat = input.view(-1)
        target_flat = target.view(-1)

        intersection = (input_flat * target_flat).sum()
        union = input_flat.sum() + target_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal loss for binary segmentation"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize FocalLoss.

        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Focal loss.

        Args:
            input: Model predictions (logits)
            target: Ground truth masks

        Returns:
            Focal loss value
        """
        # Input is logits - do NOT apply sigmoid before BCEWithLogitsLoss!
        BCE_loss = nn.BCEWithLogitsLoss(reduction="none")(input, target)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        return F_loss.mean()


class DiceBCELoss(nn.Module):
    """Combination of Dice loss and BCE loss"""

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1.0):
        """
        Initialize DiceBCELoss.

        Args:
            dice_weight: Weight for Dice loss
            bce_weight: Weight for BCE loss
            smooth: Smoothing factor for Dice loss
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined Dice + BCE loss.

        Args:
            input: Model predictions (logits)
            target: Ground truth masks

        Returns:
            Combined loss value
        """
        # Dice loss
        input_sigmoid = torch.sigmoid(input)
        input_flat = input_sigmoid.view(-1)
        target_flat = target.view(-1)

        intersection = (input_flat * target_flat).sum()
        union = input_flat.sum() + target_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice

        # BCE loss
        bce_loss = nn.BCEWithLogitsLoss()(input, target)

        # Combined loss
        total_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        return total_loss


class FocalDiceLoss(nn.Module):
    """Combination of Focal loss and Dice loss"""

    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        alpha: float = 0.25,
        gamma: float = 2.0,
        smooth: float = 1.0,
    ):
        """
        Initialize FocalDiceLoss.

        Args:
            focal_weight: Weight for Focal loss
            dice_weight: Weight for Dice loss
            alpha: Weighting factor for positive class (Focal loss)
            gamma: Focusing parameter (Focal loss)
            smooth: Smoothing factor for Dice loss
        """
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined Focal + Dice loss.

        Args:
            input: Model predictions (logits)
            target: Ground truth masks

        Returns:
            Combined loss value
        """
        # Focal loss
        input_sigmoid = torch.sigmoid(input)
        BCE_loss = nn.BCEWithLogitsLoss(reduction="none")(input, target)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        focal_loss = focal_loss.mean()

        # Dice loss
        input_flat = input_sigmoid.view(-1)
        target_flat = target.view(-1)

        intersection = (input_flat * target_flat).sum()
        union = input_flat.sum() + target_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice

        # Combined loss
        total_loss = self.focal_weight * focal_loss + self.dice_weight * dice_loss
        return total_loss


def get_loss_function(loss_type: str = "dice") -> nn.Module:
    """
    Get loss function by name.

    Args:
        loss_type: Type of loss function to return. Options:
            - 'dice': DiceLoss
            - 'bce': BCEWithLogitsLoss
            - 'ce': CrossEntropyLoss
            - 'focal': FocalLoss
            - 'dice_bce': DiceBCELoss
            - 'focal_dice': FocalDiceLoss

    Returns:
        Loss function module

    Raises:
        ValueError: If loss_type is not recognized
    """
    if loss_type == "dice":
        return DiceLoss()
    elif loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_type == "ce":
        return nn.CrossEntropyLoss()
    elif loss_type == "focal":
        return FocalLoss()
    elif loss_type == "dice_bce":
        return DiceBCELoss()
    elif loss_type == "focal_dice":
        return FocalDiceLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
