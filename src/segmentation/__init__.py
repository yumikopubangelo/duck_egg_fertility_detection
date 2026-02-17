"""
Segmentation package including U-Net implementations.
"""

from .unet import (
    UNet,
    create_unet,
    create_unet_for_eggs,
    calculate_iou,
    calculate_dice_coefficient
)
from .unet_lightweight import (
    UNetLightWeight,
    create_unet_lightweight
)
from .data_loader import EggDataset
from .losses import (
    DiceLoss,
    FocalLoss,
    DiceBCELoss,
    FocalDiceLoss,
    get_loss_function
)
from .trainer import (
    UNetTrainer,
    create_trainer
)

__all__ = [
    "UNet",
    "UNetLightWeight",
    "EggDataset",
    "DiceLoss",
    "FocalLoss",
    "DiceBCELoss",
    "FocalDiceLoss",
    "create_unet",
    "create_unet_lightweight",
    "create_unet_for_eggs",
    "calculate_iou",
    "calculate_dice_coefficient",
    "get_loss_function",
    "UNetTrainer",
    "create_trainer"
]
