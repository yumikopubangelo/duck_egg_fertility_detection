"""
UNet Implementation for Egg Segmentation

This module implements a UNet architecture for segmenting duck eggs in images.
UNet is a convolutional neural network that is widely used for biomedical image segmentation.

Architecture:
- Encoder (Downsampling path)
- Bottleneck
- Decoder (Upsampling path)
- Skip connections between encoder and decoder

References:
- Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- https://arxiv.org/abs/1505.04597
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DoubleConv(nn.Module):
    """(convolution => BN => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb148a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac90d6e60498b00b61a6f6373579
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final output convolution"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """UNet architecture for egg segmentation"""

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 1,
        bilinear: bool = True,
        dropout_rate: float = 0.0,
    ):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 512)

        # Decoder
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # Dropout
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Bottleneck
        x = self.bottleneck(x5)
        if self.dropout:
            x = self.dropout(x)

        # Decoder with dropout at each step for better regularization
        x = self.up1(x, x4)
        if self.dropout:
            x = self.dropout(x)

        x = self.up2(x, x3)
        if self.dropout:
            x = self.dropout(x)

        x = self.up3(x, x2)
        if self.dropout:
            x = self.dropout(x)

        x = self.up4(x, x1)

        # Output
        logits = self.outc(x)
        return logits


# Model factory functions


def create_unet(
    n_channels: int = 3,
    n_classes: int = 1,
    bilinear: bool = True,
    dropout_rate: float = 0.0,
    lightweight: bool = False,
):
    """Create UNet model"""
    if lightweight:
        from .unet_lightweight import UNetLightWeight

        return UNetLightWeight(n_channels, n_classes, bilinear, dropout_rate)
    else:
        return UNet(n_channels, n_classes, bilinear, dropout_rate)


def create_unet_for_eggs(
    n_channels: int = 3,
    n_classes: int = 1,
    bilinear: bool = True,
    dropout_rate: float = 0.0,
    lightweight: bool = False,
) -> UNet:
    """Create UNet model specifically for egg segmentation"""
    return create_unet(n_channels, n_classes, bilinear, dropout_rate, lightweight)


# Loss functions


# Loss functions import
from .losses import get_loss_function

# Evaluation metrics


def calculate_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate Intersection over Union (IoU)"""
    pred = torch.sigmoid(pred) > 0.5

    intersection = (pred & target).sum().float()
    union = (pred | target).sum().float()

    if union == 0:
        return 1.0

    return (intersection / union).item()


def calculate_dice_coefficient(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate Dice coefficient"""
    pred = torch.sigmoid(pred) > 0.5

    intersection = 2.0 * (pred & target).sum().float()
    union = pred.sum().float() + target.sum().float()

    if union == 0:
        return 1.0

    return (intersection / union).item()


# Model utilities


def count_parameters(model: nn.Module) -> int:
    """Count number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module) -> str:
    """Get model summary"""
    summary = []
    summary.append("Model Summary:")
    summary.append("=" * 50)

    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_shape = param.size()
            param_size = param.numel()
            total_params += param_size
            summary.append(f"{name}: {param_shape} = {param_size:,} params")

    summary.append("=" * 50)
    summary.append(f"Total Trainable Parameters: {total_params:,}")

    return "\n".join(summary)


# Model loading and saving


def save_model(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    loss: Optional[float] = None,
) -> None:
    """Save model to file"""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "loss": loss,
    }

    torch.save(checkpoint, path)


def load_model(
    model: nn.Module, path: str, optimizer: Optional[torch.optim.Optimizer] = None
) -> Tuple[int, float]:
    """Load model from file"""
    checkpoint = torch.load(path, map_location=torch.device("cpu"))

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("inf"))

    return epoch, loss
