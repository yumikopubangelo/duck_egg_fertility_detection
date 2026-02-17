"""
Lightweight UNet Implementation for Egg Segmentation

This module implements a lightweight version of the U-Net architecture
for faster inference on resource-constrained devices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


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
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final output convolution"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetLightWeight(nn.Module):
    """Lightweight UNet architecture for egg segmentation"""
    
    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 1,
        bilinear: bool = True,
        dropout_rate: float = 0.0
    ):
        """
        Initialize UNetLightWeight.
        
        Args:
            n_channels: Number of input channels (3 for RGB images)
            n_classes: Number of output channels (1 for binary segmentation)
            bilinear: Whether to use bilinear upsampling or transposed convolution
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder (Downsampling path) - using fewer channels for lightweight
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        
        # Bottleneck
        self.bottleneck = DoubleConv(256, 256)
        
        # Decoder (Upsampling path) - fixed channel dimensions
        self.up1 = Up(384, 128, bilinear)  # 256 + 128 = 384
        self.up2 = Up(192, 64, bilinear)  # 128 + 64 = 192
        self.up3 = Up(96, 32, bilinear)  # 64 + 32 = 96
        self.outc = OutConv(32, n_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the lightweight UNet.
        
        Args:
            x: Input image tensor (B x C x H x W)
        
        Returns:
            Output logits tensor (B x 1 x H x W)
        """
        # Encoder
        x1 = self.inc(x)  # (B, 32, H, W)
        x2 = self.down1(x1)  # (B, 64, H/2, W/2)
        x3 = self.down2(x2)  # (B, 128, H/4, W/4)
        x4 = self.down3(x3)  # (B, 256, H/8, W/8)
        
        # Bottleneck
        x = self.bottleneck(x4)  # (B, 256, H/8, W/8)
        if self.dropout:
            x = self.dropout(x)
        
        # Decoder with skip connections
        x = self.up1(x, x3)  # (B, 128, H/4, W/4)
        x = self.up2(x, x2)  # (B, 64, H/2, W/2)
        x = self.up3(x, x1)  # (B, 32, H, W)
        
        # Output layer
        logits = self.outc(x)  # (B, 1, H, W)
        
        return logits


def create_unet_lightweight(
    n_channels: int = 3,
    n_classes: int = 1,
    bilinear: bool = True,
    dropout_rate: float = 0.0
) -> UNetLightWeight:
    """
    Create a lightweight UNet model.
    
    Args:
        n_channels: Number of input channels
        n_classes: Number of output channels
        bilinear: Whether to use bilinear upsampling
        dropout_rate: Dropout rate
    
    Returns:
        Lightweight UNet model instance
    """
    return UNetLightWeight(
        n_channels=n_channels,
        n_classes=n_classes,
        bilinear=bilinear,
        dropout_rate=dropout_rate
    )


def count_parameters(model: nn.Module) -> int:
    """Count number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module) -> str:
    """Get detailed summary of the model architecture"""
    summary = []
    summary.append("Lightweight UNet Summary:")
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
