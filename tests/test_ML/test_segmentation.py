"""
Tests for segmentation module.
"""

import pytest
import torch
from src.segmentation import (
    UNet,
    UNetLightWeight,
    create_unet,
    create_unet_lightweight,
    DiceLoss,
    FocalLoss,
    get_loss_function
)


def test_model_creation():
    """Test that models can be created correctly"""
    # Test basic UNet
    model = UNet(n_channels=3, n_classes=1)
    assert model is not None
    
    # Test lightweight UNet
    lightweight_model = UNetLightWeight(n_channels=3, n_classes=1)
    assert lightweight_model is not None
    
    # Test factory functions
    factory_model = create_unet(lightweight=True)
    assert isinstance(factory_model, UNetLightWeight)
    
    factory_model2 = create_unet_lightweight()
    assert isinstance(factory_model2, UNetLightWeight)


def test_model_forward_pass():
    """Test that models can process input tensors"""
    # Create test input (batch size 1, 3 channels, 256x256)
    x = torch.randn(1, 3, 256, 256)
    
    # Test UNet
    model = UNet(n_channels=3, n_classes=1)
    output = model(x)
    assert output.shape == (1, 1, 256, 256)
    
    # Test lightweight UNet
    lightweight_model = UNetLightWeight(n_channels=3, n_classes=1)
    output = lightweight_model(x)
    assert output.shape == (1, 1, 256, 256)


def test_loss_functions():
    """Test that loss functions can be created and used"""
    # Create test inputs
    logits = torch.randn(1, 1, 256, 256)
    mask = torch.randint(0, 2, (1, 1, 256, 256)).float()
    
    # Test Dice loss
    dice_loss = DiceLoss()
    loss = dice_loss(logits, mask)
    assert isinstance(loss, torch.Tensor)
    
    # Test Focal loss
    focal_loss = FocalLoss()
    loss = focal_loss(logits, mask)
    assert isinstance(loss, torch.Tensor)
    
    # Test loss function factory
    for loss_type in ['dice', 'bce', 'ce', 'focal', 'dice_bce', 'focal_dice']:
        loss_func = get_loss_function(loss_type)
        assert callable(loss_func)


def test_parameter_count():
    """Test that models have the expected number of parameters"""
    # Test parameter count
    model = UNet(n_channels=3, n_classes=1)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert param_count > 18_000_000  # ~18 million parameters
    
    lightweight_model = UNetLightWeight(n_channels=3, n_classes=1)
    lightweight_param_count = sum(p.numel() for p in lightweight_model.parameters() if p.requires_grad)
    assert lightweight_param_count < 4_000_000  # ~3 million parameters
    
    assert lightweight_param_count < param_count / 4  # Lightweight should be much smaller


def test_loss_function_factory():
    """Test that loss function factory works correctly"""
    # Test different loss function types
    dice_loss = get_loss_function('dice')
    assert isinstance(dice_loss, DiceLoss)
    
    focal_loss = get_loss_function('focal')
    assert isinstance(focal_loss, FocalLoss)
    
    bce_loss = get_loss_function('bce')
    assert isinstance(bce_loss, torch.nn.BCEWithLogitsLoss)
    
    ce_loss = get_loss_function('ce')
    assert isinstance(ce_loss, torch.nn.CrossEntropyLoss)


def test_create_unet_factory():
    """Test that the create_unet factory function works correctly"""
    # Test creating standard UNet
    standard_model = create_unet(lightweight=False)
    assert isinstance(standard_model, UNet)
    
    # Test creating lightweight UNet
    lightweight_model = create_unet(lightweight=True)
    assert isinstance(lightweight_model, UNetLightWeight)
    
    # Test that both models have different parameter counts
    standard_params = sum(p.numel() for p in standard_model.parameters() if p.requires_grad)
    lightweight_params = sum(p.numel() for p in lightweight_model.parameters() if p.requires_grad)
    
    assert standard_params > lightweight_params
    assert lightweight_params < standard_params / 4  # Should be at least 4x smaller


if __name__ == "__main__":
    pytest.main([__file__, "-v"])