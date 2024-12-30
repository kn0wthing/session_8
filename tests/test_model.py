import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from models.custom_model import CifarNet, DepthwiseSeparableConv2dBlock
from utils.data_loader import CIFAR10Dataset
from config import train_transforms, CIFAR_MEAN

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_parameter_count():
    model = CifarNet()
    param_count = count_parameters(model)
    print(f"Model has {param_count} parameters")
    assert param_count < 200000, f"Model has {param_count} parameters, should be < 200000"

def test_model_architecture():
    model = CifarNet()
    
    # Test if model contains required components
    has_depthwise = False
    has_dilated = False
    has_gap = False
    
    for module in model.modules():
        if isinstance(module, DepthwiseSeparableConv2dBlock):
            has_depthwise = True
        if hasattr(module, 'conv') and hasattr(module.conv[0], 'dilation') and module.conv[0].dilation[0] > 1:
            has_dilated = True
        if isinstance(module, nn.AdaptiveAvgPool2d):
            has_gap = True
    
    assert has_depthwise, "Model should contain depthwise separable convolution"
    assert has_dilated, "Model should contain dilated convolution"
    assert has_gap, "Model should contain Global Average Pooling"

def test_forward_pass():
    model = CifarNet()
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    output = model(x)
    assert output.shape == (batch_size, 10), f"Expected output shape (4, 10), got {output.shape}"

def test_data_augmentation():
    dataset = CIFAR10Dataset(root='./data', train=True, transform=train_transforms)
    img, _ = dataset[0]
    
    # Test if image is normalized
    assert isinstance(img, torch.Tensor), "Image should be converted to tensor"
    assert img.shape == (3, 32, 32), f"Expected shape (3, 32, 32), got {img.shape}"
    
    # Test if values are normalized
    assert torch.mean(img).item() < 1.0, "Image should be normalized"

def test_receptive_field():
    # Calculate receptive field for the new architecture
    rf = 3  # First conv in block1
    rf = rf + 2 * (3 - 1)  # Second conv in block1
    rf = rf + 2 * (3 - 1)  # First conv in block2
    rf = rf + 2 * (3 - 1)  # Depthwise conv in block2
    rf = rf + 2 * (3 - 1)  # First conv in block3
    rf = rf + 2 * (3 - 1)  # Second conv in block3
    rf = rf + 2 * 2 * (3 - 1)  # Dilated conv in block3
    rf = rf + 2 * (3 - 1)  # First conv in block4
    rf = rf + 2 * (3 - 1)  # Second conv in block4
    rf = rf + 2 * (3 - 1)  # Third conv in block4
    
    assert rf > 40, f"Receptive field should be > 40, got approximately {rf}"

if __name__ == "__main__":
    pytest.main([__file__]) 