#!/usr/bin/env python3
"""
Test script for CBAM-ResNet waste classification system
Tests model creation, forward pass, and basic functionality without requiring data
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append('.')

from models.cbam_resnet import ResNet34
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_creation():
    """Test model instantiation"""
    logger.info("Testing model creation...")
    try:
        model = ResNet34(num_classes=2)
        logger.info("✓ Model created successfully")
        logger.info(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model
    except Exception as e:
        logger.error(f"✗ Model creation failed: {e}")
        return None

def test_forward_pass(model):
    """Test forward pass with dummy input"""
    logger.info("Testing forward pass...")
    try:
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Create dummy input (batch_size=1, channels=3, height=224, width=224)
        dummy_input = torch.randn(1, 3, 224, 224).to(device)

        with torch.no_grad():
            output = model(dummy_input)
            logger.info(f"✓ Forward pass successful, output shape: {output.shape}")
            logger.info(f"✓ Output: {output}")
            return True
    except Exception as e:
        logger.error(f"✗ Forward pass failed: {e}")
        return False

def test_model_components(model):
    """Test individual model components"""
    logger.info("Testing model components...")
    try:
        # Check if CBAM modules exist
        has_cbam = hasattr(model.layer1[0], 'cbam') or hasattr(model.layer2[0], 'cbam') or \
                   hasattr(model.layer3[0], 'cbam') or hasattr(model.layer4[0], 'cbam')
        if has_cbam:
            logger.info("✓ CBAM attention modules found")
        else:
            logger.warning("⚠ No CBAM modules found")

        # Check layer structure
        logger.info(f"✓ Layer 1: {len(model.layer1)} blocks")
        logger.info(f"✓ Layer 2: {len(model.layer2)} blocks")
        logger.info(f"✓ Layer 3: {len(model.layer3)} blocks")
        logger.info(f"✓ Layer 4: {len(model.layer4)} blocks")

        return True
    except Exception as e:
        logger.error(f"✗ Component test failed: {e}")
        return False

def test_imports():
    """Test all required imports"""
    logger.info("Testing imports...")
    try:
        import torch
        import torchvision
        from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from tqdm import tqdm
        logger.info("✓ All imports successful")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def main():
    logger.info("Starting CBAM-ResNet waste classification system tests...")
    logger.info("=" * 60)

    # Test imports
    if not test_imports():
        return False

    # Test model creation
    model = test_model_creation()
    if model is None:
        return False

    # Test model components
    if not test_model_components(model):
        return False

    # Test forward pass
    if not test_forward_pass(model):
        return False

    logger.info("=" * 60)
    logger.info("✓ All tests passed! System is ready for training.")
    logger.info("To train the model:")
    logger.info("1. Run: python src/download_data.py")
    logger.info("2. Run: python src/prepare_data.py")
    logger.info("3. Run: python src/train.py")
    logger.info("Or use the Jupyter notebook: notebooks/train_model.ipynb")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)