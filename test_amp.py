#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np

def test_amp():
    """Test if AMP is working properly with updated PyTorch 2.7 API"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping AMP test")
        return
    
    # Simple model for testing
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    ).to(device)
    
    # Test data
    x = torch.randn(32, 100, device=device)
    target = torch.randint(0, 10, (32,), device=device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Test new AMP API
    print("\n=== Testing NEW PyTorch 2.7 AMP API ===")
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=True)
        
        # Forward pass with autocast
        with torch.amp.autocast('cuda', enabled=True):
            output = model(x)
            loss = criterion(output, target)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print("✅ NEW AMP API works correctly!")
        print(f"Loss: {loss.item():.4f}")
        print(f"Scaler state: {scaler.state_dict()}")
        
    except Exception as e:
        print(f"❌ NEW AMP API failed: {e}")
    
    # Test old AMP API for comparison
    print("\n=== Testing OLD PyTorch AMP API ===")
    try:
        old_scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        # Forward pass with autocast
        with torch.cuda.amp.autocast(enabled=True):
            output = model(x)
            loss = criterion(output, target)
        
        # Backward pass
        old_scaler.scale(loss).backward()
        old_scaler.step(optimizer)
        old_scaler.update()
        
        print("✅ OLD AMP API still works!")
        print(f"Loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"❌ OLD AMP API failed: {e}")

if __name__ == "__main__":
    test_amp()
