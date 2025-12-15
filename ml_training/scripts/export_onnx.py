#!/usr/bin/env python3
"""
Export trained model to ONNX format - Legacy method
"""

import torch
import torch.nn as nn
from torchvision import models

# Config
MODEL_PATH = "models/bacterial_generalized/best_model.pth"
OUTPUT_PATH = "models/bacterial_generalized/bacterial_classifier.onnx"
INPUT_SIZE = 224
NUM_CLASSES = 32

def create_model():
    """Create model architecture"""
    model = models.mobilenet_v3_large(weights=None)
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 1280),
        nn.Hardswish(),
        nn.Dropout(p=0.5),
        nn.BatchNorm1d(1280),
        nn.Linear(1280, 512),
        nn.Hardswish(),
        nn.Dropout(p=0.5),
        nn.Linear(512, NUM_CLASSES)
    )
    return model

def main():
    print("📦 Loading trained model...")
    
    # Create model
    model = create_model()
    
    # Load weights
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model with accuracy: {checkpoint.get('best_acc', 'N/A'):.2f}%")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    
    # Export to ONNX using legacy exporter
    print("📦 Exporting to ONNX (legacy mode)...")
    
    # Disable dynamo for classic ONNX export
    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_PATH,
        export_params=True,
        opset_version=17,  # Higher version for compatibility
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        dynamo=False  # Use legacy export
    )
    
    print(f"✅ Model exported to: {OUTPUT_PATH}")
    
    # Verify
    import onnx
    onnx_model = onnx.load(OUTPUT_PATH)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model verified!")
    
    # Get file size
    import os
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"📊 Model size: {size_mb:.2f} MB")
    
    # Also save as PyTorch model for flexibility
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': NUM_CLASSES,
        'input_size': INPUT_SIZE,
        'accuracy': checkpoint.get('best_acc', 79.10)
    }, "models/bacterial_generalized/bacterial_classifier.pt")
    print("✅ PyTorch model also saved!")
    
    print("\n🎉 Model ready for Android deployment!")
    print(f"   ONNX: {OUTPUT_PATH}")
    print(f"   PyTorch: models/bacterial_generalized/bacterial_classifier.pt")

if __name__ == "__main__":
    main()
