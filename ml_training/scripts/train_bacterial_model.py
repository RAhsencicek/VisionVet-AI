"""
VisionVet-AI Bacterial Colony Classification Model Training
MobileNetV3-Large Transfer Learning on DIBaS Dataset

Author: VisionVet Team
Dataset: DIBaS (Digital Image of Bacterial Species) - 33 classes
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import onnx
from torch.onnx import export

# ========================
# Configuration
# ========================
class Config:
    # Paths
    DATA_DIR = "data/dibas"  # DIBaS dataset directory
    OUTPUT_DIR = "models/bacterial"
    
    # Model
    MODEL_NAME = "mobilenet_v3_large"
    NUM_CLASSES = 33
    INPUT_SIZE = 224
    
    # Training
    BATCH_SIZE = 32
    EPOCHS = 100  # Increased - will stop early if needed
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-2  # INCREASED! 1e-4 → 1e-2 (prevents overfitting)
    
    # Early Stopping
    EARLY_STOPPING = True
    PATIENCE = 7  # Stop if no improvement for 7 epochs
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Augmentation
    USE_AUGMENTATION = True
    
    print(f"🖥️  Using device: {DEVICE}")

# ========================
# Dataset Class
# ========================
class BacterialDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ========================
# Data Preparation
# ========================
def prepare_data():
    """Load and split DIBaS dataset"""
    print("📂 Loading dataset...")
    
    data_dir = Path(Config.DATA_DIR)
    image_paths = []
    labels = []
    class_names = []
    
    # Get all class folders
    class_folders = sorted([f for f in data_dir.iterdir() if f.is_dir()])
    class_to_idx = {cls.name: idx for idx, cls in enumerate(class_folders)}
    class_names = [cls.name for cls in class_folders]
    
    # Collect all images
    for class_folder in class_folders:
        class_idx = class_to_idx[class_folder.name]
        for img_path in class_folder.glob("*.jpg"):
            image_paths.append(str(img_path))
            labels.append(class_idx)
    
    print(f"✅ Found {len(image_paths)} images across {len(class_names)} classes")
    
    # Save class names
    with open(f"{Config.OUTPUT_DIR}/labels_33.txt", "w") as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    return image_paths, labels, class_names

# ========================
# Data Transforms
# ========================
def get_transforms(is_train=True):
    if is_train and Config.USE_AUGMENTATION:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(Config.INPUT_SIZE),
            
            # AGGRESSIVE AUGMENTATION FOR GENERALIZATION!
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(180),  # Full rotation instead of 20°
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # Shift image
                scale=(0.8, 1.2),      # Zoom in/out
            ),
            transforms.ColorJitter(
                brightness=0.3,  # Simulate different lighting
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            
            # Random erasing (simulate occlusions)
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# ========================
# Model Creation
# ========================
def create_model():
    """Create MobileNetV3-Large with custom classifier"""
    print("🧠 Creating MobileNetV3-Large model...")
    
    # Load pretrained MobileNetV3
    model = models.mobilenet_v3_large(pretrained=True)
    
    # Freeze early layers (optional - comment out for full fine-tuning)
    for param in model.features[:10].parameters():
        param.requires_grad = False
    
    # Replace classifier
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 1280),
        nn.Hardswish(),
        nn.Dropout(p=0.5),  # INCREASED! 0.2 → 0.5 (prevents overfitting)
        nn.Linear(1280, Config.NUM_CLASSES)
    )
    
    return model.to(Config.DEVICE)

# ========================
# Training Function
# ========================
def train_epoch(model, dataloader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
    for images, labels in pbar:
        images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{running_loss/len(pbar):.3f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss / len(dataloader), 100. * correct / total

# ========================
# Validation Function
# ========================
def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total

# ========================
# Export to ONNX
# ========================
def export_to_onnx(model, output_path="models/bacterial/mobilenet_v3_large.onnx"):
    """Export trained model to ONNX format for Android deployment"""
    print("📦 Exporting to ONNX...")
    
    model.eval()
    dummy_input = torch.randn(1, 3, Config.INPUT_SIZE, Config.INPUT_SIZE).to(Config.DEVICE)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,  # ONNX Runtime 1.19.2 compatible
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"✅ Model exported to {output_path}")
    
    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model verified successfully!")

# ========================
# Main Training Loop
# ========================
def main():
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Prepare data
    image_paths, labels, class_names = prepare_data()
    
    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"📊 Training samples: {len(X_train)}")
    print(f"📊 Validation samples: {len(X_val)}")
    
    # Create datasets
    train_dataset = BacterialDataset(X_train, y_train, get_transforms(is_train=True))
    val_dataset = BacterialDataset(X_val, y_val, get_transforms(is_train=False))
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Create model
    model = create_model()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    best_acc = 0.0
    patience_counter = 0
    
    for epoch in range(Config.EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        print(f"\n📈 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"📉 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Check overfitting
        acc_gap = train_acc - val_acc
        if acc_gap > 5:
            print(f"\n⚠️  WARNING: Train-Val gap = {acc_gap:.1f}% (possible overfitting)")
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Save best model and check early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{Config.OUTPUT_DIR}/best_model.pth")
            print(f"💾 Best model saved! Accuracy: {best_acc:.2f}%")
            patience_counter = 0  # Reset patience
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter} epoch(s)")
            
        # Early stopping
        if Config.EARLY_STOPPING and patience_counter >= Config.PATIENCE:
            print(f"\n⚠️  EARLY STOPPING triggered!")
            print(f"   No improvement for {Config.PATIENCE} consecutive epochs")
            print(f"   Best validation accuracy: {best_acc:.2f}%")
            print(f"   Stopping at epoch {epoch+1} to prevent overfitting")
            break
    
    print(f"\n{'='*60}")
    print(f"🎉 Training completed! Best validation accuracy: {best_acc:.2f}%")
    print(f"{'='*60}")
    
    # Load best model and export to ONNX
    model.load_state_dict(torch.load(f"{Config.OUTPUT_DIR}/best_model.pth"))
    export_to_onnx(model, f"{Config.OUTPUT_DIR}/mobilenet_v3_large.onnx")
    
    print("\n✅ Model ready for Android deployment!")
    print(f"📁 Model location: {Config.OUTPUT_DIR}/mobilenet_v3_large.onnx")
    print(f"📁 Labels location: {Config.OUTPUT_DIR}/labels_33.txt")

if __name__ == "__main__":
    main()
