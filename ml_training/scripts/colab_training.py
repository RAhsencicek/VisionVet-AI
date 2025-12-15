"""
🚀 VisionVet-AI Model Training - Google Colab Ready
Copy-paste this entire file to Google Colab and run!

Prerequisites:
1. Go to https://colab.research.google.com/
2. Runtime > Change runtime type > GPU
3. Paste this code
4. Run all cells (Cell > Run all)
"""

# ============================================================
# STEP 1: Setup Environment
# ============================================================
print("🔧 Installing dependencies...")
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
!pip install onnx onnxruntime pillow tqdm matplotlib -q

print("✅ Dependencies installed!")

# ============================================================
# STEP 2: Clone Project
# ============================================================
print("\n📦 Cloning VisionVet-AI project...")
!git clone https://github.com/ihoflaz/bacterial-colony-classification.git dibas_dataset
print("✅ Project cloned!")

# ============================================================
# STEP 3: Prepare Dataset
# ============================================================
print("\n📊 Preparing dataset...")
import os
import shutil
from pathlib import Path

# Create directory structure
os.makedirs("data/dibas", exist_ok=True)

# Copy dataset
dataset_source = Path("dibas_dataset/images")
dataset_target = Path("data/dibas")

if dataset_source.exists():
    print(f"Copying dataset from {dataset_source}...")
    
    # Copy all class folders
    for class_folder in dataset_source.iterdir():
        if class_folder.is_dir():
            target_folder = dataset_target / class_folder.name
            if not target_folder.exists():
                shutil.copytree(class_folder, target_folder)
                print(f"  ✓ {class_folder.name}")
    
    print("✅ Dataset prepared!")
else:
    print("⚠️  Dataset source not found, will use alternative download...")
    # Alternative: Direct download (if needed)
    !wget https://github.com/ihoflaz/bacterial-colony-classification/archive/refs/heads/main.zip
    !unzip -q main.zip
    !mv bacterial-colony-classification-main/images/* data/dibas/
    !rm -rf main.zip bacterial-colony-classification-main

# Check dataset
dataset_path = Path("data/dibas")
num_classes = len([d for d in dataset_path.iterdir() if d.is_dir()])
print(f"\n📊 Found {num_classes} bacterial classes")

# ============================================================
# STEP 4: Training Script
# ============================================================
print("\n🧠 Preparing training script...")

training_code = """
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import json
from tqdm import tqdm
import os

# ============================================================
# Configuration
# ============================================================
class Config:
    # Paths
    DATA_DIR = "data/dibas"
    OUTPUT_DIR = "models/bacterial"
    
    # Model
    NUM_CLASSES = 33
    INPUT_SIZE = 224
    
    # Training - OPTIMIZED FOR GENERALIZATION!
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-2  # Strong regularization
    
    # Early Stopping
    EARLY_STOPPING = True
    PATIENCE = 7
    
    # Augmentation
    USE_AUGMENTATION = True
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🖥️  Using device: {Config.DEVICE}")
if Config.DEVICE.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# Data Transforms - AGGRESSIVE AUGMENTATION!
# ============================================================
def get_transforms(is_train=True):
    if is_train and Config.USE_AUGMENTATION:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(Config.INPUT_SIZE),
            
            # AGGRESSIVE AUGMENTATION FOR GENERALIZATION!
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(180),  # Full rotation
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.8, 1.2),
            ),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            
            # Random erasing
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# ============================================================
# Load Dataset
# ============================================================
print("\\n📂 Loading dataset...")
dataset = datasets.ImageFolder(Config.DATA_DIR, transform=get_transforms(is_train=True))

# Split train/val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Update val transform
val_dataset.dataset = datasets.ImageFolder(Config.DATA_DIR, transform=get_transforms(is_train=False))

print(f"✅ Training samples: {len(train_dataset)}")
print(f"✅ Validation samples: {len(val_dataset)}")
print(f"✅ Number of classes: {len(dataset.classes)}")

# Save class names
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
with open(f"{Config.OUTPUT_DIR}/labels_{Config.NUM_CLASSES}.txt", "w") as f:
    for class_name in dataset.classes:
        f.write(f"{class_name}\\n")
print(f"✅ Saved class labels to {Config.OUTPUT_DIR}/labels_{Config.NUM_CLASSES}.txt")

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)

# ============================================================
# Create Model - MobileNetV3-Large
# ============================================================
print("\\n🧠 Creating MobileNetV3-Large model...")
model = models.mobilenet_v3_large(pretrained=True)

# Freeze early layers
for param in model.features[:10].parameters():
    param.requires_grad = False

# Replace classifier with STRONG DROPOUT
num_features = model.classifier[0].in_features
model.classifier = nn.Sequential(
    nn.Linear(num_features, 1280),
    nn.Hardswish(),
    nn.Dropout(p=0.5),  # STRONG dropout to prevent overfitting
    nn.Linear(1280, Config.NUM_CLASSES)
)

model = model.to(Config.DEVICE)
print(f"✅ Model created and moved to {Config.DEVICE}")

# ============================================================
# Loss, Optimizer, Scheduler
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# ============================================================
# Training Functions
# ============================================================
def train_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/len(loader), 'acc': 100.*correct/total})
    
    return running_loss/len(loader), 100.*correct/total

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation"):
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss/len(loader), 100.*correct/total

# ============================================================
# Training Loop with Early Stopping
# ============================================================
print("\\n" + "="*60)
print("🚀 STARTING TRAINING")
print("="*60)

best_acc = 0.0
patience_counter = 0

for epoch in range(Config.EPOCHS):
    print(f"\\nEpoch {epoch+1}/{Config.EPOCHS}")
    print("-"*60)
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion)
    
    print(f"📈 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"📉 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # Check overfitting
    acc_gap = train_acc - val_acc
    if acc_gap > 5:
        print(f"⚠️  WARNING: Train-Val gap = {acc_gap:.1f}% (possible overfitting)")
    
    # Learning rate scheduling
    scheduler.step(val_acc)
    
    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), f"{Config.OUTPUT_DIR}/best_model.pth")
        print(f"💾 Best model saved! Accuracy: {best_acc:.2f}%")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"⏳ No improvement for {patience_counter} epoch(s)")
    
    # Early stopping
    if Config.EARLY_STOPPING and patience_counter >= Config.PATIENCE:
        print(f"\\n⚠️  EARLY STOPPING triggered!")
        print(f"   No improvement for {Config.PATIENCE} consecutive epochs")
        print(f"   Best validation accuracy: {best_acc:.2f}%")
        print(f"   Stopping at epoch {epoch+1} to prevent overfitting")
        break

print("\\n" + "="*60)
print(f"🎉 Training completed! Best validation accuracy: {best_acc:.2f}%")
print("="*60)

# ============================================================
# Export to ONNX
# ============================================================
print("\\n📦 Exporting model to ONNX...")
model.load_state_dict(torch.load(f"{Config.OUTPUT_DIR}/best_model.pth"))
model.eval()

dummy_input = torch.randn(1, 3, Config.INPUT_SIZE, Config.INPUT_SIZE).to(Config.DEVICE)
onnx_path = f"{Config.OUTPUT_DIR}/mobilenet_v3_large.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"✅ ONNX model saved to {onnx_path}")

# ============================================================
# Download Results
# ============================================================
print("\\n📥 Preparing files for download...")
from google.colab import files

# Download ONNX model
files.download(onnx_path)
print(f"✅ Downloaded: {onnx_path}")

# Download labels
labels_path = f"{Config.OUTPUT_DIR}/labels_{Config.NUM_CLASSES}.txt"
files.download(labels_path)
print(f"✅ Downloaded: {labels_path}")

print("\\n" + "="*60)
print("🎉 ALL DONE!")
print("="*60)
print("\\nNext steps:")
print("1. Copy downloaded files to Android project:")
print("   - mobilenet_v3_large.onnx → app/src/main/assets/bacterial/")
print("   - labels_33.txt → app/src/main/assets/bacterial/")
print("2. Rebuild Android app: ./gradlew clean assembleDebug")
print("3. Install: ./gradlew installDebug")
print("4. Test with phone camera!")
print("\\n🚀 Your model is ready to recognize bacteria from phone camera!")
"""

# Write training script
with open("train.py", "w") as f:
    f.write(training_code)

print("✅ Training script prepared!")

# ============================================================
# STEP 5: Run Training
# ============================================================
print("\n" + "="*60)
print("🚀 STARTING TRAINING")
print("="*60)
print("\n⏰ This will take 2-3 hours. Go grab a coffee! ☕\n")

!python train.py

print("\n" + "="*60)
print("🎉 TRAINING COMPLETE!")
print("="*60)
print("\n✅ Model files have been automatically downloaded!")
print("✅ Check your Downloads folder for:")
print("   - mobilenet_v3_large.onnx")
print("   - labels_33.txt")
