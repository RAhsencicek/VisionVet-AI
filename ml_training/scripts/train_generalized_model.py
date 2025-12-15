"""
VisionVet-AI Generalized Bacterial Colony Classification Model Training
================================================================================
Bu script modelin resimleri EZBERLEMEK yerine ÖĞRENMESINI sağlar.

Özellikler:
- Mixup Augmentation: Farklı bakterileri karıştırarak öğrenme
- Label Smoothing: Aşırı güvenli tahminleri engelleme
- Test-Time Augmentation (TTA): Daha güvenilir tahminler
- Aggressive Regularization: Overfitting'i önleme
- Early Stopping: Tam zamanında durma

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
import random

# ========================
# Configuration
# ========================
class Config:
    # Paths
    DATA_DIR = "data/dibas"  # DIBaS dataset directory
    OUTPUT_DIR = "models/bacterial_generalized"
    
    # Model
    MODEL_NAME = "mobilenet_v3_large"
    NUM_CLASSES = 32  # DIBaS: 32 classes (Acinetobacter excluded - no images)
    INPUT_SIZE = 224
    
    # Training
    BATCH_SIZE = 16  # Smaller batch for better generalization
    EPOCHS = 100
    LEARNING_RATE = 0.0005  # Lower LR for smoother learning
    WEIGHT_DECAY = 1e-2  # Strong regularization
    
    # Early Stopping
    EARLY_STOPPING = True
    PATIENCE = 10  # More patience with strong regularization
    
    # Generalization Techniques
    DROPOUT_RATE = 0.5  # 50% dropout
    LABEL_SMOOTHING = 0.1  # Prevent overconfident predictions
    MIXUP_ALPHA = 0.2  # Mixup augmentation
    USE_MIXUP = True
    USE_TTA = True  # Test-Time Augmentation
    TTA_ITERATIONS = 5
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Augmentation
    USE_AUGMENTATION = True
    
    print(f"🖥️  Using device: {DEVICE}")

# ========================
# Mixup Augmentation
# ========================
def mixup_data(x, y, alpha=0.2):
    """
    Mixup: İki farklı resmi ve etiketlerini karıştır.
    Model hem E.coli hem Salmonella özelliklerini görür → Daha iyi genelleme!
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(Config.DEVICE)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup için özel loss hesaplama"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ========================
# Label Smoothing Loss
# ========================
class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing: Modelin %100 emin olmasını önler.
    Normal: [0, 0, 1, 0, 0] → "Bu kesinlikle E.coli!"
    Smoothed: [0.025, 0.025, 0.9, 0.025, 0.025] → "Büyük ihtimalle E.coli"
    
    Bu sayede model daha dengeli öğrenir ve ezberlemez.
    """
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

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
    
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Save class names
    with open(f"{Config.OUTPUT_DIR}/labels_33.txt", "w") as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    return image_paths, labels, class_names

# ========================
# Data Transforms - AGGRESSIVE!
# ========================
def get_train_transforms():
    """
    Çok Agresif Augmentation - Model ezberleyemez!
    
    Her resmi farklı açılardan, ışıklardan, boyutlardan gösterir.
    Model mecburen bakterinin gerçek özelliklerini öğrenir.
    """
    return transforms.Compose([
        transforms.Resize((280, 280)),  # Biraz daha büyük başla
        transforms.RandomCrop(Config.INPUT_SIZE),
        
        # 1. GEOMETRİK DÖNÜŞÜMLER
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(180),  # Tam 360° dönüş
        transforms.RandomAffine(
            degrees=15,
            translate=(0.15, 0.15),  # %15 kaydırma
            scale=(0.75, 1.25),      # %25 yakınlaştırma/uzaklaştırma
            shear=10                  # Eğme
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # Perspektif bozma
        
        # 2. RENK DÖNÜŞÜMLER (Farklı kamera/ışık simülasyonu)
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.15
        ),
        
        # 3. BULANIKLIK (Kötü kamera simülasyonu)
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0))
        ], p=0.4),
        
        # 4. RANDOM GRAYSCALE (Renk bağımsızlığı)
        transforms.RandomGrayscale(p=0.1),
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
        # 5. RANDOM ERASING (Eksik veri simülasyonu)
        transforms.RandomErasing(p=0.4, scale=(0.02, 0.15)),
    ])


def get_val_transforms():
    """Validation için basit transform"""
    return transforms.Compose([
        transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_tta_transforms():
    """Test-Time Augmentation için transform listesi"""
    return [
        # Original
        transforms.Compose([
            transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # Horizontal flip
        transforms.Compose([
            transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # Vertical flip
        transforms.Compose([
            transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # Rotate 90
        transforms.Compose([
            transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
            transforms.RandomRotation((90, 90)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # Slightly brighter
        transforms.Compose([
            transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    ]

# ========================
# Model Creation
# ========================
def create_model():
    """Create MobileNetV3-Large with strong regularization"""
    print("🧠 Creating MobileNetV3-Large model with strong regularization...")
    
    # Load pretrained MobileNetV3
    model = models.mobilenet_v3_large(pretrained=True)
    
    # Freeze early layers (transfer learning)
    # İlk katmanlar genel özellikler öğrenir (kenarlar, dokular)
    for param in model.features[:12].parameters():
        param.requires_grad = False
    
    # Replace classifier with strong dropout
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 1280),
        nn.Hardswish(),
        nn.Dropout(p=Config.DROPOUT_RATE),  # %50 dropout!
        nn.BatchNorm1d(1280),  # Batch normalization
        nn.Linear(1280, 512),
        nn.Hardswish(),
        nn.Dropout(p=Config.DROPOUT_RATE),  # Bir dropout daha!
        nn.Linear(512, Config.NUM_CLASSES)
    )
    
    return model.to(Config.DEVICE)

# ========================
# Training Function
# ========================
def train_epoch(model, dataloader, criterion, optimizer, epoch, use_mixup=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
    for images, labels in pbar:
        images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
        
        # Mixup augmentation
        if use_mixup and Config.USE_MIXUP:
            images, labels_a, labels_b, lam = mixup_data(images, labels, Config.MIXUP_ALPHA)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Loss calculation
        if use_mixup and Config.USE_MIXUP:
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            loss = criterion(outputs, labels)
        
        loss.backward()
        
        # Gradient clipping (stabilite için)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        
        # Mixup durumunda accuracy hesaplama
        if use_mixup and Config.USE_MIXUP:
            correct += (lam * predicted.eq(labels_a).float() + 
                       (1 - lam) * predicted.eq(labels_b).float()).sum().item()
        else:
            correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.3f}', 
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(dataloader), 100. * correct / total

# ========================
# Validation with TTA
# ========================
def validate(model, dataloader, criterion, use_tta=False):
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


def validate_with_tta(model, val_image_paths, val_labels, criterion):
    """Test-Time Augmentation ile validation"""
    model.eval()
    correct = 0
    total = len(val_labels)
    tta_transforms = get_tta_transforms()
    
    print("🔄 Validating with Test-Time Augmentation...")
    
    with torch.no_grad():
        for idx, (img_path, label) in enumerate(tqdm(zip(val_image_paths, val_labels), total=total)):
            image = Image.open(img_path).convert('RGB')
            
            # Her transform ile tahmin yap
            all_outputs = []
            for tta_transform in tta_transforms:
                img_tensor = tta_transform(image).unsqueeze(0).to(Config.DEVICE)
                output = model(img_tensor)
                all_outputs.append(output)
            
            # Ortalamasını al
            avg_output = torch.mean(torch.stack(all_outputs), dim=0)
            _, predicted = avg_output.max(1)
            
            if predicted.item() == label:
                correct += 1
    
    return 100. * correct / total

# ========================
# Export to ONNX
# ========================
def export_to_onnx(model, output_path):
    """Export trained model to ONNX format for Android deployment"""
    print("📦 Exporting to ONNX...")
    
    model.eval()
    dummy_input = torch.randn(1, 3, Config.INPUT_SIZE, Config.INPUT_SIZE).to(Config.DEVICE)
    
    # CPU'ya taşı (ONNX export için)
    model_cpu = model.cpu()
    dummy_input_cpu = dummy_input.cpu()
    
    torch.onnx.export(
        model_cpu,
        dummy_input_cpu,
        output_path,
        export_params=True,
        opset_version=13,
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
    print("\n" + "="*70)
    print("🧬 VisionVet-AI Generalized Model Training")
    print("   Ezberlemeden Öğrenen Model Eğitimi")
    print("="*70 + "\n")
    
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Prepare data
    image_paths, labels, class_names = prepare_data()
    
    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\n📊 Training samples: {len(X_train)}")
    print(f"📊 Validation samples: {len(X_val)}")
    
    # Aktif teknikler
    print(f"\n🎯 Aktif Generalization Teknikleri:")
    print(f"   • Aggressive Augmentation: ✓")
    print(f"   • Mixup (α={Config.MIXUP_ALPHA}): {'✓' if Config.USE_MIXUP else '✗'}")
    print(f"   • Label Smoothing ({Config.LABEL_SMOOTHING}): ✓")
    print(f"   • Dropout ({Config.DROPOUT_RATE*100}%): ✓")
    print(f"   • Weight Decay ({Config.WEIGHT_DECAY}): ✓")
    print(f"   • Early Stopping (patience={Config.PATIENCE}): ✓")
    print(f"   • Test-Time Augmentation: {'✓' if Config.USE_TTA else '✗'}")
    
    # Create datasets
    train_dataset = BacterialDataset(X_train, y_train, get_train_transforms())
    val_dataset = BacterialDataset(X_val, y_val, get_val_transforms())
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        drop_last=True  # Son eksik batch'i atla (Mixup için)
    )
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Create model
    model = create_model()
    
    # Loss with label smoothing
    criterion = LabelSmoothingLoss(Config.NUM_CLASSES, smoothing=Config.LABEL_SMOOTHING)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing scheduler (smoother learning)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # Her 10 epoch'ta restart
        T_mult=2  # Her restart'ta periyodu 2x yap
    )
    
    # Training loop
    best_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(Config.EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{Config.EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*60}")
        
        # Train with Mixup
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, epoch,
            use_mixup=Config.USE_MIXUP
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        # Update scheduler
        scheduler.step()
        
        # Log results
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\n📈 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"📉 Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.2f}%")
        
        # Check overfitting
        acc_gap = train_acc - val_acc
        if acc_gap > 5:
            print(f"\n⚠️  WARNING: Train-Val gap = {acc_gap:.1f}% (possible overfitting)")
        elif acc_gap < 3:
            print(f"\n✅ GOOD: Train-Val gap = {acc_gap:.1f}% (healthy learning)")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': {
                    'dropout': Config.DROPOUT_RATE,
                    'weight_decay': Config.WEIGHT_DECAY,
                    'label_smoothing': Config.LABEL_SMOOTHING,
                    'mixup_alpha': Config.MIXUP_ALPHA
                }
            }, f"{Config.OUTPUT_DIR}/best_model.pth")
            print(f"💾 Best model saved! Accuracy: {best_acc:.2f}%")
            patience_counter = 0
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
    
    # Final TTA validation
    if Config.USE_TTA:
        print(f"\n{'='*60}")
        print("🔄 Final validation with Test-Time Augmentation...")
        checkpoint = torch.load(f"{Config.OUTPUT_DIR}/best_model.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        tta_acc = validate_with_tta(model, X_val, y_val, criterion)
        print(f"📊 TTA Validation Accuracy: {tta_acc:.2f}%")
        print(f"📊 Normal Validation Accuracy: {best_acc:.2f}%")
        print(f"📊 TTA Improvement: {tta_acc - best_acc:+.2f}%")
    
    print(f"\n{'='*60}")
    print(f"🎉 Training completed! Best validation accuracy: {best_acc:.2f}%")
    print(f"{'='*60}")
    
    # Save training history
    with open(f"{Config.OUTPUT_DIR}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    # Load best model and export to ONNX
    checkpoint = torch.load(f"{Config.OUTPUT_DIR}/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    export_to_onnx(model, f"{Config.OUTPUT_DIR}/mobilenet_v3_large_generalized.onnx")
    
    print("\n" + "="*70)
    print("✅ Model ready for Android deployment!")
    print(f"📁 Model location: {Config.OUTPUT_DIR}/mobilenet_v3_large_generalized.onnx")
    print(f"📁 Labels location: {Config.OUTPUT_DIR}/labels_33.txt")
    print(f"📁 History location: {Config.OUTPUT_DIR}/training_history.json")
    print("="*70)
    
    print("\n💡 Pro Tip: Bu model telefon kamerasından çekilen resimleri")
    print("   daha iyi tanıyacak çünkü ezberlemek yerine öğrendi!")

if __name__ == "__main__":
    main()
