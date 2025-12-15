"""
VisionVet-AI Balanced Training Script
================================================================================
Bu script TÜM bakterileri dengeli şekilde öğreten gelişmiş bir eğitim yapar.

Özellikler:
- Class-Weighted Loss: Zayıf sınıflara daha fazla önem
- Focal Loss: Zor örneklere odaklanma
- Balanced Batch Sampling: Her batch'te dengeli sınıf dağılımı
- Kontrollü Augmentation: Genelleme + sınıf koruması

Author: VisionVet Team
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import os
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import Counter

# ========================
# Configuration
# ========================
class Config:
    DATA_DIR = "data/dibas"
    OUTPUT_DIR = "models/bacterial_balanced"
    
    MODEL_NAME = "mobilenet_v3_large"
    NUM_CLASSES = 32
    INPUT_SIZE = 224
    
    # Training - Balanced
    BATCH_SIZE = 16
    EPOCHS = 150  # Daha uzun eğitim
    LEARNING_RATE = 0.0003  # Biraz daha düşük
    WEIGHT_DECAY = 5e-3  # Orta seviye regularization
    
    # Early Stopping - Daha sabırlı
    EARLY_STOPPING = True
    PATIENCE = 20  # 10 → 20
    
    # Balanced Training
    USE_CLASS_WEIGHTS = True  # Sınıf ağırlıkları
    USE_FOCAL_LOSS = True     # Focal Loss
    FOCAL_GAMMA = 2.0         # Focal Loss gamma
    USE_BALANCED_SAMPLER = True  # Dengeli örnekleme
    
    # Augmentation - Daha dengeli
    DROPOUT_RATE = 0.4  # 0.5 → 0.4 (biraz daha az)
    LABEL_SMOOTHING = 0.05  # 0.1 → 0.05 (daha az bulanıklaştırma)
    MIXUP_ALPHA = 0.1  # 0.2 → 0.1 (daha az karıştırma)
    USE_MIXUP = True
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"🖥️  Using device: {DEVICE}")

# ========================
# Focal Loss - Zor örneklere odaklan
# ========================
class FocalLoss(nn.Module):
    """
    Focal Loss: Kolay örnekleri ignore et, zor örneklere odaklan.
    
    Eğer model bir örneği yüksek güvenle doğru tahmin ediyorsa → az loss
    Eğer model bir örneği düşük güvenle tahmin ediyorsa → yüksek loss
    
    Bu sayede model zayıf olduğu sınıflara odaklanır!
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Class weights
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# ========================
# Dataset
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
    print("📂 Loading dataset...")
    
    data_dir = Path(Config.DATA_DIR)
    image_paths = []
    labels = []
    
    class_folders = sorted([f for f in data_dir.iterdir() if f.is_dir()])
    class_to_idx = {cls.name: idx for idx, cls in enumerate(class_folders)}
    class_names = [cls.name for cls in class_folders]
    
    for class_folder in class_folders:
        class_idx = class_to_idx[class_folder.name]
        for img_path in class_folder.glob("*.jpg"):
            image_paths.append(str(img_path))
            labels.append(class_idx)
    
    print(f"✅ Found {len(image_paths)} images across {len(class_names)} classes")
    
    # Class distribution
    class_counts = Counter(labels)
    print("\n📊 Sınıf dağılımı:")
    for idx, count in sorted(class_counts.items()):
        print(f"   {class_names[idx]}: {count} resim")
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    with open(f"{Config.OUTPUT_DIR}/labels.txt", "w") as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    return image_paths, labels, class_names, class_counts


def compute_class_weights(labels, num_classes):
    """
    Sınıf ağırlıklarını hesapla.
    Az örneği olan sınıflar daha yüksek ağırlık alır.
    """
    class_counts = Counter(labels)
    total = len(labels)
    
    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        # Inverse frequency weighting
        weight = total / (num_classes * count)
        weights.append(weight)
    
    # Normalize
    weights = torch.FloatTensor(weights)
    weights = weights / weights.sum() * num_classes
    
    return weights

# ========================
# Transforms - Dengeli
# ========================
def get_train_transforms():
    """Dengeli augmentation - genelleme + sınıf koruması"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(Config.INPUT_SIZE),
        
        # Geometrik (orta seviye)
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),  # 180 → 90 (daha az ekstrem)
        transforms.RandomAffine(
            degrees=10,  # 15 → 10
            translate=(0.1, 0.1),
            scale=(0.85, 1.15),  # Daha dar aralık
        ),
        
        # Renk (orta seviye)
        transforms.ColorJitter(
            brightness=0.25,  # 0.4 → 0.25
            contrast=0.25,
            saturation=0.25,
            hue=0.1
        ),
        
        # Bulanıklık (düşük olasılık)
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ], p=0.2),  # 0.4 → 0.2
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
        # Random erasing (düşük)
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # 0.4 → 0.2
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# ========================
# Mixup (Hafifletilmiş)
# ========================
def mixup_data(x, y, alpha=0.1):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)  # En az %50 orijinal
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(Config.DEVICE)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ========================
# Model
# ========================
def create_model():
    print("🧠 Creating MobileNetV3-Large with balanced architecture...")
    
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    
    # Freeze less layers (daha fazla fine-tuning)
    for param in model.features[:8].parameters():  # 12 → 8
        param.requires_grad = False
    
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 1280),
        nn.Hardswish(),
        nn.Dropout(p=Config.DROPOUT_RATE),
        nn.BatchNorm1d(1280),
        nn.Linear(1280, 640),
        nn.Hardswish(),
        nn.Dropout(p=Config.DROPOUT_RATE * 0.8),  # İkinci dropout biraz daha az
        nn.Linear(640, Config.NUM_CLASSES)
    )
    
    return model.to(Config.DEVICE)

# ========================
# Training
# ========================
def train_epoch(model, dataloader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
    for images, labels in pbar:
        images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
        
        # Mixup (hafifletilmiş)
        if Config.USE_MIXUP and np.random.random() < 0.5:  # %50 olasılıkla
            images, labels_a, labels_b, lam = mixup_data(images, labels, Config.MIXUP_ALPHA)
            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{running_loss/len(pbar):.3f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Per-class tracking
    class_correct = [0] * Config.NUM_CLASSES
    class_total = [0] * Config.NUM_CLASSES
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
    
    # Calculate per-class accuracy
    per_class_acc = []
    for i in range(Config.NUM_CLASSES):
        if class_total[i] > 0:
            per_class_acc.append(100 * class_correct[i] / class_total[i])
        else:
            per_class_acc.append(0)
    
    return running_loss / len(dataloader), 100. * correct / total, per_class_acc


def main():
    print("\n" + "="*70)
    print("🧬 VisionVet-AI Balanced Training")
    print("   Tüm Bakterileri Dengeli Öğrenen Model")
    print("="*70 + "\n")
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Prepare data
    image_paths, labels, class_names, class_counts = prepare_data()
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\n📊 Training samples: {len(X_train)}")
    print(f"📊 Validation samples: {len(X_val)}")
    
    # Compute class weights
    if Config.USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(y_train, Config.NUM_CLASSES)
        print(f"\n⚖️  Class weights computed (range: {class_weights.min():.2f} - {class_weights.max():.2f})")
        class_weights = class_weights.to(Config.DEVICE)
    else:
        class_weights = None
    
    # Create datasets
    train_dataset = BacterialDataset(X_train, y_train, get_train_transforms())
    val_dataset = BacterialDataset(X_val, y_val, get_val_transforms())
    
    # Balanced sampler
    if Config.USE_BALANCED_SAMPLER:
        sample_weights = [1.0 / class_counts[label] for label in y_train]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, sampler=sampler, num_workers=4)
        print("⚖️  Balanced sampling enabled")
    else:
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Model
    model = create_model()
    
    # Loss function
    if Config.USE_FOCAL_LOSS:
        criterion = FocalLoss(gamma=Config.FOCAL_GAMMA, alpha=class_weights)
        print(f"🎯 Focal Loss enabled (gamma={Config.FOCAL_GAMMA})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Scheduler - her 15 epoch'ta LR düşür
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    # Training
    best_acc = 0.0
    best_min_class_acc = 0.0  # En düşük sınıf doğruluğu
    patience_counter = 0
    
    print(f"\n🎯 Hedef: Tüm sınıflarda dengeli doğruluk")
    print(f"   Patience: {Config.PATIENCE} epoch")
    print(f"   Max epochs: {Config.EPOCHS}")
    
    for epoch in range(Config.EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{Config.EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
        
        # Validate
        val_loss, val_acc, per_class_acc = validate(model, val_loader, criterion)
        
        # Scheduler step
        scheduler.step()
        
        # Stats
        min_class_acc = min(per_class_acc)
        max_class_acc = max(per_class_acc)
        
        print(f"\n📈 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"📉 Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.2f}%")
        print(f"📊 Per-class: Min={min_class_acc:.0f}%, Max={max_class_acc:.0f}%, Gap={max_class_acc-min_class_acc:.0f}%")
        
        # Overfitting check
        acc_gap = train_acc - val_acc
        if acc_gap > 10:
            print(f"⚠️  Train-Val gap = {acc_gap:.1f}% (watching for overfitting)")
        elif acc_gap < 5:
            print(f"✅ Train-Val gap = {acc_gap:.1f}% (healthy)")
        
        # Save best model (prefer high val_acc + high min_class_acc)
        combined_score = val_acc * 0.7 + min_class_acc * 0.3  # Val acc + min class acc
        
        if combined_score > best_acc * 0.7 + best_min_class_acc * 0.3:
            best_acc = val_acc
            best_min_class_acc = min_class_acc
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'per_class_acc': per_class_acc,
                'class_names': class_names
            }, f"{Config.OUTPUT_DIR}/best_model.pth")
            
            print(f"💾 Best model saved! Val: {best_acc:.2f}%, Min class: {best_min_class_acc:.0f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter} epoch(s)")
        
        # Show worst performing classes
        if epoch % 5 == 0:
            print(f"\n📋 En düşük 5 sınıf:")
            sorted_acc = sorted(enumerate(per_class_acc), key=lambda x: x[1])
            for idx, acc in sorted_acc[:5]:
                print(f"   ❌ {class_names[idx]}: {acc:.0f}%")
        
        # Early stopping
        if Config.EARLY_STOPPING and patience_counter >= Config.PATIENCE:
            print(f"\n⚠️  EARLY STOPPING at epoch {epoch+1}")
            print(f"   Best Val Accuracy: {best_acc:.2f}%")
            print(f"   Best Min Class Accuracy: {best_min_class_acc:.0f}%")
            break
    
    # Final results
    print(f"\n{'='*70}")
    print(f"🎉 Training completed!")
    print(f"   Best Validation Accuracy: {best_acc:.2f}%")
    print(f"   Best Min Class Accuracy: {best_min_class_acc:.0f}%")
    print(f"{'='*70}")
    
    # Load best model and show per-class accuracy
    checkpoint = torch.load(f"{Config.OUTPUT_DIR}/best_model.pth")
    print("\n📊 Final Per-Class Accuracy:")
    for idx, acc in enumerate(checkpoint['per_class_acc']):
        status = '✅' if acc >= 80 else '🟡' if acc >= 50 else '❌'
        print(f"   {status} {class_names[idx]}: {acc:.0f}%")
    
    # Export
    print("\n📦 Exporting model...")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Save PyTorch model
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': Config.NUM_CLASSES,
        'input_size': Config.INPUT_SIZE,
        'accuracy': best_acc,
        'class_names': class_names
    }, f"{Config.OUTPUT_DIR}/bacterial_classifier.pt")
    
    print(f"\n✅ Model saved to: {Config.OUTPUT_DIR}/bacterial_classifier.pt")
    print(f"✅ Labels saved to: {Config.OUTPUT_DIR}/labels.txt")


if __name__ == "__main__":
    main()
