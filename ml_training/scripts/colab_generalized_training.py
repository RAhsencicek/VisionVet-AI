"""
=================================================================================
🧬 VisionVet-AI: Ezberlemeden Öğrenen Model Eğitimi (Google Colab Version)
=================================================================================

Bu notebook'u Google Colab'da çalıştırarak GPU üzerinde hızlı eğitim yapabilirsiniz.

Adımlar:
1. Bu dosyayı Colab'a yükleyin veya GitHub'dan açın
2. Runtime > Change runtime type > GPU seçin
3. Tüm hücreleri çalıştırın

Colab'da çalıştırmak için bu dosyayı .ipynb formatına dönüştürün veya
doğrudan Python dosyası olarak çalıştırın.
=================================================================================
"""

# ========================
# 1. SETUP - Gerekli Kütüphaneler
# ========================

# Colab'da çalışıyorsanız, bu satırı uncomment edin:
# !pip install torch torchvision tqdm scikit-learn onnx

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
import urllib.request
import zipfile

print("✅ Kütüphaneler yüklendi!")
print(f"🖥️  PyTorch version: {torch.__version__}")
print(f"🖥️  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")

# ========================
# 2. DATASET İNDİRME - DIBaS Dataset
# ========================

def download_dibas_dataset(target_dir="data/dibas"):
    """DIBaS dataset'ini indir"""
    os.makedirs(target_dir, exist_ok=True)
    
    # DIBaS dataset URL (örnek - gerçek URL'yi kontrol edin)
    # Not: DIBaS dataset'i genellikle manuel indirme gerektirir
    print("📂 DIBaS Dataset İndirme Rehberi:")
    print("   1. https://github.com/gallardorafael/DIBaS adresine gidin")
    print("   2. Dataset'i indirin")
    print(f"   3. {target_dir} klasörüne çıkartın")
    print("   4. Her bakteri sınıfı ayrı bir klasörde olmalı")
    print("\n   Klasör yapısı:")
    print("   data/dibas/")
    print("   ├── Acinetobacter_baumanii/")
    print("   │   ├── 001.jpg")
    print("   │   └── ...")
    print("   ├── Escherichia_coli/")
    print("   │   ├── 001.jpg")
    print("   │   └── ...")
    print("   └── ...")
    
    # Check if data exists
    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
        print(f"\n✅ Dataset zaten mevcut: {target_dir}")
        return True
    else:
        print(f"\n⚠️  Dataset bulunamadı. Lütfen manuel olarak indirin.")
        return False

# ========================
# 3. CONFIGURATION
# ========================

class Config:
    # Paths
    DATA_DIR = "data/dibas"
    OUTPUT_DIR = "models/bacterial_generalized"
    
    # Model
    MODEL_NAME = "mobilenet_v3_large"
    NUM_CLASSES = 33
    INPUT_SIZE = 224
    
    # Training - Generalization optimized
    BATCH_SIZE = 16
    EPOCHS = 100
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 1e-2
    
    # Early Stopping
    EARLY_STOPPING = True
    PATIENCE = 10
    
    # Generalization Techniques
    DROPOUT_RATE = 0.5
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.2
    USE_MIXUP = True
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n🖥️  Device: {Config.DEVICE}")

# ========================
# 4. MIXUP & LABEL SMOOTHING
# ========================

def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation: İki resmi ve etiketlerini karıştır"""
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
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing: Aşırı güvenli tahminleri önle"""
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
# 5. DATASET CLASS
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
# 6. AGGRESSIVE TRANSFORMS
# ========================

def get_train_transforms():
    """AGRESİF Augmentation - Model ezberleyemez!"""
    return transforms.Compose([
        transforms.Resize((280, 280)),
        transforms.RandomCrop(Config.INPUT_SIZE),
        
        # Geometrik dönüşümler
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(180),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.15, 0.15),
            scale=(0.75, 1.25),
            shear=10
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        
        # Renk dönüşümleri
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.15
        ),
        
        # Bulanıklık
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0))
        ], p=0.4),
        
        transforms.RandomGrayscale(p=0.1),
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
        transforms.RandomErasing(p=0.4, scale=(0.02, 0.15)),
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# ========================
# 7. MODEL CREATION
# ========================

def create_model():
    """Güçlü regularization ile MobileNetV3-Large"""
    print("🧠 Creating MobileNetV3-Large with strong regularization...")
    
    model = models.mobilenet_v3_large(pretrained=True)
    
    # Erken katmanları dondur
    for param in model.features[:12].parameters():
        param.requires_grad = False
    
    # Güçlü dropout ile classifier
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 1280),
        nn.Hardswish(),
        nn.Dropout(p=Config.DROPOUT_RATE),
        nn.BatchNorm1d(1280),
        nn.Linear(1280, 512),
        nn.Hardswish(),
        nn.Dropout(p=Config.DROPOUT_RATE),
        nn.Linear(512, Config.NUM_CLASSES)
    )
    
    return model.to(Config.DEVICE)

# ========================
# 8. TRAINING FUNCTIONS
# ========================

def prepare_data():
    """Dataset'i yükle ve böl"""
    print("📂 Loading dataset...")
    
    data_dir = Path(Config.DATA_DIR)
    image_paths = []
    labels = []
    class_names = []
    
    class_folders = sorted([f for f in data_dir.iterdir() if f.is_dir()])
    class_to_idx = {cls.name: idx for idx, cls in enumerate(class_folders)}
    class_names = [cls.name for cls in class_folders]
    
    for class_folder in class_folders:
        class_idx = class_to_idx[class_folder.name]
        for img_path in class_folder.glob("*.jpg"):
            image_paths.append(str(img_path))
            labels.append(class_idx)
    
    print(f"✅ Found {len(image_paths)} images across {len(class_names)} classes")
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    with open(f"{Config.OUTPUT_DIR}/labels_33.txt", "w") as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    return image_paths, labels, class_names


def train_epoch(model, dataloader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
    for images, labels in pbar:
        images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
        
        # Mixup
        if Config.USE_MIXUP:
            images, labels_a, labels_b, lam = mixup_data(images, labels, Config.MIXUP_ALPHA)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        if Config.USE_MIXUP:
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        
        if Config.USE_MIXUP:
            correct += (lam * predicted.eq(labels_a).float() + 
                       (1 - lam) * predicted.eq(labels_b).float()).sum().item()
        else:
            correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{running_loss/len(pbar):.3f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss / len(dataloader), 100. * correct / total


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
# 9. MAIN TRAINING
# ========================

def train():
    print("\n" + "="*70)
    print("🧬 VisionVet-AI Generalized Model Training")
    print("   Ezberlemeden Öğrenen Model Eğitimi")
    print("="*70 + "\n")
    
    # Check dataset
    if not download_dibas_dataset(Config.DATA_DIR):
        print("\n❌ Dataset bulunamadı. Lütfen önce dataset'i indirin.")
        return
    
    # Prepare data
    image_paths, labels, class_names = prepare_data()
    
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\n📊 Training samples: {len(X_train)}")
    print(f"📊 Validation samples: {len(X_val)}")
    
    print(f"\n🎯 Aktif Generalization Teknikleri:")
    print(f"   • Aggressive Augmentation: ✓")
    print(f"   • Mixup (α={Config.MIXUP_ALPHA}): ✓")
    print(f"   • Label Smoothing ({Config.LABEL_SMOOTHING}): ✓")
    print(f"   • Dropout ({Config.DROPOUT_RATE*100}%): ✓")
    print(f"   • Weight Decay ({Config.WEIGHT_DECAY}): ✓")
    print(f"   • Early Stopping (patience={Config.PATIENCE}): ✓")
    
    # Create datasets
    train_dataset = BacterialDataset(X_train, y_train, get_train_transforms())
    val_dataset = BacterialDataset(X_val, y_val, get_val_transforms())
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, 
                              num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Create model
    model = create_model()
    
    # Loss and optimizer
    criterion = LabelSmoothingLoss(Config.NUM_CLASSES, smoothing=Config.LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Training loop
    best_acc = 0.0
    patience_counter = 0
    
    for epoch in range(Config.EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{Config.EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*60}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step()
        
        print(f"\n📈 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"📉 Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.2f}%")
        
        acc_gap = train_acc - val_acc
        if acc_gap > 5:
            print(f"\n⚠️  WARNING: Train-Val gap = {acc_gap:.1f}% (possible overfitting)")
        elif acc_gap < 3:
            print(f"\n✅ GOOD: Train-Val gap = {acc_gap:.1f}% (healthy learning)")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_acc': best_acc,
            }, f"{Config.OUTPUT_DIR}/best_model.pth")
            print(f"💾 Best model saved! Accuracy: {best_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter} epoch(s)")
        
        if Config.EARLY_STOPPING and patience_counter >= Config.PATIENCE:
            print(f"\n⚠️  EARLY STOPPING at epoch {epoch+1}")
            break
    
    print(f"\n{'='*60}")
    print(f"🎉 Training completed! Best validation accuracy: {best_acc:.2f}%")
    print(f"{'='*60}")
    
    print("\n💡 Model artık telefon kamerasından çekilen resimleri")
    print("   daha iyi tanıyacak çünkü ezberlemek yerine öğrendi!")

# ========================
# 10. RUN TRAINING
# ========================

if __name__ == "__main__":
    train()
