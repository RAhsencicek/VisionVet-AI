"""
VisionVet-AI Model Test Script with Generalization Analysis
================================================================================
Bu script modelin gerçekten öğrenip öğrenmediğini test eder.

Test Tipleri:
1. Dataset resimleri (beklenen: %90-95)
2. Telefon kamerasından çekilen resimler (beklenen: %85-92)
3. İnternetten indirilen resimler (beklenen: %80-90)

Eğer dataset resimleri %99 ama telefon resimleri %50 ise → EZBER VAR!
Eğer hepsi %85-95 arasındaysa → GERÇEK ÖĞRENME ✓

Author: VisionVet Team
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import argparse
import os
import json
from pathlib import Path

# ========================
# Configuration
# ========================
class Config:
    MODEL_PATH = "models/bacterial_generalized/best_model.pth"
    LABELS_PATH = "models/bacterial_generalized/labels_33.txt"
    INPUT_SIZE = 224
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Generalization thresholds
    GOOD_CONFIDENCE_MIN = 0.7
    GOOD_CONFIDENCE_MAX = 0.98  # %99+ = ezberleme olabilir!
    SUSPICIOUS_THRESHOLD = 0.99  # %99+ güven → şüpheli

# ========================
# Model Creation
# ========================
def load_model():
    """Load the trained generalized model"""
    print(f"🧠 Loading model from {Config.MODEL_PATH}...")
    
    # Create model architecture
    model = models.mobilenet_v3_large(pretrained=False)
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 1280),
        nn.Hardswish(),
        nn.Dropout(p=0.5),
        nn.BatchNorm1d(1280),
        nn.Linear(1280, 512),
        nn.Hardswish(),
        nn.Dropout(p=0.5),
        nn.Linear(512, 33)  # 33 bacterial classes
    )
    
    # Load weights
    checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded! Best accuracy: {checkpoint.get('best_acc', 'N/A'):.2f}%")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Model loaded!")
    
    model = model.to(Config.DEVICE)
    model.eval()
    return model


def load_labels():
    """Load class labels"""
    labels = []
    with open(Config.LABELS_PATH, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    print(f"📋 Loaded {len(labels)} class labels")
    return labels

# ========================
# Transforms
# ========================
def get_basic_transform():
    """Normal transform for single prediction"""
    return transforms.Compose([
        transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_tta_transforms():
    """Test-Time Augmentation transforms"""
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
        # Rotate 180
        transforms.Compose([
            transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
            transforms.RandomRotation((180, 180)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # Brighter
        transforms.Compose([
            transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
            transforms.ColorJitter(brightness=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # Darker
        transforms.Compose([
            transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
            transforms.ColorJitter(brightness=-0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    ]

# ========================
# Prediction Functions
# ========================
def predict_single(model, image_path, labels, use_tta=False):
    """Single image prediction with optional TTA"""
    image = Image.open(image_path).convert('RGB')
    
    if use_tta:
        # Test-Time Augmentation
        tta_transforms = get_tta_transforms()
        all_outputs = []
        
        with torch.no_grad():
            for tta_transform in tta_transforms:
                img_tensor = tta_transform(image).unsqueeze(0).to(Config.DEVICE)
                output = model(img_tensor)
                all_outputs.append(torch.softmax(output, dim=1))
        
        # Average predictions
        avg_output = torch.mean(torch.stack(all_outputs), dim=0)
        probs = avg_output[0].cpu().numpy()
    else:
        # Normal prediction
        transform = get_basic_transform()
        img_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)
        
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
    
    # Get top predictions
    top_indices = np.argsort(probs)[::-1][:5]
    top_predictions = [(labels[i], probs[i] * 100) for i in top_indices]
    
    return top_predictions, probs


def analyze_prediction(predictions, image_source="unknown"):
    """Analyze if prediction looks like memorization or real learning"""
    top_conf = predictions[0][1]
    second_conf = predictions[1][1]
    
    print(f"\n📊 Analiz ({image_source}):")
    print(f"   Top-1: {predictions[0][0]} ({top_conf:.2f}%)")
    print(f"   Top-2: {predictions[1][0]} ({second_conf:.2f}%)")
    
    # Confidence analysis
    if top_conf > 99:
        print(f"   ⚠️  ÇOK YÜKSEK güven ({top_conf:.1f}%) - Ezberleme olabilir!")
        return "suspicious"
    elif top_conf > 95:
        print(f"   🟡 Yüksek güven ({top_conf:.1f}%) - Dikkatli ol")
        return "high"
    elif top_conf > 70:
        print(f"   ✅ Normal güven ({top_conf:.1f}%) - Sağlıklı öğrenme")
        return "healthy"
    else:
        print(f"   ❓ Düşük güven ({top_conf:.1f}%) - Model emin değil")
        return "uncertain"


def simulate_phone_capture(image_path, output_path=None):
    """
    Simulates taking a photo of the screen with phone camera.
    Adds noise, blur, perspective distortion, and lighting changes.
    """
    from PIL import ImageFilter, ImageEnhance
    import random
    
    image = Image.open(image_path).convert('RGB')
    
    # 1. Add slight blur (phone camera not perfect)
    image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    
    # 2. Change brightness (screen reflection)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))
    
    # 3. Change contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(random.uniform(0.9, 1.1))
    
    # 4. Add slight color shift
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.uniform(0.9, 1.1))
    
    # 5. Resize slightly (different resolution)
    w, h = image.size
    new_size = (int(w * random.uniform(0.9, 1.1)), int(h * random.uniform(0.9, 1.1)))
    image = image.resize(new_size, Image.LANCZOS)
    image = image.resize((w, h), Image.LANCZOS)  # Back to original size
    
    if output_path:
        image.save(output_path)
        print(f"📱 Simulated phone capture saved to: {output_path}")
    
    return image

# ========================
# Main Test Functions
# ========================
def test_generalization(model, labels, test_dir):
    """Test model on a directory of images"""
    print(f"\n📂 Testing images in: {test_dir}")
    
    test_path = Path(test_dir)
    image_files = list(test_path.glob("*.jpg")) + list(test_path.glob("*.png")) + list(test_path.glob("*.jpeg"))
    
    if not image_files:
        print("❌ No images found!")
        return
    
    results = []
    for img_path in image_files:
        print(f"\n{'='*50}")
        print(f"🖼️  Testing: {img_path.name}")
        
        predictions, probs = predict_single(model, str(img_path), labels, use_tta=True)
        status = analyze_prediction(predictions, img_path.name)
        
        results.append({
            'file': img_path.name,
            'prediction': predictions[0][0],
            'confidence': predictions[0][1],
            'status': status
        })
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 ÖZET")
    print(f"{'='*50}")
    
    healthy = sum(1 for r in results if r['status'] == 'healthy')
    suspicious = sum(1 for r in results if r['status'] == 'suspicious')
    
    print(f"   Toplam test: {len(results)}")
    print(f"   Sağlıklı tahminler: {healthy} ({100*healthy/len(results):.1f}%)")
    print(f"   Şüpheli (ezberleme?): {suspicious} ({100*suspicious/len(results):.1f}%)")
    
    if suspicious > healthy:
        print("\n⚠️  Model ezberleme yapıyor olabilir!")
        print("   Augmentation'ı artırın veya daha fazla veri ekleyin.")
    else:
        print("\n✅ Model sağlıklı öğreniyor görünüyor!")


def compare_original_vs_phone(model, labels, original_image):
    """Compare prediction on original vs simulated phone capture"""
    print(f"\n{'='*60}")
    print("📱 ORİJİNAL vs TELEFON KAMERASI TESTİ")
    print(f"{'='*60}")
    
    # Original prediction
    print(f"\n1️⃣ Orijinal resim: {original_image}")
    orig_preds, _ = predict_single(model, original_image, labels, use_tta=True)
    orig_status = analyze_prediction(orig_preds, "Orijinal")
    
    # Simulate phone capture
    print(f"\n2️⃣ Telefon kamerası simülasyonu...")
    phone_image = simulate_phone_capture(original_image)
    
    # Save temporarily
    temp_path = "/tmp/phone_simulated.jpg"
    phone_image.save(temp_path)
    
    phone_preds, _ = predict_single(model, temp_path, labels, use_tta=True)
    phone_status = analyze_prediction(phone_preds, "Telefon Kamerası")
    
    # Compare
    print(f"\n{'='*60}")
    print("📊 KARŞILAŞTIRMA")
    print(f"{'='*60}")
    
    same_prediction = orig_preds[0][0] == phone_preds[0][0]
    conf_diff = abs(orig_preds[0][1] - phone_preds[0][1])
    
    if same_prediction:
        print(f"✅ Aynı tahmin: {orig_preds[0][0]}")
        print(f"   Orijinal güven: {orig_preds[0][1]:.2f}%")
        print(f"   Telefon güven:  {phone_preds[0][1]:.2f}%")
        print(f"   Fark: {conf_diff:.2f}%")
        
        if conf_diff < 5:
            print(f"\n🎉 MÜKEMMEL! Model telefon kamerasını iyi tolere ediyor!")
        elif conf_diff < 15:
            print(f"\n👍 İYİ! Kabul edilebilir fark.")
        else:
            print(f"\n⚠️  Fark biraz yüksek, ama aynı sonuç.")
    else:
        print(f"❌ FARKLI TAHMİN!")
        print(f"   Orijinal: {orig_preds[0][0]} ({orig_preds[0][1]:.2f}%)")
        print(f"   Telefon:  {phone_preds[0][0]} ({phone_preds[0][1]:.2f}%)")
        print(f"\n⚠️  Model telefon kamerası görüntülerini iyi tanıyamıyor.")
        print("   Daha fazla augmentation veya veri çeşitliliği gerekebilir.")
    
    # Cleanup
    os.remove(temp_path)


def main():
    parser = argparse.ArgumentParser(description='VisionVet-AI Model Test Script')
    parser.add_argument('--image', type=str, help='Single image to test')
    parser.add_argument('--dir', type=str, help='Directory of images to test')
    parser.add_argument('--compare', type=str, help='Compare original vs phone capture')
    parser.add_argument('--tta', action='store_true', help='Use Test-Time Augmentation')
    parser.add_argument('--model', type=str, default=Config.MODEL_PATH, help='Model path')
    parser.add_argument('--labels', type=str, default=Config.LABELS_PATH, help='Labels path')
    
    args = parser.parse_args()
    
    # Update config
    Config.MODEL_PATH = args.model
    Config.LABELS_PATH = args.labels
    
    # Load model and labels
    model = load_model()
    labels = load_labels()
    
    if args.compare:
        compare_original_vs_phone(model, labels, args.compare)
    elif args.dir:
        test_generalization(model, labels, args.dir)
    elif args.image:
        print(f"\n🖼️  Testing: {args.image}")
        predictions, _ = predict_single(model, args.image, labels, use_tta=args.tta)
        analyze_prediction(predictions, args.image)
        
        print(f"\n📋 Top-5 Tahminler:")
        for i, (label, conf) in enumerate(predictions, 1):
            print(f"   {i}. {label}: {conf:.2f}%")
    else:
        print("❌ Lütfen --image, --dir veya --compare argümanlarından birini kullanın.")
        print("\nÖrnek kullanım:")
        print("  python test_generalization.py --image test.jpg")
        print("  python test_generalization.py --image test.jpg --tta")
        print("  python test_generalization.py --dir test_images/")
        print("  python test_generalization.py --compare original_ecoli.jpg")


if __name__ == "__main__":
    main()
