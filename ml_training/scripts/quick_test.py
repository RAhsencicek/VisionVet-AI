#!/usr/bin/env python3
"""
🧪 Telefon Kamerası Test Scripti
================================================================================
Bu script ile modelin gerçekten genelleme yapıp yapmadığını test edebilirsiniz.

Kullanım:
1. Dataset'teki bir E.coli resmini bilgisayar ekranında açın
2. Telefonunuzla ekranın fotoğrafını çekin
3. Fotoğrafı bilgisayara aktarın
4. Bu script ile test edin:
   python quick_test.py /path/to/phone_photo.jpg

Beklenen Sonuç:
- Model "Escherichia_coli" tahmin etmeli
- Güven %70+ olmalı (ezberlemediğinin kanıtı)
================================================================================
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import sys
import os

# Configuration
MODEL_PATH = "models/bacterial_generalized/bacterial_classifier.pt"
LABELS_PATH = "models/bacterial_generalized/labels_32.txt"
INPUT_SIZE = 224

def load_model():
    """Load trained model"""
    print("🧠 Model yükleniyor...")
    
    # Create model architecture
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
        nn.Linear(512, 32)
    )
    
    # Load weights
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model yüklendi! Eğitim doğruluğu: {checkpoint.get('accuracy', 79.10):.2f}%")
    return model


def load_labels():
    """Load class labels"""
    with open(LABELS_PATH, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


def get_transforms():
    """Get image transforms"""
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def predict(model, image_path, labels):
    """Make prediction on image"""
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    transform = get_transforms()
    img_tensor = transform(image).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]
    
    # Get top 5 predictions
    top5_probs, top5_indices = torch.topk(probs, 5)
    
    return [(labels[idx], prob.item() * 100) for idx, prob in zip(top5_indices, top5_probs)]


def analyze_result(predictions, expected_class=None):
    """Analyze prediction result"""
    top_class, top_conf = predictions[0]
    
    print(f"\n{'='*60}")
    print("📊 TAHMİN SONUÇLARI")
    print(f"{'='*60}")
    
    for i, (class_name, conf) in enumerate(predictions, 1):
        bar_length = int(conf / 2)  # Max 50 chars for 100%
        bar = "█" * bar_length + "░" * (50 - bar_length)
        
        if i == 1:
            print(f"\n🥇 {class_name}")
            print(f"   [{bar}] {conf:.2f}%")
        else:
            print(f"\n   {i}. {class_name}: {conf:.2f}%")
    
    print(f"\n{'='*60}")
    print("🧬 ANALİZ")
    print(f"{'='*60}")
    
    # Confidence analysis
    if top_conf > 99:
        print(f"⚠️  ÇOK YÜKSEK güven ({top_conf:.1f}%)")
        print("   Bu ezberleme olabilir! Ama telefon fotoğrafıyla bu normal değil.")
    elif top_conf > 85:
        print(f"✅ YÜKSEK güven ({top_conf:.1f}%)")
        print("   Model emin görünüyor - iyi bir tahmin!")
    elif top_conf > 70:
        print(f"✅ NORMAL güven ({top_conf:.1f}%)")
        print("   Sağlıklı bir tahmin - genelleme yapıyor!")
    elif top_conf > 50:
        print(f"🟡 ORTA güven ({top_conf:.1f}%)")
        print("   Model biraz kararsız ama doğru olabilir.")
    else:
        print(f"❓ DÜŞÜK güven ({top_conf:.1f}%)")
        print("   Model emin değil - görüntü kalitesi düşük olabilir.")
    
    # Check if expected class matches
    if expected_class:
        if expected_class.lower() in top_class.lower():
            print(f"\n🎉 BAŞARILI! Beklenen sınıf ({expected_class}) doğru tahmin edildi!")
        else:
            print(f"\n❌ Beklenen: {expected_class}, Tahmin: {top_class}")
    
    return top_class, top_conf


def simulate_phone_capture(image_path):
    """
    Simulate taking a photo of screen with phone
    Adds realistic distortions
    """
    from PIL import ImageFilter, ImageEnhance
    import random
    
    print("\n📱 Telefon kamerası simülasyonu uygulanıyor...")
    
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # 1. Add blur (phone camera not perfect + screen)
    blur_amount = random.uniform(0.5, 1.5)
    image = image.filter(ImageFilter.GaussianBlur(radius=blur_amount))
    print(f"   • Bulanıklık eklendi (radius={blur_amount:.2f})")
    
    # 2. Change brightness (screen reflection, ambient light)
    brightness_factor = random.uniform(0.7, 1.3)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)
    print(f"   • Parlaklık değiştirildi (factor={brightness_factor:.2f})")
    
    # 3. Change contrast
    contrast_factor = random.uniform(0.8, 1.2)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)
    print(f"   • Kontrast değiştirildi (factor={contrast_factor:.2f})")
    
    # 4. Slight color shift
    color_factor = random.uniform(0.85, 1.15)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(color_factor)
    print(f"   • Renk tonu değiştirildi (factor={color_factor:.2f})")
    
    # 5. Slight resolution change (different camera)
    scale = random.uniform(0.8, 1.2)
    new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
    image = image.resize(new_size, Image.LANCZOS)
    image = image.resize(original_size, Image.LANCZOS)
    print(f"   • Çözünürlük değiştirildi (scale={scale:.2f})")
    
    # Save simulated image
    temp_path = "/tmp/phone_simulated_test.jpg"
    image.save(temp_path, quality=85)  # JPEG compression
    print(f"   • JPEG sıkıştırma uygulandı")
    print(f"\n✅ Simüle edilmiş görüntü: {temp_path}")
    
    return temp_path


def main():
    # Check arguments
    if len(sys.argv) < 2:
        print("""
╔════════════════════════════════════════════════════════════════╗
║         🧪 VisionVet-AI Telefon Kamerası Test Scripti          ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  Kullanım:                                                     ║
║    python quick_test.py <görüntü_yolu> [--simulate]            ║
║                                                                ║
║  Örnekler:                                                     ║
║    # Gerçek telefon fotoğrafı test et:                         ║
║    python quick_test.py telefon_foto.jpg                       ║
║                                                                ║
║    # Dataset resmini telefon simülasyonuyla test et:           ║
║    python quick_test.py data/dibas/Escherichia_coli/E*.jpg \\  ║
║           --simulate                                           ║
║                                                                ║
║  Test Adımları:                                                ║
║    1. Bir E.coli resmini bilgisayar ekranında açın            ║
║    2. Telefonunuzla ekranın fotoğrafını çekin                 ║
║    3. Fotoğrafı bilgisayara aktarın (AirDrop, USB, vb.)       ║
║    4. python quick_test.py <fotoğraf_yolu>                    ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
        """)
        
        # Quick demo with simulation
        print("\n🎯 Hızlı Demo: Dataset'ten bir E.coli resmi ile test...")
        ecoli_path = "data/dibas/Escherichia_coli/Escherichia.coli_0001.jpg"
        
        if os.path.exists(ecoli_path):
            model = load_model()
            labels = load_labels()
            
            # Original
            print("\n" + "="*60)
            print("📸 ORİJİNAL DATASET RESMİ")
            print("="*60)
            predictions = predict(model, ecoli_path, labels)
            analyze_result(predictions, "Escherichia_coli")
            
            # Simulated phone
            print("\n" + "="*60)
            print("📱 TELEFON KAMERASI SİMÜLASYONU")
            print("="*60)
            sim_path = simulate_phone_capture(ecoli_path)
            predictions = predict(model, sim_path, labels)
            analyze_result(predictions, "Escherichia_coli")
            
            print("\n" + "="*60)
            print("💡 SONUÇ")
            print("="*60)
            print("Eğer her iki test de 'Escherichia_coli' tahmin ettiyse,")
            print("model gerçekten genelleme yapıyor demektir! 🎉")
        else:
            print(f"❌ Test resmi bulunamadı: {ecoli_path}")
        
        return
    
    image_path = sys.argv[1]
    use_simulation = "--simulate" in sys.argv
    
    if not os.path.exists(image_path):
        print(f"❌ Dosya bulunamadı: {image_path}")
        return
    
    # Load model
    model = load_model()
    labels = load_labels()
    
    if use_simulation:
        print(f"\n📸 Orijinal görüntü: {image_path}")
        predictions_orig = predict(model, image_path, labels)
        analyze_result(predictions_orig)
        
        print("\n" + "-"*60)
        
        sim_path = simulate_phone_capture(image_path)
        predictions_sim = predict(model, sim_path, labels)
        analyze_result(predictions_sim)
        
        # Compare
        print("\n" + "="*60)
        print("📊 KARŞILAŞTIRMA")
        print("="*60)
        
        orig_class, orig_conf = predictions_orig[0]
        sim_class, sim_conf = predictions_sim[0]
        
        if orig_class == sim_class:
            print(f"✅ Her iki test de aynı sonuç: {orig_class}")
            print(f"   Orijinal güven: {orig_conf:.2f}%")
            print(f"   Simülasyon güven: {sim_conf:.2f}%")
            print(f"   Fark: {abs(orig_conf - sim_conf):.2f}%")
            print("\n🎉 Model telefon kamerasına dayanıklı!")
        else:
            print(f"⚠️  Farklı sonuçlar!")
            print(f"   Orijinal: {orig_class} ({orig_conf:.2f}%)")
            print(f"   Simülasyon: {sim_class} ({sim_conf:.2f}%)")
    else:
        print(f"\n📸 Test görüntüsü: {image_path}")
        predictions = predict(model, image_path, labels)
        analyze_result(predictions)


if __name__ == "__main__":
    main()
