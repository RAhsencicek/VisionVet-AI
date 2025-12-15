"""
Dataset Kontrolü ve İstatistik Scripti
VisionVet-AI için dataset'i analiz eder ve raporlar
"""

import os
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

def check_dataset(data_dir="data/dibas"):
    """Dataset'i kontrol et ve rapor üret"""
    
    print("="*60)
    print("📊 DATASET ANALİZİ")
    print("="*60)
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ HATA: {data_dir} bulunamadı!")
        print(f"\nLütfen önce dataset'i indirin:")
        print(f"git clone https://github.com/ihoflaz/bacterial-colony-classification.git dataset_temp")
        print(f"mkdir -p {data_dir}")
        print(f"cp -r dataset_temp/images/* {data_dir}/")
        return
    
    # Sınıfları topla
    classes = {}
    total_images = 0
    
    for class_folder in sorted(data_path.iterdir()):
        if class_folder.is_dir():
            # Görüntüleri say
            images = list(class_folder.glob("*.jpg")) + \
                    list(class_folder.glob("*.jpeg")) + \
                    list(class_folder.glob("*.png"))
            
            count = len(images)
            classes[class_folder.name] = count
            total_images += count
    
    # Rapor
    print(f"\n📁 Dataset Yolu: {data_dir}")
    print(f"📊 Toplam Sınıf Sayısı: {len(classes)}")
    print(f"🖼️  Toplam Görüntü Sayısı: {total_images}")
    print(f"📈 Ortalama Görüntü/Sınıf: {total_images/len(classes):.1f}")
    
    # Detaylı istatistikler
    print("\n" + "="*60)
    print("SINIF BAZINDA İSTATİSTİKLER")
    print("="*60)
    
    # Sıralı liste
    sorted_classes = sorted(classes.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Sınıf Adı':<40} {'Görüntü Sayısı':<15} {'Durum'}")
    print("-"*60)
    
    min_count = min(classes.values())
    max_count = max(classes.values())
    avg_count = total_images / len(classes)
    
    for class_name, count in sorted_classes:
        # Durum kontrolü
        if count < avg_count * 0.5:
            status = "⚠️ Az"
        elif count > avg_count * 1.5:
            status = "⚡ Çok"
        else:
            status = "✅ Normal"
        
        print(f"{class_name:<40} {count:<15} {status}")
    
    # Class imbalance uyarısı
    print("\n" + "="*60)
    print("CLASS IMBALANCE ANALİZİ")
    print("="*60)
    
    imbalance_ratio = max_count / min_count
    print(f"En Fazla Görüntü: {max_count} ({sorted_classes[0][0]})")
    print(f"En Az Görüntü: {min_count} ({sorted_classes[-1][0]})")
    print(f"İmbalance Oranı: {imbalance_ratio:.2f}x")
    
    if imbalance_ratio > 3:
        print("\n⚠️  UYARI: Ciddi class imbalance var!")
        print("   Çözüm: Class weights kullanın veya oversampling yapın")
    elif imbalance_ratio > 2:
        print("\n⚡ DİKKAT: Orta seviye imbalance var")
        print("   Önerilen: Class weights kullanın")
    else:
        print("\n✅ Dataset dengeli görünüyor")
    
    # Train/Val split önerisi
    print("\n" + "="*60)
    print("TRAIN/VAL SPLIT ÖNERİSİ")
    print("="*60)
    
    train_count = int(total_images * 0.8)
    val_count = total_images - train_count
    
    print(f"Training Set (80%): {train_count} görüntü")
    print(f"Validation Set (20%): {val_count} görüntü")
    print(f"Test Set (Opsiyonel): Kendi verilerinizi kullanın")
    
    # Disk kullanımı
    print("\n" + "="*60)
    print("DİSK KULLANIMI")
    print("="*60)
    
    total_size = 0
    for class_folder in data_path.iterdir():
        if class_folder.is_dir():
            for img in class_folder.glob("*"):
                total_size += img.stat().st_size
    
    total_size_mb = total_size / (1024 * 1024)
    print(f"Toplam Boyut: {total_size_mb:.1f} MB")
    print(f"Ortalama Görüntü Boyutu: {total_size_mb/total_images:.2f} MB")
    
    # Öneriler
    print("\n" + "="*60)
    print("ÖNERİLER")
    print("="*60)
    
    if total_images < 3000:
        print("⚠️  Dataset küçük! Data augmentation ŞİDDETLE önerilir")
        print("   - Random rotation, flip, color jitter kullanın")
        print("   - Transfer learning yapın (ImageNet pretrained)")
    
    if len(classes) < 33:
        print(f"⚠️  {33 - len(classes)} sınıf eksik!")
        print("   Tüm bakteriler için veri toplayın")
    
    if len(classes) > 33:
        print(f"⚠️  Fazladan {len(classes) - 33} sınıf var!")
        print("   labels_33.txt dosyasını güncelleyin")
    
    print("\n✅ Dataset kontrolü tamamlandı!")
    print("\nSıradaki adım:")
    print("  python train_bacterial_model.py")
    
    return classes

if __name__ == "__main__":
    import sys
    
    # Komut satırı argümanı
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/dibas"
    
    classes = check_dataset(data_dir)
    
    # Görselleştirme (opsiyonel)
    if classes and len(classes) > 0:
        try:
            import matplotlib.pyplot as plt
            
            # Bar chart
            sorted_classes = sorted(classes.items(), key=lambda x: x[1], reverse=True)
            names = [c[0][:20] for c in sorted_classes]  # İlk 20 karakter
            counts = [c[1] for c in sorted_classes]
            
            plt.figure(figsize=(15, 8))
            plt.bar(range(len(names)), counts, color='steelblue')
            plt.xlabel('Sınıf Adı')
            plt.ylabel('Görüntü Sayısı')
            plt.title('Sınıf Bazında Görüntü Dağılımı')
            plt.xticks(range(len(names)), names, rotation=90, ha='right')
            plt.tight_layout()
            plt.savefig('ml_training/dataset_distribution.png', dpi=150)
            print(f"\n📊 Görselleştirme kaydedildi: ml_training/dataset_distribution.png")
        except ImportError:
            print("\nℹ️  Matplotlib yüklü değil, görselleştirme atlandı")
