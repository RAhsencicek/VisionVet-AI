#!/usr/bin/env python3
"""
TIF to JPG Converter for DIBaS Dataset
================================================================================
DIBaS dataset'i TIF formatında geliyor. Bu script tüm TIF dosyalarını
JPG formatına dönüştürür.
"""

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil

def convert_tif_to_jpg(data_dir="data/dibas"):
    """Convert all TIF files to JPG"""
    print("="*60)
    print("🔄 TIF → JPG Dönüştürücü")
    print("="*60)
    
    data_path = Path(data_dir)
    
    # Find all TIF files
    tif_files = list(data_path.rglob("*.tif")) + list(data_path.rglob("*.TIF"))
    
    print(f"\n📂 Toplam TIF dosyası: {len(tif_files)}")
    
    if not tif_files:
        print("❌ TIF dosyası bulunamadı!")
        return
    
    converted = 0
    failed = 0
    
    for tif_path in tqdm(tif_files, desc="Dönüştürülüyor"):
        try:
            # Open TIF
            with Image.open(tif_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Create JPG path
                jpg_path = tif_path.with_suffix('.jpg')
                
                # Save as JPG with good quality
                img.save(jpg_path, 'JPEG', quality=95)
                
                converted += 1
            
            # Remove original TIF to save space
            os.remove(tif_path)
            
        except Exception as e:
            print(f"\n❌ Hata: {tif_path.name} - {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print("📊 ÖZET")
    print(f"{'='*60}")
    print(f"   ✅ Dönüştürülen: {converted}")
    print(f"   ❌ Başarısız: {failed}")
    
    # Count total JPG files
    jpg_files = list(data_path.rglob("*.jpg"))
    print(f"\n   📸 Toplam JPG dosyası: {len(jpg_files)}")
    
    # Count per class
    print("\n   📋 Sınıf bazında:")
    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    for class_dir in class_dirs[:10]:  # Show first 10
        jpg_count = len(list(class_dir.glob("*.jpg")))
        print(f"      {class_dir.name}: {jpg_count} resim")
    if len(class_dirs) > 10:
        print(f"      ... ve {len(class_dirs) - 10} sınıf daha")
    
    print(f"\n🎉 Dönüştürme tamamlandı!")
    print(f"   Şimdi eğitimi başlatabilirsiniz.")


if __name__ == "__main__":
    convert_tif_to_jpg()
