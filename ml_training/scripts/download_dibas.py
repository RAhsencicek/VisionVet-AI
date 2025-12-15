#!/usr/bin/env python3
"""
DIBaS Dataset İndirme Scripti
================================================================================
Bu script 33 bakteri türünün resimlerini otomatik olarak indirir ve
doğru klasör yapısına çıkartır.

Kaynak: https://doctoral.matinf.uj.edu.pl/database/dibas/
"""

import os
import urllib.request
import zipfile
import shutil
import ssl
from pathlib import Path
from tqdm import tqdm

# SSL sertifika doğrulamasını devre dışı bırak (üniversite sunucusu için gerekli)
ssl._create_default_https_context = ssl._create_unverified_context

# Dataset URL'leri
DIBAS_URLS = {
    "Acinetobacter_baumanii": "https://doctoral.matinf.uj.edu.pl/database/dibas/Acinetobacter.baumanii.zip",
    "Actinomyces_israeli": "https://doctoral.matinf.uj.edu.pl/database/dibas/Actinomyces.israeli.zip",
    "Bacteroides_fragilis": "https://doctoral.matinf.uj.edu.pl/database/dibas/Bacteroides.fragilis.zip",
    "Bifidobacterium_spp": "https://doctoral.matinf.uj.edu.pl/database/dibas/Bifidobacterium.spp.zip",
    "Candida_albicans": "https://doctoral.matinf.uj.edu.pl/database/dibas/Candida.albicans.zip",
    "Clostridium_perfringens": "https://doctoral.matinf.uj.edu.pl/database/dibas/Clostridium.perfringens.zip",
    "Enterococcus_faecium": "https://doctoral.matinf.uj.edu.pl/database/dibas/Enterococcus.faecium.zip",
    "Enterococcus_faecalis": "https://doctoral.matinf.uj.edu.pl/database/dibas/Enterococcus.faecalis.zip",
    "Escherichia_coli": "https://doctoral.matinf.uj.edu.pl/database/dibas/Escherichia.coli.zip",
    "Fusobacterium_spp": "https://doctoral.matinf.uj.edu.pl/database/dibas/Fusobacterium.zip",
    "Lactobacillus_casei": "https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.casei.zip",
    "Lactobacillus_crispatus": "https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.crispatus.zip",
    "Lactobacillus_delbrueckii": "https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.delbrueckii.zip",
    "Lactobacillus_gasseri": "https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.gasseri.zip",
    "Lactobacillus_jehnsenii": "https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.jehnsenii.zip",
    "Lactobacillus_johnsonii": "https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.johnsonii.zip",
    "Lactobacillus_paracasei": "https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.paracasei.zip",
    "Lactobacillus_plantarum": "https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.plantarum.zip",
    "Lactobacillus_reuteri": "https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.reuteri.zip",
    "Lactobacillus_rhamnosus": "https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.rhamnosus.zip",
    "Lactobacillus_salivarius": "https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.salivarius.zip",
    "Listeria_monocytogenes": "https://doctoral.matinf.uj.edu.pl/database/dibas/Listeria.monocytogenes.zip",
    "Micrococcus_spp": "https://doctoral.matinf.uj.edu.pl/database/dibas/Micrococcus.spp.zip",
    "Neisseria_gonorrhoeae": "https://doctoral.matinf.uj.edu.pl/database/dibas/Neisseria.gonorrhoeae.zip",
    "Porphyromonas_gingivalis": "https://doctoral.matinf.uj.edu.pl/database/dibas/Porfyromonas.gingivalis.zip",
    "Propionibacterium_acnes": "https://doctoral.matinf.uj.edu.pl/database/dibas/Propionibacterium.acnes.zip",
    "Proteus_spp": "https://doctoral.matinf.uj.edu.pl/database/dibas/Proteus.zip",
    "Pseudomonas_aeruginosa": "https://doctoral.matinf.uj.edu.pl/database/dibas/Pseudomonas.aeruginosa.zip",
    "Staphylococcus_aureus": "https://doctoral.matinf.uj.edu.pl/database/dibas/Staphylococcus.aureus.zip",
    "Staphylococcus_epidermidis": "https://doctoral.matinf.uj.edu.pl/database/dibas/Staphylococcus.epidermidis.zip",
    "Staphylococcus_saprophiticus": "https://doctoral.matinf.uj.edu.pl/database/dibas/Staphylococcus.saprophiticus.zip",
    "Streptococcus_agalactiae": "https://doctoral.matinf.uj.edu.pl/database/dibas/Streptococcus.agalactiae.zip",
    "Veionella_spp": "https://doctoral.matinf.uj.edu.pl/database/dibas/Veionella.zip",
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path):
    """Download a file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def extract_zip(zip_path, target_dir):
    """Extract zip file and organize files"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)


def organize_files(target_dir):
    """Move jpg files to the class folder and clean up"""
    target_path = Path(target_dir)
    
    # Find all jpg files recursively
    jpg_files = list(target_path.rglob("*.jpg")) + list(target_path.rglob("*.JPG"))
    
    for jpg_file in jpg_files:
        # Move to class folder root
        new_path = target_path / jpg_file.name
        if jpg_file != new_path:
            shutil.move(str(jpg_file), str(new_path))
    
    # Remove empty subdirectories
    for subdir in target_path.iterdir():
        if subdir.is_dir():
            shutil.rmtree(subdir)


def main():
    print("="*60)
    print("🧬 DIBaS Dataset İndirme Scripti")
    print("   33 Bakteri Türü - Toplam ~660 Görüntü")
    print("="*60)
    
    # Create target directory
    data_dir = Path("data/dibas")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Temp directory for downloads
    temp_dir = Path("data/temp_downloads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    successful = 0
    failed = []
    
    for class_name, url in DIBAS_URLS.items():
        print(f"\n📥 İndiriliyor: {class_name}")
        
        class_dir = data_dir / class_name
        zip_path = temp_dir / f"{class_name}.zip"
        
        # Skip if already downloaded
        if class_dir.exists() and len(list(class_dir.glob("*.jpg"))) > 0:
            print(f"   ✅ Zaten mevcut ({len(list(class_dir.glob('*.jpg')))} resim)")
            successful += 1
            continue
        
        try:
            # Download
            download_file(url, str(zip_path))
            
            # Create class directory
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract
            print(f"   📦 Çıkartılıyor...")
            extract_zip(str(zip_path), str(class_dir))
            
            # Organize files
            organize_files(str(class_dir))
            
            # Count images
            num_images = len(list(class_dir.glob("*.jpg")))
            print(f"   ✅ Başarılı! {num_images} resim")
            
            successful += 1
            
            # Remove zip file
            os.remove(zip_path)
            
        except Exception as e:
            print(f"   ❌ Hata: {e}")
            failed.append(class_name)
    
    # Clean up temp directory
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    # Summary
    print("\n" + "="*60)
    print("📊 ÖZET")
    print("="*60)
    print(f"   ✅ Başarılı: {successful}/{len(DIBAS_URLS)}")
    
    if failed:
        print(f"   ❌ Başarısız: {len(failed)}")
        for name in failed:
            print(f"      - {name}")
    
    # Count total images
    total_images = sum(len(list((data_dir / c).glob("*.jpg"))) for c in data_dir.iterdir() if c.is_dir())
    print(f"\n   📸 Toplam görüntü sayısı: {total_images}")
    print(f"   📁 Dataset konumu: {data_dir.absolute()}")
    
    if successful == len(DIBAS_URLS):
        print("\n🎉 Dataset başarıyla indirildi!")
        print("   Şimdi 'python colab_generalized_training.py' çalıştırabilirsiniz.")
    else:
        print("\n⚠️  Bazı dosyalar indirilemedi.")
        print("   İnternet bağlantınızı kontrol edin ve tekrar deneyin.")


if __name__ == "__main__":
    main()
