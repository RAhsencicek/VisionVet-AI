# 🚀 Hızlı Başlangıç - Model Eğitimi

## ⚡ 5 Dakikada Başlangıç

### 1. Dataset İndir (2 dk)
```bash
cd /Users/mac/VisionVet-AI
git clone https://github.com/ihoflaz/bacterial-colony-classification.git dataset_temp
mkdir -p data/dibas
cp -r dataset_temp/images/* data/dibas/
rm -rf dataset_temp
```

### 2. Environment Hazırla (2 dk)
```bash
cd ml_training
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Dataset Kontrolü (30 sn)
```bash
python check_dataset.py
```

### 4. Eğitimi Başlat (1 dk setup, sonra bekle)
```bash
python train_bacterial_model.py
```

---

## 🎯 Hızlı Karar Rehberi

### Şu Anda Ne Yapmalıyım?

```
┌─────────────────────────────────────┐
│  Mevcut model %94 doğruluk sağlıyor │
│  31/33 sınıf mükemmel çalışıyor     │
└─────────────────────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Acil mi?            │
    └─────────────────────┘
         │           │
       EVET        HAYIR
         │           │
         ▼           ▼
    ┌─────────┐  ┌──────────────┐
    │ Şimdilik│  │ Yeniden Eğit │
    │ Kullan  │  │ (4-6 saat)   │
    └─────────┘  └──────────────┘
```

---

## 💰 Maliyet Hesaplama

| Seçenek | Süre | Maliyet | Accuracy |
|---------|------|---------|----------|
| **Hiçbir şey yapma** | 0 | $0 | %94 |
| **Google Colab (ÜCRETSİZ)** | 2-3 saat | $0 | %96 |
| **Kaggle (ÜCRETSİZ)** | 2 saat | $0 | %96 |
| **Kendi GPU'n (GTX 1060)** | 2 saat | Elektrik | %96 |
| **AWS SageMaker** | 1.5 saat | $1.50 | %96 |
| **Paperspace** | 1 saat | $1.00 | %96 |

**TAVSİYE**: Google Colab ÜCRETSİZ ve yeterli! 🎉

---

## 🔥 Google Colab ile Eğitim (ÜCRETSİZ!)

### Adım 1: Colab Notebook Oluştur
1. https://colab.research.google.com/ 'a git
2. "New Notebook" tıkla
3. Runtime > Change runtime type > GPU seç

### Adım 2: Kodu Yapıştır
```python
# 1. Projeyi klonla
!git clone https://github.com/YOUR_USERNAME/VisionVet-AI.git
%cd VisionVet-AI

# 2. Dataset'i indir
!git clone https://github.com/ihoflaz/bacterial-colony-classification.git dataset_temp
!mkdir -p data/dibas
!cp -r dataset_temp/images/* data/dibas/
!rm -rf dataset_temp

# 3. Dependencies
%cd ml_training
!pip install -r requirements.txt

# 4. Eğitimi başlat
!python train_bacterial_model.py

# 5. Model'i indir (eğitim bitince)
from google.colab import files
files.download('models/bacterial/mobilenet_v3_large.onnx')
files.download('models/bacterial/labels_33.txt')
```

### Adım 3: Çalıştır
- Cell > Run All
- Kahve iç ☕
- 2-3 saat sonra model indirilecek

---

## 📊 Eğitim Sırasında İzleme

### Terminal Çıktısı:
```
Epoch 1/50: 100%|██████| 171/171 [00:45<00:00]
📈 Train Loss: 2.145 | Train Acc: 42.31%
📉 Val Loss: 1.823 | Val Acc: 51.24%
💾 Best model saved!

Epoch 10/50: 100%|██████| 171/171 [00:43<00:00]
📈 Train Loss: 0.543 | Train Acc: 85.67%
📉 Val Loss: 0.421 | Val Acc: 89.12%
💾 Best model saved!

Epoch 30/50: 100%|██████| 171/171 [00:42<00:00]
📈 Train Loss: 0.123 | Train Acc: 96.84%
📉 Val Loss: 0.198 | Val Acc: 94.56%
💾 Best model saved!
```

### Ne Beklenmeli:
- ✅ Train Loss azalmalı (2.0 → 0.1)
- ✅ Val Acc artmalı (50% → 95%)
- ⚠️ Val Loss artıyorsa → DURDUR (overfitting!)

---

## 🛑 Eğitimi Durdurma

### Eğer şunları görürseniz DURDURUN:
```
Epoch 45/50:
📈 Train Acc: 99.2%
📉 Val Acc: 87.5%  ← Train'den çok düşük = OVERFITTING!
```

**Çözüm**:
1. CTRL+C ile durdur
2. En iyi model zaten kaydedildi (`best_model.pth`)
3. ONNX'e export et
4. Kullan!

---

## 📦 Model'i Android'e Taşıma

Eğitim tamamlandıktan sonra:

```bash
# 1. Model'i Android assets'e kopyala
cp models/bacterial/mobilenet_v3_large.onnx ../app/src/main/assets/bacterial/
cp models/bacterial/labels_33.txt ../app/src/main/assets/bacterial/

# 2. App'i rebuild et
cd ..
./gradlew clean
./gradlew assembleDebug

# 3. Cihaza yükle
./gradlew installDebug

# 4. Test et!
adb shell am start -n com.visionvet.ai/.MainActivity
```

---

## 🐛 Sorun Giderme

### "CUDA out of memory"
```python
# Config sınıfında:
BATCH_SIZE = 16  # 32 → 16
```

### "Dataset not found"
```bash
# Dataset yolunu kontrol et:
ls -la data/dibas/
# 33 klasör görmelisiniz
```

### Eğitim çok yavaş
```python
# num_workers azalt (macOS'ta sık sorun):
num_workers=0  # 4 → 0
```

### Model accuracy %60'ta takılı
```python
# Learning rate artır:
LEARNING_RATE = 0.01  # 0.001 → 0.01
```

---

## 📞 Yardım

Sorun yaşıyorsanız:
1. `check_dataset.py` çalıştırın
2. Log'ları kontrol edin
3. GitHub Issues'ta sorun açın

---

## ✅ Checklist

Eğitim öncesi:
- [ ] Dataset indirildi (data/dibas/)
- [ ] Python 3.9+ yüklü
- [ ] GPU/CUDA kurulu (opsiyonel ama önerilir)
- [ ] 50GB+ disk alanı var
- [ ] requirements.txt yüklendi

Eğitim sonrası:
- [ ] Val Accuracy > %90
- [ ] Model ONNX'e export edildi
- [ ] Android assets'e kopyalandı
- [ ] App test edildi
- [ ] Overfitting yok (Train ≈ Val)

---

**Başarılar! 🚀**
