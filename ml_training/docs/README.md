# 🧠 VisionVet-AI Model Training

Bu klasör, bakteriyel koloni sınıflandırma modelini yeniden eğitmek için gerekli scriptleri içerir.

## 📋 Gereksinimler

```bash
pip install torch torchvision onnx onnxruntime scikit-learn pillow matplotlib tqdm
```

## 📂 Dataset Yapısı

DIBaS (Digital Image of Bacterial Species) dataset'ini şu şekilde organize edin:

```
data/dibas/
├── Acinetobacter_baumannii/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── Candida_albicans/
│   ├── image_001.jpg
│   └── ...
├── Escherichia_coli/
│   └── ...
└── ... (33 sınıf toplam)
```

## 🚀 Eğitim

### 1. Dataset İndir
DIBaS dataset'ini indirin ve `data/dibas/` klasörüne çıkarın.

### 2. Modeli Eğit
```bash
cd ml_training
python train_bacterial_model.py
```

### 3. Hyperparameter Ayarları
`train_bacterial_model.py` dosyasındaki `Config` sınıfını düzenleyin:

```python
class Config:
    BATCH_SIZE = 32      # GPU belleğinize göre ayarlayın
    EPOCHS = 50          # Daha fazla epoch daha iyi accuracy
    LEARNING_RATE = 0.001
    USE_AUGMENTATION = True  # Data augmentation (önerilir)
```

## 📊 Beklenen Sonuçlar

- **Eğitim Süresi**: ~2-4 saat (GPU ile)
- **Beklenen Accuracy**: %90-96
- **Model Boyutu**: ~16MB

## 📦 Model Deployment

Eğitim tamamlandıktan sonra:

1. **ONNX Model**: `models/bacterial/mobilenet_v3_large.onnx`
2. **Labels**: `models/bacterial/labels_33.txt`

Bu dosyaları Android projesine kopyalayın:

```bash
cp models/bacterial/mobilenet_v3_large.onnx ../app/src/main/assets/bacterial/
cp models/bacterial/labels_33.txt ../app/src/main/assets/bacterial/
```

## 🔧 Advanced: Fine-tuning

Daha iyi sonuçlar için:

### 1. Tüm Katmanları Eğit
```python
# train_bacterial_model.py içinde:
# Freeze satırlarını yoruma alın:
# for param in model.features[:10].parameters():
#     param.requires_grad = False
```

### 2. Learning Rate Schedule
```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
```

### 3. Test-Time Augmentation
```python
# Inference sırasında birden fazla augmented versiyonu kullan
```

## 📈 Monitoring

TensorBoard ile eğitimi takip etmek için:

```bash
pip install tensorboard
tensorboard --logdir=runs
```

## ⚠️ Common Issues

### GPU Memory Error
- `BATCH_SIZE`'ı azaltın (16 veya 8)
- `num_workers`'ı azaltın

### Overfitting
- `WEIGHT_DECAY`'i artırın
- `Dropout` oranını artırın
- Data augmentation kullanın

### Low Accuracy
- Daha fazla epoch
- Learning rate azalt
- Daha fazla data augmentation

## 🎯 Model Validation Thresholds

Eğitim sonrası, `BacterialClassifier.kt` dosyasındaki threshold'ları ayarlayın:

```kotlin
private const val MIN_LOGIT_THRESHOLD = 8.0f
private const val MAX_LOGIT_THRESHOLD = 25.0f
private const val MIN_VALID_CONFIDENCE_THRESHOLD = 40f
```

Test setinizde farklı görüntülerle deney yaparak optimal değerleri bulun.

## 📝 Notlar

- **MobileNetV3** hafif ve hızlı bir model (mobil cihazlar için ideal)
- **Transfer Learning** kullanıyoruz (ImageNet pretrained weights)
- **33 sınıf** bakteriyel koloni türü
- **ONNX format** Android'de ONNX Runtime ile çalışır

## 🤝 Contribution

Model iyileştirmeleri için PR göndermekten çekinmeyin!
