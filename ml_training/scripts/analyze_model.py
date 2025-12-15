#!/usr/bin/env python3
"""
Model Analiz Scripti - Eğitilmiş ONNX modelini analiz eder
"""

import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

def analyze_onnx_model(model_path, labels_path):
    """ONNX modelini detaylı analiz et"""
    
    print("="*60)
    print("🔬 MODEL ANALİZ RAPORU")
    print("="*60)
    
    # 1. Model dosyasını yükle
    print(f"\n📂 Model: {model_path}")
    model = onnx.load(model_path)
    
    # 2. Model boyutu
    model_size = Path(model_path).stat().st_size / (1024 * 1024)
    print(f"📊 Model Boyutu: {model_size:.2f} MB")
    
    # 3. Opset version
    opset_version = model.opset_import[0].version
    print(f"🔧 ONNX Opset: {opset_version}")
    
    # 4. Input/Output bilgileri
    print(f"\n📥 INPUT:")
    for input_tensor in model.graph.input:
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"   Name: {input_tensor.name}")
        print(f"   Shape: {shape}")
        print(f"   Type: {input_tensor.type.tensor_type.elem_type}")
    
    print(f"\n📤 OUTPUT:")
    for output_tensor in model.graph.output:
        shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"   Name: {output_tensor.name}")
        print(f"   Shape: {shape}")
        print(f"   Type: {output_tensor.type.tensor_type.elem_type}")
        num_classes = shape[-1] if shape else 0
        print(f"   ✅ Sınıf Sayısı: {num_classes}")
    
    # 5. Model validation
    try:
        onnx.checker.check_model(model)
        print(f"\n✅ Model Validation: PASSED")
    except Exception as e:
        print(f"\n❌ Model Validation: FAILED - {e}")
    
    # 6. Labels dosyasını oku
    print(f"\n📋 Labels: {labels_path}")
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"✅ Label Sayısı: {len(labels)}")
    
    # Label'ları göster
    print(f"\n📝 Bakteriler:")
    for i, label in enumerate(labels, 1):
        print(f"   {i:2d}. {label}")
    
    # 7. ONNX Runtime ile test
    print(f"\n🧪 ONNX Runtime Test:")
    try:
        session = ort.InferenceSession(model_path)
        
        # Input bilgisi
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        print(f"   Input Name: {input_name}")
        print(f"   Input Shape: {input_shape}")
        
        # Dummy input ile test
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        outputs = session.run(None, {input_name: dummy_input})
        
        output_shape = outputs[0].shape
        print(f"   Output Shape: {output_shape}")
        print(f"   ✅ Inference Test: PASSED")
        
        # Output değerleri analiz
        logits = outputs[0][0]
        print(f"\n📊 Output İstatistikleri:")
        print(f"   Min Logit: {logits.min():.4f}")
        print(f"   Max Logit: {logits.max():.4f}")
        print(f"   Mean Logit: {logits.mean():.4f}")
        print(f"   Std Logit: {logits.std():.4f}")
        
        # Softmax uygula
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()
        
        # En yüksek 5 tahmin
        top5_idx = np.argsort(probs)[-5:][::-1]
        print(f"\n🏆 Top 5 Predictions (Random Input):")
        for idx in top5_idx:
            if idx < len(labels):
                print(f"   {labels[idx]:<35} {probs[idx]*100:6.2f}%")
        
    except Exception as e:
        print(f"   ❌ Inference Test: FAILED - {e}")
    
    # 8. SONUÇ
    print(f"\n{'='*60}")
    print("📋 ÖZET")
    print(f"{'='*60}")
    
    issues = []
    
    # Kontroller
    if model_size < 10 or model_size > 30:
        issues.append(f"⚠️  Model boyutu beklenenden farklı: {model_size:.2f} MB (beklenen: 15-20 MB)")
    else:
        print(f"✅ Model boyutu normal: {model_size:.2f} MB")
    
    if num_classes != 33 and num_classes != 34:
        issues.append(f"⚠️  Sınıf sayısı yanlış: {num_classes} (beklenen: 33)")
    else:
        print(f"✅ Sınıf sayısı doğru: {num_classes}")
    
    if len(labels) != 33 and len(labels) != 34:
        issues.append(f"⚠️  Label sayısı yanlış: {len(labels)} (beklenen: 33)")
    else:
        print(f"✅ Label sayısı doğru: {len(labels)}")
    
    if opset_version < 11:
        issues.append(f"⚠️  ONNX opset eski: {opset_version} (önerilen: 12+)")
    else:
        print(f"✅ ONNX opset uygun: {opset_version}")
    
    if issues:
        print(f"\n⚠️  SORUNLAR:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print(f"\n🎉 Model tüm kontrolleri geçti!")
    
    print(f"\n{'='*60}")
    
    return {
        'model_size_mb': model_size,
        'num_classes': num_classes,
        'num_labels': len(labels),
        'opset_version': opset_version,
        'labels': labels,
        'issues': issues
    }


if __name__ == "__main__":
    # Model ve labels dosyaları
    model_path = "/Users/mac/Downloads/mobilenet_v3_large (5).onnx"
    labels_path = "/Users/mac/Downloads/labels_33 (5).txt"
    
    result = analyze_onnx_model(model_path, labels_path)
    
    print(f"\n💡 SONRAKİ ADIM:")
    print(f"   Bu modeli mevcut modelle karşılaştırmak için:")
    print(f"   python compare_models.py")
