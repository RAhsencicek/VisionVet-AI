# 🔬 Bakteriyel Koloni Sınıflandırma - VisionVet AI Entegrasyon Yol Haritası

## 📊 Mevcut Durum Analizi

### ✅ VisionVet AI Proje Yapısı (Şu An)
```
com.visionvet.ai/
├── feature/
│   ├── home/              ✅ Ana ekran (Tab navigation)
│   ├── dashboard/         ✅ İstatistikler
│   ├── scanner/           ✅ NewScanScreen (genel tarama)
│   ├── history/           ✅ Geçmiş kayıtlar
│   ├── analysis/          ✅ AnalysisDetailScreen
│   ├── settings/          ✅ Ayarlar
│   └── mnist/             ✅ MNIST test (TensorFlow Lite)
├── ml/
│   └── mnist/             ✅ MnistClassifier, DrawingView
├── core/
│   └── database/          ✅ Room DB (Analysis, Transaction, User)
└── ui/                    ✅ Tema ve ortak bileşenler
```

### 📦 Bacterial Colony Classification Projesi
**Repository:** https://github.com/ihoflaz/bacterial-colony-classification

**Temel Özellikler:**
- **Model:** MobileNetV3-Large
- **Doğruluk:** 95.45%
- **Parametre:** 4.24M
- **Model Boyutu:** 16.2 MB (ONNX)
- **Sınıf Sayısı:** 33 bakteriyel koloni türü
- **Girdi Boyutu:** 224x224 RGB
- **Framework:** ONNX Runtime Mobile / PyTorch Mobile

**Dataset:** DIBaS (Digital Image of Bacterial Species)

---

## 🎯 Entegrasyon Hedefleri

### 1. 🔬 Ana Özellik: Bakteriyel Koloni Tanıma Sistemi
VisionVet AI uygulamasına veteriner kullanımı için bakteriyel koloni sınıflandırma özelliği eklenmesi.

### 2. 📱 Kullanıcı Senaryosu
1. Veteriner kullanıcı kamera ile petri kabı fotoğrafı çeker
2. AI analizi gerçekleştirilir (MobileNetV3-Large)
3. Top-3 bakteriyel koloni tahmini gösterilir
4. Güven skorları ve detaylı bilgiler sunulur
5. Sonuç veritabanına kaydedilir
6. Geçmiş analizler history'de görüntülenir

---

## 📋 DETAYLI YOL HARİTASI

### 🔴 FAZ 1: Altyapı ve Model Entegrasyonu (2-3 gün)

#### 1.1 ✅ Dependencies Ekleme
**Dosya:** `app/build.gradle.kts`
```kotlin
dependencies {
    // ONNX Runtime Mobile
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.23.0")
    
    // Görüntü işleme için (zaten var olabilir)
    implementation("androidx.camera:camera-camera2:1.3.1")
    implementation("androidx.camera:camera-lifecycle:1.3.1")
    implementation("androidx.camera:camera-view:1.3.1")
}
```

**Gradle Sync:**
- ARM64 ABI filtresi ekle (zaten olabilir)
- MinSdk: 26 (Android 8.0+)

---

#### 1.2 📁 Model Dosyaları Yerleştirme
**Hedef Klasör:** `app/src/main/assets/bacterial/`

**Gerekli Dosyalar:**
1. `mobilenet_v3_large.onnx` (16.2 MB)
2. `labels_33.txt` (33 bakteriyel tür listesi)

**Labels Formatı (labels_33.txt):**
```
Acinetobacter_baumannii
Actinomyces_naeslundii
Bacteroides_fragilis
...
Veillonella
```

**Action Items:**
- [ ] `bacterial-colony-classification/models/exports/onnx/` klasöründen modeli kopyala
- [ ] Labels dosyasını oluştur (training_plan.md'den veya sonuçlardan)
- [ ] Assets klasörünü VisionVet AI projesine ekle

---

#### 1.3 🧠 BacterialClassifier Sınıfı Oluşturma
**Dosya:** `app/src/main/java/com/visionvet/ai/ml/bacterial/BacterialClassifier.kt`

**Sorumluluklar:**
- ONNX model yükleme
- Inference çalıştırma
- Top-K tahmin döndürme
- Session yönetimi

**Kod Taslağı:**
```kotlin
package com.visionvet.ai.ml.bacterial

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.exp

data class BacterialPrediction(
    val className: String,
    val confidence: Float,
    val classIndex: Int
)

class BacterialClassifier(context: Context) {
    
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private val labels: List<String>
    
    companion object {
        private const val MODEL_FILE = "bacterial/mobilenet_v3_large.onnx"
        private const val LABELS_FILE = "bacterial/labels_33.txt"
        private const val INPUT_SIZE = 224
        private const val NUM_CLASSES = 33
        
        // ImageNet normalization (DIBaS eğitimi bu değerlerle yapılmış)
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }
    
    init {
        // Load labels
        labels = context.assets.open(LABELS_FILE).bufferedReader().readLines()
        
        // Initialize ONNX Runtime
        ortEnv = OrtEnvironment.getEnvironment()
        val sessionOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(1)
            setInterOpNumThreads(1)
        }
        
        val modelBytes = context.assets.open(MODEL_FILE).readBytes()
        ortSession = ortEnv?.createSession(modelBytes, sessionOptions)
    }
    
    fun classify(bitmap: Bitmap, topK: Int = 3): List<BacterialPrediction> {
        // TODO: Implement preprocessing
        // TODO: Run inference
        // TODO: Apply softmax
        // TODO: Return top-K predictions
        return emptyList()
    }
    
    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val expValues = logits.map { exp(it - maxLogit) }
        val sumExp = expValues.sum()
        return expValues.map { (it / sumExp).toFloat() }.toFloatArray()
    }
    
    fun close() {
        ortSession?.close()
        ortSession = null
        ortEnv?.close()
        ortEnv = null
    }
}
```

---

#### 1.4 🖼️ ImagePreprocessor Utility
**Dosya:** `app/src/main/java/com/visionvet/ai/ml/utils/ImagePreprocessor.kt`

**Fonksiyonlar:**
- `resizeAndCenterCrop(bitmap: Bitmap, size: Int): Bitmap`
- `normalize(bitmap: Bitmap, mean: FloatArray, std: FloatArray): FloatBuffer`
- `bitmapToFloatBuffer(bitmap: Bitmap): FloatBuffer`

**Önemli Noktalar:**
- RGB sıralaması (Android Bitmap default RGB)
- [0, 1] aralığına normalize: `pixel / 255.0`
- ImageNet mean/std değerleri uygula
- NCHW formatı: `[1, 3, 224, 224]`

---

### 🟡 FAZ 2: UI ve Feature Implementation (3-4 gün)

#### 2.1 🎨 BacterialScanScreen Oluşturma
**Dosya:** `app/src/main/java/com/visionvet/ai/feature/bacterial/BacterialScanScreen.kt`

**Özellikler:**
- CameraX entegrasyonu
- Fotoğraf çekme butonu
- Galeri seçim opsiyonu
- Loading indicator (inference sırasında)
- Error handling
- Preview görüntüleme

**UI Bileşenleri:**
```kotlin
@Composable
fun BacterialScanScreen(
    onNavigateToResult: (analysisId: String) -> Unit,
    viewModel: BacterialScanViewModel = viewModel()
) {
    val cameraPermissionGranted by viewModel.cameraPermissionState.collectAsState()
    val isProcessing by viewModel.isProcessing.collectAsState()
    
    Box(modifier = Modifier.fillMaxSize()) {
        if (cameraPermissionGranted) {
            CameraPreview(
                onImageCaptured = { bitmap ->
                    viewModel.analyzeBacterialColony(bitmap)
                }
            )
        } else {
            PermissionRequest(onRequestPermission = { /* ... */ })
        }
        
        if (isProcessing) {
            LoadingOverlay()
        }
    }
}
```

---

#### 2.2 📊 BacterialResultScreen
**Dosya:** `app/src/main/java/com/visionvet/ai/feature/bacterial/BacterialResultScreen.kt`

**Görüntülenecek Bilgiler:**
1. Çekilen görüntü (thumbnail)
2. Top-3 tahmin listesi:
   - Bakteriyel tür adı
   - Güven skoru (%)
   - Progress bar
3. Detaylı bilgi kartı:
   - Analiz tarihi
   - İşlem süresi (ms)
   - Model versiyonu
4. Action butonları:
   - Kaydet
   - Paylaş
   - Yeni analiz

**Örnek UI:**
```kotlin
@Composable
fun BacterialResultScreen(
    analysisId: String,
    viewModel: BacterialResultViewModel = viewModel()
) {
    val analysis by viewModel.getAnalysis(analysisId).collectAsState()
    
    Column(modifier = Modifier.padding(16.dp)) {
        // Captured image
        AsyncImage(
            model = analysis?.imagePath,
            modifier = Modifier.size(200.dp)
        )
        
        Spacer(modifier = Modifier.height(16.dp))
        
        // Top-3 predictions
        Text("Top Tahminler", style = MaterialTheme.typography.titleLarge)
        
        analysis?.predictions?.forEachIndexed { index, pred ->
            PredictionCard(
                rank = index + 1,
                bacterialName = pred.className,
                confidence = pred.confidence
            )
        }
        
        // Action buttons
        Row(horizontalArrangement = Arrangement.SpaceEvenly) {
            Button(onClick = { /* Save */ }) { Text("Kaydet") }
            Button(onClick = { /* Share */ }) { Text("Paylaş") }
            Button(onClick = { /* New */ }) { Text("Yeni Analiz") }
        }
    }
}
```

---

#### 2.3 🗂️ BacterialScanViewModel
**Dosya:** `app/src/main/java/com/visionvet/ai/feature/bacterial/BacterialScanViewModel.kt`

**Sorumluluklar:**
- Camera permission state
- Image capture handling
- Classifier'ı çağırma
- Result'ı database'e kaydetme
- Loading state management
- Error handling

```kotlin
class BacterialScanViewModel(
    private val bacterialClassifier: BacterialClassifier,
    private val bacterialRepository: BacterialRepository
) : ViewModel() {
    
    private val _isProcessing = MutableStateFlow(false)
    val isProcessing = _isProcessing.asStateFlow()
    
    private val _analysisResult = MutableStateFlow<BacterialAnalysisResult?>(null)
    val analysisResult = _analysisResult.asStateFlow()
    
    fun analyzeBacterialColony(bitmap: Bitmap) {
        viewModelScope.launch {
            _isProcessing.value = true
            try {
                val startTime = System.currentTimeMillis()
                
                // Run classification
                val predictions = bacterialClassifier.classify(bitmap, topK = 3)
                
                val inferenceTime = System.currentTimeMillis() - startTime
                
                // Save to database
                val analysis = BacterialAnalysis(
                    imagePath = saveImageToStorage(bitmap),
                    topPrediction = predictions.first().className,
                    confidence = predictions.first().confidence,
                    predictions = predictions,
                    inferenceTime = inferenceTime,
                    timestamp = System.currentTimeMillis()
                )
                
                bacterialRepository.insertAnalysis(analysis)
                _analysisResult.value = BacterialAnalysisResult.Success(analysis)
                
            } catch (e: Exception) {
                _analysisResult.value = BacterialAnalysisResult.Error(e.message ?: "Unknown error")
            } finally {
                _isProcessing.value = false
            }
        }
    }
}
```

---

### 🟢 FAZ 3: Database ve Repository (1-2 gün)

#### 3.1 💾 BacterialAnalysis Entity
**Dosya:** `app/src/main/java/com/visionvet/ai/core/database/model/BacterialAnalysis.kt`

```kotlin
@Entity(tableName = "bacterial_analysis")
data class BacterialAnalysis(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    
    val imagePath: String,
    val topPrediction: String,
    val confidence: Float,
    
    @TypeConverters(PredictionsConverter::class)
    val predictions: List<BacterialPrediction>,
    
    val inferenceTime: Long, // milliseconds
    val timestamp: Long,
    val userId: String? = null,
    
    val notes: String? = null,
    val isSynced: Boolean = false
)

// Converter for predictions list
class PredictionsConverter {
    @TypeConverter
    fun fromPredictionList(predictions: List<BacterialPrediction>): String {
        return Gson().toJson(predictions)
    }
    
    @TypeConverter
    fun toPredictionList(json: String): List<BacterialPrediction> {
        return Gson().fromJson(json, object : TypeToken<List<BacterialPrediction>>() {}.type)
    }
}
```

---

#### 3.2 🔧 BacterialAnalysisDao
**Dosya:** `app/src/main/java/com/visionvet/ai/core/database/dao/BacterialAnalysisDao.kt`

```kotlin
@Dao
interface BacterialAnalysisDao {
    @Insert
    suspend fun insert(analysis: BacterialAnalysis): Long
    
    @Query("SELECT * FROM bacterial_analysis ORDER BY timestamp DESC")
    fun getAllAnalyses(): Flow<List<BacterialAnalysis>>
    
    @Query("SELECT * FROM bacterial_analysis WHERE id = :id")
    fun getAnalysisById(id: Long): Flow<BacterialAnalysis?>
    
    @Query("SELECT * FROM bacterial_analysis WHERE timestamp >= :startTime")
    fun getAnalysesSince(startTime: Long): Flow<List<BacterialAnalysis>>
    
    @Delete
    suspend fun delete(analysis: BacterialAnalysis)
    
    @Query("DELETE FROM bacterial_analysis WHERE id = :id")
    suspend fun deleteById(id: Long)
}
```

---

#### 3.3 🏗️ BacterialRepository
**Dosya:** `app/src/main/java/com/visionvet/ai/core/database/repository/BacterialRepository.kt`

```kotlin
class BacterialRepository(
    private val bacterialAnalysisDao: BacterialAnalysisDao
) {
    fun getAllAnalyses(): Flow<List<BacterialAnalysis>> {
        return bacterialAnalysisDao.getAllAnalyses()
    }
    
    fun getAnalysisById(id: Long): Flow<BacterialAnalysis?> {
        return bacterialAnalysisDao.getAnalysisById(id)
    }
    
    suspend fun insertAnalysis(analysis: BacterialAnalysis): Long {
        return bacterialAnalysisDao.insert(analysis)
    }
    
    suspend fun deleteAnalysis(analysis: BacterialAnalysis) {
        bacterialAnalysisDao.delete(analysis)
    }
    
    fun getRecentAnalyses(days: Int = 7): Flow<List<BacterialAnalysis>> {
        val startTime = System.currentTimeMillis() - (days * 24 * 60 * 60 * 1000L)
        return bacterialAnalysisDao.getAnalysesSince(startTime)
    }
}
```

---

### 🔵 FAZ 4: Navigation ve Integration (1 gün)

#### 4.1 🧭 Screen Routes Ekleme
**Dosya:** `app/src/main/java/com/visionvet/ai/navigation/Screen.kt`

```kotlin
sealed class Screen(val route: String) {
    // Existing screens...
    object Home : Screen("home")
    object Dashboard : Screen("dashboard")
    object Settings : Screen("settings")
    object MnistTest : Screen("mnist_test")
    
    // NEW: Bacterial screens
    object BacterialScan : Screen("bacterial_scan")
    object BacterialResult : Screen("bacterial_result/{analysisId}") {
        fun createRoute(analysisId: Long) = "bacterial_result/$analysisId"
    }
    object BacterialHistory : Screen("bacterial_history")
}
```

---

#### 4.2 📱 MainActivity Navigation Setup
**Dosya:** `app/src/main/java/com/visionvet/ai/MainActivity.kt`

```kotlin
NavHost(
    navController = navController,
    startDestination = Screen.Home.route
) {
    // ... existing composables
    
    composable(Screen.BacterialScan.route) {
        BacterialScanScreen(
            onNavigateToResult = { analysisId ->
                navController.navigate(Screen.BacterialResult.createRoute(analysisId))
            }
        )
    }
    
    composable(
        route = Screen.BacterialResult.route,
        arguments = listOf(navArgument("analysisId") { type = NavType.LongType })
    ) { backStackEntry ->
        val analysisId = backStackEntry.arguments?.getLong("analysisId") ?: 0L
        BacterialResultScreen(analysisId = analysisId)
    }
    
    composable(Screen.BacterialHistory.route) {
        BacterialHistoryScreen()
    }
}
```

---

#### 4.3 🏠 HomeView'a Bacterial Scan Tab Ekleme
**Dosya:** `app/src/main/java/com/visionvet/ai/feature/home/HomeView.kt`

```kotlin
sealed class BottomNavItem(
    val route: String,
    val title: String,
    val icon: ImageVector
) {
    object Dashboard : BottomNavItem("dashboard", "Dashboard", Icons.Default.Home)
    object NewScan : BottomNavItem("new_scan", "New Scan", Icons.Default.Add)
    object BacterialScan : BottomNavItem("bacterial_scan", "Bacterial", Icons.Default.Science) // NEW
    object History : BottomNavItem("history", "History", Icons.Default.DateRange)
    object Settings : BottomNavItem("settings", "Settings", Icons.Default.Settings)
}
```

---

### 🟣 FAZ 5: Testing ve Optimization (2 gün)

#### 5.1 🧪 Unit Tests
**Dosya:** `app/src/test/java/com/visionvet/ai/ml/bacterial/BacterialClassifierTest.kt`

**Test Senaryoları:**
- Model loading
- Preprocessing pipeline
- Inference output format
- Softmax calculation
- Top-K selection
- Error handling

---

#### 5.2 ⚡ Performance Optimization

**ONNX Runtime Ayarları:**
```kotlin
val sessionOptions = OrtSession.SessionOptions().apply {
    // Thread optimization
    setIntraOpNumThreads(2)
    setInterOpNumThreads(1)
    
    // Execution mode
    setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
    
    // NNAPI Delegate (Android 10+)
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
        addNnapi()
    }
}
```

**Performans Metrikleri:**
- Inference time: Target < 100ms
- Model load time: < 1 second
- Memory usage: < 50MB

---

#### 5.3 📊 Instrumentation Tests
**Dosya:** `app/src/androidTest/java/com/visionvet/ai/feature/bacterial/BacterialScanE2ETest.kt`

**Test Akışı:**
1. Camera permission grant
2. Image capture
3. Classification
4. Result display
5. Database save
6. History view

---

### ⚫ FAZ 6: UI/UX Polish ve Documentation (1 gün)

#### 6.1 🎨 UI/UX İyileştirmeler
- Loading animations
- Error messages (user-friendly)
- Empty states
- Skeleton screens
- Success animations

---

#### 6.2 📝 Documentation
- README güncellemesi
- Kullanım kılavuzu
- API documentation
- Model versiyonlama
- Troubleshooting guide

---

## 📊 KAPSAM KARŞILAŞTIRMASI

### Bacterial Colony Projesi vs VisionVet AI

| Özellik | Bacterial Project | VisionVet AI (Mevcut) | Entegrasyon Sonrası |
|---------|-------------------|----------------------|---------------------|
| Model Framework | PyTorch + ONNX | TensorFlow Lite | ONNX Runtime |
| Model | MobileNetV3-Large | MNIST (Digit) | MobileNetV3-Large + MNIST |
| Sınıf Sayısı | 33 (bacterial) | 10 (digits) | 33 + 10 |
| Girdi Boyutu | 224x224 | 28x28 | Her ikisi de desteklenir |
| Normalizasyon | ImageNet | Custom | İkisi ayrı pipeline |
| Database | Yok (sadece inference) | Room DB | Room DB + Bacterial entity |
| UI | Yok (mobile guide var) | Full Compose UI | Tam entegrasyon |
| Camera | Yok | CameraX | CameraX (bacterial için) |
| History | Yok | Var | Bacterial için genişletilecek |

---

## ⚠️ ZORLUKLAR VE ÇÖ ZÜMLER

### 1. Model Boyutu
**Problem:** ONNX modeli 16.2 MB  
**Çözüm:**  
- FP16 quantization (model boyutunu yarıya indirir)
- Asset compression
- On-demand download

### 2. Inference Süresi
**Problem:** Cihaz performansına bağlı gecikme  
**Çözüm:**  
- NNAPI delegate kullanımı
- Thread optimizasyonu
- Background processing

### 3. Kamera Kalitesi
**Problem:** Düşük kalite görüntülerde doğruluk azalabilir  
**Çözüm:**  
- Minimum resolution requirement
- Focus check
- Lighting guide UI

### 4. Label Yönetimi
**Problem:** 33 bakteriyel tür isminin doğru eşleştirilmesi  
**Çözüm:**  
- labels_33.txt dosyası strict sıralama
- Unit test ile doğrulama
- Hardcoded fallback

---

## 📅 ZAMAN TAHMİNİ

**Toplam:** ~10-12 gün (2-2.5 hafta)

| Faz | Görev | Süre |
|-----|-------|------|
| Faz 1 | Model Entegrasyonu | 2-3 gün |
| Faz 2 | UI Implementation | 3-4 gün |
| Faz 3 | Database | 1-2 gün |
| Faz 4 | Navigation | 1 gün |
| Faz 5 | Testing | 2 gün |
| Faz 6 | Polish | 1 gün |

---

## ✅ BAŞLANGIÇ ADIMLARı (İlk Gün)

### 1. Model Dosyalarını Hazırla
```bash
cd /Users/mac/milco/bacterial-colony-classification
# ONNX modelini kontrol et
ls -lh models/exports/onnx/

# Labels dosyasını oluştur (eğer yoksa)
python scripts/generate_labels.py
```

### 2. VisionVet AI'a Assets Klasörü Ekle
```bash
cd /Users/mac/milco/AndroidStudioProjects/OPCA/app/src/main
mkdir -p assets/bacterial
cp /Users/mac/milco/bacterial-colony-classification/models/exports/onnx/mobilenet_v3_large.onnx assets/bacterial/
```

### 3. Gradle Dependency Ekle
```kotlin
// app/build.gradle.kts
implementation("com.microsoft.onnxruntime:onnxruntime-android:1.23.0")
```

### 4. İlk Sınıfı Oluştur
```bash
mkdir -p app/src/main/java/com/visionvet/ai/ml/bacterial
touch app/src/main/java/com/visionvet/ai/ml/bacterial/BacterialClassifier.kt
```

---

## 🎯 SONUÇ

Bu yol haritası, **Bacterial Colony Classification** projesini **VisionVet AI** uygulamasına tam entegre etmek için gereken tüm adımları içermektedir.

**Ana Çıktılar:**
1. ✅ Bakteriyel koloni sınıflandırma özelliği
2. ✅ ONNX Runtime entegrasyonu
3. ✅ Tam database desteği
4. ✅ User-friendly UI
5. ✅ History ve analiz takibi

**Teknik Stack:**
- ONNX Runtime Mobile 1.23.0
- MobileNetV3-Large (95.45% accuracy)
- Jetpack Compose UI
- Room Database
- CameraX

**Sonraki Adım:** Faz 1'in ilk görevini (Dependencies ekleme) başlatmak için onay alınmalıdır.
