# 🐾 VisionVet AI

AI-powered veterinary diagnostic assistant for Android. Features bacterial colony classification using deep learning models.

## ✨ Features

- **🔬 Bacterial Colony Classification**: Identify 33 different bacterial species using MobileNetV3-Large (95.45% accuracy)
- **📸 Camera Integration**: Capture images directly from your device camera
- **🖼️ Gallery Support**: Analyze existing images from your gallery
- **📊 Analysis History**: Track and review all previous analysis results
- **💾 Offline Support**: Works completely offline - no internet required

## 🧪 Supported Bacterial Species

The app can identify 33 bacterial species from the DIBaS (Digital Image of Bacterial Species) dataset including:
- *Acinetobacter baumannii*
- *Escherichia coli*
- *Staphylococcus aureus*
- *Pseudomonas aeruginosa*
- *Candida albicans*
- And 28 more species...

## 🏗️ Tech Stack

- **Language**: Kotlin
- **UI Framework**: Jetpack Compose + Material 3
- **ML Runtime**: ONNX Runtime Mobile 1.19.2
- **Architecture**: MVVM with Repository pattern
- **Database**: Room (SQLite)
- **Camera**: CameraX

## 📱 Requirements

- Android 7.0 (API 24) or higher
- ARM64 or ARMv7 processor
- ~50MB storage space

## 🚀 Getting Started

### Prerequisites
- Android Studio Hedgehog (2023.1.1) or later
- JDK 11+
- Android SDK 36

### Build & Run

1. Clone the repository:
```bash
git clone https://github.com/RAhsencicek/VisionVet-AI.git
cd VisionVet-AI
```

2. Open in Android Studio

3. Sync Gradle and build:
```bash
./gradlew assembleDebug
```

4. Install on device/emulator:
```bash
./gradlew installDebug
```

## 📁 Project Structure

```
app/src/main/
├── java/com/visionvet/ai/
│   ├── core/
│   │   └── database/          # Room Database, DAOs, Entities
│   ├── feature/
│   │   ├── bacterial/         # Bacterial classification screens
│   │   ├── home/              # Home screen
│   │   ├── dashboard/         # Dashboard
│   │   ├── history/           # Analysis history
│   │   └── settings/          # App settings
│   ├── ml/
│   │   └── bacterial/         # BacterialClassifier (ONNX inference)
│   ├── navigation/            # Navigation setup
│   └── ui/                    # Theme & common UI components
├── assets/
│   └── bacterial/
│       ├── mobilenet_v3_large.onnx      # ML model (~16MB)
│       ├── mobilenet_v3_large.onnx.data # Model weights
│       └── labels_33.txt                 # Class labels
└── res/                       # Android resources
```

## 🧠 ML Model Details

| Property | Value |
|----------|-------|
| Architecture | MobileNetV3-Large |
| Framework | ONNX |
| Input Size | 224×224 RGB |
| Classes | 33 bacterial species |
| Accuracy | 95.45% |
| Model Size | ~16 MB |
| Inference Time | <100ms on modern devices |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [DIBaS Dataset](https://github.com/ihoflaz/bacterial-colony-classification) - Bacterial colony images
- [ONNX Runtime](https://onnxruntime.ai/) - Cross-platform ML inference
- [Jetpack Compose](https://developer.android.com/jetpack/compose) - Modern Android UI toolkit
