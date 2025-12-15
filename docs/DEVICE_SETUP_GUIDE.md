# Android Cihazı Kurulum Rehberi

## 1. Android Cihazınızda Geliştirici Seçeneklerini Aktifleştirin

### Adım 1: Ayarlar'a gidin
- Cihazınızda **Ayarlar** (Settings) uygulamasını açın

### Adım 2: Telefon Hakkında bölümünü bulun
- **Sistem** > **Telefon hakkında** (About phone) 
- Veya doğrudan **Telefon hakkında** seçeneğini arayın

### Adım 3: Yapı numarasına 7 kez dokunun
- **Yapı numarası** (Build number) satırını bulun
- Bu satıra **7 kez arka arkaya** dokunun
- "Artık bir geliştiricisisiniz!" mesajını göreceksiniz

## 2. USB Hata Ayıklamayı Aktifleştirin

### Adım 1: Geliştirici seçeneklerine gidin
- Ayarlar > **Sistem** > **Geliştirici seçenekleri**
- (Bazı cihazlarda doğrudan Ayarlar altında görünür)

### Adım 2: USB Hata Ayıklamayı açın
- **USB hata ayıklama** (USB debugging) seçeneğini **AÇIN**
- Uyarı mesajında **Tamam**'a basın

### Adım 3: (İsteğe bağlı) Diğer yararlı seçenekler
- **USB aracılığıyla yüklemeye izin ver** - AÇIN
- **Uyku modunda hata ayıklamayı koru** - AÇIN

## 3. Cihazı Mac'e Bağlayın

### Adım 1: USB kablosu ile bağlayın
- Orijinal USB kablosunu kullanın (veri transferi destekleyen)
- Cihazı Mac'inize bağlayın

### Adım 2: USB bağlantı modunu ayarlayın
- Cihazda bildirim geldiğinde **"Dosya aktarımı"** veya **"MTP"** modunu seçin

### Adım 3: Hata ayıklama iznini verin
- "USB hata ayıklamasına izin verilsin mi?" sorusuna **İZİN VER** deyin
- **"Bu bilgisayardan her zaman izin ver"** kutusunu işaretleyin

## 4. Bağlantıyı Test Etme

Terminal'de şu komutu çalıştırın:
```bash
cd /Users/mac/AndroidStudioProjects/OPCA && ./gradlew installDebug
```

## Sorun Giderme

### Cihaz görünmüyorsa:
1. USB kablosunu değiştirin
2. Farklı USB portu deneyin
3. Cihazı yeniden başlatın
4. USB hata ayıklamayı kapatıp açın

### Android sürümüne göre farklılıklar:
- **Android 11+**: Ek güvenlik izinleri gerekebilir
- **MIUI/ColorOS/OneUI**: Ek geliştirici seçenekleri olabilir

