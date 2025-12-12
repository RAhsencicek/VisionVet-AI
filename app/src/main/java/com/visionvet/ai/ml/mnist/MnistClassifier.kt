package com.visionvet.ai.ml.mnist

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp

/**
 * MNIST rakam tanıma sınıfı
 * El yazısı rakamları (0-9) tanımak için TensorFlow Lite modeli kullanır
 */
class MnistClassifier(context: Context) {
    
    private var interpreter: Interpreter? = null
    private val inputSize = 28 // MNIST images are 28x28
    private val pixelSize = 1 // Grayscale
    private val imageStd = 255.0f
    
    companion object {
        private const val MODEL_FILE = "mnist_model.tflite"
        private const val NUM_CLASSES = 10 // 0-9 digits
    }
    
    init {
        try {
            // Model dosyasını yükle
            val model = FileUtil.loadMappedFile(context, MODEL_FILE)
            interpreter = Interpreter(model)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
    
    /**
     * Bitmap görüntüsünü sınıflandır
     * @param bitmap 28x28 grayscale bitmap
     * @return ClassificationResult tanıma sonucu
     */
    fun classify(bitmap: Bitmap): ClassificationResult {
        if (interpreter == null) {
            return ClassificationResult(-1, 0.0f, FloatArray(NUM_CLASSES))
        }
        
        // Bitmap'i normalize edilmiş ByteBuffer'a dönüştür
        val inputBuffer = convertBitmapToByteBuffer(bitmap)
        
        // Çıktı için float array
        val output = Array(1) { FloatArray(NUM_CLASSES) }
        
        // Model inference
        interpreter?.run(inputBuffer, output)
        
        // Sonuçları işle
        val probabilities = output[0]
        val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: -1
        val confidence = probabilities[maxIndex]
        
        return ClassificationResult(
            predictedDigit = maxIndex,
            confidence = confidence,
            probabilities = probabilities
        )
    }
    
    /**
     * Bitmap'i model için uygun formata dönüştür
     */
    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * pixelSize)
        byteBuffer.order(ByteOrder.nativeOrder())
        
        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        
        for (pixelValue in pixels) {
            // Grayscale değeri al (R, G, B aynı olmalı)
            val gray = (pixelValue shr 16 and 0xFF)
            // Normalize et [0, 1]
            val normalizedValue = gray / imageStd
            byteBuffer.putFloat(normalizedValue)
        }
        
        return byteBuffer
    }
    
    /**
     * Kaynakları serbest bırak
     */
    fun close() {
        interpreter?.close()
        interpreter = null
    }
}

/**
 * Sınıflandırma sonucu veri sınıfı
 */
data class ClassificationResult(
    val predictedDigit: Int,
    val confidence: Float,
    val probabilities: FloatArray
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as ClassificationResult

        if (predictedDigit != other.predictedDigit) return false
        if (confidence != other.confidence) return false
        if (!probabilities.contentEquals(other.probabilities)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = predictedDigit
        result = 31 * result + confidence.hashCode()
        result = 31 * result + probabilities.contentHashCode()
        return result
    }
}
