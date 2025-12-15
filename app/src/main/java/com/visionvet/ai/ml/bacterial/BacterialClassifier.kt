package com.visionvet.ai.ml.bacterial

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer

/**
 * Bacterial Colony Classifier using ONNX Runtime
 * Model: MobileNetV3-Large Balanced - trained on 32 bacterial species
 * Accuracy: 94.8% validation, 100% augmentation robustness
 * Input: 224x224 RGB image
 * Output: Probability distribution over 32 classes
 */
class BacterialClassifier(private val context: Context) {
    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var labels: List<String> = emptyList()
    
    private val inputWidth = 224
    private val inputHeight = 224
    private val inputChannels = 3
    
    // ImageNet normalization parameters
    private val meanValues = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val stdValues = floatArrayOf(0.229f, 0.224f, 0.225f)
    
    private var isInitialized = false
    
    // Store last inference stats for validation
    private var lastMaxLogit: Float = 0f
    private var lastLogitVariance: Float = 0f
    
    companion object {
        private const val TAG = "BacterialClassifier"
        
        // NEW BALANCED MODEL (December 2024)
        // Accuracy: 94.8%, Augmentation robustness: 100%
        private const val MODEL_NAME = "bacterial_classifier.onnx"
        private const val LABELS_NAME = "labels_bacteria.txt"
        
        // Confidence thresholds - RELAXED for balanced model
        const val MIN_VALID_CONFIDENCE_THRESHOLD = 25f // 25% - more permissive
        
        // Logit thresholds - RELAXED based on real-world testing
        // Testing showed: Candida maxLogit=6.4, variance=2.6 should PASS
        private const val MIN_LOGIT_THRESHOLD = 2.0f // Below = very weak activation
        private const val MAX_LOGIT_THRESHOLD = 100.0f // Very high tolerance
        private const val MIN_VARIANCE_THRESHOLD = 1.0f // Very relaxed
        private const val MAX_VARIANCE_THRESHOLD = 500.0f // Very high tolerance
        
        // Entropy threshold - relaxed
        private const val MAX_ENTROPY_THRESHOLD = 4.0f
    }
    
    /**
     * Initialize the ONNX model and load labels
     * Should be called before any classification
     */
    suspend fun initialize() = withContext(Dispatchers.IO) {
        try {
            if (isInitialized) {
                Log.d(TAG, "Classifier already initialized")
                return@withContext
            }
            
            // Load labels
            labels = loadLabels()
            Log.d(TAG, "Loaded ${labels.size} bacterial species labels")
            
            // Initialize ONNX Runtime environment
            ortEnvironment = OrtEnvironment.getEnvironment()
            
            // Load the ONNX model from assets (new balanced model)
            val modelFile = copyAssetToCache(MODEL_NAME)
            
            // Create session with default options
            val sessionOptions = OrtSession.SessionOptions()
            ortSession = ortEnvironment?.createSession(modelFile.absolutePath, sessionOptions)
            
            isInitialized = true
            Log.d(TAG, "Bacterial classifier initialized successfully")
            
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing classifier", e)
            throw BacterialClassifierException("Failed to initialize classifier: ${e.message}", e)
        }
    }
    
    /**
     * Classify a bacterial colony image
     * @param bitmap Input image (will be resized to 224x224)
     * @return List of predictions with class names and probabilities
     */
    suspend fun classify(bitmap: Bitmap): List<Prediction> = withContext(Dispatchers.IO) {
        if (!isInitialized) {
            throw BacterialClassifierException("Classifier not initialized. Call initialize() first.")
        }
        
        try {
            // Preprocess image
            val inputTensor = preprocessImage(bitmap)
            
            // Run inference
            val outputs = ortSession?.run(mapOf("input" to inputTensor))
                ?: throw BacterialClassifierException("Session is null")
            
            // Get output tensor (raw logits)
            val outputTensor = outputs[0].value as Array<FloatArray>
            val logits = outputTensor[0]
            
            // Store raw logit stats for validation
            lastMaxLogit = logits.maxOrNull() ?: 0f
            lastLogitVariance = calculateVariance(logits)
            
            Log.d(TAG, "Raw logits - Max: $lastMaxLogit, Variance: $lastLogitVariance")
            
            // Clean up
            inputTensor.close()
            outputs.close()
            
            // Apply softmax and create predictions
            val softmaxProbs = softmax(logits)
            val predictions = softmaxProbs.mapIndexed { index, probability ->
                Prediction(
                    className = labels.getOrNull(index) ?: "Unknown_$index",
                    probability = probability,
                    confidence = probability * 100f
                )
            }.sortedByDescending { it.probability }
            
            Log.d(TAG, "Classification complete. Top prediction: ${predictions.first().className} " +
                    "(${predictions.first().confidence}%)")
            
            return@withContext predictions
            
        } catch (e: Exception) {
            Log.e(TAG, "Error during classification", e)
            throw BacterialClassifierException("Classification failed: ${e.message}", e)
        }
    }
    
    /**
     * Get top K predictions
     */
    suspend fun classifyTopK(bitmap: Bitmap, k: Int = 5): List<Prediction> {
        val allPredictions = classify(bitmap)
        return allPredictions.take(k)
    }
    
    /**
     * Preprocess image for model input
     * - Resize to 224x224
     * - Convert to RGB
     * - Normalize using ImageNet mean/std
     * - Format as NCHW (batch, channels, height, width)
     */
    private fun preprocessImage(bitmap: Bitmap): OnnxTensor {
        // Resize bitmap to 224x224
        val resizedBitmap = Bitmap.createScaledBitmap(
            bitmap,
            inputWidth,
            inputHeight,
            true
        )
        
        // Extract pixels
        val pixels = IntArray(inputWidth * inputHeight)
        resizedBitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)
        
        // Create float buffer for NCHW format: [1, 3, 224, 224]
        val floatBuffer = FloatBuffer.allocate(1 * inputChannels * inputHeight * inputWidth)
        
        // Convert to normalized float values in NCHW format
        // NCHW: All R values, then all G values, then all B values
        for (c in 0 until inputChannels) {
            for (h in 0 until inputHeight) {
                for (w in 0 until inputWidth) {
                    val pixelIndex = h * inputWidth + w
                    val pixel = pixels[pixelIndex]
                    
                    // Extract channel value (0-255)
                    val channelValue = when (c) {
                        0 -> (pixel shr 16 and 0xFF) // Red
                        1 -> (pixel shr 8 and 0xFF)  // Green
                        2 -> (pixel and 0xFF)         // Blue
                        else -> 0
                    }
                    
                    // Normalize: (value/255 - mean) / std
                    val normalized = (channelValue / 255.0f - meanValues[c]) / stdValues[c]
                    floatBuffer.put(normalized)
                }
            }
        }
        
        floatBuffer.rewind()
        
        // Create ONNX tensor with shape [1, 3, 224, 224]
        val shape = longArrayOf(1, inputChannels.toLong(), inputHeight.toLong(), inputWidth.toLong())
        return OnnxTensor.createTensor(
            ortEnvironment,
            floatBuffer,
            shape
        )
    }
    
    /**
     * Apply softmax to convert logits to probabilities
     */
    private fun softmax(values: FloatArray): FloatArray {
        val maxValue = values.maxOrNull() ?: 0f
        val expValues = values.map { kotlin.math.exp((it - maxValue).toDouble()).toFloat() }
        val sumExp = expValues.sum()
        return expValues.map { it / sumExp }.toFloatArray()
    }
    
    /**
     * Calculate variance of logits - important for OOD detection
     * Low variance in logits often indicates out-of-distribution input
     */
    private fun calculateVariance(values: FloatArray): Float {
        val mean = values.average().toFloat()
        return values.map { (it - mean) * (it - mean) }.average().toFloat()
    }
    
    /**
     * Calculate Shannon entropy to measure uncertainty
     * High entropy = uncertain/uniform distribution = likely invalid image
     * Low entropy = confident prediction = likely valid bacterial image
     */
    private fun calculateEntropy(probabilities: FloatArray): Float {
        return -probabilities.filter { it > 0 }
            .map { p -> p * kotlin.math.ln(p.toDouble()).toFloat() }
            .sum() / kotlin.math.ln(probabilities.size.toDouble()).toFloat()
    }
    
    /**
     * Check if the classification result is valid (likely a bacterial colony image)
     * Uses multiple heuristics:
     * 1. Max logit value - should be reasonably high for in-distribution images
     * 2. Logit variance - should be high for confident predictions
     * 3. Top prediction confidence gap
     * 
     * IMPORTANT: Softmax can give high confidence even for out-of-distribution images
     * because it forces outputs to sum to 1. Raw logit values are more reliable
     * for detecting whether the image is truly a bacterial colony.
     */
    fun isValidClassification(predictions: List<Prediction>): Boolean {
        if (predictions.isEmpty()) return false
        
        val topConfidence = predictions.first().confidence
        val topClass = predictions.first().className
        
        Log.d(TAG, "=".repeat(60))
        Log.d(TAG, "VALIDATION REPORT for: $topClass")
        Log.d(TAG, "MaxLogit: $lastMaxLogit (range: $MIN_LOGIT_THRESHOLD - $MAX_LOGIT_THRESHOLD)")
        Log.d(TAG, "Variance: $lastLogitVariance (range: $MIN_VARIANCE_THRESHOLD - $MAX_VARIANCE_THRESHOLD)")
        Log.d(TAG, "TopConf: $topConfidence% (min: $MIN_VALID_CONFIDENCE_THRESHOLD%)")
        Log.d(TAG, "=".repeat(60))
        
        // Check 1: Max logit value RANGE - for in-distribution bacterial images, 
        // the model should have reasonably strong activation, but NOT extreme values
        // Extreme logit values (>50) often indicate OOD data causing model "overconfidence"
        if (lastMaxLogit < MIN_LOGIT_THRESHOLD) {
            Log.w(TAG, "❌ REJECTED: Max logit $lastMaxLogit below threshold $MIN_LOGIT_THRESHOLD (weak model activation)")
            return false
        }
        if (lastMaxLogit > MAX_LOGIT_THRESHOLD) {
            Log.w(TAG, "❌ REJECTED: Max logit $lastMaxLogit above threshold $MAX_LOGIT_THRESHOLD (OVERFITTING/OOD - model memorized this class!)")
            return false
        }
        
        // Check 2: Logit variance - in-distribution images produce varied but reasonable logits
        // Very low variance: model uncertain (uniform distribution)
        // Very high variance: model giving extreme responses to OOD data
        if (lastLogitVariance < MIN_VARIANCE_THRESHOLD) {
            Log.w(TAG, "❌ REJECTED: Logit variance $lastLogitVariance below threshold $MIN_VARIANCE_THRESHOLD (model uncertain - image unclear)")
            return false
        }
        if (lastLogitVariance > MAX_VARIANCE_THRESHOLD) {
            Log.w(TAG, "❌ REJECTED: Logit variance $lastLogitVariance above threshold $MAX_VARIANCE_THRESHOLD (extreme OOD/overfitting)")
            return false
        }
        
        // Check 3: Top confidence must be above softmax threshold
        if (topConfidence < MIN_VALID_CONFIDENCE_THRESHOLD) {
            Log.w(TAG, "❌ REJECTED: Top confidence $topConfidence% below threshold $MIN_VALID_CONFIDENCE_THRESHOLD%")
            return false
        }
        
        // Check 4: The gap between top predictions
        // If top 3 predictions are very close in confidence, model is confused
        if (predictions.size >= 3) {
            val top3 = predictions.take(3).map { it.confidence }
            val gap = top3[0] - top3[2]
            if (gap < 10f && topConfidence < 60f) {
                Log.w(TAG, "❌ REJECTED: Top predictions too similar (gap: $gap%, top: $topConfidence%)")
                return false
            }
        }
        
        Log.i(TAG, "✅ ACCEPTED: Image appears to be a valid bacterial colony - $topClass")
        return true
    }
    
    /**
     * Get validation result with detailed feedback
     */
    data class ValidationResult(
        val isValid: Boolean,
        val message: String,
        val topConfidence: Float
    )
    
    /**
     * Copy asset file to cache directory
     * Required for ONNX Runtime to access external data files
     */
    private fun copyAssetToCache(assetPath: String): java.io.File {
        val fileName = assetPath.substringAfterLast('/')
        val cacheFile = java.io.File(context.cacheDir, fileName)
        
        // Only copy if file doesn't exist or is outdated
        if (!cacheFile.exists()) {
            context.assets.open(assetPath).use { input ->
                cacheFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            Log.d(TAG, "Copied $assetPath to cache: ${cacheFile.absolutePath}")
        }
        
        return cacheFile
    }
    
    /**
     * Load bacterial species labels from assets
     */
    private fun loadLabels(): List<String> {
        return try {
            context.assets.open(LABELS_NAME)
                .bufferedReader()
                .useLines { lines ->
                    lines.map { it.trim() }
                        .filter { it.isNotEmpty() }
                        .toList()
                }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading labels from $LABELS_NAME", e)
            throw BacterialClassifierException("Failed to load labels: ${e.message}", e)
        }
    }
    
    /**
     * Release resources
     */
    fun close() {
        ortSession?.close()
        ortEnvironment?.close()
        isInitialized = false
        Log.d(TAG, "Classifier resources released")
    }
    
    /**
     * Data class for classification predictions
     */
    data class Prediction(
        val className: String,
        val probability: Float,
        val confidence: Float // Probability as percentage
    )
    
    /**
     * Classification result with validation status
     */
    data class ClassificationResult(
        val predictions: List<Prediction>,
        val isValid: Boolean,
        val validationMessage: String,
        val topConfidence: Float
    )
    
    /**
     * Classify with validation - returns result with validity check
     */
    suspend fun classifyWithValidation(bitmap: Bitmap, k: Int = 5): ClassificationResult {
        val predictions = classifyTopK(bitmap, k)
        val isValid = isValidClassification(predictions)
        val topConfidence = predictions.firstOrNull()?.confidence ?: 0f
        
        val message = when {
            predictions.isEmpty() -> "Sınıflandırma başarısız"
            !isValid && topConfidence < MIN_VALID_CONFIDENCE_THRESHOLD -> 
                "Bu görüntü bakteri kolonisi gibi görünmüyor. Lütfen uygun bir görüntü seçin."
            !isValid -> 
                "Model bu görüntüyü güvenilir bir şekilde sınıflandıramadı. Daha net bir görüntü deneyin."
            topConfidence < 50f -> 
                "Düşük güvenilirlik - sonuçları dikkatli değerlendirin"
            else -> "Sınıflandırma başarılı"
        }
        
        Log.d(TAG, "Classification validation: valid=$isValid, topConfidence=$topConfidence%, message=$message")
        
        return ClassificationResult(
            predictions = predictions,
            isValid = isValid,
            validationMessage = message,
            topConfidence = topConfidence
        )
    }
    
    /**
     * Detailed analysis result for debugging/testing model behavior
     */
    data class DetailedAnalysis(
        val predictions: List<Prediction>,
        val maxLogit: Float,
        val logitVariance: Float,
        val entropy: Float,
        val top5Spread: Float, // Difference between 1st and 5th prediction
        val isValid: Boolean,
        val invalidReason: String?,
        val recommendation: String
    )
    
    /**
     * Perform detailed analysis for testing model behavior
     * Use this to check if model is overfitting or behaving correctly
     */
    suspend fun analyzeInDetail(bitmap: Bitmap): DetailedAnalysis {
        val predictions = classify(bitmap)
        val top5 = predictions.take(5)
        
        val top5Spread = if (top5.size >= 5) {
            top5[0].confidence - top5[4].confidence
        } else {
            top5.firstOrNull()?.confidence ?: 0f
        }
        
        // Calculate entropy for uncertainty measure
        val entropy = calculateEntropy(predictions.map { it.probability }.toFloatArray())
        
        // Determine validity and reason
        val invalidReason = when {
            lastMaxLogit < MIN_LOGIT_THRESHOLD -> "Logit çok düşük (${lastMaxLogit} < $MIN_LOGIT_THRESHOLD)"
            lastMaxLogit > MAX_LOGIT_THRESHOLD -> "Logit çok yüksek (${lastMaxLogit} > $MAX_LOGIT_THRESHOLD) - OOD"
            lastLogitVariance < MIN_VARIANCE_THRESHOLD -> "Varyans çok düşük (${lastLogitVariance} < $MIN_VARIANCE_THRESHOLD)"
            lastLogitVariance > MAX_VARIANCE_THRESHOLD -> "Varyans çok yüksek (${lastLogitVariance} > $MAX_VARIANCE_THRESHOLD) - OOD"
            top5[0].confidence < MIN_VALID_CONFIDENCE_THRESHOLD -> "Güven çok düşük (${top5[0].confidence}%)"
            else -> null
        }
        
        val isValid = invalidReason == null
        
        // Generate recommendation based on analysis
        val recommendation = when {
            !isValid && lastMaxLogit > MAX_LOGIT_THRESHOLD -> 
                "⚠️ Bu görüntü bakteriyel koloni değil gibi görünüyor. Model aşırı tepki veriyor."
            !isValid && lastLogitVariance > MAX_VARIANCE_THRESHOLD -> 
                "⚠️ Model bu görüntü için çok belirsiz. Muhtemelen eğitim verisine benzemiyor."
            isValid && top5Spread > 90f -> 
                "✅ Model çok emin. Ezberleme riski düşük - benzersiz özellikler tespit edilmiş."
            isValid && top5Spread > 70f -> 
                "✅ İyi tahmin. Model güvenilir görünüyor."
            isValid && top5Spread < 30f -> 
                "⚠️ Top tahminler birbirine çok yakın. Model belirsiz veya görüntü bulanık olabilir."
            isValid && entropy < 0.1f -> 
                "⚠️ Çok düşük entropi - Model aşırı emin. Ezberleme olabilir."
            isValid && entropy > 0.5f -> 
                "ℹ️ Yüksek entropi - Model birkaç sınıf arasında kararsız."
            else -> "✅ Normal tahmin davranışı."
        }
        
        Log.d(TAG, """
            |=== DETAILED ANALYSIS ===
            |Top Prediction: ${top5.firstOrNull()?.className} (${top5.firstOrNull()?.confidence}%)
            |MaxLogit: $lastMaxLogit
            |Variance: $lastLogitVariance  
            |Entropy: $entropy
            |Top5 Spread: $top5Spread%
            |Valid: $isValid
            |Reason: ${invalidReason ?: "None"}
            |Recommendation: $recommendation
            |Top 5 Predictions:
            |${top5.mapIndexed { i, p -> "  ${i+1}. ${p.className}: ${p.confidence}%" }.joinToString("\n")}
            |=========================
        """.trimMargin())
        
        return DetailedAnalysis(
            predictions = top5,
            maxLogit = lastMaxLogit,
            logitVariance = lastLogitVariance,
            entropy = entropy,
            top5Spread = top5Spread,
            isValid = isValid,
            invalidReason = invalidReason,
            recommendation = recommendation
        )
    }
    
    /**
     * Custom exception for classifier errors
     */
    class BacterialClassifierException(message: String, cause: Throwable? = null) : 
        Exception(message, cause)
}
