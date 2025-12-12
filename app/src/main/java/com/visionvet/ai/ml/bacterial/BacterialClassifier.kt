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
 * Model: MobileNetV3-Large trained on 33 bacterial species
 * Input: 224x224 RGB image
 * Output: Probability distribution over 33 classes
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
    
    companion object {
        private const val TAG = "BacterialClassifier"
        private const val MODEL_NAME = "mobilenet_v3_large.onnx"
        private const val LABELS_NAME = "labels_33.txt"
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
            
            // Load the ONNX model from assets
            // Copy model files to cache directory for ONNX Runtime to access external data
            val modelFile = copyAssetToCache("bacterial/$MODEL_NAME")
            val dataFile = copyAssetToCache("bacterial/$MODEL_NAME.data")
            
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
            
            // Get output tensor
            val outputTensor = outputs[0].value as Array<FloatArray>
            val probabilities = outputTensor[0]
            
            // Clean up
            inputTensor.close()
            outputs.close()
            
            // Apply softmax and create predictions
            val softmaxProbs = softmax(probabilities)
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
            context.assets.open("bacterial/$LABELS_NAME")
                .bufferedReader()
                .useLines { lines ->
                    lines.map { it.trim() }
                        .filter { it.isNotEmpty() }
                        .toList()
                }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading labels", e)
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
     * Custom exception for classifier errors
     */
    class BacterialClassifierException(message: String, cause: Throwable? = null) : 
        Exception(message, cause)
}
