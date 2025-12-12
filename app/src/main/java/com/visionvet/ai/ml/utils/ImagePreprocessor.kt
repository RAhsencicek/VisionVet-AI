package com.visionvet.ai.ml.utils

import android.graphics.Bitmap
import android.graphics.Matrix

/**
 * Image preprocessing utilities for machine learning models
 * Provides methods for resizing, cropping, and normalizing images
 */
object ImagePreprocessor {
    
    /**
     * Resize bitmap to target dimensions
     * @param bitmap Source bitmap
     * @param width Target width
     * @param height Target height
     * @param filter Enable bilinear filtering for better quality
     * @return Resized bitmap
     */
    fun resize(
        bitmap: Bitmap,
        width: Int,
        height: Int,
        filter: Boolean = true
    ): Bitmap {
        return Bitmap.createScaledBitmap(bitmap, width, height, filter)
    }
    
    /**
     * Center crop bitmap to square aspect ratio
     * Useful for preparing images for models that expect square inputs
     * @param bitmap Source bitmap
     * @param size Output size (width and height)
     * @return Center-cropped square bitmap
     */
    fun centerCrop(bitmap: Bitmap, size: Int): Bitmap {
        val dimension = minOf(bitmap.width, bitmap.height)
        val x = (bitmap.width - dimension) / 2
        val y = (bitmap.height - dimension) / 2
        
        val cropped = Bitmap.createBitmap(bitmap, x, y, dimension, dimension)
        return resize(cropped, size, size)
    }
    
    /**
     * Resize bitmap maintaining aspect ratio and pad to square
     * Alternative to center crop that preserves entire image
     * @param bitmap Source bitmap
     * @param targetSize Target size for width and height
     * @param paddingColor Background color for padding (ARGB format)
     * @return Resized and padded square bitmap
     */
    fun resizeWithPadding(
        bitmap: Bitmap,
        targetSize: Int,
        paddingColor: Int = android.graphics.Color.BLACK
    ): Bitmap {
        val aspectRatio = bitmap.width.toFloat() / bitmap.height.toFloat()
        
        val (newWidth, newHeight) = if (aspectRatio > 1) {
            // Landscape: width is larger
            targetSize to (targetSize / aspectRatio).toInt()
        } else {
            // Portrait: height is larger
            (targetSize * aspectRatio).toInt() to targetSize
        }
        
        // Resize maintaining aspect ratio
        val resized = resize(bitmap, newWidth, newHeight)
        
        // Create square bitmap with padding
        val output = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(output)
        canvas.drawColor(paddingColor)
        
        // Calculate padding offsets to center the image
        val left = (targetSize - newWidth) / 2f
        val top = (targetSize - newHeight) / 2f
        
        canvas.drawBitmap(resized, left, top, null)
        
        return output
    }
    
    /**
     * Rotate bitmap by specified degrees
     * @param bitmap Source bitmap
     * @param degrees Rotation angle (positive = clockwise)
     * @return Rotated bitmap
     */
    fun rotate(bitmap: Bitmap, degrees: Float): Bitmap {
        val matrix = Matrix().apply {
            postRotate(degrees)
        }
        return Bitmap.createBitmap(
            bitmap,
            0,
            0,
            bitmap.width,
            bitmap.height,
            matrix,
            true
        )
    }
    
    /**
     * Convert bitmap to grayscale
     * @param bitmap Source bitmap
     * @return Grayscale bitmap
     */
    fun toGrayscale(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val grayscale = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        
        val canvas = android.graphics.Canvas(grayscale)
        val paint = android.graphics.Paint()
        val colorMatrix = android.graphics.ColorMatrix()
        colorMatrix.setSaturation(0f)
        val filter = android.graphics.ColorMatrixColorFilter(colorMatrix)
        paint.colorFilter = filter
        canvas.drawBitmap(bitmap, 0f, 0f, paint)
        
        return grayscale
    }
    
    /**
     * Normalize pixel values for ImageNet models
     * Converts 0-255 RGB values to normalized float values using ImageNet statistics
     * Mean: [0.485, 0.456, 0.406]
     * Std: [0.229, 0.224, 0.225]
     * 
     * @param pixelValue Original pixel value (0-255)
     * @param channel Color channel (0=R, 1=G, 2=B)
     * @return Normalized float value
     */
    fun normalizeImageNet(pixelValue: Int, channel: Int): Float {
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        
        val normalized = pixelValue / 255.0f
        return (normalized - mean[channel]) / std[channel]
    }
    
    /**
     * Extract normalized RGB values from pixel
     * Used for preparing batch processing of images
     * @param pixel ARGB pixel value
     * @return Triple of normalized (R, G, B) values for ImageNet models
     */
    fun extractNormalizedRGB(pixel: Int): Triple<Float, Float, Float> {
        val r = (pixel shr 16) and 0xFF
        val g = (pixel shr 8) and 0xFF
        val b = pixel and 0xFF
        
        return Triple(
            normalizeImageNet(r, 0),
            normalizeImageNet(g, 1),
            normalizeImageNet(b, 2)
        )
    }
    
    /**
     * Prepare bitmap for bacterial classification model
     * - Center crops to square
     * - Resizes to 224x224
     * - Converts to RGB (removes alpha)
     * @param bitmap Source bitmap
     * @return Preprocessed bitmap ready for model input
     */
    fun prepareBacterialInput(bitmap: Bitmap): Bitmap {
        // Center crop to square
        val cropped = centerCrop(bitmap, 224)
        
        // Ensure RGB format (remove alpha channel if present)
        if (cropped.config == Bitmap.Config.ARGB_8888) {
            val rgb = Bitmap.createBitmap(
                cropped.width,
                cropped.height,
                Bitmap.Config.ARGB_8888
            )
            val canvas = android.graphics.Canvas(rgb)
            canvas.drawColor(android.graphics.Color.WHITE)
            canvas.drawBitmap(cropped, 0f, 0f, null)
            return rgb
        }
        
        return cropped
    }
    
    /**
     * Prepare bitmap for MNIST digit recognition
     * - Converts to grayscale
     * - Resizes to 28x28
     * - Inverts colors (white digit on black background)
     * @param bitmap Source bitmap
     * @return Preprocessed bitmap ready for MNIST model
     */
    fun prepareMnistInput(bitmap: Bitmap): Bitmap {
        val grayscale = toGrayscale(bitmap)
        val resized = resize(grayscale, 28, 28, filter = true)
        
        // MNIST expects white digits on black background
        // Invert if needed (check center pixel intensity)
        val pixels = IntArray(resized.width * resized.height)
        resized.getPixels(pixels, 0, resized.width, 0, 0, resized.width, resized.height)
        
        val centerPixel = pixels[pixels.size / 2]
        val brightness = (centerPixel shr 16 and 0xFF) + 
                        (centerPixel shr 8 and 0xFF) + 
                        (centerPixel and 0xFF)
        
        // If center is dark, assume digit is dark on light - invert
        if (brightness < 384) { // 384 = 128 * 3
            for (i in pixels.indices) {
                val r = 255 - (pixels[i] shr 16 and 0xFF)
                val g = 255 - (pixels[i] shr 8 and 0xFF)
                val b = 255 - (pixels[i] and 0xFF)
                pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
            resized.setPixels(pixels, 0, resized.width, 0, 0, resized.width, resized.height)
        }
        
        return resized
    }
}
