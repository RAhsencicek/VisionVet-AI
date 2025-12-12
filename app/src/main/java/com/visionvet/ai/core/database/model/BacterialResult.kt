package com.visionvet.ai.core.database.model

import androidx.room.Entity
import androidx.room.PrimaryKey
import androidx.room.TypeConverters
import com.visionvet.ai.core.database.converter.BacterialConverters
import java.text.SimpleDateFormat
import java.util.*

@Entity(tableName = "bacterial_results")
@TypeConverters(BacterialConverters::class)
data class BacterialResult(
    @PrimaryKey
    val id: String,
    val userId: String,
    val timestamp: Long,
    val imagePath: String,
    val predictions: List<BacterialPrediction>,
    val notes: String = "",
    val location: String = "",
    val isUploaded: Boolean = false,
    val uploadTimestamp: Long? = null
) {
    val topPrediction: BacterialPrediction?
        get() = predictions.maxByOrNull { it.confidence }
    
    val formattedDate: String
        get() {
            val formatter = SimpleDateFormat("dd MMM yyyy, HH:mm", Locale.getDefault())
            return formatter.format(Date(timestamp))
        }
    
    val isHighConfidence: Boolean
        get() = (topPrediction?.confidence ?: 0f) >= 80f
}

data class BacterialPrediction(
    val className: String,
    val displayName: String,
    val confidence: Float,
    val probability: Float
) {
    val formattedConfidence: String
        get() = String.format("%.1f%%", confidence)
}
