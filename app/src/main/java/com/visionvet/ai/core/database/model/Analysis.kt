package com.visionvet.ai.core.database.model

import android.graphics.Bitmap
import androidx.room.Entity
import androidx.room.PrimaryKey
import androidx.room.TypeConverters
import com.visionvet.ai.core.database.converter.Converters
import java.text.SimpleDateFormat
import java.util.*

@Entity(tableName = "analyses")
@TypeConverters(Converters::class)
data class Analysis(
    @PrimaryKey
    val id: String,
    val userId: String, // Kullanıcı ID'si eksikti
    val imageData: ByteArray? = null,
    val imageBitmap: Bitmap? = null,
    val location: String,
    val timestamp: Long,
    val notes: String,
    val results: List<ParasiteResult>,
    val analysisType: AnalysisType = AnalysisType.PARASITE, // Eksik property eklendi
    val isUploaded: Boolean = false,
    val uploadTimestamp: Long? = null
) {
    val formattedDate: String
        get() {
            val formatter = SimpleDateFormat("MMM dd, yyyy HH:mm", Locale.getDefault())
            return formatter.format(Date(timestamp))
        }

    val dominantParasite: ParasiteType?
        get() = results.maxByOrNull { it.confidence }?.type

    fun getHighestConfidenceParasite(): Pair<ParasiteType, Double> {
        val highestResult = results.maxByOrNull { it.confidence }
        return if (highestResult != null) {
            Pair(highestResult.type, highestResult.confidence)
        } else {
            Pair(ParasiteType.ASCARIS, 0.0)
        }
    }

    fun getAnalysisDateAsDate(): Date {
        return Date(timestamp)
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as Analysis

        if (id != other.id) return false
        if (userId != other.userId) return false
        if (imageData != null) {
            if (other.imageData == null) return false
            if (!imageData.contentEquals(other.imageData)) return false
        } else if (other.imageData != null) return false
        if (location != other.location) return false
        if (timestamp != other.timestamp) return false
        if (notes != other.notes) return false
        if (results != other.results) return false
        if (analysisType != other.analysisType) return false
        if (isUploaded != other.isUploaded) return false
        if (uploadTimestamp != other.uploadTimestamp) return false

        return true
    }

    override fun hashCode(): Int {
        var result = id.hashCode()
        result = 31 * result + userId.hashCode()
        result = 31 * result + (imageData?.contentHashCode() ?: 0)
        result = 31 * result + location.hashCode()
        result = 31 * result + timestamp.hashCode()
        result = 31 * result + notes.hashCode()
        result = 31 * result + results.hashCode()
        result = 31 * result + analysisType.hashCode()
        result = 31 * result + isUploaded.hashCode()
        result = 31 * result + (uploadTimestamp?.hashCode() ?: 0)
        return result
    }
}

data class ParasiteResult(
    val type: ParasiteType,
    val confidence: Double,
    val detectionDate: Long
)
