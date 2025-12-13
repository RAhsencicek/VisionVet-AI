package com.visionvet.ai.core.database.model

import androidx.room.Entity
import androidx.room.Ignore
import androidx.room.PrimaryKey
import java.text.SimpleDateFormat
import java.util.*

/**
 * Analysis Entity - Simplified for Bacterial Analysis only
 */
@Entity(tableName = "analyses")
data class Analysis(
    @PrimaryKey
    val id: String = UUID.randomUUID().toString(),
    val bacteriaName: String,
    val confidence: Float,
    val imageUri: String? = null,
    val timestamp: Long = System.currentTimeMillis(),
    val notes: String = "",
    val isValid: Boolean = true
) {
    // Computed properties - not stored in database
    val formattedDate: String
        get() {
            val formatter = SimpleDateFormat("dd MMM yyyy, HH:mm", Locale.getDefault())
            return formatter.format(Date(timestamp))
        }
    
    val confidencePercent: String
        get() = "${(confidence * 100).toInt()}%"
}
