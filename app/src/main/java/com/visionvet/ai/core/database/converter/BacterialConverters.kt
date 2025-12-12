package com.visionvet.ai.core.database.converter

import androidx.room.TypeConverter
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.visionvet.ai.core.database.model.BacterialPrediction

class BacterialConverters {
    private val gson = Gson()
    
    @TypeConverter
    fun fromPredictionList(predictions: List<BacterialPrediction>): String {
        return gson.toJson(predictions)
    }
    
    @TypeConverter
    fun toPredictionList(json: String): List<BacterialPrediction> {
        val type = object : TypeToken<List<BacterialPrediction>>() {}.type
        return gson.fromJson(json, type)
    }
}
