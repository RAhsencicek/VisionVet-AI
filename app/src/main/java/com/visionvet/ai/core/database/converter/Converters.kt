package com.visionvet.ai.core.database.converter

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.room.TypeConverter
import com.visionvet.ai.core.database.model.ParasiteResult
import com.visionvet.ai.core.database.model.ParasiteType
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.io.ByteArrayOutputStream

class Converters {

    @TypeConverter
    fun fromBitmap(bitmap: Bitmap?): ByteArray? {
        if (bitmap == null) return null
        val outputStream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, outputStream)
        return outputStream.toByteArray()
    }

    @TypeConverter
    fun toBitmap(byteArray: ByteArray?): Bitmap? {
        if (byteArray == null) return null
        return BitmapFactory.decodeByteArray(byteArray, 0, byteArray.size)
    }

    @TypeConverter
    fun fromParasiteResultList(results: List<ParasiteResult>): String {
        return Gson().toJson(results)
    }

    @TypeConverter
    fun toParasiteResultList(resultsString: String): List<ParasiteResult> {
        val listType = object : TypeToken<List<ParasiteResult>>() {}.type
        return Gson().fromJson(resultsString, listType)
    }

    @TypeConverter
    fun fromParasiteType(type: ParasiteType): String {
        return type.name
    }

    @TypeConverter
    fun toParasiteType(typeName: String): ParasiteType {
        return ParasiteType.valueOf(typeName)
    }
}
