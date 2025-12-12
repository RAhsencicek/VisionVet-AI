package com.visionvet.ai.core.database

import androidx.room.TypeConverter
import com.visionvet.ai.core.database.model.TransactionType
import com.visionvet.ai.core.database.model.TransactionStatus

class Converters {
    @TypeConverter
    fun fromTransactionType(type: TransactionType): String = type.name

    @TypeConverter
    fun toTransactionType(type: String): TransactionType = TransactionType.valueOf(type)

    @TypeConverter
    fun fromTransactionStatus(status: TransactionStatus): String = status.name

    @TypeConverter
    fun toTransactionStatus(status: String): TransactionStatus = TransactionStatus.valueOf(status)
}
