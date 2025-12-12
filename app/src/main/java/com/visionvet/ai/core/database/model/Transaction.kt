package com.visionvet.ai.core.database.model

import androidx.room.Entity
import androidx.room.PrimaryKey
import java.util.*

@Entity(tableName = "transactions")
data class Transaction(
    @PrimaryKey
    val id: String = UUID.randomUUID().toString(),
    val title: String,
    val description: String,
    val amount: Double,
    val type: TransactionType,
    val status: TransactionStatus,
    val timestamp: Long = System.currentTimeMillis(),
    val category: String = "",
    val imageUrl: String? = null
)

enum class TransactionType {
    INCOME,
    EXPENSE,
    TRANSFER
}

enum class TransactionStatus {
    PENDING,
    COMPLETED,
    FAILED,
    CANCELLED
}
