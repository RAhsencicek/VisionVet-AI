package com.visionvet.ai.core.database.model

import androidx.room.Entity
import androidx.room.PrimaryKey
import java.util.Date
import java.util.UUID

@Entity(tableName = "users")
data class User(
    @PrimaryKey
    val id: String = UUID.randomUUID().toString(),
    val userId: String,
    val fullName: String,
    val email: String,
    val profileImageUrl: String? = null,
    val role: String = "user",
    val isActive: Boolean = true,
    val lastLoginDate: Long = System.currentTimeMillis(),
    val createdDate: Long = System.currentTimeMillis()
) {
    fun getLastLoginDateAsDate(): Date = Date(lastLoginDate)
    fun getCreatedDateAsDate(): Date = Date(createdDate)
}
