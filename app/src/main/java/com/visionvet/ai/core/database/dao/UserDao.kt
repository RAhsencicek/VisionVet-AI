package com.visionvet.ai.core.database.dao

import androidx.room.*
import com.visionvet.ai.core.database.model.User
import kotlinx.coroutines.flow.Flow

@Dao
interface UserDao {
    @Query("SELECT * FROM users WHERE userId = :userId")
    suspend fun getUserByUserId(userId: String): User?

    @Query("SELECT * FROM users WHERE email = :email")
    suspend fun getUserByEmail(email: String): User?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertUser(user: User)

    @Update
    suspend fun updateUser(user: User)

    @Delete
    suspend fun deleteUser(user: User)

    @Query("UPDATE users SET lastLoginDate = :loginDate WHERE userId = :userId")
    suspend fun updateLastLoginDate(userId: String, loginDate: Long)
}
