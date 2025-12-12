package com.visionvet.ai.core.database.dao

import androidx.room.*
import com.visionvet.ai.core.database.model.BacterialResult
import kotlinx.coroutines.flow.Flow

@Dao
interface BacterialResultDao {
    @Query("SELECT * FROM bacterial_results ORDER BY timestamp DESC")
    fun getAllResults(): Flow<List<BacterialResult>>
    
    @Query("SELECT * FROM bacterial_results WHERE userId = :userId ORDER BY timestamp DESC")
    fun getResultsByUserId(userId: String): Flow<List<BacterialResult>>
    
    @Query("SELECT * FROM bacterial_results WHERE id = :id")
    suspend fun getResultById(id: String): BacterialResult?
    
    @Query("SELECT * FROM bacterial_results WHERE userId = :userId ORDER BY timestamp DESC LIMIT :limit")
    suspend fun getRecentResults(userId: String, limit: Int = 10): List<BacterialResult>
    
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertResult(result: BacterialResult)
    
    @Update
    suspend fun updateResult(result: BacterialResult)
    
    @Delete
    suspend fun deleteResult(result: BacterialResult)
    
    @Query("DELETE FROM bacterial_results WHERE userId = :userId")
    suspend fun deleteAllResultsByUserId(userId: String)
    
    @Query("SELECT COUNT(*) FROM bacterial_results WHERE userId = :userId")
    suspend fun getResultCountByUserId(userId: String): Int
    
    @Query("SELECT * FROM bacterial_results WHERE userId = :userId AND isUploaded = 0 ORDER BY timestamp DESC")
    suspend fun getUnuploadedResults(userId: String): List<BacterialResult>
    
    @Query("SELECT * FROM bacterial_results WHERE userId = :userId AND timestamp BETWEEN :startTime AND :endTime ORDER BY timestamp DESC")
    suspend fun getResultsByDateRange(userId: String, startTime: Long, endTime: Long): List<BacterialResult>
}
