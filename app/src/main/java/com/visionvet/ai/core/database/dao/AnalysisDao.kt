package com.visionvet.ai.core.database.dao

import androidx.room.*
import com.visionvet.ai.core.database.model.Analysis
import kotlinx.coroutines.flow.Flow

@Dao
interface AnalysisDao {
    @Query("SELECT * FROM analyses WHERE userId = :userId ORDER BY timestamp DESC")
    fun getAnalysesByUserId(userId: String): Flow<List<Analysis>>

    @Query("SELECT * FROM analyses WHERE id = :id")
    suspend fun getAnalysisById(id: String): Analysis?

    @Query("SELECT * FROM analyses WHERE userId = :userId ORDER BY timestamp DESC LIMIT :limit")
    suspend fun getRecentAnalyses(userId: String, limit: Int = 10): List<Analysis>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAnalysis(analysis: Analysis)

    @Update
    suspend fun updateAnalysis(analysis: Analysis)

    @Delete
    suspend fun deleteAnalysis(analysis: Analysis)

    @Query("DELETE FROM analyses WHERE userId = :userId")
    suspend fun deleteAllAnalysesByUserId(userId: String)

    @Query("SELECT COUNT(*) FROM analyses WHERE userId = :userId")
    suspend fun getAnalysisCountByUserId(userId: String): Int
}
