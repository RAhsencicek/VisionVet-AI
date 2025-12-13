package com.visionvet.ai.core.database.dao

import androidx.room.*
import com.visionvet.ai.core.database.model.Analysis
import kotlinx.coroutines.flow.Flow

/**
 * Analysis DAO - Simplified for bacterial analysis
 */
@Dao
interface AnalysisDao {
    @Query("SELECT * FROM analyses ORDER BY timestamp DESC")
    fun getAllAnalyses(): Flow<List<Analysis>>

    @Query("SELECT * FROM analyses WHERE id = :id")
    suspend fun getAnalysisById(id: String): Analysis?

    @Query("SELECT * FROM analyses ORDER BY timestamp DESC LIMIT :limit")
    suspend fun getRecentAnalyses(limit: Int = 10): List<Analysis>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAnalysis(analysis: Analysis)

    @Update
    suspend fun updateAnalysis(analysis: Analysis)

    @Delete
    suspend fun deleteAnalysis(analysis: Analysis)

    @Query("DELETE FROM analyses WHERE id = :id")
    suspend fun deleteAnalysisById(id: String)

    @Query("DELETE FROM analyses")
    suspend fun deleteAllAnalyses()

    @Query("SELECT COUNT(*) FROM analyses")
    suspend fun getAnalysisCount(): Int
}
