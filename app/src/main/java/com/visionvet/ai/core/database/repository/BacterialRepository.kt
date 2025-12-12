package com.visionvet.ai.core.database.repository

import com.visionvet.ai.core.database.dao.BacterialResultDao
import com.visionvet.ai.core.database.model.BacterialResult
import kotlinx.coroutines.flow.Flow

class BacterialRepository(
    private val bacterialResultDao: BacterialResultDao
) {
    fun getAllResults(): Flow<List<BacterialResult>> =
        bacterialResultDao.getAllResults()
    
    fun getResultsByUserId(userId: String): Flow<List<BacterialResult>> =
        bacterialResultDao.getResultsByUserId(userId)
    
    suspend fun getResultById(id: String): BacterialResult? =
        bacterialResultDao.getResultById(id)
    
    suspend fun getRecentResults(userId: String, limit: Int = 10): List<BacterialResult> =
        bacterialResultDao.getRecentResults(userId, limit)
    
    suspend fun insertResult(result: BacterialResult) =
        bacterialResultDao.insertResult(result)
    
    suspend fun updateResult(result: BacterialResult) =
        bacterialResultDao.updateResult(result)
    
    suspend fun deleteResult(result: BacterialResult) =
        bacterialResultDao.deleteResult(result)
    
    suspend fun deleteAllResults(userId: String) =
        bacterialResultDao.deleteAllResultsByUserId(userId)
    
    suspend fun getResultCount(userId: String): Int =
        bacterialResultDao.getResultCountByUserId(userId)
    
    suspend fun getUnuploadedResults(userId: String): List<BacterialResult> =
        bacterialResultDao.getUnuploadedResults(userId)
    
    suspend fun getResultsByDateRange(
        userId: String,
        startTime: Long,
        endTime: Long
    ): List<BacterialResult> =
        bacterialResultDao.getResultsByDateRange(userId, startTime, endTime)
    
    suspend fun markAsUploaded(resultId: String) {
        val result = getResultById(resultId)
        result?.let {
            val updated = it.copy(
                isUploaded = true,
                uploadTimestamp = System.currentTimeMillis()
            )
            updateResult(updated)
        }
    }
}
