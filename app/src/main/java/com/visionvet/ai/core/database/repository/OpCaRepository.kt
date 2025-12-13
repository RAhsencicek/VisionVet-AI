package com.visionvet.ai.core.database.repository

import com.visionvet.ai.core.database.dao.UserDao
import com.visionvet.ai.core.database.dao.AnalysisDao
import com.visionvet.ai.core.database.model.User
import com.visionvet.ai.core.database.model.Analysis
import kotlinx.coroutines.flow.Flow

/**
 * Repository for database operations
 * Simplified for bacterial analysis
 */
class OpCaRepository(
    private val userDao: UserDao,
    private val analysisDao: AnalysisDao
) {
    // User operations (keeping for future auth implementation)
    suspend fun getUserByUserId(userId: String): User? = userDao.getUserByUserId(userId)

    suspend fun getUserByEmail(email: String): User? = userDao.getUserByEmail(email)

    suspend fun insertUser(user: User) = userDao.insertUser(user)

    suspend fun updateUser(user: User) = userDao.updateUser(user)

    suspend fun updateLastLoginDate(userId: String, loginDate: Long) =
        userDao.updateLastLoginDate(userId, loginDate)

    // Analysis operations
    fun getAllAnalyses(): Flow<List<Analysis>> = analysisDao.getAllAnalyses()

    suspend fun getAnalysisById(id: String): Analysis? = analysisDao.getAnalysisById(id)

    suspend fun getRecentAnalyses(limit: Int = 10): List<Analysis> =
        analysisDao.getRecentAnalyses(limit)

    suspend fun insertAnalysis(analysis: Analysis) = analysisDao.insertAnalysis(analysis)

    suspend fun updateAnalysis(analysis: Analysis) = analysisDao.updateAnalysis(analysis)

    suspend fun deleteAnalysis(analysis: Analysis) = analysisDao.deleteAnalysis(analysis)

    suspend fun deleteAnalysisById(id: String) = analysisDao.deleteAnalysisById(id)

    suspend fun deleteAllAnalyses() = analysisDao.deleteAllAnalyses()

    suspend fun getAnalysisCount(): Int = analysisDao.getAnalysisCount()
}
